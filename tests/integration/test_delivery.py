"""The outbox's caller: what it takes for a decided reminder to actually leave.

The bridge itself was written long before anything could run it — `deliver_once` and
`deliver_for_home` had no production caller, and the two delivery routes the operations
doc lists exist in no code at all, so the honest reading is that no artifact had ever
left an outbox on a real machine. These tests cover the part that was missing rather
than the part that was already careful: a drain reachable from outside the plugin, and a
sink that reports only what it can prove about itself.
"""
from __future__ import annotations

import json
import threading

import pytest

from hermes_memory.ids import timestamp
from hermes_memory.proactive.delivery import (DeliveryPolicy, bounded_drain, command_sink,
                                              deliver_ready, local_sink)
from hermes_memory.proactive.outbox import Outbox
from hermes_memory.proactive.policy import POLICY_VERSION, AttentionPolicy
from hermes_memory.prospective.due_events import DueEventLog
from hermes_memory.prospective.goals import GoalStore

OWNER = "jugaadu"
UTC = "UTC"
MORNING = "2026-09-15T09:00:00+00:00"
EPOCH = 1_789_462_800.0


def a_ready_pipeline(store, *, moment=MORNING, at=EPOCH):
    """goal -> due event -> intent -> decision -> artifact, as the pipeline writes it.

    The clock is an argument because a drain revalidates against the time it runs at: an
    artifact prepared for one fixed September morning is refused as expired by a loop
    running in a different year, and the test would then prove nothing about the loop.
    """
    policy = AttentionPolicy(store, owner_principal=OWNER)
    # An artifact only leases out of a memory whose owner has agreed to be told: shadow
    # mode is the default, and a drain that ignored it would be the switch nobody pulled.
    policy.configure(actor=OWNER, timezone_name=UTC, max_immediate_per_day=10,
                     cooldown_minutes=0, shadow=False)
    events = DueEventLog(store)
    goals = GoalStore(store, events=events, owner_principal=OWNER)
    subject = Outbox(store, policy=policy, owner_principal=OWNER)

    def ready(payload="The client is still waiting on the invoice."):
        decision = policy.decide(topic="general", at=moment)
        goal_id = goals.propose(title="Send the invoice", statement="Client waiting.",
                                timezone_name=UTC, due=timestamp(moment),
                                proposed_by=OWNER, proposed_kind="owner")["id"]
        event_id = store.db.execute("SELECT id FROM due_events WHERE goal_id=?",
                                    (goal_id,)).fetchone()[0]
        claim = events.claim(event_id, holder="worker", at=at)
        intent = events.ack(event_id=event_id, token=claim.token,
                            decision="awaiting_analysis",
                            policy_version=POLICY_VERSION)["intent"]
        written = policy.record(decision, intent_id=intent, goal_id=goal_id, revision=1)
        artifact = subject.prepare(decision_id=written["id"], kind="notify_owner",
                                  topic="general", payload=payload)
        return artifact["id"]

    return ready


@pytest.fixture()
def queue(store):
    """A real decision path — goal, due event, recorded decision — and one artifact on it."""
    return a_ready_pipeline(store)


def allowed(destination="local:jugaadu"):
    return DeliveryPolicy(enabled=True, destination=destination, owner_principal=OWNER)


# -- the drain -----------------------------------------------------------------

def test_a_ready_artifact_leaves_and_is_recorded_as_having_left(store, queue, tmp_path):
    identifier = queue()
    home = tmp_path / "homes" / "default"

    report = deliver_ready(store, policy=allowed(), sink=local_sink(home), limit=3,
                           at=EPOCH)

    assert report["delivered"] == 1, report
    assert report["reports"][0]["artifact"] == identifier
    landed = list((home / "memory" / "delivered").glob("*.md"))
    assert len(landed) == 1
    text = landed[0].read_text(encoding="utf-8")
    assert "The client is still waiting on the invoice." in text
    assert text.startswith(f"hermes-memory {identifier} "), \
        "the correlation line is what a transport can echo back and be believed"


def test_a_local_file_is_reported_as_accepted_and_not_as_confirmation(
        store, queue, tmp_path):
    """Nobody carried this anywhere. A file appearing is our own word, not a receipt."""
    queue()

    report = deliver_ready(store, policy=allowed(), sink=local_sink(tmp_path / "h"),
                           limit=1, at=EPOCH)

    assert report["reports"][0]["state"] == "accepted_unverified"
    assert report["reports"][0]["ok"] is False, \
        "`ok` means proved; `delivered` means handed over, and the two are not the same"
    assert report["delivered"] == 1
    assert store.db.execute("SELECT state FROM outbox").fetchone()[0] == "accepted_unverified"


def test_a_drained_artifact_is_not_sent_twice_by_the_next_run(store, queue, tmp_path):
    queue()
    home = tmp_path / "h"
    assert deliver_ready(store, policy=allowed(), sink=local_sink(home), limit=2,
                         at=EPOCH)["delivered"] == 1
    again = deliver_ready(store, policy=allowed(), sink=local_sink(home), limit=2,
                          at=EPOCH)

    assert again["delivered"] == 0
    assert again["reason"] == "nothing is ready to send right now"
    assert len(list((home / "memory" / "delivered").glob("*.md"))) == 1


def test_a_transport_that_dies_mid_send_is_uncertain_and_not_replayed(store, queue,
                                                                       tmp_path):
    """The sink raises after the bytes may have gone: that is the state this exists for."""
    identifier = queue()

    def half_written(body):
        (tmp_path / "partial").write_text(body[:20], encoding="utf-8")
        raise OSError("the disk went away")

    report = deliver_ready(store, policy=allowed(), sink=half_written, limit=1,
                           at=EPOCH)

    assert report["delivered"] == 0 and report["ok"] is False
    assert report["reports"][0]["state"] == "uncertain"
    assert "not be replayed blind" in report["reports"][0]["reason"]
    replayed = deliver_ready(store, policy=allowed(), sink=local_sink(tmp_path / "h"),
                             limit=1, at=EPOCH)
    assert replayed["delivered"] == 0, \
        f"{identifier} was already handed over once; a second copy is the worse failure"


# -- a transport that is another program ---------------------------------------

def a_transport(tmp_path, body: str) -> str:
    """Write a transport program and hand back the argv that runs it."""
    script = tmp_path / "transport.py"
    script.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def test_a_transport_that_reads_the_correlation_back_proves_the_send(store, queue,
                                                                     tmp_path):
    """The one receipt that earns `confirmed` without knowing the digest out of band."""
    queue()
    command = a_transport(tmp_path, """
import json, sys
body = sys.stdin.read()
print(json.dumps({"success": True, "platform": "telegram", "chat_id": "787655730",
                  "message_id": 42, "echo": body.splitlines()[0]}))
""")

    report = deliver_ready(store, policy=allowed("telegram:787655730"),
                           sink=command_sink([command], destination="telegram:787655730"),
                           limit=1, at=EPOCH)

    assert report["reports"][0]["state"] == "confirmed", report
    proof = json.loads(store.db.execute("SELECT proof FROM outbox").fetchone()[0])
    assert proof["message_id"] == 42 and proof["chat_id"] == "787655730", \
        "the transport's own answer is stored beside the artifact, not thrown away"
    assert store.db.execute("SELECT state FROM outbox").fetchone()[0] == "confirmed"


def test_a_transport_that_only_says_sent_is_believed_as_far_as_it_says(store, queue,
                                                                       tmp_path):
    queue()
    command = a_transport(tmp_path, """
import json, sys
sys.stdin.read()
print(json.dumps({"success": True, "platform": "telegram", "chat_id": "787655730",
                  "message_id": 43}))
""")

    report = deliver_ready(store, policy=allowed("telegram:787655730"),
                           sink=command_sink([command], destination="telegram:787655730"),
                           limit=1, at=EPOCH)

    assert report["reports"][0]["state"] == "accepted_unverified"
    assert report["delivered"] == 1, "it did go out; nobody proved the owner was shown it"


def test_a_transport_that_messaged_somebody_else_is_not_a_delivery(store, queue,
                                                                    tmp_path):
    """The approved destination is checked against where the program says it actually went."""
    queue()
    command = a_transport(tmp_path, """
import json, sys
sys.stdin.read()
print(json.dumps({"success": True, "chat_id": "999999999"}))
""")

    report = deliver_ready(store, policy=allowed("telegram:787655730"),
                           sink=command_sink([command], destination="telegram:787655730"),
                           limit=1, at=EPOCH)

    assert report["delivered"] == 0
    assert report["reports"][0]["state"] == "uncertain", \
        "bytes left this machine toward a stranger's chat; that is not retryable quietly"
    assert "not the approved destination" in store.db.execute(
        "SELECT reason FROM outbox").fetchone()[0]


def test_a_transport_program_that_does_not_exist_is_refused_before_anything_is_leased(
        store, queue):
    """A misspelled command must not spend the owner's reminders as `uncertain`."""
    queue()
    with pytest.raises(ValueError, match="not a program this installation can run"):
        command_sink(["/no/such/hermes-send"], destination="telegram:787655730")
    assert store.db.execute("SELECT state FROM outbox").fetchone()[0] == "prepared"


def test_a_failing_transport_leaves_the_artifact_uncertain_and_says_the_exit(store, queue,
                                                                             tmp_path):
    queue()
    command = a_transport(tmp_path, """
import sys
sys.stderr.write("bot token rejected\\n")
sys.exit(1)
""")

    report = deliver_ready(store, policy=allowed("telegram:787655730"),
                           sink=command_sink([command], destination="telegram:787655730"),
                           limit=1, at=EPOCH)

    assert report["delivered"] == 0
    assert report["reports"][0]["state"] == "uncertain"
    assert "exited 1" in store.db.execute("SELECT reason FROM outbox").fetchone()[0]


def test_a_transport_that_answers_in_pretty_printed_json_is_still_read(store, queue,
                                                                        tmp_path):
    """`hermes send --json` indents. Reading only the last line reads a closing brace."""
    queue()
    command = a_transport(tmp_path, """
import json, sys
sys.stdin.read()
print(json.dumps({"success": True, "platform": "telegram", "chat_id": "787655730",
                  "message_id": 44}, indent=2))
""")

    deliver_ready(store, policy=allowed("telegram:787655730"),
                  sink=command_sink([command], destination="telegram:787655730"),
                  limit=1, at=EPOCH)

    proof = json.loads(store.db.execute("SELECT proof FROM outbox").fetchone()[0])
    assert proof["chat_id"] == "787655730" and proof["message_id"] == 44
    assert "destination_unconfirmed" not in proof, \
        "the transport named the chat; the receipt must not say nobody did"


def test_concurrent_drains_never_close_each_other_s_delivery(store, queue, tmp_path):
    """A lease must not suppress a row another drain already took.

    Eight drains over twelve artifacts reproduce the window wide open: the candidate list
    is a snapshot, so a row can be claimed between the SELECT and the revalidation that
    follows it. Before the guard the losing drain died with "already attempted", and the
    artifact it was trying to close had already been delivered to the owner.
    """
    identifiers = [queue(payload=f"reminder {index}") for index in range(12)]
    outcomes, errors = [], []

    def drain(worker: int) -> None:
        try:
            from hermes_memory.storage.evidence import EvidenceStore

            with EvidenceStore(store.path) as own:
                outcomes.append(deliver_ready(own, policy=allowed(),
                                              sink=local_sink(tmp_path / f"h{worker}"),
                                              limit=12, holder=f"drain-{worker}"))
        except Exception as error:
            errors.append(f"{type(error).__name__}: {str(error)[:160]}")

    threads = [threading.Thread(target=drain, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors[:3]
    handed = sum(item["delivered"] for item in outcomes)
    states = store.db.execute("SELECT state, count(*) FROM outbox GROUP BY state").fetchall()
    assert handed == len(identifiers), (handed, len(identifiers))
    assert dict(states).get("accepted_unverified") == len(identifiers), dict(states)
    landed = len(list((tmp_path).rglob("*.md")))
    assert landed == len(identifiers), \
        f"{landed} files for {len(identifiers)} artifacts: something was sent twice"


# -- what refuses --------------------------------------------------------------

def test_an_installation_with_no_destination_says_why_rather_than_showing_an_empty_queue(
        store, queue):
    queue()
    blocked = DeliveryPolicy(enabled=False, destination=None, owner_principal=OWNER)

    report = deliver_ready(store, policy=blocked, sink=lambda body: None, limit=1)

    assert report["delivered"] == 0 and "disabled by default" in report["reason"]
    assert store.db.execute("SELECT state FROM outbox").fetchone()[0] == "prepared", \
        "a refusal leaves the artifact waiting, not spent"


@pytest.mark.parametrize("target,why", [
    ("signal", "not a concrete destination"),
    ("broadcast:everyone", "not a private destination"),
])
def test_a_destination_that_is_not_one_private_channel_is_refused_by_name(
        store, queue, target, why):
    queue()
    policy = DeliveryPolicy(enabled=True, destination=target, owner_principal=OWNER)

    report = deliver_ready(store, policy=policy, sink=lambda body: None, limit=1)

    assert why in report["reason"]


def test_an_operators_hold_stops_the_drain_without_turning_the_switch_off(store, queue,
                                                                          tmp_path):
    """The hold is temporary and the switch is configuration; both refuse, and say which."""
    queue()

    report = deliver_ready(store, policy=allowed(), sink=local_sink(tmp_path / "h"),
                           limit=1, held="delivery is paused for this installation by "
                                          "the owner")

    assert report["delivered"] == 0 and "paused for this installation" in report["reason"]
    assert not (tmp_path / "h" / "memory" / "delivered").exists()


@pytest.mark.parametrize("limit", [0, -1, 26, "3", None, True])
def test_an_unbounded_or_absurd_drain_is_refused_before_anything_is_opened(limit):
    with pytest.raises(ValueError, match="between 1 and 25"):
        bounded_drain(limit)


def test_a_sink_that_names_nothing_cannot_record_a_delivery(store, queue):
    queue()
    with pytest.raises(ValueError, match="name a transport"):
        deliver_ready(store, policy=allowed(), limit=1)


# -- the door ------------------------------------------------------------------

@pytest.fixture()
def installed(tmp_path, monkeypatch):
    """A running installation with one enrolled profile, the way `enroll` makes one."""
    root = tmp_path / "hm"
    root.mkdir()
    (root / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={tmp_path / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(root))
    monkeypatch.delenv("HERMES_MEMORY_DATA_DIR", raising=False)
    home = root / "profiles" / "work"
    home.mkdir(parents=True)
    from hermes_memory.cli import main

    assert _run(main, "init")[0] == 0
    plan = _run(main, "enroll", "--hermes-home", str(home))[1]
    assert _run(main, "enroll", "--hermes-home", str(home), "--review",
                plan["review_digest"], "--actor", OWNER)[0] == 0
    return home


def test_the_door_refuses_to_guess_whose_memory_it_is_draining(installed):
    """The home is the destination as well as the lookup, so it is never inferred."""
    from hermes_memory.cli import main

    code, message = _errors(main, "deliver")
    assert code == 2 and "--hermes-home or --profile" in message


def test_the_door_reports_the_transport_it_would_have_used(installed, monkeypatch):
    """Named, enrolled, and still nothing sent: the report says which of those it is at.

    The transport comes from the destination the owner configured. It is not assumed: a
    door that named a local directory on an installation that never asked for one would be
    reporting a destination nobody approved.
    """
    from hermes_memory.cli import main

    monkeypatch.setenv("HERMES_MEMORY_DELIVERY_TARGET", f"local:{OWNER}")
    code, report = _run(main, "deliver", "--hermes-home", str(installed))
    assert code == 0
    assert report["profile"] == "work" and report["delivered"] == 0
    assert "disabled by default" in report["reason"]
    assert report["transport"] == str(installed / "memory" / "delivered")


def test_a_door_with_no_destination_names_no_transport_rather_than_the_nearest_folder(
        installed):
    from hermes_memory.cli import main

    code, report = _run(main, "deliver", "--hermes-home", str(installed))
    assert code == 0 and report["transport"] is None
    assert "disabled by default" in report["reason"]


def test_the_door_refuses_to_write_a_file_and_call_it_a_telegram_message(installed,
                                                                         monkeypatch):
    """A destination that needs a transport, and no transport named, is not a send."""
    from hermes_memory.cli import main

    monkeypatch.setenv("HERMES_MEMORY_DELIVERY_ENABLED", "true")
    monkeypatch.setenv("HERMES_MEMORY_DELIVERY_TARGET", "telegram:787655730")
    code, message = _errors(main, "deliver", "--hermes-home", str(installed))
    assert code == 2 and "HERMES_MEMORY_DELIVERY_COMMAND" in message


def _run(main, *argv):
    captured = pytest.MonkeyPatch()
    out = []
    captured.setattr("builtins.print", lambda *a, **k: out.append(a[0] if a else ""))
    try:
        code = main(list(argv))
    finally:
        captured.undo()
    return code, json.loads(out[-1])


def _errors(main, *argv):
    written = []
    patch = pytest.MonkeyPatch()
    patch.setattr("sys.stderr.write", lambda text: written.append(text))
    try:
        try:
            code = main(list(argv))
        except SystemExit as exit_code:      # argparse refuses before main can
            code = int(exit_code.code or 2)
    finally:
        patch.undo()
    return code, "".join(written)


# -- the timer that sends without a caller -------------------------------------

def _profile_db(installed):
    """The enrolled profile's own database, resolved the way the runtime resolves it."""
    from hermes_memory.config import load_settings
    from hermes_memory.install.profiles import ProfileRegistry

    registry = ProfileRegistry.reading(load_settings())
    try:
        return registry.resolve(installed).scoped(load_settings()).db_path
    finally:
        registry.db.close()


def _owner_said_yes(monkeypatch, destination=None):
    monkeypatch.setenv("HERMES_MEMORY_DELIVERY_ENABLED", "true")
    monkeypatch.setenv("HERMES_MEMORY_DELIVERY_TARGET", destination or f"local:{OWNER}")


def test_the_installed_drain_sends_what_the_pass_prepared(installed, monkeypatch):
    """Nothing types `deliver`: this is the half that makes a reminder actually arrive.

    The drain is run against a real enrolled installation rather than a hand-made policy,
    because the thing under test is the resolution from *installation* to *profile* to
    *transport* — the part that had no caller at all.
    """
    import time

    from hermes_memory.config import load_settings
    from hermes_memory.ids import now
    from hermes_memory.proactive.delivery import drain_installed, last_drain
    from hermes_memory.storage.evidence import EvidenceStore

    _owner_said_yes(monkeypatch)
    with EvidenceStore(_profile_db(installed)) as store:
        identifier = a_ready_pipeline(store, moment=now(), at=time.time())()
        outcome = drain_installed(load_settings(), limit=3)

        assert outcome["delivered"] == 1, outcome
        assert [item["profile"] for item in outcome["profiles"]] == ["work"]
        assert store.db.execute("SELECT state FROM outbox WHERE id=?",
                                (identifier,)).fetchone()[0] == "accepted_unverified"
        assert last_drain(store)["delivered"] == 1
    assert len(list((installed / "memory" / "delivered").glob("*.md"))) == 1


def test_a_drain_with_nothing_to_say_writes_one_line_rather_than_one_a_period(
        installed, monkeypatch):
    """A minute-long poll is 1440 drains a day; the audit is not where that belongs.

    The line exists so `status` can see the loop from another process. Recording every
    no-op would bury the evidence the audit is supposed to be, so only a *change* is
    written — and a loop that is quietly idle is proved by the backlog reading instead.
    """
    from hermes_memory.config import load_settings
    from hermes_memory.proactive.delivery import drain_installed
    from hermes_memory.storage.evidence import EvidenceStore

    _owner_said_yes(monkeypatch)
    with EvidenceStore(_profile_db(installed)) as store:
        for _ in range(5):
            assert drain_installed(load_settings(), limit=3)["delivered"] == 0
        assert store.db.execute("SELECT count(*) FROM audit WHERE action='delivery_drain'"
                                ).fetchone()[0] == 1


def test_a_send_and_then_a_quiet_drain_are_two_different_lines(installed, monkeypatch):
    import time

    from hermes_memory.config import load_settings
    from hermes_memory.ids import now
    from hermes_memory.proactive.delivery import drain_installed, last_drain
    from hermes_memory.storage.evidence import EvidenceStore

    _owner_said_yes(monkeypatch)
    with EvidenceStore(_profile_db(installed)) as store:
        a_ready_pipeline(store, moment=now(), at=time.time())()
        drain_installed(load_settings(), limit=3)
        drain_installed(load_settings(), limit=3)
        lines = store.db.execute(
            "SELECT metadata FROM audit WHERE action='delivery_drain'").fetchall()
        assert len(lines) == 2, lines
        assert last_drain(store)["delivered"] == 0


def test_a_destination_with_no_transport_program_is_said_once_instead_of_silently(
        installed, monkeypatch):
    """The owner named Telegram, nobody named the program that speaks it: not a file, and
    not silence either."""
    from hermes_memory.config import load_settings
    from hermes_memory.proactive.delivery import drain_installed, last_drain
    from hermes_memory.storage.evidence import EvidenceStore

    _owner_said_yes(monkeypatch, "telegram:787655730")
    with EvidenceStore(_profile_db(installed)) as store:
        outcome = drain_installed(load_settings(), limit=3)
        assert outcome["delivered"] == 0
        assert "no transport program is configured" in outcome["profiles"][0]["reason"]
        assert "no transport program" in last_drain(store)["reason"]


def test_an_installation_that_never_switched_delivery_on_records_no_drain(installed):
    """Refusal by configuration is not an event; it is the state of the machine."""
    from hermes_memory.config import load_settings
    from hermes_memory.proactive.delivery import drain_installed
    from hermes_memory.storage.evidence import EvidenceStore

    with EvidenceStore(_profile_db(installed)) as store:
        outcome = drain_installed(load_settings(), limit=3)
        assert outcome["delivered"] == 0 and "disabled by default" in (
            outcome["profiles"][0]["reason"])
        assert store.db.execute("SELECT count(*) FROM audit WHERE action='delivery_drain'"
                                ).fetchone()[0] == 0
