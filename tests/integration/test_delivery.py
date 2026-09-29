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


@pytest.fixture()
def queue(store):
    """A real decision path — goal, due event, recorded decision — and one artifact on it."""
    policy = AttentionPolicy(store, owner_principal=OWNER)
    # An artifact only leases out of a memory whose owner has agreed to be told: shadow
    # mode is the default, and a drain that ignored it would be the switch nobody pulled.
    policy.configure(actor=OWNER, timezone_name=UTC, max_immediate_per_day=10,
                     cooldown_minutes=0, shadow=False)
    events = DueEventLog(store)
    goals = GoalStore(store, events=events, owner_principal=OWNER)
    subject = Outbox(store, policy=policy, owner_principal=OWNER)

    def ready(payload="The client is still waiting on the invoice."):
        decision = policy.decide(topic="general", at=MORNING)
        goal_id = goals.propose(title="Send the invoice", statement="Client waiting.",
                                timezone_name=UTC, due=timestamp(MORNING),
                                proposed_by=OWNER, proposed_kind="owner")["id"]
        event_id = store.db.execute("SELECT id FROM due_events WHERE goal_id=?",
                                    (goal_id,)).fetchone()[0]
        claim = events.claim(event_id, holder="worker", at=EPOCH)
        intent = events.ack(event_id=event_id, token=claim.token,
                            decision="awaiting_analysis",
                            policy_version=POLICY_VERSION)["intent"]
        written = policy.record(decision, intent_id=intent, goal_id=goal_id, revision=1)
        artifact = subject.prepare(decision_id=written["id"], kind="notify_owner",
                                  topic="general", payload=payload)
        return artifact["id"]

    return ready


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


def test_the_door_reports_the_transport_it_would_have_used(installed):
    """Named, enrolled, and still nothing sent: the report says which of those it is at."""
    from hermes_memory.cli import main

    code, report = _run(main, "deliver", "--hermes-home", str(installed))
    assert code == 0
    assert report["profile"] == "work" and report["delivered"] == 0
    assert "disabled by default" in report["reason"]
    assert report["transport"] == str(installed / "memory" / "delivered")


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
