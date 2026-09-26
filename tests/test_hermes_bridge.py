"""The two plugin entry points that reach outside the machine: enrollment and delivery.

Both are refused by default, both name the owner as the only party that can change that,
and both say what they refused rather than quietly doing nothing.
"""
from __future__ import annotations

import importlib
import io
import json

import pytest
from hermes_memory.ids import timestamp
from hermes_memory.install.profiles import InstallationError, ProfileRegistry
from hermes_memory.proactive.outbox import Outbox
from hermes_memory.proactive.policy import POLICY_VERSION, AttentionPolicy
from hermes_memory.prospective.due_events import DueEventLog
from hermes_memory.prospective.goals import GoalStore

from plugin_loader import load_plugin

OWNER = "owner-principal"
UTC = "UTC"
MORNING = "2026-09-15T09:00:00+00:00"
EPOCH = 1789462800.0


@pytest.fixture()
def plugin():
    module = load_plugin("hm_bridge")
    module.setup = importlib.import_module(f"{module.__name__}.setup")
    module.delivery = importlib.import_module(f"{module.__name__}.delivery")
    return module


@pytest.fixture()
def instance(tmp_path, monkeypatch):
    """An installation home with an owner and nothing enrolled."""
    root = tmp_path / "instance"
    root.mkdir()
    (root / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={root / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(root))
    monkeypatch.delenv("HERMES_MEMORY_DELIVERY_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_DELIVERY_TARGET", raising=False)
    return root


@pytest.fixture()
def outbox(store, policy):
    events = DueEventLog(store)
    goals = GoalStore(store, events=events, owner_principal=OWNER)
    subject = Outbox(store, policy=policy, owner_principal=OWNER, clock=lambda: EPOCH)

    def ready(kind="notify_owner", payload="The client is still waiting on the invoice."):
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
        artifact = subject.prepare(decision_id=written["id"], kind=kind, topic="general",
                                   payload=payload)
        return artifact

    return {"outbox": subject, "ready": ready, "policy": policy}


@pytest.fixture()
def policy(store):
    subject = AttentionPolicy(store, owner_principal=OWNER)
    subject.configure(actor=OWNER, timezone_name=UTC, quiet_from="22:00",
                      quiet_until="06:00", max_immediate_per_day=10, cooldown_minutes=0,
                      shadow=False)
    return subject


def enrolled(plugin, instance, home, *, profile="work", owner=OWNER):
    ledger = plugin.setup.registry(instance_home=instance)
    proposal = ledger.plan(profile, home)
    result = ledger.enroll(profile, home, actor=owner,
                           review_digest=proposal["review_digest"])
    ledger.db.close()
    return result


# -- setup: the reviewed diff --------------------------------------------------

def test_a_plan_names_every_path_it_would_create_and_writes_none_of_them(
        plugin, instance, tmp_path):
    home = tmp_path / "homes" / "work"
    proposal = plugin.setup.plan(home, instance_home=instance)

    assert proposal["profile"] == "work"
    assert proposal["data_dir"] == str(instance / "profiles" / "work")
    assert proposal["bank_id"] == "hermes-work"
    assert any("profiles/work" in line for line in proposal["adds"])
    assert proposal["command"].startswith("hermes-memory enroll --profile work")
    assert proposal["review_digest"] and not home.exists()
    assert not (instance / "profiles").exists()


def test_a_plan_says_what_it_leaves_alone(plugin, instance, tmp_path):
    """Each of these is a thing a setup run has previously been observed to touch."""
    unchanged = " ".join(plugin.setup.plan(tmp_path / "homes" / "work",
                                          instance_home=instance)["unchanged"])
    for promise in ("main model", "native Hermes memory", "ports", "connectors",
                    "selected"):
        assert promise in unchanged


def test_setup_enrolls_nothing_hermes_did_not_show_an_owner(plugin, instance, tmp_path):
    """The wizard ends with a command, because a digest the code computes for itself
    is not evidence that a person read it."""
    home = tmp_path / "homes" / "work"
    report = plugin.setup.report(home, instance_home=instance)
    assert report["enrolled"] is False
    assert "--actor <your principal>" in report["next_command"]
    assert report["model_config_unchanged"] is True
    assert report["why"]
    with pytest.raises(InstallationError, match="not enrolled"):
        plugin.setup.registry(instance_home=instance).profile("work")


def test_enrollment_needs_the_digest_of_the_proposal_that_was_made(
        plugin, instance, tmp_path):
    home = tmp_path / "homes" / "work"
    proposal = plugin.setup.plan(home, instance_home=instance)
    with pytest.raises(InstallationError, match="does not match"):
        plugin.setup.enroll(home, actor=OWNER, review_digest="0" * 64,
                           instance_home=instance)
    assert plugin.setup.pending(home, instance_home=instance)["enrolled"] is False

    done = plugin.setup.enroll(home, actor=OWNER, review_digest=proposal["review_digest"],
                              instance_home=instance)
    assert done["changed"] is True
    state = plugin.setup.pending(home, instance_home=instance)
    assert state == {"enrolled": True, "profile": "work", "bank_id": "hermes-work",
                     "data_dir": str(instance / "profiles" / "work"), "command": None,
                     "reason": None}


@pytest.mark.parametrize("actor", ["agent", "", None])
def test_nobody_but_the_owner_can_enroll(plugin, instance, tmp_path, actor):
    home = tmp_path / "homes" / "work"
    proposal = plugin.setup.plan(home, instance_home=instance)
    with pytest.raises(InstallationError, match="only the owner"):
        plugin.setup.enroll(home, actor=actor, review_digest=proposal["review_digest"],
                            instance_home=instance)


def test_retiring_takes_the_name_off_the_ledger_not_the_data_off_the_disk(
        plugin, instance, tmp_path):
    home = tmp_path / "homes" / "work"
    enrolled(plugin, instance, home)
    (instance / "profiles" / "work").mkdir(parents=True)
    (instance / "profiles" / "work" / "canonical.db").write_text("evidence",
                                                                encoding="utf-8")
    result = plugin.setup.retire(home, actor=OWNER, reason="machine retired",
                                instance_home=instance)
    assert result["changed"] is True
    assert (instance / "profiles" / "work" / "canonical.db").read_text(encoding="utf-8") \
        == "evidence"
    assert plugin.setup.pending(home, instance_home=instance)["enrolled"] is False


def test_the_profile_is_named_from_the_home_and_the_home_alone(plugin, instance, tmp_path):
    assert plugin.setup.plan(tmp_path / "profiles" / "default",
                             instance_home=instance)["profile"] == "default"
    assert plugin.setup.plan(tmp_path / ".hermes",
                             instance_home=instance)["profile"] == "default"
    assert plugin.setup.plan(tmp_path / "Work Laptop",
                             instance_home=instance)["profile"] == "work-laptop"


def test_the_ledger_is_the_installations_not_a_profiles(plugin, instance, tmp_path):
    """Pointing the registry at a profile home would give it an unreviewed private map."""
    ledger = plugin.setup.registry(instance_home=tmp_path / "profile-home")
    assert ledger.db.execute("PRAGMA database_list").fetchone()[2] == str(
        (tmp_path / "profile-home").resolve() / "installation.db")
    ledger.db.close()


# -- delivery: one artifact, to the owner, with a receipt ----------------------

def settings_for(plugin, instance, tmp_path, monkeypatch, **values):
    from hermes_memory.config import load_settings

    lines = [f"HERMES_MEMORY_{key}={value}" for key, value in values.items()]
    (instance / "hermes-memory.env").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(instance))
    return load_settings()


def enabled_policy(plugin, **overrides):
    arguments = {"enabled": True, "destination": "signal:owner-1234",
                 "owner_principal": OWNER}
    arguments.update(overrides)
    return plugin.delivery.DeliveryPolicy(**arguments)


def test_delivery_is_refused_before_an_owner_asks_for_it(plugin, outbox, store):
    calls = []
    report = plugin.delivery.deliver_once(
        outbox["outbox"], policy=enabled_policy(plugin, enabled=False),
        sink=lambda body: calls.append(body))
    assert report == {"ok": True, "delivered": False, "attempted": False,
                      "reason": "provider-initiated delivery is disabled by default"}
    assert calls == []
    assert outbox["outbox"].pending() == [], "a refused run must not even lease one"


@pytest.mark.parametrize("destination", [None, "", "   ", "signal:", "signal:  ",
                                         "bot-chat:room", "all:everyone",
                                         "broadcast:x", "not-a-destination"])
def test_an_approved_destination_has_to_be_one_destination(plugin, destination):
    blocked = enabled_policy(plugin, destination=destination).refusal()
    assert blocked, f"{destination!r} was accepted as a delivery target"
    if destination in ("bot-chat:room", "all:everyone", "broadcast:x"):
        assert "private destination" in blocked


def test_enabling_delivery_without_a_destination_is_a_startup_error(tmp_path, monkeypatch):
    from hermes_memory.config import SettingError, load_settings

    home = tmp_path / "home"
    home.mkdir()
    (home / "hermes-memory.env").write_text("HERMES_MEMORY_DELIVERY_ENABLED=true\n",
                                            encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.delenv("HERMES_MEMORY_DELIVERY_TARGET", raising=False)
    with pytest.raises(SettingError, match="DELIVERY_TARGET"):
        load_settings()


def test_a_delivery_with_no_transport_refuses_to_record_a_receipt(plugin, outbox):
    """Writing nowhere and then confirming something would be the worst report here."""
    with pytest.raises(ValueError, match="name a transport"):
        plugin.delivery.deliver_once(outbox["outbox"], policy=enabled_policy(plugin),
                                     at=EPOCH)


def test_one_ready_artifact_reaches_the_transport_with_its_own_digest(
        plugin, outbox, store):
    prepared = outbox["ready"]()
    seen = []

    def sink(body):
        seen.append(body)
        return {"message_id": "msg-1", "digest": prepared["payload_digest"]}

    report = plugin.delivery.deliver_once(outbox["outbox"],
                                          policy=enabled_policy(plugin), sink=sink,
                                          at=EPOCH)
    assert report["ok"] is True and report["state"] == "confirmed"
    assert seen == [f"hermes-memory {prepared['id']} {prepared['payload_digest'][:12]}\n"
                    "The client is still waiting on the invoice."]
    assert outbox["outbox"].get(prepared["id"]).state == "confirmed"
    assert report["delivered"] is True


def test_a_transport_that_says_sent_without_a_digest_is_not_confirmation(
        plugin, outbox):
    """'sent' is a report of the transport's intent, not of the owner's inbox."""
    prepared = outbox["ready"]()
    report = plugin.delivery.deliver_once(
        outbox["outbox"], policy=enabled_policy(plugin),
        sink=lambda body: {"sent": True}, at=EPOCH)
    assert report["ok"] is False and report["state"] == "accepted_unverified"
    assert report["delivered"] is True, "it was sent as far as anyone can tell"
    assert outbox["outbox"].get(prepared["id"]).state == "accepted_unverified"


def test_a_digest_that_is_not_this_artifacts_is_uncertainty(plugin, outbox):
    prepared = outbox["ready"]()
    report = plugin.delivery.deliver_once(
        outbox["outbox"], policy=enabled_policy(plugin),
        sink=lambda body: {"digest": "0" * 64}, at=EPOCH)
    assert report["state"] == "uncertain"
    assert outbox["outbox"].get(prepared["id"]).state == "uncertain"
    again = plugin.delivery.deliver_once(
        outbox["outbox"], policy=enabled_policy(plugin), sink=lambda body: None, at=EPOCH)
    assert again["delivered"] == 0 or again["attempted"] is False, \
        "an unresolved send is never replayed blind"


def test_a_transport_that_dies_after_the_handover_began_leaves_it_uncertain(
        plugin, outbox):
    prepared = outbox["ready"]()
    exploded = []

    def sink(body):
        exploded.append(body)
        raise RuntimeError("connection reset after write")

    report = plugin.delivery.deliver_once(outbox["outbox"], policy=enabled_policy(plugin),
                                          sink=sink, at=EPOCH)
    assert report["attempted"] is True and report["state"] == "uncertain"
    assert outbox["outbox"].get(prepared["id"]).state == "uncertain"


def test_an_artifact_addressed_to_anyone_but_the_current_owner_goes_back(
        plugin, outbox):
    """The outbox proves the bytes are intact; it cannot prove the owner still is.

    A principal change after queueing is the case this catches: the artifact is
    still genuinely the one that was prepared, and it is still not ours to send to
    whoever the owner is now.
    """
    prepared = outbox["ready"]()
    calls = []
    report = plugin.delivery.deliver_once(
        outbox["outbox"], policy=enabled_policy(plugin, owner_principal="a-new-owner"),
        sink=lambda body: calls.append(body), at=EPOCH)
    assert calls == [] and report["attempted"] is False
    assert "someone other than the owner" in report["reason"]
    assert outbox["outbox"].get(prepared["id"]).state == "prepared", \
        "a refusal is not a delivery, and the owner may still want it"


def test_bytes_changed_under_the_transport_are_not_this_artifact_any_more(
        plugin, outbox, store):
    """A readdressed row fails its own digest, and a suppressed artifact is not sent."""
    prepared = outbox["ready"]()
    store.db.execute("UPDATE outbox SET recipient='someone-else' WHERE id=?",
                     (prepared["id"],))
    calls = []
    report = plugin.delivery.deliver_once(
        outbox["outbox"], policy=enabled_policy(plugin),
        sink=lambda body: calls.append(body), at=EPOCH)
    assert calls == [] and report["attempted"] is False
    assert outbox["outbox"].get(prepared["id"]).state == "suppressed"


def test_a_draft_is_never_handed_to_a_transport(plugin, outbox):
    """A draft is a thing the owner approves and sends; that is the whole point."""
    prepared = outbox["ready"](kind="draft", payload="Draft reply to the client.")
    calls = []
    report = plugin.delivery.deliver_once(
        outbox["outbox"], policy=enabled_policy(plugin),
        sink=lambda body: calls.append(body), at=EPOCH)
    assert calls == [] and "draft" in report["reason"]
    assert outbox["outbox"].get(prepared["id"]).state == "prepared"


def test_the_body_is_written_out_with_its_correlation_for_a_script_transport(
        plugin, outbox):
    prepared = outbox["ready"]()
    stream = io.StringIO()
    report = plugin.delivery.deliver_once(outbox["outbox"], policy=enabled_policy(plugin),
                                          out=stream, at=EPOCH)
    assert stream.getvalue().startswith(f"hermes-memory {prepared['id']} "
                                        f"{prepared['payload_digest'][:12]}")
    assert report["state"] == "accepted_unverified", \
        "a completed script is not a delivered receipt"


def test_quiet_hours_hold_the_artifact_back_without_sending_it(plugin, outbox, store):
    prepared = outbox["ready"]()
    late = enabled_policy(plugin)
    # 23:30Z is inside the configured quiet window, so the lease revalidates it away.
    report = plugin.delivery.deliver_once(outbox["outbox"], policy=late,
                                          sink=lambda body: None, at=EPOCH + 52200)
    assert report["attempted"] is False and report["delivered"] is False
    assert outbox["outbox"].get(prepared["id"]).state in ("prepared", "suppressed")


# -- deliver_for_home: the same guard from a separate process ------------------

def test_a_home_with_no_profile_delivers_nothing(plugin, instance, tmp_path):
    from hermes_memory.config import load_settings

    outbox_home = tmp_path / "homes" / "work"
    with pytest.raises(plugin.client.BindingError,
                       match="no installation ledger"):
        plugin.delivery.deliver_for_home(outbox_home, settings=load_settings(),
                                         sink=lambda body: None)


def test_delivery_is_one_profiles_own_business(plugin, instance, tmp_path, monkeypatch):
    """Two profiles on one machine have two outboxes and two owner decisions."""
    settings = settings_for(plugin, instance, tmp_path, monkeypatch,
                            DATA_DIR=instance / "data", INFERENCE_ENABLED="false",
                            OWNER_PRINCIPAL=OWNER, DELIVERY_ENABLED="true",
                            DELIVERY_TARGET="signal:owner-1234")
    enrolled(plugin, instance, tmp_path / "homes" / "work")
    activity = plugin.client.bind(tmp_path / "homes" / "work", settings=settings)
    from hermes_memory.storage.evidence import EvidenceStore

    with EvidenceStore(activity.db_path) as profile_store:
        assert profile_store.db.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0
    report = plugin.delivery.deliver_for_home(tmp_path / "homes" / "work",
                                             settings=settings, sink=lambda body: None)
    assert report["profile"] == "work" and report["bank_id"] == "hermes-work"
    assert report["delivered"] == 0
    assert report["reason"] == "nothing is ready to send right now"
    activity.close()


def test_a_disabled_profile_reports_why_rather_than_an_empty_queue(
        plugin, instance, tmp_path, monkeypatch):
    settings = settings_for(plugin, instance, tmp_path, monkeypatch,
                            DATA_DIR=instance / "data", INFERENCE_ENABLED="false",
                            OWNER_PRINCIPAL=OWNER)
    enrolled(plugin, instance, tmp_path / "homes" / "work")
    report = plugin.delivery.deliver_for_home(tmp_path / "homes" / "work",
                                              settings=settings, sink=lambda body: None)
    assert report["ok"] is True and report["delivered"] == 0
    assert "disabled by default" in report["reason"]


@pytest.mark.parametrize("limit", [0, -3, 26, "5", None, True])
def test_a_delivery_run_takes_a_bounded_number_of_artifacts(plugin, instance, limit):
    with pytest.raises(ValueError, match="between 1 and 25"):
        plugin.delivery.deliver_for_home(instance, limit=limit, sink=lambda body: None)


def test_delivery_needs_an_owner_to_notify(plugin):
    """No principal named means nobody may be told anything, not even themselves."""
    blocked = enabled_policy(plugin, owner_principal=None).refusal()
    assert blocked and "owner principal" in blocked


def test_a_run_delivers_this_profiles_artifacts_and_nobody_elses(
        plugin, instance, tmp_path, monkeypatch):
    """Two profiles, two outboxes, and one of them has something to say."""
    settings = settings_for(plugin, instance, tmp_path, monkeypatch,
                            DATA_DIR=instance / "data", INFERENCE_ENABLED="false",
                            OWNER_PRINCIPAL=OWNER, DELIVERY_ENABLED="true",
                            DELIVERY_TARGET="signal:owner-1234")
    homes = {"work": tmp_path / "homes" / "work",
             "personal": tmp_path / "homes" / "personal"}
    for name, home in homes.items():
        enrolled(plugin, instance, home, profile=name)

    from hermes_memory.storage.evidence import EvidenceStore

    work = plugin.client.bind(homes["work"], settings=settings)
    work.data_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    with EvidenceStore(work.db_path) as store:
        attention = AttentionPolicy(store, owner_principal=OWNER)
        attention.configure(actor=OWNER, timezone_name=UTC, quiet_from="22:00",
                            quiet_until="06:00", max_immediate_per_day=10,
                            cooldown_minutes=0, shadow=False)
        events = DueEventLog(store)
        goals = GoalStore(store, events=events, owner_principal=OWNER)
        subject = Outbox(store, policy=attention, owner_principal=OWNER,
                         clock=lambda: EPOCH)
        decision = attention.decide(topic="general", at=MORNING)
        goal_id = goals.propose(title="Send the invoice", statement="Client waiting.",
                                timezone_name=UTC, due=timestamp(MORNING),
                                proposed_by=OWNER, proposed_kind="owner")["id"]
        event_id = store.db.execute("SELECT id FROM due_events WHERE goal_id=?",
                                    (goal_id,)).fetchone()[0]
        claim = events.claim(event_id, holder="worker", at=EPOCH)
        intent = events.ack(event_id=event_id, token=claim.token,
                            decision="awaiting_analysis",
                            policy_version=POLICY_VERSION)["intent"]
        written = attention.record(decision, intent_id=intent, goal_id=goal_id, revision=1)
        prepared = subject.prepare(decision_id=written["id"], kind="notify_owner",
                                   topic="general", payload="The invoice is still owed.")
        assert prepared["prepared"] is True
    work.close()

    seen = []
    report = plugin.delivery.deliver_for_home(homes["work"], settings=settings,
                                              sink=seen.append, at=EPOCH)
    assert report["delivered"] == 1, report
    assert "The invoice is still owed." in seen[0]
    assert report["profile"] == "work"

    untouched = []
    elsewhere = plugin.delivery.deliver_for_home(homes["personal"], settings=settings,
                                                 sink=untouched.append, at=EPOCH)
    assert untouched == [] and elsewhere["delivered"] == 0
    assert elsewhere["reason"] == "nothing is ready to send right now", \
        "the other profile's archive was not even opened"
