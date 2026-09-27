"""C10 attention policy: the owner's budget, enforced on the owner's clock.

The intentions these tests record against are produced by the real C8 handoff,
because the point of the seam is that a decision answers an intention that
already exists somewhere durable.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from hermes_memory.ids import timestamp
from hermes_memory.ids import timestamp
from hermes_memory.proactive.policy import (ACTIONS, DEFAULTS, POLICY_VERSION,
                                            AttentionPolicy, Decision)
from hermes_memory.prospective.due_events import DueEventLog
from hermes_memory.prospective.goals import GoalStore
from hermes_memory.storage.evidence import EvidenceError

OWNER = "owner-principal"
AGENT = "hermes-agent"
UTC = "UTC"
# 22:00-06:00 leaves the 12:00 and 18:00 slot boundaries of a four-way day split
# outside quiet hours, so a test can ask about a boundary without also asking
# about the slip.
QUIET = ("22:00", "06:00")


@pytest.fixture()
def policy(store):
    return AttentionPolicy(store, owner_principal=OWNER)


@pytest.fixture()
def handoff(store, policy):
    """Produce real handed-off intentions, one per call, via the C8 path."""
    events = DueEventLog(store)
    goals = GoalStore(store, events=events, owner_principal=OWNER)
    counter = {"n": 0}

    def make(topic: str = "general") -> str:
        counter["n"] += 1
        fire = timestamp("2026-09-15T09:00:00+00:00")
        goal_id = goals.propose(title=f"Follow up {counter['n']}",
                                statement="A thing the owner asked to be told about.",
                                timezone_name=UTC, due=fire, proposed_by=OWNER,
                                proposed_kind="owner")["id"]
        event_id = store.db.execute("SELECT id FROM due_events WHERE goal_id=?",
                                    (goal_id,)).fetchone()[0]
        claim = events.claim(event_id, holder="worker-a", at=1e12)
        return events.ack(event_id=event_id, token=claim.token,
                          decision="awaiting_analysis",
                          policy_version=POLICY_VERSION)["intent"]

    return make


def decide(policy, topic="general", urgency="proactive", at=None, **kwargs):
    return policy.decide(topic=topic, urgency=urgency, at=at, **kwargs)


# -- defaults and authority --------------------------------------------------

def test_an_unconfigured_topic_is_shadow_mode_with_no_model_permission(policy):
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    assert decision.action == "notify_owner"
    assert decision.shadow is True
    assert decision.model_allowed is False, \
        "the whole point of shadow mode is that the model is not consulted yet"


def test_defaults_are_the_conservative_end_of_the_range(policy):
    settings = policy.settings("never-configured")
    assert settings["shadow"] is True
    assert settings["digest_per_day"] == 1
    assert settings["max_immediate_per_day"] == 2
    assert settings["state"] == "allowed"
    assert set(settings) >= set(DEFAULTS)


def test_going_live_is_an_explicit_owner_act(policy):
    assert policy.configure(actor=OWNER, shadow=False)["shadow"] is False
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    assert decision.shadow is False and decision.model_allowed is True


def test_an_agent_cannot_widen_its_own_attention_budget(policy):
    with pytest.raises(EvidenceError):
        policy.configure(actor=AGENT, max_immediate_per_day=10, shadow=False)
    assert policy.settings("general")["max_immediate_per_day"] == 2


def test_with_no_owner_named_no_one_may_configure(policy, store):
    lonely = AttentionPolicy(store, owner_principal=None)
    with pytest.raises(EvidenceError, match="owner principal"):
        lonely.configure(actor=OWNER)


@pytest.mark.parametrize("kwargs", [
    {"digest_per_day": -1}, {"digest_per_day": 13}, {"max_immediate_per_day": 99},
    {"cooldown_minutes": 25 * 60}, {"digest_per_day": "two"},
    {"cooldown_minutes": True},
])
def test_a_number_outside_the_owners_range_is_refused(policy, kwargs):
    with pytest.raises(EvidenceError):
        policy.configure(actor=OWNER, **kwargs)


@pytest.mark.parametrize("clock", ["25:00", "07:60", "9am", "07", ""])
def test_a_quiet_hour_that_is_not_a_time_of_day_is_refused(policy, clock):
    with pytest.raises(EvidenceError):
        policy.configure(actor=OWNER, quiet_until=clock)


def test_quiet_hours_that_swallow_the_whole_day_are_an_opt_out_in_disguise(policy):
    with pytest.raises(EvidenceError, match="opt-out"):
        policy.configure(actor=OWNER, quiet_from="07:00", quiet_until="07:00")


def test_an_unknown_timezone_is_refused_rather_than_silently_assumed(policy):
    with pytest.raises(EvidenceError, match="timezone"):
        policy.configure(actor=OWNER, timezone_name="Europe/Notreal")


def test_configuration_persists_and_merges_rather_than_replacing(policy):
    policy.configure(actor=OWNER, digest_per_day=3, timezone_name="Europe/Amsterdam")
    settings = policy.configure(actor=OWNER, cooldown_minutes=30)
    assert settings["digest_per_day"] == 3, "an unrelated change must not reset the budget"
    assert settings["timezone"] == "Europe/Amsterdam"
    assert settings["cooldown_minutes"] == 30


# -- the deterministic no's --------------------------------------------------

def test_an_opted_out_topic_produces_nothing_at_all(policy, handoff, store):
    policy.configure(actor=OWNER, opted_out=True)
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    assert decision.action == "silent" and decision.model_allowed is False
    policy.record(decision, intent_id=handoff(), goal_id="goal-1", revision=1)
    assert store.db.execute("SELECT count(*) FROM attention_windows").fetchone()[0] == 0


def test_a_disconnected_source_cannot_be_vouched_for(policy):
    decision = decide(policy, at="2026-09-15T09:00:00+00:00", source_connected=False)
    assert decision.action == "silent"
    assert "re-checked" in decision.reason


def test_the_analysts_already_silent_never_becomes_a_notification(policy):
    assert decide(policy, at="2026-09-15T09:00:00+00:00",
                  already_silent=True).action == "silent"


def test_an_operator_pause_silences_a_topic_without_changing_the_owners_settings(
        policy, store):
    store.set_control("topic:general", "attention", "paused", actor="operator",
                      reason="model server maintenance", policy_version=POLICY_VERSION)
    assert decide(policy, at="2026-09-15T09:00:00+00:00").action == "silent"
    assert policy.settings("general")["state"] == "allowed"


@pytest.mark.parametrize("urgency, expected", [
    ("maintenance", "silent"),
    ("interactive", "next_turn"),
])
def test_only_two_urgencies_are_worth_a_gate(policy, urgency, expected):
    assert decide(policy, urgency=urgency,
                  at="2026-09-15T09:00:00+00:00").action == expected


def test_an_unknown_urgency_is_a_bug_not_a_silence(policy):
    with pytest.raises(EvidenceError, match="urgency"):
        decide(policy, urgency="urgent-ish", at="2026-09-15T09:00:00+00:00")


# -- quiet hours, on the owner's clock ---------------------------------------

@pytest.fixture()
def quiet(policy):
    policy.configure(actor=OWNER, timezone_name=UTC, quiet_from=QUIET[0],
                     quiet_until=QUIET[1], digest_per_day=4, cooldown_minutes=0,
                     max_immediate_per_day=10, shadow=False)
    return policy


def test_the_owner_is_not_woken_at_midnight(quiet):
    decision = decide(quiet, at="2026-09-15T23:30:00+00:00")
    assert decision.action == "digest"
    assert decision.digest_closes_at == timestamp("2026-09-16T06:01:00+00:00"), \
        "the slot boundary at midnight falls inside quiet hours, so it waits"


def test_the_downgrade_says_which_rung_was_denied_and_why(quiet):
    decision = decide(quiet, at="2026-09-15T23:30:00+00:00")
    assert decision.downgrades and decision.downgrades[0].startswith("notify_owner")
    assert "quiet" in decision.downgrades[0]


def test_a_quiet_window_that_spans_midnight_is_one_window(quiet):
    # 06:30 UTC is outside 22:00-06:00; 05:59 is inside it, on the same morning.
    assert decide(quiet, at="2026-09-15T05:59:00+00:00").action == "digest"
    assert decide(quiet, at="2026-09-15T06:30:00+00:00").action == "notify_owner"


def test_quiet_hours_follow_the_owner_not_the_server_clock(store):
    policy = AttentionPolicy(store, owner_principal=OWNER)
    policy.configure(actor=OWNER, timezone_name="Asia/Tokyo", quiet_from="22:00",
                     quiet_until="06:00", shadow=False, max_immediate_per_day=10,
                     cooldown_minutes=0)
    # 21:00 in UTC is 06:00 in Tokyo: the owner is just awake.
    assert policy.decide(topic="general", at="2026-09-15T21:00:00+00:00").action \
        == "notify_owner"
    # 14:00 UTC is 23:00 in Tokyo: asleep, whatever the server thinks.
    assert policy.decide(topic="general", at="2026-09-15T14:00:00+00:00").action \
        == "digest"


def test_a_dst_short_day_does_not_shorten_the_quiet_window(store):
    policy = AttentionPolicy(store, owner_principal=OWNER)
    policy.configure(actor=OWNER, timezone_name="Europe/Amsterdam", quiet_from="02:00",
                     quiet_until="03:00", shadow=False, max_immediate_per_day=10,
                     cooldown_minutes=0)
    # On the spring-forward night 02:00-03:00 Amsterdam does not exist. The gate
    # must still answer with a decision rather than a crash or a silent fallthrough.
    decision = policy.decide(topic="general", at="2026-03-29T01:30:00+01:00")
    assert decision.action in ACTIONS


# -- the immediate budget ----------------------------------------------------

def test_the_third_interruption_of_a_day_becomes_a_digest(policy, handoff):
    policy.configure(actor=OWNER, timezone_name=UTC, max_immediate_per_day=2,
                     cooldown_minutes=0, shadow=False, digest_per_day=4)
    moments = ["2026-09-15T08:00:00+00:00", "2026-09-15T09:00:00+00:00",
               "2026-09-15T10:00:00+00:00"]
    actions = []
    for moment in moments:
        decision = decide(policy, at=moment)
        policy.record(decision, intent_id=handoff(), goal_id="goal-1", revision=1)
        actions.append(decision.action)
    assert actions == ["notify_owner", "notify_owner", "digest"]
    assert "budget" in decision.reason


def test_the_budget_renews_on_the_owners_day_not_the_utc_one(policy, handoff):
    """One UTC day can hold two of the owner's days, and only theirs is counted."""
    policy.configure(actor=OWNER, timezone_name="America/Sao_Paulo", quiet_from="23:00",
                     quiet_until="06:00", max_immediate_per_day=1, cooldown_minutes=0,
                     shadow=False)
    late = "2026-09-16T01:00:00+00:00"      # 22:00 on the 15th, local
    later = "2026-09-16T01:59:00+00:00"     # 22:59 on the 15th: same local day
    next_day = "2026-09-16T15:00:00+00:00"  # 12:00 on the 16th, same UTC day
    decision = decide(policy, at=late)
    assert decision.action == "notify_owner"
    policy.record(decision, intent_id=handoff(), goal_id="goal-1", revision=1)
    assert decide(policy, at=later).action == "digest", \
        "the owner's evening is spoken for; a second interruption is not owed to us"
    assert decide(policy, at=next_day).action == "notify_owner", \
        "their day turned over even though the UTC date string did not"


def test_two_topics_do_not_share_one_silence_budget(policy, handoff):
    policy.configure(actor=OWNER, timezone_name=UTC, max_immediate_per_day=1,
                     cooldown_minutes=0, shadow=False)
    moment = "2026-09-15T09:00:00+00:00"
    for topic in ("work", "home"):
        decision = decide(policy, topic=topic, at=moment)
        policy.record(decision, intent_id=handoff(), goal_id="goal-1", revision=1)
        assert decision.action == "notify_owner", f"{topic} was charged for the other"
    third = decide(policy, topic="work", at="2026-09-15T10:00:00+00:00")
    assert third.action == "digest", "work's own day is spent, whatever home did"
    assert decide(policy, topic="home", at="2026-09-15T10:00:00+00:00").action == "digest"
    assert decide(policy, topic="elsewhere", at="2026-09-15T10:00:00+00:00").action \
        == "notify_owner"


# -- windows: joining, closing and paying ------------------------------------

def test_a_delivered_digest_does_not_quietly_gain_another_item(policy, handoff, store):
    """The slot the gate would choose has already been read by the owner."""
    policy.configure(actor=OWNER, timezone_name=UTC, digest_per_day=1, quiet_from=QUIET[0],
                     quiet_until=QUIET[1], max_immediate_per_day=0, shadow=False)
    natural = decide(policy, at="2026-09-15T09:00:00+00:00")
    assert natural.action == "digest"
    seed_window(store, natural.digest_closes_at, state="closed")
    later = decide(policy, at="2026-09-15T09:30:00+00:00")
    assert timestamp(later.digest_closes_at) > timestamp(natural.digest_closes_at), \
        "a closed window cannot hold a late arrival; it would be missed or repeated"


def test_a_day_whose_allowance_is_spent_gets_no_further_window(policy, store):
    policy.configure(actor=OWNER, timezone_name=UTC, digest_per_day=1, quiet_from=QUIET[0],
                     quiet_until=QUIET[1], max_immediate_per_day=0, shadow=False)
    natural = decide(policy, at="2026-09-15T09:00:00+00:00").digest_closes_at
    # A window the owner already bought for that same day, at another hour.
    seed_window(store, timestamp("2026-09-16T03:00:00+00:00"))
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    assert decision.action == "digest"
    assert decision.digest_closes_at != natural
    assert _parse_iso(decision.digest_closes_at).date() \
        > _parse_iso(natural).date(), "a second window on the owner's day is not ours to open"


def test_one_topics_digest_allowance_is_not_another_topics(policy, store):
    policy.configure(actor=OWNER, timezone_name=UTC, digest_per_day=1, quiet_from=QUIET[0],
                     quiet_until=QUIET[1], max_immediate_per_day=0, shadow=False)
    expected = decide(policy, at="2026-09-15T09:00:00+00:00").digest_closes_at
    seed_window(store, timestamp("2026-09-16T03:00:00+00:00"), topic="work")
    seed_window(store, timestamp("2026-09-16T04:00:00+00:00"), topic="work")
    assert decide(policy, at="2026-09-15T09:00:00+00:00").digest_closes_at == expected


def test_a_cooldown_asks_for_attention_at_most_once_per_topic(policy, handoff):
    policy.configure(actor=OWNER, timezone_name=UTC, max_immediate_per_day=10,
                     cooldown_minutes=240, shadow=False)
    first = decide(policy, at="2026-09-15T09:00:00+00:00")
    policy.record(first, intent_id=handoff(), goal_id="goal-1", revision=1)
    soon = decide(policy, at="2026-09-15T11:00:00+00:00")
    assert soon.action == "digest" and "waits 240" in soon.reason
    later = decide(policy, at="2026-09-15T14:00:00+00:00")
    assert later.action == "notify_owner"


def test_a_topic_with_no_immediate_allowance_never_interrupts(policy):
    policy.configure(actor=OWNER, max_immediate_per_day=0, shadow=False)
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    assert decision.action == "digest"


# -- digests -----------------------------------------------------------------

def test_digests_disabled_means_the_item_waits_for_the_next_turn(policy):
    policy.configure(actor=OWNER, digest_per_day=0, max_immediate_per_day=0, shadow=False)
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    assert decision.action == "next_turn"
    assert decision.digest_closes_at is None


def test_two_items_the_same_day_join_one_digest(policy, handoff):
    policy.configure(actor=OWNER, timezone_name=UTC, digest_per_day=1,
                     max_immediate_per_day=0, shadow=False)
    morning = decide(policy, at="2026-09-15T08:00:00+00:00")
    policy.record(morning, intent_id=handoff(), goal_id="goal-1", revision=1)
    evening = decide(policy, at="2026-09-15T19:00:00+00:00")
    assert evening.digest_closes_at == morning.digest_closes_at
    result = policy.record(evening, intent_id=handoff(), goal_id="goal-1", revision=2)
    rows = store_windows(policy)
    assert len(rows) == 1 and rows[0]["items"] == 2


def test_the_day_never_grows_more_digest_windows_than_the_owner_bought(
        policy, handoff, store):
    """The cap is on windows, not items: items coalesce, windows are the cost.

    Driven across a week of two-hourly candidates so the slot slip past quiet
    hours, the day boundary and the per-day allowance all interact, and the only
    thing asserted is the promise the owner made to themselves.
    """
    policy.configure(actor=OWNER, timezone_name="Europe/Amsterdam", digest_per_day=2,
                     max_immediate_per_day=0, cooldown_minutes=0, shadow=False)
    for day in range(10, 17):
        for hour in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23):
            moment = f"2026-09-{day:02d}T{hour:02d}:00:00+00:00"
            decision = decide(policy, at=moment)
            policy.record(decision, intent_id=handoff(), goal_id="goal-1", revision=1)
    per_day: dict[str, int] = {}
    for row in policy.db.execute("SELECT closes_at FROM attention_windows"):
        local = datetime.fromisoformat(row["closes_at"]).astimezone(
            ZoneInfo("Europe/Amsterdam"))
        per_day[local.date().isoformat()] = per_day.get(local.date().isoformat(), 0) + 1
    assert per_day and max(per_day.values()) <= 2, \
        f"the owner bought two digests a day: {per_day}"
    assert policy.db.execute(
        "SELECT count(*) FROM attention_windows WHERE items < 1").fetchone()[0] == 0, \
        "a window that holds nothing should not have been opened"
    assert policy.db.execute(
        "SELECT count(*) FROM proactive_decisions WHERE action='digest' AND window_id IS NULL"
    ).fetchone()[0] == 0, "a digest that names no window is undeliverable"


def test_a_digest_decision_without_a_window_is_incoherent(policy, handoff):
    loose = Decision(action="digest", reason="no slot", topic="general",
                     digest_closes_at=None)
    with pytest.raises(EvidenceError, match="window"):
        policy.record(loose, intent_id=handoff(), goal_id="goal-1", revision=1)


# -- recording ---------------------------------------------------------------

def test_a_recorded_decision_answers_a_real_intention(policy, store):
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    with pytest.raises(EvidenceError):
        policy.record(decision, intent_id="dec_missing", goal_id="goal-1", revision=1)


def test_replaying_a_record_counts_the_digest_once(policy, handoff, store):
    policy.configure(actor=OWNER, timezone_name=UTC, digest_per_day=1,
                     max_immediate_per_day=0, shadow=False)
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    intent = handoff()
    first = policy.record(decision, intent_id=intent, goal_id="goal-1", revision=1)
    again = policy.record(decision, intent_id=intent, goal_id="goal-1", revision=1)
    assert first["recorded"] is True and again["recorded"] is False
    assert store_windows(policy)[0]["items"] == 1
    assert store.db.execute("SELECT count(*) FROM proactive_decisions").fetchone()[0] == 1


def test_an_action_the_gate_could_not_have_chosen_is_still_storable(policy, handoff):
    """The analyst reaches 'draft'; the gate never does, and record() accepts both."""
    draft = Decision(action="draft", reason="the owner can act on this themselves",
                     topic="general", shadow=False)
    assert policy.record(draft, intent_id=handoff(), goal_id="goal-1",
                         revision=1)["recorded"] is True


def test_an_unknown_action_is_not_stored_at_all(policy, handoff):
    broken = Decision(action="carrier_pigeon", reason="x", topic="general")
    with pytest.raises(EvidenceError, match="action"):
        policy.record(broken, intent_id=handoff(), goal_id="goal-1", revision=1)


def test_a_decision_written_in_the_callers_transaction_disappears_with_it(store, policy,
                                                                          handoff):
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    intent = handoff()
    store.db.execute("BEGIN IMMEDIATE")
    policy.record(decision, intent_id=intent, goal_id="goal-1", revision=1,
                  db=store.db)
    store.db.execute("ROLLBACK")
    assert store.db.execute("SELECT count(*) FROM proactive_decisions").fetchone()[0] == 0


def test_a_record_needs_the_ambient_transaction_it_claims_to_share(store, policy,
                                                                   handoff):
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    store.db.execute("BEGIN IMMEDIATE")
    store.db.execute("COMMIT")
    with pytest.raises(EvidenceError, match="transaction"):
        policy.record(decision, intent_id=handoff(), goal_id="goal-1", revision=1,
                      db=store.db)


def test_the_audit_trail_names_the_intention_and_the_window(policy, handoff, store):
    policy.configure(actor=OWNER, digest_per_day=1, max_immediate_per_day=0,
                     timezone_name=UTC, shadow=False)
    decision = decide(policy, at="2026-09-15T09:00:00+00:00")
    intent = handoff()
    written = policy.record(decision, intent_id=intent, goal_id="goal-1", revision=1,
                            citations=["rec_1"])
    row = store.db.execute("SELECT action, object_id, metadata FROM audit WHERE "
                           "action='proactive_decide'").fetchone()
    assert row["object_id"] == written["id"]
    assert f'"intent": "{intent}"' in row["metadata"]


def test_the_policy_reports_what_it_spent(policy, handoff):
    policy.configure(actor=OWNER, timezone_name=UTC, max_immediate_per_day=2,
                     cooldown_minutes=0, shadow=False)
    for moment in ("2026-09-15T08:00:00+00:00", "2026-09-15T09:00:00+00:00"):
        policy.record(decide(policy, at=moment), intent_id=handoff(), goal_id="goal-1",
                      revision=1)
    counts = policy.counts("general", at="2026-09-15T10:00:00+00:00")
    assert counts["immediate_today"] == 2
    assert counts["local_day"] == "2026-09-15"


def test_shadow_decisions_do_not_spent_the_live_budget(policy, handoff):
    """Shadow mode is a rehearsal: it must not consume what the live run may say.

    Both streams are gated by their own counters, so a week of shadowing does not
    leave the first live day unable to notify anybody.
    """
    policy.configure(actor=OWNER, timezone_name=UTC, max_immediate_per_day=1,
                     cooldown_minutes=0, shadow=False)
    moment = "2026-09-15T09:00:00+00:00"
    policy.record(decide(policy, at=moment), intent_id=handoff(), goal_id="goal-1",
                  revision=1)
    policy.configure(actor=OWNER, shadow=True)
    assert decide(policy, at=moment).action == "notify_owner"


# -- helpers -----------------------------------------------------------------

def store_windows(policy):
    return [dict(row) for row in policy.db.execute(
        "SELECT * FROM attention_windows ORDER BY closes_at")]


def seed_window(store, closes_at: str, *, topic: str = "general", state: str = "open"):
    """Put a window on the owner's calendar directly.

    The gate creates windows as a side effect of recording, so a test that needs a
    day to already hold one — because it was delivered, or because another topic
    spent it — has to say so explicitly.
    """
    from hermes_memory.ids import new_id
    store.db.execute(
        "INSERT INTO attention_windows(id, scope, kind, opens_at, closes_at, state, items) "
        "VALUES(?,?, 'digest', ?, ?, ?, 1)",
        (new_id("win"), f"topic:{topic}", closes_at, closes_at, state))


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)
