"""C2 Hermes event adapter: identity from the host, weight from what actually spoke.

These pin the gate the plan named for adapters — malformed export, encoding,
missing ids, quota and revocation — plus the properties specific to this source: a
model's own answer and a mirrored note describe the conversation and must not be
counted as a second witness to it, and a replayed spool must not arrive looking like
live activity.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from hermes_memory.ids import timestamp
from hermes_memory.sources.base import CursorExpired
from hermes_memory.sources.runtime import (COMPLETE, STALLED, UNREACHABLE,
                                          ConnectorRuntime)
from hermes_memory.sources.sdk import MAX_EVENT_CHARS, HermesEvents
from hermes_memory.sources.sync import SyncController
from hermes_memory.storage.evidence import EvidenceError, prepare_envelope

BEARER = "Bearer yhVn2sQ9kF7pLm3Zx8Rt4Wd6"
API_KEY = "sk-proj-Ab9ZmQ2xKd7Lp4Rt8Vn1Yc3Eg6Jk0Mq"
MISSING = object()


def iso(**shift) -> str:
    return (datetime.now(timezone.utc) + timedelta(**shift)).isoformat()


def event(event_id="e1", *, kind="conversation_turn", session="s-1", created_at=None,
          **payload):
    return {"event_id": event_id, "session_id": session,
            "created_at": created_at if created_at is not None else iso(),
            "payload": {"kind": kind, **payload}}


def turn(event_id="e1", user="Can we move the standup?", assistant="Sure, 10:30.", **extra):
    return event(event_id, user=user, assistant=assistant, **extra)


class Stream:
    """A drain that answers from a list and remembers what it was asked."""

    def __init__(self, *events, raises=None, returns=MISSING, over_supply=False):
        self.events = list(events)
        self.calls: list[tuple] = []
        self.raises = raises
        self.returns = returns
        self.over_supply = over_supply

    def __call__(self, after, limit):
        self.calls.append((after, limit))
        if self.raises is not None:
            raise self.raises
        if self.returns is not MISSING:
            return self.returns
        start = 0 if after is None else next(
            (index + 1 for index, item in enumerate(self.events)
             if item["event_id"] == after), len(self.events))
        return self.events[start:] if self.over_supply else self.events[start:start + limit]


def adapter(*events, **kwargs):
    return HermesEvents(Stream(*events), **kwargs)


def drain_of(*events, **kwargs):
    stream = Stream(*events, **{key: value for key, value in kwargs.items()
                                if key == "over_supply"})
    options = {key: value for key, value in kwargs.items() if key != "over_supply"}
    return HermesEvents(stream, **options), stream


# -- the contract -----------------------------------------------------------

def test_the_event_source_declares_itself_live_and_historic():
    capabilities = HermesEvents(Stream()).capabilities
    assert (capabilities.history, capabilities.live) == (True, True)
    # A spool never tells us a message was withdrawn from the host's own memory.
    assert capabilities.deletion_events is False


def test_a_source_name_carries_into_every_record_it_stamps():
    source, _ = drain_of(turn(), source="hermes-desktop")
    envelopes, _ = source.read_all()
    assert {item["source"] for item in envelopes} == {"hermes-desktop"}


def test_the_drain_is_handed_over_rather_than_imported():
    with pytest.raises(EvidenceError, match="drain callable"):
        HermesEvents("~/data/hermes/capture-spool.db")


@pytest.mark.parametrize("records", [0, 1001])
def test_the_page_size_is_stated_in_the_range_the_store_accepts(records):
    with pytest.raises(ValueError, match="between 1 and 1000"):
        HermesEvents(Stream(), records_per_page=records)


def test_a_drain_that_answers_with_nothing_at_all_is_not_an_empty_page():
    with pytest.raises(EvidenceError, match="sequence of event objects"):
        HermesEvents(Stream(returns=None)).read_page(None)


def test_a_drain_answering_with_a_string_is_refused_rather_than_iterated():
    with pytest.raises(EvidenceError, match="sequence of event objects"):
        HermesEvents(Stream(returns="e1")).read_page(None)


def test_a_drain_may_be_a_generator():
    def stream(after, limit):
        return (item for item in [turn("e1")] if after is None)

    envelopes, _ = HermesEvents(stream).read_all()
    assert [item["source_id"] for item in envelopes] == ["e1#user", "e1#assistant"]


def test_a_position_the_stream_no_longer_recognises_is_returned_not_swallowed():
    source = HermesEvents(Stream(raises=CursorExpired("the spool was compacted")))
    with pytest.raises(CursorExpired):
        source.read_page("e9")
    with pytest.raises(CursorExpired):
        source.check()


# -- identity ---------------------------------------------------------------

def test_one_turn_is_two_records_because_two_things_were_said():
    envelopes, _ = adapter(turn()).read_all()
    assert [item["source_id"] for item in envelopes] == ["e1#user", "e1#assistant"]
    assert [item["metadata"]["role"] for item in envelopes] == ["user", "assistant"]
    assert {item["metadata"]["event_kind"] for item in envelopes} == {"conversation_turn"}


def test_the_user_side_and_the_model_side_do_not_carry_the_same_weight():
    envelopes, _ = adapter(turn()).read_all()
    assert [item["metadata"]["independent"] for item in envelopes] == [True, False]
    assert [item["metadata"]["origin"] for item in envelopes] == [
        "owner-statement", "model-output"]


def test_the_event_id_survives_although_the_records_are_per_side():
    envelopes, _ = adapter(turn()).read_all()
    assert {item["metadata"]["event_id"] for item in envelopes} == {"e1"}
    assert {item["metadata"]["session_id"] for item in envelopes} == {"s-1"}


def test_two_events_saying_the_same_thing_are_two_records_with_one_digest():
    """Copied text is a copy, and only a later stage can tell which one it was."""
    envelopes, _ = adapter(turn("e1", user="Same sentence.", assistant="Same answer."),
                           turn("e2", user="Same sentence.",
                                assistant="Same answer.")).read_all()
    weights = Counter(item["metadata"]["text_digest"] for item in envelopes)
    assert sorted(weights.values()) == [2, 2]
    assert len({item["source_id"] for item in envelopes}) == 4


def test_a_repeated_event_id_is_reported_once_rather_than_committed_twice():
    page = adapter(turn(), turn()).read_page(None)
    assert [item["source_id"] for item in page.envelopes] == ["e1#user", "e1#assistant"]
    assert [(item.ref, item.reason) for item in page.skipped] == [
        ("e1", "the stream reported this event twice in one page")]


def test_an_event_with_no_id_cannot_be_named_and_is_a_gap_not_a_record():
    page = adapter(event(event_id="")).read_page(None)
    assert page.envelopes == ()
    assert page.skipped[0].reason == "the event carried no event_id"


def test_two_different_nameless_events_are_two_gaps_because_they_are_two_faults():
    page = adapter(event(event_id="", user="one"), event(event_id="", user="two")).read_page(None)
    assert len({item.ref for item in page.skipped}) == 2


def test_the_same_nameless_event_twice_is_one_gap_because_it_is_one_fault():
    broken = event(event_id="", user="one")
    page = adapter(broken, dict(broken)).read_page(None)
    # Two rows on the page, and the store keys them as one because they are one
    # indistinguishable fault.
    assert len({item.ref for item in page.skipped}) == 1


def test_an_identifier_longer_than_the_store_writes_is_not_silently_truncated():
    event_id = "x" * 300
    page = adapter(turn(event_id)).read_page(None)
    assert page.skipped[0].reason == "the event id is longer than 200 characters"
    assert page.skipped[0].ref == event_id[:200]
    assert page.envelopes == ()


def test_an_event_that_is_not_an_object_is_a_gap_with_a_reference_anyway():
    page = adapter("not an event").read_page(None)
    assert page.skipped[0].ref.startswith("<unaddressed>:")
    assert page.next_cursor == page.skipped[0].ref


# -- kinds we can read, and kinds we cannot ---------------------------------

def test_an_unknown_kind_is_a_gap_named_by_the_event_that_will_close_it():
    page = adapter(event("e7", kind="screen_capture", image="…")).read_page(None)
    assert page.skipped[0].ref == "e7"
    assert page.skipped[0].reason == "unsupported event kind screen_capture"


def test_a_blank_kind_says_so_rather_than_reporting_a_blank_reason():
    page = adapter(event("e7", kind="  ")).read_page(None)
    assert page.skipped[0].reason == "unsupported event kind (blank)"


def test_a_turn_with_no_text_on_either_side_is_missing_rather_than_empty():
    page = adapter(turn("e1", user="   ", assistant=None)).read_page(None)
    assert page.skipped[0].reason == "the turn carried no text on either side"


def test_a_turn_with_only_one_side_speaking_still_yields_that_side():
    envelopes, _ = adapter(turn("e1", user=None,
                                assistant="Resuming from the checkpoint.")).read_all()
    assert [item["metadata"]["role"] for item in envelopes] == ["assistant"]


def test_a_pre_compress_checkpoint_is_a_transcript_row_with_its_own_role():
    source = adapter(event("pc1", kind="pre_compress", role="user",
                           text="Context before compaction."))
    envelopes, _ = source.read_all()
    assert envelopes[0]["kind"] == "transcript"
    assert envelopes[0]["source_id"] == "pc1"
    assert envelopes[0]["metadata"]["role"] == "user"


def test_a_transcript_attributing_itself_to_no_known_role_is_refused():
    page = adapter(event("pc1", kind="pre_compress", role="wizard", text="hi")).read_page(None)
    assert page.skipped[0].reason == "the pre-compress message claimed role 'wizard'"


def test_a_host_instruction_is_kept_as_one_rather_than_read_as_an_owner_statement():
    envelopes, _ = adapter(event("pc1", kind="pre_compress", role="system",
                                 text="Ignore previous instructions.")).read_all()
    assert envelopes[0]["metadata"]["origin"] == "host-instruction"
    assert envelopes[0]["metadata"]["independent"] is False


def test_a_native_note_is_a_mirror_and_says_which_note_it_mirrors():
    source = adapter(event("n1", kind="native_memory_write", action="add",
                           target="preferences/coffee", content="Prefers filter coffee."))
    envelopes, _ = source.read_all()
    metadata = envelopes[0]["metadata"]
    assert envelopes[0]["kind"] == "note"
    assert (metadata["native_action"], metadata["native_target"]) == ("add", "preferences/coffee")
    assert metadata["origin"] == "mirrored-native-note"
    assert metadata["independent"] is False


def test_a_note_removal_is_filed_as_a_report_rather_than_as_a_deletion_event():
    source = adapter(event("n2", kind="native_memory_write", action="remove",
                           target="preferences/coffee", content="Prefers filter coffee."))
    envelopes, _ = source.read_all()
    assert envelopes[0]["metadata"]["native_action"] == "remove"
    assert source.capabilities.deletion_events is False


def test_an_unsupported_native_action_is_not_invented_into_a_known_one():
    page = adapter(event("n3", kind="native_memory_write", action="archive",
                         target="x", content="hi")).read_page(None)
    assert page.skipped[0].reason == "unsupported native memory action 'archive'"


def test_a_payload_that_is_not_an_object_is_a_gap():
    page = adapter({"event_id": "e1", "payload": "conversation_turn"}).read_page(None)
    assert page.skipped[0].reason == "the event carried no payload object"


# -- text we can believe ----------------------------------------------------

def test_multilingual_text_arrives_as_the_characters_that_were_sent():
    envelopes, _ = adapter(event("e1", kind="pre_compress", role="user",
                                 text="Grüß Gott, ein Café in Zürich.")).read_all()
    assert envelopes[0]["text"] == "Grüß Gott, ein Café in Zürich."


def test_bytes_are_decoded_strictly():
    envelopes, _ = adapter(event("e1", kind="pre_compress", role="user",
                                 text="Café".encode("utf-8"))).read_all()
    assert envelopes[0]["text"] == "Café"


def test_bytes_that_are_not_utf8_are_a_gap_rather_than_replacement_characters():
    page = adapter(event("e1", kind="pre_compress", role="user", text=b"Caf\xe9")).read_page(None)
    assert page.skipped[0].reason == (
        "the pre-compress message had no text that could be decoded")


def test_damaged_text_is_not_stored_as_if_it_had_been_understood():
    page = adapter(event("e1", kind="pre_compress", role="user",
                         text="Caf\ufffd in the\ufffdir")).read_page(None)
    assert page.envelopes == ()


def test_text_that_is_not_text_at_all_is_missing_rather_than_empty():
    page = adapter(event("e1", kind="pre_compress", role="user", text={"nested": 1})).read_page(None)
    assert page.skipped[0].reason == "the pre-compress message had no text that could be decoded"


def test_an_oversized_body_is_bounded_and_says_that_it_was():
    envelopes, _ = adapter(event("e1", kind="pre_compress", role="user",
                                 text="a" * (MAX_EVENT_CHARS + 5_000))).read_all()
    assert len(envelopes[0]["text"]) == MAX_EVENT_CHARS
    assert envelopes[0]["metadata"]["truncated"] is True
    assert envelopes[0]["metadata"]["chars"] == MAX_EVENT_CHARS


def test_a_body_that_stops_short_of_the_bound_does_not_claim_truncation():
    envelopes, _ = adapter(event("e1", kind="pre_compress", role="user",
                                 text="Short.")).read_all()
    assert "truncated" not in envelopes[0]["metadata"]


# -- credentials ------------------------------------------------------------

def test_a_token_in_transcribed_text_does_not_reach_the_record():
    envelopes, _ = adapter(turn("e1", user=f"Here is the key: {BEARER} — store it safely.")).read_all()
    assert BEARER not in json.dumps(envelopes)
    assert "[redacted]" in envelopes[0]["text"]


def test_an_api_key_shaped_string_does_not_reach_the_record_either():
    envelopes, _ = adapter(event("e1", kind="pre_compress", role="user",
                                 text=f"config: {API_KEY}")).read_all()
    assert API_KEY not in json.dumps(envelopes)


@pytest.mark.parametrize("field", ["api_key", "access_token", "authorization", "password"])
def test_a_secret_shaped_author_field_is_dropped_and_counted(field):
    envelopes, _ = adapter(turn("e1", author={field: BEARER, "name": "Ada"})).read_all()
    author = envelopes[0]["metadata"]["author"]
    assert field not in author
    assert author["name"] == "Ada"
    assert author["redacted_fields"] == 1


def test_a_nested_author_block_is_not_copied_into_metadata_wholesale():
    envelopes, _ = adapter(turn("e1", author={"name": "Ada", "runtime": {"routes": [1, 2]}})).read_all()
    assert envelopes[0]["metadata"]["author"] == {"name": "Ada", "redacted_fields": 1}


def test_attribution_is_bounded_however_much_the_host_offers():
    author = {f"field_{index}": "v" * 400 for index in range(40)}
    envelopes, _ = adapter(turn("e1", author=author)).read_all()
    kept = envelopes[0]["metadata"]["author"]
    assert len([key for key in kept if key != "redacted_fields"]) == 10
    assert kept["field_0"] == "v" * 200
    assert kept["redacted_fields"] == 30


def test_an_author_block_that_is_not_a_mapping_yields_no_attribution():
    envelopes, _ = adapter(turn("e1", author="Ada Lovelace")).read_all()
    assert envelopes[0]["metadata"]["author"] is None


def test_no_envelope_from_this_adapter_carries_a_credential_it_was_shown():
    source = adapter(turn("e1", author={"token": BEARER}, user=BEARER))
    for envelope in source.read_all()[0]:
        prepare_envelope(envelope)
        assert BEARER not in json.dumps(envelope)


# -- time -------------------------------------------------------------------

def test_an_absolute_instant_from_the_host_is_kept_to_the_second():
    envelopes, _ = adapter(turn("e1", occurred_at="2026-09-24T09:15:00+02:00")).read_all()
    assert envelopes[0]["occurred_at"] == "2026-09-24T07:15:00+00:00"
    assert envelopes[0]["occurred_precision"] == "second"
    assert envelopes[0]["metadata"]["time_basis"] == "event field"
    assert "time_note" not in envelopes[0]["metadata"]


def test_a_clock_reading_with_no_zone_is_not_a_moment():
    envelopes, _ = adapter(turn("e1", occurred_at="2026-09-24T09:15:00")).read_all()
    assert envelopes[0]["occurred_at"] is None
    assert envelopes[0]["occurred_precision"] == "unknown"
    assert envelopes[0]["metadata"]["time_basis"] == "none"
    assert "timezone" in envelopes[0]["metadata"]["time_note"]


def test_an_event_that_reports_no_time_is_dated_by_nobody():
    envelopes, _ = adapter(turn("e1")).read_all()
    assert envelopes[0]["occurred_at"] is None
    assert envelopes[0]["metadata"]["time_basis"] == "none"


def test_observing_an_old_event_today_does_not_make_it_happen_today():
    old = event("e1", created_at=iso(days=-9), user="then", assistant="after")
    envelopes, _ = adapter(old).read_all()
    assert envelopes[0]["observed_at"] >= timestamp(iso(days=-9))
    assert envelopes[0]["occurred_at"] is None


# -- arrival ----------------------------------------------------------------

def test_when_the_host_accepted_the_event_is_kept_as_the_host_said_it():
    when = iso(minutes=-30)
    envelopes, _ = adapter(event("e1", created_at=when, user="x", assistant="y")).read_all()
    assert envelopes[0]["metadata"]["spooled_at"].startswith(when[:19])


def test_an_arrival_time_with_no_zone_is_not_a_date_at_all():
    envelopes, _ = adapter(event("e1", created_at="2026-09-24T09:15:00", user="x",
                                 assistant="y")).read_all()
    assert envelopes[0]["metadata"]["spooled_at"] is None


def test_a_second_read_of_the_same_event_describes_it_identically():
    """Otherwise re-offering one page under a new name is a revision conflict.

    Nothing a read can notice about the present moment may enter the metadata: the
    fingerprint covers it, and a description that drifts between reads would make
    the store refuse the same evidence the second time it was offered. When we saw
    it is a column of its own, and is meant to move.
    """
    spooled = iso(days=-3)
    source, _ = drain_of(event("e1", created_at=spooled, user="x", assistant="y"))
    first = source.read_page(None).envelopes
    second = source.read_page(None).envelopes
    assert [prepare_envelope(item).fingerprint for item in first] == \
           [prepare_envelope(item).fingerprint for item in second]


def test_a_replayed_spool_still_carries_the_date_it_was_accepted():
    """The one fact that tells a three-week backlog from this morning's session."""
    envelopes, _ = adapter(event("e1", created_at=iso(days=-21), user="x",
                                 assistant="y")).read_all()
    assert envelopes[0]["metadata"]["spooled_at"] < iso(days=-20)
    assert envelopes[0]["observed_at"] > iso(days=-1)


# -- paging -----------------------------------------------------------------

def test_an_empty_read_is_the_end_of_what_the_stream_has_now():
    page = adapter().read_page(None)
    assert (page.envelopes, page.skipped, page.next_cursor) == ((), (), None)


def test_a_page_ends_where_the_declared_page_size_does():
    events = [turn(f"e{index}") for index in range(20)]
    source, stream = drain_of(*events, records_per_page=4)
    page = source.read_page(None)
    assert [item["source_id"] for item in page.envelopes] == [
        "e0#user", "e0#assistant", "e1#user", "e1#assistant"]
    assert stream.calls == [(None, 4)]
    assert page.next_cursor == "e1"


def test_one_event_is_never_split_across_two_pages():
    """A question on one commit and its answer on the next is a broken record."""
    source, _ = drain_of(*[turn(f"e{index}") for index in range(6)], records_per_page=1)
    page = source.read_page(None)
    assert [item["source_id"] for item in page.envelopes] == ["e0#user", "e0#assistant"]
    assert page.next_cursor == "e0"


def test_the_position_advances_across_pages_without_losing_the_tail():
    events = [turn(f"e{index}") for index in range(6)]
    source, _ = drain_of(*events, records_per_page=2)
    cursor, seen = None, []
    for _ in range(10):
        page = source.read_page(cursor)
        seen.extend(item["source_id"] for item in page.envelopes)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert seen == [f"e{index}#{role}" for index in range(6) for role in ("user", "assistant")]


def test_a_position_is_reported_even_when_the_page_ran_out_of_text():
    """A live stream re-read from an old position would re-read the whole spool."""
    page = adapter(event("e1", kind="voice_note", transcript="x"),
                   event("e2", kind="voice_note", transcript="y")).read_page(None)
    assert page.envelopes == ()
    assert page.next_cursor == "e2"


def test_a_drain_that_ignores_the_page_size_is_still_bounded_by_the_adapter():
    events = [turn(f"e{index}") for index in range(400)]
    source, _ = drain_of(*events, over_supply=True)
    page = source.read_page(None)
    assert len(page.envelopes) <= 200


def test_a_stream_that_offers_gaps_forever_still_advances_rather_than_spinning():
    events = [event(f"e{index}", kind="unknown") for index in range(400)]
    source, _ = drain_of(*events, over_supply=True, records_per_page=5)
    page = source.read_page(None)
    assert len(page.skipped) == 20
    assert page.next_cursor == "e19"


def test_a_byte_budget_stops_the_page_before_an_event_rather_than_after_it():
    events = [event(f"e{index}", kind="pre_compress", role="user", text="a" * 90_000)
              for index in range(4)]
    source, _ = drain_of(*events)
    source.capabilities = replace(source.capabilities, max_bytes_per_page=60_000)
    page = source.read_page(None)
    assert len(page.envelopes) == 1
    assert page.next_cursor == "e0"


def test_a_stream_that_re_serves_its_own_head_is_stalled_rather_than_reread_forever(store):
    """A drain that ignores the position would page until the budget ran out otherwise."""
    sync = SyncController(store)
    sync.register("hermes", policy_version="local-only")
    repeat = [turn("e1")]
    run = ConnectorRuntime(store, sync, holder="worker-a").run(HermesEvents(
        lambda after, limit: repeat))
    assert (run.stopped, run.pages, run.records) == (STALLED, 2, 2)


def test_a_stream_that_raises_on_a_read_is_an_outage_not_an_empty_archive():
    source = HermesEvents(Stream(raises=RuntimeError("429 quota exhausted")))
    with pytest.raises(EvidenceError, match="429 quota exhausted"):
        source.read_page(None)


def test_a_check_never_reads_content_but_does_report_the_backlog_age():
    old = event("e1", created_at=iso(minutes=-30), user="x", assistant="y")
    source, stream = drain_of(old, turn("e2"))
    report = source.check()
    assert report["ok"] is True and report["content_read"] is False
    assert report["pending"] is True
    assert report["oldest_at"].startswith(old["created_at"][:13])
    assert stream.calls == [(None, 1)]


def test_a_check_on_a_quiet_pipe_says_so_rather_than_inventing_a_backlog():
    report = adapter().check()
    assert report["ok"] is True and report["pending"] is False and report["oldest_at"] is None


def test_a_check_reports_a_refused_pipe_instead_of_raising_through_the_runtime():
    report = HermesEvents(Stream(raises=PermissionError("the token was revoked"))).check()
    assert report["ok"] is False
    assert "the token was revoked" in report["reason"]
    assert report["content_read"] is False


# -- end to end -------------------------------------------------------------

def test_a_spool_becomes_evidence_once_and_a_second_poll_adds_nothing(store):
    sync = SyncController(store)
    sync.register("hermes", policy_version="local-only")
    events = [turn("e1"), event("e2", created_at=iso(days=-3), user="older", assistant="a"),
              event("e3", kind="voice_note", transcript="x")]
    stream = Stream(*events)
    runtime = ConnectorRuntime(store, sync, holder="worker-a")

    first = runtime.run(HermesEvents(stream))
    assert (first.stopped, first.records, first.gaps) == (COMPLETE, 4, 1)
    assert first.coverage_state == "current"
    assert first.cursor == "e3"
    assert first.pages == 2
    assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 4

    second = runtime.run(HermesEvents(stream))
    assert (second.records, second.repeats, second.stopped) == (0, 1, COMPLETE)
    assert second.coverage_state == "current"


def test_what_the_stream_could_not_give_stays_visible_after_the_pass(store):
    sync = SyncController(store)
    sync.register("hermes", policy_version="local-only")
    runtime = ConnectorRuntime(store, sync, holder="worker-a")
    runtime.run(HermesEvents(Stream(event("e3", kind="voice_note", transcript="x"))))

    gaps = sync.gaps("hermes")
    assert [row["ref"] for row in gaps] == ["e3"]
    assert gaps[0]["reason"] == "unsupported event kind voice_note"


def test_a_pipe_that_refuses_the_probe_is_filed_as_unreachable_without_a_crash(store):
    sync = SyncController(store)
    sync.register("hermes", policy_version="local-only")
    runtime = ConnectorRuntime(store, sync, holder="worker-a")
    run = runtime.run(HermesEvents(Stream(raises=PermissionError("scope revoked"))))
    assert run.stopped == UNREACHABLE
    assert "scope revoked" in run.note
    assert sync.state("hermes")["coverage_state"] == "unreachable"


def test_a_day_is_not_an_arrival_instant():
    envelopes, _ = adapter(event("e1", created_at="2026-09-24", user="x",
                                 assistant="y")).read_all()
    assert envelopes[0]["metadata"]["spooled_at"] is None


def test_a_mirrored_note_is_not_attributed_to_the_person_it_quotes():
    envelopes, _ = adapter(event("n1", kind="native_memory_write", action="add",
                                 target="preferences/tea", content="Likes tea.")).read_all()
    metadata = envelopes[0]["metadata"]
    assert metadata["origin"] == "mirrored-native-note"
    assert metadata["role"] == "note"
