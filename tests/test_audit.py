"""C14 audit: the ledger is only worth having if reading it back cannot lie.

The three failure modes this suite is aimed at are a filter that ignores its argument
and prints everything, a metadata blob that carries a credential into a log line, and
a report that quietly writes while it is reading.
"""
from __future__ import annotations

import json

import pytest

from hermes_memory.operations.audit import MAX_METADATA_FIELDS, AuditTrail
from hermes_memory.sources.base import Skipped
from hermes_memory.storage.evidence import EvidenceError

from conftest import envelope

OWNER = "owner:judge"
AGENT = "hermes-agent"
SECRET = "sk-supersecretvalue1"
STAMP = "2026-09-25T12:00:00+00:00"


def insert(store, table, **columns):
    names = ", ".join(columns)
    store.db.execute(f"INSERT INTO {table}({names}) "
                     f"VALUES({','.join('?' * len(columns))})", list(columns.values()))


def audited(store, action, object_id, metadata=None, at=STAMP):
    insert(store, "audit", action=action, object_id=object_id, created_at=at,
           metadata=json.dumps(metadata if metadata is not None else {}))
    return store.db.execute("SELECT max(id) FROM audit").fetchone()[0]


@pytest.fixture()
def trail(store):
    return AuditTrail(store)


def a_record(store, **overrides):
    return store.commit(envelope(**overrides))["id"]


# -- the ledger --------------------------------------------------------------

def test_the_newest_rows_come_first_and_no_more_than_were_asked_for(store, trail):
    ids = [audited(store, action, "rec_a", at=f"2026-01-0{index}:00:00+00:00")
           for index, action in enumerate(("evidence_commit", "evidence_hide",
                                           "evidence_show"), start=1)]
    assert [item["id"] for item in trail.recent(limit=2)] == [ids[2], ids[1]]
    assert [item["id"] for item in trail.recent(limit=3)] == [ids[2], ids[1], ids[0]]


def test_a_filter_that_matches_nothing_returns_nothing(store, trail):
    audited(store, "evidence_commit", "rec_a")
    assert trail.recent(action="erasure_confirm") == []
    assert trail.recent(object_id="rec_not_here") == []


def test_an_unknown_action_is_not_mistaken_for_no_filter(store, trail):
    audited(store, "evidence_commit", "rec_a")
    audited(store, "evidence_hide", "rec_a")
    assert [item["action"] for item in trail.recent(action="evidence_hide")] == [
        "evidence_hide"]


def test_a_blank_filter_is_an_error_rather_than_a_wildcard(trail):
    with pytest.raises(EvidenceError, match="nonempty text"):
        trail.recent(action="  ")
    with pytest.raises(EvidenceError, match="nonempty text"):
        trail.recent(object_id="")


@pytest.mark.parametrize("limit", [0, -1, 5000, "many", True, None])
def test_the_row_bound_is_enforced_before_anything_is_read(store, trail, limit):
    audited(store, "evidence_commit", "rec_a")
    with pytest.raises(EvidenceError, match="between 1 and"):
        trail.recent(limit=limit)


def test_rows_are_filtered_by_the_actor_named_in_the_metadata(store, trail):
    audited(store, "evidence_hide", "rec_a", {"actor": OWNER})
    audited(store, "evidence_show", "rec_a", {"actor": AGENT})
    audited(store, "evidence_commit", "rec_a", {"source": "gmail"})
    assert [item["action"] for item in trail.recent(actor=OWNER)] == ["evidence_hide"]
    assert trail.recent(actor="nobody") == []


def test_what_has_happened_here_is_counted_by_kind(store, trail):
    audited(store, "evidence_commit", "rec_a")
    audited(store, "evidence_commit", "rec_b")
    audited(store, "erasure_confirm", "er-1")
    assert trail.actions() == {"evidence_commit": 2, "erasure_confirm": 1}


def test_an_actor_filter_matches_the_whole_name_not_a_piece_of_it(store, trail):
    audited(store, "evidence_hide", "rec_a", {"actor": OWNER})
    audited(store, "evidence_hide", "rec_b", {"actor": f"{OWNER}-impersonating"})
    only = trail.recent(actor=OWNER)
    assert [item["object_id"] for item in only] == ["rec_a"]


def test_who_has_been_acting_is_counted_from_the_rows_saying_so(store, trail):
    for _ in range(3):
        audited(store, "evidence_hide", "rec_x", {"actor": OWNER})
    audited(store, "evidence_hide", "rec_y", {"actor": AGENT})
    audited(store, "evidence_commit", "rec_z", {"source": "gmail"})
    assert list(trail.by_actor()) == [OWNER, AGENT], "busiest actor first"
    assert trail.by_actor() == {OWNER: 3, AGENT: 1}


def test_owner_only_decisions_are_attributed_to_the_person_who_made_them(store, trail):
    for account in ("a", "b"):
        insert(store, "identity_accounts", id=account, namespace="email",
               identifier=f"{account}@example.com", normalized=f"{account}@example.com",
               created_at=STAMP)
    insert(store, "identity_candidates", id="cand-1", account_a="a", account_b="b",
           rule="same-name", rule_version="v1", basis="name", evidence="[]",
           proposed_by=AGENT, proposed_kind="agent", proposed_at=STAMP,
           state="confirmed", decided_by=OWNER, decided_at=STAMP)
    insert(store, "identity_edges", id="edge-1", account_a="a", account_b="b",
           candidate_id="cand-1", state="active", confirmed_by=OWNER, confirmed_at=STAMP)
    assert trail.decided_by("identity") == [{"actor": OWNER, "decisions": 1}]
    assert trail.decided_by("identity-decisions") == [{"actor": OWNER, "decisions": 1}]


def test_an_invented_decision_category_is_refused(store, trail):
    with pytest.raises(EvidenceError, match="unknown decision category"):
        trail.decided_by("feelings")


# -- secrets and shape -------------------------------------------------------

def test_a_credential_in_audit_metadata_never_reaches_the_report(store, trail):
    audited(store, "evidence_hide", "rec_a", {"actor": OWNER, "note": f"key {SECRET} used"})
    dumped = json.dumps(trail.recent())
    assert SECRET not in dumped
    assert "[redacted]" in dumped


def test_a_wide_metadata_blob_is_bounded_and_says_so(store, trail):
    fields = {f"f{index}": index for index in range(MAX_METADATA_FIELDS + 5)}
    audited(store, "snapshot_create", "snap-1", fields)
    entry = trail.recent()[0]
    assert len([key for key in entry if key not in ("id", "action", "object_id", "at")]) == \
        MAX_METADATA_FIELDS + 1
    assert entry["additional_fields"] == 5


def test_metadata_that_is_not_an_object_is_reported_as_that(store, trail):
    insert(store, "audit", action="x", object_id="o", created_at=STAMP,
           metadata=json.dumps(["not", "an", "object"]))
    insert(store, "audit", action="y", object_id="o", created_at=STAMP, metadata="{oops")
    shapes = {item["action"]: item for item in trail.recent()}
    assert shapes["x"]["unexpected_shape"] == "list"
    assert shapes["y"]["unparseable"] is True


def test_long_metadata_values_are_cut_down_rather_than_streamed(store, trail):
    audited(store, "erasure_preview", "er-1", {"preview": "private text " * 500})
    value = trail.recent()[0]["preview"]
    assert len(value) <= 400


# -- one thing's history -----------------------------------------------------

def test_a_timeline_answers_when_it_stopped_being_private(store, trail):
    record = a_record(store)
    store.hide(record, reason="mine only", actor=OWNER)
    report = trail.timeline(record)
    assert report["retrievable"] == {"in_search_index": False, "hidden": True,
                                     "hidden_reason": "mine only", "replaced_by": None,
                                     "tombstoned_at": None}
    assert [item["action"] for item in report["audit"]] == ["evidence_hide",
                                                            "evidence_commit"]
    assert [item["change"] for item in report["changes"]] == ["add", "hide"]
    assert report["evidence"]["source"] == "gmail"


def test_a_timeline_does_not_carry_the_evidence_it_describes(store, trail):
    record = a_record(store, text="My diagnosis was confirmed on Tuesday.")
    report = trail.timeline(record)
    assert "text" not in report
    assert "diagnosis" not in json.dumps(report, default=str)


def test_evidence_text_appears_only_when_it_is_asked_for_and_then_redacted(store, trail):
    record = a_record(store, text=f"diagnosis: {SECRET}")
    report = trail.timeline(record, include_text=True)
    assert SECRET not in report["text"]
    assert "[redacted]" in report["text"]
    assert len(report["text"]) <= 400


def test_a_timeline_reports_support_and_what_was_built_from_it(store, trail):
    record = a_record(store)
    insert(store, "assertions", id="asr-1", subject="owner", predicate="mood",
           value="calm", category="profile", evidence_kind="explicit_statement",
           record_id=record, quote_start=0, quote_end=4, quote="The ",
           status="confirmed", created_by=OWNER, created_at=STAMP)
    insert(store, "derived_citations", artifact_id="sum-1", kind="summary",
           coverage="full", record_id=record, added_at=STAMP)
    report = trail.timeline(record)
    assert report["supported_by"][0]["id"] == "asr-1"
    assert report["built_into"][0]["artifact_id"] == "sum-1"


def test_an_unknown_record_has_no_timeline_rather_than_an_empty_one(store, trail):
    with pytest.raises(EvidenceError, match="no record"):
        trail.timeline("rec_nope")


def test_a_forgotten_record_still_has_its_history(store, trail):
    record = a_record(store)
    insert(store, "erasure_ledger", id="er-1", source="gmail", requested_at=STAMP,
           requested_by=OWNER, requester_kind="owner", reason="private", preview="{}",
           preview_digest="d" * 16, state="complete", epoch=1)
    insert(store, "tombstones", record_id=record, intent_id="er-1",
           fingerprint="f" * 16, deleted_at=STAMP)
    store.db.execute("UPDATE records SET deleted=1 WHERE id=?", (record,))
    report = trail.timeline(record)
    assert report["evidence"]["forgotten"] is True
    assert report["retrievable"]["tombstoned_at"] == STAMP


# -- a source's own account --------------------------------------------------

def test_a_source_history_shows_pages_gaps_and_controls(store, trail, sync):
    sync.register("gmail", policy_version="local-only")
    fence = sync.acquire("gmail", holder="worker")
    sync.publish(fence, "page-1", [envelope()], skipped=[Skipped("gmail.csv#4",
                                                                 "a row that is not a number")])
    sync.release(fence)
    sync.pause("gmail", actor=OWNER, reason="privacy review", policy_version="v1",
               stages=("capture",))
    report = trail.source_history("gmail")
    assert report["pages"][0]["records"] == 1
    assert report["gaps"][0]["reason"] == "a row that is not a number"
    assert report["gaps"][0]["ref"] == "gmail.csv#4"
    assert report["controls"] == [
        {"stage": "capture", "state": "paused", "actor": OWNER, "reason": "privacy review",
         "policy_version": "v1", "changed_at": report["controls"][0]["changed_at"]}]
    assert report["connector"]["generation"] == 1


def test_an_unregistered_source_is_not_reported_as_having_done_nothing(trail):
    with pytest.raises(EvidenceError, match="no connector named"):
        trail.source_history("gmail")


def test_a_source_with_no_pages_yet_says_so(trail, sync):
    sync.register("gmail", policy_version="local-only")
    report = trail.source_history("gmail")
    assert report["pages"] == [] and report["gaps"] == []


# -- the change journal ------------------------------------------------------

def test_the_journal_starts_one_past_the_point_the_caller_last_saw(store, trail):
    a_record(store)
    assert [item["seq"] for item in trail.journal(after=0)] == [1]
    assert trail.journal(after=1) == []


def test_a_journal_position_that_is_not_a_number_is_refused(trail):
    with pytest.raises(EvidenceError, match="nonnegative journal sequence"):
        trail.journal(after=-1)
    with pytest.raises(EvidenceError, match="nonnegative journal sequence"):
        trail.journal(after="first")


# -- the reading is a reading ------------------------------------------------

def test_reading_the_ledger_writes_nothing(store, trail):
    a_record(store)
    store.db.execute("INSERT INTO audit(action, object_id, created_at, metadata) "
                     "VALUES('x','y',?, '{}')", (STAMP,))
    tables = [row[0] for row in store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]

    def footprint():
        return {table: store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in tables}

    before = footprint()
    trail.recent()
    trail.actions()
    trail.by_actor()
    trail.journal(after=0)
    assert footprint() == before
    assert store.db.in_transaction is False


def test_a_registered_connector_is_reported_as_never_having_run(store, trail, sync):
    sync.register("gmail", policy_version="local-only")
    report = trail.source_history("gmail")
    assert report["connector"]["coverage_state"] == "unknown"
    assert report["connector"]["generation"] == 1
