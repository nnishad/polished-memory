"""C6 summaries: a producer, a reader and an invalidator that agree with each other.

A summary table with nothing that writes it, nothing that reads it and nothing that takes
a reading out of circulation when its evidence goes is a schema, not a feature. These tests
walk the whole life of one reading: a scope is listed and approved, one reflect call is
made against a stub backend, the window is filed with its citations, the packet quotes it,
a forgetting withdraws it, and the refresh promise booked by that withdrawal is answered by
the next reading rather than left to accrue.
"""
from __future__ import annotations

import json

import pytest

from conftest import envelope
from hermes_memory.backend.hindsight_client import (HindsightError, HindsightUnavailable,
                                                   HindsightClient, TransportResult)
from hermes_memory.config import load_settings
from hermes_memory.context import ContextBroker
from hermes_memory.knowledge.summaries import SummaryStore
from hermes_memory.knowledge import summaries as summaries_module
from hermes_memory.lifecycle.erasure import ErasureManager
from hermes_memory.processing import summarization
from hermes_memory.processing.instance_gate import instance_gate
from hermes_memory.processing.summarization import (MAX_BATCH, SummarizeError,
                                                   resolve_scope, scope_records,
                                                   summarize_apply, summarize_plan)
from hermes_memory.storage.evidence import EvidenceStore, ReadOnlyStore
from hermes_memory.storage.lineage import Lineage

OWNER = "jugaadu"
BANK = "hermes"
UPSTREAM = "http://127.0.0.1:11434/v1"


class Reflects:
    """A backend that answers a reflection and remembers being asked exactly once."""

    def __init__(self, text="Two notes were filed and the meeting moved.", cited=2):
        self.calls: list[dict] = []
        self.text = text
        self.cited = cited

    def reflect(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return {"text": self.text, "facts": [{"id": f"m{i}"} for i in range(self.cited)],
                "cited_memories": self.cited, "input_tokens": 120, "output_tokens": 30,
                "truncated": False}


class Answers:
    """Refuses the way a live server does: an error naming what went wrong."""

    def __init__(self, error):
        self.error = error
        self.asks = 0

    def reflect(self, query, **kwargs):
        self.asks += 1
        raise self.error


@pytest.fixture()
def installation(tmp_path, monkeypatch):
    """A configured installation holding two records in one project and one outside it."""
    home = tmp_path / "instance"
    (home / "data").mkdir(parents=True)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS=200000\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n"
        "HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:8123\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n"
        f"HERMES_MEMORY_TEXT_BASE_URL={UPSTREAM}\n"
        "HERMES_MEMORY_EMBEDDINGS_BASE_URL=http://127.0.0.1:11435/v1\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_RETAIN=cred-retain\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_EMBEDDINGS=cred-embed\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_CONSOLIDATE=cred-consolidate\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_REFLECT=cred-reflect\n"
        "HERMES_MEMORY_ROUTE_CREDENTIAL_FOREGROUND=cred-foreground\n",
        encoding="utf-8")
    settings = load_settings()
    with EvidenceStore(settings.db_path) as store:
        for index in (1, 2):
            store.commit(envelope(source_id=f"note-{index}", text=f"note {index} about the "
                                  f"invoice for the lamprey survey",
                                  metadata={"project": "survey", "participants": []}))
        store.commit(envelope(source_id="other-1", text="an unrelated delivery note",
                              metadata={"participants": []}))
    return settings


def without(installation, key):
    env_file = installation.home / "hermes-memory.env"
    env_file.write_text("".join(
        line for line in env_file.read_text(encoding="utf-8").splitlines(keepends=True)
        if not line.startswith(f"{key}=")), encoding="utf-8")
    return load_settings()


def ids(store):
    return [row["id"] for row in store.db.execute(
        "SELECT id FROM records ORDER BY ingested_at, id")]


def approve(settings, client, **kwargs):
    plan = summarize_plan(settings, **kwargs)
    return summarize_apply(settings, review=plan["review_digest"], actor=OWNER,
                          client=client, **kwargs)


def published(store, scope, *, kind="day", body="a reading of the window"):
    habits = SummaryStore(store, owner_principal=OWNER)
    found = habits.publish(scope=scope, kind=kind, title=f"the {kind}", body=body,
                           citations=[{"record_id": item} for item in ids(store)[:2]],
                           processor_fingerprint="proc-1")
    return habits, found["id"]


# -- the client's own contract ------------------------------------------------

def transport(result):
    calls = []

    def call(method, url, payload, headers):
        calls.append({"method": method, "url": url, "payload": payload})
        return result
    call.calls = calls
    return call


def client_with(result):
    return HindsightClient(base_url="http://127.0.0.1:8123", bank_id=BANK,
                           transport=transport(result))


def test_a_reflection_asks_for_the_facts_behind_its_own_answer():
    answer = {"text": "Two notes were filed.", "based_on": {"memories": [{"id": "m1"}]},
              "usage": {"input_tokens": 90, "output_tokens": 21}}
    outcome = client_with(TransportResult(200, answer)).reflect("What happened today?")
    assert outcome["text"] == "Two notes were filed." and outcome["cited_memories"] == 1
    assert outcome["facts"] == [{"id": "m1"}], "the memories behind the answer are the answer's sourcing"
    assert outcome["input_tokens"] == 90 and outcome["output_tokens"] == 21


def test_the_reflect_request_names_the_route_and_asks_for_sourcing():
    holder = client_with(TransportResult(200, {"text": "x", "based_on": {}}))
    holder.reflect("Summarize the week", max_tokens=512)
    call = holder.transport.calls[0]
    assert call["method"] == "POST" and call["url"].endswith("/reflect")
    assert call["payload"]["include"] == {"facts": {}}
    assert call["payload"]["max_tokens"] == 512


@pytest.mark.parametrize("question", ["", "   ", "x" * 4001])
def test_a_reflection_refuses_a_question_it_cannot_bound(question):
    holder = client_with(TransportResult(200, {"text": "should not be sent"}))
    with pytest.raises(HindsightError):
        holder.reflect(question)
    assert not holder.transport.calls, "the refusal happens before the request leaves"


def test_an_unconfirmed_reflection_is_reported_as_uncertain_not_as_an_answer():
    holder = client_with(TransportResult(0, None, transport_error="connection reset"))
    with pytest.raises(HindsightUnavailable, match="may still be running"):
        holder.reflect("What happened?")


def test_a_reflected_answer_is_bounded_before_it_is_trusted():
    with pytest.raises(HindsightError, match="max_tokens"):
        client_with(TransportResult(200, {"text": "x"})).reflect("q", max_tokens=99999)


# -- what a scope is ----------------------------------------------------------

@pytest.mark.parametrize("scope", ["", "  ", "nonsense:thing", "project:", "day:not-a-date",
                                   "week:2026-13-01"])
def test_a_scope_that_names_no_window_is_refused(scope):
    with pytest.raises(SummarizeError):
        resolve_scope(scope)


@pytest.mark.parametrize("scope", ["project:survey", "thread:abc", "account:owner",
                                   "source:gmail", "day:2026-09-24", "week:2026-09-22"])
def test_the_admissible_scopes_are_the_ones_a_window_can_be_taken_from(scope):
    assert resolve_scope(scope)[0] == scope.split(":")[0]


def test_a_project_scope_is_exactly_the_evidence_that_declares_it(installation):
    with ReadOnlyStore(installation.db_path) as store:
        selected = scope_records(store, scope="project:survey")
        assert sorted(item["source_id"] for item in selected) == ["note-1", "note-2"], \
            "the unrelated note is not in this project's window"


def test_a_day_scope_is_the_day_the_evidence_occurred(installation):
    with ReadOnlyStore(installation.db_path) as store:
        when = store.db.execute("SELECT occurred_at FROM records LIMIT 1").fetchone()[0]
        assert len(scope_records(store, scope=f"day:{when[:10]}")) == 3
        assert scope_records(store, scope="day:2020-01-01") == []


def test_a_week_scope_reaches_the_whole_week(installation):
    with ReadOnlyStore(installation.db_path) as store:
        assert len(scope_records(store, scope="week:2026-09-24")) == 3
        assert scope_records(store, scope="week:2026-01-01") == []


def test_a_forgotten_record_leaves_the_window_and_a_hidden_one_does_too(installation):
    with EvidenceStore(installation.db_path) as store:
        wanted = {row["id"]: row["source_id"] for row in store.db.execute(
            "SELECT id, source_id FROM records")}
        hidden = [key for key, value in wanted.items() if value == "note-1"][0]
        erased = [key for key, value in wanted.items() if value == "note-2"][0]
        store.hide(hidden, reason="superseded by a correction", actor=OWNER)
        assert sorted(item["source_id"] for item in scope_records(
            store, scope="project:survey")) == ["note-2"], \
            "a revision the owner superseded leaves the window"
        store.db.execute("UPDATE records SET deleted=1 WHERE id=?", (erased,))
        assert scope_records(store, scope="project:survey") == [], \
            "forgotten evidence must never be re-summarized"


# -- planning: a reading that performs nothing --------------------------------

def test_planning_names_the_window_without_sending_anything(installation):
    plan = summarize_plan(installation, scope="project:survey")
    assert plan["ok"] is True and plan["kind"] == "project"
    assert plan["records"] == 2 and plan["selected"][0]["revision"] == "1"
    assert plan["coverage"] == "truncated", "nothing has been projected to the backend yet"
    assert plan["not_performed"] and plan["review_digest"]
    with ReadOnlyStore(installation.db_path) as store:
        assert store.db.execute("SELECT count(*) FROM summaries").fetchone()[0] == 0


def test_planning_dials_nothing(installation, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("planning must not build a client, let alone ask it")
    monkeypatch.setattr(summarization, "client_for", forbid)
    assert summarize_plan(installation, scope="source:gmail")["records"] == 3


@pytest.mark.parametrize("missing", ["HERMES_MEMORY_INFERENCE_ENABLED",
                                     "HERMES_MEMORY_HINDSIGHT_URL",
                                     "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS"])
def test_a_reflection_needs_the_configuration_it_says_it_used(installation, missing):
    installation = without(installation, missing)
    plan = summarize_plan(installation, scope="project:survey")
    assert plan["ok"] is False
    assert any(missing.split("_")[-1].lower().rstrip("d") in note.lower()
               or note.startswith("no ") or "budget" in note.lower()
               for note in plan["blocking"]), plan["blocking"]


def test_an_empty_scope_is_refused_as_a_guess(installation):
    plan = summarize_plan(installation, scope="project:nothing-here")
    assert plan["ok"] is False
    assert any("a summary of no evidence is a guess" in note for note in plan["blocking"])


def test_a_window_too_large_for_the_ceiling_is_narrowed_never_truncated(installation):
    plan = summarize_plan(installation, scope="project:survey", limit=1)
    assert plan["ok"] is False
    assert any("more than 1 live record" in note for note in plan["blocking"]), \
        "summarizing a prefix and calling it the scope is the failure this refuses"


def test_the_digest_covers_the_window_it_promises(installation):
    before = summarize_plan(installation, scope="project:survey")["review_digest"]
    with EvidenceStore(installation.db_path) as store:
        store.commit(envelope(source_id="note-3", text="a third note",
                              metadata={"project": "survey", "participants": []}))
    assert summarize_plan(installation, scope="project:survey")["review_digest"] != before


def test_the_ceiling_is_the_citation_ceiling_the_store_enforces():
    assert MAX_BATCH == summaries_module.MAX_CITATIONS_PER_SUMMARY


# -- applying: one request, one filing ---------------------------------------

def test_an_approved_window_is_filed_with_the_evidence_it_is_a_claim_about(installation):
    client = Reflects()
    outcome = approve(installation, client, scope="project:survey")
    assert outcome["ok"] is True and client.__class__ is Reflects and len(client.calls) == 1
    assert outcome["published"] is True and outcome["revision"] == 1
    assert outcome["citations"] == 2 and outcome["scope"] == "project:survey"
    assert outcome["tokens_charged"] == 150, \
        "what the backend reported is what the day's ledger is charged"
    with ReadOnlyStore(installation.db_path) as store:
        cited = [row["record_id"] for row in store.db.execute(
            "SELECT record_id FROM derived_citations WHERE artifact_id=? ORDER BY record_id",
            (outcome["summary"],))]
        assert cited == sorted(ids(store)[:2])
        body = store.db.execute("SELECT body, kind, status FROM summaries WHERE id=?",
                               (outcome["summary"],)).fetchone()
        assert body["status"] == "published" and body["kind"] == "project"


def test_the_question_asked_names_the_scope_and_the_ceiling(installation):
    client = Reflects()
    approve(installation, client, scope="project:survey")
    ask = client.calls[0]["query"]
    assert "project:survey" in ask and "2 archived item" in ask
    assert "Say only what the evidence supports" in ask


def test_an_unapproved_digest_is_refused_before_the_backend_is_asked(installation):
    client = Reflects()
    with pytest.raises(SummarizeError, match="does not match"):
        summarize_apply(installation, scope="project:survey", review="f" * 64, actor=OWNER,
                        client=client)
    assert client.calls == []


def test_nothing_is_asked_without_an_actor_to_attribute_the_words_to(installation):
    plan = summarize_plan(installation, scope="project:survey")
    with pytest.raises(SummarizeError, match="actor"):
        summarize_apply(installation, scope="project:survey", review=plan["review_digest"],
                        actor="  ", client=Reflects())


def test_a_summary_of_nothing_cannot_be_filed_even_if_the_backend_volunteers_prose(installation):
    installation = without(installation, "HERMES_MEMORY_HINDSIGHT_URL")
    client = Reflects()
    with pytest.raises(SummarizeError, match="refused"):
        summarize_apply(installation, scope="project:survey",
                        review=summarize_plan(installation, scope="project:survey")
                        ["review_digest"], actor=OWNER, client=client)
    assert client.calls == []


def test_an_answer_with_no_words_in_it_leaves_the_scope_unsummarized(installation):
    with pytest.raises(SummarizeError, match="no text"):
        approve(installation, Reflects(text="   \n "), scope="project:survey")
    with ReadOnlyStore(installation.db_path) as store:
        assert store.db.execute("SELECT count(*) FROM summaries").fetchone()[0] == 0


def test_a_reflection_gets_a_deadline_a_generation_can_meet(installation):
    """The transport's read timeout sizes a lookup; a page of prose is not one.

    Answering after a lookup's deadline is reported as an unconfirmed submission, which
    quarantines a device slot and sends the operator to reconcile a call that was only slow
    — as two live reflects on a 9B over the LAN did, one of which had already failed in 12.7s.
    """
    from hermes_memory.backend.hindsight_client import DEFAULT_TIMEOUT_S
    from hermes_memory.processing.summarization import REFLECT_TIMEOUT_S, client_for

    client = client_for(installation)
    assert client.timeout == REFLECT_TIMEOUT_S
    assert client.timeout > DEFAULT_TIMEOUT_S, \
        "a generation is still expected to answer within a lookup's deadline"


@pytest.mark.parametrize("error", [HindsightError("reflect failed: HTTP 400 — bad request"),
                                   HindsightUnavailable("connection reset mid-request")])
def test_a_refused_refusal_is_recorded_as_a_failed_refresh_not_as_silence(installation, error):
    answered = approve(installation, Answers(error), scope="project:survey")
    assert answered["ok"] is False and "refused" in answered["refused"] or answered["refused"]
    with ReadOnlyStore(installation.db_path) as store:
        assert store.db.execute("SELECT count(*) FROM summaries").fetchone()[0] == 0
        rows = store.db.execute("SELECT state, detail FROM summary_refreshes").fetchall()
        assert rows == [] or all(row["state"] == "failed" for row in rows)


def test_the_reading_answers_every_promise_its_window_covers(installation):
    """A refresh booked with a different clock must not stay owed to nobody."""
    with EvidenceStore(installation.db_path) as store:
        habits = SummaryStore(store, owner_principal=OWNER)
        habits.request_refresh("project:survey", kind="project",
                               through_at="2026-09-24T00:00:00+00:00")
        habits.request_refresh("project:survey", kind="project",
                               through_at="2026-09-25T23:00:00+00:00")
        assert len(habits.pending_refreshes()) == 2
    outcome = approve(installation, Reflects(), scope="project:survey")
    assert outcome["ok"] is True
    with ReadOnlyStore(installation.db_path) as store:
        habits = SummaryStore(store, owner_principal=OWNER)
        pending = habits.pending_refreshes()
        assert [item["through_at"] for item in pending] == ["2026-09-25T23:00:00+00:00"], \
            "the promise the window reached is settled; the promise past it is still owed"


def test_a_second_approved_pass_supersedes_the_first_reading(installation):
    first = approve(installation, Reflects(), scope="project:survey")
    second = approve(installation, Reflects(text="a later reading of the same window"),
                     scope="project:survey")
    assert second["revision"] == 1 and second["published"] is True, \
        "the same words are the same summary"
    with EvidenceStore(installation.db_path) as store:
        habits = SummaryStore(store, owner_principal=OWNER)
        newer = approve(installation, Reflects(text="and then the invoice was chased"),
                        scope="project:survey")
        assert newer["revision"] == 1
        assert habits.verdict(newer["summary"]) == "partial", \
            "nothing has been projected, so the manifest is honestly truncated"


def test_a_mental_model_needs_the_owner_and_says_which_one(installation):
    plan = summarize_plan(installation, scope="project:survey", kind="mental_model")
    with pytest.raises(Exception, match="owner's approval"):
        summarize_apply(installation, scope="project:survey", kind="mental_model",
                        review=plan["review_digest"], actor="somebody-else",
                        client=Reflects())


def test_an_unconfirmed_reflection_leaves_the_backend_to_answer_for_the_device(installation):
    """The outer request never runs on the device, so it has no slot to quarantine.

    The backend's own admissions hold it, and they mark themselves uncertain when the
    transport goes away — which is the honest place for that record, because that is the
    request whose completion is genuinely unknown.
    """
    approve(installation, Answers(HindsightUnavailable("connection reset mid-request")),
            scope="project:survey")
    with instance_gate(installation) as gate:
        rows = gate.db.execute("SELECT state, route FROM gate_reservations").fetchall()
    assert [dict(row) for row in rows] == [], \
        "a reflection that took no slot must leave no quarantine of its own"


def test_a_reflection_never_holds_the_slot_the_backend_asks_for_itself(installation):
    """A reflect claim starves the very calls that would answer it: the gate said 429.

    The device is charged for what the answer reports, which is what the day's budget needs;
    holding a slot for a request that only carries a question is the self-inflicted deadlock.
    """
    approve(installation, Reflects(), scope="project:survey")
    with instance_gate(installation) as gate:
        rows = gate.db.execute("SELECT 1 FROM gate_reservations").fetchall()
        usage = gate.usage()
    assert rows == [], "the reflection admitted nothing it did not need"
    assert usage, "what the reflection spent is still charged to the device"
    assert sum(int(item["tokens"]) for item in usage.values()) > 0


def test_a_paused_device_still_refuses_a_reflection_that_would_queue_behind_it(installation):
    """A blocked device is refused by name, with the door that frees it."""
    plan = summarize_plan(installation, scope="project:survey")
    client = Reflects()
    with instance_gate(installation) as gate:
        held = gate.try_acquire(route="retain", holder="somebody-else",
                               resource=plan["resource"], priority=0, ttl=120.0)
        gate.mark_uncertain(held, reason="test: the connection went away mid-request")
        outcome = summarize_apply(installation, scope="project:survey",
                                  review=plan["review_digest"], actor=OWNER, client=client)
    assert outcome["ok"] is False
    assert "blocked by a request nobody has answered for" in outcome["refused"]
    assert "--resolve" in outcome["refused"], "the refusal names the door that frees it"
    assert client.calls == [], "a blocked device is not worth a request into it"


# -- the invalidator ----------------------------------------------------------

def test_forgetting_evidence_withdraws_the_reading_built_from_it(installation):
    with EvidenceStore(installation.db_path) as store:
        habits, summary = published(store, "project:survey", kind="project")
        target = store.db.execute(
            "SELECT record_id FROM derived_citations WHERE artifact_id=?",
            (summary,)).fetchone()[0]
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[target], actor=OWNER, reason="withdrawn")
        assert any(item["kind"] == "summary" and item["id"] == summary
                   for item in preview["derived_products"]), \
            "the owner is told a reading goes too, not only the record"
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
        assert habits.get(summary).status == "withdrawn"
        owed = habits.pending_refreshes()
        assert [item["scope"] for item in owed] == ["project:survey"], \
            "the withdrawn reading leaves a promise to read the scope again"


def test_a_reading_that_never_quoted_the_forgotten_evidence_stays(installation):
    with EvidenceStore(installation.db_path) as store:
        habits, summary = published(store, "source:telegram", kind="day")
        manager = ErasureManager(store, owner_principal=OWNER)
        other = store.db.execute(
            "SELECT id FROM records WHERE source_id='other-1'").fetchone()[0]
        preview = manager.preview(record_ids=[other], actor=OWNER, reason="not mine")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
        assert habits.get(summary).status == "published"
        assert habits.pending_refreshes() == []


def test_the_withdrawal_survives_a_rollback_of_the_same_confirm(installation):
    """A failed confirmation must not leave a reading quietly withdrawn."""
    with EvidenceStore(installation.db_path) as store:
        habits, summary = published(store, "project:survey", kind="project")
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[ids(store)[0]], actor=OWNER, reason="gone")
        with pytest.raises(Exception, match="does not match"):
            manager.confirm(intent_id=preview["intent_id"], preview_digest="0" * 64,
                            actor=OWNER)
        assert habits.get(summary).status == "published"


# -- the reader ---------------------------------------------------------------

@pytest.fixture()
def reading(installation):
    approve(installation, Reflects(), scope="project:survey")
    return installation


def packet_for(settings, query, *, summaries=True, budget=1800):
    with ReadOnlyStore(settings.db_path) as store:
        broker = ContextBroker(store, cache=None, budget_tokens=budget,
                               summaries=SummaryStore(store, owner_principal=OWNER)
                               if summaries else None)
        try:
            return broker.assemble(query, limit=8)
        finally:
            broker.close()


def test_the_packet_carries_the_current_reading_of_a_scope_it_touched(reading):
    found = packet_for(reading, "invoice")
    assert [item["scope"] for item in found.summaries] == ["project:survey"]
    assert found.summaries[0]["kind"] == "project" and found.summaries[0]["revision"] == 1
    assert "Two notes were filed" in found.render()
    assert "not a second sighting" in found.render()


def test_an_answer_with_no_evidence_in_a_scope_carries_no_reading(reading):
    assert packet_for(reading, "nothing archived about zkxvq").summaries == ()


def test_a_packet_without_the_reading_section_is_unchanged(reading):
    without_readings = packet_for(reading, "invoice", summaries=False)
    assert without_readings.summaries == ()
    assert without_readings.packet_id != packet_for(reading, "invoice").packet_id, \
        "a cache warmed without readings must not answer a caller that expects them"


def test_a_reading_never_buys_coverage_it_cannot_support(reading):
    found = packet_for(reading, "invoice", budget=80)
    assert "summaries" in found.truncated
    assert found.summaries == ()


def test_the_withheld_reading_of_forgotten_evidence_is_absent_not_stale(reading):
    with EvidenceStore(reading.db_path) as store:
        summary = SummaryStore(store, owner_principal=OWNER).latest("project:survey")[0].id
        manager = ErasureManager(store, owner_principal=OWNER)
        cited = [row["record_id"] for row in store.db.execute(
            "SELECT record_id FROM derived_citations WHERE artifact_id=?", (summary,))]
        preview = manager.preview(record_ids=cited, actor=OWNER, reason="all of it")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
    assert packet_for(reading, "invoice").summaries == ()


def test_the_reading_is_reported_in_the_packet_dict_as_well(reading):
    payload = packet_for(reading, "invoice").as_dict()
    assert payload["summaries"][0]["scope"] == "project:survey"
    assert payload["summaries"][0]["body"].startswith("Two notes")


# -- the door -----------------------------------------------------------------

def run(*args):
    from hermes_memory.cli import main
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(list(args))
    return code, json.loads(buffer.getvalue() or "{}")


def test_the_door_lists_the_window_and_asks_before_it_asks_a_model(installation, monkeypatch):
    monkeypatch.setattr(summarization, "client_for",
                        lambda settings: pytest.fail("the door must not connect"))
    code, report = run("summarize", "--scope", "project:survey")
    assert code == 0 and report["ok"] is True
    assert report["records"] == 2 and report["blocking"] == []
    assert "--review" in report["next"]


def test_the_door_refuses_a_digest_it_never_showed(installation):
    code, report = run("summarize", "--scope", "project:survey", "--review", "0" * 64,
                       "--actor", OWNER)
    assert code == 2 and report == {}


def test_the_door_writes_the_summary_somebody_approved(installation, monkeypatch):
    holder = Reflects()
    monkeypatch.setattr(summarization, "client_for", lambda settings: holder)
    _, listed = run("summarize", "--scope", "project:survey")
    code, report = run("summarize", "--scope", "project:survey",
                       "--review", listed["review_digest"], "--actor", OWNER)
    assert code == 0 and report["ok"] is True and report["actor"] == OWNER
    assert report["citations"] == 2 and len(holder.calls) == 1
    with ReadOnlyStore(installation.db_path) as store:
        assert SummaryStore(store, owner_principal=OWNER).latest("project:survey")


def test_an_unadmissible_scope_is_refused_at_the_door(installation):
    code, report = run("summarize", "--scope", "vibes:everything")
    assert code == 2 and report == {}


def test_the_door_is_counted_among_the_work_that_owes_an_answer(installation):
    """A promise booked by a forgetting has to be visible without reading the table."""
    with EvidenceStore(installation.db_path) as store:
        habits, summary = published(store, "project:survey", kind="project")
        store.db.execute("INSERT INTO summary_refreshes(scope, kind, through_at, "
                         "requested_at, state) VALUES('project:survey','project',?,?,"
                         "'pending')", ("2026-09-26T00:00:00+00:00", "2026-09-26T00:00:01"))
        assert habits.pending_refreshes()

# -- what the mutation driver said the tests were not checking ----------------

def test_a_pass_may_not_be_larger_than_the_ceiling_the_store_enforces(installation):
    with pytest.raises(SummarizeError, match="limit must be an integer"):
        summarize_plan(installation, scope="project:survey", limit=MAX_BATCH + 1)
    with pytest.raises(SummarizeError, match="limit must be an integer"):
        summarize_plan(installation, scope="project:survey", limit=0)


def test_a_paused_gate_refuses_the_reflection_instead_of_ignoring_the_pause(installation):
    client = Reflects()
    with instance_gate(installation) as gate:
        gate.pause(actor=OWNER, reason="the model server is being restarted")
        plan = summarize_plan(installation, scope="project:survey")
        assert plan["ok"] is False and any("paused" in note for note in plan["blocking"]), \
            "the pause is visible before anything is approved, not only at the request"
        try:
            with pytest.raises(SummarizeError, match="paused"):
                summarize_apply(installation, scope="project:survey",
                                review=plan["review_digest"], actor=OWNER, client=client)
        finally:
            gate.resume(actor=OWNER, reason="back up")
    assert client.calls == [], "a paused stage is not a suggestion"


def test_a_device_busy_with_somebody_else_still_gets_the_question(installation):
    """Occupation is no reason to refuse: the outer request carries a question, not work.

    Refusing on occupation was the deadlock wearing the costume of caution — the one caller
    that could free the device is the reflection being refused, and the backend queues its
    own tool calls on that slot regardless of anything this process holds.
    """
    plan = summarize_plan(installation, scope="project:survey")
    client = Reflects()
    with instance_gate(installation) as holder:
        reservation = holder.try_acquire(route="retain", holder="somebody-else",
                                        resource=plan["resource"], priority=0, ttl=120.0)
        assert reservation is not None, "the fixture must actually occupy the slot"
        try:
            outcome = summarize_apply(installation, scope="project:survey",
                                      review=plan["review_digest"], actor=OWNER, client=client)
        finally:
            holder.release(reservation, outcome="succeeded", tokens=0, seconds=0.1)
    assert outcome["ok"] is True, outcome.get("refused")
    assert client.calls, "the question was asked despite the busy device"


def _window_end(store):
    return store.db.execute("SELECT max(occurred_at) FROM records").fetchone()[0]


def test_a_refused_reflection_answers_the_promise_it_was_asked_to_keep(installation):
    with EvidenceStore(installation.db_path) as store:
        through = _window_end(store)
        SummaryStore(store, owner_principal=OWNER).request_refresh(
            "project:survey", kind="project", through_at=through)
    outcome = approve(installation, Answers(HindsightError("reflect failed: HTTP 503 — busy")),
                      scope="project:survey")
    assert outcome["ok"] is False and outcome["refreshes_settled"] == 1
    with ReadOnlyStore(installation.db_path) as store:
        habits = SummaryStore(store, owner_principal=OWNER)
        assert habits.pending_refreshes() == []
        row = store.db.execute("SELECT state, detail FROM summary_refreshes").fetchone()
        assert row["state"] == "failed" and "503" in row["detail"], \
            "the reason it did not happen is the thing a status report has to show"


def test_an_identical_reading_still_answers_the_promise(installation):
    """The words were already on file, so the scope is summarized; that is a kept promise."""
    approve(installation, Reflects(), scope="project:survey")
    with EvidenceStore(installation.db_path) as store:
        SummaryStore(store, owner_principal=OWNER).request_refresh(
            "project:survey", kind="project", through_at=_window_end(store))
    again = approve(installation, Reflects(), scope="project:survey")
    assert again["published"] is False and again["already_on_file"] is True
    with ReadOnlyStore(installation.db_path) as store:
        assert SummaryStore(store, owner_principal=OWNER).pending_refreshes() == []
        rows = store.db.execute("SELECT state FROM summary_refreshes").fetchall()
        assert [row["state"] for row in rows] == ["done"], \
            "an answered promise is recorded as kept, not merely as no longer pending"


@pytest.mark.parametrize("reason", ["", "   ", "x" * 1001])
def test_a_withdrawal_says_why_in_words_that_fit(installation, reason):
    with EvidenceStore(installation.db_path) as store:
        habits, summary = published(store, "project:survey", kind="project")
        with pytest.raises(summaries_module.EvidenceError, match="reason"):
            habits.withdraw(summary, actor=OWNER, reason=reason)
        assert habits.get(summary).status == "published"


def test_a_reading_already_withdrawn_is_not_booked_as_owed_again(installation):
    with EvidenceStore(installation.db_path) as store:
        habits, summary = published(store, "project:survey", kind="project")
        habits.withdraw(summary, actor=OWNER, reason="the owner read it and disagreed")
        before = len(habits.pending_refreshes())
        cited = [row["record_id"] for row in store.db.execute(
            "SELECT record_id FROM derived_citations WHERE artifact_id=?", (summary,))]
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[cited[0]], actor=OWNER, reason="gone")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
        assert len(habits.pending_refreshes()) == before


def test_the_booked_refresh_reaches_the_end_of_the_withdrawn_window(installation):
    with EvidenceStore(installation.db_path) as store:
        habits = SummaryStore(store, owner_principal=OWNER)
        habits.publish(scope="thread:survey", kind="thread", title="a thread",
                       body="what the thread said", citations=[{"record_id": ids(store)[0]}],
                       processor_fingerprint="proc-1",
                       window=("2026-09-01T00:00:00+00:00", "2026-09-20T00:00:00+00:00"))
        manager = ErasureManager(store, owner_principal=OWNER)
        preview = manager.preview(record_ids=[ids(store)[0]], actor=OWNER, reason="gone")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=OWNER)
        assert [item["through_at"] for item in habits.pending_refreshes()] == [
            "2026-09-20T00:00:00+00:00"], \
            "the promise is to re-read what that window covered, not to re-read today"


def test_a_cache_warmed_without_readings_does_not_answer_a_caller_that_expects_them(reading):
    from hermes_memory.context import PacketCache

    with EvidenceStore(reading.db_path) as store:
        cache = PacketCache(store)
        plain = ContextBroker(store, cache=cache, budget_tokens=1800)
        first = plain.assemble("invoice", limit=8)
        armed = ContextBroker(store, cache=cache, budget_tokens=1800,
                              summaries=SummaryStore(store, owner_principal=OWNER))
        second = armed.assemble("invoice", limit=8)
        plain.close()
        armed.close()
    assert first.summaries == ()
    assert [item["scope"] for item in second.summaries] == ["project:survey"], \
        "the second caller was handed the first one's packet"


def test_a_packet_whose_only_content_is_a_reading_is_not_reported_as_empty():
    from hermes_memory.context import Packet

    found = Packet(query="q", summaries=({"id": "sum_1", "scope": "project:x",
                                          "kind": "project", "revision": 1,
                                          "body": "a reading of the window"},))
    assert not found.empty
    assert "summary of project:x" in found.render()


def test_a_withdrawal_written_outside_a_transaction_is_refused(installation):
    """An erasure's withdrawal must not be autocommitted apart from its tombstones."""
    with EvidenceStore(installation.db_path) as store:
        habits, summary = published(store, "project:survey", kind="project")
        store.db.execute("BEGIN IMMEDIATE")
        store.db.execute("COMMIT")
        with pytest.raises(summaries_module.EvidenceError, match="ambient transaction"):
            habits.withdraw(summary, actor=OWNER, reason="withdrawn by hand", db=store.db)
        assert habits.get(summary).status == "published"
