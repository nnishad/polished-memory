"""C2 MCP adapter: one mapped thing, no discovery, and no room for a redirect.

The plan's gate for this adapter is the set of things an MCP server could do to a
connector that believed it: offer prompts as if they were evidence, hand over a
credential in a field nobody mapped, redirect the reader to a resource the operator
never authorised, or revoke the grant mid-flight. Each of those is a case below.
"""
from __future__ import annotations

import json

import pytest

from hermes_memory.sources import mcp
from hermes_memory.sources.base import CursorExpired
from hermes_memory.sources.mcp import MAX_MCP_TEXT_CHARS, McpSource, SourceMap
from hermes_memory.sources.runtime import COMPLETE, UNREACHABLE, ConnectorRuntime
from hermes_memory.sources.sync import SyncController
from hermes_memory.storage.evidence import EvidenceError

BEARER = "Bearer yhVn2sQ9kF7pLm3Zx8Rt4Wd6"
THREAD_MAP = {"name": "threads", "tool": "list_threads", "id": "thread_id",
              "text": "subject", "time": "updated_at", "kind": "message",
              "scope": "read:threads"}


class Client:
    """A server stub that records every call and refuses to be asked anything else."""

    def __init__(self, answers=None, *, status=None, raises=None):
        self.answers = dict(answers or {})
        self.calls: list[tuple] = []
        self.status_report = status
        self.raises = raises

    def call_tool(self, name, arguments):
        return self._record(("tool", name, arguments))

    def read_resource(self, uri):
        return self._record(("resource", uri))

    def _record(self, call):
        self.calls.append(call)
        if self.raises is not None:
            raise self.raises
        answer = self.answers.get(call[1])
        if isinstance(answer, Exception):
            raise answer
        return answer

    # Discovery, prompts and sampling: present so that touching one is a failure the
    # suite can see, rather than a method the adapter could quietly reach for.
    def list_tools(self):
        raise AssertionError("the adapter must not enumerate what a server can do")

    def list_resources(self):
        raise AssertionError("the adapter must not browse a server's resources")

    def list_prompts(self):
        raise AssertionError("a prompt is not evidence to enumerate")

    def complete(self, *args, **kwargs):
        raise AssertionError("the adapter must not ask a server to complete anything")

    def create_message(self, *args, **kwargs):
        raise AssertionError("remote sampling is not this connector's to request")


def status_client(report=None):
    """A client that reports its own standing, which may be a refusal."""
    client = Client({"list_threads": {"items": []}})

    def status():
        if isinstance(report, Exception):
            raise report
        return report

    client.status = status
    return client


NOTHING = object()


def source(mapping=None, *, client=None, answer=NOTHING, **kwargs):
    client = client if client is not None else Client(
        {"list_threads": {"items": []} if answer is NOTHING else answer})
    return McpSource(client, mapping or THREAD_MAP, **kwargs), client


def item(thread_id="t1", subject="The invoice is overdue.", **extra):
    return {"thread_id": thread_id, "subject": subject,
            "updated_at": "2026-09-24T09:15:00+02:00", **extra}


# -- the map is the operator's, and it is checked ---------------------------

def test_a_map_names_one_channel_rather_than_a_menu():
    with pytest.raises(EvidenceError, match="one channel"):
        SourceMap.from_mapping({"tool": "a", "resource": "b://c"})
    with pytest.raises(EvidenceError, match="must name the tool or resource"):
        SourceMap.from_mapping({"name": "orphan"})


def test_a_prompt_is_refused_because_it_is_an_instruction_not_a_record():
    """Filed as evidence it would come back later as though somebody had said it."""
    with pytest.raises(EvidenceError, match="instruction to a model"):
        SourceMap.from_mapping({"name": "summarise", "prompt": "weekly_digest"})


def test_a_misspelt_time_field_is_refused_rather_than_loading_and_dropping_dates():
    with pytest.raises(EvidenceError, match="unknown source map fields"):
        SourceMap.from_mapping({"tool": "list_threads", "timestamp": "updated_at"})


def test_a_channel_named_with_nothing_behind_it_is_refused():
    with pytest.raises(EvidenceError, match="has no name"):
        SourceMap.from_mapping({"tool": "   "})


def test_a_map_that_is_not_an_object_is_refused():
    with pytest.raises(EvidenceError, match="object of declared fields"):
        SourceMap.from_mapping("list_threads")


def test_a_channel_that_is_neither_tool_nor_resource_is_refused():
    with pytest.raises(EvidenceError, match="channel"):
        SourceMap(target="x", name="x", channel="queue")


def test_a_missing_required_field_is_named_rather_than_defaulted():
    with pytest.raises(EvidenceError, match="nonempty text"):
        SourceMap(name="", target="list_threads")


def test_declared_names_are_trimmed_however_they_were_written():
    subject = SourceMap(name="  threads  ", target=" list_threads ")
    assert (subject.name, subject.target) == ("threads", "list_threads")


def test_a_map_defaults_to_the_obvious_field_names_but_states_them():
    subject = SourceMap.from_mapping({"tool": "list_threads"})
    assert (subject.record_id, subject.text, subject.time, subject.revision) == (
        "id", "text", None, None)
    assert subject.name == "list_threads"


def test_a_tool_map_needs_a_client_that_can_call_tools():
    class ResourceOnly:
        def read_resource(self, uri):
            return {"items": []}

    with pytest.raises(EvidenceError, match="no call_tool"):
        McpSource(ResourceOnly(), THREAD_MAP)


def test_a_resource_map_needs_a_client_that_can_read_resources():
    class ToolOnly:
        def call_tool(self, name, arguments):
            return {"items": []}

    with pytest.raises(EvidenceError, match="no read_resource"):
        McpSource(ToolOnly(), {"name": "doc", "resource": "file:///notes.md"})


def test_a_source_map_needs_the_client_it_is_read_through():
    with pytest.raises(EvidenceError, match="needs the client"):
        McpSource(None, THREAD_MAP)


def test_the_source_is_named_after_the_map_unless_the_operator_names_it():
    default, _ = source(answer={"items": []})
    assert default.source == "mcp-threads"
    named, _ = source(answer={"items": []}, source="workmail-threads")
    assert named.source == "workmail-threads"


def test_a_page_size_the_store_would_refuse_is_refused_here_instead():
    with pytest.raises(ValueError, match="between 1 and 1000"):
        McpSource(Client(), THREAD_MAP, records_per_page=0)


def test_a_live_map_says_so_because_a_channel_that_grows_is_not_an_export():
    quiet, _ = source(answer={"items": []})
    assert quiet.capabilities.live is False
    growing, _ = source(answer={"items": []}, live=True)
    assert growing.capabilities.live is True


# -- nothing is discovered --------------------------------------------------

def test_only_the_mapped_tool_is_ever_called():
    subject, client = source(answer={"items": [item()]})
    subject.read_page(None)
    assert client.calls == [("tool", "list_threads", {"limit": 100})]


def test_a_position_is_handed_back_to_the_same_target_and_to_nothing_else():
    subject, client = source(answer={"items": [item()], "next_cursor": "c2"})
    subject.read_page(None)
    subject.read_page("c2")
    assert [call[2] for call in client.calls] == [
        {"limit": 100}, {"limit": 100, "after": "c2"}]


def test_a_resource_is_read_at_its_mapped_uri_once_and_cannot_be_resumed():
    client = Client({"file:///notes.md": {"contents": [{"id": "n1", "text": "A note."}]}})
    subject = McpSource(client, {"name": "notes", "resource": "file:///notes.md",
                                 "id": "id", "text": "text"})
    page = subject.read_page(None)
    assert client.calls == [("resource", "file:///notes.md")]
    assert len(page.envelopes) == 1
    with pytest.raises(CursorExpired, match="no position to resume"):
        subject.read_page("anything")


def test_a_next_uri_in_an_answer_does_not_become_the_next_read():
    """A server that could point the connector elsewhere would not need permission."""
    subject, client = source(answer={"items": [item()], "next_uri": "file:///etc/passwd",
                                     "next_cursor": "c2"})
    subject.read_page("c2")
    assert [(call[0], call[1]) for call in client.calls] == [("tool", "list_threads")]


# -- one item, one record ---------------------------------------------------

def test_a_mapped_item_becomes_one_record_of_the_declared_kind():
    subject, _ = source(answer={"items": [item()]})
    envelopes, skipped = subject.read_all()
    assert skipped == []
    assert len(envelopes) == 1
    record = envelopes[0]
    assert record["source_id"] == "t1"
    assert record["text"] == "The invoice is overdue."
    assert record["kind"] == "message"
    assert record["occurred_at"] == "2026-09-24T07:15:00+00:00"
    assert record["metadata"]["scope"] == "read:threads"
    assert record["metadata"]["mcp_target"] == "list_threads"


def test_an_item_with_no_identity_is_a_gap_naming_the_field_that_was_wanted():
    subject, _ = source(answer={"items": [{"subject": "nameless", "updated_at": None}]})
    page = subject.read_page(None)
    assert page.envelopes == ()
    assert page.skipped[0].reason == "the item carries no 'thread_id' to identify it by"


def test_the_same_thing_offered_twice_in_one_answer_is_read_once():
    subject, _ = source(answer={"items": [item(), item()]})
    page = subject.read_page(None)
    assert len(page.envelopes) == 1
    assert page.skipped[0].reason == "the server offered this revision twice in one answer"


def test_an_item_that_is_not_an_object_is_still_attributed_to_something():
    subject, _ = source(answer={"items": ["a bare string"]})
    page = subject.read_page(None)
    assert page.skipped[0].ref == "threads#0"
    assert page.skipped[0].reason == "the server yielded a non-object"


def test_an_empty_text_field_is_missing_rather_than_empty():
    subject, _ = source(answer={"items": [item(subject="   ")]})
    page = subject.read_page(None)
    assert page.skipped[0].reason == "the item's 'subject' field holds no readable text"


def test_a_versioned_resource_becomes_revisions_of_one_record_rather_than_new_records():
    subject, _ = source(client=Client({"fetch_doc": {
        "items": [{"id": "policy", "text": "v1 wording", "version": "1"},
                  {"id": "policy", "text": "v2 wording", "version": "2"}]}}),
        mapping={"name": "policy", "tool": "fetch_doc", "id": "id", "text": "text",
                 "revision": "version"})
    page = subject.read_page(None)
    assert [(item["source_id"], item["revision"]) for item in page.envelopes] == [
        ("policy", "1"), ("policy", "2")]


def test_an_unversioned_source_revises_at_one_rather_than_by_content():
    subject, _ = source(answer={"items": [item()]})
    assert subject.read_page(None).envelopes[0]["revision"] == "1"


def test_no_undeclared_field_of_a_server_answer_reaches_the_record():
    subject, _ = source(answer={"items": [item(
        internal_url="https://intranet.example/secrets", authorization=BEARER,
        attachment_bytes="Zm9v")]})
    envelope = subject.read_page(None).envelopes[0]
    assert BEARER not in json.dumps(envelope)
    assert "intranet.example" not in json.dumps(envelope)
    assert "Zm9v" not in json.dumps(envelope)
    assert sorted(envelope["metadata"]["undeclared_fields_dropped"]) == [
        "attachment_bytes", "authorization", "internal_url"]


def test_an_answer_of_only_declared_fields_claims_no_dropping():
    subject, _ = source(answer={"items": [item()]})
    assert "undeclared_fields_dropped" not in subject.read_page(None).envelopes[0]["metadata"]


def test_a_credential_in_a_mapped_text_field_is_removed_before_it_is_stored():
    subject, _ = source(answer={"items": [item(subject=f"rotate this: {BEARER} today")]} )
    envelope = subject.read_page(None).envelopes[0]
    assert BEARER not in json.dumps(envelope)
    assert "[redacted]" in envelope["text"]


# -- text and time ----------------------------------------------------------

def test_multilingual_text_is_kept_as_the_characters_that_were_sent():
    subject, _ = source(answer={"items": [item(subject="Die Rechnung ist überfällig.")]})
    assert subject.read_page(None).envelopes[0]["text"] == "Die Rechnung ist überfällig."


def test_bytes_are_decoded_strictly_and_damaged_bytes_are_a_gap():
    subject, _ = source(answer={"items": [item(subject=b"F\xe4lle"), item("t2", subject=b"ok")]})
    page = subject.read_page(None)
    assert page.skipped[0].reason.endswith("holds no readable text")
    assert page.envelopes[0]["text"] == "ok"


def test_text_that_arrives_already_damaged_is_refused():
    subject, _ = source(answer={"items": [item(subject="Caf\ufffd in the\ufffdir")]})
    assert subject.read_page(None).envelopes == ()


def test_an_oversized_body_is_bounded_and_says_that_it_was():
    subject, _ = source(answer={"items": [item(subject="a" * (MAX_MCP_TEXT_CHARS + 900))]})
    envelope = subject.read_page(None).envelopes[0]
    assert len(envelope["text"]) == MAX_MCP_TEXT_CHARS
    assert envelope["metadata"]["truncated"] is True
    assert envelope["metadata"]["chars"] == MAX_MCP_TEXT_CHARS


def test_a_time_the_map_did_not_declare_is_not_hunted_for():
    """A field called 'timestamp' is a guess, and a guess becomes a wrong date."""
    subject, _ = source(
        mapping={"name": "threads", "tool": "list_threads", "id": "thread_id",
                 "text": "subject"},
        answer={"items": [item()]})
    envelope = subject.read_page(None).envelopes[0]
    assert envelope["occurred_at"] is None
    assert envelope["metadata"]["time_basis"] == "none"


def test_a_clock_reading_with_no_zone_is_not_a_moment():
    subject, _ = source(answer={"items": [item(updated_at="2026-09-24T09:15:00")]})
    envelope = subject.read_page(None).envelopes[0]
    assert envelope["occurred_at"] is None
    assert envelope["occurred_precision"] == "unknown"
    assert "timezone" in envelope["metadata"]["time_note"]


def test_a_declared_time_field_is_attributed_to_itself():
    subject, _ = source(answer={"items": [item()]})
    envelope = subject.read_page(None).envelopes[0]
    assert envelope["metadata"]["time_basis"] == "updated_at"


def test_a_date_only_time_is_precise_as_a_day():
    subject, _ = source(answer={"items": [item(updated_at="2026-09-24")]})
    envelope = subject.read_page(None).envelopes[0]
    assert (envelope["occurred_at"], envelope["occurred_precision"]) == (
        "2026-09-24T00:00:00+00:00", "day")


# -- bounds and refusals ----------------------------------------------------

def test_a_page_stops_at_the_declared_size_and_keeps_the_servers_position():
    items = [item(f"t{index}") for index in range(250)]
    subject, _ = source(answer={"items": items, "next_cursor": "c2"}, records_per_page=10)
    page = subject.read_page(None)
    assert len(page.envelopes) == 10
    assert page.next_cursor.startswith("mcp-page-v1:")
    second = subject.read_page(page.next_cursor)
    assert second.envelopes[0]["source_id"] == "t10"


def test_a_full_page_with_no_position_to_resume_from_reports_what_it_could_not_carry():
    items = [item(f"t{index}") for index in range(30)]
    subject, _ = source(answer={"items": items}, records_per_page=10)
    page = subject.read_page(None)
    assert len(page.envelopes) == 10
    assert not page.skipped
    assert page.next_cursor.startswith("mcp-page-v1:")
    records, gaps = subject.read_all()
    assert len(records) == 30 and not gaps


def test_an_answer_longer_than_one_page_will_hold_is_reported_not_quietly_cut(monkeypatch):
    monkeypatch.setattr(mcp, "_MAX_ITEMS", 4)
    items = [item(f"t{index}") for index in range(12)]
    subject, _ = source(answer={"items": items}, records_per_page=5)
    page = subject.read_page(None)
    assert len(page.envelopes) <= 5
    assert any("one page takes at most 4" in row.reason for row in page.skipped)


def test_a_byte_budget_stops_a_page_without_losing_the_rest():
    items = [item(f"t{index}", subject="a" * 30_000) for index in range(6)]
    subject, _ = source(answer={"items": items, "next_cursor": "more"})
    subject.capabilities = mcp.replace(subject.capabilities, max_bytes_per_page=50_000)
    page = subject.read_page(None)
    assert len(page.envelopes) == 1
    assert page.next_cursor.startswith("mcp-page-v1:")
    second = subject.read_page(page.next_cursor)
    assert second.envelopes[0]["source_id"] == "t1"


@pytest.mark.parametrize("answer", [None, "a string", {"no_items": []}, {"items": "abc"},
                                    {"items": {"a": 1}}])
def test_an_answer_that_is_not_a_list_of_items_is_refused_rather_than_guessed(answer):
    subject, _ = source(answer=answer)
    with pytest.raises(EvidenceError, match="items"):
        subject.read_page(None)


def test_a_cursor_the_server_could_not_read_back_is_refused():
    subject, _ = source(answer={"items": [item()], "next_cursor": {"opaque": True}})
    with pytest.raises(EvidenceError, match="accept back"):
        subject.read_page(None)


def test_a_numeric_cursor_is_handed_back_as_the_text_a_cursor_has_to_be():
    subject, client = source(answer={"items": [item()], "nextCursor": 17})
    subject.read_page(None)
    subject.read_page("17")
    assert client.calls[-1][2]["after"] == "17"


def test_a_server_that_does_not_answer_is_an_outage_not_an_empty_channel():
    subject, _ = source(client=Client(raises=ConnectionError("connection reset by peer")))
    with pytest.raises(EvidenceError, match="did not answer"):
        subject.read_page(None)


# -- what the source says about itself --------------------------------------

def test_the_probe_reports_the_wiring_without_spending_a_read():
    subject, client = source(answer={"items": [item()]})
    report = subject.check()
    assert report["ok"] is True and report["content_read"] is False
    assert report["target"] == "list_threads" and report["scope"] == "read:threads"
    assert client.calls == []


def test_a_client_that_reports_no_standing_says_so_rather_than_implying_health():
    subject, _ = source(answer={"items": []})
    report = subject.check()
    assert "no standing" in report["note"]


def test_a_revoked_grant_is_named_as_one_rather_than_as_a_network_fault():
    subject = McpSource(status_client({"authorized": False,
                                       "reason": "the token was revoked by the account"}),
                        THREAD_MAP)
    report = subject.check()
    assert report["ok"] is False
    assert report["coverage_state"] == "revoked"


def test_a_status_that_could_not_be_given_is_still_reported_as_a_refusal():
    subject = McpSource(status_client(PermissionError("insufficient_scope")), THREAD_MAP)
    report = subject.check()
    assert report["ok"] is False and report["coverage_state"] == "revoked"


def test_a_status_that_is_not_a_report_leaves_the_verdict_to_the_first_read():
    subject = McpSource(status_client("up"), THREAD_MAP)
    assert subject.check()["ok"] is True


def test_an_authorized_standing_is_passed_through_as_the_server_stated_it():
    subject = McpSource(status_client({"authorized": True}), THREAD_MAP)
    assert subject.check()["authorized"] is True


# -- end to end -------------------------------------------------------------

def test_a_mapped_channel_becomes_evidence_once_and_a_second_poll_adds_nothing(store):
    sync = SyncController(store)
    sync.register("mcp-threads", policy_version="private-api")
    answer = {"items": [item("t1"), item("t2")], "next_cursor": None}
    runtime = ConnectorRuntime(store, sync, holder="worker-a")

    first = runtime.run(source(answer=answer)[0])
    assert (first.stopped, first.records, first.coverage_state) == (COMPLETE, 2, "current")

    second = runtime.run(source(answer=answer)[0])
    assert (second.records, second.repeats) == (0, 1)


def test_a_grant_that_was_revoked_is_recorded_as_revoked_not_as_an_outage(store):
    sync = SyncController(store)
    sync.register("mcp-threads", policy_version="private-api")
    subject = McpSource(status_client({"authorized": False,
                                       "reason": "invalid_grant: re-authorise this source"}),
                        THREAD_MAP)
    run = ConnectorRuntime(store, sync, holder="worker-a").run(subject)
    assert run.stopped == UNREACHABLE
    assert sync.state("mcp-threads")["coverage_state"] == "revoked"


def test_an_over_long_cursor_is_bounded_to_what_the_store_can_hold():
    subject, _ = source(answer={"items": [item()], "next_cursor": "c" * 5000})
    page = subject.read_page(None)
    assert len(page.next_cursor) == 2000


def test_a_time_like_field_is_not_hunted_for_when_the_map_declares_none():
    subject, _ = source(
        mapping={"name": "threads", "tool": "list_threads", "id": "thread_id",
                 "text": "subject"},
        answer={"items": [item(timestamp="2026-09-24T09:15:00+02:00",
                               date="2026-09-24T09:15:00+02:00")]})
    envelope = subject.read_page(None).envelopes[0]
    assert envelope["occurred_at"] is None
    assert envelope["occurred_precision"] == "unknown"


def test_the_item_cap_bounds_the_work_rather_than_only_the_report(monkeypatch):
    monkeypatch.setattr(mcp, "_MAX_ITEMS", 4)
    items = [item(f"t{index}") for index in range(12)]
    subject, _ = source(answer={"items": items}, records_per_page=50)
    page = subject.read_page(None)
    assert len(page.envelopes) == 4
    assert any("one page takes at most 4" in row.reason for row in page.skipped)


def test_a_status_that_is_not_a_report_is_said_rather_than_nodded_through():
    subject = McpSource(status_client("up"), THREAD_MAP)
    report = subject.check()
    assert report["ok"] is True
    assert "status was not a report" in report["note"]
