"""C2 email export adapter: identity, provenance and the gaps a message leaves.

The gate items are the ones the plan named: a malformed export, timezone ambiguity,
multilingual text, missing ids, attachment errors, and the rule that copied text is
context rather than a second piece of evidence.
"""
from __future__ import annotations

import base64
import json
import pathlib
from dataclasses import replace

import pytest

from hermes_memory.ids import timestamp
from hermes_memory.sources.base import CursorExpired
from hermes_memory.sources.email import GmailSource, MAX_MESSAGE_BYTES, EmailSource
from hermes_memory.sources.email import attachment_descriptors
from hermes_memory.storage.evidence import EvidenceError, EvidenceStore
from hermes_memory.storage.evidence import prepare_envelope

AT = timestamp("2026-09-15T09:00:00+00:00")


def eml(message_id="<a1@example.com>", *, date="Tue, 15 Sep 2026 09:14:02 +0200",
        subject="Re: invoice 27", extra="", body="Die Zahlung ist am Montag fällig.\r\n"
        "> The invoice is overdue.\r\n", content="text/plain; charset=utf-8"):
    return (f"Message-ID: {message_id}\r\n"
            f"Date: {date}\r\n"
            "From: Anna <anna@example.com>\r\n"
            "To: me@example.com, boss@example.com\r\n"
            f"Subject: {subject}\r\n"
            "In-Reply-To: <z9@example.com>\r\n"
            "References: <q1@example.com> <z9@example.com>\r\n"
            "Received: from mail.example.com (1.2.3.4)\r\n"
            f"{extra}"
            f"Content-Type: {content}\r\n\r\n"
            f"{body}").encode("utf-8")


def tree(root: pathlib.Path, files: dict[str, bytes | str]):
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
    return root


def source(root, **kwargs):
    return EmailSource(root, **kwargs)


def ids_of(page):
    return [item["source_id"] for item in page.envelopes]


def one(root, **kwargs) -> dict:
    page = source(root, **kwargs).read_page(None)
    assert len(page.envelopes) == 1, [gap.reason for gap in page.skipped]
    return page.envelopes[0]


# -- declaring what it can do -------------------------------------------------

def test_the_check_counts_files_without_reading_one(tmp_path, monkeypatch):
    tree(tmp_path, {"inbox.eml": eml()})
    monkeypatch.setattr(pathlib.Path, "read_bytes",
                        lambda self: (_ for _ in ()).throw(AssertionError("content read")))
    report = source(tmp_path).check()

    assert report["ok"] is True and report["candidates"] == 1
    assert report["content_read"] is False


def test_a_missing_directory_reports_instead_of_raising(tmp_path):
    report = source(tmp_path / "absent").check()

    assert report["ok"] is False and "not a directory" in report["reason"]


def test_an_email_export_is_history_only(tmp_path):
    capabilities = source(tmp_path).capabilities

    assert capabilities.history and not capabilities.live
    assert not capabilities.deletion_events, "a file export cannot report forgetting"


# -- identity -----------------------------------------------------------------

def test_the_message_id_is_the_identity(tmp_path):
    item = one(tree(tmp_path, {"inbox.eml": eml()}))

    assert item["source_id"] == "a1@example.com"
    assert item["metadata"]["message_id"] == "a1@example.com"
    assert item["revision"] == "1", "a header is immutable; a re-read is not a new version"


def test_a_message_without_an_id_is_a_named_gap_not_a_guessed_one(tmp_path):
    root = tree(tmp_path, {"inbox.eml": eml(message_id="")})
    page = source(root).read_page(None)

    assert page.envelopes == ()
    assert "Message-ID" in page.skipped[0].reason


def test_a_re_read_export_produces_the_same_records(tmp_path):
    root = tree(tmp_path, {"inbox.eml": eml()})

    first, second = source(root).read_all(), source(root).read_all()

    assert [item["source_id"] for item in first[0]] == [item["source_id"] for item in second[0]]
    assert [item["text"] for item in first[0]] == [item["text"] for item in second[0]]


def test_each_message_in_a_mbox_is_its_own_record_and_its_own_gap(tmp_path):
    root = tree(tmp_path, {"out.mbox": (
        "From - Tue Sep 15 09:00:00 2026\n"
        "Message-ID: <m0@example.com>\nDate: Tue, 15 Sep 2026 09:00:00 +0200\n"
        "Subject: first\nContent-Type: text/plain\n\nOne.\n\n"
        "From - Wed Sep 16 09:00:00 2026\n"
        "Date: Wed, 16 Sep 2026 09:00:00 +0200\nSubject: second\n"
        "Content-Type: text/plain\n\nNo id here.\n\n")})
    page = source(root).read_page(None)

    assert ids_of(page) == ["m0@example.com"]
    assert page.skipped[0].ref == "out.mbox#1", "which message failed, not just which file"


# -- time ---------------------------------------------------------------------

def test_an_offset_in_the_header_is_carried_through(tmp_path):
    item = one(tree(tmp_path, {"inbox.eml": eml()}))

    assert item["occurred_at"] == "2026-09-15T07:14:02+00:00"
    assert item["occurred_precision"] == "second"
    assert "time_note" not in item["metadata"]


@pytest.mark.parametrize("date, note", [
    ("Tue, 15 Sep 2026 09:14:02", "carried no timezone offset"),
    ("the day after the invoice", "unparsable"),
])
def test_an_ambiguous_time_stays_ambiguous(tmp_path, date, note):
    item = one(tree(tmp_path, {"inbox.eml": eml(date=date)}))

    assert item["occurred_at"] is None and item["occurred_precision"] == "unknown"
    assert note in item["metadata"]["time_note"]


def test_no_date_header_is_not_answered_with_ingestion_time(tmp_path):
    root = tree(tmp_path, {"inbox.eml": eml().replace(
        b"Date: Tue, 15 Sep 2026 09:14:02 +0200\r\n", b"")})
    item = one(root)

    assert item["occurred_at"] is None and "no Date header" in item["metadata"]["time_note"]
    assert item["observed_at"]


# -- text and provenance ------------------------------------------------------

@pytest.mark.parametrize("body", [
    "Grüß Gott — 請求書は月曜締切です。\r\n",
    "La réunion est reportée à jeudi.\r\n",
    "Η συνάντηση μετατέθηκε. Привет.\r\n",
])
def test_multilingual_text_survives_the_read(tmp_path, body):
    root = tree(tmp_path, {"inbox.eml": eml(body=body)})
    item = one(root)

    assert "\ufffd" not in item["text"]
    assert item["text"].startswith(body.split(" \r")[0][:12])


def test_html_is_read_as_text_and_the_signature_is_not_body(tmp_path):
    root = tree(tmp_path, {"inbox.eml": eml(
        content="text/html; charset=utf-8",
        body="<p>Café <b>meeting</b> moved.</p><script>evil()</script>"
             "<p>signature</p>")})
    item = one(root)

    assert "Café meeting moved." in item["text"]
    assert "evil()" not in item["text"] and "<" not in item["text"]
    assert "evil" not in item["text"] and "<p>" not in item["text"]
    assert item["metadata"]["media_type"].startswith("text/html/")


def test_a_quoted_reply_keeps_its_provenance_as_headers(tmp_path):
    item = one(tree(tmp_path, {"inbox.eml": eml()}))
    meta = item["metadata"]

    assert meta["reply_to"] == "z9@example.com"
    assert meta["references"] == ["q1@example.com", "z9@example.com"]
    assert meta["quoted_lines"] == 1, "copied text is context, not a second event"


def test_a_forwarded_message_says_so(tmp_path):
    root = tree(tmp_path, {"inbox.eml": eml(body="Begin forwarded message:\r\n"
                                                "> They never called back.\r\n")})
    assert one(root)["metadata"]["forwarded"] is True


def test_participants_are_kept_separately_from_the_sender(tmp_path):
    meta = one(tree(tmp_path, {"inbox.eml": eml()}))["metadata"]

    assert meta["from"] == {"name": "Anna", "address": "anna@example.com"}
    assert [person["address"] for person in meta["participants"]] == [
        "me@example.com", "boss@example.com"]


def test_the_body_is_bound_and_the_truncation_is_recorded(tmp_path):
    root = tree(tmp_path, {"inbox.eml": eml(body="x" * 300_000)})
    item = one(root)

    assert len(item["text"]) == 120_000
    assert item["metadata"]["truncated"] is True


# -- attachments and bounds ---------------------------------------------------

def test_attachments_are_named_but_not_copied(tmp_path):
    payload = base64.b64encode(b"%PDF-1.4\n").decode()
    root = tree(tmp_path, {"inbox.eml": eml(
        extra='Content-Type: multipart/mixed; boundary="B"\r\n\r\n'
              '--B\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nSee attached.\r\n'
              '--B\r\nContent-Type: application/pdf\r\n'
              'Content-Disposition: attachment; filename="invoice-27.pdf"\r\n'
              'Content-Transfer-Encoding: base64\r\n\r\n' + payload + "\r\n--B--\r\n",
        content="text/plain")})
    item = one(root)
    attached = item["metadata"]["attachments"]

    assert attached == [{"name": "invoice-27.pdf", "media_type": "application/pdf",
                         "bytes": 9, "inline": False, "ingested": False}]
    assert "%PDF" not in item["text"]


def test_a_message_over_the_per_message_bound_is_a_gap(tmp_path):
    root = tree(tmp_path, {"huge.eml": eml(body="x" * (MAX_MESSAGE_BYTES + 10))})
    page = source(root).read_page(None)

    assert page.envelopes == ()
    assert "bound" in page.skipped[0].reason


def test_a_message_with_no_text_part_is_a_gap_not_a_crash(tmp_path):
    root = tree(tmp_path, {"inbox.eml": eml(
        extra='Content-Type: multipart/mixed; boundary="B"\r\n\r\n'
              '--B\r\nContent-Type: image/png\r\nContent-Transfer-Encoding: base64\r\n'
              '\r\naWQ0\r\n--B--\r\n', content="text/plain", body="")})
    page = source(root).read_page(None)

    assert page.envelopes == ()
    assert "no text part" in page.skipped[0].reason


def test_text_that_did_not_survive_its_character_set_is_not_kept(tmp_path):
    root = tree(tmp_path, {"inbox.eml": (
        b"Message-ID: <e5@example.com>\r\nDate: Tue, 15 Sep 2026 09:14:02 +0200\r\n"
        b"Subject: broken\r\nContent-Type: text/plain; charset=us-ascii\r\n\r\n"
        b"caf\xe9 was not ascii")})
    page = source(root).read_page(None)

    assert page.envelopes == ()
    assert "character set" in page.skipped[0].reason


def test_the_charset_a_part_declared_is_recorded(tmp_path):
    head = (b"Message-ID: <f6@example.com>\r\nDate: Tue, 15 Sep 2026 09:14:02 +0200\r\n"
            b"Subject: turkish\r\nContent-Type: text/plain; charset=iso-8859-9\r\n\r\n")
    root = tree(tmp_path, {"inbox.eml": head + "Grüß GOTT".encode("iso-8859-9")})
    item = one(root)

    assert item["metadata"]["media_type"] == "text/plain/iso-8859-9"
    assert "\ufffd" not in item["text"]


def test_a_symlink_out_of_the_export_is_not_ingested(tmp_path):
    outside = tmp_path.parent / "elsewhere.eml"
    outside.write_bytes(eml("<elsewhere@example.com>"))
    tree(tmp_path, {"inbox.eml": eml()})
    (tmp_path / "link.eml").symlink_to(outside)

    page = source(tmp_path).read_page(None)

    assert ids_of(page) == ["a1@example.com"]
    assert "elsewhere" not in json.dumps(page.envelopes)
    assert source(tmp_path).check()["candidates"] == 1


def test_a_hidden_file_in_the_export_is_left_alone(tmp_path):
    tree(tmp_path, {"inbox.eml": eml(), ".tmp.eml": eml("<hidden@example.com>"),
                   "draft": eml("<no-suffix@example.com>")})
    page = source(tmp_path).read_page(None)
    assert ids_of(page) == ["a1@example.com"]


# -- paging -------------------------------------------------------------------

def test_pages_are_bounded_and_advance_by_position(tmp_path):
    tree(tmp_path, {f"m{i}.eml": eml(message_id=f"<m{i}@example.com>") for i in range(5)})
    adapter = source(tmp_path, records_per_page=2)

    first = adapter.read_page(None)
    second = adapter.read_page(first.next_cursor)
    third = adapter.read_page(second.next_cursor)

    assert ids_of(first) == ["m0@example.com", "m1@example.com"]
    assert ids_of(second) == ["m2@example.com", "m3@example.com"]
    assert ids_of(third) == ["m4@example.com"] and third.next_cursor is None


def test_a_page_reports_the_relays_it_claims(tmp_path):
    meta = one(tree(tmp_path, {"inbox.eml": eml()}))["metadata"]

    assert meta["relays"] == 1
    assert meta["bytes"] == (tmp_path / "inbox.eml").stat().st_size


# -- the store's own view -----------------------------------------------------

def test_envelopes_are_accepted_by_the_canonical_commit(tmp_path, store):
    tree(tmp_path, {"inbox.eml": eml()})
    page = source(tmp_path).read_page(None)

    prepared = prepare_envelope(page.envelopes[0])
    assert prepared.source == "email" and prepared.kind == "email"
    record = store.commit(page.envelopes[0])
    assert store.get(record["id"]).text.startswith("Die Zahlung")


def test_an_attachment_helper_is_usable_on_its_own(tmp_path):
    import email
    import email.policy

    message = email.message_from_bytes(eml(), policy=email.policy.default)
    assert attachment_descriptors(message) == []


def test_a_blank_subject_and_no_recipients_is_still_a_record(tmp_path):
    root = tree(tmp_path, {"inbox.eml": eml(subject="",
                                            body="A note with no header value.\r\n")})
    meta = one(root)["metadata"]

    assert meta["subject"] is None and meta["participants"]


# -- the live connector -----------------------------------------------------

import base64 as _b64

from hermes_memory.sources.runtime import COMPLETE, PAUSED, ConnectorRuntime
from hermes_memory.sources.sync import SyncController


def encoded(raw: bytes) -> str:
    return _b64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class Mailbox:
    """A Gmail-shaped transport that records every call it was asked to make."""

    def __init__(self, *, pages=None, history=None, bodies=None, profile=None,
                 raises=None, message_raises=None):
        self.pages = list(pages or [{}])
        self.history_pages = list(history or [{}])
        self.bodies = dict(bodies or {})
        self.profile_report = profile if profile is not None else {
            "address": "me@example.com", "scopes": ["https://mail.google.com/"]}
        self.raises = raises
        self.message_raises = message_raises
        self.calls: list[tuple] = []

    def profile(self):
        self.calls.append(("profile",))
        if self.raises is not None:
            raise self.raises
        return self.profile_report

    def messages(self, *, query, page_token, max_results):
        self.calls.append(("messages", query, page_token, max_results))
        if self.raises is not None:
            raise self.raises
        return self.pages[min(len(self.calls) - 1, len(self.pages) - 1)]

    def message(self, message_id):
        self.calls.append(("message", message_id))
        if self.message_raises is not None:
            raise self.message_raises
        return self.bodies.get(message_id)

    def history(self, *, since, query):
        self.calls.append(("history", since, query))
        if self.raises is not None:
            raise self.raises
        return self.history_pages[0] if self.history_pages else {}


def gmail(api=None, **kwargs) -> GmailSource:
    options = {"query": "label:work", "enabled": True}
    options.update(kwargs)
    return GmailSource(api if api is not None else Mailbox(), **options)


def entry(message_id="m1", *, thread="t1", raw=None, when="1758000000000", body=eml()):
    return {"id": message_id, "threadId": thread, "internalDate": when,
            "raw": encoded(body) if raw is None else raw}


def test_the_live_connector_is_off_until_the_owner_switches_it_on():
    api = Mailbox()
    subject = GmailSource(api, query="label:work")
    report = subject.check()
    assert report["ok"] is False and "not enabled" in report["reason"]
    with pytest.raises(EvidenceError, match="owner action"):
        subject.read_page(None)
    assert api.calls == []


def test_reading_somebodys_mail_needs_an_operator_written_scope():
    with pytest.raises(TypeError, match="required keyword-only"):
        GmailSource(Mailbox())
    with pytest.raises(EvidenceError, match="operator-written query"):
        GmailSource(Mailbox(), query="   ")
    with pytest.raises(EvidenceError, match="operator-written query"):
        GmailSource(Mailbox(), query=17)


@pytest.mark.parametrize("missing", ["profile", "messages", "message", "history"])
def test_a_transport_that_cannot_do_one_of_the_four_is_refused_at_wiring(missing):
    api = Mailbox()
    setattr(api, missing, None)
    with pytest.raises(EvidenceError, match=f"offers no {missing}"):
        GmailSource(api, query="label:work", enabled=True)


def test_the_probe_reports_the_grant_and_reads_no_message():
    api = Mailbox(pages=[{"items": [entry()]}])
    report = gmail(api).check()
    assert report["ok"] is True and report["content_read"] is False
    assert report["address"] == "me@example.com" and report["readonly"] is True
    assert api.calls == [("profile",)]


def test_a_revoked_grant_is_named_as_one_rather_than_as_a_wobbly_connection():
    api = Mailbox(raises=PermissionError("403 Insufficient Permission: invalid_grant"))
    report = gmail(api).check()
    assert report["ok"] is False and report["coverage_state"] == "revoked"


def test_an_endpoint_that_is_simply_down_is_not_called_a_revocation():
    report = gmail(Mailbox(raises=OSError("connection timed out"))).check()
    assert report["ok"] is False and "coverage_state" not in report


def test_a_quota_refusal_on_a_read_is_an_outage_carrying_the_servers_words():
    api = Mailbox(raises=RuntimeError("429 User rate exceeded, retry in 30 seconds"))
    with pytest.raises(EvidenceError, match="429 User rate exceeded"):
        gmail(api).read_page(None)


def test_the_message_is_identified_by_its_own_header_rather_than_by_mailbox_numbers():
    page = gmail(Mailbox(pages=[{"items": [entry()]}])).read_page(None)
    assert [item["source_id"] for item in page.envelopes] == ["a1@example.com"]
    assert page.envelopes[0]["metadata"]["gmail_id"] == "m1"
    assert page.envelopes[0]["metadata"]["thread_id"] == "t1"


def test_the_live_read_and_the_export_of_one_message_agree_about_what_it_said(tmp_path):
    """Overlapping channels must not disagree, and must not merge into one row."""
    root = tree(tmp_path, {"inbox.eml": eml()})
    export = EmailSource(root).read_page(None).envelopes[0]
    live = gmail(Mailbox(pages=[{"items": [entry()]}])).read_page(None).envelopes[0]
    assert live["source_id"] == export["source_id"] == "a1@example.com"
    assert live["text"] == export["text"]
    assert live["source"] != export["source"]
    assert live["metadata"]["quoted_lines"] == export["metadata"]["quoted_lines"]


def test_the_labels_a_message_carries_are_not_part_of_its_fingerprint():
    """They change when mail is read or filed, and would make one message two revisions."""
    api = Mailbox(pages=[{"items": [entry()]}])
    page = gmail(api).read_page(None)
    assert "label" not in json.dumps(page.envelopes[0]).lower()


def test_the_relay_time_is_kept_beside_the_time_the_author_claimed():
    page = gmail(Mailbox(pages=[{"items": [entry(when="1758000000000")]}])).read_page(None)
    metadata = page.envelopes[0]["metadata"]
    assert metadata["received_at"] == "2025-09-16T05:20:00+00:00"
    assert page.envelopes[0]["occurred_at"] == "2026-09-15T07:14:02+00:00"


def test_a_relay_stamp_that_is_not_a_plausible_instant_is_nobody_s_date():
    for value in ("soon", "-1", "99999999999999", None):
        page = gmail(Mailbox(pages=[{"items": [entry(when=value)]}])).read_page(None)
        assert page.envelopes[0]["metadata"]["received_at"] is None, value


def test_a_listing_without_a_body_spends_one_more_call_rather_than_losing_the_message():
    api = Mailbox(pages=[{"items": [{"id": "m1", "threadId": "t1"}]}],
                  bodies={"m1": {"raw": encoded(eml()), "threadId": "t7"}})
    page = gmail(api).read_page(None)
    assert ("message", "m1") in api.calls
    assert page.envelopes[0]["metadata"]["thread_id"] == "t7"


def test_a_message_that_cannot_be_fetched_is_a_gap_named_by_its_id():
    api = Mailbox(pages=[{"items": [{"id": "m1"}]}], message_raises=RuntimeError("404 not found"))
    page = gmail(api).read_page(None)
    assert page.envelopes == ()
    assert page.skipped[0].ref == "gmail:m1"
    assert "404 not found" in page.skipped[0].reason


def test_a_body_that_is_not_base64url_is_a_gap_rather_than_an_empty_message():
    page = gmail(Mailbox(pages=[{"items": [entry(raw="!!!!not base64!!!!")]}])).read_page(None)
    assert page.skipped[0].reason.startswith("the raw message could not be decoded")


def test_a_message_bigger_than_the_bound_is_refused_whole_rather_than_in_parts():
    oversized = "a" * (MAX_MESSAGE_BYTES // 3 * 4 + 8)
    page = gmail(Mailbox(pages=[{"items": [entry(raw=oversized)]}])).read_page(None)
    assert page.skipped[0].reason.endswith("byte per-message bound")


def test_an_entry_with_no_id_cannot_be_got_at_and_is_a_gap():
    page = gmail(Mailbox(pages=[{"items": [{"threadId": "t1"}]}])).read_page(None)
    assert page.skipped[0].ref == "gmail#0"
    assert page.skipped[0].reason == "the mailbox listed an entry with no id"


def test_one_message_listed_twice_in_one_answer_is_recorded_once():
    page = gmail(Mailbox(pages=[{"items": [entry(), entry()]}])).read_page(None)
    assert len(page.envelopes) == 1
    assert page.skipped[0].reason == "the mailbox offered this message twice"


def test_a_backfill_pages_by_the_query_token_then_switches_to_the_history_head():
    api = Mailbox(pages=[{"items": [entry()], "next_page_token": "p2", "history_id": "h9"},
                         {"items": [entry("m2", body=eml("<a2@example.com>"))],
                          "history_id": "h9"}])
    subject = gmail(api)
    first = subject.read_page(None)
    assert first.next_cursor == "p:p2"
    second = subject.read_page(first.next_cursor)
    assert second.next_cursor == "h:h9"
    assert api.calls[-1] == ("messages", "label:work", "p2", 100)


def test_a_history_read_that_found_nothing_is_current_rather_than_a_moving_position():
    api = Mailbox(pages=[{"items": [entry()], "history_id": "h9"}],
                  history=[{"items": [], "history_id": "h9"}])
    subject = gmail(api)
    caught_up = subject.read_page("h:h9")
    assert caught_up.next_cursor is None
    assert api.calls[-1] == ("history", "h9", "label:work")


def test_a_history_read_that_found_something_advances_to_the_new_head():
    arrived = entry("m2", body=eml("<a2@example.com>"), when=None)
    api = Mailbox(history=[{"items": [arrived], "history_id": "h10"}])
    page = gmail(api).read_page("h:h9")
    assert page.next_cursor == "h:h10"
    assert [item["source_id"] for item in page.envelopes] == ["a2@example.com"]


def test_a_position_this_connector_never_wrote_starts_the_reading_over():
    with pytest.raises(CursorExpired, match="not one this connector writes"):
        gmail(Mailbox()).read_page("g8f2h4")


def test_a_message_that_left_the_mailbox_is_reported_and_not_quietly_forgotten():
    api = Mailbox(pages=[{"items": [], "removed": ["m7"], "history_id": "h9"}])
    page = gmail(api).read_page(None)
    assert [row.ref for row in page.skipped] == ["m7"]
    assert "does not propagate a deletion" in page.skipped[0].reason
    assert page.next_cursor == "h:h9"


def test_a_full_page_says_where_it_stopped_rather_than_dropping_the_rest():
    entries = [entry(f"m{index}", body=eml(f"<{index}@example.com>")) for index in range(6)]
    api = Mailbox(pages=[{"items": entries, "next_page_token": "p2", "history_id": "h9"}])
    page = gmail(api, records_per_page=2).read_page(None)
    assert len(page.envelopes) == 2
    assert page.skipped[-1].reason.endswith("it is read again")
    assert page.next_cursor == "p:p2"


def test_a_live_mailbox_and_its_poll_reach_the_store_once(store):
    sync = SyncController(store)
    sync.register("gmail", policy_version="private-api")
    runtime = ConnectorRuntime(store, sync, holder="worker-a")
    mailbox = Mailbox(pages=[{"items": [entry()], "history_id": "h9"}],
                      history=[{"items": [], "history_id": "h9"}])
    first = runtime.run(gmail(mailbox))
    assert (first.stopped, first.records, first.coverage_state) == (COMPLETE, 1, "current")
    assert first.cursor == "h:h9"
    assert ("history", "h9", "label:work") in mailbox.calls

    again = Mailbox(pages=[{"items": [entry()], "history_id": "h9"}],
                    history=[{"items": [], "history_id": "h9"}])
    second = runtime.run(gmail(again))
    assert (second.records, second.repeats, second.coverage_state) == (0, 1, "current")


def test_a_stopped_connector_never_reaches_for_the_mailbox(store):
    sync = SyncController(store)
    sync.register("gmail", policy_version="private-api")
    store.set_control("gmail", "capture", "paused", actor="owner-principal",
                      reason="the owner stopped it", policy_version="v1")
    api = Mailbox(pages=[{"items": [entry()]}])
    run = ConnectorRuntime(store, sync, holder="worker-a").run(gmail(api))
    assert run.stopped == PAUSED and api.calls == []


def test_a_signature_is_not_part_of_what_the_message_said(tmp_path):
    root = tree(tmp_path, {"sig.eml": eml(
        body="The transfer cleared on Friday.\r\n-- \r\nAnna Reyes\r\n07700 900982\r\n")})
    text = one(root)["text"]
    assert text.strip() == "The transfer cleared on Friday."
    assert "07700" not in text


def test_a_page_stops_at_its_byte_budget_rather_than_its_count(tmp_path):
    big = eml("<b1@example.com>", body="x" * 90_000)
    root = tree(tmp_path, {"1.eml": big, "2.eml": big, "3.eml": big})
    adapter = source(root)
    adapter.capabilities = replace(adapter.capabilities, max_bytes_per_page=100_000)
    page = adapter.read_page(None)
    assert len(page.envelopes) == 1
    assert page.next_cursor == "1.eml"


def test_more_attachments_than_the_list_holds_are_capped_and_said(tmp_path):
    parts = "".join(
        f"--B\r\nContent-Type: application/octet-stream\r\n"
        f"Content-Disposition: attachment; filename=\"part{i}.bin\"\r\n"
        f"Content-Transfer-Encoding: base64\r\n\r\nSVBG\r\n" for i in range(30))
    root = tree(tmp_path, {"many.eml": eml(
        extra='Content-Type: multipart/mixed; boundary="B"\r\n\r\n--B\r\n'
              'Content-Type: text/plain; charset=utf-8\r\n\r\nThirty files.\r\n'
              + parts + "--B--\r\n")})
    found = one(root)["metadata"]["attachments"]
    assert len(found) == 20
    assert all(row["ingested"] is False for row in found)


def test_a_message_with_no_text_at_all_is_distinguished_from_an_empty_one(tmp_path):
    binary = tree(tmp_path, {"only-image.eml": eml(
        extra='Content-Type: image/png\r\nContent-Transfer-Encoding: base64\r\n\r\n'
              'iVBORw0KGgo=\r\n', content="image/png", body="")})
    empty = tree(tmp_path / "second", {"blank.eml": eml(body="   \r\n")})
    assert source(binary).read_page(None).skipped[0].reason == "no text part could be decoded"
    assert source(empty).read_page(None).skipped[0].reason == "the text part is empty"
