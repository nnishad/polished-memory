"""C2 WhatsApp export: a chat log without ids, without zones, and without lies.

The gate items are the ones an export actually hits: timezone ambiguity, wrapped
messages, media placeholders that are not the media, system notices, malformed
lines and re-reads of an append-only file.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from dataclasses import replace

from hermes_memory.sources.whatsapp_export import WhatsAppExport
from hermes_memory.storage.evidence import prepare_envelope

CHAT = (
    "15/09/2026, 09:14 - Anna: Die Zahlung ist am Montag fällig\n"
    "   und ich habe sie schon erinnert\n"
    "[15/09/2026, 09:15:02] Bob: <attached: 00000012-PHOTO.jpg>\n"
    "Messages and calls are end-to-end encrypted.\n"
    "[15/09/2026, 09:16] Anna: Hello ✅\n"
    "16/09/2026, 20:01 - Bob: invoice 27 is attached\n")


def chat(root: pathlib.Path, text: str = CHAT, name: str = "_chat.txt") -> pathlib.Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return root


def read(root, **kwargs):
    return WhatsAppExport(root, **kwargs).read_page(None)


def texts(page):
    return [item["text"] for item in page.envelopes]


# -- declaring what it can do -------------------------------------------------

def test_the_check_counts_chats_without_reading_them(tmp_path, monkeypatch):
    chat(tmp_path)
    monkeypatch.setattr(pathlib.Path, "read_text",
                        lambda self, **kw: (_ for _ in ()).throw(AssertionError("read")))
    report = WhatsAppExport(tmp_path).check()

    assert report["ok"] is True and report["chats"] == 1
    assert report["content_read"] is False


def test_an_export_declares_itself_history_only(tmp_path):
    capabilities = WhatsAppExport(tmp_path).capabilities

    assert capabilities.history and not capabilities.live
    assert not capabilities.deletion_events, "a text export cannot report forgetting"


# -- time, which is the hard part --------------------------------------------

def test_a_zoneless_export_leaves_every_message_unplaced(tmp_path):
    page = read(chat(tmp_path))

    assert all(item["occurred_at"] is None for item in page.envelopes)
    assert all(item["occurred_precision"] == "unknown" for item in page.envelopes)
    assert "states no timezone" in page.envelopes[0]["metadata"]["time_note"]


def test_the_zone_the_operator_named_is_the_zone_used(tmp_path):
    berlin = read(chat(tmp_path), timezone_name="Europe/Berlin")
    saopaulo = read(chat(tmp_path), timezone_name="America/Sao_Paulo")

    assert berlin.envelopes[-1]["occurred_at"] == "2026-09-16T18:01:00+00:00"
    assert saopaulo.envelopes[-1]["occurred_at"] == "2026-09-16T23:01:00+00:00"
    assert berlin.envelopes[-1]["metadata"]["time_basis"] == "Europe/Berlin"


def test_an_invented_zone_is_refused_rather_than_defaulted(tmp_path):
    with pytest.raises(ValueError, match="not guessed"):
        WhatsAppExport(tmp_path, timezone_name="Mars/Olympus")


def test_a_clock_reading_without_seconds_is_minute_precise(tmp_path):
    root = chat(tmp_path, "[15/09/2026, 09:16] Anna: at a minute\n"
                          "[15/09/2026, 09:17:31] Bob: to the second\n")
    page = read(root, timezone_name="UTC")

    assert [item["occurred_precision"] for item in page.envelopes] == ["minute", "second"]


def test_the_american_and_european_date_forms_are_both_read(tmp_path):
    root = chat(tmp_path, "[9/15/2026, 9:14:02 AM] Anna: morning\n"
                         "15/09/2026, 09:15 - Bob: after\n", name="_chat_us.txt")
    page = read(root, timezone_name="UTC")

    assert [item["occurred_at"] for item in page.envelopes] == [
        "2026-09-15T09:14:02+00:00", "2026-09-15T09:15:00+00:00"]


# -- message structure --------------------------------------------------------

def test_a_wrapped_message_stays_one_message(tmp_path):
    page = read(chat(tmp_path), timezone_name="UTC")

    assert page.envelopes[0]["text"].startswith("Die Zahlung")
    assert "ich habe sie schon erinnert" in page.envelopes[0]["text"]
    assert page.envelopes[0]["metadata"]["lines"] == 2


def test_media_placeholders_are_gaps_naming_the_message(tmp_path):
    page = read(chat(tmp_path), timezone_name="UTC")

    assert [gap.ref for gap in page.skipped] == ["_chat.txt#1"]
    assert "media placeholder" in page.skipped[0].reason
    assert not any("attached:" in text for text in texts(page))


def test_a_system_notice_is_its_own_record_and_not_evidence(tmp_path):
    page = read(chat(tmp_path), timezone_name="Europe/Berlin")
    notices = [item for item in page.envelopes if item["kind"] == "system_notice"]

    assert [item["source_id"] for item in notices] == ["_chat.txt#2"]
    assert notices[0]["metadata"]["system"] is True
    assert "no timestamp" in notices[0]["metadata"]["time_note"]


def test_prose_about_leaving_is_not_mistaken_for_a_notice(tmp_path):
    root = chat(tmp_path, "[15/09/2026, 09:16] Anna: they left at noon\n")
    page = read(root, timezone_name="UTC")

    assert page.envelopes[0]["kind"] == "message"
    assert page.envelopes[0]["text"] == "they left at noon"


def test_an_orphan_line_before_the_first_message_is_reported(tmp_path):
    root = chat(tmp_path, "a line with no stamp at all\n"
                          "[15/09/2026, 09:16] Anna: hi\n")
    page = read(root, timezone_name="UTC")

    assert [gap.ref for gap in page.skipped] == ["_chat.txt#0"]
    assert len(page.envelopes) == 1


def test_an_empty_export_is_no_records_and_no_crash(tmp_path):
    assert read(chat(tmp_path, "")).envelopes == ()


def test_multilingual_and_emoji_text_survives(tmp_path):
    root = chat(tmp_path, "[15/09/2026, 09:16] Anna: 請求書は月曜締切です ✅ Привет\n")
    text = read(root, timezone_name="UTC").envelopes[0]["text"]

    assert "請求書は月曜締切です" in text and "✅" in text and "Привет" in text
    assert "￼" not in text and "\ufffd" not in text


def test_a_file_that_is_not_utf8_is_reported(tmp_path):
    (tmp_path / "_chat.txt").write_bytes(b"[15/09/2026, 09:16] Anna: \xff\xfe broken\n")
    page = read(tmp_path, timezone_name="UTC")

    assert page.envelopes == ()
    assert "read failed" in page.skipped[0].reason


# -- identity and paging ------------------------------------------------------

def test_the_position_is_the_identity_and_survives_a_re_read(tmp_path):
    root = chat(tmp_path)

    first = read(root, timezone_name="UTC")
    second = read(root, timezone_name="UTC")

    assert [item["source_id"] for item in first.envelopes] == \
        [item["source_id"] for item in second.envelopes]
    assert first.envelopes[0]["revision"] == "1"


def test_appending_to_the_export_does_not_renumber_what_was_read(tmp_path):
    root = chat(tmp_path)
    before = [item["source_id"] for item in read(root, timezone_name="UTC").envelopes]

    (root / "_chat.txt").write_text(CHAT + "[17/09/2026, 08:00] Bob: later\n",
                                    encoding="utf-8")
    after = read(root, timezone_name="UTC").envelopes

    assert [item["source_id"] for item in after][:len(before)] == before
    assert after[-1]["source_id"] == "_chat.txt#5"


def test_two_chats_page_independently_by_position(tmp_path):
    chat(tmp_path, name="ann/_chat.txt")
    chat(tmp_path, name="bob/_chat.txt")
    adapter = WhatsAppExport(tmp_path, timezone_name="UTC", records_per_page=2)

    first = adapter.read_page(None)
    second = adapter.read_page(first.next_cursor)

    assert [item["metadata"]["chat"] for item in first.envelopes][:1] == ["ann/_chat.txt"]
    assert second.next_cursor is None or second.envelopes
    records, gaps = adapter.read_all()
    assert {item["metadata"]["chat"] for item in records} == \
        {"ann/_chat.txt", "bob/_chat.txt"}


def test_a_symlinked_chat_out_of_the_export_is_not_read(tmp_path):
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("[15/09/2026, 09:16] Anna: not for you\n", encoding="utf-8")
    chat(tmp_path)
    (tmp_path / "_chat2.txt").symlink_to(outside)

    assert read(tmp_path, timezone_name="UTC").envelopes


def test_a_page_is_bounded_and_the_next_cursor_continues_it(tmp_path):
    body = "".join(f"[15/09/2026, 09:{minute:02d}] Anna: message {minute}\n"
                   for minute in range(12))
    root = chat(tmp_path, body)
    adapter = WhatsAppExport(root, timezone_name="UTC", records_per_page=5)

    page = adapter.read_page(None)

    assert len(page.envelopes) == 5
    assert page.next_cursor is not None
    records, gaps = adapter.read_all()
    assert len(records) == 12 and not gaps


# -- the store's own view -----------------------------------------------------

def test_envelopes_are_accepted_by_the_canonical_commit(tmp_path, store):
    page = read(chat(tmp_path), timezone_name="UTC")
    prepared = prepare_envelope(page.envelopes[0])

    assert prepared.source == "whatsapp" and prepared.occurred_at
    record = store.commit(page.envelopes[0])
    assert store.get(record["id"]).metadata["author"] == "Anna"


def test_a_message_with_nothing_in_it_is_a_gap_rather_than_an_empty_record(tmp_path):
    root = chat(tmp_path, "[15/09/2026, 09:14] Anna: \n"
                         "15/09/2026, 09:15 - Bob: A real message\n")
    page = WhatsAppExport(root).read_page(None)
    assert [gap.reason for gap in page.skipped] == ["the message is empty"]
    assert [item["text"] for item in page.envelopes] == ["A real message"]


def test_a_link_out_of_the_export_directory_is_not_a_chat(tmp_path):
    outside = tmp_path.parent / "elsewhere_chat.txt"
    outside.write_text("15/09/2026, 09:14 - Anna: Nobody pointed the connector here\n",
                       encoding="utf-8")
    root = chat(tmp_path)
    (root / "link_chat.txt").symlink_to(outside)
    page = WhatsAppExport(root).read_page(None)
    assert "elsewhere" not in json.dumps(page.envelopes)
    assert WhatsAppExport(root).check()["chats"] == 1


def test_a_chat_log_too_big_for_one_page_says_so_rather_than_vanishing(tmp_path):
    root = chat(tmp_path)
    chat(tmp_path, "15/09/2026, 09:14 - Anna: " + "x" * 5000, name="_huge_chat.txt")
    subject = WhatsAppExport(root)
    subject.capabilities = replace(subject.capabilities, max_bytes_per_page=2000)
    page = subject.read_page(None)
    oversized = [gap for gap in page.skipped if gap.ref == "_huge_chat.txt"]
    assert len(oversized) == 1 and "byte per-page bound" in oversized[0].reason
    assert subject.check()["chats"] == 2
