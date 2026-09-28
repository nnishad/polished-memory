"""C2 adapter contract: bounded paging, malformed input and honest time handling."""
from __future__ import annotations

import json

import pytest

from hermes_memory.sources.base import (REDACTED, Page, normalize_text,
                                        normalize_time, part_container, revocation)
from hermes_memory.sources.files import FileSource
from hermes_memory.storage.evidence import EvidenceError


def tree(root, files: dict[str, str | bytes]):
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body if isinstance(body, bytes) else body.encode("utf-8"))
    return root


def drained(source: FileSource):
    """Read every page, asserting the cursor always terminates."""
    envelopes, skipped, cursor, pages = [], [], None, 0
    while True:
        page = source.read_page(cursor)
        envelopes.extend(page.envelopes)
        skipped.extend(page.skipped)
        pages += 1
        assert pages < 50, "paging did not terminate"
        if page.next_cursor is None:
            return envelopes, skipped, pages
        cursor = page.next_cursor


# -- time --------------------------------------------------------------------

def test_an_untimestamped_note_has_no_invented_event_time(tmp_path):
    tree(tmp_path, {"note.md": "# Moved house\n"})
    (record,), _, _ = drained(FileSource(tmp_path))
    assert record["occurred_at"] is None
    assert record["occurred_precision"] == "unknown"
    assert "no time reported" in record["metadata"]["time_note"]


def test_an_offset_free_datetime_is_not_silently_assumed_local_or_utc(tmp_path):
    tree(tmp_path, {"note.md": "date: 2026-03-04 17:00\nbody\n"})
    (record,), _, _ = drained(FileSource(tmp_path))
    assert record["occurred_at"] is None
    assert "no timezone offset" in record["metadata"]["time_note"]


@pytest.mark.parametrize("text, expected", [
    ("2026-03-04T17:00:00+09:00", "second"),
    ("2026-03-04T17:00:00Z", "second"),
    ("2026-03-04T17:00+00:00", "minute"),
    ("2026-03-04", "day"),
])
def test_unambiguous_times_keep_their_precision(text, expected):
    occurred, precision, note = normalize_time(text)
    assert precision == expected
    assert occurred is not None and note in (None, "date-only source time, day precision")


def test_garbage_times_degrade_to_unknown_rather_than_raising():
    for value in ("yesterday", "17/03/2026", 1_700_000_000, None, ""):
        occurred, precision, note = normalize_time(value)
        assert (occurred, precision) == (None, "unknown"), value
        assert note


# -- malformed and hostile input --------------------------------------------

def test_invalid_bytes_are_reported_not_replacement_mojibake(tmp_path):
    tree(tmp_path, {"broken.txt": b"\xff\xfe\x00not text"})
    records, skipped, _ = drained(FileSource(tmp_path))
    assert records == []
    assert "UTF-8" in skipped[0].reason


def test_an_empty_file_is_a_reported_gap(tmp_path):
    tree(tmp_path, {"blank.md": "   \n"})
    records, skipped, _ = drained(FileSource(tmp_path))
    assert records == [] and skipped[0].reason == "file is empty"


def test_one_malformed_export_does_not_stop_the_others(tmp_path):
    tree(tmp_path, {"0-good.md": "kept\n", "1-bad.json": "{not json",
                   "2-good.json": json.dumps([{"id": "a", "text": "also kept"}])})
    records, skipped, _ = drained(FileSource(tmp_path))
    assert {record["source_id"] for record in records} == {"0-good.md", "a"}
    assert "1-bad.json" in skipped[0].ref and "malformed" in skipped[0].reason


def test_a_symlink_leaving_the_export_directory_is_not_ingested(tmp_path):
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("not part of the export\n", encoding="utf-8")
    tree(tmp_path, {"real.md": "in scope\n"})
    (tmp_path / "link.md").symlink_to(outside)
    records, _, _ = drained(FileSource(tmp_path))
    assert [record["source_id"] for record in records] == ["real.md"]


def test_dotfiles_and_unsupported_types_are_excluded_not_forgotten(tmp_path):
    tree(tmp_path, {".hidden.md": "x\n", "notes.md": "y\n", "image.png": "z"})
    records, skipped, _ = drained(FileSource(tmp_path))
    assert [record["source_id"] for record in records] == ["notes.md"]
    assert skipped == [], "out-of-scope files are not a coverage gap"


# -- structure ---------------------------------------------------------------

def test_json_arrays_and_wrapped_lists_become_individual_records(tmp_path):
    tree(tmp_path, {
        "plain.json": json.dumps([{"id": "m1", "text": "first", "time": "2026-01-02T03:04:05Z"}]),
        "wrapped.json": json.dumps({"messages": [{"body": "second"}, {"body": "   "},
                                                 {"text": "third"}]}),
        "lines.jsonl": "\n".join(json.dumps({"text": f"line {i}"}) for i in range(3)),
    })
    records, skipped, _ = drained(FileSource(tmp_path))
    texts = sorted(record["text"] for record in records)
    assert texts == ["first", "line 0", "line 1", "line 2", "second", "third"]
    assert skipped == []
    assert next(r for r in records if r["text"] == "first")["occurred_precision"] == "second"


def test_multilingual_text_survives_the_round_trip(tmp_path):
    tree(tmp_path, {"jp.txt": "会議は木曜の午後3時に変わりました。\n"})
    (record,), _, _ = drained(FileSource(tmp_path))
    assert record["text"].startswith("会議")
    assert record["metadata"]["media_type"] == "text/plain"


def test_an_edited_file_becomes_a_new_revision_of_the_same_source(tmp_path):
    tree(tmp_path, {"note.md": "v1\n"})
    first, _, _ = drained(FileSource(tmp_path))
    tree(tmp_path, {"note.md": "v2 edited\n"})
    second, _, _ = drained(FileSource(tmp_path))
    assert first[0]["source_id"] == second[0]["source_id"] == "note.md"
    assert first[0]["revision"] != second[0]["revision"]


def test_an_unchanged_file_replays_with_an_identical_revision(tmp_path):
    tree(tmp_path, {"note.md": "same\n"})
    first, _, _ = drained(FileSource(tmp_path))
    second, _, _ = drained(FileSource(tmp_path))
    assert first[0]["revision"] == second[0]["revision"]


# -- paging ------------------------------------------------------------------

def test_pages_respect_the_record_bound_and_reach_the_same_total(tmp_path):
    tree(tmp_path, {f"n{i:02d}.md": f"body {i}\n" for i in range(12)})
    source = FileSource(tmp_path, records_per_page=5)
    _, _, pages = drained(source)
    assert pages == 3
    records, _, _ = drained(FileSource(tmp_path, records_per_page=5))
    assert len(records) == 12


def test_paging_order_is_stable_across_runs(tmp_path):
    tree(tmp_path, {f"n{i:02d}.md": f"body {i}\n" for i in range(9)})
    first = [r["source_id"] for r in drained(FileSource(tmp_path, records_per_page=2))[0]]
    second = [r["source_id"] for r in drained(FileSource(tmp_path, records_per_page=2))[0]]
    assert first == second == [f"n{i:02d}.md" for i in range(9)]


def test_a_page_of_only_unreadable_files_still_advances(tmp_path):
    tree(tmp_path, {f"n{i:02d}.txt": b"\xff\xfe" for i in range(6)})
    tree(tmp_path, {"z-final.md": "reached\n"})
    source = FileSource(tmp_path, records_per_page=2)
    records, skipped, pages = drained(source)
    assert [r["source_id"] for r in records] == ["z-final.md"]
    assert len(skipped) == 6


def test_check_never_reads_file_content(tmp_path, monkeypatch):
    tree(tmp_path, {"note.md": "secret body\n"})
    source = FileSource(tmp_path)
    monkeypatch.setattr("pathlib.Path.read_bytes",
                        lambda self: (_ for _ in ()).throw(AssertionError("content read")))
    assert source.check()["ok"] is True


def test_a_missing_root_reports_instead_of_raising(tmp_path):
    report = FileSource(tmp_path / "absent").check()
    assert report["ok"] is False and "not a directory" in report["reason"]
    assert drained(FileSource(tmp_path / "absent"))[0] == []


# -- adapter to store --------------------------------------------------------

def test_a_full_drain_commits_once_and_replays_as_duplicates(store, tmp_path):
    from hermes_memory.sources.sync import SyncController

    tree(tmp_path, {f"n{i:02d}.md": f"body number {i}\n" for i in range(7)})
    source = FileSource(tmp_path, source="files", records_per_page=3)
    sync = SyncController(store)
    sync.register("files", policy_version="local-only")
    fence = sync.acquire("files", holder="worker", ttl=3600)

    def commit_all(run: int) -> list[dict]:
        outcomes = []
        for index, page in enumerate(_pages(source)):
            outcomes.append(sync.publish(fence, f"page-{index}", page.envelopes,
                                         next_cursor=f"after-{run}-{index}"))
        return outcomes

    first = commit_all(1)
    assert len([record for outcome in first for record in outcome["ids"]]) == 7
    assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 7
    assert sum(outcome["records"] for outcome in first) == 7

    cursor_before = sync.state("files")["cursor"]
    replayed = commit_all(2)
    assert all(outcome["duplicate"] for outcome in replayed), "replay must write nothing"
    assert sync.state("files")["cursor"] == cursor_before, "a replay must not move the cursor"
    assert store.db.execute("SELECT count(*) FROM records").fetchone()[0] == 7
    assert store.db.execute(
        "SELECT count(*) FROM change_journal WHERE change='add'").fetchone()[0] == 7


def _pages(source):
    cursor = None
    while True:
        page = source.read_page(cursor)
        yield page
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


# -- the shared normalization the adapters lean on --------------------------

def test_a_source_that_never_reaches_its_end_is_stopped(tmp_path):
    class NeverEnds(FileSource):
        def read_page(self, cursor):
            return Page(envelopes=(), next_cursor="always-ahead")

    with pytest.raises(EvidenceError, match="did not finish within 3 pages"):
        NeverEnds(tmp_path).read_all(max_pages=3)


def test_an_adapter_that_cannot_name_itself_stamps_nothing(tmp_path):
    with pytest.raises(EvidenceError, match="declare a source"):
        FileSource(tmp_path, source="").envelope(source_id="x", text="y")


def test_bytes_are_decoded_strictly_and_damaged_text_is_refused():
    assert normalize_text(b"Caf\xc3\xa9") == "Café"
    assert normalize_text(b"Caf\xe9") is None
    assert normalize_text("Caf\ufffd") is None
    assert normalize_text({"not": "text"}) is None
    assert normalize_text("   ") is None


@pytest.mark.parametrize("secret", [
    "Bearer yhVn2sQ9kF7pLm3Zx8Rt4Wd6", "sk-proj-Ab9ZmQ2xKd7Lp4Rt8Vn1Yc3Eg6Jk0Mq",
    "ghp_16A0nB3cD4eF5gH6iJ7kL8mN9oP0qR1sT2uV", "xoxb-123456789012-abcdefGHIJKL",
    "AKIA1F3BCDEGHIJKLM2N", "AIzaSyB3cD4eF5gH6iJ7kL8mN9oP0qR1sT2uV3w"])
def test_every_credential_shape_the_suite_can_name_is_taken_out_of_text(secret):
    assert secret not in normalize_text(f"the key is {secret}, keep it safe")
    assert REDACTED in normalize_text(f"the key is {secret}, keep it safe")


def test_a_source_saying_the_grant_is_gone_is_a_different_answer():
    assert revocation("403 insufficient_scope for this account") == {
        "coverage_state": "revoked"}
    assert revocation("invalid_grant") == {"coverage_state": "revoked"}
    assert revocation("connection reset by peer") == {}
    assert revocation(None) == {}


# -- naming a part of a thing ------------------------------------------------

def test_an_id_naming_a_part_reports_the_whole_it_belonged_to():
    assert part_container("att-1#user") == "att-1"
    assert part_container("chat.txt#7") == "chat.txt"
    assert part_container("session-end:e1:c2f5#0") == "session-end:e1:c2f5"
    # One suffix, stripped once: an adapter appends to the id it was handed, so the parent of
    # `thread#42#att-3` is the attachment's own message, not the mailbox `thread` names.
    assert part_container("thread#42#att-3") == "thread#42"


def test_an_id_that_names_a_whole_has_no_container_above_it():
    assert part_container("att-1") is None
    assert part_container("#0") is None, "a leading separator invents no parent"
    assert part_container("") is None
