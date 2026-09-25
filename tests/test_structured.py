"""C2 structured fixtures: units, quality, devices, intervals and gaps."""
from __future__ import annotations

import json
import pathlib

from datetime import datetime, timedelta, timezone

import pytest
from dataclasses import replace

from hermes_memory.sources.structured import StructuredSource, summarise

WEIGHT = (
    "timestamp,kg,device,quality\n"
    "2026-09-01T07:00:00+00:00,71.4,scale-a,good\n"
    "2026-09-02T07:00:00+00:00,71.9,scale-a,good\n"
    "2026-09-03T07:00:00+00:00,,scale-a,low battery\n"
    "2026-09-10T07:00:00+00:00,70.2,scale-a,good\n")
HEART = [
    {"time": "2026-09-05T06:00:00Z", "value": 58, "unit": "bpm", "device": "band"},
    {"time": "2026-09-05T06:05:00Z", "value": 61, "unit": "bpm", "device": "band"},
    {"time": "2026-09-05T06:10:00Z", "value": 63, "unit": "bpm", "device": "band"},
    {"time": "2026-09-05T07:30:00Z", "value": 74, "unit": "bpm", "device": "band"}]


def files(root: pathlib.Path, mapping: dict[str, str]) -> pathlib.Path:
    for name, text in mapping.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def read(root, **kwargs):
    return StructuredSource(root, **kwargs).read_page(None)


def series(page):
    return [item for item in page.envelopes if item["kind"] == "measurement_series"]


# -- what a measurement keeps -------------------------------------------------

def test_the_unit_and_device_are_kept_even_when_the_column_names_them(tmp_path):
    root = files(tmp_path, {"weight.csv": WEIGHT})
    page = read(root, per_sample=True)
    meta = page.envelopes[0]["metadata"]

    assert meta["unit"] == "kg" and meta["device"] == "scale-a"
    assert meta["measure"] == "weight" and meta["quality"] == "good"
    assert meta["value"] == pytest.approx(71.4)


def test_a_row_with_no_number_is_a_gap_naming_the_row(tmp_path):
    root = files(tmp_path, {"weight.csv": WEIGHT})
    page = read(root, per_sample=True)

    assert [gap.ref for gap in page.skipped] == ["weight.csv#2"]
    assert "no measurement value" in page.skipped[0].reason


def test_two_numeric_columns_is_a_question_not_a_guess(tmp_path):
    root = files(tmp_path, {"body.csv":
                            "timestamp,kg,%fat,device\n2026-09-01T07:00:00+00:00,71.4,18.2,scale-a\n"})
    page = read(root, per_sample=True)

    assert page.envelopes == ()
    assert "no measurement value" in page.skipped[0].reason


def test_an_explicit_unit_column_beats_the_text(tmp_path):
    root = files(tmp_path, {"hr.jsonl": "\n".join(json.dumps(item) for item in HEART)})
    page = read(root, per_sample=True)

    assert page.envelopes[0]["metadata"]["unit"] == "bpm"
    assert page.envelopes[0]["metadata"]["device"] == "band"


def test_a_time_that_cannot_be_placed_is_kept_as_an_unplaced_value(tmp_path):
    rows = HEART + [{"time": "last tuesday", "value": 999, "unit": "bpm", "device": "band"}]
    root = files(tmp_path, {"hr.jsonl": "\n".join(json.dumps(item) for item in rows)})
    grouped = series(read(root))[0]

    assert grouped["metadata"]["samples"] == 5
    assert grouped["metadata"]["unplaced_times"] == 1
    assert grouped["occurred_at"] == "2026-09-05T06:00:00+00:00"


# -- series and statistics ----------------------------------------------------

def test_a_series_is_one_record_per_device_measure_unit(tmp_path):
    root = files(tmp_path, {
        "hr.jsonl": "\n".join(json.dumps(item) for item in HEART),
        "hr2.jsonl": "\n".join(json.dumps({**item, "device": "other band"})
                               for item in HEART[:2])})
    grouped = series(read(root))

    assert len(grouped) == 2
    assert {item["metadata"]["device"] for item in grouped} == {"band", "other band"}


def test_the_statistics_are_reproducible_from_the_rows(tmp_path):
    root = files(tmp_path, {"hr.jsonl": "\n".join(json.dumps(item) for item in HEART)})
    stats = series(read(root))[0]["metadata"]["statistics"]

    assert stats == {"count": 4, "min": 58.0, "max": 74.0, "mean": pytest.approx(64.0),
                     "median": pytest.approx(62.0), "stdev": pytest.approx(6.9761, abs=0.001),
                     "interval_seconds": 300.0, "gaps": [
                         {"after": "2026-09-05T06:10:00+00:00", "seconds": 4800.0}]}


def test_a_quiet_stretch_is_reported_as_a_gap(tmp_path):
    root = files(tmp_path, {"hr.jsonl": "\n".join(json.dumps(item) for item in HEART)})
    gaps = series(read(root))[0]["metadata"]["gaps"]

    assert len(gaps) == 1 and gaps[0]["seconds"] == 4800.0


def test_quality_counts_survive_grouping(tmp_path):
    root = files(tmp_path, {"weight.csv": WEIGHT})
    grouped = series(read(root))[0]

    assert grouped["metadata"]["quality"] == {"good": 3}


def test_summarise_declines_to_invent_numbers():
    assert summarise([])["count"] == 0
    one = summarise([2.0])
    assert one["mean"] == 2.0 and one["stdev"] is None and one["gaps"] == []


def test_per_sample_mode_emits_the_rows_themselves(tmp_path):
    root = files(tmp_path, {"hr.jsonl": "\n".join(json.dumps(item) for item in HEART)})
    page = read(root, per_sample=True)

    assert len(page.envelopes) == 4
    assert {item["kind"] for item in page.envelopes} == {"measurement"}


# -- formats and failures -----------------------------------------------------

def test_a_wrapped_json_object_of_samples_is_read(tmp_path):
    root = files(tmp_path, {"hr.json": json.dumps({"samples": HEART})})
    assert len(read(root, per_sample=True).envelopes) == 4


def test_a_tsv_is_read_with_its_own_delimiter(tmp_path):
    root = files(tmp_path, {"hr.tsv":
                            "time\tvalue\tunit\n2026-09-05T06:00:00Z\t58\tbpm\n"})
    item = read(root, per_sample=True).envelopes[0]

    assert item["metadata"]["value"] == 58.0 and item["metadata"]["unit"] == "bpm"


def test_a_malformed_fixture_is_a_gap(tmp_path):
    root = files(tmp_path, {"broken.csv": "timestamp,kg\n,,"})
    page = read(root)

    assert page.envelopes == () and page.skipped


def test_non_utf8_is_reported_not_replaced(tmp_path):
    path = tmp_path / "hr.jsonl"
    path.write_bytes(b'{"time": "2026-09-05T06:00:00Z", "value": "\xff\xfe"}\n')
    page = read(tmp_path)

    assert page.envelopes == () and "UTF-8" in page.skipped[0].reason


def test_a_symlinked_fixture_is_not_read(tmp_path):
    outside = tmp_path / "other.csv"
    outside.write_text("time,value\n2026-09-05T06:00:00Z,58\n", encoding="utf-8")
    files(tmp_path, {"hr.jsonl": json.dumps(HEART[0])})
    (tmp_path / "linked.csv").symlink_to(outside)

    assert read(tmp_path).next_cursor in (None, "hr.jsonl")


def test_pages_are_bounded_by_series(tmp_path):
    root = files(tmp_path, {f"s{i}.csv": "time,ml\n2026-09-05T06:00:00Z,10\n"
                            for i in range(6)})
    adapter = StructuredSource(root, records_per_page=2)

    first = adapter.read_page(None)
    second = adapter.read_page(first.next_cursor)

    assert len(first.envelopes) == 2 and len(second.envelopes) == 2


def test_the_check_reports_the_grouping_it_will_use(tmp_path):
    root = files(tmp_path, {"hr.jsonl": json.dumps(HEART[0])})

    assert StructuredSource(root).check()["grouping"] == "series"
    assert StructuredSource(root, per_sample=True).check()["grouping"] == "sample"


def test_envelopes_are_accepted_by_the_canonical_commit(tmp_path, store):
    root = files(tmp_path, {"hr.jsonl": "\n".join(json.dumps(item) for item in HEART)})
    item = read(root, per_sample=True).envelopes[0]

    record = store.commit(item)
    stored = store.get(record["id"])
    assert stored.metadata["unit"] == "bpm"
    assert "58" in stored.text


def test_a_value_that_is_not_a_number_is_a_gap_rather_than_a_zero(tmp_path):
    root = files(tmp_path, {"temp.csv": "timestamp,value,device\n"
                                      "2026-09-01T07:00:00+00:00,n/a,probe\n"
                                      "2026-09-01T08:00:00+00:00,18.4,probe\n"})
    page = StructuredSource(root, per_sample=True).read_page(None)
    assert [item["metadata"]["value"] for item in page.envelopes] == [18.4]
    assert "is not a number" in page.skipped[0].reason


def test_a_sample_from_no_known_device_says_so_rather_than_inventing_one(tmp_path):
    root = files(tmp_path, {"temp.csv": "timestamp,value\n2026-09-01T07:00:00+00:00,18.4\n"})
    series = StructuredSource(root).read_page(None).envelopes[0]
    assert series["metadata"]["device"] == "unattributed"
    assert "unattributed" in series["source_id"]


def test_a_sample_that_reported_no_quality_is_not_reported_as_good(tmp_path):
    root = files(tmp_path, {"temp.csv": "timestamp,value\n2026-09-01T07:00:00+00:00,18.4\n"})
    series = StructuredSource(root).read_page(None).envelopes[0]
    assert series["metadata"]["quality"] == {"unknown": 1}


def test_a_unit_change_mid_series_starts_a_second_series_rather_than_continuing(tmp_path):
    root = files(tmp_path, {"temp.csv": "timestamp,value,unit\n"
                                        "2026-09-01T07:00:00+00:00,18.4,C\n"
                                        "2026-09-01T08:00:00+00:00,65.1,F\n"})
    page = StructuredSource(root).read_page(None)
    units = sorted(item["metadata"]["unit"] for item in page.envelopes)
    assert units == ["C", "F"]
    assert [item["metadata"]["samples"] for item in sorted(page.envelopes,
                                                           key=lambda row: row["metadata"]["unit"])] \
        == [1, 1]


def test_a_long_quiet_stretch_is_capped_rather_than_endless(tmp_path):
    # 101 steps of a minute and 99 of a thousand: most of them are gaps, and the
    # list has to stop somewhere or a series description outweighs its samples.
    moments = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    for index in range(200):
        moments.append(moments[-1] + timedelta(seconds=60 if index < 101 else 60_000))
    rows = ["timestamp,value,unit"] + [moment.isoformat() + ",20.0,C" for moment in moments]
    root = files(tmp_path, {"long.csv": "\n".join(rows) + "\n"})
    series = StructuredSource(root).read_page(None).envelopes[0]
    statistics_of = series["metadata"]["statistics"]
    assert statistics_of["count"] == 201
    assert len(statistics_of["gaps"]) == 50
    assert statistics_of["interval_seconds"] == 60


def test_a_log_written_out_of_order_does_not_count_a_step_backwards(tmp_path):
    root = files(tmp_path, {"temp.csv": "timestamp,value,device\n"
                                        "2026-09-01T09:00:00+00:00,20.0,probe\n"
                                        "2026-09-01T07:00:00+00:00,19.0,probe\n"
                                        "2026-09-01T08:00:00+00:00,19.5,probe\n"})
    series = StructuredSource(root).read_page(None).envelopes[0]
    # One forward step of an hour survives out of three written pairs.
    assert series["metadata"]["interval_seconds"] == 3600.0


def test_a_fixture_too_big_for_one_page_is_reported_rather_than_skipped(tmp_path):
    root = files(tmp_path, {"big.csv": "timestamp,value\n"
                                       + "2026-09-01T07:00:00+00:00,20.0\n" * 3000,
                            "small.csv": "timestamp,value\n2026-09-01T07:00:00+00:00,20.0\n"})
    subject = StructuredSource(root)
    subject.capabilities = replace(subject.capabilities, max_bytes_per_page=4000)
    page = subject.read_page(None)
    assert [gap.ref for gap in page.skipped if gap.ref.endswith(".csv")] == ["big.csv"]
    assert "byte per-page bound" in page.skipped[0].reason
    assert subject.check()["candidates"] == 2
