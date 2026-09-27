"""C13 typed measurement reads: windows, units, gaps and the refusal to average nonsense.

The ingest side already keeps units, quality, device and gaps (`sources/structured.py`),
and this is the other half the plan asks for: a question about a *window* somebody chose
later. Everything here is arithmetic over stored samples, so each test can name the exact
number it expects and a model is never asked to average a heart rate.
"""
from __future__ import annotations

import json

import pytest

from conftest import envelope
from hermes_memory.storage.evidence import EvidenceError
from hermes_memory.storage.measurements import Measurements, summarise

FIRST = "2026-03-01T08:00:00+00:00"


def sample(store, value, *, at=FIRST, measure="weight", unit="kg", device="scale-a",
           quality="good", source="csv", source_id=None, kind="measurement",
           metadata=None):
    """One stored sample, in the shape the structured adapter writes."""
    body = {"measure": measure, "value": value, "unit": unit, "device": device,
            "quality": quality}
    body.update(metadata or {})
    return store.commit(envelope(
        source=source, source_id=source_id or f"{measure}-{value}-{at}", revision="1",
        kind=kind, text=f"{measure} {value}{unit} at {at}", occurred_at=at,
        occurred_precision="second" if at else "unknown",
        metadata=body))["id"]


@pytest.fixture()
def readings(store):
    return Measurements(store)


# -- a window ----------------------------------------------------------------

def test_a_window_reads_the_samples_inside_it_and_says_so(store, readings):
    sample(store, 70.0, at="2026-03-01T08:00:00+00:00")
    sample(store, 71.0, at="2026-03-02T08:00:00+00:00")
    sample(store, 72.0, at="2026-03-03T08:00:00+00:00")
    outside = sample(store, 99.0, at="2026-04-01T08:00:00+00:00")

    report = readings.series("weight", since="2026-03-01T00:00:00+00:00",
                             until="2026-03-31T23:59:59+00:00")

    assert report["samples"] == 3
    assert report["statistics"]["mean"] == 71.0
    assert report["statistics"]["min"] == 70.0 and report["statistics"]["max"] == 72.0
    assert outside not in report["records"]
    assert report["first_at"] == "2026-03-01T08:00:00+00:00"
    assert report["last_at"] == "2026-03-03T08:00:00+00:00"


def test_a_bound_without_a_timezone_is_refused_not_compared_as_text(readings):
    """"2026-03-01" sorts below every timestamp of that day, so a loose bound would
    quietly drop the samples it was written to keep."""
    for bound in ("2026-03-01", "last tuesday"):
        with pytest.raises(EvidenceError, match="timezone-aware"):
            readings.series("weight", since=bound)
        with pytest.raises(EvidenceError, match="timezone-aware"):
            readings.series("weight", until=bound)


def test_a_device_narrows_the_reading_and_leaves_the_other_one_alone(store, readings):
    sample(store, 70.0, device="scale-a")
    sample(store, 80.0, device="body-fat", source_id="other")
    report = readings.series("weight", device="scale-a")
    assert report["samples"] == 1 and report["statistics"]["mean"] == 70.0
    assert readings.series("weight")["samples"] == 2, "no filter means both devices"
    assert readings.series("weight", device="missing")["samples"] == 0


# -- units -------------------------------------------------------------------

def test_a_series_that_changed_units_refuses_to_be_averaged(store, readings):
    """The arithmetic is the danger, not the answer.

    A mean across two units is a number that looks like a measurement, and the only
    honest reading is the one that names both units and computes nothing.
    """
    sample(store, 70.0, unit="kg")
    sample(store, 71.0, unit="kg")
    sample(store, 156.0, unit="lb")

    report = readings.series("weight")

    assert report["statistics"] is None
    assert report["unit_conflict"] == ["", "kg", "lb"] or \
        report["unit_conflict"] == ["kg", "lb"], report["unit_conflict"]
    assert "kg" in report["refused"] and "lb" in report["refused"]
    assert report["units_seen"] == {"kg": 2, "lb": 1}


def test_naming_a_unit_reads_that_unit_and_reports_what_was_left_out(store, readings):
    sample(store, 70.0, unit="kg")
    sample(store, 156.0, unit="lb")
    report = readings.series("weight", unit="kg")
    assert report["statistics"]["mean"] == 70.0
    assert report["units_seen"] == {"kg": 1, "lb": 1}, "the excluded unit stays visible"


def test_a_bare_number_series_is_still_a_series(store, readings):
    """Some sources carry no unit at all; that is one unit, not a conflict."""
    sample(store, 60.0, unit=None)
    sample(store, 80.0, unit=None, source_id="second")
    report = readings.series("weight")
    assert report["statistics"]["mean"] == 70.0
    assert report["units_seen"] == {"": 2}


def test_the_citations_are_of_the_samples_that_were_read(store, readings):
    """A row excluded from the arithmetic is not evidence for the answer.

    Listing every row the query touched would let a reading cite a lb sample as proof
    of a kg mean.
    """
    kept = sample(store, 70.0, unit="kg")
    dropped = sample(store, 156.0, unit="lb", source_id="imperial")
    report = readings.series("weight", unit="kg")
    assert report["records"] == [kept]
    assert dropped not in report["records"]
    assert report["units_seen"] == {"kg": 1, "lb": 1}


def test_one_sided_window_still_names_what_it_could_not_place(store, readings):
    sample(store, 70.0, at="2026-03-01T08:00:00+00:00")
    sample(store, 71.0, at="2026-04-01T08:00:00+00:00", source_id="later")
    sample(store, 72.0, at=None, source_id="undated")

    since_only = readings.series("weight", since="2026-03-15T00:00:00+00:00")
    assert since_only["samples"] == 1
    assert since_only["unplaced_excluded"] == 1

    until_only = readings.series("weight", until="2026-03-15T00:00:00+00:00")
    assert until_only["samples"] == 1
    assert until_only["unplaced_excluded"] == 1, "an open end still drops the undated"


def test_the_excluded_count_is_stated_in_the_terms_that_were_asked(store, readings):
    """A reader who asked about kg is told how many *kg* samples the window dropped."""
    sample(store, 70.0, unit="kg")
    sample(store, 156.0, unit="lb", at=None, source_id="imperial-and-undated")

    assert readings.series("weight", since="2026-01-01T00:00:00+00:00",
                           unit="kg")["unplaced_excluded"] is None
    assert readings.series("weight", since="2026-01-01T00:00:00+00:00",
                           unit="lb")["unplaced_excluded"] == 1


def test_a_sample_that_named_no_unit_is_counted_in_the_terms_that_were_asked(store,
                                                                            readings):
    """A unitless row belongs to the unitless series, not to every series."""
    sample(store, 70.0, unit="kg", at=None, source_id="kg-and-undated")
    sample(store, 71.0, unit=None, at=None, source_id="bare-and-undated")

    assert readings.series("weight", since="2026-01-01T00:00:00+00:00",
                           unit="kg")["unplaced_excluded"] == 1
    assert readings.series("weight", since="2026-01-01T00:00:00+00:00",
                           unit="lb")["unplaced_excluded"] is None
    assert readings.series("weight", since="2026-01-01T00:00:00+00:00",
                           )["unplaced_excluded"] == 2, "no unit named means all of them"


# -- the shape of the answer -------------------------------------------------

def test_the_citation_list_is_bounded_because_a_model_reads_it(store, readings):
    """Twenty-five ids is a citation; thirty is a context window spent on bookkeeping."""
    for index in range(30):
        sample(store, float(index), source_id=f"s{index}",
               at=f"2026-03-01T{index % 24:02d}:00:00+00:00")
    report = readings.series("weight")
    assert report["samples"] == 30
    assert len(report["records"]) == 25
    assert report["records_more"] == 5


def test_a_refused_read_still_cites_only_the_rows_it_read(store, readings):
    """The bound is what the reading is made of, so it has to bite before the answer."""
    for index in range(5):
        sample(store, float(index), source_id=f"s{index}",
               at=f"2026-03-01T{index:02d}:00:00+00:00")
    report = Measurements(store, max_samples=4).series("weight")
    assert report["statistics"] is None
    assert len(report["records"]) == 4, "the sample over the bound is not in the reading"


def test_two_samples_are_enough_to_disagree(store, readings):
    sample(store, 70.0, source_id="one")
    sample(store, 72.0, source_id="two")
    assert readings.series("weight")["statistics"]["stdev"] == pytest.approx(1.414214,
                                                                            abs=1e-5)


def test_a_name_of_any_length_is_not_a_query(readings):
    with pytest.raises(EvidenceError, match="at most 200 characters"):
        readings.series("weight" * 60)
    with pytest.raises(EvidenceError, match="at most 200 characters"):
        readings.series("weight", device="scale" * 60)


def test_the_listing_has_a_bound_too(readings):
    with pytest.raises(EvidenceError, match="limit must be an integer between 1 and 500"):
        readings.available(limit=501)


# -- time it cannot place ----------------------------------------------------

def test_an_unplaced_sample_counts_in_an_open_reading_and_is_named_in_a_window(
        store, readings):
    sample(store, 70.0, at="2026-03-01T08:00:00+00:00")
    sample(store, 74.0, at=None, source_id="undated")

    open_reading = readings.series("weight")
    assert open_reading["samples"] == 2
    assert open_reading["unplaced_times"] == 1
    assert "unplaced_excluded" not in open_reading, "an open window excluded nothing"
    assert open_reading["statistics"]["interval_seconds"] is None, \
        "no cadence can be inferred across a sample with no time"

    bounded = readings.series("weight", since="2026-03-01T00:00:00+00:00",
                              until="2026-03-31T00:00:00+00:00")
    assert bounded["samples"] == 1
    assert bounded["unplaced_excluded"] == 1, "the window did not hide it silently"


def test_a_quiet_stretch_is_reported_as_a_gap_not_smoothed_over(store, readings):
    sample(store, 70.0, at="2026-03-01T08:00:00+00:00", source_id="d1")
    sample(store, 70.5, at="2026-03-02T08:00:00+00:00", source_id="d2")
    sample(store, 71.0, at="2026-03-03T08:00:00+00:00", source_id="d3")
    sample(store, 71.5, at="2026-03-20T08:00:00+00:00", source_id="d4")

    gaps = readings.series("weight")["statistics"]["gaps"]
    assert len(gaps) == 1
    assert gaps[0]["after"] == "2026-03-03T08:00:00+00:00"
    assert gaps[0]["seconds"] == pytest.approx(17 * 86400)


# -- what the reading is made of --------------------------------------------

def test_a_forgotten_sample_stops_counting_in_the_mean(store, readings):
    """The reading is of the live set, not of what was last averaged."""
    kept = sample(store, 70.0, source_id="kept")
    gone = sample(store, 90.0, source_id="gone")
    assert readings.series("weight")["statistics"]["mean"] == 80.0
    store.hide(gone, reason="the owner withdrew it", actor="owner")
    report = readings.series("weight")
    assert report["statistics"]["mean"] == 70.0
    assert kept in report["records"] and gone not in report["records"]


def test_a_sample_that_is_not_a_number_is_reported_rather_than_skipped_quietly(store,
                                                                              readings):
    sample(store, 70.0, source_id="fine")
    sample(store, "not-a-number", source_id="broken")
    report = readings.series("weight")
    assert report["samples"] == 1 and report["unreadable"] == 1
    assert report["statistics"]["mean"] == 70.0


def test_an_empty_reading_says_empty_rather_than_zero(store, readings):
    report = readings.series("sleep")
    assert report["samples"] == 0
    assert report["statistics"]["count"] == 0
    assert report["statistics"]["mean"] is None
    assert "empty reading" in report["note"] and "0" not in report["note"]


def test_a_read_that_would_average_the_whole_archive_refuses(store, readings):
    for index in range(4):
        sample(store, float(index), source_id=f"s{index}")
    subject = Measurements(store, max_samples=3)
    report = subject.series("weight")
    assert report["statistics"] is None
    assert "more than 3 samples" in report["refused"]
    assert "narrow the window" in report["refused"]


def test_an_ingested_series_record_is_never_read_as_a_sample(store, readings):
    """A summary of numbers is not another number to average.

    The structured adapter writes both kinds against the same `measure`, so the read
    has to select samples by kind or it would double-count its own description.
    """
    sample(store, 70.0, source_id="one")
    sample(store, 72.0, source_id="two")
    sample(store, 71.0, kind="measurement_series", source_id="series",
           metadata={"statistics": summarise([70.0, 72.0])})
    report = readings.series("weight")
    assert report["samples"] == 2
    assert report["statistics"]["mean"] == 71.0


def test_the_quality_of_the_samples_travels_with_the_reading(store, readings):
    sample(store, 70.0, quality="good", source_id="q1")
    sample(store, 71.0, quality="estimated", source_id="q2")
    sample(store, 72.0, quality="estimated", source_id="q3")
    assert readings.series("weight")["quality"] == {"good": 1, "estimated": 2}


# -- what can be asked at all ------------------------------------------------

def test_the_listing_names_what_the_store_can_answer(store, readings):
    sample(store, 70.0, measure="weight", unit="kg", source="csv")
    sample(store, 61.0, measure="heart_rate", unit="bpm", device="watch-a",
           source="jsonl", source_id="hr")
    hidden = sample(store, 71.0, measure="weight", unit="kg", source_id="hidden-one")
    store.hide(hidden, reason="withdrawn", actor="owner")

    series = readings.available()

    assert [(item["measure"], item["device"], item["unit"], item["samples"])
            for item in series] == [("heart_rate", "watch-a", "bpm", 1),
                                    ("weight", "scale-a", "kg", 1)]
    assert series[0]["sources"] == ["jsonl"] and series[1]["sources"] == ["csv"]
    assert series[1]["first_at"] == series[1]["last_at"] == FIRST
    assert readings.available()[1]["samples"] == 1, "a hidden sample is not in the count"


def test_a_store_with_no_measurements_answers_with_an_empty_listing(readings):
    assert readings.available() == []


def test_a_bogus_bound_or_name_is_refused(readings):
    with pytest.raises(EvidenceError, match="measure must be nonempty"):
        readings.series("  ")
    with pytest.raises(EvidenceError, match="limit"):
        readings.available(limit=0)
    with pytest.raises(EvidenceError, match="max_samples"):
        Measurements(readings.store, max_samples=0)
