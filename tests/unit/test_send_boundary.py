"""F1.a: the pre-send boundary checks personal-memory claims before delivery.

The detector is deterministic and cheap; evidence is the packet the turn injected
(the agent may assert what it was given, nothing more); degradation is reported
rather than silent; and every non-trivial decision leaves a receipt row.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_memory.config import load_settings
from hermes_memory.processing.boundary import SendBoundary, memory_sentences


def packet(*pairs):
    items = [SimpleNamespace(id=rid, text=text) for rid, text in pairs]
    return SimpleNamespace(items=items, packet_id="ctx_test")


def boundary(store, mode, monkeypatch=None, verify=None):
    settings = load_settings()
    if verify is not None and monkeypatch is not None:
        monkeypatch.setattr("hermes_memory.processing.boundary.verify_memory_claims",
                            verify)
    return SendBoundary(settings=settings, store=store, mode=mode)


def test_the_detector_finds_personal_claims_and_ignores_small_talk():
    draft = ("You are allergic to peanuts, so I avoided them. "
             "The weather looks nice today. "
             "I recall you said the Jaipur ticket is not booked yet.")
    found = memory_sentences(draft, ["Nisha is allergic to peanuts and avoids them."])
    assert any("allergic" in sentence for sentence in found)
    assert any("Jaipur" in sentence for sentence in found)
    assert not any("weather" in sentence for sentence in found)


def test_off_mode_skips_without_a_receipt(store):
    verdict = boundary(store, "off").check("You are allergic to peanuts.",
                                           packet=packet(("r1", "peanuts")))
    assert verdict.disposition == "pass" and verdict.mode == "off"
    assert verdict.receipt == {}
    rows = store.db.execute("SELECT count(*) AS n FROM boundary_receipts").fetchone()
    assert rows["n"] == 0


def test_a_non_memory_draft_passes_without_a_model_or_a_receipt(store):
    verdict = boundary(store, "enforce").check("The weather looks nice today.")
    assert verdict.disposition == "pass"
    assert verdict.detector["skipped_reason"] == "no_memory_claims"
    assert verdict.receipt == {}


def test_asserting_beyond_the_injected_packet_is_insufficient_without_a_model(store):
    verdict = boundary(store, "enforce").check(
        "You are allergic to peanuts and must avoid them.", packet=None)
    assert verdict.disposition == "revise"
    assert verdict.action["mode"] == "downgrade"
    assert verdict.action["unverified_sentences"] == [
        "You are allergic to peanuts and must avoid them."]
    rows = store.db.execute("SELECT disposition, mode FROM boundary_receipts").fetchall()
    assert [(r["disposition"], r["mode"]) for r in rows] == [("revise", "enforce")]


def test_warn_records_and_sends_rather_than_holding(store):
    verdict = boundary(store, "warn").check(
        "You are allergic to peanuts and must avoid them.", packet=None)
    assert verdict.disposition == "revise"
    assert verdict.action["mode"] == "none"


def test_a_supported_claim_passes_in_enforce(store, monkeypatch):
    def verify(settings, store_, claims, account_id=None):
        return {"ok": True, "all_supported": True, "text": claims[0]["text"],
                "verification": {"rejected": [], "deterministic_rejections": [],
                                 "claims_digest": "dig"}}
    verdict = boundary(store, "enforce", monkeypatch, verify).check(
        "You are allergic to peanuts.", packet=packet(("r1", "Nisha is allergic to peanuts.")))
    assert verdict.disposition == "pass"
    assert verdict.action == {"mode": "none"}
    assert verdict.claims[0]["label"] == "supported"


def test_a_contradicted_claim_is_held_in_enforce(store, monkeypatch):
    def verify(settings, store_, claims, account_id=None):
        return {"ok": True, "all_supported": False, "text": "",
                "verification": {"rejected": [{"claim_index": 0, "label": "contradicted"}],
                                 "deterministic_rejections": [], "claims_digest": "dig"}}
    verdict = boundary(store, "enforce", monkeypatch, verify).check(
        "You are not allergic to peanuts.", packet=packet(("r1", "allergic to peanuts")))
    assert verdict.disposition == "block"
    assert verdict.action["mode"] == "hold"


def test_an_unreachable_verifier_holds_in_enforce_and_warns_in_warn(store):
    # Inference is disabled in the test environment, so verification raises; the
    # boundary must not pass silently in either mode.
    held = boundary(store, "enforce").check(
        "You are allergic to peanuts.", packet=packet(("r1", "allergic to peanuts")))
    assert held.disposition == "block" and held.action["mode"] == "hold"
    assert held.detector["degraded"] is True
    warned = boundary(store, "warn").check(
        "You are allergic to peanuts.", packet=packet(("r1", "allergic to peanuts")))
    assert warned.disposition == "revise"
    assert warned.action["notes"][0].startswith("boundary degraded to warn")


def test_an_unknown_mode_is_refused(store):
    with pytest.raises(Exception, match="unknown boundary mode"):
        SendBoundary(settings=load_settings(), store=store, mode="yolo")
