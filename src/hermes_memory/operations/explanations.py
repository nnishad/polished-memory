"""C14 explanations: answer "why was this said, or why was it not" from stored rows.

Two questions used to be unanswerable in the retired installation. "Why did this come
back?" was answered by re-running the query and hoping for the same shape, and "why
did I not hear about it?" was answered by the model that had been shown a prompt it
could not be shown again. Here both answers are assembled from what is already on
disk: the visibility rows, citations, decisions, policy settings and receipts that
were written at the moment they became true. Nothing is re-asked and nothing is
generated, so the answer is the same one an hour later.

Anything that could carry private text — a payload, a goal statement, a prompt packet
— is left out unless the caller asks for it by name, and is still redacted when it is.
The packet id is reported; the packet itself is not.
"""
from __future__ import annotations

import json
from typing import Any

from ..ids import now, timestamp
from ..proactive.outbox import Outbox
from ..proactive.policy import AttentionPolicy
from ..sources.base import redact_secrets
from ..storage.evidence import EvidenceError
from .audit import MAX_DETAIL_CHARS, AuditTrail

__all__ = ["Explanations"]

# Outbox states that mean somebody still owes the artifact something.
UNSETTLED = ("prepared", "leased", "attempted")
ANSWERED = ("confirmed", "suppressed", "expired")


class Explanations:
    """Read-only "why" over one store."""

    def __init__(self, store, *, settings: Any = None, policy: AttentionPolicy | None = None,
                 outbox: Outbox | None = None, audit: AuditTrail | None = None,
                 blobs=None):
        from ..storage.blobs import BlobStore

        self.store = store
        self.db = store.db
        self.owner = getattr(settings, "owner_principal", None)
        self.policy = policy or AttentionPolicy(store, owner_principal=self.owner)
        self.outbox = outbox or Outbox(store, policy=self.policy,
                                       owner_principal=self.owner)
        self.audit = audit or AuditTrail(store)
        self.blobs = blobs or BlobStore(store)

    # -- retrieval -----------------------------------------------------------

    def retrieval(self, record_id: str, *, limit: int = 20) -> dict[str, Any]:
        """Can this be retrieved at all, and what does retrieving it support?

        The answer is a list of gates with what the store said about each, so a "no"
        names the row responsible rather than leaving the caller to guess between
        forgotten, hidden and never indexed.
        """
        row = self._record(record_id)
        visibility = self.db.execute(
            "SELECT * FROM record_visibility WHERE record_id=?", (row["id"],)).fetchone()
        indexed = self.db.execute("SELECT 1 FROM record_fts WHERE id=?",
                                  (row["id"],)).fetchone() is not None
        hidden = bool(visibility and visibility["hidden"])
        forgotten = bool(row["deleted"])
        gates = [
            {"gate": "not forgotten", "met": not forgotten,
             "evidence": None if not forgotten else "the record is deleted in place"},
            {"gate": "visible to retrieval", "met": not hidden,
             "evidence": None if not hidden else redact_secrets(
                 str(visibility["reason"]))[:MAX_DETAIL_CHARS]},
            {"gate": "in the search index", "met": indexed,
             "evidence": None if indexed else "no full-text entry exists for this record"},
        ]
        return {
            "record_id": row["id"],
            "assembled_at": now(),
            "retrievable": all(item["met"] for item in gates),
            "gates": gates,
            "coordinates": {"source": row["source"], "source_id": row["source_id"],
                            "revision": row["revision"], "kind": row["kind"],
                            "occurred_at": row["occurred_at"],
                            "occurred_precision": row["occurred_precision"],
                            "observed_at": row["observed_at"],
                            "ingested_at": row["ingested_at"]},
            "supports": {
                "assertions": [{"id": item["id"], "predicate": item["predicate"],
                                "status": item["status"]} for item in self.db.execute(
                    "SELECT id, predicate, status FROM assertions WHERE record_id=? "
                    "ORDER BY created_at, id", (row["id"],)).fetchall()],
                "artifacts": [{"artifact_id": item["artifact_id"], "kind": item["kind"],
                               "coverage": item["coverage"]} for item in self.db.execute(
                    "SELECT artifact_id, kind, coverage FROM derived_citations "
                    "WHERE record_id=? ORDER BY artifact_id", (row["id"],)).fetchall()],
            },
            "held": [item.as_dict() for item in self.blobs.list_for(row["id"])],
            "history": self.audit.for_object(row["id"], limit=limit),
        }

    # -- notifications -------------------------------------------------------

    def notification(self, artifact_id: str, *, include_private: bool = False) -> dict[str, Any]:
        """Why this was said, or why it is still sitting unsaid.

        The chain is read from the bottom up — artifact, decision, intention, due
        event, goal — because each rung recorded its own reason at the time, and a
        summary written now would be a fresh claim rather than a record of one.
        """
        artifact = self.outbox.get(_text(artifact_id, "artifact id"))
        if artifact is None:
            raise EvidenceError(f"no artifact {artifact_id!r} in this outbox")
        decision = self.db.execute("SELECT * FROM proactive_decisions WHERE id=?",
                                   (artifact.decision_id,)).fetchone()
        intent = self.db.execute("SELECT * FROM decision_intents WHERE id=?",
                                 (decision["intent_id"],)).fetchone() if decision else None
        event = self.db.execute("SELECT * FROM due_events WHERE id=?",
                                (intent["event_id"],)).fetchone() if intent else None
        goal = self.db.execute("SELECT * FROM goals WHERE id=?",
                               (decision["goal_id"],)).fetchone() if decision else None
        try:
            still_valid = self.outbox.revalidate(artifact.id).as_dict()
        except EvidenceError as error:
            still_valid = {"ok": False, "stage": "unknown", "reason": str(error)[:200]}
        out: dict[str, Any] = {
            "artifact_id": artifact.id,
            "assembled_at": now(),
            "answer": _answer(artifact.state, still_valid, artifact.reason),
            "state": artifact.state,
            "kind": artifact.kind,
            "topic": artifact.topic,
            "recipient": artifact.recipient,
            "attempts": int(artifact.attempts),
            "reason": redact_secrets(str(artifact.reason or ""))[:MAX_DETAIL_CHARS] or None,
            "policy_version": artifact.policy_version,
            "created_at": artifact.created_at,
            "delivered_at": artifact.delivered_at,
            "expires_at": artifact.expires_at,
            "payload_digest": artifact.payload_digest,
            "decided": None if decision is None else (
                {"action": decision["action"], "reason": decision["reason"],
                 "policy_version": decision["policy_version"],
                 "shadow": bool(decision["shadow"]),
                 "model_used": bool(decision["model_used"]),
                 "packet_id": decision["packet_id"],
                 "decided_at": decision["decided_at"],
                 "cited_records": _cited(decision["citations"])}),
            "handed_off": None if intent is None else (
                {"kind": intent["kind"], "state": intent["state"],
                 "created_at": intent["created_at"]}),
            "trigger": None if event is None else (
                {"fire_at": event["fire_at"], "reason": event["reason"],
                 "precision": event["precision"], "state": event["state"],
                 "decided_at": event["decided_at"], "decision_kind": event["decision_kind"]}),
            "goal": None if goal is None else (
                {"id": goal["id"], "status": goal["status"], "revision": int(goal["revision"]),
                 "due_at": goal["due_at"], "due_precision": goal["due_precision"],
                 "snoozed_until": goal["snoozed_until"], "created_kind": goal["created_kind"],
                 "confirmed_by": goal["confirmed_by"]}),
            "policy_now": {**self.policy.settings(artifact.topic),
                           **{"attention_today": self.policy.counts(artifact.topic)}},
            # One decision can prepare more than one thing — a note to the owner and a
            # draft for somebody else. "Did I get two messages about this?" is answered by
            # naming the others, not by reasoning about what the decision probably did.
            "from_the_same_decision": [
                {"id": item.id, "kind": item.kind, "state": item.state,
                 "recipient": item.recipient}
                for item in self.outbox.for_decision(artifact.decision_id)
                if item.id != artifact.id],
            "still_sendable": still_valid,
            "history": self.audit.for_object(artifact.id, limit=20),
        }
        if include_private:
            out["payload"] = redact_secrets(str(artifact.payload))[:2000]
            out["evidence_spans"] = [redact_secrets(str(span))[:120]
                                     for span in (artifact.evidence or ())][:20]
            if goal is not None:
                out["goal_title"] = redact_secrets(str(goal["title"]))[:MAX_DETAIL_CHARS]
        return out

    def suppressed(self, *, topic: str = "general", at: str | None = None,
                   limit: int = 20) -> dict[str, Any]:
        """Every recent reason the owner was not interrupted, with the policy behind it.

        This is the question the old installation could not answer at all: a message
        that was decided against leaves no trace in a log, so "why didn't you tell me"
        had no evidence to be answered from. *at* names the moment to judge the quiet
        hours against, because the question is always about a past one.
        """
        _text(topic, "topic")
        if not 1 <= int(limit) <= 200:
            raise EvidenceError("limit must be between 1 and 200 rows")
        rows = self.db.execute(
            "SELECT d.id, d.action, d.reason, d.shadow, d.model_used, d.decided_at, "
            "d.goal_id, d.revision, "
            "(SELECT count(*) FROM outbox o WHERE o.decision_id=d.id) AS artifacts "
            "FROM proactive_decisions d WHERE d.topic=? ORDER BY d.decided_at DESC, d.id "
            "LIMIT ?", (topic, int(limit))).fetchall()
        moment = now() if at is None else timestamp(at)
        settings = self.policy.settings(topic)
        return {
            "topic": topic,
            "assembled_at": now(),
            "asked_about": moment,
            "policy": settings,
            "attention_today": self.policy.counts(topic, at=moment),
            "quiet_until": self.policy.next_waking(topic, moment),
            # An empty list would otherwise read as "nothing was held back", when the
            # likelier answer is that this topic has no policy at all.
            "topics_under_policy": [str(item["topic"]) for item in self.policy.topics()],
            "decided_against": [{
                "decision_id": row["id"], "action": row["action"],
                "reason": redact_secrets(str(row["reason"]))[:MAX_DETAIL_CHARS],
                "shadow": bool(row["shadow"]), "model_used": bool(row["model_used"]),
                "goal_id": row["goal_id"], "revision": int(row["revision"]),
                "artifact_prepared": bool(row["artifacts"]), "decided_at": row["decided_at"],
            } for row in rows],
        }

    # -- what the archive learned --------------------------------------------

    def lesson(self, lesson_id: str, *, include_private: bool = False) -> dict[str, Any]:
        """Every version of a habit, what it cites, what was scored against it.

        Nothing here asks the model whether the rule was a good one. The support and the
        objections were filed by somebody who could know, the runs were scored by an
        authorized evaluator, and the evidence either is on disk or is not.
        """
        from ..learning.evaluation import EvaluationLedger
        from ..ids import record_pk
        from ..learning.lessons import LessonStore
        from ..learning.outcomes import OutcomeLog

        wanted = _text(lesson_id, "lesson id")
        log = OutcomeLog(self.store, owner_principal=self.owner)
        habits = LessonStore(self.store, outcomes=log, owner_principal=self.owner)
        if habits.get(wanted) is None:
            raise EvidenceError(f"no lesson {wanted!r} in this store")
        ledger = EvaluationLedger.reading(self.store, owner_principal=self.owner)
        review = {str(item["id"]): item for item in habits.needs_review(limit=200)}
        versions = []
        for item in habits.versions(wanted):
            key = f"{item.id}@{item.version}"
            told = log.for_subject("lesson", key)
            run = ledger.latest(item.id, item.version)
            support = sum(1 for entry in told if entry.counts_as_support
                          and entry.valence == "success")
            against = sum(1 for entry in told if entry.counts_as_support
                          and entry.valence == "failure")
            entry = {
                "lesson": key, "version": int(item.version), "status": item.status,
                "proposed_by": item.created_by, "proposed_kind": item.created_kind,
                "applicability": dict(item.applicability or {}),
                "prerequisites": list(item.prerequisites or ()),
                "exceptions": list(item.exceptions or ()),
                "contrary_cases": list(item.contrary_cases or ()),
                "evidence": [{"citation": str(span), "record": record_pk(span),
                              "readable": self.store.live_and_visible(record_pk(span))}
                             for span in (item.evidence or ())],
                "support": support, "against": against,
                "outcomes": [record.as_dict() for record in told][-20:],
                "evaluation": None if run is None else run.as_dict(),
                "awaiting_owner_review": key in review,
                "review_reason": (review.get(key) or {}).get("reason"),
            }
            if include_private:
                entry["text"] = redact_secrets(str(item.text))[:MAX_DETAIL_CHARS]
            versions.append(entry)
        return {"lesson_id": wanted, "assembled_at": now(), "versions": versions,
                "current": habits.get(wanted).version,
                "note": "a candidate teaches nothing; only an owner or a passed run makes "
                        "a lesson active, and both are named above"}

    def summary(self, summary_id: str, *, include_private: bool = False) -> dict[str, Any]:
        """Why this reading is on the table, or why it is not being shown.

        The citations come back with their live state, because the reason a summary is
        withheld is always one of these rows having been forgotten, corrected or hidden —
        and an owner reading the answer needs to see which.
        """
        from ..knowledge.summaries import SummaryStore
        from ..storage.lineage import Lineage

        wanted = _text(summary_id, "summary id")
        habits = SummaryStore(self.store, owner_principal=self.owner)
        found = habits.read(wanted)
        if not found.get("found"):
            raise EvidenceError(f"no summary {wanted!r} in this store")
        found["assembled_at"] = now()
        found["cited"] = [
            {"record": str(row["record_id"]),
             "state": "erased" if row["deleted"] else "hidden" if row["hidden"]
             else "visible",
             "coverage": str(row["coverage"])}
            for row in Lineage(self.store).citations_of(wanted)]
        # A withheld reading does not report its own scope, so the promise is matched
        # against the row rather than against whatever the answer chose to show.
        stored = habits.get(wanted)
        found["refresh_owed"] = [
            item for item in habits.pending_refreshes(limit=50)
            if stored is not None and item["scope"] == stored.scope]
        if not include_private:
            # The body is a re-reading of private evidence; it travels only when named.
            found.pop("body", None)
        return found

    # -- prospective memory --------------------------------------------------

    def goal(self, goal_id: str, *, include_private: bool = False) -> dict[str, Any]:
        """Why this is on the list, and what its clocks have done since."""
        row = self.db.execute("SELECT * FROM goals WHERE id=?",
                              (_text(goal_id, "goal id"),)).fetchone()
        if row is None:
            raise EvidenceError(f"no goal {goal_id!r} in this store")
        events = self.db.execute(
            "SELECT id, revision, fire_at, reason, precision, state, claimed_by, decided_at, "
            "decision_kind FROM due_events WHERE goal_id=? ORDER BY fire_at, id",
            (row["id"],)).fetchall()
        predicates = self.db.execute(
            "SELECT id, revision, kind, state, evidence_record_id, evaluated_at "
            "FROM goal_predicates WHERE goal_id=? ORDER BY revision, kind",
            (row["id"],)).fetchall()
        out = {
            "goal_id": row["id"],
            "assembled_at": now(),
            "status": row["status"],
            "revision": int(row["revision"]),
            "created_by": row["created_by"],
            "created_kind": row["created_kind"],
            "confirmed_by": row["confirmed_by"],
            "due": {"due_at": row["due_at"], "precision": row["due_precision"],
                    "timezone": row["timezone"], "snoozed_until": row["snoozed_until"],
                    "expires_at": row["expires_at"]},
            "why_now": _why_now(row, [dict(item) for item in events]),
            "due_events": [dict(item) for item in events],
            "predicates": [dict(item) for item in predicates],
            "history": self.audit.for_object(row["id"], limit=20),
        }
        if include_private:
            out["title"] = redact_secrets(str(row["title"]))[:MAX_DETAIL_CHARS]
            out["statement"] = redact_secrets(str(row["statement"]))[:2000]
        return out

    # -- helpers -------------------------------------------------------------

    def _record(self, record_id: str):
        row = self.db.execute("SELECT * FROM records WHERE id=?",
                              (_text(record_id, "record id"),)).fetchone()
        if row is None:
            raise EvidenceError(f"no record {record_id!r} in this store")
        return row


def _answer(state: str, still_valid: dict[str, Any], reason: str | None = None) -> str:
    """The one-line answer, from the two records that already exist about it.

    The artifact's own reason wins where it has one: it was written by whoever
    stopped or lost the send, at the moment they did it. The revalidation verdict is a
    fallback for the states that carry no reason of their own.
    """
    said = redact_secrets(str(reason))[:MAX_DETAIL_CHARS] if reason else None
    if state == "confirmed":
        return "delivered, and the host said so"
    if state == "accepted_unverified":
        return "the host took it and has not yet proved it reached the owner"
    if state == "uncertain":
        return ("a send was attempted and its outcome is unknown: "
                + (said or "nothing was assumed"))
    if state == "suppressed":
        return "withheld before sending: " + (said or str(still_valid.get("reason"))
                                              or "no reason was recorded")
    if state == "expired":
        return "it was never claimed, and its own expiry passed"
    if state in UNSETTLED:
        return ("prepared and waiting for the host transport"
                + ("" if still_valid.get("ok") else
                   f"; it can no longer be sent: {still_valid.get('reason')}"))
    return f"the artifact is {state}"


def _why_now(goal: Any, events: list[dict[str, Any]]) -> str:
    """Why this goal is or is not worth announcing at this moment, from its own rows."""
    if goal["status"] != "active":
        return f"it is {goal['status']}, so nothing about it is due"
    if goal["snoozed_until"]:
        return f"it was put off until {goal['snoozed_until']} by the owner"
    open_events = [item for item in events if item["state"] in ("pending", "claimed")]
    if not open_events:
        return "its due events have all been decided about"
    return (f"{len(open_events)} due event(s) outstanding, the next at "
            f"{open_events[0]['fire_at']} ({open_events[0]['reason']})")


def _cited(raw: Any) -> list[str]:
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item)[:120] for item in parsed[:50]]


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{label} must be nonempty text")
    return value.strip()[:200]
