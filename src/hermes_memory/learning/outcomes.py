"""C11 outcomes: who reported what, and what it is actually evidence of.

Three kinds of statement look alike in a log and mean completely different things.
A host receipt says the world changed and can be checked against the artifact the
host acted on. An owner report says the owner experienced an outcome. An assistant
claim says somebody *said* it worked — which is a fact about a message, and the
reason this file exists: the retired system let a success claim feed learning, so a
confident assistant could train the archive into repeating its own mistakes.

Nothing here stores a verdict about a lesson. Support is counted from these rows on
read, so a retracted receipt takes its weight off immediately.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..ids import digest, now
from ..storage.evidence import EvidenceError

__all__ = ["OutcomeLog", "Outcome", "KINDS", "VALENCES", "CHECKABLE", "SUBJECTS"]

KINDS = ("host_receipt", "owner_report", "assistant_claim")
VALENCES = ("success", "failure", "unknown")
# Only these may ever raise a lesson's support. An assistant claim is deliberately
# absent: it is recorded, retrieved and never counted.
CHECKABLE = ("host_receipt", "owner_report")
SUBJECTS = ("lesson", "goal", "artifact", "task")


@dataclass(frozen=True)
class Outcome:
    id: str
    subject_kind: str
    subject_id: str
    kind: str
    valence: str
    note: str
    evidence: tuple[str, ...]
    actor: str
    recorded_at: str

    @classmethod
    def from_row(cls, row) -> "Outcome":
        try:
            evidence = tuple(json.loads(row["evidence"] or "[]"))
        except (TypeError, json.JSONDecodeError):
            evidence = ()
        return cls(id=row["id"], subject_kind=row["subject_kind"],
                   subject_id=row["subject_id"], kind=row["kind"], valence=row["valence"],
                   note=row["note"], evidence=evidence, actor=row["actor"],
                   recorded_at=row["recorded_at"])

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "subject": f"{self.subject_kind}:{self.subject_id}",
                "kind": self.kind, "valence": self.valence, "note": self.note,
                "evidence": list(self.evidence), "actor": self.actor,
                "recorded_at": self.recorded_at, "counts": self.counts_as_support}

    @property
    def counts_as_support(self) -> bool:
        return self.kind in CHECKABLE


class OutcomeLog:
    def __init__(self, store, *, owner_principal: str | None = None, outbox=None,
                 identities=None):
        self.store = store
        self.db = store.db
        self.owner_principal = owner_principal
        self.outbox = outbox
        self.identities = identities

    def record(self, *, subject_kind: str, subject_id: str, kind: str, valence: str,
               note: str, actor: str, evidence: Iterable[str] = (),
               db=None) -> dict[str, Any]:
        """File one report about one subject, and say what it can be used for."""
        if subject_kind not in SUBJECTS:
            raise EvidenceError(f"unknown outcome subject {subject_kind!r}")
        if kind not in KINDS:
            raise EvidenceError(f"unknown outcome kind {kind!r}; expected one of {KINDS}")
        if valence not in VALENCES:
            raise EvidenceError(f"unknown valence {valence!r}")
        subject = _text(subject_id, "subject id", 120)
        teller = _text(actor, "actor", 120)
        words = _text(note, "note", 600)
        spans = [str(item) for item in evidence][:24]
        self._check_authority(kind=kind, actor=teller, valence=valence, evidence=spans,
                              subject_kind=subject_kind, subject_id=subject)
        # One delivery is one observation, even if retold with a new note/actor.
        identity = ([subject_kind, subject, kind, sorted(set(spans))] if kind == "host_receipt"
                    else [subject_kind, subject, kind, valence, words, teller])
        outcome_id = "outc_" + digest(identity)[:32]
        connection = db if db is not None else self.db
        if db is not None and not db.in_transaction:
            raise EvidenceError("recording an outcome needs an ambient transaction")
        cursor = connection.execute(
            "INSERT OR IGNORE INTO outcomes(id, subject_kind, subject_id, kind, valence, "
            "note, evidence, actor, recorded_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (outcome_id, subject_kind, subject, kind, valence, words,
             json.dumps(spans, ensure_ascii=False)[:4000], teller, now()))
        recorded = int(cursor.rowcount or 0) > 0
        if recorded:
            self.store._audit("outcome_recorded", outcome_id, {
                "subject": f"{subject_kind}:{subject}", "kind": kind,
                "valence": valence, "actor": teller,
                "counts_as_support": kind in CHECKABLE})
        return {"id": outcome_id, "recorded": recorded,
                "counts_as_support": kind in CHECKABLE,
                "meaning": ("this is evidence about the world" if kind in CHECKABLE
                            else "this is a statement about a message, and is never "
                                 "counted as support")}

    def for_subject(self, subject_kind: str, subject_id: str, *,
                    kind: str | None = None) -> list[Outcome]:
        clause = " AND kind=?" if kind else ""
        rows = self.db.execute(
            f"SELECT * FROM outcomes WHERE subject_kind=? AND subject_id=?{clause} "
            "ORDER BY recorded_at, id",
            [subject_kind, subject_id] + ([kind] if kind else [])).fetchall()
        return [Outcome.from_row(row) for row in rows]

    def tally(self, subject_kind: str, subject_id: str) -> dict[str, Any]:
        """What has actually happened to this subject, weighted by what can be checked."""
        rows = self.db.execute(
            "SELECT kind, valence, count(*) AS n FROM outcomes WHERE subject_kind=? AND "
            "subject_id=? GROUP BY kind, valence", (subject_kind, subject_id)).fetchall()
        tally = {f"{row['kind']}:{row['valence']}": int(row["n"]) for row in rows}
        support = sum(int(row["n"]) for row in rows
                      if row["kind"] in CHECKABLE and row["valence"] == "success")
        against = sum(int(row["n"]) for row in rows
                      if row["kind"] in CHECKABLE and row["valence"] == "failure")
        claimed = sum(int(row["n"]) for row in rows if row["kind"] == "assistant_claim")
        return {"support": support, "against": against, "claimed": claimed,
                "detail": tally,
                "net": support - against,
                "note": ("assistant claims are excluded from support and against; "
                         f"{claimed} of them are on file")}

    # -- authority -----------------------------------------------------------

    def _check_authority(self, *, kind: str, actor: str, valence: str,
                         evidence: Sequence[str], subject_kind: str, subject_id: str) -> None:
        if kind == "host_receipt":
            if not evidence:
                raise EvidenceError("a host receipt requires nonempty delivery evidence")
            if self.outbox is None:
                raise EvidenceError(
                    "a host receipt needs the outbox to check it against: without a "
                    "delivery record to point at, this is a claim wearing a receipt's "
                    "clothes")
            for span in evidence:
                artifact = self.outbox.get(_artifact_id(span))
                if artifact is None:
                    raise EvidenceError(f"the receipt names {span!r}, which is not an "
                                        "artifact this store ever prepared")
                if artifact.state not in ("confirmed", "accepted_unverified"):
                    raise EvidenceError(
                        f"{artifact.id} is {artifact.state}, so nothing reached anybody; "
                        "a receipt has to be written against a send that happened")
                if subject_kind != "artifact" or subject_id != artifact.id or valence != "success":
                    raise EvidenceError("a delivery receipt proves only that exact artifact's delivery; "
                                        "use owner reports/evaluation for task, goal or lesson outcomes")
        elif kind == "owner_report":
            if self.owner_principal is None or actor != self.owner_principal:
                raise EvidenceError(
                    "only the owner principal can report what the owner experienced; an "
                    "agent speaking for them is how a wish becomes evidence")
        elif kind == "assistant_claim" and valence == "success":
            # Allowed, and never counted. Recorded here so the reason survives into
            # the audit trail rather than living in a comment.
            pass

    def retract(self, outcome_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        """Take a report off the record; anything leaning on it re-resolves on read."""
        if self.owner_principal is None or actor != self.owner_principal:
            raise EvidenceError("only the owner may withdraw a report from the record")
        row = self.db.execute("SELECT * FROM outcomes WHERE id=?", (outcome_id,)).fetchone()
        if row is None:
            raise EvidenceError(f"unknown outcome {outcome_id!r}")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("DELETE FROM outcomes WHERE id=?", (outcome_id,))
            self.store._audit("outcome_retracted", outcome_id,
                              {"reason": _text(reason, "reason", 400)[:400],
                               "subject": f"{row['subject_kind']}:{row['subject_id']}"})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": outcome_id, "retracted": True}


def _artifact_id(span: str) -> str:
    text = str(span).split("#", 1)[0]
    return text if text.startswith("out_") else "out_" + text


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise EvidenceError(f"{label} must be nonempty text of at most {maximum} "
                            "characters")
    return " ".join(value.split())
