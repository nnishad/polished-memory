"""C10 outbox: durable artifacts, and an honest account of who was told what.

The framework makes the thing to say; Hermes delivers it. Nothing here holds a
messaging credential, and nothing here claims a delivery it did not see a receipt
for. The states are all about the artifact, and the difference between
``confirmed``, ``accepted_unverified`` and ``uncertain`` is the whole point: a host
that says "sent" without a checkable digest has told us about its own intent, not
about the owner's inbox.

Two rules are load-bearing. **Revalidate before delivery**, because a prepared
artifact is a claim about a moment that has since passed — evidence can be erased,
a goal completed, a topic opted out, a digest already sent. **No blind resend**,
because the one thing worse than a missing notification is the third copy of it.
An artifact that reached ``attempted`` has been handed to something that can put
plaintext in front of a person, and no read-only receipt bridge can take that back.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from ..ids import digest, new_id, now, timestamp
from ..learning.lessons import record_pk
from ..storage.evidence import EvidenceError

__all__ = ["Outbox", "Artifact", "ArtifactClaim", "DELIVERABLE", "OPEN_STATES",
           "TERMINAL", "DEFAULT_LEASE_S"]

DELIVERABLE = ("next_turn", "digest", "notify_owner", "draft")
OPEN_STATES = ("prepared", "leased")
UNCHECKABLE = ("attempted", "uncertain")
TERMINAL = ("confirmed", "accepted_unverified", "suppressed", "expired")
DEFAULT_LEASE_S = 120.0
MAX_PAYLOAD_CHARS = 6000


@dataclass(frozen=True)
class Artifact:
    id: str
    decision_id: str
    kind: str
    topic: str
    recipient: str
    payload: str
    payload_digest: str
    evidence: tuple[str, ...]
    policy_version: str
    expires_at: str | None
    state: str
    reason: str | None
    attempts: int
    created_at: str
    updated_at: str
    delivered_at: str | None

    @classmethod
    def from_row(cls, row) -> "Artifact":
        try:
            evidence = tuple(json.loads(row["evidence"] or "[]"))
        except (TypeError, json.JSONDecodeError):
            evidence = ()
        return cls(id=row["id"], decision_id=row["decision_id"], kind=row["kind"],
                   topic=row["topic"], recipient=row["recipient"], payload=row["payload"],
                   payload_digest=row["payload_digest"], evidence=evidence,
                   policy_version=row["policy_version"], expires_at=row["expires_at"],
                   state=row["state"], reason=row["reason"], attempts=int(row["attempts"]),
                   created_at=row["created_at"], updated_at=row["updated_at"],
                   delivered_at=row["delivered_at"])

    def as_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        out = {"id": self.id, "decision": self.decision_id, "kind": self.kind,
               "topic": self.topic, "recipient": self.recipient, "state": self.state,
               "reason": self.reason, "attempts": self.attempts,
               "payload_digest": self.payload_digest, "evidence": list(self.evidence),
               "expires_at": self.expires_at, "created_at": self.created_at,
               "delivered_at": self.delivered_at}
        if include_payload:
            out["payload"] = self.payload
        return out


@dataclass(frozen=True)
class ArtifactClaim:
    artifact: Artifact
    token: str


@dataclass(frozen=True)
class Revalidation:
    ok: bool
    stage: str
    reason: str
    retry_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "stage": self.stage, "reason": self.reason,
                "retry_at": self.retry_at}


class Outbox:
    def __init__(self, store, *, policy, owner_principal: str | None = None, clock=None,
                 goals=None, events=None):
        self.store = store
        self.db = store.db
        self.policy = policy
        self.owner_principal = owner_principal
        self.goals = goals
        self.events = events
        self._clock = clock

    def _now(self) -> float:
        return time.time() if self._clock is None else float(self._clock())

    @contextmanager
    def _writing(self, db: sqlite3.Connection | None):
        connection = db or self.db
        if db is not None and not db.in_transaction:
            raise EvidenceError("this write needs an ambient transaction")
        owns = db is None
        if owns:
            connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            if owns:
                connection.execute("ROLLBACK")
            raise
        else:
            if owns:
                connection.execute("COMMIT")

    # -- preparation ---------------------------------------------------------

    def prepare(self, *, decision_id: str, kind: str, topic: str, payload: str,
                evidence: Iterable[str] = (), recipient: str | None = None,
                expires_at: str | None = None,
                db: sqlite3.Connection | None = None) -> dict[str, Any]:
        """Make one durable artifact for one decision. Replay returns the original.

        The payload is hashed as it is stored, so the digest the host is asked to
        echo back is a statement about these exact bytes rather than about whatever
        string the model produced earlier in the pipeline.
        """
        if kind not in DELIVERABLE:
            raise EvidenceError(f"{kind!r} is not a deliverable artifact; 'silent' is "
                                "the absence of one")
        row = self.db.execute("SELECT * FROM proactive_decisions WHERE id=?",
                              (decision_id,)).fetchone()
        if row is None:
            raise EvidenceError(f"unknown decision {decision_id!r}; an artifact answers a "
                                "recorded decision, not an intention on its own")
        text = _payload_text(payload)
        bound = recipient or self.owner_principal
        if bound is None:
            raise EvidenceError("no owner principal is named, so nobody may be addressed")
        if bound != self.owner_principal:
            # Notify-and-draft. A third-party send belongs to Hermes with the
            # owner's approval, never to a component that can only see evidence.
            raise EvidenceError(
                f"{bound!r} is not the owner: this framework notifies the owner and "
                "drafts for their approval, it does not message anyone else")
        if kind == "digest" and row["window_id"] is None:
            raise EvidenceError("a digest artifact needs the window it belongs to")
        stamp = now()
        artifact_id = "out_" + digest(["outbox", decision_id, kind])[:32]
        digest_value = digest(["v1", topic, bound, text, list(evidence)])
        with self._writing(db) as connection:
            existing = connection.execute("SELECT * FROM outbox WHERE id=?",
                                          (artifact_id,)).fetchone()
            if existing is not None:
                return {**Artifact.from_row(existing).as_dict(), "prepared": False,
                        "note": "this decision already has that artifact; nothing was "
                                "queued twice"}
            connection.execute(
                "INSERT INTO outbox(id, decision_id, kind, topic, recipient, payload, "
                "payload_digest, evidence, policy_version, expires_at, state, created_at, "
                "updated_at) VALUES(?,?,?,?,?,?,?,?,?,?, 'prepared', ?, ?)",
                (artifact_id, decision_id, kind, topic, bound, text, digest_value,
                 json.dumps(list(evidence), ensure_ascii=False)[:4000],
                 str(row["policy_version"]),
                 timestamp(expires_at) if expires_at else None, stamp, stamp))
            self.store._audit("outbox_prepare", artifact_id,
                              {"decision": decision_id, "kind": kind, "topic": topic,
                               "evidence": len(list(evidence)), "digest": digest_value[:16]})
        return {**self.get(artifact_id).as_dict(), "prepared": True}

    # -- reading -------------------------------------------------------------

    def get(self, artifact_id: str) -> Artifact | None:
        row = self.db.execute("SELECT * FROM outbox WHERE id=?", (artifact_id,)).fetchone()
        return Artifact.from_row(row) if row else None

    def for_decision(self, decision_id: str) -> list[Artifact]:
        rows = self.db.execute("SELECT * FROM outbox WHERE decision_id=? ORDER BY kind",
                               (decision_id,)).fetchall()
        return [Artifact.from_row(row) for row in rows]

    def pending(self, *, state: str = "prepared", limit: int = 25) -> list[Artifact]:
        if state not in ("prepared", "leased", "attempted", "uncertain"):
            raise EvidenceError(f"cannot list pending artifacts in state {state!r}")
        rows = self.db.execute("SELECT * FROM outbox WHERE state=? ORDER BY created_at, id "
                               "LIMIT ?", (state, _bounded(limit))).fetchall()
        return [Artifact.from_row(row) for row in rows]

    # -- delivery ------------------------------------------------------------

    def lease(self, *, holder: str, kind: str | None = None, at: float | None = None,
              lease_s: float = DEFAULT_LEASE_S) -> ArtifactClaim | None:
        """Take one artifact that is safe to deliver right now.

        Revalidation happens *inside* the lease, not before it: otherwise two
        workers can both see a clean artifact and one of them delivers it after the
        other has already suppressed it.
        """
        if not isinstance(holder, str) or not holder.strip():
            raise EvidenceError("a lease names who holds it")
        if not isinstance(lease_s, (int, float)) or not 1 <= lease_s <= 3600:
            raise EvidenceError("lease_s must be between 1 and 3600 seconds")
        moment = self._now() if at is None else float(at)
        self.recover(at=moment)
        clause = " AND kind=?" if kind else ""
        rows = self.db.execute(
            f"SELECT id FROM outbox WHERE state='prepared'{clause} ORDER BY created_at, id "
            "LIMIT ?", ([kind] if kind else []) + [_bounded(20)]).fetchall()
        for row in rows:
            artifact = self.get(str(row["id"]))
            check = self.revalidate(artifact.id, at=moment)
            if not check.ok:
                if check.retry_at is not None:
                    continue  # not yet, not never: leave it prepared for a later lease
                # The list above is a snapshot taken a moment ago. Another drain may have
                # claimed this one in between, and an artifact that is mid-flight somewhere
                # else is not this lease's to close: suppressing it would both throw here
                # and unsend a delivery somebody else already carried out.
                if artifact.state != "prepared":
                    continue
                try:
                    self.suppress(artifact.id, reason=check.reason)
                except EvidenceError:
                    continue  # lost that race by a heartbeat, so there is nothing to close
                continue
            token = new_id("send")
            self.db.execute("BEGIN IMMEDIATE")
            try:
                moved = self.db.execute(
                    "UPDATE outbox SET state='leased', lease_token=?, lease_until=?, "
                    "held_by=?, attempts=attempts+1, updated_at=? WHERE id=? "
                    "AND state='prepared'",
                    (token, moment + float(lease_s), holder, now(), artifact.id))
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            if int(moved.rowcount or 0):
                return ArtifactClaim(artifact=self.get(artifact.id), token=token)
        return None

    def release(self, *, artifact_id: str, token: str,
                reason: str = "the holder did not send it") -> Artifact:
        """Give an unsent artifact back. Only the holder, and only before the handover.

        Without this, a worker that leased something and then decided against it
        could only leave it leased until recovery called the send uncertain — which
        would report a delivery risk that never existed.
        """
        with self._writing(None) as connection:
            row = connection.execute("SELECT * FROM outbox WHERE id=?",
                                     (artifact_id,)).fetchone()
            if row is None:
                raise EvidenceError(f"unknown artifact {artifact_id!r}")
            if row["state"] != "leased" or row["lease_token"] != token:
                raise EvidenceError(_refusal_reason(str(row["state"])))
            connection.execute(
                "UPDATE outbox SET state='prepared', reason=?, lease_token=NULL, "
                "lease_until=NULL, held_by=NULL, updated_at=? WHERE id=?",
                (reason[:400], now(), artifact_id))
        return self.get(artifact_id)

    def attempt(self, *, artifact_id: str, token: str) -> Artifact:
        """Say that the handover has begun, before beginning it.

        This is the write that makes a crash survivable: an artifact stuck in
        ``attempted`` is one a person may already have seen, which is a different
        and much more serious fact than one nobody received. The token and its
        deadline survive this transition, so the receipt can still be fenced to the
        holder and a dead handover still counts as one.
        """
        with self._writing(None) as connection:
            row = connection.execute("SELECT * FROM outbox WHERE id=?",
                                     (artifact_id,)).fetchone()
            if row is None:
                raise EvidenceError(f"unknown artifact {artifact_id!r}")
            if row["state"] != "leased" or row["lease_token"] != token:
                raise EvidenceError(_refusal_reason(str(row["state"])))
            connection.execute(
                "UPDATE outbox SET state='attempted', updated_at=? WHERE id=?",
                (now(), artifact_id))
        return self.get(artifact_id)

    def confirm(self, *, artifact_id: str, token: str, proof: Any = None) -> Artifact:
        """Record what the host came back with — and what it did not prove.

        A proof whose digest is not this artifact's digest is uncertainty, not
        confirmation: something was delivered, and it was not this.
        """
        state, reason = _receipt(proof, self.get(artifact_id))
        with self._writing(None) as connection:
            row = connection.execute("SELECT * FROM outbox WHERE id=?",
                                     (artifact_id,)).fetchone()
            if row is None:
                raise EvidenceError(f"unknown artifact {artifact_id!r}")
            if row["state"] != "attempted":
                raise EvidenceError(_refusal_reason(str(row["state"])))
            if row["lease_token"] != token:
                raise EvidenceError(
                    "this receipt is not from the holder of the handover; whoever began "
                    "the send is the only one who can report how it ended")
            connection.execute(
                "UPDATE outbox SET state=?, reason=?, proof=?, lease_token=NULL, "
                "lease_until=NULL, delivered_at=?, updated_at=? WHERE id=?",
                (state, reason, json.dumps(proof, ensure_ascii=False)[:2000] if proof else None,
                 now() if state == "confirmed" else None, now(), artifact_id))
            _carry_to_intent(connection, str(row["decision_id"]),
                             "delivered" if state == "confirmed" else state)
        self.store._audit("outbox_receipt", artifact_id, {"state": state,
                                                          "reason": reason[:200]})
        return self.get(artifact_id)

    def uncertain(self, *, artifact_id: str, token: str | None = None,
                  reason: str = "the delivery outcome was not established") -> Artifact:
        """The send is unresolved. It is not put back in the queue."""
        with self._writing(None) as connection:
            row = connection.execute("SELECT * FROM outbox WHERE id=?",
                                     (artifact_id,)).fetchone()
            if row is None:
                raise EvidenceError(f"unknown artifact {artifact_id!r}")
            if row["state"] not in ("leased", "attempted", "uncertain"):
                raise EvidenceError(f"{artifact_id} is {row['state']}; only an unfinished "
                                    "send can become uncertain")
            if token is not None and row["lease_token"] not in (None, token):
                raise EvidenceError("this lease belongs to someone else")
            connection.execute("UPDATE outbox SET state='uncertain', reason=?, "
                               "lease_token=NULL, lease_until=NULL, updated_at=? "
                               "WHERE id=?",
                               (reason[:400], now(), artifact_id))
            _carry_to_intent(connection, str(row["decision_id"]), "uncertain")
        self.store._audit("outbox_uncertain", artifact_id, {"reason": reason[:200]})
        return self.get(artifact_id)

    def suppress(self, artifact_id: str, *, reason: str,
                 db: sqlite3.Connection | None = None) -> Artifact:
        with self._writing(db) as connection:
            row = connection.execute("SELECT * FROM outbox WHERE id=?",
                                     (artifact_id,)).fetchone()
            if row is None:
                raise EvidenceError(f"unknown artifact {artifact_id!r}")
            if row["state"] not in OPEN_STATES:
                raise EvidenceError(
                    f"{artifact_id} is already {row['state']}; a delivered artifact "
                    "cannot be unsent, and pretending otherwise would be the "
                    "read-only-receipt lie this state machine exists to avoid")
            connection.execute("UPDATE outbox SET state='suppressed', reason=?, "
                               "lease_token=NULL, lease_until=NULL, updated_at=? WHERE id=?",
                               (reason[:400], now(), artifact_id))
            _carry_to_intent(connection, str(row["decision_id"]), "suppressed")
        self.store._audit("outbox_suppress", artifact_id, {"reason": reason[:200]})
        return self.get(artifact_id)

    def expire(self, *, at: str | None = None) -> list[str]:
        """Stale queued work is dropped visibly, never delivered as if it were news."""
        moment = timestamp(at) if at else now()
        rows = self.db.execute("SELECT id FROM outbox WHERE state IN ('prepared', 'leased') "
                               "AND expires_at IS NOT NULL AND expires_at < ?",
                               (moment,)).fetchall()
        expired: list[str] = []
        for row in rows:
            self._set(str(row["id"]), state="expired", reason="the owner's moment for this "
                                                             "artifact passed")
            expired.append(str(row["id"]))
        return expired

    def recover(self, *, at: float | None = None) -> list[str]:
        """A lapsed handover is uncertain. It does not go back in the queue.

        Both halves of the send are swept: a holder that died after leasing and one
        that died after handing the plaintext to the host are different failures,
        but neither may be answered with a second attempt whose outcome nobody saw.
        """
        moment = self._now() if at is None else float(at)
        rows = self.db.execute("SELECT id, held_by, state FROM outbox WHERE "
                               "state IN ('leased', 'attempted') AND lease_until < ?",
                               (moment,)).fetchall()
        for row in rows:
            self.uncertain(artifact_id=str(row["id"]),
                           reason=f"{row['held_by']}'s lease lapsed while "
                                  f"{row['state']}")
        return [str(row["id"]) for row in rows]

    # -- revalidation --------------------------------------------------------

    def revalidate(self, artifact_id: str, *, at: float | None = None) -> Revalidation:
        """Is this artifact still true, still allowed, and still safe to send?

        Called by the lease and again by the host before the handover. Each failure
        says which of the five things changed, because "it could not be sent" is not
        an answer anyone can act on.
        """
        artifact = self.get(artifact_id)
        if artifact is None:
            raise EvidenceError(f"unknown artifact {artifact_id!r}")
        if artifact.state not in OPEN_STATES:
            return Revalidation(False, "state", f"the artifact is {artifact.state}")
        if digest(["v1", artifact.topic, artifact.recipient, artifact.payload,
                   list(artifact.evidence)]) != artifact.payload_digest:
            return Revalidation(False, "digest", "the stored payload no longer matches its "
                                                 "digest; the bytes were changed under us")
        gone = [span for span in artifact.evidence if not self._span_live(span)]
        if gone:
            return Revalidation(False, "evidence",
                                f"{len(gone)} of {len(artifact.evidence)} cited spans is no "
                                "longer on file; the owner forgot or withdrew what this "
                                "artifact was about")
        goal = self.db.execute(
            "SELECT g.status, g.revision AS current_revision, d.revision AS wanted, "
            "g.snoozed_until FROM proactive_decisions d JOIN goals g ON g.id = d.goal_id "
            "WHERE d.id=?", (artifact.decision_id,)).fetchone()
        if goal is not None and goal["status"] != "active":
            return Revalidation(False, "goal", f"the goal is {goal['status']}; a completed "
                                               "promise is not a notification")
        if goal is not None and int(goal["wanted"]) != int(goal["current_revision"]):
            return Revalidation(False, "revision",
                                "the goal was revised after this artifact was prepared, so "
                                "it describes a version the owner has already changed")
        settings = self.policy.settings(artifact.topic)
        if settings["state"] == "opted_out":
            return Revalidation(False, "opt_out", "the owner opted this topic out while the "
                                                  "artifact was queued")
        if self.store.stage_is_paused(f"topic:{artifact.topic}", "attention"):
            return Revalidation(False, "pause", "an operator paused this topic's attention")
        if artifact.kind == "digest":
            window = self.db.execute(
                "SELECT w.state FROM attention_windows w JOIN proactive_decisions d "
                "ON d.window_id = w.id WHERE d.id=?", (artifact.decision_id,)).fetchone()
            if window is not None and window["state"] != "open":
                return Revalidation(False, "window", "the digest window this belongs to has "
                                                     "already been delivered")
        moment = self._now() if at is None else float(at)
        waking = self.policy.next_waking(artifact.topic, _iso(moment))
        if waking is not None:
            return Revalidation(False, "quiet", "it is the owner's quiet hours",
                                retry_at=waking)
        if artifact.expires_at and timestamp(artifact.expires_at) <= _iso(moment):
            return Revalidation(False, "expiry", "the artifact expired")
        if artifact.policy_version != _policy_version(self.db, artifact.decision_id):
            return Revalidation(False, "policy", "the policy that decided this artifact is "
                                                 "no longer the one in force")
        return Revalidation(True, "ok", "still true, still allowed, still the owner's")

    def _span_live(self, span: str) -> bool:
        """Does the cited evidence still exist and is it still visible?

        A derived citation names a backend document, so it resolves through the
        projection map; either way an erased record revokes the artifact that quoted
        it, and the artifact is suppressed rather than delivered on stale support.
        """
        pk = record_pk(span)
        if self.store.live_and_visible(pk):
            return True
        row = self.db.execute("SELECT record_id FROM backend_documents WHERE document_id=? "
                              "LIMIT 1", (pk,)).fetchone()
        return bool(row) and self.store.live_and_visible(str(row["record_id"]))

    def _set(self, artifact_id: str, **fields) -> None:
        fields.setdefault("updated_at", now())
        keys = ", ".join(f"{key}=?" for key in fields)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(f"UPDATE outbox SET {keys} WHERE id=?",
                            (*fields.values(), artifact_id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise


def _receipt(proof: Any, artifact: Artifact) -> tuple[str, str]:
    if isinstance(proof, dict) and proof.get("digest"):
        if str(proof["digest"]) == artifact.payload_digest:
            return "confirmed", "the host echoed this artifact's own digest back"
        return "uncertain", ("the host reported a send for a different digest, so something "
                             "reached the owner that is not this artifact")
    if (isinstance(proof, dict) and proof.get("correlation") and proof.get("verified")):
        # A transport that read the message back off its own wire and found this artifact's
        # correlation in it. The claim of read-back is what earns the difference: an echo of
        # the bytes it was handed proves the transport got them, not that anyone was shown
        # them, so without `verified` this falls through to accepted_unverified below.
        parts = str(proof["correlation"]).split()
        if (len(parts) == 3 and parts[0] == "hermes-memory" and parts[1] == artifact.id
                and artifact.payload_digest.startswith(parts[2])):
            return "confirmed", ("the transport read this artifact's own correlation back "
                                 "from what it sent")
        return "uncertain", ("the correlation the transport read back is not this artifact's, "
                             "so something else reached the owner")
    if isinstance(proof, dict) and proof.get("sent") is True:
        return "accepted_unverified", ("the host says it sent this and gave nothing to check "
                                       "it against")
    if proof is None:
        return "accepted_unverified", "the host acknowledged without a receipt"
    return "uncertain", f"the receipt is not understood: {str(proof)[:120]}"


def _refusal_reason(state: str) -> str:
    if state in UNCHECKABLE:
        return (f"an artifact in {state!r} has already been handed to something that can put "
                "plaintext in front of the owner. Re-sending it blind is the failure this "
                "state machine exists to prevent; a corrected item is a new decision")
    if state in TERMINAL:
        return f"the artifact is {state} and finished; it cannot be re-opened"
    return f"the artifact is {state}, which is not a live lease"


def _carry_to_intent(connection, decision_id: str, state: str) -> None:
    """Mirror the artifact's fate onto the intention behind it.

    One decision may hold a next_turn and a digest artifact, so the intention is
    only delivered when something was, and only uncertain when nothing can be.
    """
    row = connection.execute(
        "SELECT d.intent_id AS intent FROM proactive_decisions d WHERE d.id=?",
        (decision_id,)).fetchone()
    if row is None:
        return
    if state == "delivered":
        connection.execute("UPDATE decision_intents SET state='delivered', updated_at=? "
                           "WHERE id=? AND state IN ('prepared', 'awaiting_analysis')",
                           (now(), row["intent"]))
    elif state in ("suppressed", "uncertain"):
        remaining = connection.execute(
            "SELECT count(*) FROM outbox WHERE decision_id=? AND state IN "
            "('prepared', 'leased', 'attempted', 'confirmed')", (decision_id,)).fetchone()[0]
        if not int(remaining):
            connection.execute("UPDATE decision_intents SET state=?, updated_at=? WHERE id=? "
                               "AND state IN ('prepared', 'awaiting_analysis')",
                               (state, now(), row["intent"]))


def _policy_version(db, decision_id: str) -> str:
    row = db.execute("SELECT policy_version FROM proactive_decisions WHERE id=?",
                     (decision_id,)).fetchone()
    return str(row["policy_version"]) if row else ""


def _payload_text(payload: Any) -> str:
    if isinstance(payload, dict):
        payload = str(payload.get("text") or "")
    if not isinstance(payload, str) or not payload.strip():
        raise EvidenceError("an artifact must carry text somebody can read")
    if len(payload) > MAX_PAYLOAD_CHARS:
        raise EvidenceError(f"an artifact may not exceed {MAX_PAYLOAD_CHARS} characters; "
                            "a wall of text is not a notification")
    return " ".join(payload.split())


def _iso(moment: Any) -> str:
    """An instant as UTC ISO, from either half of the clock the caller may hold.

    Lease arithmetic is monotonic seconds and policy arithmetic is wall clock, and
    a caller that has one should not have to convert to use the other.
    """
    if isinstance(moment, str):
        return timestamp(moment)
    if not isinstance(moment, (int, float)) or abs(float(moment)) > 4e9:
        raise EvidenceError(f"{moment!r} is not a usable instant")
    return datetime.fromtimestamp(float(moment), tz=timezone.utc).isoformat()


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= 200 else 20
