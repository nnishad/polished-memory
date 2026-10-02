"""Questions to the owner, and the one reply that may settle a decision.

Everything else in this package decides *whether* the framework may say something. This
module is for the case where what it needs is not to say something but to be told: an erasure
preview only the owner can confirm, two accounts that may or may not be one person, a goal an
agent proposed and nobody adopted. Those already wait behind `owner --list`, in a store with
no way to reach a human.

Three rules hold the design up.

*The fence is computed, never supplied.* A question stores a digest of the row it is about,
derived here from the archive's own state, so an answer settles exactly what the owner was
shown. If anything about that row changed meanwhile, the question is void rather than
answered — confirming a preview you were never shown is the failure this framework exists to
make impossible, and a transport is no place to reintroduce it.

*The code is written down once, in the message.* Every question carries a fresh one-use code;
only its digest is stored, so reading the database — including reading it as the agent, which
may ask about anything — does not hand out the ability to answer. A reply without a code is
nobody's decision, however much it reads like one. What the code does *not* do is keep the
answer away from the agent: a messaging client replies by quoting, so the question and its code
come back in what the conversation can read. The code says the sender read the message; that it
came from the owner rather than from the model is carried by the channel the host attests and by
this module refusing to let any other door decide what it is asking about.

*Nothing here is a default.* A reply settling an irreversible act needs the owner's switch,
an approved *private* destination, and the same attention budget a reminder spends: a
question and a notification are both interruptions, and two allowances would let each wake
the owner as often as the other.
"""
from __future__ import annotations

import json
import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator

from ..ids import digest, new_id, now, timestamp
from ..storage.evidence import EvidenceError
from ..storage.transactions import write_transaction
from .policy import AttentionPolicy

__all__ = ["DEFAULT_TTL_S", "FENCES", "Inquiry", "InquiryStore", "REPLIES_STAGE",
           "code_for", "code_in", "fence"]

#: The stage the owner's switch is recorded under. Absence means off: a text message that
#: can erase evidence is not a default, it is something a person decided.
REPLIES_STAGE = "owner_replies"
DEFAULT_TTL_S = 7 * 24 * 3600
MAX_TTL_S = 30 * 24 * 3600
#: What a question can be about, and how its row is found. The digest covers the whole row,
#: so any change to it moves the fence; `awaiting` is the state that still needs the owner.
FENCES: dict[str, tuple[str, tuple[str, ...], str, tuple[str, ...]]] = {
    "identity": ("identity_candidates", ("id",), "state", ("pending",)),
    "identity-rejection": ("identity_candidates", ("id",), "state", ("pending",)),
    "edge-revocation": ("identity_edges", ("id",), "state", ("active",)),
    "assertion": ("assertions", ("id",), "status", ("candidate",)),
    "assertion-retraction": ("assertions", ("id",), "status", ("confirmed",)),
    "lesson-activation": ("lessons", ("id", "version"), "status", ("candidate",)),
    "lesson-retraction": ("lessons", ("id", "version"), "status", ("active",)),
    "lesson-confirmation": ("lessons", ("id", "version"), "status",
                            ("active", "candidate")),
    "lesson-contradiction": ("lessons", ("id", "version"), "status",
                             ("active", "candidate")),
    "goal-activation": ("goals", ("id",), "status", ("candidate",)),
    "goal-completion": ("goals", ("id",), "status", ("candidate", "active")),
    "goal-cancellation": ("goals", ("id",), "status", ("candidate", "active")),
}
# A code a person can type from a phone, without an alphabet where 0/O and 1/I look alike.
# Six characters is ~887 million combinations, and every wrong one is written down.
ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
CODE_LENGTH = 6
_CODE_IN_TEXT = re.compile(r"\b([%s]{%d})\b" % (ALPHABET, CODE_LENGTH))
_STATES = ("open", "sent", "answered", "expired", "withdrawn", "void")


def code_for() -> str:
    """A fresh one-use code. It exists in this return value and in the owner's message."""
    return "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))


def code_in(text: str) -> str | None:
    """The code inside a reply, if the owner's answer carried one."""
    answer = _ANSWER.fullmatch(str(text or "").strip())
    if answer and answer.group("code"):
        return answer.group("code").upper()
    found = _CODE_IN_TEXT.search(str(text or "").upper())
    return found.group(0) if found else None


#: The words that turn a code into an instruction. The code proves who sent the reply and
#: says nothing about what to do with it, so a door that read the code and not the words
#: would confirm the very erasure the owner was refusing. English only, on purpose: a reply
#: in another word for yes is refused with the two words that would have worked, rather than
#: guessed at by a sentiment model nobody audited.
YES_WORDS = frozenset({"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm",
                       "confirmed", "approve", "approved", "agree", "proceed", "correct",
                       "exactly"})
NO_WORDS = frozenset({"no", "n", "nope", "nah", "dont", "never", "stop", "reject",
                      "rejected", "deny", "denied", "refuse", "refused", "wrong"})
_ANSWER = re.compile(
    r"(?P<word>" + "|".join(sorted(YES_WORDS | NO_WORDS)) + r")"
    r"(?:\s+(?P<code>[" + ALPHABET + r"]{6}))?[.!]?", re.IGNORECASE)


def _polarity(text: str) -> str | None:
    """An explicit yes/no command, not sentiment extracted from arbitrary prose.

    Both words in one reply is also None: "no, not that one, yes the other" is a conversation,
    not a decision, and the owner is still holding the code that can have either ending.
    """
    answer = _ANSWER.fullmatch(str(text or "").strip())
    if answer is None:
        return None
    return "yes" if answer.group("word").lower() in YES_WORDS else "no"


def fence(store, *, decision: str, subject_id: str) -> dict[str, Any]:
    """The digest of the thing a question would be about, and whether it still awaits.

    Whole-row rather than a chosen handful of columns: picking which fields matter is how a
    fence ends up blind to the change that mattered. The subject id may carry a version
    (`chase-invoice@3`), which is parsed the same way the owner door parses it.
    """
    from ..operations.decisions import lesson_ref

    if decision == "forgetting":
        from ..lifecycle.erasure import ErasureManager

        found = ErasureManager(store, owner_principal=None).preview_current(
            intent_id=subject_id)
        return {"found": found["digest"] is not None, "awaiting": bool(found["awaiting"]),
                "digest": str(found.get("digest") or ""),
                "reason": str(found.get("reason") or "")}
    table, keys, state_column, awaiting = FENCES[decision]
    if table == "lessons":
        # A lesson is keyed by name *and* version, and the subject carries both: `name@3` is
        # how the owner door writes it, so the question cannot quietly mean "the newest one".
        named, number = lesson_ref(subject_id)
        if number is None:
            raise EvidenceError("a lesson question names its version: `chase-invoice@3`")
        values = [named, str(number)]
    else:
        values = [subject_id]
    placeholders = " AND ".join(f"{key}=?" for key in keys)
    row = store.db.execute(f"SELECT * FROM {table} WHERE {placeholders}", values).fetchone()
    if row is None:
        return {"found": False, "awaiting": False, "digest": "",
                "reason": f"no {table} row {subject_id!r}"}
    fields = {key: row[key] for key in row.keys()}
    return {"found": True,
            "awaiting": str(fields.get(state_column)) in awaiting,
            "digest": digest([decision, table,
                              json.dumps(fields, sort_keys=True, default=str)]),
            "reason": "" if str(fields.get(state_column)) in awaiting else
            f"it is already {fields.get(state_column)}"}


@dataclass(frozen=True)
class Inquiry:
    """One question. `as_dict` cannot leak the code, because the store never kept it."""

    id: str
    topic: str
    decision: str
    subject_id: str
    subject_digest: str
    question: str
    choices: tuple[str, ...]
    state: str
    reason: str | None
    epoch: int
    attempts: int
    asked_at: str
    expires_at: str
    sent_at: str | None
    answered_at: str | None
    answer: str | None
    answer_channel: str | None
    settled: dict[str, Any] | None

    @classmethod
    def from_row(cls, row) -> "Inquiry":
        settled = json.loads(str(row["settled"] or "{}")) or None
        return cls(id=str(row["id"]), topic=str(row["topic"]),
                   decision=str(row["decision"]), subject_id=str(row["subject_id"]),
                   subject_digest=str(row["subject_digest"]),
                   question=str(row["question"]),
                   choices=tuple(json.loads(str(row["choices"] or "[]"))),
                   state=str(row["state"]), reason=row["reason"],
                   epoch=int(row["epoch"]), attempts=int(row["attempts"]),
                   asked_at=str(row["asked_at"]), expires_at=str(row["expires_at"]),
                   sent_at=row["sent_at"], answered_at=row["answered_at"],
                   answer=row["answer"], answer_channel=row["answer_channel"],
                   settled=settled)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "topic": self.topic, "decision": self.decision,
                "subject_id": self.subject_id, "subject_digest": self.subject_digest,
                "question": self.question, "choices": list(self.choices),
                "state": self.state, "reason": self.reason, "epoch": self.epoch,
                "attempts": self.attempts, "asked_at": self.asked_at,
                "expires_at": self.expires_at, "sent_at": self.sent_at,
                "answered_at": self.answered_at, "answer": self.answer,
                "answer_channel": self.answer_channel, "settled": self.settled}

    def body(self, code: str, *, destination: str) -> str:
        """The message the owner sees, and the only place the code is ever written.

        It says what answering will do, because an owner who replies yes to a message that
        did not say what it was confirming has not consented to anything.
        """
        lines = [f"hermes-memory question {self.id} {self.subject_digest[:12]}",
                 self.question,
                 f"It would settle: {self.decision} ({self.subject_id})."]
        if self.choices:
            lines.append("Answer with one of: " + ", ".join(self.choices))
        local = _parse(self.expires_at).astimezone()
        lines.append(f"Until {local.strftime('%d %b %H:%M')} — after that it has to be "
                     "asked again.")
        lines.append(f"To settle it by reply, use `yes {code}`. "
                     f"`no {code}` stops the asking and decides nothing; "
                     "it does not reject the underlying proposal.")
        lines.append(f"It reaches you on {destination}, and settles nothing without the "
                     "code. Otherwise reply `hermes-memory owner` for the door.")
        return "\n".join(lines)


class InquiryStore:
    """The asking, the sending and the answering of owner questions."""

    def __init__(self, store, *, owner_principal: str | None = None,
                 policy: AttentionPolicy | None = None):
        self.store = store
        self.db = store.db
        self.owner_principal = owner_principal
        self.policy = policy or AttentionPolicy(store, owner_principal=owner_principal)

    @contextmanager
    def _writing(self) -> Iterator[Any]:
        with write_transaction(self.db):
            yield self.db

    # -- the owner's switch --------------------------------------------------

    def replies_allowed(self) -> tuple[bool, str]:
        """Whether a chat reply may settle a decision here, and who said so."""
        hold = self.store.control("global", REPLIES_STAGE)
        if hold is None:
            return False, ("nobody has switched replies on, so a question is answered "
                           "through `hermes-memory owner`")
        if hold["state"] != "active":
            return False, f"replies are switched off by {hold['actor']}: {hold['reason']}"
        return True, f"allowed by {hold['actor']} since {hold['changed_at']}"

    def allow_replies(self, *, actor: str, on: bool, reason: str,
                      policy_version: str = "inquiry-1") -> dict[str, Any]:
        """Owner-only. A capability to settle an erasure by text message is not a default."""
        if not actor or actor != self.owner_principal:
            raise EvidenceError("only the owner principal may allow a chat reply to settle "
                                "one of their decisions")
        text = " ".join(str(reason or "").split())
        if len(text) < 3:
            raise EvidenceError("switching replies on or off has to say why")
        self.store.set_control("global", REPLIES_STAGE, "active" if on else "paused",
                               actor=actor, reason=text[:1000],
                               policy_version=policy_version)
        self.store._audit("inquiry_replies_switch", f"global:{REPLIES_STAGE}",
                          {"actor": actor, "on": bool(on), "reason": text[:200]})
        return {"replies_enabled": bool(on), "actor": actor, "reason": text}

    # -- asking --------------------------------------------------------------

    def ask(self, *, decision: str, subject_id: str, question: str,
            topic: str = "general", choices: tuple[str, ...] = (),
            ttl_s: int = DEFAULT_TTL_S, reason: str | None = None) -> dict[str, Any]:
        with self._writing():
            return self._ask(decision=decision, subject_id=subject_id, question=question,
                             topic=topic, choices=choices, ttl_s=ttl_s, reason=reason)

    def _ask(self, *, decision: str, subject_id: str, question: str,
             topic: str, choices: tuple[str, ...], ttl_s: int,
             reason: str | None) -> dict[str, Any]:
        """Open one question about one thing, and hold it until the owner may be asked.

        Nothing is sent from here: the caller that owns a transport calls ``send_next``, so a
        question can never be delivered by the code that formed it. The digest comes from the
        archive rather than from the asker, which is what stops a question being asked about
        a state that never existed.
        """
        from ..operations.decisions import ACTS

        if decision not in ACTS:
            raise EvidenceError(f"{decision!r} is not an owner decision; a question can only "
                                f"be about {list(ACTS)}")
        subject = " ".join(str(subject_id or "").split())
        if not subject or len(subject) > 200:
            raise EvidenceError("a question names the thing it asks about")
        text = " ".join(str(question or "").split())
        if not 10 <= len(text) <= 900:
            raise EvidenceError("the question has to be between 10 and 900 characters")
        options = tuple(" ".join(str(item).split())[:40] for item in (choices or ())[:4])
        if not isinstance(ttl_s, int) or isinstance(ttl_s, bool) \
                or not 60 <= ttl_s <= MAX_TTL_S:
            raise EvidenceError(f"ttl_s must be between 60 and {MAX_TTL_S} seconds")
        pinned = fence(self.store, decision=decision, subject_id=subject)
        if not pinned["found"]:
            raise EvidenceError(f"there is nothing to ask about: {pinned['reason']}")
        if not pinned["awaiting"]:
            return {"asked": False, "ok": True, "reason": f"nothing is awaited: "
                    f"{pinned['reason']}", "fence": pinned}
        asked = now()
        expires = (_parse(asked) + timedelta(seconds=ttl_s)).isoformat()
        identifier = "inq_" + digest(["inquiry", decision, subject,
                                      pinned["digest"]])[:24]
        existing = self.get(identifier)
        if existing is not None:
            if existing.state in ("void", "expired"):
                # A question that never reached the owner, or reached them too late, is not a
                # question that was asked and refused: the row that recorded the failure would
                # otherwise stand in the way of asking again forever. `withdrawn` and
                # `answered` are the owner's own answers, and those stay closed.
                with self._writing():
                    self.db.execute(
                        "UPDATE inquiries SET state='open', asked_at=?, expires_at=?, "
                        "next_try_at=?, reason=NULL, code_digest=NULL, updated_at=?, epoch=?, "
                        "question=?, choices=?, topic=?, delivery_state='pending', "
                        "lease_token=NULL, lease_until=NULL, held_by=NULL, proof=NULL, "
                        "correlation=NULL, sent_at=NULL, answered_at=NULL, answer=NULL, "
                        "answer_channel=NULL, settled=NULL WHERE id=?",
                        (asked, expires, asked, asked, self.store.epoch(), text,
                         json.dumps(list(options)), topic, identifier))
                    self.store._audit("inquiry_reopened", identifier,
                                      {"was": existing.state, "decision": decision,
                                       "subject": subject})
                return {**self.get(identifier).as_dict(), "asked": True,
                        "note": f"asked again: it was {existing.state}"}
            # The same question about the same state is the same row, and the code already
            # in the owner's hand is the one that answers it. Re-minting here would leave a
            # sent message quoting a code that no longer works.
            return {**existing.as_dict(), "asked": False,
                    "note": ("this is already asked; the code that went out still answers it"
                             if existing.state in ("open", "sent") else
                             f"this already ended as {existing.state}; asking it again from "
                             "here would undo that")}
        with self._writing():
            self.db.execute(
                "INSERT INTO inquiries(id, topic, decision, subject_id, subject_digest, "
                "question, choices, state, reason, epoch, attempts, asked_at, expires_at, "
                "next_try_at, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?, 'open', ?,?,0,?,?,?,?,?)",
                (identifier, topic, decision, subject, pinned["digest"], text,
                 json.dumps(list(options), ensure_ascii=False),
                 " ".join(str(reason or "").split())[:500] or None,
                 int(self.store.epoch()), asked, expires, asked, asked, asked))
            self.store._audit("inquiry_asked", identifier,
                              {"decision": decision, "subject": subject,
                               "digest": pinned["digest"][:16], "topic": topic,
                               "epoch": int(self.store.epoch())})
        return {**self.get(identifier).as_dict(), "asked": True}

    # -- reading -------------------------------------------------------------

    def get(self, inquiry_id: str) -> Inquiry | None:
        row = self.db.execute("SELECT * FROM inquiries WHERE id=?", (inquiry_id,)).fetchone()
        return Inquiry.from_row(row) if row else None

    def for_subject(self, decision: str, subject_id: str) -> Inquiry | None:
        row = self.db.execute("SELECT * FROM inquiries WHERE decision=? AND subject_id=? "
                              "ORDER BY asked_at DESC, id LIMIT 1",
                              (decision, subject_id)).fetchone()
        return Inquiry.from_row(row) if row else None

    def list(self, *, states: tuple[str, ...] = _STATES, limit: int = 50) -> list[Inquiry]:
        wanted = tuple(item for item in states if item in _STATES)
        if not wanted:
            return []
        rows = self.db.execute(
            "SELECT * FROM inquiries WHERE state IN ({}) ORDER BY asked_at, id "
            "LIMIT ?".format(",".join("?" * len(wanted))),
            (*wanted, _cap(limit))).fetchall()
        return [Inquiry.from_row(row) for row in rows]

    def _pin(self, inquiry: Inquiry) -> dict[str, Any]:
        """What the archive says now about the state the owner was shown."""
        try:
            return fence(self.store, decision=inquiry.decision,
                         subject_id=inquiry.subject_id)
        except EvidenceError as error:
            return {"found": False, "awaiting": False, "digest": "", "reason": str(error)}

    def stands_as_shown(self, inquiry: Inquiry) -> bool:
        """Whether the row behind a question is still the row the owner was told about."""
        pinned = self._pin(inquiry)
        return bool(pinned["found"] and pinned["awaiting"]
                    and pinned["digest"] == inquiry.subject_digest)

    def live_for(self, *, decision: str, subject_id: str) -> Inquiry | None:
        """The question being asked right now about this exact decision, if there is one."""
        row = self.db.execute("SELECT * FROM inquiries WHERE decision=? AND subject_id=? "
                              "AND state IN ('open','sent') ORDER BY state='sent' DESC, "
                              "asked_at LIMIT 1",
                              (str(decision), str(subject_id))).fetchone()
        return Inquiry.from_row(row) if row is not None else None

    def settled_elsewhere(self, *, limit: int = 20) -> list[Inquiry]:
        """Live questions whose subject has already moved out from under them.

        A reading, not a verdict, and deliberately not `_moved`: an answer voids its own
        question on this evidence, while the doctor has to see the same thing without
        erasing it. A `goal-activation` still marked `sent` over a goal that is already
        `active` means somebody decided it through a door that never looked here.
        """
        return [item for item in self.list(states=("open", "sent"), limit=limit)
                if not self.stands_as_shown(item)]

    # -- sending -------------------------------------------------------------

    def send_next(self, *, sink: Callable[[str], Any], destination: str, holder: str,
                  limit: int = 5) -> dict[str, Any]:
        """Ask the owner up to ``limit`` questions, through the transport they named.

        Consent is the reminder's consent: ``decide`` says whether this topic may interrupt
        at all right now. A refusal that means *not yet* holds the question and says when to
        try again; only a refusal that means *not here* voids it.
        """
        reports = []
        for _ in range(max(1, _cap(limit))):
            report = self._send_one(sink=sink, destination=destination, holder=holder)
            reports.append(report)
            if not report.get("attempted"):
                break
        sent = sum(1 for item in reports if item.get("sent"))
        tail = reports[-1].get("reason") if reports else "nothing is being asked"
        return {"ok": all(item.get("ok", True) for item in reports), "sent": sent,
                "reason": (f"{sent} question(s) asked; " if sent else "") + tail,
                "reports": reports}

    def _send_one(self, *, sink: Callable[[str], Any], destination: str,
                  holder: str) -> dict[str, Any]:
        # Admission is committed before I/O. A crash after this point is an unknown
        # send, never an automatically retryable lease timeout.
        with self._writing():
            prepared = self._prepare_send(destination=destination, holder=holder,
                                          profile_home=getattr(sink, "hermes_home", None))
        if "body" not in prepared:
            return prepared
        inquiry = self.get(prepared["inquiry"])
        token, body = prepared["token"], prepared["body"]
        try:
            proof = sink(body)
        except Exception as error:
            proof = {"error": f"{type(error).__name__}: {str(error)[:180]}"}
        explicit = isinstance(proof, dict) and type(proof.get("sent")) is bool
        sent = explicit and proof["sent"] is True
        failed = explicit and proof["sent"] is False
        status = "sent" if sent else "failed" if failed else "uncertain"
        reason = ("the owner was asked" if sent else
                  f"delivery {status}: {str((proof or {}).get('error', 'no explicit acknowledgement'))[:200]}"
                  if isinstance(proof, dict) else "delivery uncertain: no explicit acknowledgement")
        stamp = now()
        with self._writing():
            changed = self.db.execute(
                "UPDATE inquiries SET state=?, delivery_state=?, sent_at=?, updated_at=?, "
                "proof=?, correlation=?, lease_token=NULL, lease_until=NULL, held_by=NULL, "
                "reason=? WHERE id=? AND lease_token=? AND state='open' AND epoch=?",
                ("sent" if sent else "open", "pending" if failed else status,
                 stamp if sent else None, stamp, json.dumps(_proof(proof)),
                 f"hermes-memory question {inquiry.id} {digest(body)[:12]}",
                 None if sent else reason, inquiry.id, token, inquiry.epoch)).rowcount
            # Keep the transport fact even when expiry/revocation superseded its question.
            safe = _proof(proof)
            self.db.execute(
                "UPDATE inquiry_deliveries SET state=?, proof=?, reason=?, platform=?, "
                "chat_id=?, message_id=?, thread_id=?, updated_at=? WHERE id=?",
                (status, json.dumps(safe), reason, safe.get("platform"),
                 str(safe["chat_id"]) if safe.get("chat_id") is not None else None,
                 str(safe["message_id"]) if safe.get("message_id") is not None else None,
                 str(safe["thread_id"]) if safe.get("thread_id") is not None else None,
                 stamp, token))
            self.store._audit("inquiry_delivery", inquiry.id,
                              {"delivery": token, "state": status, "current": bool(changed),
                               "channel": destination, "proof": safe})
        return {"attempted": True, "sent": bool(sent and changed), "ok": bool(sent and changed),
                "inquiry": inquiry.id, "delivery": token, "delivery_state": status,
                "reason": reason if changed else "delivery completed for a superseded question"}

    def _prepare_send(self, *, destination: str, holder: str,
                      profile_home: str | None) -> dict[str, Any]:
        moment = now()
        self.expire_due()
        row = self.db.execute(
            "SELECT * FROM inquiries WHERE state='open' AND delivery_state='pending' "
            "AND lease_token IS NULL AND expires_at>? AND (next_try_at IS NULL OR "
            "next_try_at<=?) ORDER BY asked_at, id LIMIT 1", (moment, moment)).fetchone()
        if row is None:
            return {"attempted": False, "sent": False, "ok": True,
                    "reason": "nothing is being asked"}
        inquiry = Inquiry.from_row(row)
        if inquiry.epoch != self.store.epoch():
            self._void(inquiry.id, reason="the archive epoch changed before delivery")
            return {"attempted": True, "sent": False, "ok": True,
                    "inquiry": inquiry.id, "reason": "question generation revoked"}
        allowed, who = self.replies_allowed()
        if not allowed:
            self._void(inquiry.id, reason=f"a reply settles nothing here: {who}")
            return {"attempted": True, "sent": False, "ok": True, "inquiry": inquiry.id,
                    "reason": who}
        if self._moved(inquiry):
            return {"attempted": True, "sent": False, "ok": True, "inquiry": inquiry.id,
                    "reason": "the archive moved under the preview this asked about"}
        gate = self.policy.decide(topic=inquiry.topic, urgency="proactive", at=moment)
        if gate.action != "notify_owner":
            hold_until = _retry_at(gate, moment)
            self.db.execute("UPDATE inquiries SET next_try_at=?, reason=?, updated_at=? "
                            "WHERE id=?", (hold_until, gate.reason[:500], now(), inquiry.id))
            return {"attempted": False, "sent": False, "ok": True, "inquiry": inquiry.id,
                    "retry_at": hold_until,
                    "reason": f"held, not refused: {gate.reason}"}
        token = new_id("ask")
        code = code_for()
        body = inquiry.body(code, destination=destination)
        with self._writing():
            claimed = self.db.execute(
                "UPDATE inquiries SET lease_token=?, lease_until=?, held_by=?, "
                "attempts=attempts+1, code_digest=?, updated_at=?, delivery_state='sending' "
                "WHERE id=? AND state='open' AND delivery_state='pending' AND lease_token IS NULL",
                (token, _seconds() + 300.0, holder,
                 digest(["inquiry-code", inquiry.id, code]), now(), inquiry.id)).rowcount
            if not int(claimed or 0):
                return {"attempted": False, "sent": False, "ok": True,
                        "reason": "someone else is already asking this one"}
        self.db.execute(
            "INSERT INTO inquiry_deliveries(id,inquiry_id,epoch,destination,profile_home,"
            "body_digest,state,created_at,updated_at) VALUES(?,?,?,?,?,?,'sending',?,?)",
            (token, inquiry.id, inquiry.epoch, destination, profile_home, digest(body), moment, moment))
        return {"inquiry": inquiry.id, "token": token, "body": body}

    # -- answering -----------------------------------------------------------

    def for_native_reply(self, *, platform: str, chat_id: str, message_id: str,
                         thread_id: str, profile_home: str) -> tuple[Inquiry, str] | None:
        """Resolve a persisted send, never a quotation, nearest question or latest session."""
        rows = self.db.execute(
            "SELECT inquiry_id,id,epoch,thread_id,body_digest FROM inquiry_deliveries "
            "WHERE state='sent' AND platform=? AND chat_id=? AND message_id=? "
            "AND profile_home=?",
            (platform, chat_id, message_id, profile_home)).fetchall()
        matches = [row for row in rows if str(row["thread_id"] or "") == thread_id]
        if len(matches) != 1:
            return None
        row = matches[0]
        inquiry = self.get(row["inquiry_id"])
        if inquiry is None or inquiry.epoch != int(row["epoch"]):
            return None
        # Reopening replaces the active delivery even within the same epoch.
        expected = f"hermes-memory question {inquiry.id} {row['body_digest'][:12]}"
        if str(row_value(self.db, "correlation", inquiry.id) or "") != expected:
            return None
        return inquiry, str(row["id"])

    def answer_native(self, *, reply: str, platform: str, chat_id: str,
                      message_id: str, thread_id: str, profile_home: str,
                      transport_home: str, author_id: str) -> dict[str, Any]:
        """Trusted host ingress only; model tools cannot provide this transport attestation.

        Natural short replies require a Telegram owner DM and the same credential home
        that sent the question. Shared-bot satellites fail closed. Irreversible forgetting
        keeps the explicit code confirmation protocol.
        """
        with self._writing():
            if platform != "telegram" or author_id != chat_id or not author_id \
                    or transport_home != profile_home:
                return {"ok": False, "settled": False, "reason": "native reply identity is not an owner DM"}
            matched = self.for_native_reply(platform=platform, chat_id=chat_id,
                                            message_id=message_id, thread_id=thread_id,
                                            profile_home=profile_home)
            if matched is None:
                return {"ok": False, "settled": False, "reason": "no unique current question matches this reply"}
            inquiry, delivery = matched
            if inquiry.decision == "forgetting" and not code_in(reply):
                return {"ok": False, "settled": False, "inquiry": inquiry.id,
                        "reason": "forgetting requires the explicit yes/no code confirmation"}
            parsed = _ANSWER.fullmatch(reply.strip())
            has_code = bool(parsed and parsed.group("code"))
            return self._answer(reply=reply, inquiry_id=inquiry.id,
                                channel=f"{platform}:{chat_id}",
                                native_delivery=delivery if not has_code else None)

    def dossier(self, inquiry: Inquiry) -> dict[str, Any]:
        """Direct, attributed context. No code or ability to authorize is exposed here."""
        preview = None
        if inquiry.decision in FENCES:
            table, keys, _, _ = FENCES[inquiry.decision]
            values = inquiry.subject_id.split("@") if table == "lessons" else [inquiry.subject_id]
            row = self.db.execute(f"SELECT * FROM {table} WHERE " +
                                  " AND ".join(f"{key}=?" for key in keys), values).fetchone()
            if row:
                preview = dict(row)
        else:
            row = self.db.execute("SELECT preview FROM erasure_ledger WHERE id=?",
                                  (inquiry.subject_id,)).fetchone()
            preview = json.loads(row[0]) if row else None
        return {"inquiry": inquiry.id, "question": inquiry.question, "decision": inquiry.decision,
                "subject": inquiry.subject_id, "subject_digest": inquiry.subject_digest,
                "epoch": inquiry.epoch, "state": inquiry.state,
                "delivery_state": row_value(self.db, "delivery_state", inquiry.id),
                "expires_at": inquiry.expires_at, "stands_as_shown": self.stands_as_shown(inquiry),
                "current_subject": preview, "reason": inquiry.reason, "outcome": inquiry.settled,
                "authority": "Framework proposal/context, not a user instruction or authorization"}

    def answer(self, *, reply: str, inquiry_id: str | None = None,
               channel: str | None = None) -> dict[str, Any]:
        # Lock before reading the generation, grants, epoch and subject. Nested owner
        # operations use savepoints; they cannot commit the inquiry's transaction.
        with self._writing():
            return self._answer(reply=reply, inquiry_id=inquiry_id, channel=channel)

    def _answer(self, *, reply: str, inquiry_id: str | None,
                channel: str | None, native_delivery: str | None = None) -> dict[str, Any]:
        """Settle one question with the owner's reply, or say precisely why it did not.

        The code is the authority and it is checked first. No code, a code that is not live,
        a question whose fence moved: in every one of those the reply is recorded as the text
        it is and nothing is decided, because the caller here is a relay that can put any
        word in this field.
        """
        from ..operations.decisions import settle, summarise

        text = " ".join(str(reply or "").split())
        if not text:
            raise EvidenceError("an answer has to be the owner's words")
        named = self.get(inquiry_id) if inquiry_id else None
        code = code_in(text)
        if named is None and code is None:
            return {"settled": False, "ok": True,
                    "reason": "the reply names no question and carries no code, so there is "
                              "nothing for it to decide"}
        inquiry = named if named is not None else self._by_code(code)
        if inquiry is None:
            # A code that belongs to nothing is still an attempt worth one line: a guess that
            # went unre marked is a guess that can be tried again.
            self.store._audit("inquiry_refused", "unknown-code",
                              {"reason": "no question is open under a code like this"})
            return {"settled": False, "ok": True,
                    "reason": f"no question is open under {code}; it was answered, voided, "
                              "or the code is a guess"}
        if code is None and native_delivery is None:
            return {"settled": False, "ok": True, "inquiry": inquiry.id,
                    "reason": "the reply points at a question but carries no code: the code "
                              "is what says this came from the owner rather than from "
                              "somebody who read the listing"}
        allowed, who = self.replies_allowed()
        if not allowed:
            return {"settled": False, "ok": False, "inquiry": inquiry.id,
                    "reason": f"a reply settles nothing here: {who}"}
        if self.store.stage_is_paused("global", "delivery"):
            return {"settled": False, "ok": False, "inquiry": inquiry.id,
                    "reason": "an operator is holding delivery, and an answer that cannot be "
                              "re-asked is not one to take under a hold"}
        if int(self.store.epoch()) != inquiry.epoch:
            self._void(inquiry.id, reason=f"the memory was reset since this was asked "
                                          f"(epoch {inquiry.epoch}, now "
                                          f"{int(self.store.epoch())})")
            return {"settled": False, "ok": False, "inquiry": inquiry.id,
                    "reason": "the archive was reset since this was asked, so the question "
                              "is void and the answer with it"}
        if inquiry.state == "answered":
            return {"settled": False, "ok": True, "inquiry": inquiry.id,
                    "settled_before": inquiry.settled,
                    "reason": "it was already answered, and an owner decision is not taken "
                              "twice"}
        if inquiry.state != "sent":
            return {"settled": False, "ok": False, "inquiry": inquiry.id,
                    "reason": f"the question is {inquiry.state}, so there is nothing live to "
                              "answer"}
        if native_delivery is None and digest(["inquiry-code", inquiry.id, code]) != str(row_value(
                self.db, "code_digest", inquiry.id) or ""):
            self.db.execute("UPDATE inquiries SET attempts=attempts+1, updated_at=?, "
                            "reason=? WHERE id=?",
                            (now(), "a reply arrived with a code that is not this "
                                    "question's", inquiry.id))
            self.store._audit("inquiry_refused", inquiry.id,
                              {"reason": "code mismatch", "decision": inquiry.decision,
                               "subject": inquiry.subject_id})
            return {"settled": False, "ok": False, "inquiry": inquiry.id,
                    "reason": "that code does not belong to this question, and it is written "
                              "down: a wrong code is either a mistyped one or somebody "
                              "guessing"}
        if _parse(inquiry.expires_at) <= _parse(now()):
            self.db.execute("UPDATE inquiries SET state='expired', updated_at=?, reason=? "
                            "WHERE id=?",
                            (now(), "answered after it expired", inquiry.id))
            return {"settled": False, "ok": False, "inquiry": inquiry.id,
                    "reason": "it expired before the answer came; asking again settles it, "
                              "answering the old one does not"}
        wanted = _polarity(text)
        if wanted is None:
            # The code says who sent it; only their words say what to do. A reply that says
            # neither is left open rather than guessed at, because an erasure confirmed on the
            # strength of "hmm about the fern thing" is not a confirmed erasure.
            return {"settled": False, "ok": True, "inquiry": inquiry.id,
                    "reason": "the code is right and the words are neither a yes nor a no, so "
                              "nothing is decided: reply `yes <code>` to settle it, or open "
                              "`hermes-memory owner --list` to decide it at the door"}
        if wanted == "no":
            self.withdraw(inquiry_id=inquiry.id,
                          reason=f"the owner declined it on {channel or 'the owner channel'}")
            self.store._audit("inquiry_declined", inquiry.id,
                              {"decision": inquiry.decision, "subject": inquiry.subject_id,
                               "channel": channel, "said": text[:200]})
            return {"settled": False, "ok": True, "declined": True, "inquiry": inquiry.id,
                    "reason": "the owner said no, which settles nothing: the decision stays as "
                              "open as it was and this question stops being asked about it. "
                              "Refusing it properly is `hermes-memory owner`"}
        if self._moved(inquiry):
            return {"settled": False, "ok": False, "inquiry": inquiry.id,
                    "reason": "the thing it asked about has changed since the owner was "
                              "shown it, so this answer is about a decision that is not open"}
        try:
            outcome = settle(self.store, owner_principal=self.owner_principal,
                             name=inquiry.decision, subject_id=inquiry.subject_id,
                             actor=self.owner_principal or "",
                             reason=f"replied on {channel or 'the owner channel'}: "
                                    f"{text}"[:900],
                             preview_digest=inquiry.subject_digest,
                             via_inquiry=inquiry.id)
        except EvidenceError as error:
            # The owner said yes and the door underneath still refused. That refusal is the
            # point, and the question stays live so the reason can be read and fixed.
            self.db.execute("UPDATE inquiries SET updated_at=?, reason=? WHERE id=?",
                            (now(), f"the decision refused it: {str(error)[:200]}",
                             inquiry.id))
            return {"settled": False, "ok": False, "inquiry": inquiry.id,
                    "reason": f"the answer was good and the door said no: {error}"}
        stamp = now()
        said = summarise(inquiry.decision, outcome)
        self.db.execute(
            "UPDATE inquiries SET state='answered', answered_at=?, updated_at=?, answer=?, "
            "answer_channel=?, settled=?, reason=NULL WHERE id=?",
            (stamp, stamp, text[:2000], channel,
             json.dumps({"outcome": outcome, "said": said}, ensure_ascii=False)[:4000],
             inquiry.id))
        self.store._audit("inquiry_answered", inquiry.id,
                          {"decision": inquiry.decision, "subject": inquiry.subject_id,
                           "channel": channel, "outcome": _proof(outcome)})
        return {"settled": True, "ok": True, "inquiry": inquiry.id,
                "decision": inquiry.decision, "outcome": outcome, "reason": said}

    def _by_code(self, code: str | None) -> Inquiry | None:
        """Which question a code belongs to, live or already spent.

        Solved by re-deriving the digest over the questions that could still be answered —
        the stored value is one-way, so there is nothing to look up by. An answered question
        is included on purpose: a code used twice should hear "that was already answered",
        which is a different fact from "no such code" and the one the owner needs.
        """
        if not code:
            return None
        rows = self.db.execute("SELECT id, code_digest FROM inquiries WHERE state IN "
                               "('sent','answered') ORDER BY sent_at DESC, id").fetchall()
        for row in rows:
            if digest(["inquiry-code", str(row["id"]), code]) == str(row["code_digest"]):
                return self.get(str(row["id"]))
        return None

    def _moved(self, inquiry: Inquiry) -> bool:
        """Void the question if the archive no longer matches what was shown."""
        pinned = self._pin(inquiry)
        if pinned["found"] and pinned["awaiting"] \
                and pinned["digest"] == inquiry.subject_digest:
            return False
        self._void(inquiry.id, reason=f"it no longer stands as it was shown: "
                                      f"{pinned['reason'] or 'the row changed'}")
        return True

    # -- the rest of a question's life ---------------------------------------

    def withdraw(self, *, inquiry_id: str, reason: str) -> dict[str, Any]:
        """Stop asking about this state.

        ``void`` and ``withdrawn`` are different endings: a void row means the question
        outlived the fact it was asked about, so asking again is right, while a withdrawn one
        means somebody chose to stop — and a pass that reopened it would be arguing.
        """
        inquiry = self.get(inquiry_id)
        if inquiry is None:
            raise EvidenceError(f"no question {inquiry_id!r}")
        if inquiry.state not in ("open", "sent"):
            return {**inquiry.as_dict(), "withdrawn": False,
                    "reason": f"it is already {inquiry.state}"}
        self._void(inquiry_id, state="withdrawn",
                   reason=f"withdrawn: {' '.join(str(reason).split())}"[:480])
        return {**self.get(inquiry_id).as_dict(), "withdrawn": True}

    def expire_due(self) -> dict[str, Any]:
        """Close the questions that outlived their own answer window."""
        moment = now()
        rows = self.db.execute("SELECT id FROM inquiries WHERE state IN ('open','sent') "
                               "AND expires_at<=?", (moment,)).fetchall()
        with self._writing():
            for row in rows:
                self.db.execute("UPDATE inquiries SET state='expired', updated_at=?, "
                                "reason=? WHERE id=?",
                                (moment, "the owner did not answer within its window",
                                 row["id"]))
        if rows:
            self.store._audit("inquiry_expired", f"count:{len(rows)}",
                              {"count": len(rows),
                               "ids": [str(row["id"]) for row in rows][:20]})
        return {"expired": len(rows)}

    def _void(self, inquiry_id: str, *, reason: str, state: str = "void") -> None:
        self.db.execute("UPDATE inquiries SET state=?, updated_at=?, reason=? WHERE id=?",
                        (state, now(), reason[:500], inquiry_id))
        self.store._audit("inquiry_void", inquiry_id, {"reason": reason[:200],
                                                       "state": state})


def row_value(db, column: str, inquiry_id: str) -> Any:
    row = db.execute(f"SELECT {column} FROM inquiries WHERE id=?", (inquiry_id,)).fetchone()
    return row[column] if row else None


def _proof(value: Any) -> dict[str, Any]:
    """What the transport said about a question, keeping only the parts that are facts."""
    if isinstance(value, dict):
        return {key: item for key, item in value.items()
                if key in ("sent", "platform", "chat_id", "message_id", "thread_id", "verified",
                           "correlation", "path", "sha256", "mirrored", "session_key",
                           "session_id", "transport_profile")}
    return {"reported": str(value)[:200]} if value else {}


def _retry_at(gate, moment: str) -> str:
    """When a held question may be asked again: the next waking the policy named.

    A gate that downgraded to a digest or held for quiet hours has already worked out the
    moment it would be willing; a cooldown-only refusal says so in its reason and gets a
    plain re-try rather than a guess at its arithmetic.
    """
    named = getattr(gate, "digest_closes_at", None)
    if named:
        return timestamp(str(named))
    return (_parse(moment) + timedelta(minutes=15)).isoformat()


def _parse(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _seconds() -> float:
    return datetime.now(timezone.utc).timestamp()


def _cap(value: int, maximum: int = 25) -> int:
    return value if isinstance(value, int) and 1 <= value <= maximum else 5
