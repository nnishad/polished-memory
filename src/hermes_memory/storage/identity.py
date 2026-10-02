"""C7 — identity accounts, deterministic candidates and owner-only confirmation.

The authority split is the whole point: an agent may *propose*, because it can
see a thread with two addresses in it. Only the owner may *confirm*, because a
wrong join silently merges two people's evidence and every later retrieval
inherits the mistake. Confirmation is therefore unreachable from any
agent-role credential, and a proposal records the rule and evidence that
produced it so the decision can be judged rather than trusted.
"""
from __future__ import annotations

from ..storage.transactions import write_transaction

import json
import re
from typing import Any, Sequence

from ..ids import digest, intervals_overlap, now, record_pk, timestamp
from .evidence import EvidenceError

__all__ = ["IdentityStore", "evidence_accounts", "in_scope",
           "citations_in_scope", "namespace_of", "normalize_account", "parse_account",
           "PENDING", "CONFIRMED", "REJECTED", "STALE", "RULES"]

PENDING = "pending"
CONFIRMED = "confirmed"
REJECTED = "rejected"
STALE = "stale"

# Rules that may produce a candidate. Each is deterministic: it fires on exact
# structural agreement, never on similarity. A rule absent from this table
# cannot propose anything, which is how "a model suggested these are the same
# person" is kept out of canonical identity.
RULES = {
    "email-thread-participant": "1",
    "email-normalized-equal": "1",
    "phone-e164-equal": "1",
    "explicit-alias-declared": "1",
    "source-account-self": "1",
}

# Explicitly refused: these look like evidence and are not.
_REFUSED_RULES = {
    "display-name-match": "a shared display name is not a shared identity",
    "co-occurrence": "appearing together is not being the same person",
    "model-suggested": "a model may describe an association; it may not persist one",
    "fuzzy-name": "name similarity is not structural agreement",
}

_EMAIL = re.compile(r"^(?P<local>[^@]+)@(?P<domain>[^@]+)$")
_PHONE_ALLOWED = re.compile(r"^[+\d][\d\s\-().]*$")
#: Marks a value that is somebody's serialization of an account rather than an address —
#: a JSON object, a Python repr, a display form. No real address contains any of these.
_SERIALIZED = re.compile(r"[{}[\]<>\"'\\]")


class IdentityStore:
    def __init__(self, store, *, owner_principal: str | None = None):
        self.store = store
        self.db = store.db
        self.owner_principal = owner_principal

    # -- accounts ------------------------------------------------------------

    def account(self, namespace: str, identifier: str, *, label: str | None = None) -> str:
        """Register or fetch an account. Identifiers are case- and form-faithful."""
        namespace = _check_namespace(namespace)
        kept, normalized = normalize_account(namespace, identifier)
        account_id = "acct_" + digest([namespace, normalized])[:32]
        with write_transaction(self.db):
            self.db.execute(
                "INSERT INTO identity_accounts(id, namespace, identifier, normalized, label, "
                "state, created_at) VALUES(?,?,?,?,?, 'active', ?) "
                "ON CONFLICT(namespace, normalized) DO UPDATE SET "
                "label=COALESCE(excluded.label, identity_accounts.label)",
                (account_id, namespace, kept, normalized, label, now()),
            )
        return account_id

    def get_account(self, account_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM identity_accounts WHERE id=?", (account_id,)).fetchone()
        return dict(row) if row else None

    def resolve(self, namespace: str, identifier: str) -> str | None:
        """Look an address up without creating it.

        Registration is a deliberate act; resolving a stranger's address out of
        a message must never mint an account behind the owner's back.
        """
        namespace = _check_namespace(namespace)
        _, normalized = normalize_account(namespace, identifier)
        row = self.db.execute(
            "SELECT id FROM identity_accounts WHERE namespace=? AND normalized=? "
            "AND state='active'", (namespace, normalized)).fetchone()
        return row["id"] if row else None

    # -- candidates ----------------------------------------------------------

    def propose(self, *, account_a: str, account_b: str, rule: str, basis: str,
                evidence: Sequence[str], proposed_by: str, proposed_kind: str = "agent",
                rule_version: str | None = None) -> dict[str, Any]:
        """Open a candidate join. Deterministic rules only; one candidate per pair.

        A second proposal about accounts the owner already joined — directly or through
        someone else — is not queued: there is nothing left for them to decide, and the
        list they read is the thing worth protecting here.
        """
        if rule in _REFUSED_RULES:
            raise EvidenceError(f"rule {rule!r} cannot propose an identity: {_REFUSED_RULES[rule]}")
        if rule not in RULES:
            raise EvidenceError(
                f"unknown identity rule {rule!r}; admissible rules are {sorted(RULES)}")
        version = rule_version or RULES[rule]
        if version != RULES[rule]:
            raise EvidenceError(f"rule {rule!r} is at version {RULES[rule]}, not {version}")
        pair = _ordered_pair(account_a, account_b)
        if pair[0] == pair[1]:
            raise EvidenceError("an account cannot be a candidate for itself")
        for account_id in pair:
            if self.get_account(account_id) is None:
                raise EvidenceError(f"unknown account {account_id!r}")
        if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
            raise EvidenceError("evidence must be a list of record ids")
        if not evidence or len(evidence) > 200:
            raise EvidenceError("a candidate cites between 1 and 200 evidence records")
        for citation in evidence:
            if not isinstance(citation, str) or not citation.startswith("rec_"):
                raise EvidenceError(f"evidence must be canonical record ids, got {citation!r}")
            if not self.store.live_and_visible(citation):
                raise EvidenceError(
                    f"evidence record {citation!r} does not resolve to live evidence; a "
                    "candidate must be supportable by something retrievable")
        if not isinstance(basis, str) or not basis.strip() or len(basis) > 1000:
            raise EvidenceError("basis must be nonempty text of at most 1000 characters")

        if self.same_person(pair[0], pair[1]):
            # The owner has already decided these are one person, so there is no
            # question to put in front of them. A candidate row here would sit on the
            # list the owner reads, asking to confirm what is already true.
            return {"candidate_id": None, "state": CONFIRMED, "reopened": False,
                    "already_joined": True,
                    "note": "these accounts are already one person by a confirmed decision"}

        # The pair, not the pair and the rule: the schema allows one candidate per pair,
        # so a second rule pointing at the same two accounts reopens the same row
        # rather than colliding with it.
        candidate_id = "cand_" + digest([pair[0], pair[1]])[:32]
        with write_transaction(self.db):
            existing = self.db.execute(
                "SELECT state FROM identity_candidates WHERE id=?", (candidate_id,)).fetchone()
            if existing:
                # A rejected candidate is durable: re-proposing must not quietly
                # revive a decision the owner already made.
                if existing["state"] in (REJECTED, CONFIRMED):
                    return {"candidate_id": candidate_id, "state": existing["state"],
                            "reopened": False}
                self.db.execute(
                    "UPDATE identity_candidates SET rule=?, rule_version=?, basis=?, "
                    "evidence=?, proposed_by=?, proposed_kind=?, proposed_at=? WHERE id=?",
                    (rule, version, basis, json.dumps(list(evidence), sort_keys=True),
                     proposed_by, proposed_kind, now(), candidate_id),
                )
                return {"candidate_id": candidate_id, "state": PENDING, "reopened": False}
            self.db.execute(
                "INSERT INTO identity_candidates(id, account_a, account_b, rule, rule_version, "
                "basis, evidence, proposed_by, proposed_kind, proposed_at, state) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (candidate_id, pair[0], pair[1], rule, version, basis,
                 json.dumps(list(evidence), sort_keys=True), proposed_by, proposed_kind,
                 now(), PENDING),
            )
            self.store._audit("identity_propose", candidate_id,
                              {"rule": rule, "proposed_by": proposed_by,
                               "proposed_kind": proposed_kind, "evidence": len(evidence)})
        return {"candidate_id": candidate_id, "state": PENDING, "reopened": True,
                "rule": rule, "rule_version": version}

    def pending(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or not 1 <= limit <= 200:
            raise EvidenceError("limit must be between 1 and 200")
        rows = self.db.execute(
            """
            SELECT c.*, a.identifier AS identifier_a, b.identifier AS identifier_b
            FROM identity_candidates c
            JOIN identity_accounts a ON a.id=c.account_a
            JOIN identity_accounts b ON b.id=c.account_b
            WHERE c.state=? ORDER BY c.proposed_at, c.id LIMIT ?
            """, (PENDING, limit)).fetchall()
        return [dict(row) for row in rows]

    # -- decisions -----------------------------------------------------------

    def confirm(self, *, candidate_id: str, actor: str, reason: str,
                valid_from: str | None = None, valid_until: str | None = None) -> dict[str, Any]:
        """Owner-only. Creates the canonical edge and refuses overlapping claims."""
        self._require_owner(actor)
        _check_reason(reason)
        with write_transaction(self.db):
            candidate = self._candidate_or_raise(candidate_id)
            if candidate["state"] == CONFIRMED:
                return {"state": CONFIRMED, "edge_id": None,
                        "note": "already confirmed; no second edge was created"}
            if candidate["state"] == REJECTED:
                raise EvidenceError(
                    f"candidate {candidate_id!r} was rejected by {candidate['decided_by']!r}; "
                    "re-proposing evidence is required before it can be confirmed")
            if candidate["state"] == STALE:
                raise EvidenceError(
                    f"candidate {candidate_id!r} lost its supporting evidence; re-propose it")
            self._refuse_conflicting(candidate["account_a"], candidate["account_b"],
                                    valid_from, valid_until)
            start = timestamp(valid_from) if valid_from else None
            end = timestamp(valid_until) if valid_until else None
            if start and end and end < start:
                raise EvidenceError("valid_until precedes valid_from")
            edge_id = "iedge_" + digest([candidate_id, start, end])[:32]
            self.db.execute(
                "INSERT INTO identity_edges(id, account_a, account_b, candidate_id, valid_from, "
                "valid_until, state, confirmed_by, confirmed_at) VALUES(?,?,?,?,?,?,'active',?,?)",
                (edge_id, candidate["account_a"], candidate["account_b"], candidate_id,
                 start, end, actor, now()),
            )
            self.db.execute(
                "UPDATE identity_candidates SET state=?, decided_by=?, decided_at=?, "
                "decision_reason=? WHERE id=?",
                (CONFIRMED, actor, now(), reason, candidate_id),
            )
            self.store._audit("identity_confirm", edge_id,
                              {"actor": actor, "candidate": candidate_id})
        return {"state": CONFIRMED, "edge_id": edge_id}

    def reject(self, *, candidate_id: str, actor: str, reason: str) -> dict[str, Any]:
        """Owner-only, and durable: a rejection survives later re-proposals."""
        self._require_owner(actor)
        _check_reason(reason)
        with write_transaction(self.db):
            candidate = self._candidate_or_raise(candidate_id)
            if candidate["state"] == CONFIRMED:
                raise EvidenceError(
                    "a confirmed identity cannot be rejected; revoke the edge instead, so the "
                    "record shows it was once believed")
            self.db.execute(
                "UPDATE identity_candidates SET state=?, decided_by=?, decided_at=?, "
                "decision_reason=? WHERE id=?",
                (REJECTED, actor, now(), reason, candidate_id),
            )
            self.store._audit("identity_reject", candidate_id, {"actor": actor})
        return {"state": REJECTED, "candidate_id": candidate_id}

    def revoke(self, *, edge_id: str, actor: str, reason: str) -> dict[str, Any]:
        self._require_owner(actor)
        _check_reason(reason)
        with write_transaction(self.db):
            edge = self.db.execute(
                "SELECT * FROM identity_edges WHERE id=?", (edge_id,)).fetchone()
            if edge is None:
                raise EvidenceError(f"unknown identity edge {edge_id!r}")
            if edge["state"] != "active":
                return {"state": edge["state"], "note": "already revoked"}
            self.db.execute(
                "UPDATE identity_edges SET state='revoked', revoked_at=?, revoked_by=?, "
                "revocation_reason=? WHERE id=?",
                (now(), actor, reason, edge_id),
            )
            self.store._audit("identity_revoke", edge_id, {"actor": actor})
        return {"state": "revoked", "edge_id": edge_id}

    def _require_owner(self, actor: str) -> None:
        if self.owner_principal is None:
            raise EvidenceError(
                "no owner principal is configured, so identity cannot be confirmed or "
                "revoked; set HERMES_MEMORY_OWNER_PRINCIPAL")
        if actor != self.owner_principal:
            raise EvidenceError(
                f"identity decisions require the owner principal, not {actor!r}")

    def _candidate_or_raise(self, candidate_id: str):
        row = self.db.execute(
            "SELECT * FROM identity_candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            raise EvidenceError(f"unknown candidate {candidate_id!r}")
        return row

    def _refuse_conflicting(self, account_a: str, account_b: str,
                            valid_from: str | None, valid_until: str | None) -> None:
        """Refuse a confirmation that contradicts a decision the owner already made.

        Merging groups is normal — that is what transitive identity means. What
        is not acceptable is silently uniting two accounts the owner explicitly
        declared to be different people, or stacking a second active edge over
        the same pair for the same interval.
        """
        duplicate = self.db.execute(
            "SELECT id FROM identity_edges WHERE state='active' "
            "AND ((account_a=? AND account_b=?) OR (account_a=? AND account_b=?))",
            (account_a, account_b, account_b, account_a)).fetchone()
        if duplicate and _intervals_overlap(
                *(self.db.execute("SELECT valid_from, valid_until FROM identity_edges WHERE id=?",
                                  (duplicate["id"],)).fetchone()), valid_from, valid_until):
            raise EvidenceError(
                f"an active edge already links this pair over an overlapping interval "
                f"({duplicate['id']}); revoke it before confirming another")

        group_a, group_b = self._group(account_a, None), self._group(account_b, None)
        placeholders_a = ",".join("?" * len(group_a))
        placeholders_b = ",".join("?" * len(group_b))
        rejected = self.db.execute(
            f"SELECT account_a, account_b FROM identity_candidates WHERE state=? "
            f"AND ((account_a IN ({placeholders_a}) AND account_b IN ({placeholders_b})) "
            f"OR (account_a IN ({placeholders_b}) AND account_b IN ({placeholders_a})))",
            [REJECTED] + group_a + group_b + group_b + group_a).fetchone()
        if rejected:
            raise EvidenceError(
                "the owner already rejected a join between these two identities; confirming "
                "this edge would merge them anyway, so that rejection must be revisited first")

    # -- queries -------------------------------------------------------------

    def group(self, account_id: str, *, at: str | None = None) -> list[str]:
        """The accounts one person, as confirmed for a moment (by default, now).

        The validity interval is honoured and not merely the ``active`` state. An
        edge confirmed "these were the same person through 2024" that keeps joining
        after 2025 has turned a time-limited decision into a permanent one, and
        every later read inherits the widening.
        """
        return self._group(account_id, _moment(at))

    def same_person(self, account_a: str, account_b: str, *, at: str | None = None) -> bool:
        """Confirmed, active, time-valid edges only. Nothing else joins accounts."""
        return account_b in self.group(account_a, at=at)

    def _group(self, account_id: str, at: str | None) -> list[str]:
        """One walk, shared by the set form and the pairwise form.

        ``at=None`` ignores the calendar. Only the conflict check may do that:
        merging two groups the owner separated is the thing they decided against,
        however the validity intervals happen to fall.
        """
        seen = {account_id}
        frontier = [account_id]
        while frontier:
            placeholders = ",".join("?" * len(frontier))
            rows = self.db.execute(
                f"SELECT account_a, account_b, valid_from, valid_until FROM identity_edges "
                f"WHERE state='active' AND (account_a IN ({placeholders}) "
                f"OR account_b IN ({placeholders}))",
                frontier + frontier).fetchall()
            frontier = []
            for row in rows:
                if not _interval_contains(row["valid_from"], row["valid_until"], at):
                    continue
                for candidate in (row["account_a"], row["account_b"]):
                    if candidate not in seen:
                        seen.add(candidate)
                        frontier.append(candidate)
        return sorted(seen)

    def invalidate_stale(self) -> dict[str, Any]:
        """Drop candidates whose cited evidence has been forgotten or hidden.

        An edge confirmed on evidence that no longer exists is a claim nobody
        can re-check, so confirmed edges are reported for owner review rather
        than silently revoked.
        """
        rows = self.db.execute(
            "SELECT id, evidence, state FROM identity_candidates WHERE state IN (?,?)",
            (PENDING, CONFIRMED)).fetchall()
        stale, needs_review = [], []
        with write_transaction(self.db):
            for row in rows:
                cited = json.loads(row["evidence"])
                alive = [rid for rid in cited
                         if self.store.live_and_visible(rid)]
                if alive:
                    continue
                if row["state"] == PENDING:
                    self.db.execute(
                        "UPDATE identity_candidates SET state=?, decided_at=?, "
                        "decision_reason='all cited evidence is gone' WHERE id=?",
                        (STALE, now(), row["id"]))
                    stale.append(row["id"])
                else:
                    needs_review.append(row["id"])
            if stale or needs_review:
                # Only a change is an event. The background pass calls this every period, and a
                # ledger that records a sweep of nothing fills with rows that look like
                # decisions and say none.
                self.store._audit("identity_invalidate", "identity_candidates",
                                  {"stale": len(stale), "needs_review": len(needs_review)})
        return {"stale": stale, "confirmed_needing_review": needs_review}

    # -- topics --------------------------------------------------------------

    def link_topic(self, *, account_id: str, topic: str, kind: str = "structural",
                   source_record_id: str | None = None) -> None:
        """Structural topics may feed rules; labels may only retrieve candidates."""
        if kind not in {"structural", "label"}:
            raise EvidenceError("topic kind must be 'structural' or 'label'")
        if not isinstance(topic, str) or not topic.strip() or len(topic) > 500:
            raise EvidenceError("topic must be nonempty text of at most 500 characters")
        if self.get_account(account_id) is None:
            raise EvidenceError(f"unknown account {account_id!r}")
        with write_transaction(self.db):
            self.db.execute(
                "INSERT INTO topic_links(id, account_id, topic, kind, source_record_id, "
                "observed_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(account_id, topic, kind) DO UPDATE SET "
                "source_record_id=COALESCE(excluded.source_record_id, topic_links.source_record_id), "
                "observed_at=excluded.observed_at",
                ("topic_" + digest([account_id, topic, kind])[:32], account_id, topic.strip(),
                 kind, source_record_id, now()),
            )

    def accounts_for_topic(self, topic: str, *, kind: str = "structural") -> list[str]:
        rows = self.db.execute(
            "SELECT account_id FROM topic_links WHERE topic=? AND kind=? ORDER BY account_id",
            (topic, kind)).fetchall()
        return [row["account_id"] for row in rows]


def normalize_account(namespace: str, identifier: str) -> tuple[str, str]:
    """Return (identifier as written, normalized form used for matching).

    Email local parts keep their case because some providers treat it as
    significant, while the domain is folded — the reverse of what naive
    lowercasing does. Phone numbers keep an explicit international form.
    """
    if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 500:
        raise EvidenceError("identifier must be nonempty text of at most 500 characters")
    value = identifier.strip()
    if _SERIALIZED.search(value) or (namespace != "phone" and re.search(r"\s", value)):
        raise EvidenceError(
            f"{value!r} is not an account address; pass the address itself — one unbroken "
            "token like priya@example.com, not a display name, an object or a serialized blob")
    if namespace == "email":
        match = _EMAIL.match(value)
        if not match:
            raise EvidenceError(f"{value!r} is not an email address")
        return value, f"{match['local']}@{match['domain'].lower()}"
    if namespace == "phone":
        if not _PHONE_ALLOWED.match(value):
            raise EvidenceError(f"{value!r} is not a phone number")
        digits = re.sub(r"[\s\-().]", "", value)
        if not digits.startswith("+"):
            raise EvidenceError(
                f"phone {value!r} has no explicit international prefix; guessing a country "
                "code would merge accounts across countries")
        return value, digits
    return value, value.lower()


def parse_account(value: Any) -> tuple[str, str]:
    """Take (namespace, address) out of whatever a caller handed over.

    Two shapes are admissible because they are the two this system already speaks: a
    bare address, and the namespace-and-address pair a record's participants carry —
    which is the form a model quotes back when it proposes an identity from something
    it recalled. Anything else is refused here rather than stringified: a serialized
    object used to be stored as an account's address, so the same person proposed in
    two shapes never joined and a normalized-equal rule could not fire on them.
    """
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        unknown = sorted(keys - {"namespace", "address"})
        if unknown:
            raise EvidenceError(
                f"unknown account field(s) {unknown}; an account object carries a namespace "
                "and an address")
        missing = sorted(key for key in ("namespace", "address")
                         if not str(value.get(key) or "").strip())
        if missing:
            raise EvidenceError(
                f"an account object is missing {missing}; say which namespace the address is "
                "in rather than leaving it out")
        return _check_namespace(str(value["namespace"])), str(value["address"]).strip()
    if not isinstance(value, str):
        raise EvidenceError(
            f"an account is a bare address or a namespace-and-address pair, not "
            f"{type(value).__name__}")
    text = value.strip()
    if not text:
        raise EvidenceError("an account address must not be empty")
    return namespace_of(text), text


def namespace_of(value: str) -> str:
    """Classify an address written on its own.

    A wrong guess merges account kinds — a handle holding an `@` becoming an email — so
    a caller that knows the namespace says so in the account object and this is only the
    fallback for one that does not.
    """
    if "@" in value:
        return "email"
    if value.startswith("+") and any(char.isdigit() for char in value):
        return "phone"
    return "handle"


def _check_namespace(value: str) -> str:
    if value not in {"email", "phone", "handle", "profile", "source_account"}:
        raise EvidenceError(
            f"unknown identity namespace {value!r}; admissible are email, phone, handle, "
            "profile, source_account")
    return value


def _check_reason(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise EvidenceError("reason must be nonempty text of at most 1000 characters")
    return value


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _intervals_overlap(a_from, a_until, b_from, b_until) -> bool:
    return intervals_overlap(a_from, a_until, b_from, b_until)


def _interval_contains(start, end, at) -> bool:
    if start and at and at < start:
        return False
    if end and at and at > end:
        return False
    return True


def _moment(at: str | None) -> str:
    """A query moment in the form the stored intervals are written in.

    A naive or unparseable stamp is refused rather than compared as text: "2024-06-01"
    sorts below every stored timestamp of that year, so a loose guess would silently
    answer as if it were asked at the very start of the interval.
    """
    if at is None:
        return now()
    try:
        return timestamp(at)
    except ValueError as error:
        raise EvidenceError(
            f"a query moment must be a timezone-aware timestamp: {error}") from error


def in_scope(claims: set[str], identifiers: set[str]) -> bool:
    """The one rule every account-scoped read applies to a set of claims.

    Evidence that names nobody is shared, because quarantining unscoped notes would
    empty the archive without protecting anyone in it. Evidence that names someone is
    shown only to that someone's confirmed group — and a caller the installation knows
    nothing about (an empty group) therefore sees the unscoped part of the archive.
    """
    return not claims or bool(identifiers and claims & identifiers)


def citations_in_scope(store, identity: "IdentityStore | None", citations, *,
                       identifiers: set[str]) -> bool:
    """May this caller be shown everything a derived artifact is standing on?

    A lesson cites where it came from and the record says who it is about, so the
    scope is read from the store rather than stored with the lesson: a join the owner
    confirms later widens what a lesson may be taught to, and one they revoke narrows
    it, without anyone editing the lesson. Each citation is judged by ``in_scope`` on
    its own, which is the same judgment the record would face if it were retrieved
    directly — a lesson must not be able to carry a span across a boundary the span
    could not cross by itself.
    """
    for citation in citations or ():
        record = store.get(record_pk(str(citation)))
        if record is None:
            # Gone evidence is not this check's business: whether a lesson still has
            # its support is decided where support is checked.
            continue
        if not in_scope(evidence_accounts(identity, record), identifiers):
            return False
    return True


def evidence_accounts(identity: "IdentityStore | None", evidence) -> set[str]:
    """Which accounts a record says it belongs to, resolved through identity only.

    An address that is not a registered account scopes nothing. That direction
    matters: unregistered participants are ordinary noise in imported mail, and
    letting noise quarantine a record would empty the archive without protecting
    anyone in it.
    """
    claims = {str(value) for value in (evidence.metadata.get("account_ids") or [])
              if str(value).strip()}
    if identity is None:
        return claims
    for participant in evidence.metadata.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        namespace, address = participant.get("namespace"), participant.get("address")
        if not namespace or not address:
            continue
        try:
            resolved = identity.resolve(str(namespace), str(address))
        except EvidenceError:
            # A stranger's address in someone else's message is not a bug here,
            # and it must never become an argument for showing more.
            resolved = None
        if resolved:
            claims.add(resolved)
    return claims
