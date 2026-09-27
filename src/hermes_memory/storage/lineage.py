"""C1 lineage: what a piece of evidence turned into, and what replaced it.

The store already holds the edges — ``record_dependencies``, ``record_visibility``,
``derived_citations``, an assertion's own quote — but each caller so far walked them
its own way and stopped at whatever its task needed. That is how an erasure can be
correct about a record and blind to the summary that quotes it. This module is the
one traversal, and it is deliberately read-only: lineage is an account of what
happened, so nothing here may change it.

Two shapes matter. A **chain** is time: a revision superseded by a revision, with the
superseded rows kept because "what did we believe then" is a real question. A
**closure** is consequence: every record reachable through the dependency edges, plus
every derived artifact that quotes any of them. Derived artifacts are terminal — a
summary quotes records and never another summary — so a closure is two levels deep
however far the record graph runs, and that is what makes an impact list something an
owner can actually read before confirming a deletion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..ids import digest
from .evidence import EvidenceError

__all__ = ["Lineage", "CLOSURE_CAP"]

CLOSURE_CAP = 2000
# Every place a record can be leaned on. Adding a derived product means adding it
# here, which is the one place a new artifact type has to be announced.
# A None kind means the table names it itself. The ledger records the artifact's own kind
# (`summary:project`, `lesson`), and the radius report groups by family — the part before
# the colon — because a lesson called a "summary" is a wrong answer in the one preview an
# owner confirms against, while the families are what the cleanup paths act on.
ARTIFACT_SOURCES: tuple[tuple[str | None, str, str, str | None], ...] = (
    (None, "derived_citations", "artifact_id", "kind"),
    ("assertion", "assertions", "id", None),
)


class Lineage:
    def __init__(self, store):
        self.store = store
        self.db = store.db

    # -- revisions and chains ------------------------------------------------

    def revisions(self, source: str, source_id: str, *,
                  include_hidden: bool = True) -> list[dict[str, Any]]:
        """Every revision of one source item, oldest first.

        Corrections are new rows, not edits, so this is the history rather than the
        current state — and the history has to stay answerable after a correction.
        """
        clause = "" if include_hidden else " AND r.deleted=0 AND COALESCE(v.hidden,0)=0"
        rows = self.db.execute(
            f"""
            SELECT r.id, r.revision, r.occurred_at, r.observed_at, r.ingested_at, r.kind,
                   r.deleted, COALESCE(v.hidden, 0) AS hidden, v.replacement_id, v.reason
            FROM records r LEFT JOIN record_visibility v ON v.record_id = r.id
            WHERE r.source=? AND r.source_id=?{clause}
            ORDER BY r.ingested_at, r.revision
            """, (source, source_id)).fetchall()
        return [dict(row) for row in rows]

    def chain(self, record_id: str) -> dict[str, Any]:
        """Where this revision sits in its line: its head, and the way there.

        A visibility row with no replacement is a hide, not a supersession, and the
        two read differently: one says "this was corrected", the other "this was
        withdrawn". Following the wrong one would present withdrawn evidence as
        merely out of date.
        """
        head = record_id
        ancestors: list[str] = []
        current = record_id
        successor = self.db.execute(
            "SELECT replacement_id, reason FROM record_visibility WHERE record_id=? AND "
            "hidden=1 AND replacement_id IS NOT NULL", (current,)).fetchone()
        seen = {current}
        while successor is not None:
            nxt = str(successor["replacement_id"])
            if nxt in seen:
                # A supersession cycle is corrupt data, not a puzzle to solve by
                # walking it: report what was found rather than looping on it.
                return {"id": record_id, "head": current, "ancestors": ancestors,
                        "cycle_at": nxt, "truncated": True}
            seen.add(nxt)
            ancestors.append(nxt)
            current = nxt
            successor = self.db.execute(
                "SELECT replacement_id FROM record_visibility WHERE record_id=? AND "
                "hidden=1 AND replacement_id IS NOT NULL", (current,)).fetchone()
        return {"id": record_id, "head": current, "ancestors": ancestors,
                "replaced_by": ancestors[0] if ancestors else None, "truncated": False}

    def current(self, source: str, source_id: str) -> str | None:
        """The revision that stands today, or nothing if none of them do."""
        rows = self.db.execute(
            """
            SELECT r.id FROM records r LEFT JOIN record_visibility v ON v.record_id = r.id
            WHERE r.source=? AND r.source_id=? AND r.deleted=0
              AND COALESCE(v.hidden, 0)=0
            ORDER BY r.ingested_at DESC, r.revision DESC LIMIT 1
            """, (source, source_id)).fetchone()
        return str(rows["id"]) if rows else None

    # -- consequence ---------------------------------------------------------

    def parents(self, record_id: str) -> list[str]:
        rows = self.db.execute("SELECT parent_id FROM record_dependencies WHERE child_id=?",
                               (record_id,)).fetchall()
        return [str(row["parent_id"]) for row in rows]

    def closure(self, record_ids: Iterable[str], *,
                cap: int = CLOSURE_CAP) -> tuple[list[str], bool]:
        """Every record reachable through the dependency edges, targets included.

        Returns the set and whether the cap was reached, because a truncated
        consequence list has to be said out loud rather than read as complete.

        Cycle-safe and capped: a self-referential edge would otherwise spin, and a
        hub record in a long-lived archive can reach thousands — an impact list that
        says "and everything" tells the owner nothing they can consent to.
        """
        seen: set[str] = set()
        frontier = [str(item) for item in record_ids if item]
        truncated = False
        while frontier:
            if len(seen) >= cap:
                truncated = True
                break
            batch = [item for item in frontier if item not in seen][:cap]
            if len(batch) < len(frontier):
                truncated = True
            if not batch:
                break
            seen.update(batch)
            placeholders = ",".join("?" * len(batch))
            rows = self.db.execute(
                f"SELECT child_id FROM record_dependencies WHERE parent_id IN "
                f"({placeholders})", batch).fetchall()
            frontier = [str(row["child_id"]) for row in rows]
        return sorted(seen), truncated

    def artifacts(self, record_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Derived products that quote any of these records, directly."""
        targets = [str(item) for item in record_ids if item]
        if not targets:
            return []
        placeholders = ",".join("?" * len(targets))
        found: list[dict[str, Any]] = []
        for kind, table, column, kind_column in ARTIFACT_SOURCES:
            selected = (f"{column} AS artifact" if kind_column is None
                        else f"{column} AS artifact, {kind_column} AS kind")
            rows = self.db.execute(
                f"SELECT DISTINCT {selected} FROM {table} WHERE record_id IN "
                f"({placeholders})", targets).fetchall()
            found.extend({"kind": kind if kind is not None else
                          str(row[kind_column]).partition(":")[0],
                          "id": str(row["artifact"])} for row in rows)
        return sorted(found, key=lambda item: (item["kind"], item["id"]))

    def citations_of(self, artifact_id: str) -> list[dict[str, Any]]:
        """What a derived artifact was built from, with each source's live state.

        The point of reading it this way rather than from a stored verdict is that
        the answer changes when the evidence does: an erased record shows up here as
        erased next to the summary that still quotes it.
        """
        rows = self.db.execute(
            """
            SELECT c.record_id, c.kind, c.coverage, c.quote, c.quote_start, c.quote_end,
                   r.deleted, COALESCE(v.hidden, 0) AS hidden, c.added_at
            FROM derived_citations c
            LEFT JOIN records r ON r.id = c.record_id
            LEFT JOIN record_visibility v ON v.record_id = c.record_id
            WHERE c.artifact_id=? ORDER BY c.added_at, c.record_id
            """, (artifact_id,)).fetchall()
        out = []
        for row in rows:
            entry = dict(row)
            entry["live"] = not (row["deleted"] or row["hidden"])
            out.append(entry)
        return out

    # -- explanation and integrity -------------------------------------------

    def explain(self, record_id: str, *, accounts: Sequence[str] | None = None) -> dict[str, Any]:
        """Why this record is or is not on the table, in terms a status report can show."""
        row = self.db.execute(
            """
            SELECT r.id, r.source, r.source_id, r.revision, r.kind, r.occurred_at,
                   r.occurred_precision, r.observed_at, r.ingested_at, r.deleted,
                   r.metadata, COALESCE(v.hidden, 0) AS hidden, v.replacement_id,
                   v.reason AS visibility_reason
            FROM records r LEFT JOIN record_visibility v ON v.record_id = r.id
            WHERE r.id=?
            """, (record_id,)).fetchone()
        if row is None:
            tombstone = self.db.execute(
                "SELECT intent_id, deleted_at FROM tombstones WHERE record_id=?",
                (record_id,)).fetchone()
            return {"id": record_id, "present": False,
                    "state": "erased" if tombstone else "unknown",
                    "reason": ("forgotten by erasure intent "
                               + str(tombstone["intent_id"]) if tombstone else
                               "this store never held that id"),
                    "retrievable": False}
        state = ("erased" if row["deleted"] else
                 "hidden" if row["hidden"] else "visible")
        reason = row["visibility_reason"]
        if row["deleted"]:
            # An erasure writes a tombstone rather than a visibility row, so the two
            # have to be read together or "why is this gone" has no answer for the
            # commonest way something goes.
            tombstone = self.db.execute(
                "SELECT intent_id, deleted_at FROM tombstones WHERE record_id=?",
                (record_id,)).fetchone()
            if tombstone is not None:
                reason = (f"erased by intent {tombstone['intent_id']} at "
                          f"{tombstone['deleted_at']}")
        dependents = self.db.execute(
            "SELECT child_id FROM record_dependencies WHERE parent_id=?",
            (record_id,)).fetchall()
        return {"id": record_id, "present": True, "state": state,
                "source": row["source"], "source_id": row["source_id"],
                "revision": row["revision"], "kind": row["kind"],
                "occurred_at": row["occurred_at"],
                "occurred_precision": row["occurred_precision"],
                "ingested_at": row["ingested_at"],
                "reason": reason,
                "replaced_by": row["replacement_id"],
                "superseded_by": self.chain(record_id).get("replaced_by"),
                "children": [str(item["child_id"]) for item in dependents],
                "quoted_by": self.artifacts([record_id]),
                "accounts": list(accounts if accounts is not None
                                 else self._accounts(record_id)),
                "retrievable": state == "visible"}

    def dangling(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Edges that point at nothing — the trace of a restore or a bad erase.

        A dependency row whose parent was erased is normal; one whose parent never
        existed means something was written outside the transaction that should have
        owned it, and that is worth reporting rather than quietly pruning.
        """
        rows = self.db.execute(
            """
            SELECT 'dependency' AS edge, d.child_id AS from_id, d.parent_id AS to_id,
                   'record_dependencies' AS table_name
            FROM record_dependencies d LEFT JOIN records p ON p.id = d.parent_id
            WHERE p.id IS NULL
            UNION ALL
            SELECT 'citation', c.artifact_id, c.record_id, 'derived_citations'
            FROM derived_citations c LEFT JOIN records r ON r.id = c.record_id
            WHERE r.id IS NULL
            UNION ALL
            SELECT 'attachment', a.id, a.record_id, 'attachments'
            FROM attachments a LEFT JOIN records r ON r.id = a.record_id
            WHERE r.id IS NULL
            LIMIT ?
            """, (_bounded(limit),)).fetchall()
        return [dict(row) for row in rows]

    def orphans(self, *, limit: int = 50) -> list[str]:
        """Visible records nothing and nobody can reach: captured, indexed, unlinked.

        Not an error by itself — most evidence is a leaf — but a large count usually
        means a connector committed without its participants, which is a source
        contract problem the doctor should surface rather than a data mystery.
        """
        rows = self.db.execute(
            """
            SELECT r.id FROM records r
            LEFT JOIN record_visibility v ON v.record_id = r.id
            LEFT JOIN record_dependencies d ON d.child_id = r.id
            WHERE r.deleted=0 AND COALESCE(v.hidden,0)=0 AND d.child_id IS NULL
            ORDER BY r.ingested_at, r.id LIMIT ?
            """, (_bounded(limit),)).fetchall()
        return [str(row["id"]) for row in rows]

    def _accounts(self, record_id: str) -> list[str]:
        """Which accounts this record claims to belong to, resolved through identity.

        Reported for explanation only. Authorisation is the broker's decision on
        every read; a lineage row that named an account is not a grant.
        """
        from .identity import IdentityStore, evidence_accounts
        evidence = self.store.get(record_id, include_hidden=True)
        if evidence is None:
            return []
        return sorted(evidence_accounts(IdentityStore(self.store), evidence))


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= 500 else 50
