"""C3 — connector leases, fences, cursors and consumer replay."""
from __future__ import annotations


import time
import uuid
from typing import Any, Callable, Sequence

from ..ids import digest, now
from ..storage.evidence import EvidenceError, prepare_envelope

__all__ = ["Fence", "StaleFence", "SyncController", "STAGES", "DOWNSTREAM_STAGES"]

# Pause naming is explicit: a normal pause leaves capture running and stops the
# stages that would spend compute or speak to anyone. Stopping ingestion too has
# to be asked for by name, because an operator who says "pause this source"
# almost never means "lose what arrives while it is paused".
DOWNSTREAM_STAGES = ("formation", "proactivity")
STAGES = ("capture",) + DOWNSTREAM_STAGES


class StaleFence(EvidenceError):
    """The writer's authority expired. Its output must not be committed."""


class Fence:
    """Proof that a connector may still write, checked inside the write transaction."""

    __slots__ = ("source", "generation", "epoch", "lease", "holder", "_clock")

    def __init__(self, source: str, generation: int, epoch: int, lease: str, holder: str,
                 clock: Callable[[], float] = time.time):
        self.source = source
        self.generation = generation
        self.epoch = epoch
        self.lease = lease
        self.holder = holder
        # The fence carries the clock that issued its lease. Validity is a claim
        # about a specific clock reading; falling back to wall time here would
        # make a lease issued by a test clock expire the instant it was minted.
        self._clock = clock

    def __repr__(self) -> str:
        return f"Fence(source={self.source!r}, generation={self.generation}, epoch={self.epoch})"

    def validate(self, db, *, at: float | None = None) -> None:
        row = db.execute(
            "SELECT generation, lease, lease_until FROM connectors WHERE source=?",
            (self.source,),
        ).fetchone()
        if row is None:
            raise StaleFence(f"source {self.source!r} has no connector registration")
        if row["generation"] != self.generation:
            raise StaleFence(
                f"source {self.source!r} was reconfigured (generation "
                f"{self.generation} -> {row['generation']}); restart the connector"
            )
        epoch = db.execute("SELECT value FROM memory_epoch WHERE id=1").fetchone()[0]
        if epoch != self.epoch:
            raise StaleFence(
                f"memory epoch advanced {self.epoch} -> {epoch}; data fetched under the "
                "old epoch is not authoritative"
            )
        # Ordered after the epoch check: a reset also clears leases, and
        # "another holder took it" would send an operator looking for a
        # connector that does not exist.
        if row["lease"] is None:
            raise StaleFence(f"source {self.source!r} lease was released")
        if row["lease"] != self.lease:
            raise StaleFence(f"source {self.source!r} lease was taken by another holder")
        if (row["lease_until"] or 0) <= (self._clock() if at is None else at):
            raise StaleFence(
                f"source {self.source!r} lease expired; a stalled connector must not "
                "commit a page it fetched under a lease it no longer holds"
            )


class SyncController:
    """One transaction per page: receipts, cursor and journal advance together.

    Fetching upstream and forming downstream are deliberately separated. The
    cursor only ever moves once the page's evidence is durable, so a crash
    replays the page rather than skipping it.
    """

    def __init__(self, store, *, clock: Callable[[], float] = time.time):
        self.store = store
        self.clock = clock
        self.db = store.db

    # -- registration --------------------------------------------------------

    def register(self, source: str, *, policy_version: str) -> dict[str, Any]:
        """Make *source* known without disturbing an existing connector."""
        _check_name(source)
        if policy_version not in _KNOWN_POLICIES and policy_version != "unconfigured":
            raise EvidenceError(f"unknown connector policy {policy_version!r}")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                """
                INSERT INTO connectors(source, generation, policy_version, coverage_state, updated_at)
                VALUES(?, 1, ?, 'unknown', ?)
                ON CONFLICT(source) DO UPDATE SET
                    policy_version=excluded.policy_version, updated_at=excluded.updated_at
                """,
                (source, policy_version, now()),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.state(source)

    def reconfigure(self, source: str, *, policy_version: str, actor: str, reason: str) -> int:
        """Start a new connector generation and strand every older writer.

        The cursor is cleared rather than kept: a re-mapped account, a changed
        endpoint or a narrowed scope means the old position no longer describes
        the same stream, and resuming it would silently skip evidence.
        """
        _check_name(source)
        _check_reason(reason)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT generation FROM connectors WHERE source=?", (source,)).fetchone()
            if row is None:
                raise EvidenceError(f"source {source!r} is not registered")
            generation = row["generation"] + 1
            self.db.execute(
                """
                UPDATE connectors SET generation=?, policy_version=?, lease=NULL, lease_until=NULL,
                    holder=NULL, cursor=NULL, cursor_kind='opaque', coverage_state='unknown',
                    last_success_at=NULL, updated_at=?
                WHERE source=?
                """,
                (generation, policy_version, now(), source),
            )
            self.store._audit("connector_reconfigure", source,
                              {"actor": actor, "reason": reason, "generation": generation})
            self.db.execute("COMMIT")
            return generation
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def state(self, source: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM connectors WHERE source=?", (source,)).fetchone()
        if row is None:
            raise EvidenceError(f"source {source!r} is not registered")
        out = dict(row)
        out["lease_active"] = bool(out["lease"]) and (out["lease_until"] or 0) > self.clock()
        return out

    # -- leasing -------------------------------------------------------------

    def acquire(self, source: str, *, holder: str, ttl: float = 60.0) -> Fence:
        """Take or re-take the lease. Someone else's unexpired lease is refused.

        Overwriting a live lease would let a second connector interleave with
        the first; re-taking one's own lease is the intended crash-recovery
        path, so a holder that died mid-page can resume it.
        """
        _check_name(source)
        if not 1 <= ttl <= 3600:
            raise EvidenceError("ttl must be between 1 and 3600 seconds")
        token = uuid.uuid4().hex
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT generation, holder, lease_until FROM connectors WHERE source=?", (source,)
            ).fetchone()
            if row is None:
                raise EvidenceError(f"source {source!r} is not registered")
            held = row["holder"] and (row["lease_until"] or 0) > self.clock()
            if held and row["holder"] != holder:
                raise EvidenceError(
                    f"source {source!r} is leased to another connector until "
                    f"{row['lease_until']:.0f}"
                )
            epoch = self.db.execute("SELECT value FROM memory_epoch WHERE id=1").fetchone()[0]
            self.db.execute(
                "UPDATE connectors SET lease=?, lease_until=?, holder=?, updated_at=? WHERE source=?",
                (token, self.clock() + ttl, holder, now(), source),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return Fence(source, row["generation"], epoch, token, holder, clock=self.clock)

    def renew(self, fence: Fence, *, ttl: float = 60.0) -> Fence:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            fence.validate(self.db)
            self.db.execute(
                "UPDATE connectors SET lease_until=? WHERE source=? AND lease=?",
                (self.clock() + ttl, fence.source, fence.lease),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return fence

    def release(self, fence: Fence) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT lease FROM connectors WHERE source=?", (fence.source,)).fetchone()
            if row is not None and row["lease"] == fence.lease:
                self.db.execute(
                    "UPDATE connectors SET lease=NULL, lease_until=NULL, holder=NULL, updated_at=?"
                    " WHERE source=?",
                    (now(), fence.source),
                )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    # -- page commit ---------------------------------------------------------

    def publish(self, fence: Fence, page_token: str, envelopes: Sequence[dict[str, Any]],
                *, next_cursor: str | None = None) -> dict[str, Any]:
        """Commit one page and advance the cursor as a single transaction.

        Idempotent per (source, generation, page_token). A page that is replayed
        after a crash is recognised by fingerprint before any write, so retries
        cannot duplicate evidence; a *different* page under the same token is a
        source-level contradiction and is refused.
        """
        _check_name(fence.source)
        if not isinstance(envelopes, Sequence) or isinstance(envelopes, (str, bytes)):
            raise EvidenceError("envelopes must be a sequence")
        if len(envelopes) > 1000:
            raise EvidenceError("a page may carry at most 1000 records")
        token = _check_token(page_token)
        # Validated before the write lock: a malformed page should fail cheap,
        # and nothing partially prepared may reach the store.
        prepared = [prepare_envelope(envelope) for envelope in envelopes]
        for item in prepared:
            if item.source != fence.source:
                raise EvidenceError(
                    f"page claims source {item.source!r} but the fence is for {fence.source!r}"
                )
        fingerprint = digest([[p.source_id, p.revision, p.fingerprint] for p in prepared])

        self.db.execute("BEGIN IMMEDIATE")
        try:
            fence.validate(self.db)
            seen = self.db.execute(
                "SELECT fingerprint, record_count FROM source_pages "
                "WHERE source=? AND generation=? AND page_token=?",
                (fence.source, fence.generation, token),
            ).fetchone()
            if seen:
                if seen["fingerprint"] != fingerprint:
                    raise EvidenceError(
                        f"page {token!r} was already committed with different content"
                    )
                ids, duplicate, written = [], True, seen["record_count"]
            else:
                ids = [self.store.write_prepared(self.db, item,
                                                 generation=fence.generation)[0]
                       for item in prepared]
                self.db.execute(
                    "INSERT INTO source_pages(source, generation, page_token, record_count, "
                    "fingerprint, committed_at) VALUES(?,?,?,?,?,?)",
                    (fence.source, fence.generation, token, len(prepared), fingerprint, now()),
                )
                if next_cursor is not None:
                    self.db.execute(
                        "UPDATE connectors SET cursor=?, coverage_state='current', "
                        "last_success_at=?, updated_at=? WHERE source=?",
                        (str(next_cursor)[:2000], now(), now(), fence.source),
                    )
                duplicate, written = False, len(prepared)
            # A replayed page writes nothing, but it still has to close the
            # transaction it opened: leaving it idle would hold the write lock
            # and make every later commit fail with 'within a transaction'.
            self.db.execute("COMMIT")
            return {"duplicate": duplicate, "records": written, "ids": ids,
                    "cursor": self._cursor(fence.source)}
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def _cursor(self, source: str) -> str | None:
        row = self.db.execute("SELECT cursor FROM connectors WHERE source=?", (source,)).fetchone()
        return row["cursor"] if row else None

    def mark_gap(self, fence: Fence, *, coverage_state: str, reason: str) -> None:
        """Record that coverage is partial. A gap must stay visible, not silent."""
        if coverage_state not in {"unknown", "current", "partial", "stale", "unreachable", "revoked"}:
            raise EvidenceError(f"unknown coverage state {coverage_state!r}")
        _check_reason(reason)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            fence.validate(self.db)
            self.db.execute(
                "UPDATE connectors SET coverage_state=?, updated_at=? WHERE source=?",
                (coverage_state, now(), fence.source),
            )
            self.store._audit("source_coverage", fence.source,
                              {"state": coverage_state, "reason": reason})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    # -- downstream replay ---------------------------------------------------

    def changes(self, *, consumer: str, limit: int = 200) -> list[dict[str, Any]]:
        """Everything committed since this consumer's own checkpoint.

        Each consumer advances independently, so a backend that is offline
        never holds the upstream cursor or another consumer back.
        """
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise EvidenceError("limit must be between 1 and 1000")
        since = self.checkpoint(consumer=consumer)
        rows = self.db.execute(
            "SELECT seq, source, generation, epoch, record_id, change, committed_at "
            "FROM change_journal WHERE seq>? ORDER BY seq LIMIT ?",
            (since, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def checkpoint(self, *, consumer: str) -> int:
        _check_name(consumer)
        row = self.db.execute(
            "SELECT seq FROM consumer_checkpoints WHERE consumer=?", (consumer,)
        ).fetchone()
        return int(row["seq"]) if row else 0

    def advance(self, *, consumer: str, seq: int) -> None:
        """Move a consumer forward only. Rewinding is a separate, deliberate act."""
        _check_name(consumer)
        if not isinstance(seq, int) or seq < 0:
            raise EvidenceError("seq must be a nonnegative integer")
        current = self.checkpoint(consumer=consumer)
        if seq < current:
            raise EvidenceError(f"consumer {consumer!r} may not rewind {current} -> {seq}")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO consumer_checkpoints(consumer, seq, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(consumer) DO UPDATE SET seq=excluded.seq, updated_at=excluded.updated_at",
                (consumer, seq, now()),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def journal_tail(self, seq: int) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT source, record_id, change, epoch, committed_at FROM change_journal "
            "WHERE seq=(SELECT MAX(seq) FROM change_journal WHERE seq<=?)", (seq,)
        ).fetchone()
        return dict(row) if row else {}

    # -- pause semantics -----------------------------------------------------

    def pause(self, source: str, *, actor: str, reason: str, policy_version: str,
              stages: Sequence[str] = DOWNSTREAM_STAGES) -> list[str]:
        """Pause processing stages for one source. Capture keeps running by default."""
        return self._set_stages(source, "paused", actor=actor, reason=reason,
                                policy_version=policy_version, stages=stages)

    def resume(self, source: str, *, actor: str, reason: str, policy_version: str,
               stages: Sequence[str] = DOWNSTREAM_STAGES) -> list[str]:
        return self._set_stages(source, "active", actor=actor, reason=reason,
                                policy_version=policy_version, stages=stages)

    def pause_capture(self, source: str, *, actor: str, reason: str, policy_version: str) -> list[str]:
        """The explicitly named ingestion stop, separate from a normal pause."""
        return self._set_stages(source, "paused", actor=actor, reason=reason,
                                policy_version=policy_version, stages=("capture",))

    def _set_stages(self, source, state, *, actor, reason, policy_version, stages) -> list[str]:
        _check_name(source)
        _check_reason(reason)
        unknown = set(stages) - set(STAGES)
        if unknown or not stages:
            raise EvidenceError(f"stages must come from {STAGES}")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for stage in stages:
                self.store.set_control(source, stage, state, actor=actor, reason=reason,
                                       policy_version=policy_version)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return list(stages)

    def paused_stages(self, source: str) -> list[str]:
        return [stage for stage in STAGES if self.store.stage_is_paused(source, stage)]


_KNOWN_POLICIES = frozenset({"local-only", "private-api", "disabled"})


def _check_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise EvidenceError("name must be nonempty text of at most 200 characters")
    return value


def _check_reason(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise EvidenceError("reason must be nonempty text of at most 1000 characters")
    return value


def _check_token(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise EvidenceError("page_token must be nonempty text of at most 500 characters")
    return value
