"""C3 — connector leases, fences, cursors and consumer replay."""
from __future__ import annotations


import time
import uuid
from typing import Any, Callable, Sequence

from ..ids import digest, now
from ..storage.evidence import EvidenceError, prepare_envelope
from .base import Skipped

__all__ = ["Fence", "StaleFence", "SyncController", "STAGES", "DOWNSTREAM_STAGES",
           "COVERAGE_STATES"]

# An operator who pauses a source means that source: ingestion, the compute that
# forms it downstream, and anything that could speak on its behalf. Stopping only
# the ingestion half — keeping what we have, working on it, saying nothing — is a
# different instruction and has its own name, so neither has to be guessed at.
DOWNSTREAM_STAGES = ("formation", "proactivity")
STAGES = ("capture",) + DOWNSTREAM_STAGES

# "current" is the only state that says we have everything the source would give.
# Every other value means some answer would be about a partial world.
COVERAGE_STATES = frozenset({"unknown", "current", "partial", "stale", "unreachable",
                            "revoked"})


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
        """Extend a lease, or refuse to. The fence is checked inside the write
        transaction, so a holder whose lease has passed to someone else is told so
        rather than left reading and believing it had written back into authority.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            fence.validate(self.db)
            self.db.execute(
                "UPDATE connectors SET lease_until=?, updated_at=? WHERE source=? AND lease=?",
                (self.clock() + ttl, now(), fence.source, fence.lease),
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
                *, next_cursor: str | None = None,
                skipped: Sequence[Any] = ()) -> dict[str, Any]:
        """Commit one page and advance the cursor as a single transaction.

        Idempotent per (source, generation, page_token). A page that is replayed
        after a crash is recognised by fingerprint before any write, so retries
        cannot duplicate evidence; a *different* page under the same token is a
        source-level contradiction and is refused. The runtime derives a token from
        page content when the source cannot name its own page, which is what keeps a
        moving tail from being read as a contradiction — see
        ``ConnectorRuntime._token``.

        What the source could not give us on this page is recorded here rather than
        logged, in the same transaction: a gap that is not durable is a gap nobody
        will ever close, and one that arrives later has to be able to prove it did.
        """
        _check_name(fence.source)
        if isinstance(envelopes, (str, bytes)) or not isinstance(envelopes, Sequence):
            raise EvidenceError("a page carries a sequence of envelopes")
        if len(envelopes) > 1000:
            raise EvidenceError("a page may carry at most 1000 records")
        if len(skipped) > 1000:
            raise EvidenceError("a page may report at most 1000 gaps")
        token = _check_token(page_token)
        gaps = [(_ref_of(item), _reason_of(item)) for item in skipped]
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
                "SELECT fingerprint FROM source_pages "
                "WHERE source=? AND generation=? AND page_token=?",
                (fence.source, fence.generation, token),
            ).fetchone()
            if seen:
                if seen["fingerprint"] != fingerprint:
                    raise EvidenceError(
                        f"page {token!r} was already committed with different content"
                    )
                ids, duplicate, written = [], True, 0
            else:
                rows = [self.store.write_prepared(self.db, item,
                                                  generation=fence.generation)
                        for item in prepared]
                ids = [row[0] for row in rows]
                written = sum(1 for row in rows if not row[1])
                self.db.execute(
                    "INSERT INTO source_pages(source, generation, page_token, record_count, "
                    "fingerprint, committed_at) VALUES(?,?,?,?,?,?)",
                    (fence.source, fence.generation, token, len(prepared), fingerprint, now()),
                )
                duplicate = False
            # Reaching the end of a source is what 'current' means; a page with a
            # cursor behind it is a backfill in progress, whatever the caller hoped.
            self.db.execute(
                "UPDATE connectors SET coverage_state=?, last_success_at=?, updated_at=? "
                "WHERE source=?",
                ("current" if next_cursor is None else "partial", now(), now(), fence.source))
            if not duplicate and next_cursor is not None:
                # Only a page that actually landed moves the position: replaying one
                # that was already committed must not walk past the source's head.
                self.db.execute("UPDATE connectors SET cursor=? WHERE source=?",
                                (str(next_cursor)[:2000], fence.source))
            for ref, reason in gaps:
                self.db.execute(
                    "INSERT INTO source_gaps(source, generation, ref, reason, first_seen_at, "
                    "last_seen_at, cleared_at) VALUES(?,?,?,?,?,?,NULL) "
                    "ON CONFLICT(source, generation, ref) DO UPDATE SET reason=excluded.reason, "
                    "last_seen_at=excluded.last_seen_at, cleared_at=NULL",
                    (fence.source, fence.generation, ref, reason, now(), now()))
            self._clear_gaps(fence, [item.source_id for item in prepared])
            # A replayed page writes nothing, but it still has to close the
            # transaction it opened: leaving it idle would hold the write lock
            # and make every later commit fail with 'within a transaction'.
            self.db.execute("COMMIT")
            return {"duplicate": duplicate, "records": len(prepared), "new": written,
                    "ids": ids, "gaps": len(gaps),
                    "coverage_state": "current" if next_cursor is None else "partial",
                    "cursor": self._cursor(fence.source)}
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def _clear_gaps(self, fence: Fence, source_ids: Sequence[str]) -> int:
        """Close a gap the source has since handed over.

        Matched by source id rather than by position, because the point of a gap is
        "this thing is missing", and the thing arriving later is the only evidence
        that would ever close it.
        """
        if not source_ids:
            return 0
        placeholders = ",".join("?" * len(source_ids))
        return int(self.db.execute(
            "UPDATE source_gaps SET cleared_at=? WHERE source=? AND generation=? AND "
            f"cleared_at IS NULL AND ref IN ({placeholders})",
            [now(), fence.source, fence.generation, *source_ids]).rowcount or 0)

    def _cursor(self, source: str) -> str | None:
        row = self.db.execute("SELECT cursor FROM connectors WHERE source=?", (source,)).fetchone()
        return row["cursor"] if row else None

    def restart(self, fence: Fence, *, reason: str) -> None:
        """Drop the stored position so the next pass reads from the beginning.

        A cursor the source will not accept is not a lease problem: the position
        simply no longer describes the stream. Nothing is deleted. Evidence already
        committed stays, and re-reading what the source will give again is idempotent
        per record, so a restart costs quota and cannot invent duplicates.
        """
        _check_reason(reason)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            fence.validate(self.db)
            self.db.execute(
                "UPDATE connectors SET cursor=NULL, cursor_kind='opaque', coverage_state="
                "'partial', updated_at=? WHERE source=?",
                (now(), fence.source),
            )
            self.store._audit("connector_restart", fence.source,
                              {"reason": reason, "generation": fence.generation})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def mark_gap(self, fence: Fence, *, coverage_state: str, reason: str) -> None:
        """Record that coverage is partial. A gap must stay visible, not silent."""
        if coverage_state not in COVERAGE_STATES:
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

    def gaps(self, source: str, *, generation: int | None = None,
             include_cleared: bool = False, limit: int = 200) -> list[dict[str, Any]]:
        """What this connector knows it did not get.

        Open gaps are the honest version of a coverage claim: the source said there
        was something there and could not hand it over. The default is the *current*
        generation, because a reconfigured connector is a different question than a
        flaky one: what the old credentials could not reach is history, not debt.
        """
        _check_name(source)
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise EvidenceError("the gap list is between 1 and 1000 rows")
        where = ["source=?", "generation=?", "cleared_at IS NULL"]
        arguments: list[Any] = [
            source, int(generation if generation is not None
                        else self.state(source)["generation"])]
        if include_cleared:
            where.pop()
        rows = self.db.execute(
            f"SELECT generation, ref, reason, first_seen_at, last_seen_at, cleared_at "
            f"FROM source_gaps WHERE {' AND '.join(where)} "
            f"ORDER BY last_seen_at, ref LIMIT ?",
            [*arguments, limit]).fetchall()
        return [dict(row) for row in rows]

    # -- pause semantics -----------------------------------------------------

    def pause(self, source: str, *, actor: str, reason: str, policy_version: str,
              stages: Sequence[str] = STAGES) -> list[str]:
        """Pause a source end to end. A narrower stop is named for what it stops."""
        return self._set_stages(source, "paused", actor=actor, reason=reason,
                                policy_version=policy_version, stages=stages)

    def resume(self, source: str, *, actor: str, reason: str, policy_version: str,
               stages: Sequence[str] = STAGES) -> list[str]:
        return self._set_stages(source, "active", actor=actor, reason=reason,
                                policy_version=policy_version, stages=stages)

    def pause_capture(self, source: str, *, actor: str, reason: str, policy_version: str) -> list[str]:
        """The explicitly named ingestion stop, separate from a normal pause."""
        return self._set_stages(source, "paused", actor=actor, reason=reason,
                                policy_version=policy_version, stages=("capture",))

    def resume_capture(self, source: str, *, actor: str, reason: str,
                       policy_version: str) -> list[str]:
        """Lift the ingestion stop and nothing else.

        A source held from capture usually still has its downstream stages paused, and
        resuming all three because somebody typed ``--resume`` would speak on evidence
        whose reading was never the thing being restarted.
        """
        return self._set_stages(source, "active", actor=actor, reason=reason,
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


def _ref_of(item: Any) -> str:
    ref = item.ref if isinstance(item, Skipped) else item[0]
    if not isinstance(ref, str) or not ref.strip():
        raise EvidenceError("a reported gap needs the source's own reference to it")
    return ref[:500]


def _reason_of(item: Any) -> str:
    reason = item.reason if isinstance(item, Skipped) else item[1]
    if not isinstance(reason, str) or not reason.strip():
        raise EvidenceError("a reported gap needs a reason; 'nothing' is not one")
    return reason[:1000]
