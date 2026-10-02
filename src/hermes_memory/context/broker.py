"""C9 context broker: one bounded packet, assembled from what is actually available.

Two channels, reported separately. The lexical channel is local and answers in
milliseconds with no model in the loop; the derived channel asks the backend for
facts and may be down, slow, saturated, or simply not configured. A packet that
silently omits one reads exactly like a packet that found nothing, and the
difference decides whether the agent is entitled to say anything at all.

Everything is bounded: the packet has a token ceiling, the derived channel has a
deadline, and a channel that exceeds either is dropped rather than allowed to
make the whole answer late or the whole turn long.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import replace
from typing import Any, Callable, Iterable

from ..ids import digest
from ..backend.support import SupportError, SupportResolver
from ..storage.identity import (citations_in_scope, evidence_accounts, IdentityStore,
                           in_scope)
from .cache import PacketCache
from .lexical import AVAILABLE, LexicalChannel
from .packet import (CONFLICTING, PARTIAL, SUPPORTED, UNKNOWN, Channels, EvidenceItem,
                     Packet)

__all__ = ["ContextBroker", "Packet", "EvidenceItem", "Channels", "PacketCache",
           "CONFLICTING", "PARTIAL", "SUPPORTED", "UNKNOWN"]

# A derived channel that was never tried or never configured is not a failure;
# a channel that was tried and could not be reached is.
_ATTEMPTED_FAILED = {"unavailable", "timeout", "partial", "unreachable", "paused", "denied"}
# Below this, a prefix of an answer carries more risk of misleading than any
# chance of helping, so the budget is left unused instead.
_MIN_PREFIX_TOKENS = 40
# Readings of a scope, not a fourth retrieval channel: a few of them, each quoted at
# arm's length, so a digest never occupies the room the evidence needs.
_SUMMARY_SPAN = 600
_MAX_SUMMARIES = 3
# The metadata fields that name a scope a summary can be written about, in the same
# grammar `hermes-memory summarize --scope` accepts.
_SCOPE_FIELDS = ("project", "thread", "account")


def _item(evidence, *, text: str | None = None, span_truncated: bool = False) -> EvidenceItem:
    return EvidenceItem(
        id=evidence.id, source=evidence.source, source_id=evidence.source_id,
        text=evidence.text if text is None else text,
        occurred_at=evidence.occurred_at, occurred_precision=evidence.occurred_precision,
        observed_at=evidence.observed_at, channel="lexical", score=1.0,
        span_truncated=span_truncated)


class ContextBroker:
    """Assemble a packet within a token ceiling and a wall-clock deadline."""

    def __init__(self, store: Any, *, client: Any = None, identity: IdentityStore | None = None,
                 assertions: Any = None, summaries: Any = None,
                 budget_tokens: int = 1800,
                 derived_timeout_s: float = 4.0,
                 cache: PacketCache | None | bool = True, account_id: str | None = None,
                 clock: Callable[[], float] = time.monotonic, estimator=None):
        if not isinstance(budget_tokens, int) or not 1 <= budget_tokens <= 200_000:
            raise ValueError("budget_tokens must be an integer between 1 and 200000")
        if not isinstance(derived_timeout_s, (int, float)) or not 0 < derived_timeout_s <= 30:
            raise ValueError("derived_timeout_s must be between 0 and 30 seconds")
        self.store = store
        self.client = client
        # Identity resolution needs only the store, so a broker built without an
        # injected IdentityStore still cannot answer for a named account out of
        # somebody else's evidence. An injected one (a profile's own, with its owner
        # principal) is used as given.
        self.identity = IdentityStore(store) if identity is None else identity
        self.assertions = assertions
        # A SummaryStore, or None. Read-only here: the packet may quote a reading, and the
        # broker never publishes, withdraws or refreshes one.
        self.summaries = summaries
        self.budget_tokens = budget_tokens
        self.derived_timeout_s = float(derived_timeout_s)
        self.account_id = account_id
        self.clock = clock
        self.lexical = LexicalChannel(store)
        # Chars/3 is a conservative stand-in; a real tokenizer can be injected.
        self.estimate = estimator or (lambda text: max(1, len(text) // 3))
        if isinstance(cache, PacketCache):
            self.cache = cache
        elif cache is True:
            self.cache = PacketCache(store, clock=clock)
        else:  # None or False: every turn goes back to the store.
            self.cache = None
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._closed = False

    # -- assembly ------------------------------------------------------------

    def assemble(self, query: str, *, limit: int = 8, include_derived: bool = True,
                 sources: Iterable[str] | None = None,
                 window: tuple[str | None, str | None] | None = None,
                 account_id: str | None = None, commitments: Iterable[dict] = (),
                 lessons: Iterable[dict] = ()) -> Packet:
        """Build the packet for one question. Raises only on a caller bug."""
        query = (query or "").strip()
        if not query:
            raise ValueError("a context query must not be empty")
        limit = limit if isinstance(limit, int) and 1 <= limit <= 50 else 8
        caller = self.account_id if account_id is None else account_id
        scope = self._scope(query, limit=limit, include_derived=include_derived,
                            sources=sources, window=window, account_id=caller,
                            commitments=commitments, lessons=lessons)
        started = self.clock()
        if self.cache is not None:
            cached = self.cache.get(query, account_id=caller, scope=scope)
            if cached is not None:
                return replace(cached, took_ms=int((self.clock() - started) * 1000))

        # Stamp first: anything the archive changes after this point is a change
        # this packet did not see, and has to be reported rather than hidden.
        epoch, revision = self.store.watermark()
        truncated: list[str] = []

        outcome = self.lexical.gather(query, limit=max(limit * 3, 20))
        if outcome.dropped_terms:
            truncated.append("terms")
        allowed, withheld = self._authorize(outcome.items, account_id=caller)
        considered = [evidence for evidence in allowed
                      if _usable(evidence, sources=sources, window=window)]
        conflicts = self._conflicts(considered)

        spent = 0
        items: list[EvidenceItem] = []
        # A lesson is a conclusion drawn from someone's records, so it carries the same
        # scoping as the records it names: teaching it across an identity boundary
        # repeats that person's evidence in someone else's answer.
        teachable, lessons_withheld = self._authorize_lessons(lessons, account_id=caller)
        if lessons_withheld:
            truncated.append("lessons")
        # Commitments and lessons are short and always worth their lines, so they
        # reserve first; a long evidence list must not crowd out a due obligation.
        kept_commitments, spent = self._bounded(commitments, spent, key="title", cap=4)
        kept_lessons, spent = self._bounded(teachable, spent, key="text", cap=4)

        # Typed claims are checked, attributable and short, so they go in before
        # the raw spans that would crowd them out.
        asserted, assertion_conflicts, assertion_note, spent = self._assertions(query, spent, caller=caller,
                                                                             sources=sources, window=window)
        conflicts = conflicts + assertion_conflicts
        if assertion_note:
            truncated.append("assertions")

        for evidence in considered:
            if len(items) >= limit:
                truncated.append("results")
                break
            cost = self.estimate(evidence.text) + 12
            if spent + cost > self.budget_tokens:
                truncated.append("packet")
                # A prefix of the best match beats an empty packet: the caller
                # asked for a ceiling, not for amnesia. A sliver of room is not
                # worth a line that could only mislead, so this stops early.
                room = self.budget_tokens - spent - 12
                if room >= _MIN_PREFIX_TOKENS:
                    prefix = evidence.text[:room * 3]
                    spent += self.estimate(prefix) + 12
                    items.append(_item(evidence, text=prefix, span_truncated=True))
                break
            spent += cost
            items.append(_item(evidence))

        # Readings of the scopes this answer touched, taken after the evidence so a digest
        # never crowds out the record it was written from.
        readings, spent, dropped_readings, reading_note = self._summaries(
            considered, spent, caller=caller, sources=sources, window=window)
        if dropped_readings:
            truncated.append("summaries")

        facts: tuple[dict[str, Any], ...] = ()
        derived_state, derived_detail = "not_attempted", ""
        if include_derived:
            facts, derived_state, derived_detail, derived_truncated, source_facts = self._derive(query, limit)
            checked = tuple(fact for fact in facts
                            if self._fact_allowed(fact, caller=caller, sources=sources, window=window,
                                                  source_facts=source_facts))
            if len(checked) != len(facts):
                derived_state, derived_detail = "partial", "unverified or unauthorized fact provenance withheld"
            facts = checked
            truncated.extend(derived_truncated)
            # Derived text is charged too. A backend that returns an essay would
            # otherwise overrun the ceiling the caller asked us to hold.
            kept = []
            for fact in facts:
                text = str(fact.get("text") or fact.get("content") or "")[:600]
                cost = self.estimate(text) + 12
                if spent + cost > self.budget_tokens:
                    truncated.append("packet")
                    break
                spent += cost
                kept.append(fact)
            facts = tuple(kept)

        items, revoked, store_moved = self._recheck(items, stamp=(epoch, revision))
        store_moved = store_moved or scope != self._scope(
            query, limit=limit, include_derived=include_derived, sources=sources,
            window=window, account_id=caller, commitments=commitments, lessons=lessons)
        if revoked:
            truncated.append("revoked_during_recall")
        if store_moved:
            # A reset or trust revocation happened while we were on the network.
            # The packet still says what it found, but it cannot be reused.
            truncated.append("store_moved")
            # Do not deliver derivative sections built under a stale authority stamp.
            facts, asserted, readings, kept_lessons, conflicts = (), (), (), (), ()
            evidence = [self.store.get(item.id) for item in items]
            authorized, _ = self._authorize([e for e in evidence if e is not None], account_id=caller)
            current = {e.id for e in authorized}
            items = [item for item in items if item.id in current]

        notes = [text for text in (outcome.detail, derived_detail, assertion_note,
                                   reading_note) if text]
        if revoked:
            notes.append(f"{revoked} item(s) were forgotten or hidden during retrieval")
        if lessons_withheld:
            notes.append(f"{lessons_withheld} lesson(s) belong to evidence this caller "
                         "is not joined to")
        if store_moved:
            notes.append("the archive changed during retrieval; this packet is not cached")
        channels = Channels(lexical=outcome.state, derived=derived_state,
                            detail="; ".join(notes)[:400])

        packet = Packet(
            query=query, items=tuple(items), assertions=asserted, facts=facts,
            summaries=readings,
            lessons=kept_lessons, commitments=kept_commitments,
            channels=channels,
            coverage=self._coverage(items=items, facts=facts, conflicts=conflicts,
                                    channels=channels, truncated=truncated),
            truncated=tuple(dict.fromkeys(truncated)),
            tokens_used=spent, took_ms=int((self.clock() - started) * 1000),
            conflicts=conflicts, epoch=epoch, revision=revision, withheld=withheld,
        )
        packet = _with_id(packet)
        if self.cache is not None and not store_moved:
            self.cache.put(query, packet, account_id=caller, scope=scope)
        return packet

    def close(self) -> None:
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # -- channels ------------------------------------------------------------

    def _derive(self, query, limit):
        """Ask the backend, and give up on deadline rather than making the turn late."""
        if self.client is None:
            return (), "not_configured", "", (), ()
        try:
            outcome = self._recall_within_deadline(query)
        except FutureTimeout:
            # The request was abandoned, not cancelled: the socket stays open
            # until the client's own timeout fires. Nothing downstream may
            # conclude that the backend did no work.
            return (), "timeout", f"derived channel exceeded {self.derived_timeout_s}s", (), ()
        except Exception as error:
            # Unreachable, 5xx, unsupported capability, paused stage: to the
            # caller these all mean this half of the answer is missing.
            return (), "unavailable", f"{type(error).__name__}: {error}"[:200], (), ()
        truncated = [str(name) for name in outcome.truncated]
        facts = tuple(dict(fact) for fact in tuple(outcome.results)[:limit])
        if len(outcome.results) > limit:
            truncated.append("results")
        complete = getattr(outcome, "provenance_complete", not truncated)
        state = "available" if complete else "partial"
        detail = "" if complete else "provenance came back truncated"
        return facts, state, detail, tuple(dict.fromkeys(truncated)), outcome.source_facts

    def _recall_within_deadline(self, query):
        executor = self._executor()
        future = executor.submit(self.client.recall, query,
                                 max_tokens=min(self.budget_tokens, 4096), types=None)
        try:
            return future.result(timeout=self.derived_timeout_s)
        except FutureTimeout:
            future.cancel()
            raise

    def _executor(self) -> ThreadPoolExecutor:
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("the broker is closed")
            if self._pool is None:
                # Two workers: a slow backend may stall one, and the next turn
                # should still be able to try.
                self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="derived")
            return self._pool

    # -- authority and quality ----------------------------------------------

    def _authorize(self, matches, *, account_id):
        """Split what this caller may see from what it must not reach.

        Candidates never widen access: an unconfirmed join is a hypothesis, and
        acting on it as if it were true would expose one person's evidence in
        another's context. A record that claims no account at all is shared,
        because it is not account-scoped evidence; withholding unscoped notes
        would empty the store without protecting anyone. An account with no
        confirmed join therefore sees exactly the evidence that names it, or
        names nobody.
        """
        identifiers = set(self.identity.group(account_id)) if account_id else set()
        allowed = [evidence for evidence in matches
                   if in_scope(evidence_accounts(self.identity, evidence), identifiers)]
        return allowed, len(matches) - len(allowed)

    def _authorize_lessons(self, lessons, *, account_id):
        """Which of this caller's practice it may be told, and what was dropped.

        The rule is the evidence rule, applied one step back: every span a lesson
        cites has to be a span this caller could be shown. A caller the installation
        cannot identify therefore gets the practice derived from unscoped evidence
        only, exactly as the evidence list itself does.
        """
        given = list(lessons or ())
        identifiers = set(self.identity.group(account_id)) if account_id else set()
        kept = [lesson for lesson in given
                if citations_in_scope(self.store, self.identity,
                                      (lesson or {}).get("evidence") or (),
                                      identifiers=identifiers)]
        return kept, len(given) - len(kept)

    def _coverage(self, *, items, facts, conflicts, channels, truncated):
        """How much of the question this packet answers — never how true it is."""
        if conflicts:
            return CONFLICTING
        degraded = (channels.lexical != AVAILABLE
                    or channels.derived in _ATTEMPTED_FAILED
                    or bool(set(truncated) & {"packet", "terms", "store_moved",
                                              "revoked_during_recall", "assertions",
                                              "source_facts", "chunks", "results",
                                              "lessons", "summaries"}))
        if not items and not facts:
            # "Nothing here" and "we could not look" are different answers, and
            # only the first entitles the agent to say the archive is empty.
            return PARTIAL if degraded or not channels.any_available else UNKNOWN
        if not items:
            # A backend statement with no canonical evidence behind it is a
            # hypothesis. The provenance ledger can point at a document that has
            # since been corrected, and only the local store knows that.
            return PARTIAL
        return PARTIAL if degraded else SUPPORTED

    def _assertions(self, query, spent, *, caller, sources, window):
        """Time-valid typed claims about what the archive asserts, plus their disputes."""
        if self.assertions is None:
            return (), (), "", spent
        try:
            found = self.assertions.matching(query, limit=6)
        except Exception as error:
            # A section we could not read is missing, not settled: it goes in the
            # caveats as a degradation, never in the conflict list.
            return (), (), f"assertions could not be read: {error}"[:160], spent
        kept: list[dict[str, Any]] = []
        for assertion in found:
            if not self._fact_allowed({"record_id": assertion.record_id}, caller=caller,
                                      sources=sources, window=window):
                continue
            cost = self.estimate(f"{assertion.subject} {assertion.predicate} "
                                 f"{assertion.value}") + 24
            if spent + cost > self.budget_tokens:
                break
            spent += cost
            kept.append(assertion.as_dict())
        conflicts: list[str] = []
        for subject in dict.fromkeys(item["subject"] for item in kept):
            for clash in self.assertions.contradictions(subject=subject):
                if not all(self._fact_allowed({"record_id": item.record_id}, caller=caller,
                                              sources=sources, window=window) for item in clash.assertions):
                    continue
                values = " vs ".join(sorted({item.value for item in clash.assertions})[:3])
                conflicts.append(f"{clash.subject} {clash.predicate} is disputed: {values}")
        return tuple(kept), tuple(dict.fromkeys(conflicts)), "", spent

    def _fact_allowed(self, fact, *, caller, sources, window, source_facts=()):
        """Only locally resolved canonical provenance may cross the context boundary."""
        try:
            ids = SupportResolver(self.store, bank_id=getattr(self.client, "bank_id", None),
                                  source_facts=source_facts).resolve(fact)
        except (SupportError, TypeError, ValueError):
            return False
        evidence = [self.store.get(identifier) for identifier in ids]
        if any(item is None or not self.store.live_and_visible(item.id)
               or not _usable(item, sources=sources, window=window) for item in evidence):
            return False
        allowed = self._authorize(evidence, account_id=caller)[1] == 0
        if allowed:
            fact["record_ids"] = list(ids)
        return allowed

    def _summaries(self, considered, spent, *, caller=None, sources=None, window=None):
        """The current reading of each scope this answer touches, if one is on file.

        Scopes are read off the authorized evidence rather than guessed from the query's
        wording, so a summary can only reach a caller who was already shown evidence in its
        window. ``latest`` withholds a summary whose evidence fell away, and an unsupported
        reading is absent here rather than replaced by the older one that still is on file.
        """
        if self.summaries is None or not considered:
            return (), spent, 0, ""
        dropped = 0
        kept: list[dict[str, Any]] = []
        for scope in self._scopes(considered):
            try:
                current = self.summaries.latest(scope, account_id=caller)
            except Exception as error:
                # A section that could not be read is missing, not empty: the caller is
                # told, and the readings that were already paid for stay in.
                return tuple(kept), spent, dropped + 1, (
                    f"summaries could not be read: {error}"[:160])
            for summary in current[:1]:
                support = self.summaries.ledger.resolve(summary.id, account_id=caller)
                if (not support.evidence or support.verified != support.cited
                        or any(not _usable(item, sources=sources, window=window)
                               for item in support.evidence)):
                    dropped += 1
                    continue
                text = summary.body[:_SUMMARY_SPAN]
                cost = self.estimate(text) + 20
                if spent + cost > self.budget_tokens:
                    dropped += 1
                    continue
                spent += cost
                kept.append({"id": summary.id, "scope": summary.scope, "kind": summary.kind,
                             "title": summary.title, "revision": summary.revision,
                             "window_from": summary.window_from, "window_to": summary.window_to,
                             "stale": self.summaries.is_stale(summary), "body": text,
                             "truncated": len(summary.body) > len(text)})
        return tuple(kept), spent, dropped, ""

    def _scopes(self, considered):
        """Which readings this evidence belongs to, most specific first."""
        named, sourced = [], []
        for evidence in considered:
            metadata = getattr(evidence, "metadata", None) or {}
            for field in _SCOPE_FIELDS:
                value = str(metadata.get(field) or "").strip()
                if value and f"{field}:{value}" not in named:
                    named.append(f"{field}:{value}")
            source = f"source:{evidence.source}"
            if evidence.source and source not in sourced:
                sourced.append(source)
        return (named + sourced)[:_MAX_SUMMARIES]

    def _conflicts(self, candidates):
        """Live accounts that declare incompatible values for the same attribute.

        Only declared disagreements count. Deciding that two free texts
        contradict each other is a model judgement, and a model judgement
        dressed up as a retrieval fact is how a wrong answer becomes
        authoritative. So this reads ``metadata['claims']`` and the occurred_at
        of records that name a subject, and invents nothing.
        """
        subjects: dict[str, dict[str, dict[str, str]]] = {}
        for evidence in candidates:
            subject = str(evidence.metadata.get("subject") or "").strip()
            if not subject:
                continue
            declared = evidence.metadata.get("claims")
            claims = dict(declared) if isinstance(declared, dict) else {}
            if evidence.occurred_at:
                claims.setdefault("occurred_at", evidence.occurred_at)
            for attribute, value in claims.items():
                if not isinstance(value, (str, int, float, bool)):
                    continue
                shown = str(value).strip()
                if not shown:
                    continue
                bucket = subjects.setdefault(subject, {}).setdefault(str(attribute), {})
                bucket.setdefault(shown.casefold(), shown)
        conflicts = []
        for subject in sorted(subjects):
            for attribute in sorted(subjects[subject]):
                values = sorted(subjects[subject][attribute].values())
                if len(values) > 1:
                    conflicts.append(f"{subject} {attribute} is disputed: "
                                     + " vs ".join(values[:3]))
        return tuple(conflicts[:8])

    def _recheck(self, items, *, stamp):
        """Drop what a concurrent forgetting or correction has already removed.

        The lexical read happened before the network call, so an item can be
        forgotten while this packet is in flight. The plan is explicit that
        scope, epoch, visibility and revisions are rechecked after the network,
        and a stale line surviving that window is an exposure, not a race.
        """
        epoch, revision = self.store.watermark()
        kept = [item for item in items if self.store.live_and_visible(item.id)]
        return kept, len(items) - len(kept), (epoch, revision) != stamp

    # -- helpers -------------------------------------------------------------

    def _bounded(self, entries, spent, *, key, cap):
        """Take the first few of a section, and stop when the ceiling says so."""
        kept: tuple[dict[str, Any], ...] = ()
        for entry in tuple(entries or ())[:cap]:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get(key) or entry.get("text") or "")
            if not text.strip():
                continue
            cost = self.estimate(text) + 12
            if spent + cost > self.budget_tokens:
                break
            spent += cost
            kept = kept + (entry,)
        return kept, spent

    def _scope(self, query, **options) -> tuple[str, ...]:
        """Everything that changes the right answer, so the cache cannot mix it up."""
        parts = [f"limit={options['limit']}", f"derived={int(bool(options['include_derived']))}",
                 f"sources={','.join(sorted(options['sources'] or ())) or '-'}",
                 f"window={options['window'] or '-'}",
                 f"account={options['account_id'] or '-'}",
                 # Natural edge expiry changes authority without a database write.
                 f"joined={digest(self.identity.group(options['account_id'])) if options['account_id'] else '-'}",
                 f"budget={self.budget_tokens}",
                 # Whether readings are on changes the answer, so a packet warmed without
                 # them is never handed to a caller that expects them.
                 f"readings={int(self.summaries is not None)}",
                 # The caller's own material changes the answer as much as a filter does:
                 # a packet warmed for one task must not be handed to a different task that
                 # earned different practice, and a due obligation is not interchangeable.
                 f"lessons={_fingerprint(options.get('lessons'))}",
                 f"commitments={_fingerprint(options.get('commitments'))}"]
        return tuple(parts)


def _fingerprint(items: Iterable[Any] | None) -> str:
    """A short digest of caller-supplied material, or "-" when there was none.

    The identity of a lesson is its id and version, and of a commitment its title and due
    time: enough to tell two different sets apart, short enough to be a cache key."""
    found = sorted(_identity(item) for item in (items or ()))
    return "-" if not found else digest(found)[:16]


def _identity(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)[:120]
    named = item.get("id") or item.get("lesson") or item.get("title") or item.get("text")
    version = item.get("version")
    return f"{str(named)[:100]}@{version}" if version is not None else str(named)[:120]


def _usable(evidence, *, sources, window) -> bool:
    if sources and evidence.source not in set(sources):
        return False
    return _in_window(evidence.occurred_at, window)


def _in_window(occurred_at, window):
    if not window:
        return True
    if occurred_at is None:
        return True  # an undated item is not excluded by a filter it cannot satisfy
    start, end = window
    if start and occurred_at < start:
        return False
    if end and occurred_at > end:
        return False
    return True


def _with_id(packet: Packet) -> Packet:
    packet_id = "ctx_" + digest([packet.query, packet.epoch, packet.revision,
                                 [item.id for item in packet.items],
                                 [str(fact)[:80] for fact in packet.facts],
                                 [str(item.get("id")) for item in packet.summaries],
                                 list(packet.conflicts), packet.coverage])[:24]
    return replace(packet, packet_id=packet_id)
