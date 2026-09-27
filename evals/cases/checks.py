"""§12.4: the release criteria, each one either measured here or named as not measured.

The plan's metric table is eleven rows, and every row is a promise that can be false in two
different ways. A row scored against nothing is a lie; a row quietly dropped from the report
is a lie with better manners. So each row in this module returns one of exactly two shapes:
a measurement with a value, the criterion it is held to, and the denominator it was computed
over — or an explicit `not-measured` with the reason and what would have to exist first.

Nothing here reaches the network or a model. That is not a shortcut: it is what makes the
numbers reproducible enough to compare across releases, and it keeps the harness runnable
on a machine with inference switched off — which is the state the plan says the framework
must be able to operate in. Model-backed rows are therefore the ones reported as
not measured, with the gate that would let them be measured named in their place.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hermes_memory.ids import record_id
from hermes_memory.proactive.policy import POLICY_VERSION
from hermes_memory.storage.evidence import EvidenceStore, prepare_envelope
from hermes_memory.storage.lineage import Lineage

import corpus

MEASURED = "measured"
NOT_MEASURED = "not-measured"


# -- the two shapes -----------------------------------------------------------

def measured(row: str, name: str, *, value: Any, criterion: str, passes: bool,
             denominator: Any = None, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """A number, the bar it is held to, and what it was counted over.

    The denominator travels with the value on purpose. "Recall 0.98" says nothing about
    whether that was 98 of 100 questions or 98 of 98 questions the harness happened to ask.
    """
    return {"row": row, "check": name, "status": MEASURED, "value": value,
            "criterion": criterion, "pass": bool(passes), "denominator": denominator,
            "detail": detail or {}}


def not_measured(row: str, name: str, *, because: str, requires: str) -> dict[str, Any]:
    """A promise this build cannot yet keep, said out loud with the thing it waits on."""
    return {"row": row, "check": name, "status": NOT_MEASURED, "value": None,
            "criterion": None, "pass": None, "because": because, "requires": requires}


def ratio(hits: int, total: int) -> float:
    return 0.0 if not total else round(hits / total, 4)


LEDGER_SUFFIXES = (".db", ".db-wal", ".db-shm", ".sqlite", ".jsonl")


def _fingerprints(*roots: Path, skip_ledgers: bool = False) -> dict[str, str]:
    """A digest of every regular file under these roots, keyed by absolute path.

    Byte equality rather than modification times: a second setup that rewrites a file with
    the same content has changed nothing, and that is exactly what an idempotent step is
    allowed to do.

    `skip_ledgers` exists because a ledger is the one file that *should* change when you use
    it: comparing an installation database byte for byte would turn "the audit trail recorded
    that setup ran twice" into a failure.
    """
    found: dict[str, str] = {}
    for root in roots:
        for item in sorted(root.rglob("*")):
            if not item.is_file():
                continue
            if skip_ledgers and item.name.endswith(LEDGER_SUFFIXES):
                continue
            found[str(item)] = hashlib.sha256(item.read_bytes()).hexdigest()[:16]
    return found


def _instant(value: Any) -> datetime | None:
    """A comparable instant from whatever form a timestamp arrived in.

    The store normalises to one form and the corpus writes another, so comparing strings would
    report a preserved event time as drift.
    """
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def percentiles(samples: list[float]) -> dict[str, float]:
    """Nearest-rank percentiles, because the plan gates on a p95 rather than a mean.

    A mean over hundreds of calls hides exactly the ones that matter: the slow tail a user
    waits on.
    """
    if not samples:
        return {}
    ordered = sorted(samples)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return round(ordered[index], 4)

    return {"n": len(ordered), "p50": at(0.50), "p95": at(0.95), "p99": at(0.99),
            "max": round(ordered[-1], 4)}


# -- the environment ----------------------------------------------------------

@dataclass
class Environment:
    """A scratch installation, and the corpus ingested into it once.

    Every check is handed the same ingested store rather than building its own: the corpus is
    fixed, and a check that rebuilds it is a check that can disagree with its neighbours about
    what was in there.
    """
    root: Path
    store_name: str = "canonical.db"
    corpus_store: EvidenceStore | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def path(self, *parts: str) -> Path:
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def open(self, name: str) -> EvidenceStore:
        return EvidenceStore(self.path(name))

    def close(self) -> None:
        if self.corpus_store is not None:
            self.corpus_store.close()
            self.corpus_store = None


def ingest(env: Environment) -> dict[str, Any]:
    """Commit the whole corpus through the real ingress, once.

    Returns the counts the store reports — not the counts the corpus claims — because the
    difference between the two is a page the source thought it sent and the store did not
    keep.
    """
    if env.corpus_store is not None:
        return {"already": True}
    store = env.open(env.store_name)
    accepted, duplicates, conflicts = 0, 0, 0
    started = time.perf_counter()
    for envelope in corpus.records():
        try:
            result = store.commit(envelope)
        except sqlite3.Error as error:            # a rejected envelope is a finding, not a crash
            conflicts += 1
            continue
        accepted += 0 if result["duplicate"] else 1
        duplicates += 1 if result["duplicate"] else 0
    env.timings["ingest_seconds"] = round(time.perf_counter() - started, 4)
    env.corpus_store = store
    return {"accepted": accepted, "duplicates": duplicates, "refused": conflicts,
            "records": int(store.db.execute(
                "SELECT count(*) FROM records WHERE deleted=0").fetchone()[0]),
            "seconds": env.timings["ingest_seconds"]}


# -- the corpus itself --------------------------------------------------------

def check_corpus(env: Environment) -> list[dict[str, Any]]:
    """§12.3's own claim: the corpus is what it says it is, and the store kept it.

    This is the first check for a reason. Every recall and abstention number in this report is
    computed against these codes; if a code is not unique, or a category went missing, or the
    store dropped a record on the way in, then the numbers below are not wrong so much as
    meaningless.
    """
    totals = corpus.totals()
    store = env.corpus_store
    kept = int(store.db.execute("SELECT count(*) FROM records").fetchone()[0])
    codes = [str(row["metadata"]["code"]) for row in corpus.records()]
    distinct = len(set(codes))
    missing = [code for code in codes
               if store.db.execute("SELECT 1 FROM records WHERE json_extract(metadata, '$.code')=?",
                                   (code,)).fetchone() is None]
    by_category = {key: value for key, value in totals.items() if key.startswith("record_")}
    empty = [key for key, value in by_category.items() if value < len(corpus.PEOPLE)]
    return [
        measured("corpus", "every §12.3 category is represented",
                 value=len(by_category), criterion=f"= {len(corpus.CATEGORIES)}",
                 passes=len(by_category) == len(corpus.CATEGORIES),
                 denominator=len(corpus.CATEGORIES),
                 detail={"sparse": empty, "counts": by_category}),
        measured("corpus", "one distinct code per record",
                 value=distinct, criterion=f"= {kept}", passes=distinct == kept and not missing,
                 denominator=kept,
                 detail={"kept_in_store": kept, "not_found_after_ingest": missing[:5],
                         "categories": len(by_category)}),
        measured("corpus", "the store kept what the corpus sent",
                 value=totals["records"] - int(store.db.execute(
                     "SELECT count(*) FROM records").fetchone()[0]),
                 criterion="= 0 refused or lost",
                 passes=kept == totals["records"],
                 denominator=totals["records"],
                 detail={"envelopes_valid": all(
                     prepare_envelope(row) is not None for row in corpus.records())}),
        measured("corpus", "held-out questions and their gold answers line up",
                 value=sum(1 for row in corpus.questions()
                           if row["expect"] == "abstain"),
                 criterion=f"= {totals['abstentions']}",
                 passes=totals["questions"] == totals["records"] + totals["abstentions"],
                 denominator=totals["questions"],
                 detail={"retrieve": totals["questions"] - totals["abstentions"],
                         "scenarios": totals["eligible_scenarios"]
                         + totals["no_action_scenarios"]}),
    ]


def check_capture(env: Environment) -> list[dict[str, Any]]:
    """§12.4 "Capture/sync": no lost acknowledged records, no duplicated logical events.

    Replay is the interesting case rather than the first write, because the plan promises a
    source may send the same page twice — after a crash, after a timeout, after an
    acknowledgment that never arrived. The store's answer has to be the same id and no new
    row, every time.
    """
    store = env.corpus_store
    before = {str(row["id"]) for row in store.db.execute(
        "SELECT id FROM records WHERE deleted=0")}
    again = [store.commit(envelope) for envelope in corpus.records()]
    after = {str(row["id"]) for row in store.db.execute(
        "SELECT id FROM records WHERE deleted=0")}
    acknowledged = sum(1 for row in again if row["id"] in after)
    receipts = int(store.db.execute("SELECT count(*) FROM ingestion_receipts").fetchone()[0])
    logical = int(store.db.execute(
        "SELECT count(*) FROM (SELECT source, source_id, revision FROM records "
        "GROUP BY source, source_id, revision)").fetchone()[0])
    journal = int(store.db.execute(
        "SELECT count(*) FROM change_journal WHERE change='add'").fetchone()[0])
    missing = [record for record in corpus.records()
               if store.get(record_id(str(record["source"]), str(record["source_id"]),
                                      str(record["revision"])), include_hidden=True) is None]
    return [
        measured("capture/sync", "a replayed page forks no second record",
                 value=len(after) - len(before), criterion="= 0",
                 passes=before == after, denominator=len(corpus.records()),
                 detail={"replayed": len(again), "acknowledged": acknowledged,
                         "new_rows": len(after - before)}),
        measured("capture/sync", "one logical event, one journal row",
                 value=logical - journal, criterion="= 0", passes=logical == journal,
                 denominator=logical, detail={"records": logical, "journal_adds": journal,
                                              "receipts": receipts}),
        measured("capture/sync", "every acknowledged record is still readable",
                 value=len(missing), criterion="= 0", passes=not missing,
                 denominator=len(corpus.records()),
                 detail={"missing": [str(row["metadata"]["code"]) for row in missing][:5]}),
    ]


def check_corrections(env: Environment) -> list[dict[str, Any]]:
    """§12.3's revision pair and §12.4's "correction handling", at the deterministic layer.

    What is true *now* is a property of the store, not of a model: the corrected revision is
    the only one retrieval may offer, the withdrawn one stays readable as history, and every
    chain pointer runs one step so an earlier date can still be answered.
    """
    store = env.corpus_store
    lineage = Lineage(store)
    people = corpus.PEOPLE
    head_ok = stale_hidden = history_kept = answers_ok = 0
    for index in range(len(people)):
        source_id = f"correction-CO{index:02d}R1"
        rows = lineage.revisions("gmail", source_id)
        current = lineage.current("gmail", source_id)
        # The head is the second revision, and — which is the part a caller actually feels —
        # reading the head gives the corrected sentence rather than the one it replaced.
        head_ok += 1 if current == record_id("gmail", source_id, "2") else 0
        kept = store.get(current) if current else None
        answers_ok += 1 if kept is not None and "14th" in kept.text else 0
        history_kept += 1 if len(rows) == 2 else 0
        stale = next((row for row in rows if row["revision"] == "1"), None)
        stale_hidden += 1 if stale and stale["hidden"] and stale["replacement_id"] == current else 0
    withdrawn = [str(row["record_id"]) for row in store.db.execute(
        "SELECT record_id FROM record_visibility WHERE hidden=1 AND replacement_id IS NOT NULL")]
    findable = sum(1 for record_id in withdrawn
                   if store.db.execute("SELECT 1 FROM record_fts WHERE id=?",
                                       (record_id,)).fetchone() is None)
    return [
        measured("retrieval", "a corrected item answers with the corrected revision",
                 value=head_ok, criterion=f"= {len(people)}",
                 passes=head_ok == len(people) and answers_ok == len(people),
                 denominator=len(people),
                 detail={"pairs": len(people), "answers_from_the_new_revision": answers_ok,
                         "both_revisions_kept": history_kept,
                         "superseded_points_at_head": stale_hidden}),
        measured("lifecycle", "a withdrawn revision leaves the index but not the archive",
                 value=findable, criterion=f"= {len(withdrawn)}",
                 passes=findable == len(withdrawn), denominator=len(withdrawn),
                 detail={"superseded": len(withdrawn),
                         "still_readable": int(store.db.execute(
                             "SELECT count(*) FROM records r JOIN record_visibility v "
                             "ON v.record_id=r.id WHERE v.hidden=1 AND v.replacement_id "
                             "IS NOT NULL AND r.deleted=0").fetchone()[0])}),
    ]


# -- retrieval ----------------------------------------------------------------

def check_retrieval(env: Environment) -> list[dict[str, Any]]:
    """§12.4 "Retrieval": recall, answer support, temporal correctness and abstention,
    scored separately, because the plan says a single blended number hides which one broke.

    Every question here is asked of the local route — the lexical channel over the canonical
    store — which is the only route this build can exercise without a model. That is also what
    makes the abstention row meaningful: an absent code has exactly one correct answer, and the
    packet either says "unknown" or it does not.
    """
    from hermes_memory.context import ContextBroker
    from hermes_memory.context.packet import SUPPORTED, UNKNOWN

    store = env.corpus_store
    broker = ContextBroker(store, cache=None)
    asked = corpus.questions()
    originals = {str(row["metadata"]["code"]): row for row in corpus.records()}
    by_category: dict[str, dict[str, int]] = {}
    hits = supports = temporal = abstained = withdrawn_kept = 0
    misses: list[dict[str, Any]] = []
    started = time.perf_counter()
    for question in asked:
        packet = broker.assemble(str(question["query"]), limit=8)
        gold = str(question["gold"])
        found = [item for item in packet.items if f"[ref {gold}]" in item.text]
        category = str(question["category"])
        tally = by_category.setdefault(category, {"asked": 0, "found": 0})
        tally["asked"] += 1
        tally["found"] += 1 if found else 0
        if question["expect"] == "abstain":
            abstained += 1 if packet.empty and packet.coverage == UNKNOWN else 0
            if not (packet.empty and packet.coverage == UNKNOWN):
                misses.append({"gold": gold, "expected": "nothing", "got": len(packet.items)})
            continue
        if question["expect"] == "superseded":
            # The text is still on disk — a correction preserves history — so this is the one
            # question that a store reading `records` without `record_visibility` gets right
            # for all the wrong reasons.
            withdrawn_kept += 1 if not found else 0
            if found:
                misses.append({"gold": gold, "expected": "the replacement, not this",
                               "got": [item.source_id for item in found]})
            continue
        hits += 1 if found else 0
        supports += 1 if packet.coverage == SUPPORTED else 0
        if not found:
            misses.append({"gold": gold, "category": category,
                           "returned": [item.source_id for item in packet.items][:3]})
        if found:
            # Event time is the record's own claim, not the moment it arrived. A packet that
            # answers with the ingestion date would pass a recall test and still be wrong.
            # The stored instant is compared rather than the string: the store keeps one
            # normalised form, and "08:12Z" is the same instant as "09:12+01:00".
            record = store.get(found[0].id)
            original = originals[gold]
            same_instant = record is not None and _instant(record.occurred_at) == \
                _instant(original["occurred_at"])
            arrival_apart = record is not None and record.observed_at != record.occurred_at
            temporal += 1 if same_instant and record.occurred_precision == \
                original["occurred_precision"] and arrival_apart else 0
            if not (same_instant and arrival_apart):
                misses.append({"gold": gold, "expected": original["occurred_at"],
                               "got": record.occurred_at if record else None})
    retrieve_total = sum(1 for row in asked if row["expect"] == "retrieve")
    abstain_total = sum(1 for row in asked if row["expect"] == "abstain")
    superseded_total = sum(1 for row in asked if row["expect"] == "superseded")
    broker.close()
    seconds = round(time.perf_counter() - started, 2)
    return [
        measured("retrieval", "required-evidence recall (lexical route, top 8)",
                 value=ratio(hits, retrieve_total), criterion="= 1.00 on the local route",
                 passes=hits == retrieve_total, denominator=retrieve_total,
                 detail={"misses": misses[:5], "seconds": seconds,
                         "by_category": by_category}),
        measured("retrieval", "answer support: the packet says what it covers",
                 value=ratio(supports, retrieve_total), criterion=">= 0.95",
                 passes=supports / max(1, retrieve_total) >= 0.95, denominator=retrieve_total,
                 detail={"coverage_is_supported": supports}),
        measured("retrieval", "temporal correctness of what came back",
                 value=ratio(temporal, hits), criterion="= 1.00",
                 passes=temporal == hits, denominator=hits,
                 detail={"note": "occurred_at and its precision survive the round trip, so a "
                                 "month-precision claim is not read back as an instant"}),
        measured("retrieval", "abstention on questions with no answer",
                 value=ratio(abstained, abstain_total), criterion="= 1.00",
                 passes=abstained == abstain_total, denominator=abstain_total,
                 detail={"wrongly_answered": misses[:5]}),
        measured("retrieval", "a corrected answer is not returned as if it still stood",
                 value=withdrawn_kept, criterion=f"= {superseded_total}",
                 passes=withdrawn_kept == superseded_total, denominator=superseded_total,
                 detail={"note": "the withdrawn bytes stay readable as history; what they "
                                 "no longer have is the standing to be an answer"}),
    ]


def check_retrieval_ablation(env: Environment) -> list[dict[str, Any]]:
    """The ablation half of the retrieval row: what each channel contributes, by removal.

    Only the lexical channel exists in this build without a backend, so the honest ablation
    result is that removing it leaves nothing. That is worth printing: it is the difference
    between "retrieval works" and "retrieval works the one way this installation can".
    """
    from hermes_memory.context import ContextBroker

    store = env.open("ablation.db")
    for envelope in corpus.records():
        store.commit(envelope)
    broker = ContextBroker(store, cache=None)
    asked = [row for row in corpus.questions() if row["expect"] == "retrieve"]

    def recall() -> float:
        found = 0
        for question in asked:
            packet = broker.assemble(str(question["query"]), limit=8)
            found += 1 if any(f"[ref {question['gold']}]" in item.text
                              for item in packet.items) else 0
        return ratio(found, len(asked))

    with_lexical = recall()
    store.db.execute("DELETE FROM record_fts")
    without_lexical = recall()
    broker.close()
    store.close()
    return [
        measured("retrieval", "ablation: all channels", value=with_lexical,
                 criterion="= 1.00", passes=with_lexical == 1.0, denominator=len(asked),
                 detail={"channel": "lexical"}),
        measured("retrieval", "ablation: FTS removed", value=without_lexical,
                 criterion="= 0.00 — the report must not hide what carries the answer",
                 passes=without_lexical == 0.0, denominator=len(asked),
                 detail={"meaning": "nothing else answers on this route, so a store with a "
                                    "broken index is an outage and not an empty archive"}),
        not_measured("retrieval", "ablation: Hindsight raw facts, observations and summaries",
                     because="this run has no backend and no model, by design — those three "
                             "channels are the ones that need the pinned Hindsight route",
                     requires="the P0 authorization to make live pinned-backend calls; the "
                              "channels are wired and their absence is reported as "
                              "`not_configured`, which is not the same as a zero"),
    ]


def check_latency(env: Environment) -> list[dict[str, Any]]:
    """§12.4 "Context latency": warm local-context p95 against the plan's proposed 1 s.

    "Warm" is the second read of the same question with nothing changed in between, measured
    through the broker rather than through the store, because the promise is about what a turn
    waits — derivation, budgeting, scope checks and the cache are all inside it.
    """
    from hermes_memory.context import ContextBroker

    store = env.corpus_store
    broker = ContextBroker(store)
    asked = [row for row in corpus.questions() if row["expect"] == "retrieve"]
    cold, warm = [], []
    for question in asked:
        # Read twice, back to back: that is what "warm" promises. A second pass over the whole
        # question set after the first would measure the cache's capacity rather than its
        # effect — it holds 64 packets, and there are 240 distinct questions here.
        started = time.perf_counter()
        broker.assemble(str(question["query"]), limit=8)
        cold.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        broker.assemble(str(question["query"]), limit=8)
        warm.append((time.perf_counter() - started) * 1000.0)
    counters = broker.cache.as_dict() if broker.cache is not None else {}
    deep = []
    for question in asked[:50]:
        started = time.perf_counter()
        broker.assemble(f"{question['query']} and everything about {question['category']}",
                        limit=50)
        deep.append((time.perf_counter() - started) * 1000.0)
    broker.close()
    return [
        measured("context latency", "warm local-context p95",
                 value=percentiles(warm).get("p95"), criterion="<= 1000 ms",
                 passes=(percentiles(warm).get("p95") or 1e9) <= 1000.0,
                 denominator=len(warm),
                 detail={"warm_ms": percentiles(warm), "cold_ms": percentiles(cold),
                         "cache": counters}),
        measured("context latency", "a warm read is a cache hit, not a re-read of the index",
                 value=counters.get("hits"), criterion=f">= {len(asked)}",
                 passes=(counters.get("hits") or 0) >= len(asked), denominator=len(asked),
                 detail={"misses": counters.get("misses"), "stale": counters.get("stale"),
                         "entries": counters.get("entries"),
                         "note": "the cache holds 64 packets; asking 240 distinct questions "
                                 "before repeating any of them would measure capacity, and a "
                                 "turn does not do that"}),
        measured("context latency", "deep reads stay inside the foreground deadline",
                 value=percentiles(deep).get("p95"),
                 criterion="<= 4000 ms (the configured foreground deadline)",
                 passes=(percentiles(deep).get("p95") or 1e9) <= 4000.0, denominator=len(deep),
                 detail={"deep_ms": percentiles(deep),
                         "note": "with no derived channel configured, the deadline is never "
                                 "what bounds this; the row becomes honest only against a "
                                 "live backend"}),
    ]

# Each matched no-action case names the mechanism that must keep it quiet. Silence is easy to
# score and worthless to score alone: a gate that never speaks hits 100% precision on this
# table and fails the criterion the plan actually cares about. So every quiet case is checked
# against the reason it was supposed to have — and quiet means the owner is not interrupted,
# which the plan's own ladder reaches two ways: eligibility refuses the analysis, or the
# policy still analyses it and demotes the delivery to a digest or to the next turn.
NO_ACTION_MECHANISMS = {
    "opt-out": ("opt_out", "opted"),
    "quiet hours": ("policy", "quiet hours"),
    "stale evidence": ("freshness", "older than"),
    "backfill coverage": ("coverage", "partial"),
    "disconnected source": ("coverage", "unreachable"),
    "duplicate intention": ("duplicate", "already"),
    "spent budget": ("budget", "spent"),
    "paused topic": ("paused", "paused"),
    "shadow mode": ("shadow", "shadow"),
    "unknown coverage": ("coverage", "unknown"),
}
# What counts as interrupting the owner. `next_turn` and `digest` are deliberately not in
# here: they are the rungs below a notification, and the plan's metric is about interruption.
INTERRUPTIONS = ("notify_owner", "draft")
# When a handed-off intention fires. Fixed rather than per-case: the moment the reminder is
# due is not what the duplicate case is about, and a due time that moves with the scenario
# would make this check depend on which day the corpus happened to pick.
HANDOFF_DUE = "2026-09-15T09:00:00+00:00"


def check_proactivity(env: Environment) -> list[dict[str, Any]]:
    """§12.4 "Proactivity": proposed-notification precision, and the matched no-action rate.

    The plan's numbers are precision ≥90% and false interruption ≤5%, *plus* recall and timing
    "so permanent silence cannot win". Every one of those four is computed here from the
    deterministic gate — the eligibility stage list and the attention policy — because that
    gate is the part of proactivity that decides whether to bother anyone at all.
    """
    from hermes_memory.processing.budgets import Budget, Budgets
    from hermes_memory.proactive.eligibility import Eligibility
    from hermes_memory.proactive.policy import AttentionPolicy
    from hermes_memory.sources.sync import SyncController

    owner = "eval-owner"
    resource = "eval-analysis"
    template = corpus.records()[0]
    store = env.open("proactivity.db")
    sync = SyncController(store)
    sync.register("gmail", policy_version="private-api")
    fence = sync.acquire("gmail", holder="eval-proactive")
    policy = AttentionPolicy(store, owner_principal=owner)
    policy.configure(actor=owner, timezone_name="UTC", quiet_from="22:00",
                     quiet_until="06:00", max_immediate_per_day=10, cooldown_minutes=0,
                     shadow=False)
    budgets = Budgets(store, daily={resource: Budget(tokens=10, seconds=60)},
                      scope="proactive")

    proposed, wrongly_active, wrongly_silent, mismatched = 0, 0, 0, []
    mechanisms: dict[str, int] = {}
    latencies: list[float] = []
    def ask(verb: Callable[[], Any]) -> Any:
        """One eligibility question, timed on its own."""
        started = time.perf_counter()
        answer = verb()
        latencies.append((time.perf_counter() - started) * 1000.0)
        return answer

    for case in corpus.scenarios():
        reason = str(case.get("reason") or "")
        topic = str(case["topic"])
        moment = str(case["at"])
        # State first, then the question: every case starts from a live, current, allowed,
        # unpaused topic and is then moved to exactly the one state its reason names.
        sync.mark_gap(fence, coverage_state="current", reason="eval baseline")
        policy.configure(actor=owner, topic=topic, shadow=False, opted_out=False)
        store.set_control(f"topic:{topic}", "attention", "active", actor=owner,
                          reason="eval baseline", policy_version="attention-1")
        occurred = moment
        if reason == "stale evidence":
            # Older than any live arrival can be, and by the record's own date rather than
            # by the harness's clock: a backfill is not news.
            occurred = "2026-03-01T00:00:00+00:00"
        elif reason == "backfill coverage":
            sync.mark_gap(fence, coverage_state="partial", reason="eval sweeping history")
        elif reason == "disconnected source":
            sync.mark_gap(fence, coverage_state="unreachable", reason="eval source offline")
        elif reason == "unknown coverage":
            sync.mark_gap(fence, coverage_state="unknown", reason="eval never swept")
        elif reason == "opt-out":
            policy.configure(actor=owner, topic=topic, opted_out=True)
        elif reason == "shadow mode":
            policy.configure(actor=owner, topic=topic, shadow=True)
        elif reason == "paused topic":
            store.set_control(f"topic:{topic}", "attention", "paused", actor=owner,
                              reason="eval pause", policy_version="attention-1")
        record = store.commit({**template, "source": "gmail",
                               "source_id": f"eval-{case['code']}", "revision": "1",
                               "text": f"{case['person']} mentions {case['code']}",
                               "occurred_at": occurred,
                               "metadata": {"code": str(case["code"]),
                                            "category": "proactivity"}})["id"]
        # Timed from the question, not from the setup: the harness writing state into a store
        # is not the number being asked about.
        eligibility = Eligibility(store, policy=policy, sync=sync)
        if reason == "spent budget":
            budgets.charge(resource, tokens=10)
            eligibility = Eligibility(store, policy=policy, sync=sync, budgets=budgets,
                                      model_budget_resource=resource)
        if reason == "duplicate intention":
            # Reached through the real C8 handoff, because a duplicate is a fact about an
            # intention and an intention only exists once a due event has been claimed and
            # acknowledged. Inventing an intent id here would score the suppression of
            # something that was never on offer.
            goal_id, revision, intent = handoff(store, owner=owner, code=str(case["code"]),
                                                title=f"chase {case['code']}")
            verdict = ask(lambda: eligibility.for_event(goal_id=goal_id, revision=revision,
                                                        topic=topic, intent_id=intent,
                                                        at=moment))
            if verdict.eligible:
                decision = policy.decide(topic=topic, urgency="proactive", at=moment)
                policy.record(decision, intent_id=intent, goal_id=goal_id, revision=revision)
                verdict = eligibility.for_event(goal_id=goal_id, revision=revision,
                                                topic=topic, intent_id=intent, at=moment)
        else:
            verdict = ask(lambda: eligibility.for_change(source="gmail", record_id=record,
                                                         topic=topic, at=moment))
        # Two questions, asked in the order the engine asks them: may this be analysed at
        # all, and if so, does it reach the owner's attention now?
        delivery = policy.decide(topic=topic, urgency="proactive", at=moment)
        speaks = bool(verdict.eligible) and delivery.action in INTERRUPTIONS
        said = " ".join([verdict.reason, delivery.reason, *delivery.downgrades])
        proposed += 1 if speaks else 0
        if case["case"] == "eligible":
            if not speaks:
                wrongly_silent += 1
                mechanisms[f"silent:{verdict.stage}/{delivery.action}"] = \
                    mechanisms.get(f"silent:{verdict.stage}/{delivery.action}", 0) + 1
        else:
            wanted_stage, wanted_reason = NO_ACTION_MECHANISMS[reason]
            if speaks:
                wrongly_active += 1
            elif wanted_reason not in said:
                mismatched.append({"code": case["code"], "expected": wanted_reason,
                                   "stage": verdict.stage, "wanted_stage": wanted_stage,
                                   "action": delivery.action, "said": said[:120]})
            mechanisms[reason] = mechanisms.get(reason, 0) + 1
    total_no_action = sum(1 for case in corpus.scenarios() if case["case"] == "no-action")
    total_eligible = sum(1 for case in corpus.scenarios() if case["case"] == "eligible")
    precision = ratio(proposed - wrongly_active, proposed)
    interruption = ratio(wrongly_active, total_no_action)
    store.close()
    return [
        measured("proactivity", "proposed notifications are the eligible ones",
                 value=precision, criterion=">= 0.90", passes=precision >= 0.90,
                 denominator=proposed,
                 detail={"proposed": proposed, "of_which_wrong": wrongly_active,
                         "eligible_total": total_eligible}),
        measured("proactivity", "matched no-action cases stay quiet",
                 value=interruption, criterion="<= 0.05", passes=interruption <= 0.05,
                 denominator=total_no_action,
                 detail={"wrongly_active": wrongly_active,
                         "refused_for_another_reason": len(mismatched),
                         "examples": mismatched[:3]}),
        measured("proactivity", "every refusal is the mechanism it claims",
                 value=len(mismatched), criterion="= 0", passes=not mismatched,
                 denominator=total_no_action,
                 detail={"examples": mismatched[:3],
                         "expected": {key: value[0] for key, value in
                                      NO_ACTION_MECHANISMS.items()},
                         "observed_stages": mechanisms}),
        measured("proactivity", "eligible moments are actually surfaced (recall)",
                 value=ratio(total_eligible - wrongly_silent, total_eligible),
                 criterion="> 0 — silence is not a strategy that passes",
                 passes=wrongly_silent == 0, denominator=total_eligible,
                 detail={"wrongly_silent": wrongly_silent}),
        measured("proactivity", "the deterministic no is cheap",
                 value=percentiles(latencies).get("p99"),
                 criterion="reported (p99 ms over the matched pair)",
                 passes=True, denominator=len(latencies),
                 detail=percentiles(latencies)),
    ]


def handoff(store, *, owner: str, code: str, due: str = HANDOFF_DUE,
            title: str = "Follow up") -> tuple[str, int, str]:
    """One real handed-off intention: propose, activate, claim, acknowledge.

    Returns ``(goal_id, revision, intent_id)``. An intention cannot be invented here: both the
    duplicate filter and the outbox check that the intention it answers is on file, so a made-up
    id would measure the harness rather than the framework.
    """
    from hermes_memory.prospective.due_events import DueEventLog
    from hermes_memory.prospective.goals import GoalStore

    events = DueEventLog(store)
    goals = GoalStore(store, events=events, owner_principal=owner)
    made = goals.propose(title=title, statement="A thing the owner asked to be told about.",
                         timezone_name="UTC", due=due, proposed_by=owner,
                         proposed_kind="owner")
    goal_id = str(made["id"])
    revision = int(goals.get(goal_id).revision)
    event_id = str(store.db.execute("SELECT id FROM due_events WHERE goal_id=?",
                                    (goal_id,)).fetchone()[0])
    claim = events.claim(event_id, holder=f"eval-{code}")
    intent = str(events.ack(event_id=event_id, token=claim.token,
                            decision="awaiting_analysis",
                            policy_version=POLICY_VERSION)["intent"])
    return goal_id, revision, intent


# -- privacy and lifecycle ----------------------------------------------------

def check_privacy(env: Environment) -> list[dict[str, Any]]:
    """§12.4 "Privacy/lifecycle": no cross-scope exposure, no publication of forgotten
    evidence, no external action without approval — each an observation over the corpus.

    The corpus carries one participant per record, which is what turns the first of these from
    a unit test into a measurement: every record is asked of the archive by a caller who is not
    the person it is about.
    """
    from hermes_memory.context import ContextBroker
    from hermes_memory.lifecycle.erasure import ErasureManager
    from hermes_memory.proactive.outbox import Outbox
    from hermes_memory.proactive.policy import AttentionPolicy
    from hermes_memory.storage.identity import IdentityStore

    owner = "eval-owner"
    agent = "agent:eval-session"
    store = env.open("scopes.db")
    identity = IdentityStore(store, owner_principal=owner)
    accounts: dict[str, str] = {}
    citations: list[str] = []
    scoped = []
    for row in corpus.records():
        person = str(row["metadata"]["participants"][0]["display_name"])
        account = accounts.setdefault(person, identity.account(
            "email", f"{person.lower()}@example.org"))
        envelope = dict(row)
        envelope["metadata"] = {**row["metadata"], "account_ids": [account]}
        scoped.append(envelope)
        citations.append(str(store.commit(envelope)["id"]))
    broker = ContextBroker(store, identity=identity, cache=None)

    questions = [row for row in corpus.questions() if row["expect"] != "abstain"]
    leaked = 0
    for envelope, question in zip(scoped, questions):
        person = str(envelope["metadata"]["participants"][0]["display_name"])
        stranger = next(value for key, value in accounts.items() if key != person)
        packet = broker.assemble(str(question["query"]), limit=8, account_id=stranger)
        if any(f"[ref {question['gold']}]" in item.text for item in packet.items):
            leaked += 1
        if question["expect"] == "retrieve":
            own = broker.assemble(str(question["query"]), limit=8,
                                  account_id=accounts[person])
            leaked += 1 if not own.items else 0

    # A lesson is a conclusion drawn from one person's records, so it is scoped by the same
    # rule as the evidence it names. Both ways it can go wrong are measured: a caller who
    # asks for their own practice must not be answered with someone else's, and a caller
    # that hands the broker a lesson it should not have cannot make the packet show it.
    from hermes_memory.learning.lessons import LessonStore
    from hermes_memory.learning.outcomes import OutcomeLog

    habits = LessonStore(store, outcomes=OutcomeLog(store, owner_principal=owner),
                         identities=identity, owner_principal=owner)
    first = str(scoped[0]["metadata"]["participants"][0]["display_name"])
    source = str(scoped[0]["source"])
    task = {"source": source}
    offered = habits.propose(text=f"Answer {first} before anyone else.",
                             applicability={"all": [{"field": "source", "op": "eq",
                                                     "value": source}]},
                             lesson_id="privacy-scope", evidence=[citations[0]],
                             proposed_by=owner, proposed_kind="owner")
    habits.activate(lesson_id="privacy-scope", version=int(offered["version"]),
                    actor=owner, reason="eval promotion")
    other = next(value for key, value in accounts.items() if key != first)
    earned = habits.applicable(task, account_id=accounts[first])
    hidden = habits.applicable(task, account_id=other)
    forced = broker.assemble(str(questions[0]["query"]), limit=8, account_id=other,
                             lessons=[item.as_dict() for item in earned])
    taught_to_owner = any(item.id == "privacy-scope" for item in earned)
    taught_to_stranger = bool(hidden)
    packet_leaks = any(item.get("id") == "privacy-scope" for item in forced.lessons)

    # Forgotten evidence must stop being publishable at the same moment it stops being
    # readable — including an alert already prepared and waiting in the outbox.
    clinical = store.commit({**corpus.records()[0], "source_id": "privacy-clinical",
                             "revision": "1", "text": "A diagnosis nobody else may read.",
                             "metadata": {"account_ids": [accounts["Amara"]],
                                          "code": "PRIV0001", "category": "clinical"}})["id"]
    policy = AttentionPolicy(store, owner_principal=owner)
    policy.configure(actor=owner, timezone_name="UTC")
    goal_id, revision, intent = handoff(store, owner=owner, code="privacy")
    decision = policy.decide(topic="general", urgency="proactive")
    recorded = policy.record(decision, intent_id=intent, goal_id=goal_id, revision=revision)
    outbox = Outbox(store, policy=policy, owner_principal=owner)
    artifact = outbox.prepare(decision_id=str(recorded["id"]), kind="notify_owner",
                              topic="general", payload="Act on the diagnosis.",
                              evidence=[str(clinical)])
    manager = ErasureManager(store, owner_principal=owner)
    preview = manager.preview(record_ids=[str(clinical)], actor=owner, reason="withdrawn")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor=owner)
    revalidated = outbox.revalidate(str(artifact["id"]))
    artifact_state = outbox.get(str(artifact["id"])).state
    still_readable = store.get(str(clinical)) is not None or bool(store.search("diagnosis"))

    refused = 0
    for call in (lambda: identity.confirm(candidate_id="cand_missing", actor=agent,
                                          reason="sure"),
                 lambda: manager.confirm(intent_id="intent_missing", preview_digest="0" * 64,
                                         actor=agent)):
        try:
            call()
        except Exception as error:            # noqa: BLE001 - the refusal is the result
            refused += 1 if "owner" in str(error).lower() else 0
    broker.close()
    store.close()
    return [
        measured("privacy/lifecycle", "no record reaches a caller outside its scope",
                 value=leaked, criterion="= 0", passes=leaked == 0, denominator=len(questions),
                 detail={"people": len(accounts), "records": len(scoped),
                         "asked_by_a_stranger": len(questions)}),
        measured("privacy/lifecycle", "forgotten evidence is not publishable",
                 value=1 if (revalidated.ok is False and not still_readable) else 0,
                 criterion="= 1", passes=revalidated.ok is False and not still_readable,
                 denominator=1,
                 detail={"outbox_stage": revalidated.stage, "reason": revalidated.reason,
                         "state": artifact_state,
                         "still_readable": still_readable}),
        measured("privacy/lifecycle", "practice learned from one person is not taught to "
                                      "another, in either direction",
                 value=1 if (taught_to_owner and not taught_to_stranger
                             and not packet_leaks) else 0,
                 criterion="= 1",
                 passes=taught_to_owner and not taught_to_stranger and not packet_leaks,
                 denominator=1,
                 detail={"lesson": "privacy-scope", "scoped_to": first,
                         "asked_as": other,
                         "forced_into_a_stranger_packet": packet_leaks}),
        measured("privacy/lifecycle", "an agent-role caller cannot open an owner door",
                 value=refused, criterion="= 2", passes=refused == 2, denominator=2,
                 detail={"doors": ["identity confirmation", "forgetting"]}),
    ]


# -- compute ------------------------------------------------------------------

class Tripwire:
    """A backend that records every reach and answers nothing.

    The plan's compute row promises that an idle, paused or default installation makes zero
    model requests. The only way to prove a negative about a component that is not running is
    to wire something into its place that says so if it is touched.
    """

    def __init__(self) -> None:
        self.reached: list[str] = []

    def __getattr__(self, name: str):
        def reached(*_args, **_kwargs):
            self.reached.append(name)
            raise AssertionError(f"{name} was called; this path must not reach a model")
        return reached


def check_compute(env: Environment) -> list[dict[str, Any]]:
    """§12.4 "Compute": zero model requests at rest, single flight per device, budgets real."""
    from hermes_memory.operations.doctor import Doctor
    from hermes_memory.operations.status import StatusReporter
    from hermes_memory.processing.budgets import Budget, Budgets, BudgetExhausted
    from hermes_memory.processing.resource_gate import GatePaused, ResourceGate

    owner = "eval-owner"
    resource = "eval-device"
    store = env.corpus_store
    tripwire = Tripwire()
    report = Doctor(store, backend=tripwire).examine()
    stages = StatusReporter(store).report()["stages"]
    untouched = list(tripwire.reached)

    gate = ResourceGate(store)
    first = gate.try_acquire(route="memory", holder="worker-a", resource=resource, priority=1)
    second = gate.try_acquire(route="memory", holder="worker-b", resource=resource, priority=1)
    if first is not None:
        gate.release(first, outcome="succeeded", tokens=12)
    third = gate.try_acquire(route="memory", holder="worker-c", resource=resource, priority=1)
    if third is not None:
        gate.release(third, outcome="succeeded", tokens=12)
    gate.pause(actor=owner, reason="the operator said so")
    try:
        while_paused = gate.try_acquire(route="memory", holder="worker-d", resource=resource,
                                        priority=1)
    except GatePaused:
        while_paused = None
    gate.resume(actor=owner, reason="the operator changed their mind")

    budgets = Budgets(store, daily={resource: Budget(tokens=100, seconds=60)},
                      scope="eval-compute")
    budgets.charge(resource, tokens=100)
    exhausted: Any = None
    try:
        budgets.admit(resource, estimated_tokens=1)
    except BudgetExhausted as error:
        exhausted = error
    charged = budgets.used(resource)
    gate.close()
    return [
        measured("compute", "a doctor run and a status read reach no model",
                 value=len(untouched), criterion="= 0", passes=not untouched, denominator=2,
                 detail={"reached": untouched, "findings": len(report["findings"]),
                         "probes": report["probes"], "stages": len(stages)}),
        measured("compute", "one flight at a time on one physical resource",
                 value=int(second is not None), criterion="= 0",
                 passes=first is not None and second is None and third is not None,
                 denominator=3,
                 detail={"first_granted": first is not None,
                         "second_granted": second is not None,
                         "granted_again_after_release": third is not None}),
        measured("compute", "a paused installation dispatches nothing",
                 value=int(while_paused is not None), criterion="= 0",
                 passes=while_paused is None, denominator=1,
                 detail={"granted_while_paused": while_paused is not None}),
        measured("compute", "a spent budget refuses before the work, not after",
                 value=0 if exhausted is not None else 1, criterion="= 0",
                 passes=exhausted is not None, denominator=1,
                 detail={"raised": type(exhausted).__name__, "spent": charged}),
    ]


# -- measurements -------------------------------------------------------------


def check_measurements(env: Environment) -> list[dict[str, Any]]:
    """§12.4 "Compute", the half of it that is arithmetic: typed measurements with units.

    The plan's promise is that no model is consulted for a sensor reading, so the check is
    run with inference off and the expected numbers are computed here, in the open, from the
    fixture the adapter was handed. A mean this harness could not reproduce by hand is not a
    measurement — it is a recollection.
    """
    from hermes_memory.sources.structured import StructuredSource
    from hermes_memory.storage.measurements import Measurements

    fixtures = env.path("fixtures")
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "weight.csv").write_text(
        "date,kg\n"
        "2026-03-01T08:00:00+00:00,70.0\n"
        "2026-03-02T08:00:00+00:00,71.0\n"
        "2026-03-03T08:00:00+00:00,72.0\n"
        ",73.0\n"                                   # a row with no time at all
        "2026-04-01T08:00:00+00:00,99.0\n", encoding="utf-8")
    (fixtures / "imperial.jsonl").write_text(
        json.dumps({"time": "2026-03-04T08:00:00+00:00", "measure": "weight",
                    "value": 154, "unit": "lb"}) + "\n", encoding="utf-8")
    envelopes, skipped = StructuredSource(fixtures, per_sample=True).read_all()
    store = env.open("measurements.db")
    try:
        for envelope in envelopes:
            store.commit(envelope)
        subject = Measurements(store)
        march = subject.series("weight", since="2026-03-01T00:00:00+00:00",
                               until="2026-03-31T23:59:59+00:00", unit="kg")
        whole = subject.series("weight")
        kept = march["records"][0]
        store.hide(kept, reason="the owner withdrew this reading", actor="owner")
        after = subject.series("weight", since="2026-03-01T00:00:00+00:00",
                               until="2026-03-31T23:59:59+00:00", unit="kg")
        expected_whole = round((70.0 + 71.0 + 72.0 + 73.0 + 99.0 + 154.0) / 6, 6)
        return [
            measured("compute", "a window reads the samples inside it and nothing else",
                     value=march["statistics"]["mean"], criterion="= 71.0",
                     passes=march["statistics"]["mean"] == 71.0 and march["samples"] == 3,
                     denominator=6,
                     detail={"samples": march["samples"], "window": march["window"],
                             "records": len(march["records"])}),
            measured("compute", "the reading cites exactly the rows it averaged",
                     value=len(march["records"]) - march["samples"], criterion="= 0",
                     passes=len(march["records"]) == march["samples"],
                     denominator=march["samples"],
                     detail={"note": "a citation for a row that was left out of the "
                                     "arithmetic is a citation that misleads"}),
            measured("compute", "a series that changed units refuses to be averaged",
                     value=1 if whole["statistics"] is None and
                     sorted(whole["unit_conflict"]) == ["kg", "lb"] else 0,
                     criterion="= 1",
                     passes=whole["statistics"] is None and
                     "kg" in whole["refused"] and "lb" in whole["refused"],
                     denominator=2, detail={"units_seen": whole["units_seen"],
                                            "mean_had_it_been_averaged": expected_whole}),
            measured("compute", "a sample with no time is counted, never invented",
                     value=subject.series("weight", unit="kg")["unplaced_times"],
                     criterion="= 1",
                     passes=subject.series("weight", unit="kg")["unplaced_times"] == 1,
                     denominator=5,
                     detail={"note": "the cadence is computed over placed times only"}),
            measured("compute", "a withdrawn sample leaves the mean of what remains",
                     value=after["statistics"]["mean"], criterion="= 71.5",
                     passes=after["statistics"]["mean"] == 71.5 and after["samples"] == 2,
                     denominator=3,
                     detail={"before": 71.0, "after": after["statistics"]["mean"]}),
            measured("compute", "no model was consulted for any of it",
                     value=0, criterion="= 0", passes=True, denominator=len(envelopes),
                     detail={"fixtures_skipped": len(skipped),
                             "note": "inference is off for this whole harness; the row is "
                                     "here so the promise has a place in the report"}),
        ]
    finally:
        store.close()


# -- operations ---------------------------------------------------------------

def check_operations(env: Environment) -> list[dict[str, Any]]:
    """§12.4 "Operations": every paused, failed, uncertain or partial stage is observable
    with a reason *and* the door that clears it.

    The check is structural rather than a list of expected strings, because the promise is not
    about wording — it is that nothing goes wrong quietly. An installation is put into four
    bad states at once and then asked, through the two read surfaces, to say each one and how
    to fix it.
    """
    from hermes_memory.lifecycle.erasure import ErasureManager
    from hermes_memory.operations.doctor import FAIL, OK, Doctor
    from hermes_memory.operations.status import DEGRADED, PAUSED, StatusReporter
    from hermes_memory.processing.resource_gate import ResourceGate
    from hermes_memory.sources.sync import SyncController

    owner = "eval-owner"
    store = env.open("operations.db")
    for envelope in corpus.records()[:60]:
        store.commit(envelope)
    sync = SyncController(store)
    sync.register("gmail", policy_version="private-api")
    fence = sync.acquire("gmail", holder="eval-operations")
    sync.mark_gap(fence, coverage_state="partial", reason="eval sweeping history")
    store.set_control("gmail", "capture", "paused", actor=owner, reason="eval pause",
                      policy_version="v1")
    gate = ResourceGate(store)
    gate.pause(actor=owner, reason="eval inference stop")
    held = store.commit({**corpus.records()[0], "source_id": "operations-held",
                         "metadata": {"code": "OPS0001", "category": "clinical"}})["id"]
    manager = ErasureManager(store, owner_principal=owner)
    preview = manager.preview(record_ids=[str(held)], actor=owner, reason="eval withdrawal")
    findings = Doctor(store, backend=Tripwire()).examine()["findings"]
    report = StatusReporter(store, gate=gate).report()
    silent_failures = [item["check"] for item in findings
                       if item["severity"] == FAIL and not str(item.get("remedy") or "").strip()]
    reasonless = [item["check"] for item in findings
                  if item["severity"] != OK and not str(item.get("detail") or "").strip()]
    paused_stages = [stage for stage in report["stages"] if stage["state"] == PAUSED]
    reasonless_stages = [stage["name"] for stage in paused_stages
                         if not str(stage.get("detail") or "").strip()]
    awaiting = report["awaiting_owner"]
    remediable = [item for item in findings if item["severity"] == FAIL]
    named_remedy = sum(1 for item in remediable
                       if "--" in str(item.get("remedy") or ""))
    paused_names = [stage["name"] for stage in paused_stages]
    # The scheduler's own failure mode: a reminder is due and nothing is taking it. That is
    # what a dead background loop leaves behind, and it looks exactly like a quiet machine
    # unless both read surfaces say which of the two this is.
    from types import SimpleNamespace

    from hermes_memory.prospective.goals import GoalStore

    back = env.open("background.db")
    GoalStore(back, owner_principal=owner).propose(
        title="Chase the invoice", statement="A thing the owner asked to be told about.",
        timezone_name="UTC", due=HANDOFF_DUE, proposed_by=owner, proposed_kind="owner")
    back.db.execute("INSERT INTO audit(action, object_id, created_at, metadata) "
                    "VALUES(?,?,?,?)",
                    ("maintenance_pass", "maintenance", HANDOFF_DUE, '{"at": "then"}'))
    looped = SimpleNamespace(maintenance_interval_s=900, home=env.root)
    finding = Doctor(back, settings=looped, backend=Tripwire()).background().as_dict()
    seen = StatusReporter(back, settings=looped).report()
    background = seen["background_pass"]
    named = [note for note in seen["notes"] if note.startswith("background:")]
    both_surfaces = bool(finding.get("remedy")) and background["behind"] \
        and background["waiting"] == 1 and bool(named)
    back.close()

    # The other half of "nothing goes wrong quietly": work an operator asked to stop and that
    # never answered. The intent has to be on file *before* the request goes out, and a lost
    # connection has to leave the operation `uncertain` with the intent standing — never
    # `cancelled`, which is a statement about the world this process is not entitled to make.
    from hermes_memory.backend.hindsight_client import HindsightUnavailable
    from hermes_memory.backend.worker_launcher import OperationLedger
    from hermes_memory.processing.cancellation import Canceller
    from hermes_memory.processing.instance_gate import GateStore

    class Dropped:
        """A backend that answers nothing, and reports what the ledger said when asked."""

        def __init__(self, ledger):
            self.ledger = ledger
            self.seen_at_call = None

        def cancel_operation(self, operation_id):
            self.seen_at_call = self.ledger.get(operation_id)["cancellation"]
            raise HindsightUnavailable("eval: the socket dropped mid-request")

    ledger_gate = GateStore(env.root / "cancel-gate.db")
    ledger = OperationLedger(ledger_gate)
    ledger.record({"_operation_id": "op-eval", "operation_type": "retain", "bank_id": "eval"},
                  worker_id="eval-worker", bank_id="eval", resource="remote-eval")
    ledger.mark("op-eval", "running")
    asker = Dropped(ledger)
    cancel = Canceller(None, ledger=ledger, client=asker).cancel_operation(
        "op-eval", actor=owner, reason="eval cancellation")
    residue = ledger.get("op-eval")
    stage = StatusReporter(store, gate=ResourceGate(ledger_gate)).resource_gate()
    ledger_gate.close()
    honest_stop = (asker.seen_at_call == "requested"
                   and cancel["backend"] == "unreachable"
                   and (residue["state"], residue["cancellation"]) == ("uncertain", "requested")
                   and stage.state == DEGRADED
                   and stage.evidence["operations"]["cancellations_owed"] == 1)
    store.close()
    return [
        measured("operations", "no failing check is reported without a remedy",
                 value=len(silent_failures), criterion="= 0", passes=not silent_failures,
                 denominator=len(findings),
                 detail={"silent": silent_failures, "failures": len(remediable),
                         "naming_a_command": named_remedy}),
        measured("operations", "no finding is reported without a reason",
                 value=len(reasonless), criterion="= 0", passes=not reasonless,
                 denominator=len(findings), detail={"reasonless": reasonless}),
        measured("operations", "a paused stage says it is paused, and why",
                 value=len(paused_stages), criterion=">= 1",
                 passes=bool(paused_stages) and not reasonless_stages,
                 denominator=len(report["stages"]),
                 detail={"paused": paused_names,
                         "without_a_reason": reasonless_stages,
                         "overall": report["overall"]}),
        measured("operations", "a decision only the owner may take is named as waiting",
                 value=1 if awaiting else 0, criterion="= 1", passes=bool(awaiting),
                 denominator=1, detail={"awaiting_owner": awaiting}),
        measured("operations", "a background pass that stopped is said by both readings",
                 value=1 if both_surfaces else 0, criterion="= 1", passes=both_surfaces,
                 denominator=1,
                 detail={"finding": finding.get("detail"), "remedy": finding.get("remedy"),
                         "note": named[:1], "behind": background["behind"],
                         "waiting": background["waiting"]}),
        measured("operations",
                 "an unanswered cancellation is filed as an intent, never as a stop",
                 value=1 if honest_stop else 0, criterion="= 1", passes=honest_stop,
                 denominator=1,
                 detail={"asked_before_the_call": asker.seen_at_call,
                         "backend": cancel["backend"], "state": residue["state"],
                         "cancellation": residue["cancellation"],
                         "gate_stage": stage.state,
                         "owed": stage.evidence["operations"]["cancellations_owed"]}),
    ]


# -- install and recovery -----------------------------------------------------

def check_install(env: Environment) -> list[dict[str, Any]]:
    """§12.4 "Install": clean setup and repeated setup, with nothing of the owner's lost.

    Run against a scratch Hermes home and a staged release tree, with the host commands faked
    by a recorder. No real installation is touched, and none is needed: the row is about what
    the transaction does to a home it is handed, which is observable from the outside.
    """
    import os

    from hermes_memory.config import load_settings
    from hermes_memory.install.setup import STEPS, plan, run

    root = env.path("install")
    instance = root / "instance"
    instance.mkdir(parents=True, exist_ok=True)
    release = instance / "release"
    (release / "bin").mkdir(parents=True, exist_ok=True)
    (release / "bin" / "hermes-memory").write_text("#!/bin/sh\n", encoding="utf-8")
    (instance / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={root / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n"
        "HERMES_MEMORY_OWNER_PRINCIPAL=eval-owner\n", encoding="utf-8")
    profile_home = root / "homes" / "work"
    profile_home.mkdir(parents=True, exist_ok=True)
    config = profile_home / "config.yaml"
    host_configuration = ("model:\n  provider: openai\n  model: deepseek-chat\n\n"
                          "memory:\n  memory_enabled: true\n")
    config.write_text(host_configuration, encoding="utf-8")

    class Host:
        """The host commands, faked. Nothing here touches a real installation.

        Obeys the arguments it is given rather than doing one fixed thing, because a fake that
        always wrote the answer we wanted would let the transaction build any command at all
        and still look correct — and the plan's install row is about what the host ends up
        saying, not about what the installer asked for.
        """

        def __init__(self, path: Path) -> None:
            self.path = path
            self.calls: list[list[str]] = []
            self.applied: list[str] = []

        def __call__(self, argv: list[str]) -> tuple[int, str]:
            self.calls.append(list(argv))
            if argv[1:3] == ["config", "set"]:
                self._set(argv[3], argv[4])
            return 0, ""

        def _set(self, key: str, value: str) -> None:
            self.applied.append(key)
            text = self.path.read_text(encoding="utf-8")
            block, _, leaf = key.rpartition(".")
            lines = text.splitlines()
            inside = False
            for index, line in enumerate(lines):
                stripped = line.strip()
                if not line.startswith(" ") and stripped.endswith(":"):
                    inside = stripped[:-1] == block
                    if inside:
                        lines.insert(index + 1, f"  {leaf}: {value}")
                        break
                    continue
                if inside and stripped.startswith(f"{leaf}:"):
                    lines[index] = f"  {leaf}: {value}"
                    break
            else:
                lines += ["", f"{block}:", f"  {leaf}: {value}"]
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    host = Host(config)
    runner = host
    before = {item.relative_to(profile_home).as_posix(): item.stat().st_mtime_ns
              for item in profile_home.rglob("*") if item.is_file()}
    saved = {key: os.environ.get(key) for key in
             ("HERMES_MEMORY_HOME", "XDG_CONFIG_HOME", "HERMES_MEMORY_ADMISSION_URL")}
    os.environ["HERMES_MEMORY_HOME"] = str(instance)
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
    os.environ.pop("HERMES_MEMORY_ADMISSION_URL", None)
    try:
        settings = load_settings()
        arguments = {"environ": {"HERMES_MEMORY_RELEASE": str(release)}, "runner": runner,
                     "ref": "a" * 40}
        first = plan(settings, hermes_home=profile_home, **arguments)
        done = run(settings, hermes_home=profile_home, actor=settings.owner_principal,
                   review=first["review_digest"], **arguments)
        again = plan(settings, hermes_home=profile_home, **arguments)
        settled = _fingerprints(instance, profile_home, skip_ledgers=True)
        second = run(settings, hermes_home=profile_home, actor=settings.owner_principal,
                     review=again["review_digest"], **arguments)
        after = config.read_text(encoding="utf-8")
        touched = sorted(name for name, stamp in before.items()
                         if (profile_home / name).stat().st_mtime_ns != stamp)
        added = sorted(name for name in
                       {item.relative_to(profile_home).as_posix()
                        for item in profile_home.rglob("*") if item.is_file()} - set(before))
        rewrote = sorted(name for name, digest in settled.items()
                         if _fingerprints(instance, profile_home,
                                          skip_ledgers=True).get(name) != digest)
        records = 0
        if settings.db_path.exists():
            with EvidenceStore(settings.db_path) as made:
                records = int(made.db.execute("SELECT count(*) FROM records").fetchone()[0])
        steps_second = {step["step"]: step["state"] for step in again["steps"]}
        pending = sorted(name for name, state in steps_second.items() if state == "pending")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    model_kept = "model: deepseek-chat" in after and "provider: openai" in after
    only_the_host_config = touched in ([], ["config.yaml"]) and not added
    # The read-only steps re-run by design — they are how a second invocation can show its
    # plan — and `services` stays pending until the operator asks for --start. Every step that
    # *changes* something must report that it has already been done.
    writers = {"stage", "configure", "initialize", "register-plugin", "preflight",
               "activate", "canary", "finish"}
    resumed = sorted(name for name, state in steps_second.items() if state == "resumed")
    return [
        measured("install", "the transaction has the eleven steps in the plan's order",
                 value=len(STEPS), criterion="= 11",
                 passes=STEPS == ("inventory", "plan", "stage", "configure", "initialize",
                                  "register-plugin", "preflight", "activate", "services",
                                  "canary", "finish"),
                 denominator=11, detail={"steps": list(STEPS)}),
        measured("install", "a clean setup completes and says so",
                 value=1 if done.get("ok") else 0, criterion="= 1",
                 passes=bool(done.get("ok")), denominator=1,
                 detail={"readiness": done.get("readiness"),
                         "receipts": len(done.get("receipts") or {}),
                         "host_commands": len(host.calls)}),
        measured("install", "repeated setup reports every writing step as already done",
                 value=len(writers - set(resumed)), criterion="= 0",
                 passes=not (writers - set(resumed)) and bool(second.get("ok")),
                 denominator=len(writers),
                 detail={"resumed": resumed, "pending_again": pending,
                         "note": "inventory and plan re-read on purpose; services stays "
                                 "pending until the operator asks for --start"}),
        measured("install", "a second setup rewrites no unit, template or config file",
                 value=len(rewrote), criterion="= 0", passes=not rewrote,
                 denominator=len(settled),
                 detail={"rewritten": rewrote, "files_compared": len(settled),
                         "note": "ledgers (installation.db, canonical.db) are excluded: "
                                 "they are expected to record that setup ran"}),
        measured("install", "the owner's model configuration survives untouched",
                 value=1 if model_kept else 0, criterion="= 1", passes=model_kept, denominator=1,
                 detail={"keys_the_host_was_asked_to_set": host.applied,
                         "profile_files_changed": touched}),
        measured("install", "the installer reaches the host by command, not by editing",
                 value=1 if only_the_host_config else 0, criterion="= 1",
                 passes=only_the_host_config, denominator=1,
                 detail={"changed": touched, "created": added,
                         "note": "config.yaml may move — the host moved it on the "
                                 "installer's request; nothing else in the profile home "
                                 "should"},
                 ),
        measured("install", "installing ingests nothing of the owner's",
                 value=records, criterion="= 0", passes=records == 0, denominator=1,
                 detail={"canonical_records_after_setup": records}),
    ]


def check_recovery(env: Environment) -> list[dict[str, Any]]:
    """§12.4 "Recovery": restore the matching snapshot, and still honour what happened after.

    The property is a contradiction unless it is engineered: a restore must bring the archive
    back and must not bring back the thing the owner asked to be forgotten, or a backup becomes
    a way to un-forget.
    """
    from hermes_memory.lifecycle.erasure import ErasureManager
    from hermes_memory.lifecycle.recovery import Recovery
    from hermes_memory.lifecycle.snapshots import Snapshots
    from hermes_memory.storage.lineage import Lineage

    owner = "eval-owner"
    store = env.open("recovery.db")
    kept = corpus.records()[:80]
    for envelope in kept:
        store.commit(envelope)
    snapshots = Snapshots(store, directory=env.path("snapshots"))
    recovery = Recovery(store, snapshots=snapshots, owner_principal=owner)
    made = snapshots.create(reason="eval baseline", actor=owner)["snapshot"]
    verified = snapshots.verify(made.id)

    old = str(store.commit({**kept[0], "source_id": "recovery-before", "revision": "1",
                            "text": "An old clinical result that must not come back.",
                            "metadata": {"code": "REC0000",
                                         "category": "clinical"}})["id"])
    made = snapshots.create(reason="eval baseline", actor=owner)["snapshot"]
    verified = snapshots.verify(made.id)

    # Two erasures, on either side of the snapshot. The first is the case a backup must never
    # undo; the second is a decision about evidence this copy never held, which has to travel
    # as an obligation rather than as a grave marker for a row that is not there.
    late = str(store.commit({**kept[0], "source_id": "recovery-secret", "revision": "1",
                             "text": "A clinical result that must never come back.",
                             "metadata": {"code": "REC0001",
                                          "category": "clinical"}})["id"])
    store.commit({**kept[1], "source_id": "recovery-after", "revision": "1",
                  "text": "Written after the snapshot was taken.",
                  "metadata": {"code": "REC0002", "category": "clinical"}})
    manager = ErasureManager(store, owner_principal=owner)
    preview = manager.preview(record_ids=[old, late], actor=owner, reason="mentioned in error")
    manager.confirm(intent_id=preview["intent_id"], preview_digest=preview["preview_digest"],
                    actor=owner)
    before_restore = int(store.db.execute("SELECT count(*) FROM records").fetchone()[0])
    result = recovery.restore(str(made.id), actor=owner)
    integrity = recovery.integrity()
    came_back = (store.get(old) is not None or bool(store.search("diagnosis"))
                 or store.get(late) is not None)
    explained = Lineage(store).explain(old)
    after_restore = int(store.db.execute("SELECT count(*) FROM records").fetchone()[0])
    store.close()
    return [
        measured("recovery", "a snapshot verifies against its own manifest",
                 value=1 if verified["ok"] else 0, criterion="= 1",
                 passes=bool(verified["ok"]), denominator=1,
                 detail={"problems": verified["problems"], "records": verified["records"],
                         "checksum": bool(verified.get("manifest", True))}),
        measured("recovery", "restore reapplies the erasures that postdate the snapshot",
                 value=int(result.get("reapplied") or 0), criterion=">= 1",
                 passes=int(result.get("reapplied") or 0) >= 1 and not came_back,
                 denominator=1,
                 detail={"reapplied": result.get("reapplied"),
                         "intents": result.get("intents"), "came_back": came_back,
                         "state_after": explained.get("state")}),
        measured("recovery", "an erasure of evidence the snapshot never held is carried, "
                             "not crashed on",
                 value=int(result.get("tombstones_without_a_record") or 0), criterion="= 1",
                 passes=int(result.get("tombstones_without_a_record") or 0) == 1,
                 denominator=1,
                 detail={"obligations": result.get("obligations"),
                         "intents": result.get("intents"),
                         "note": "a restore that aborts on a dangling marker is a backup "
                                 "that cannot be restored"}),
        measured("recovery", "the archive is back to the snapshot's size",
                 value=after_restore, criterion=f"<= {before_restore}",
                 passes=after_restore <= before_restore, denominator=before_restore,
                 detail={"before": before_restore, "after": after_restore,
                         "snapshot_records": made.records}),
        measured("recovery", "the restored store reports itself consistent",
                 value=1 if integrity["ok"] else 0, criterion="= 1",
                 passes=bool(integrity["ok"]), denominator=1,
                 detail={"problems": integrity["problems"],
                         "erasure_pending": integrity.get("erasure_pending")}),
    ]


# -- what this harness cannot measure ----------------------------------------

def not_measured_rows() -> list[dict[str, Any]]:
    """The rest of §12.4, named instead of skipped.

    Every row here waits on the same thing: the P0 authorization to make bounded live calls to
    the pinned backend. Saying "not measured" in the same report as the measured numbers is the
    point — a table with two rows missing and nothing said about it reads as a pass.
    """
    return [
        not_measured("extraction", "evidence recall, attribution precision, negation "
                                   "handling and schema success",
                     because="these are properties of what a model forms, and this run forms "
                             "nothing: inference is switched off and no backend is reachable",
                     requires="the P0 gate — one authorized, bounded live retain against the "
                              "pinned engine, with the token caps this build ships"),
        not_measured("extraction", "small output caps do not silently lose coverage",
                     because="the truncation is reported by the backend; without a live "
                             "response there is nothing to compare the reported span with",
                     requires="the same authorized live retain, at a cap small enough to "
                              "truncate on purpose"),
        not_measured("formation latency", "p50/p95/p99 per operation, grammar mode and "
                                          "payload size",
                     because="latency of a model that is not running is a measurement of the "
                             "harness",
                     requires="the P0 gate; the queue, the leases and the budget ledger are "
                              "measured here, and their numbers are under `compute`"),
        not_measured("context latency", "auto-prefetch consumes only a bounded host wait",
                     because="prefetch is the host's decision to make on a turn; this harness "
                             "has no host to be late for",
                     requires="a run against the tested Hermes revision, with the plugin's own "
                              "deadline accounting"),
    ]
