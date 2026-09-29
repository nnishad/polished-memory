#!/usr/bin/env python3
"""Heavy load and complex scenarios against one scratch installation.

The unit and integration suites say a flow is correct; this says it holds at scale, under
concurrency, and against a process that is not the one that wrote the row. Nothing here
touches the owner's real archive — a synthetic record in a live memory is a fact the agent
will later repeat back to them — and nothing here calls a model, because formation quality
is measured by `run_synthetic.py` and formation *throughput under contention* is measured
by the gate ledger, which needs no inference to prove that two workers never hold one lease.

    uv run python evals/run_load.py [--records 3000] [--keep]

Exit is non-zero if any invariant breaks. Timings are printed because a correct system that
takes nine minutes to drain an outbox is a different system from one that takes nine seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hermes_memory.backend.document_map import DocumentMap          # noqa: E402
from hermes_memory.lifecycle.erasure import ErasureManager          # noqa: E402
from hermes_memory.processing.maintenance import Maintenance       # noqa: E402
from hermes_memory.prospective.due_events import DueEventLog        # noqa: E402
from hermes_memory.prospective.goals import GoalStore               # noqa: E402
from hermes_memory.storage.evidence import EvidenceStore            # noqa: E402
from hermes_memory.storage.identity import IdentityStore            # noqa: E402
from hermes_memory.proactive.policy import AttentionPolicy          # noqa: E402

OWNER = "load-owner"
CLI = sys.executable
PEOPLE = ["Amara", "Bjorn", "Chidi", "Dani", "Elin", "Farid", "Gao", "Hana",
          "Ines", "Jorge", "Kira", "Lambert"]
SEEDS = ("the invoice was settled by transfer",
         "the site visit moved to the second Thursday",
         "a valve seal on pump three is weeping again",
         "the survey closes once two hundred replies land",
         "her passport renewal needs a countersignature",
         "the quotation expires at the end of the quarter",
         "the backup ran but the verification step did not",
         "two calendars disagree about the kickoff")
FAILURES: list[str] = []


def report(name: str, **facts: object) -> None:
    print(f"  {name}: " + ", ".join(f"{key}={value}" for key, value in facts.items()),
          flush=True)


def check(name: str, condition: bool, **facts: object) -> None:
    report(("ok   " if condition else "FAIL ") + name, **facts)
    if not condition:
        FAILURES.append(name)


def envelope(source: str, identifier: str, text: str, index: int) -> dict:
    return {
        "source": source, "source_id": identifier, "revision": "1", "kind": "email",
        "text": text, "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                                   time.gmtime()),
        "occurred_at": f"2026-09-{(index % 27) + 1:02d}T09:00:00+00:00",
        "occurred_precision": "second",
        "metadata": {"participants": [{"address": f"{identifier}@example.test"}]},
    }


def build(root: Path, records: int) -> dict:
    """One installation, from init to a store full of evidence."""
    home = root / "instance"
    data = home / "data"
    profile_home = root / "homes" / "load"
    profile_home.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           "HERMES_MEMORY_HOME": str(home),
           "HERMES_MEMORY_DATA_DIR": str(data),
           "HERMES_MEMORY_OWNER_PRINCIPAL": OWNER,
           "HERMES_MEMORY_INFERENCE_ENABLED": "false",
           # A scratch installation delivers to its own home: the drain must actually
           # run, or the concurrency it is meant to measure never happens.
           "HERMES_MEMORY_DELIVERY_ENABLED": "true",
           "HERMES_MEMORY_DELIVERY_TARGET": f"local:{OWNER}"}
    started = time.perf_counter()
    subprocess.run([CLI, "-m", "hermes_memory.cli", "init"], env=env, check=True,
                   capture_output=True)
    plan = subprocess.run([CLI, "-m", "hermes_memory.cli", "enroll",
                           "--hermes-home", str(profile_home)], env=env, check=True,
                          capture_output=True)
    digest = json.loads(plan.stdout)["review_digest"]
    subprocess.run([CLI, "-m", "hermes_memory.cli", "enroll", "--hermes-home",
                    str(profile_home), "--review", digest, "--actor", OWNER], env=env,
                   check=True, capture_output=True)
    # The enroll assigned this profile its own store. Loading the installation's default
    # and then draining the profile's would measure nothing at all — which is exactly what
    # the first run of this harness did, silently.
    ledger = sqlite3.connect(home / "installation.db")
    row = ledger.execute("SELECT data_dir FROM profiles WHERE hermes_home=?",
                         (str(profile_home),)).fetchone()
    ledger.close()
    store_path = Path(row[0]) / "canonical.db"
    report("installation", seconds=round(time.perf_counter() - started, 2),
           store=str(store_path))
    return {"env": env, "home": home, "data": data, "db": store_path,
            "profile_home": profile_home}


# -- 1. ingest -----------------------------------------------------------------

def phase_ingest(inst: dict, records: int) -> list[str]:
    print(f"\ningest: {records} records across four connectors, with corrections")
    started = time.perf_counter()
    identifiers: list[str] = []
    with EvidenceStore(inst["db"]) as store:
        for index in range(records):
            person = PEOPLE[index % len(PEOPLE)]
            source = ("gmail", "whatsapp", "calendar", "notes")[index % 4]
            identifier = f"{source}-{index:06d}"
            text = (f"{person}: {SEEDS[index % len(SEEDS)]}. "
                    f"Reference {source.upper()}-{index:05d}.")
            result = store.commit(envelope(source, identifier, text, index))
            identifiers.append(result["id"])
            if index % 17 == 0:      # a later message that supersedes the first
                correction = store.commit(envelope(
                    source, f"{identifier}-r2",
                    f"{person}: correction — {SEEDS[(index + 3) % len(SEEDS)]}.", index))
                identifiers.append(correction["id"])
        store.commit(envelope("gmail", "planted-quiet",
                              "Fenland Hydrology Board met at Ramsey and reported a "
                              "chlorine exceedance on the third main.", 1))
        stored = int(store.db.execute("SELECT count(*) FROM records").fetchone()[0])
        fts = int(store.db.execute("SELECT count(*) FROM record_fts").fetchone()[0])
    elapsed = time.perf_counter() - started
    report("committed", records=len(identifiers), stored=stored, fts_rows=fts,
           per_second=round(len(identifiers) / elapsed), seconds=round(elapsed, 2))
    check("every commit reached the FTS index", stored == fts, stored=stored, fts=fts)
    check("the corpus is the size asked for, corrections included", stored > records,
          stored=stored, asked=records)
    return identifiers


# -- 2. retrieval --------------------------------------------------------------

def phase_retrieval(inst: dict, queries: int = 400) -> None:
    print(f"\nretrieval: {queries} queries, including the shapes the planner had to learn")
    rng = random.Random(7)
    with EvidenceStore(inst["db"]) as store:
        latencies, misses = [], []
        for index in range(queries):
            plain = index % 3 != 1
            words = rng.choice(SEEDS).split()
            needle = (" ".join(words[:3]) if plain else rng.choice(
                ('quote:"chlorine exceedance"', "valve AND weeping", "passport OR renewal",
                 "survey NOT closes", '"pump three" -seal', "RAMSEY")))
            started = time.perf_counter()
            hits = store.search(needle, limit=20)
            latencies.append(time.perf_counter() - started)
            if plain and not hits:
                misses.append(needle)
        exact = store.search("chlorine exceedance", limit=10)
        hostile = store.search('bad:"unclosed', limit=5)
    latencies.sort()
    report("search", queries=queries,
           p50_ms=round(latencies[len(latencies) // 2] * 1000, 2),
           p95_ms=round(latencies[int(len(latencies) * 0.95)] * 1000, 2),
           p99_ms=round(latencies[int(len(latencies) * 0.99)] * 1000, 2),
           no_answer=len(misses))
    check("a planted fact is retrievable by its own words", bool(exact), hits=len(exact))
    check("malformed query syntax answers instead of throwing", isinstance(hostile, list),
          hits=len(hostile))
    check("every plain query found something", len(misses) == 0, misses=misses[:3])


# -- 3. proactivity and concurrent drains --------------------------------------

def phase_delivery(inst: dict, goals: int = 250) -> None:
    print(f"\nproactivity: {goals} overdue promises, then twelve drains racing one outbox")
    with EvidenceStore(inst["db"]) as store:
        policy = AttentionPolicy(store, owner_principal=OWNER)
        policy.configure(actor=OWNER, timezone_name="UTC",
                         cooldown_minutes=0, quiet_from="00:00", quiet_until="00:01",
                         max_immediate_per_day=10, shadow=False)
        events = DueEventLog(store)
        goals_store = GoalStore(store, events=events, owner_principal=OWNER)
        for index in range(goals):
            goals_store.propose(title=f"Chase task {index}", statement="It is owed.",
                                timezone_name="UTC",
                                due="2026-09-20T08:00:00+00:00", proposed_by=OWNER,
                                proposed_kind="owner")
    with EvidenceStore(inst["db"]) as store:
        started = time.perf_counter()
        result = Maintenance(store, owner_principal=OWNER,
                             clock=time.time).pass_now(sections=("proactive",))
        report("pass", **result["proactive"])
        prepared = int(store.db.execute(
            "SELECT count(*) FROM outbox WHERE state='prepared'").fetchone()[0])
    seconds = time.perf_counter() - started
    report("prepared", artifacts=prepared, seconds=round(seconds, 2))

    written = inst["profile_home"] / "memory" / "delivered"
    def drain(_: int) -> dict:
        completed = subprocess.run(
            [CLI, "-m", "hermes_memory.cli", "deliver", "--hermes-home",
             str(inst["profile_home"]), "--limit", "25"], env=inst["env"],
            capture_output=True, text=True, timeout=600)
        try:
            return json.loads(completed.stdout)
        except ValueError:
            # A drain that does not answer in its own report format is the failure worth
            # seeing, so keep what it said rather than turning it into a number.
            return {"code": completed.returncode,
                    "stderr": completed.stderr.strip()[-300:]}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=12) as pool:
        sent = list(pool.map(drain, range(12)))
    handed = sum(item.get("delivered", 0) for item in sent if "delivered" in item)
    broken = [item for item in sent if "delivered" not in item]
    with EvidenceStore(inst["db"]) as store:
        states = dict(store.db.execute("SELECT state, count(*) FROM outbox GROUP BY state"))
        duplicates = int(store.db.execute(
            "SELECT count(*) FROM (SELECT id FROM outbox GROUP BY id HAVING "
            "count(*) > 1)").fetchone()[0])
    landed = len(list(written.glob("*.md"))) if written.is_dir() else 0
    report("drains", delivered=handed, unreported=len(broken),
           sample=[item.get("reason") or item for item in sent][:4],
           seconds=round(time.perf_counter() - started, 2),
           files_on_disk=landed, states=states)
    check("every drain answered with a report", not broken,
          broken=[item.get("stderr", "")[:120] for item in broken][:2])
    check("no artifact was left unaccounted for",
          sum(count for state, count in states.items() if state in
              ("accepted_unverified", "prepared")) == prepared,
          prepared=prepared, states=states)
    check("files on disk equal artifacts handed over", landed == states.get(
        "accepted_unverified", 0), landed=landed, handed_over=states.get(
        "accepted_unverified", 0))
    check("no duplicate artifact rows", duplicates == 0, duplicates=duplicates)


# -- 4. identity ---------------------------------------------------------------

def phase_identity(inst: dict, candidates: int = 120) -> None:
    print(f"\nidentity: {candidates} same-person proposals, half confirmed, half left for "
          "the owner")
    confirmed = 0
    with EvidenceStore(inst["db"]) as store:
        identities = IdentityStore(store, owner_principal=OWNER)
        for index in range(candidates):
            basis = store.commit(envelope("gmail", f"id-a-{index}",
                                          f"{PEOPLE[index % len(PEOPLE)]} writes from work",
                                          index))["id"]
            other = store.commit(envelope("whatsapp", f"id-b-{index}",
                                          f"{PEOPLE[index % len(PEOPLE)]} from a second "
                                          "number", index))["id"]
            left = identities.account("email", f"person-{index}@example.test")
            right = identities.account("phone", f"+49151{index:08d}")
            made = identities.propose(account_a=left, account_b=right,
                                      rule="explicit-alias-declared",
                                      basis=f"the owner labelled both as {index}",
                                      evidence=[basis, other], proposed_by="agent:load")
            if index % 2 == 0:
                identities.confirm(candidate_id=made["candidate_id"], actor=OWNER,
                                   reason="recognised by the owner")
                confirmed += 1
        pending = len(identities.pending(limit=200))
        edges = int(store.db.execute("SELECT count(*) FROM identity_edges").fetchone()[0])
    report("identity", proposed=candidates, confirmed=confirmed, refused_or_open=pending,
           edges=edges)
    check("every confirmation produced an edge", edges == confirmed, edges=edges,
          confirmed=confirmed)
    check("the half nobody confirmed is still waiting for them",
          pending == candidates - confirmed, pending=pending)


# -- 5. erasure ---------------------------------------------------------------

def count_owed(store) -> int:
    return int(store.db.execute(
        "SELECT count(*) FROM erasure_targets WHERE state!='verified'").fetchone()[0])


class ABackend:
    """A derived backend that answers honestly: it deletes, and it can be asked again."""

    def __init__(self):
        self.absent: set[str] = set()

    def delete_document(self, document_id):
        self.absent.add(document_id)
        return {"deleted": True}

    def document_state(self, document_id):
        return {"document_id": document_id,
                "state": "absent" if document_id in self.absent else "present",
                "count": 0 if document_id in self.absent else 4}


def phase_erasure(inst: dict, targets: int = 400) -> None:
    print(f"\nerasure: {targets} records withdrawn, with a backend that owes read-backs")
    with EvidenceStore(inst["db"]) as store:
        forget = ErasureManager(store, owner_principal=OWNER)
        mapped = 0
        private: list[str] = []
        for index in range(targets):
            identifier = store.commit(envelope(
                "notes", f"x-{index}",
                f"A private matter about {PEOPLE[index % len(PEOPLE)]}.", index))["id"]
            DocumentMap(store).begin(identifier, "1")
            mapped += 1
            private.append(identifier)
        before = int(store.db.execute("SELECT count(*) FROM records").fetchone()[0])
        started = time.perf_counter()
        batch, intents = [], []
        # Withdraw the records this phase wrote, not whatever the query happens to reach
        # first: erasing the earlier corpus would leave these very rows answering a search
        # for text that was supposedly forgotten, and the check below would be a lie.
        for identifier in private:
            batch.append(identifier)
            if len(batch) == 20:
                preview = forget.preview(record_ids=batch, actor=OWNER, reason="withdrawn")
                forget.confirm(intent_id=preview["intent_id"],
                               preview_digest=preview["preview_digest"], actor=OWNER)
                intents.append(preview["intent_id"])
                batch = []
        local_seconds = time.perf_counter() - started
        after_local = int(store.db.execute(
            "SELECT count(*) FROM records WHERE deleted=0").fetchone()[0])
        owed = count_owed(store)
        backend = ABackend()
        started = time.perf_counter()
        settled, passes = 0, 0
        # Bounded by the ledger's own cap, so a debt of any size is paid in passes — and a
        # second pass over a settled debt is the case that must stay quiet rather than
        # re-dispatch something already verified gone.
        while passes < 40 and count_owed(store):
            outcome = forget.discharge(client=backend, limit=200)
            settled += len(outcome["settled"])
            passes += 1
            if not outcome["attempted"]:
                break
        remaining = count_owed(store)
        seconds = time.perf_counter() - started
        complete = int(store.db.execute(
            "SELECT count(*) FROM erasure_ledger WHERE state='complete'").fetchone()[0])
    report("erasure", mapped=mapped, forgotten=before - after_local,
           local_seconds=round(local_seconds, 2), backend_seconds=round(seconds, 2),
           obligations_owed_before=owed, settled=settled, passes=passes,
           still_owed=remaining, complete_intents=complete)
    check("the local fence removed exactly what was withdrawn",
          before - after_local == targets, removed=before - after_local, asked=targets)
    check("every backend obligation was paid and read back", remaining == 0,
          still_owed=remaining)
    with EvidenceStore(inst["db"]) as store:
        leaked = store.search("private matter", limit=5)
    check("forgotten text does not answer a search", all(
        "private matter" not in item.text.lower() for item in leaked), hits=len(leaked))


# -- 6. concurrency against a live store --------------------------------------

def phase_concurrency(inst: dict, threads: int = 24, per_thread: int = 60) -> None:
    print(f"\nconcurrency: {threads} threads × {per_thread} mixed operations on one store")
    errors: list[str] = []

    def worker(index: int) -> None:
        try:
            with EvidenceStore(inst["db"]) as store:
                for step in range(per_thread):
                    if (index + step) % 5 == 0:
                        store.commit(envelope("notes", f"c-{index}-{step}",
                                              f"concurrent note {index}/{step}", step))
                    elif (index + step) % 5 == 1:
                        store.search(f"note {index}", limit=5)
                    elif (index + step) % 5 == 2:
                        int(store.db.execute("SELECT count(*) FROM records").fetchone()[0])
                    elif (index + step) % 5 == 3:
                        store.control("global", "delivery")
                    else:
                        store.stage_is_paused("global", "delivery")
        except Exception as error:
            errors.append(f"{type(error).__name__}: {str(error)[:120]}")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(worker, range(threads)))
    elapsed = time.perf_counter() - started
    report("stress", operations=threads * per_thread, seconds=round(elapsed, 2),
           errors=len(errors), sample=errors[:3])
    check("no operation surfaced a lock or a torn read", not errors, errors=errors[:3])


# -- 7. the reports agree ------------------------------------------------------

def phase_reports(inst: dict) -> None:
    print("\nreports: doctor and status against the ledger they describe")
    doctor = subprocess.run([CLI, "-m", "hermes_memory.cli", "doctor", "--hermes-home",
                            str(inst["profile_home"])], env=inst["env"],
                           capture_output=True, text=True, timeout=600)
    status = subprocess.run([CLI, "-m", "hermes_memory.cli", "status", "--hermes-home",
                            str(inst["profile_home"])], env=inst["env"],
                            capture_output=True, text=True, timeout=600)
    parsed = json.loads(doctor.stdout)
    states = json.loads(status.stdout)["stages"]["states"]
    with EvidenceStore(inst["db"]) as store:
        open_obligations = int(store.db.execute(
            "SELECT count(*) FROM erasure_targets WHERE state!='verified'").fetchone()[0])
    report("doctor", ok=parsed["ok"], severity=parsed["severity"],
           warn=sum(1 for item in parsed["findings"] if item["severity"] == "warn"))
    report("status", states=states)
    check("nothing is half-forgotten in the reports", open_obligations == 0,
          open_obligations=open_obligations)
    with EvidenceStore(inst["db"]) as store:
        unproven = int(store.db.execute(
            "SELECT count(*) FROM outbox WHERE state='accepted_unverified'").fetchone()[0])
    # A clean doctor is not the invariant — an honest one is. Drains that handed artifacts
    # over without a carrier's receipt *should* degrade the delivery stage, and a doctor
    # that stayed quiet about them would be the bug.
    check("the delivery stage is degraded exactly as far as the unproven sends say",
          (states["delivery"] == "degraded") == (unproven > 0), delivery=states["delivery"],
          unproven=unproven)
    check("the doctor says so in as many words",
          any("delivery proof" in str(item.get("detail")) or "unproven" in str(item)
              for item in parsed["findings"] if item["severity"] != "ok"),
          findings=[item["severity"] for item in parsed["findings"]])
    check("status answers", status.returncode == 0 and bool(states), code=status.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=3000)
    parser.add_argument("--root", default="")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    root = Path(args.root or Path.home() / "tmp" / f"hermes-load-{time.strftime('%H%M%S')}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    started = time.perf_counter()
    try:
        inst = build(root, args.records)
        phase_ingest(inst, args.records)
        phase_retrieval(inst)
        phase_delivery(inst)
        phase_identity(inst)
        phase_erasure(inst)
        phase_concurrency(inst)
        phase_reports(inst)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    total = time.perf_counter() - started
    print(f"\n{len(FAILURES)} invariant failure(s) in {total:.1f}s"
          + (f" — {', '.join(FAILURES)}" if FAILURES else "")
          + ("" if args.keep or FAILURES else f"   (kept: {root})"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
