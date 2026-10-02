"""Opt-in synthetic canonical → Hindsight → Qwen → broker validation.

No personal corpus, Telegram send or model fallback. Keep reconciliation ledgers.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import tempfile
import time
import uuid
import urllib.request
from pathlib import Path

from hermes_memory.backend.hindsight_client import operation_state
from hermes_memory.backend.support import SupportResolver
from hermes_memory.config import load_settings
from hermes_memory.context.broker import ContextBroker
from hermes_memory.processing.formation import backend_client, formation_apply, formation_plan
from hermes_memory.processing.instance_gate import gate_path
from hermes_memory.processing.verification import verify_memory_claims
from hermes_memory.processing.summarization import summarize_plan, summarize_apply
from hermes_memory.storage.evidence import EvidenceStore
from run_memory_intelligence import isolated_settings

FIXTURES = (
    ("arjun", "Arjun ko chai nahi pasand; woh subah bina sugar wali coffee peeta hai."),
    ("meera", "मीरा शुक्रवार सुबह नौ बजे जयपुर जाने वाली ट्रेन से यात्रा करेगी।"),
    ("nisha", "Nisha is allergic to peanuts and must avoid food containing peanuts."),
    ("ravi", "Ravi prefers sweet tea every morning and does not like coffee."),
    ("dev", "Dev will travel to Pune by bus on Sunday afternoon."),
    ("karan", "Karan likes almonds and has no known nut allergy."),
)
QUERIES = (
    ("english", "What does Arjun drink in the morning?", "arjun"),
    ("hinglish", "Arjun subah kya peeta hai, chai ya coffee?", "arjun"),
    ("hindi", "अर्जुन सुबह क्या पीता है?", "arjun"),
    ("english", "When and how is Meera travelling to Jaipur?", "meera"),
    ("hinglish", "Meera Jaipur kab aur kaise jayegi?", "meera"),
    ("hindi", "मीरा जयपुर कब और कैसे जाएगी?", "meera"),
    ("english", "Which food must Nisha avoid because of her allergy?", "nisha"),
    ("hinglish", "Nisha ko allergy ki wajah se kya nahi khana chahiye?", "nisha"),
    ("hindi", "निशा को एलर्जी के कारण कौन सा खाना नहीं खाना चाहिए?", "nisha"),
)


def rerank_events(settings, after=0):
    with sqlite3.connect(gate_path(settings).as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(
            "SELECT rowid AS sequence,event,detail FROM gate_ledger WHERE route='rerank' AND rowid>? ORDER BY rowid", (after,))]


def check(settings, *, exercise_outage=False):
    scratch = Path(tempfile.mkdtemp(prefix="hermes-reranked-flow-"))
    bank = "eval-" + uuid.uuid4().hex
    scoped = isolated_settings(settings, scratch, bank)
    client = backend_client(scoped)
    client.health()
    existing = rerank_events(settings)
    marker = existing[-1]["sequence"] if existing else 0
    result = {"synthetic_only": True, "bank": bank, "ledger": str(scoped.db_path),
              "checks": [], "cleanup": [], "failure": None}
    with EvidenceStore(scoped.db_path) as store:
        records = {}
        for source_id, text in FIXTURES:
            records[source_id] = store.commit({"source": "synthetic", "source_id": source_id, "revision": "1",
                "kind": "message", "text": text, "observed_at": "2026-10-02T12:00:00+00:00",
                "occurred_at": None, "occurred_precision": "unknown",
                "metadata": {"role": "user", "origin": "owner-statement", "independent": True}})["id"]
    started = time.monotonic()
    try:
        proposal = formation_plan(scoped, limit=len(FIXTURES))
        result["formation"] = formation_apply(scoped, actor="synthetic-evaluation", review=proposal["review_digest"],
            limit=len(FIXTURES), max_jobs=len(FIXTURES), client=client, follow_poll_s=2, follow_max_polls=30)
        with EvidenceStore(scoped.db_path) as store:
            broker = ContextBroker(store, client=client, cache=False, derived_timeout_s=settings.foreground_deadline_s)
            try:
                for language, query, target in QUERIES:
                    before = time.monotonic()
                    outcome = client.recall(query, max_tokens=600)
                    resolver = SupportResolver(store, bank_id=bank, source_facts=outcome.source_facts)
                    supported = [resolver.resolve(fact) for fact in outcome.results]
                    packet = broker.assemble(query)
                    ranked_target = bool(supported and records[target] in supported[0])
                    # Native semantic facts occupy Packet.facts, not lexical Packet.items.
                    packet_support = [resolver.resolve(fact) for fact in packet.facts]
                    packet_target = (any(item.id == records[target] for item in packet.items)
                                     or any(records[target] in ids for ids in packet_support))
                    result["checks"].append({"language": language, "query": query,
                        "top_result_has_expected_support": ranked_target, "broker_contains_expected_evidence": packet_target,
                        "channels": packet.channels.as_dict(), "backend": outcome.as_dict(),
                        "facts": list(outcome.results), "seconds": round(time.monotonic() - before, 4),
                        "passed": ranked_target and packet_target and packet.channels.derived == "available"})
                reflection = client.reflect("What should Nisha avoid eating and why?", max_tokens=512, budget="low", include_support=True)
                reflection_resolver = SupportResolver(store, bank_id=bank, source_facts=reflection["source_facts"])
                reflection_support = [reflection_resolver.resolve(fact) for fact in reflection["facts"]]
                result["reflection"] = {"text": reflection["text"], "cited_memories": reflection["cited_memories"],
                    "validation_scope": "retrieval and canonical citation resolution only; not prose entailment",
                    "passed": "peanut" in reflection["text"].lower()
                    and any(records["nisha"] in ids for ids in reflection_support)}
                # Resolve the native reflection's sources, then check explicit
                # candidate claims via the same service exposed to Hermes.
                native_ids = {ref for refs in reflection_support for ref in refs}
                if records["nisha"] not in native_ids:
                    raise AssertionError("native reflection did not resolve Nisha's canonical evidence")
                record = store.get(records["nisha"])
                checked = verify_memory_claims(scoped, store, [
                    {"text": text, "record_ids": [record.id],
                     "evidence": [{"record_id": record.id, "quote": record.text}]}
                    for text in ("Nisha is allergic to peanuts.", "Nisha weighs 90 kg.",
                                 "Nisha has a severe peanut allergy.")])
                result["claim_verification"] = {"text": checked["text"],
                    "verification": checked["verification"],
                    "passed": checked["verification"]["accepted"] == 1
                              and checked["text"] == "Nisha is allergic to peanuts."}
                plan = summarize_plan(scoped, scope="source:synthetic")
                publication = summarize_apply(scoped, scope="source:synthetic",
                    review=plan["review_digest"], actor="synthetic-evaluation")
                result["verified_summary_publication"] = publication
                store.hide(records["arjun"], reason="synthetic visibility safety test", actor="synthetic-evaluation")
                packet = broker.assemble(QUERIES[0][1])
                visible_resolver = SupportResolver(store, bank_id=bank)
                result["hidden_evidence_excluded"] = (
                    all(item.id != records["arjun"] for item in packet.items)
                    and all(records["arjun"] not in visible_resolver.resolve(fact) for fact in packet.facts))
                if exercise_outage:
                    # Explicit fault injection against only the owner-named model service.
                    subprocess.run(["systemctl", "--user", "stop", "hermes-memory-reranker.service"], check=True)
                    try:
                        offline = broker.assemble("Nisha")
                        result["outage"] = {"channels": offline.channels.as_dict(),
                            "no_semantic_facts": not offline.facts,
                            "lexical_still_available": offline.channels.lexical == "available" and bool(offline.items)}
                        result["outage"]["passed"] = (result["outage"]["no_semantic_facts"]
                            and result["outage"]["lexical_still_available"]
                            and offline.channels.derived in {"unavailable", "timeout"})
                    finally:
                        subprocess.run(["systemctl", "--user", "start", "hermes-memory-reranker.service"], check=True)
                        deadline = time.monotonic() + 30
                        while True:
                            try:
                                with urllib.request.urlopen("http://127.0.0.1:8185/health", timeout=1) as response:
                                    assert json.load(response)["model_loaded"]
                                break
                            except (OSError, AssertionError):
                                if time.monotonic() >= deadline:
                                    raise RuntimeError("reranker did not recover after synthetic outage")
                                time.sleep(.25)
                    recovered = broker.assemble("Nisha")
                    result["recovery_channels"] = recovered.channels.as_dict()
                    result["recovered"] = (recovered.channels.derived in {"available", "partial"}
                        and any(records["nisha"] in visible_resolver.resolve(fact) for fact in recovered.facts))
            finally:
                broker.close()
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
    finally:
        with EvidenceStore(scoped.db_path) as store:
            mappings = [dict(row) for row in store.db.execute(
                "SELECT document_id,operation_id FROM backend_documents WHERE bank_id=?", (bank,))]
        for mapping in mappings:
            try:
                operation = mapping["operation_id"]
                if operation and operation_state(client.operation(operation)) not in {"completed", "failed", "cancelled", "not_found"}:
                    client.cancel_operation(operation)
                    if operation_state(client.operation(operation)) not in {"completed", "failed", "cancelled", "not_found"}:
                        raise ValueError("operation has not settled; keep document reconciliation identity")
                client.delete_document(mapping["document_id"])
                result["cleanup"].append({"document": mapping["document_id"],
                                          "state": client.document_state(mapping["document_id"])["state"]})
            except Exception as error:
                result["cleanup"].append({"document": mapping["document_id"], "state": "unverified", "reason": str(error)})
    events = rerank_events(settings, marker)
    released = [json.loads(event["detail"]) for event in events if event["event"] == "released"]
    result["reranker_gate_events"] = events
    result["reranker_successful_calls"] = sum(row.get("outcome") == "succeeded" for row in released)
    result["reranker_actual_padded_tokens"] = sum(row.get("tokens", 0) for row in released)
    result["seconds"] = round(time.monotonic() - started, 4)
    result["passed"] = (not result["failure"] and len(result["checks"]) == len(QUERIES)
        and all(row["passed"] for row in result["checks"]) and result.get("hidden_evidence_excluded", False)
        and result.get("reflection", {}).get("passed", False)
        and result.get("claim_verification", {}).get("passed", False)
        and result.get("verified_summary_publication", {}).get("ok", False)
        and result["reranker_successful_calls"] >= len(QUERIES) * 2
        and len(result["cleanup"]) == len(FIXTURES) and all(row["state"] == "absent" for row in result["cleanup"])
        and (not exercise_outage or result.get("outage", {}).get("passed", False) and result.get("recovered", False)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exercise-outage", action="store_true", help="briefly stop/restart only the owned reranker")
    args = parser.parse_args()
    if not args.live:
        parser.error("requires explicit --live synthetic inference authorization")
    result = check(load_settings(), exercise_outage=args.exercise_outage)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"checks", "formation", "reranker_gate_events"}}, default=str))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
