"""Opt-in synthetic model checks through owned admission, never personal archives.

Prints a reproducible receipt. A small smoke test is not a language-quality gate.
No production schema migration, runtime switch or Telegram delivery is performed.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path

from hermes_memory.backend.hindsight_client import HindsightError, operation_state
from hermes_memory.config import load_settings, scoped_settings
from hermes_memory.ids import digest
from hermes_memory.processing.formation import backend_client, formation_apply, formation_plan
from hermes_memory.processing.summarization import client_for
from hermes_memory.processing.synthesis import SYNTHESIS_CONTRACT
from hermes_memory.storage.evidence import EvidenceStore

CASES = (
    ("roman-hinglish", "Mujhe chai nahi chahiye, coffee pasand hai.", "What does the speaker want to drink?"),
    ("devanagari-hindi", "मुझे चाय नहीं चाहिए, कॉफी पसंद है।", "What does the speaker want to drink?"),
    ("english", "I do not want tea; I like coffee.", "What does the speaker want to drink?"),
)


def isolated_settings(settings, scratch, bank):
    root = Path(scratch).resolve()
    scoped = scoped_settings(settings, profile="synthetic-eval", data_dir=root,
                             bank_id=bank, credential_scope=settings.credential_scope)
    assert_isolated(settings, scoped, root)
    return scoped


def assert_isolated(original, scoped, scratch):
    """Fail before opening a writer or backend socket if any canonical path escaped."""
    root = Path(scratch).resolve()
    for name in ("data_dir", "db_path", "blob_dir"):
        path = getattr(scoped, name).resolve()
        if not path.is_relative_to(root) or path == getattr(original, name).resolve():
            raise ValueError(f"synthetic isolation refused: {name} is not an independent scratch path")
    if scoped.bank_id == original.bank_id or not scoped.bank_id.startswith("eval-"):
        raise ValueError("synthetic isolation refused: no distinct evaluation bank")


def synthesis(settings, count):
    rows = []
    for language, text, query in CASES[:count]:
        started = time.monotonic()
        try:
            output = client_for(settings).synthesize(query, evidence=[{
                "record_id": "synthetic-" + language, "revision": "1", "source": "synthetic",
                "role": "user", "text": text, "occurred_at": None}], max_tokens=512)
            row = {"language": language, "state": "model_support_checked",
                   "claims": output["claims"], "input_tokens": output["input_tokens"],
                   "output_tokens": output["output_tokens"], "verification": output["verification"],
                   "faithfulness": "manual_review_still_required"}
        except HindsightError as error:
            row = {"language": language, "state": "withheld", "reason": str(error)}
        rows.append({**row, "seconds": round(time.monotonic() - started, 4)})
    return {"mode": "synthesis", "results": rows, "contract": SYNTHESIS_CONTRACT}


def roundtrip(settings, count):
    # Every remote document and its durable operation identity has a local ledger
    # before submission. An isolated bank cannot contaminate ordinary recall.
    # Keep the durable synthetic ledger even after a failed/uncertain cleanup;
    # deleting it on exception would discard the only reconciliation identities.
    with tempfile.TemporaryDirectory(prefix="hermes-memory-synthetic-", delete=False) as scratch:
        bank = "eval-" + uuid.uuid4().hex
        scoped = isolated_settings(settings, scratch, bank)
        client = backend_client(scoped)
        client.health()  # an unavailable service creates no remote bank or documents
        with EvidenceStore(scoped.db_path) as store:
            for language, text, _query in CASES[:count]:
                store.commit({"source": "synthetic", "source_id": language, "revision": "1",
                    "kind": "message", "text": text, "observed_at": "2026-10-02T12:00:00+00:00",
                    "occurred_at": None, "occurred_precision": "unknown",
                    "metadata": {"role": "user", "origin": "owner-statement", "independent": True}})
        results, cleanup = [], []
        report = None
        started = time.monotonic()
        failure = None
        try:
            plan = formation_plan(scoped, limit=count)
            report = formation_apply(scoped, actor="synthetic-evaluation", review=plan["review_digest"],
                                     limit=count, max_jobs=count, client=client,
                                     follow_poll_s=2.0, follow_max_polls=30)
            for language, _text, query in CASES[:count]:
                t = time.monotonic()
                outcome = client.recall(query, max_tokens=300)
                results.append({"language": language, "seconds": round(time.monotonic() - t, 4),
                                "backend_outcome": outcome.as_dict(),
                                "faithfulness": "manual_review_required"})
        except (HindsightError, ValueError) as error:
            failure = str(error)
        finally:
            with EvidenceStore(scoped.db_path) as store:
                mappings = [dict(row) for row in store.db.execute(
                    "SELECT document_id,operation_id FROM backend_documents WHERE bank_id=?", (bank,))]
            for mapping in mappings:
                try:
                    operation = mapping["operation_id"]
                    if operation and operation_state(client.operation(operation)) not in {
                            "completed", "failed", "cancelled", "not_found"}:
                        client.cancel_operation(operation)
                        if operation_state(client.operation(operation)) not in {
                                "completed", "failed", "cancelled", "not_found"}:
                            cleanup.append({"document_id": mapping["document_id"],
                                            "state": "pending_operation_settlement"})
                            continue
                    client.delete_document(mapping["document_id"])
                    cleanup.append({"document_id": mapping["document_id"],
                                    "state": client.document_state(mapping["document_id"])["state"]})
                except HindsightError as error:
                    cleanup.append({"document_id": mapping["document_id"], "state": "unverified",
                                    "reason": str(error)})
        return {"mode": "roundtrip", "bank_id": bank,
                "synthetic_ledger": str(scoped.db_path), "failure": failure,
                "seconds": round(time.monotonic() - started, 4), "formation": report,
                "results": results, "cleanup": cleanup,
                "bank_metadata": "empty bank shell retained; no bank deletion API in this client"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="explicitly authorize bounded synthetic inference")
    parser.add_argument("--mode", choices=("synthesis", "roundtrip"), default="synthesis")
    parser.add_argument("--cases", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if not args.live:
        parser.error("live evaluation requires --live; it never silently contacts a model")
    settings = load_settings()
    receipt = {"version": "memory-intelligence-smoke-v1", "synthetic_only": True,
               "corpus_digest": digest(CASES), "model": settings.text_route.model,
               "embedding_model": settings.embeddings_route.model if settings.embeddings_route else None,
               "quality_gate": "not_established_by_three_smoke_cases",
               **(synthesis(settings, args.cases) if args.mode == "synthesis" else roundtrip(settings, args.cases))}
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
