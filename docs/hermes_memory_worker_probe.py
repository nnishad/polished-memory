"""One identified async retain in an isolated bank; exercise the real worker."""
import json
import time

from hermes_memory.backend.hindsight_client import HindsightClient
from hermes_memory.backend.worker_launcher import OperationLedger
from hermes_memory.config import load_settings, scoped_secret
from hermes_memory.processing.instance_gate import GateStore, gate_path

# Persisted before dispatch; do not change this identity and retry a lost answer.
SUBMISSION = "be9ff9d4-a3ce-4669-b42a-299b9523bc64"
DOCUMENT = "hermes-worker-test-bb72c84"


def main():
    settings = load_settings()
    client = HindsightClient(base_url=settings.hindsight_url, bank_id="hermes-worker-probe",
                             api_key=scoped_secret(settings, settings.hindsight_api_key_env))
    existing = client.operation(SUBMISSION)
    if existing.get("status") == "not_found":
        client.retain_async([{
            "document_id": DOCUMENT,
            "content": "On October 1, 2026, the synthetic worker test recorded that the red square was approved by the test operator.",
            "metadata": {"probe": "worker-lifecycle"},
        }], submission_id=SUBMISSION)
    state = None
    for _ in range(100):
        state = client.operation(SUBMISSION).get("status")
        if state in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.5)
    print(json.dumps({"submission": SUBMISSION, "backend_state": state}), flush=True)
    if state not in {"completed", "failed", "cancelled"}:
        print("Outcome unresolved; no resubmission or deletion performed.", flush=True)
        return 1
    try:
        units = client.document_state(DOCUMENT).get("count", 0)
        returned = len(client.recall("synthetic worker test red square", max_tokens=64).results)
        operation = client.operation(SUBMISSION)
        # Hindsight's batch submission is a parent, not an executable task.
        # Its child identities are the work the worker claims and accounts for.
        child_ids = [row["operation_id"] for row in operation.get("child_operations", [])]
        execution_ids = child_ids or [SUBMISSION]
        with GateStore(gate_path(settings), create=False) as store:
            recorded = [OperationLedger(store).get(identity) for identity in execution_ids]
        finished = all(row and row["state"] == "finished" for row in recorded)
        print(json.dumps({"derived_units": units, "recalled": returned,
                          "worker_accounted_finished": finished}), flush=True)
        return 0 if state == "completed" and units and returned and finished else 1
    finally:
        client.delete_document(DOCUMENT)
        print("Synthetic document deleted from isolated worker probe bank.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
