"""Stopping work that was started, and saying honestly what is still unknown.

Three things are involved and only two of them are ours. The queued job is ours: cancelling
it stops the next dispatch. The ledger's record of a backend operation is ours to write and
not ours to finish. The backend itself may be unreachable, may have completed while this
request was in flight, or may carry on after being told to stop — so a cancellation that
could not be confirmed leaves the operation `uncertain` with its intent marked `requested`,
which is a statement about the world, rather than `cancelled`, which would be a wish.

The order is the whole design. For an operation, the intent is written before anything is
called, so a crash between the two leaves "somebody asked this to stop and nobody knows more"
rather than a silently un-cancellable operation. For a job, the queue row closes first: that is
the half we control, and shutting it before the phone is picked up is what stops a second
dispatch racing the cancellation. Nothing here assumes an answer it did not get.
"""
from __future__ import annotations

from typing import Any

from ..storage.evidence import EvidenceError

__all__ = ["Canceller", "BACKEND_OUTCOMES", "run", "outstanding"]

# What the backend said, as the report names it. ``not_attempted`` is a supported answer: an
# installation with no route has no socket to call, and the intent stands regardless.
BACKEND_OUTCOMES = ("confirmed", "unsupported", "unreachable", "refused", "not_attempted")

# Distinguishes "work out whether there is a backend" from "there is none, on purpose".
_DERIVE = object()


class Canceller:
    """The composition the two halves could not perform on their own."""

    def __init__(self, store, *, jobs=None, ledger=None, client=None):
        self.store = store
        self._jobs = jobs
        self.ledger = ledger
        self.client = client

    @property
    def jobs(self):
        """The queue, built only if this cancellation needs it.

        A reading of the operation ledger has no business opening a profile store, and a
        caller that hands in neither gets an honest refusal rather than an AttributeError.
        """
        from .jobs import JobQueue

        if self._jobs is None:
            if self.store is None:
                raise EvidenceError("this canceller has neither a store nor a queue, so it "
                                    "can report on operations and stop nothing")
            self._jobs = JobQueue(self.store)
        return self._jobs

    # -- the two doors ---------------------------------------------------------

    def cancel_job(self, job_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        """Stop a queued job, and the backend operation it already handed off to."""
        actor, reason = _named(actor, "actor"), _named(reason, "reason")
        job = self.jobs.get(job_id)
        if job is None:
            raise EvidenceError(f"no queued job {job_id!r}")
        state = self.jobs.cancel(job_id, actor=actor, reason=reason)
        operations = []
        if job.backend_operation_id:
            operations.append(self.cancel_operation(str(job.backend_operation_id),
                                                    actor=actor, reason=reason))
        return {"job": job_id, "job_state": state, "actor": actor,
                "reason": reason[:400], "operations": operations,
                "spent": {"tokens": int(job.tokens_used), "attempts": int(job.attempts)},
                "note": ("the queue will not offer this job again. What the backend does "
                         "with an operation it already holds is reported per operation "
                         "above, never assumed")}

    def cancel_operation(self, operation_id: str, *, actor: str,
                         reason: str) -> dict[str, Any]:
        """Ask that one backend operation stop, and record what was actually learned."""
        actor, reason = _named(actor, "actor"), _named(reason, "reason")
        if self.ledger is None:
            return {"operation": operation_id, "backend": "not_attempted",
                    "cancellation": "absent",
                    "note": "no admission ledger is wired here, so this operation was never "
                            "recorded as ours to cancel"}
        record = self.ledger.get(operation_id)
        if not record:
            raise EvidenceError(f"no record of operation {operation_id!r}")
        intent = self.ledger.request_cancellation(operation_id, actor=actor)
        report: dict[str, Any] = {"operation": operation_id, "kind": record.get("kind"),
                                 "state_before": record.get("state"),
                                 "resource": record.get("resource"),
                                 "bank": record.get("bank_id"),
                                 "cancellation": intent.get("cancellation"),
                                 "actor": actor, "reason": reason[:400]}
        if intent.get("cancellation") != "requested":
            # The ledger would not take the intent, because the row has already settled. The
            # answer is in the record and did not come from this request, so no call goes out
            # and no credit is taken for a send that happened. Which states count as settled
            # stays the ledger's own business rather than a list copied here.
            report.update({"backend": "not_attempted",
                           "note": f"the operation is already recorded as {record['state']}, "
                                   "and a cancellation cannot reach into the past"})
            return report
        if self.client is None:
            report.update({"backend": "not_attempted",
                           "note": "no backend route is configured, so nothing was told; the "
                                   "intent is on file and the operation may still be running "
                                   "wherever it was started"})
            return report
        report.update(self._ask(operation_id))
        return report

    # -- the call ------------------------------------------------------------

    def _ask(self, operation_id: str) -> dict[str, Any]:
        from ..backend.capabilities import UnsupportedCapability
        from ..backend.hindsight_client import HindsightError, HindsightUnavailable

        try:
            answer = self.client.cancel_operation(operation_id)
        except UnsupportedCapability as error:
            # The pinned revision does not offer a cancellation endpoint. That is not a
            # failure to cancel; it is an answer about the backend's shape.
            return {"backend": "unsupported", "error": str(error)[:300],
                    "state_after": self._state_of(operation_id),
                    "note": "this backend version has no cancellation operation; the intent "
                            "stands and the worker will not start the next attempt"}
        except HindsightUnavailable as error:
            return {"backend": "unreachable", "error": str(error)[:300],
                    **self._mark(operation_id, "uncertain",
                                 error=f"cancel could not be confirmed: {str(error)[:200]}"),
                    "note": "the request may still be running upstream; this is recorded as "
                            "uncertain rather than as cancelled"}
        except HindsightError as error:
            return {"backend": "refused", "error": str(error)[:300],
                    "state_after": self._state_of(operation_id),
                    "note": "the backend refused the cancellation, so nothing about the "
                            "operation changed here"}
        moved = self._mark(operation_id, "cancelled", error=None)
        # An acknowledgement can arrive for a row this ledger already settled — the worker
        # finished it while this request was in flight. Both facts are reported; the
        # convenient one is not written over the other.
        return {"backend": "confirmed",
                "answer": {key: value for key, value in dict(answer or {}).items()
                           if key in ("state", "operation_id", "status", "cancelled")} or None,
                **moved,
                "note": ("the backend acknowledged the stop; this ledger's own record had "
                         "already settled and is left as it stands"
                         if moved.get("refused_transition") else
                         "the backend acknowledged it; the ledger now says so as well")}

    def _mark(self, operation_id: str, state: str, *,
              error: str | None) -> dict[str, Any]:
        moved = self.ledger.mark(operation_id, state, error=error)
        confirmed = None
        if state == "cancelled":
            confirmed = self.ledger.confirm_cancellation(operation_id)["cancellation"]
        report = {"state_after": moved.get("state"),
                  "cancellation": confirmed or moved.get("cancellation")}
        if moved.get("refused_transition"):
            report["refused_transition"] = moved["refused_transition"]
        return report

    def _state_of(self, operation_id: str) -> Any:
        return self.ledger.get(operation_id).get("state")

    # -- the reading ---------------------------------------------------------

    def owed(self, *, limit: int = 25) -> list[dict[str, Any]]:
        """Cancellations asked for that nobody has confirmed.

        The absence of an answer is a fact that outlives the process that asked, which is why
        this is a table read rather than a callback. It reads the *admission* ledger, where an
        operation lives, and not the profile store, where the queue does — different files on
        purpose.
        """
        if self.ledger is None or not self.ledger.present:
            return []
        return self.ledger.cancellation_owed(limit=_bounded(limit))


def _named(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise EvidenceError(f"{label} must be named: a cancellation is somebody's decision, "
                            "and it is the only way to tell whose later")
    return text[:200]


def _bounded(value: int) -> int:
    return value if isinstance(value, int) and 1 <= value <= 200 else 20


# -- the doors ---------------------------------------------------------------

def run(settings, *, job: str | None = None, operation: str | None = None,
        actor: str = "", reason: str = "", store=None,
        client: Any = _DERIVE) -> dict[str, Any]:
    """Cancel one thing by name, over the installation this settings object names.

    The admission ledger is opened writable, because filing an intent is a durable write — but
    only when its file already exists. Creating a ledger to be told there is nothing in it
    would be this command inventing state. A queued job does not need that ledger: the queue
    is ours, and closing it is the whole of what can be done here.
    """
    from ..backend.worker_launcher import OperationLedger
    from ..processing.instance_gate import gate_path, instance_gate
    from ..storage.evidence import EvidenceStore

    if bool(job) == bool(operation):
        raise EvidenceError("a cancellation names exactly one of a queued job or a backend "
                            "operation — not both, and not neither")
    path = settings.db_path
    if not store and not path.exists():
        return {"ok": False, "refused": f"no canonical store at {path}; run `hermes-memory "
                                        "init` or `hermes-memory setup` first"}
    ledger_at = gate_path(settings)
    if operation and not ledger_at.is_file():
        return {"ok": False, "refused": f"no admission ledger at {ledger_at}; nothing has "
                                        "been dispatched from this installation, so there is "
                                        "no backend operation to stop"}
    opened = store or EvidenceStore(path)
    gate = instance_gate(settings) if ledger_at.is_file() else None
    try:
        if client is _DERIVE:
            client = _configured_client(settings)
        canceller = Canceller(opened, client=client,
                              ledger=None if gate is None else OperationLedger(gate.store))
        if job:
            report = canceller.cancel_job(job, actor=actor, reason=reason)
        else:
            report = canceller.cancel_operation(str(operation), actor=actor, reason=reason)
    finally:
        if gate is not None:
            gate.close()
        if store is None:
            opened.close()
    report["ok"] = True
    report["profile"] = getattr(settings, "profile", None)
    return report


def outstanding(settings, *, limit: int = 25) -> dict[str, Any]:
    """The intents on file with no answer — a reading, so it creates and migrates nothing."""
    from ..backend.worker_launcher import OperationLedger
    from ..processing.instance_gate import gate_path, status_gate

    gate = status_gate(settings)
    if gate is None:
        return {"ledger": "absent", "at": str(gate_path(settings)), "owed": [],
                "note": "nothing has been queued from this installation yet"}
    try:
        ledger = OperationLedger(gate.store)
        if not ledger.present:
            return {"ledger": "predates operations", "owed": [],
                    "note": "this admission ledger has no operation table; opening the gate "
                            "for a real dispatch brings it up"}
        return {"ledger": "present", "owed": Canceller(None, ledger=ledger).owed(
            limit=limit)}
    finally:
        gate.close()


def _configured_client(settings) -> Any:
    """The backend this installation would call, or ``None`` when it has no route.

    No route is a supported answer rather than an error to raise: the intent still goes on
    file, and the report says nothing was told to anybody.
    """
    from .formation import FormationError, backend_client

    try:
        return backend_client(settings)
    except FormationError:
        return None
