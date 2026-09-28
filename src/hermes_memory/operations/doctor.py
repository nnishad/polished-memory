"""C14 doctor: name the broken part, and say what would actually fix it.

Status answers "what is happening"; the doctor answers "why is it not". The two are
separate because they are asked at different times and by different people: status is
watched, the doctor is run when something is wrong, and a run of it must never make
the situation worse. So every default check is a read over the canonical store — no
writes, no inference, and no outbound request. A probe is an explicit, bounded action
the operator asks for by name, and the report says which probes ran so a later reader
can tell what was actually looked at.

Findings carry a remedy where one exists. "degraded" is not an answer; "run setup, or
pause the gmail connector" is.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..ids import now
from ..sources.base import redact_secrets
from .status import (DEGRADED, PAUSED, UNCONFIGURED, StatusReporter, snapshot)

__all__ = ["Doctor", "Finding", "OK", "WARN", "FAIL", "SEVERITIES"]

OK = "ok"
WARN = "warn"
FAIL = "fail"
SEVERITIES = (OK, WARN, FAIL)

# A record scan looks at the newest evidence only. A credential in archive form is
# still a credential, but a doctor that reads every row is one nobody runs twice.
SCAN_RECORDS = 500
# How old an uncheckpointed WAL may get before it is worth mentioning, in bytes.
WAL_WARN_BYTES = 64 * 1024 * 1024
#: What the probe retains. It is an assertion about a synthetic event with a date and an
#: object in it, and that shape is deliberate: the engine's retain answers before the model
#: has decided what, if anything, the text is worth, and a self-describing sentence — which is
#: what this probe used to retain — is the sort of thing an extractor answers "nothing" to. On
#: this installation that sentence produced no memory unit at all about a third of the time,
#: so the door reported a working backend as silently broken. A fact-shaped synthetic sentence
#: produced exactly one unit every time.
PROBE_TEXT = ("The synthetic probe recorded that gate reservation 118 was settled by "
              "the operator on 2026-09-02.")
#: Asked of the same words the document was written with, so a finding of nothing is about
#: the search path and not about the door's choice of phrasing.
PROBE_QUERY = "synthetic probe gate reservation 118"
#: The id the probe retains under. It cannot contain `_` or `~`, because the engine escapes
#: them when it composes chunk ids and an escaped chunk cannot be traced back to a document —
#: and this framework refuses such an id rather than sending it, so a name written with an
#: underscore here fails the probe before it ever reaches the backend.
PROBE_DOCUMENT_ID = "doctor-probe"
#: How many times, and how far apart, the door looks for the units its retain should have
#: caused. Derivation is a model call, so it may lag the write; waiting for it is what keeps a
#: slow answer from being reported as a missing one. Measured on the live engine the units were
#: there immediately every time, so this window is the bound, not the expected path — and the
#: time it did spend is in the evidence, so an operator can tell the two apart.
PROBE_SETTLE_READS = 4
PROBE_SETTLE_WAIT_S = 10.0


@dataclass(frozen=True)
class Finding:
    """One check's answer, with the action that follows from it."""

    check: str
    severity: str
    detail: str
    remedy: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}; "
                             f"findings are {list(SEVERITIES)}")

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.check, "severity": self.severity, "detail": self.detail,
                "remedy": self.remedy, **self.evidence}


class Doctor:
    """Read-only diagnosis of one installation."""

    def __init__(self, store, *, settings: Any = None, status: StatusReporter | None = None,
                 backend: Callable[[], Any] | None = None,
                 probe_bank: str = "hermes-doctor-probe"):
        self.store = store
        self.db = store.db
        self.settings = settings
        self.status = status or StatusReporter(store, settings=settings)
        self._backend = backend
        self.probe_bank = probe_bank

    # -- entry point ---------------------------------------------------------

    def examine(self, *, connectivity: bool = False, synthetic: bool = False,
                profile: str | None = None) -> dict[str, Any]:
        """Run every check. Probes only happen when they were asked for by name.

        ``synthetic`` implies ``connectivity``: a round trip that writes to the
        backend is not something to attempt against a backend that is down.
        """
        checks = [self.layout, self.database, self.schema, self.configuration,
                  self.coverage, self.queue, self.background, self.provenance, self.lineage,
                  self.erasure, self.delivery, self.gate, self.inference_outcomes,
                  self.credentials_in_records,
                  self.leases, self.backend_ledger, self.release]
        with snapshot(self.db):
            findings = [check() for check in checks]
        if connectivity or synthetic:
            findings.append(self.backend_connectivity())
        if synthetic:
            findings.append(self.backend_synthetic_round_trip())
        worst = max((SEVERITIES.index(item.severity) for item in findings), default=0)
        return {
            "checked_at": now(),
            "profile": profile or "default",
            "severity": SEVERITIES[worst],
            "ok": worst < SEVERITIES.index(FAIL),
            "exit_code": 1 if worst >= SEVERITIES.index(FAIL) else 0,
            "probes": {"connectivity": bool(connectivity or synthetic),
                       "synthetic": bool(synthetic)},
            "findings": [item.as_dict() for item in findings],
            "actions": [item.remedy for item in findings if item.remedy],
        }

    # -- the installation itself ---------------------------------------------

    def layout(self) -> Finding:
        """The owned paths must exist and must be nobody else's to read."""
        settings = self.settings
        if settings is None:
            return Finding("layout", WARN, "no configuration was supplied",
                           "run setup, or pass --hermes-home for this profile")
        missing = [str(path) for path in (settings.data_dir, settings.db_path,
                                          settings.blob_dir) if not os.path.exists(path)]
        if missing:
            return Finding("layout", FAIL, f"missing: {', '.join(missing)}",
                           "run setup to create the owned data directory",
                           {"missing": missing})
        exposed = [str(settings.data_dir)] if _world_readable(settings.data_dir) else []
        if exposed:
            return Finding("layout", FAIL, f"{', '.join(exposed)} is readable beyond the "
                                           "owner; it holds private evidence",
                           f"chmod 700 {settings.data_dir}", {"exposed": exposed})
        return Finding("layout", OK, "the owned paths exist and are owner-only",
                       evidence={"data_dir": str(settings.data_dir)})

    def database(self) -> Finding:
        """Corruption and journal shape, read from SQLite itself."""
        check = self.db.execute("PRAGMA quick_check").fetchone()[0]
        if str(check).lower() != "ok":
            return Finding("database", FAIL, f"quick_check says: {check}",
                           "restore from a backup, then re-run doctor", {"check": check})
        mode = str(self.db.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if mode != "wal":
            return Finding("database", FAIL, f"journal mode is {mode!r}, not 'wal'",
                           "the durability guarantees assume WAL; restart under this build",
                           {"journal_mode": mode})
        path = getattr(self.settings, "db_path", None)
        wal = _size_of(str(path) + "-wal") if path else 0
        evidence = {"journal_mode": mode, "wal_bytes": wal,
                    "pages": int(self.db.execute("PRAGMA page_count").fetchone()[0])}
        if wal > WAL_WARN_BYTES:
            return Finding("database", WARN,
                           f"the write-ahead log is {wal // 1024 // 1024} MiB, waiting to "
                           "be checkpointed",
                           "let an idle moment pass, or stop the services and run a "
                           "checkpoint during maintenance",
                           evidence)
        return Finding("database", OK, "the store is intact and in WAL mode",
                       evidence=evidence)

    def schema(self) -> Finding:
        """A store this build cannot read is not a store to run against."""
        from ..storage.migrations import MIGRATIONS, current_version

        applied = current_version(self.db)
        head = len(MIGRATIONS)
        evidence = {"applied": applied, "head": head}
        if applied > head:
            return Finding("schema", FAIL,
                           f"the store is at migration {applied}; this build knows {head}",
                           "upgrade the framework — running an older build against a newer "
                           "schema is refused on purpose", evidence)
        if applied < head:
            return Finding("schema", WARN, f"{head - applied} migration(s) outstanding",
                           "run setup or upgrade to apply them", evidence)
        return Finding("schema", OK, f"schema is at {applied} of {head}", evidence=evidence)

    def configuration(self) -> Finding:
        """What the operator asked for, and the contradictions in it."""
        settings = self.settings
        if settings is None:
            return Finding("configuration", WARN, "no configuration was supplied",
                           "run setup")
        notes = []
        if not settings.hindsight_url:
            notes.append("no backend URL, so capture and local recall only")
        if settings.hindsight_url and not settings.inference_enabled:
            notes.append("a backend is configured but inference is not enabled")
        if settings.inference_enabled and settings.background_budget_tokens <= 0:
            notes.append("inference is enabled with no background budget, so nothing "
                         "will ever form")
        if not settings.owner_principal:
            notes.append("no owner principal, so no confirmation, revocation or delivery "
                         "can be attributed")
        if not settings.gate_token:
            notes.append("no gate token, so the loopback mapper cannot admit a caller")
        if notes:
            return Finding("configuration", WARN, "; ".join(notes),
                           "set the named values in the owned env file",
                           {"capture_only": settings.capture_only})
        return Finding("configuration", OK,
                       "inference is routed to an approved local or LAN endpoint with a "
                       "budget and an owner",
                       evidence={"capture_only": settings.capture_only,
                                 "allowed_hosts": sorted(settings.allowed_inference_hosts),
                                 "budget": settings.background_budget_tokens})

    # -- what the pipeline is doing ------------------------------------------

    def coverage(self) -> Finding:
        report = self.status.capture()
        if report.state == UNCONFIGURED:
            return Finding("coverage", WARN, "no source is registered",
                           "add one with setup or import a fixture",
                           {"state": report.state})
        spool = report.evidence.get("spool") or {}
        if report.evidence.get("spool_unread"):
            # Named separately from the degraded case below because the action differs: the
            # source is not unreachable and nobody paused it. The host handed turns over and
            # this installation has never opened the file they are in.
            return Finding("coverage", FAIL, report.detail,
                           "`hermes-memory maintain` drains the host spool into the record set "
                           "once; if the pending count keeps growing between passes, the "
                           "background loop is not running — see the `background` check",
                           {"state": report.state, "spool": spool,
                            "unhealthy": report.evidence.get("unhealthy")})
        degraded = report.state == DEGRADED
        return Finding("coverage", _severity_for(report.state), report.detail,
                       "run the connector again, or pause it if the source is gone"
                       if degraded else None,
                       {"state": report.state,
                        "unhealthy": report.evidence.get("unhealthy"),
                        "open_gaps": report.evidence.get("open_gaps")})

    def queue(self) -> Finding:
        observations = self.status.observations()
        counts = observations.evidence.get("queue") or {}
        stuck = observations.evidence.get("stuck") or {}
        age = self.status.queue_age()
        if stuck:
            return Finding("queue", FAIL, f"{stuck} job(s) will not be retried by "
                                          "themselves",
                           "reconcile or cancel them; a quarantined job's inputs are "
                           "named in the job row", {"stuck": stuck, "age": age})
        if age.get("stale"):
            return Finding("queue", WARN,
                           f"the oldest waiting job has waited {age['oldest_seconds']}s",
                           "nothing drains this queue by itself: run `hermes-memory form` "
                           "to work it, or resume a stage an operator paused",
                           # Named rather than called "unattended": the stage report uses
                           # that word for the opposite claim, and two meanings in one key is
                           # how a reader ends up certain of the wrong thing.
                           {"queue": counts, "age": age, "operator_needed": True})
        if observations.evidence.get("formation_owed"):
            unprojected = observations.evidence.get("unprojected")
            # The queue holds no live work, and that is exactly the finding: formation is an
            # approved act, so nothing queues itself, and records can sit captured and unasked
            # forever while a queue of finished jobs looks worked off.
            return Finding("queue", WARN,
                           f"{unprojected} live record(s) have no backend projection and "
                           "nothing is queued to form them",
                           "`hermes-memory form` shows the list and prices it; the pass itself "
                           "is the owner's to approve with --review, because it spends a device "
                           "and a model call",
                           {"unprojected": unprojected, "queue": counts, "age": age,
                            "state": observations.state})
        return Finding("queue", _severity_for(observations.state), observations.detail,
                       evidence={"queue": counts, "age": age,
                                 "state": observations.state})

    def background(self) -> Finding:
        """The zero-inference pass: is it scheduled, and is it actually running?

        Separate from ``queue`` because the two fail differently. An empty job queue with
        reminders going unswept is a loop that died, and a machine with nothing to do looks
        the same from the queue alone.
        """
        report = self.status.background_pass()
        if not report["scheduled"]:
            return Finding("background", WARN,
                           "nothing runs the background pass on its own, so due reminders "
                           "wait for somebody to notice them",
                           "start the runtime unit, or set "
                           "HERMES_MEMORY_MAINTENANCE_INTERVAL_S to how often it should "
                           "look; `hermes-memory maintain` is the pass itself", report)
        if report["failed_section"]:
            return Finding("background", FAIL,
                           f"the background pass raised in section "
                           f"{report['failed_section']!r}: {report['error']}",
                           "the loop is running but the pass never finishes, so every "
                           "section after it is not happening on any period — reproduce it "
                           "with `hermes-memory maintain`, which raises the same error in "
                           "your own terminal", report)
        if report["behind"] and report["waiting"]:
            return Finding("background", WARN,
                           f"{report['waiting']} reminder(s) are due and {report['note']}",
                           "check the runtime unit is up, or run `hermes-memory maintain` "
                           "once by hand", report)
        return Finding("background", OK, report["note"], evidence=report)

    def provenance(self) -> Finding:
        summaries = self.status.summaries()
        unsupported = summaries.evidence.get("unsupported") or {}
        if unsupported.get("artifacts"):
            return Finding("provenance", FAIL,
                           f"{unsupported['artifacts']} derived artifact(s) cite evidence "
                           "that no longer resolves",
                           "withdraw or rebuild them; a citation that cannot be opened is "
                           "not support", {"details": unsupported.get("details")})
        return Finding("provenance", _severity_for(summaries.state), summaries.detail,
                       evidence={"state": summaries.state})

    def lineage(self) -> Finding:
        """Edges that point at nothing, and evidence nothing can reach.

        A dependency row whose parent record is absent entirely is a write that happened
        outside the transaction that owns it — a restore that skipped a parent, or an erase
        that took a row and left its edges. Both are the provenance gap §14 names, and both
        are countable without opening a record, so no private text reaches a report on the way
        to saying that something is missing.
        """
        from ..storage.lineage import Lineage

        lineage = Lineage(self.store)
        edges = lineage.dangling(limit=25)
        orphans = lineage.orphans(limit=25)
        if edges:
            return Finding("lineage", FAIL,
                           f"{len(edges)} lineage edge(s) cite a record that is not in this "
                           "store",
                           "`hermes-memory explain` the artifact each edge belongs to; a "
                           "restore re-verifies them and an erasure should have taken the "
                           "edge with the row",
                           {"edges": edges[:5], "orphans": len(orphans)})
        return Finding("lineage", OK, "every recorded dependency resolves",
                       evidence={"orphans": len(orphans),
                                 "note": "an unreachable record is not a fault — most "
                                         "evidence is a leaf — but a large count means a "
                                         "connector committed without its participants"})

    def erasure(self) -> Finding:
        backlog = self.status.erasure_backlog()
        if backlog["obligations_open"]:
            return Finding("erasure", FAIL,
                           f"{backlog['obligations_open']} erasure obligation(s) are not "
                           "verified gone everywhere",
                           "run the backend reconcile pass, then verify each target",
                           backlog)
        if backlog["awaiting_owner"]:
            return Finding("erasure", WARN,
                           f"{backlog['awaiting_owner']} intent(s) await the owner's "
                           "confirmation",
                           "an agent cannot confirm an erasure: `hermes-memory owner --list` "
                           "shows the digest, and `owner --confirm-forgetting` applies it",
                           backlog)
        return Finding("erasure", OK, "nothing is half-forgotten", evidence=backlog)

    def delivery(self) -> Finding:
        report = self.status.delivery()
        evidence = report.evidence
        if evidence.get("unproven"):
            return Finding("delivery", FAIL,
                           f"{evidence['unproven']} artifact(s) have no delivery proof",
                           "correlate the host receipts, or suppress what cannot be proved",
                           {"by_state": evidence.get("by_state")})
        if evidence.get("past_expiry"):
            return Finding("delivery", FAIL,
                           f"{evidence['past_expiry']} artifact(s) aged out unsent",
                           "the host transport is not claiming the outbox; start it",
                           {"by_state": evidence.get("by_state")})
        return Finding("delivery", _severity_for(report.state), report.detail,
                       evidence={"state": report.state,
                                 "in_flight": evidence.get("in_flight")})

    def gate(self) -> Finding:
        report = self.status.resource_gate()
        operations = report.evidence.get("operations") or {}
        # One severity path for every stage: the reading decides, and the remedy
        # follows it. A second FAIL branch here would mean two places that have to
        # agree about how bad "uncertain" is, and only one of them would be tested.
        remedy = ("the resource stays blocked until an operator settles them; `hermes-memory "
                  "gate` names the reservation and `gate --resolve` settles it with a stated "
                  "outcome and reason" if report.state == DEGRADED else None)
        if operations.get("unattributed"):
            # An operation nobody could charge is the launcher saying it could not place the
            # task on a route, which is a version-contract question, not a stuck device.
            remedy = ("a backend operation matched no route; run "
                      "`python -m hermes_memory.backend.worker_launcher --check` in the "
                      "backend environment and read the revision contract")
        if operations.get("cancellations_owed"):
            # Somebody asked work to stop and no answer came back. The intent is durable, so
            # this is a queue with a name on it rather than a general airlessness.
            remedy = ("a cancellation was asked for with no answer; `hermes-memory "
                      "cancel --list` names the operations, and asking again is the way to "
                      "settle each one")
        return Finding("gate", _severity_for(report.state), report.detail, remedy,
                       {"usage": report.evidence.get("usage"),
                        "blocked": report.evidence.get("blocked"),
                        "uncertain": report.evidence.get("uncertain"),
                        "operations": operations,
                        "state": report.state})

    def release(self) -> Finding:
        """Is this the release its own compatibility manifest describes?

        The manifest is generated from the code that enforces each fact in it, so a
        difference means one of two things: somebody edited the file to make a claim the code
        does not support, or a tree was patched without a release being cut. Both are worth
        saying out loud, and neither is visible from any other check here.
        """
        from ..install.compatibility import manifest_path, read_shipped, verify

        checked = verify(settings=self.settings)
        path = manifest_path(self.settings)
        if checked["ok"]:
            shipped = read_shipped(path=path)
            digests = checked["digests"]
            # "agrees" is about the claims, which is what a running build can be held to. The
            # digest names the tree the manifest was written from, and a tree that has been
            # edited since is not a contradiction of this finding — so both numbers are shown
            # rather than one of them described as agreement.
            return Finding("release", OK,
                           f"{path.name}'s claims match this build; written at framework "
                           f"{shipped['framework_digest'][:12]}"
                           + ("" if digests["agree"]
                              else f", this tree at {digests['framework'][:12]}")
                           + f", pinned to Hindsight {shipped['hindsight']['engine_pinned']}",
                           evidence={"manifest": str(path),
                                     "framework_digest": shipped["framework_digest"],
                                     "tree_digest": digests["framework"],
                                     "pinned": shipped["hindsight"]["engine_pinned"],
                                     "schema": shipped["schema"]["evidence_migrations"]})
        if checked.get("absent"):
            # A wheel installed on its own carries no manifest, which is a fact about the
            # installation rather than a broken memory: nothing here contradicts this build,
            # there is simply nothing that states what it is compatible with. Worth a WARN
            # and a remedy that names a release tree, not a FAIL that tells somebody to
            # regenerate a file their machine has no source for.
            return Finding("release", WARN,
                           "; ".join(checked["differences"])[:300], checked["remedy"],
                           {"manifest": None, "differences": checked["differences"]})
        return Finding("release", FAIL,
                       f"{path.name} no longer says what this build does: "
                       + "; ".join(checked["differences"])[:300],
                       checked["remedy"],
                       {"manifest": str(path), "differences": checked["differences"]})

    def credentials_in_records(self) -> Finding:
        """A credential in a record is in every prompt built from it. Find it here."""
        rows = self.db.execute(
            "SELECT id, text, metadata FROM records WHERE deleted=0 ORDER BY ingested_at "
            "DESC LIMIT ?", (SCAN_RECORDS,)).fetchall()
        hits = [row["id"] for row in rows
                if redact_secrets(row["text"]) != row["text"]
                or redact_secrets(row["metadata"]) != row["metadata"]]
        if hits:
            return Finding("credentials", FAIL,
                           f"{len(hits)} recent record(s) carry what looks like a secret",
                           "erase them through the erasure ledger, then rotate the "
                           "credential at its source",
                           {"records": hits[:20],
                            "sample": redact_secrets(rows[0]["text"])[:120] if hits else None,
                            "scanned": len(rows)})
        return Finding("credentials", OK,
                       "no credential-shaped text in the newest evidence",
                       evidence={"scanned": len(rows)})

    def leases(self) -> Finding:
        """Fences left behind by a worker that died holding them."""
        moment = datetime.now(timezone.utc).timestamp()
        stale = self.db.execute(
            "SELECT source, holder, lease_until FROM connectors WHERE lease IS NOT NULL "
            "AND lease_until < ? ORDER BY lease_until", (moment,)).fetchall()
        held = [f"{row['source']} by {row['holder']}" for row in stale]
        if held:
            return Finding("leases", WARN,
                           f"{len(held)} connector lease(s) outlived their holder: "
                           f"{', '.join(held[:5])}",
                           "the next pass takes the lease once it expires; a source that "
                           "always shows this is running two ways at once",
                           {"stale": held})
        return Finding("leases", OK, "no lease is being held past its word",
                       evidence={"held": len(stale)})

    # -- the backend, without touching it ------------------------------------

    def client(self) -> Any:
        """The configured backend client, or None. Built lazily; nothing here talks yet."""
        if self._backend is not None:
            return self._backend()
        settings = self.settings
        if settings is None or not settings.hindsight_url:
            return None
        from ..backend.hindsight_client import HindsightClient
        from ..config import scoped_secret

        # Scoped through the profile ledger: a doctor run for one profile must not
        # sign its probe with another profile's key.
        key = scoped_secret(settings, settings.hindsight_api_key_env)
        return HindsightClient(base_url=settings.hindsight_url, bank_id=self.probe_bank,
                               api_key=key)

    def backend_ledger(self) -> Finding:
        """What the projection ledger says, which needs no socket to be true."""
        report = self.status.backend()
        return Finding("backend", _severity_for(report.state), report.detail,
                       "reconcile the failed submissions; the document map names them"
                       if report.state == DEGRADED else None,
                       {"by_state": report.evidence.get("by_state"),
                        "banks": report.evidence.get("banks"),
                        "superseded_epoch": report.evidence.get("superseded_epoch"),
                        "state": report.state})

    def route_outcomes(self) -> dict[str, dict[str, Any]]:
        """What the admission ledger says came back, per route. Empty without a gate."""
        gate = getattr(self.status, "gate", None)
        return gate.last_outcomes() if gate is not None else {}

    def inference_outcomes(self) -> Finding:
        """The newest settled dispatch on each route, and whether it succeeded.

        A capability census can only say that a route is served. This is the installation's
        own evidence of what a route did when it was asked for work, and a route that keeps
        failing is a different finding from a route that was never routed — they have
        different remedies, and only one of them is a model decision.
        """
        outcomes = self.route_outcomes()
        if not outcomes:
            return Finding("inference", OK,
                           "no dispatch has settled, so no route has been observed answering",
                           evidence={"observed": {}})
        observed = {name: {"outcome": str(row["outcome"]), "resource": row["resource"],
                           "state": row["state"], "settled_at": row["settled_at"]}
                    for name, row in sorted(outcomes.items())}
        stalled = sorted(name for name, item in observed.items()
                         if item["outcome"] != "succeeded")
        detail = ", ".join(f"{name} last answered {item['outcome']}"
                           for name, item in observed.items())
        if stalled:
            return Finding("inference", WARN,
                           f"{len(stalled)} route(s) did not answer their newest dispatch with "
                           f"a success: {detail}",
                           "a served route is not a working one: the model configured for it "
                           "is the part that answers, so check what runs that route before "
                           "enabling a stage that needs its answer",
                           {"stalled": stalled, "observed": observed})
        return Finding("inference", OK,
                       f"every route answered its newest dispatch: {detail}",
                       evidence={"observed": observed})

    def backend_connectivity(self) -> Finding:
        """An explicit request: is anything answering. Never run by default."""
        client = self.client()
        if client is None:
            return Finding("backend-connectivity", WARN,
                           "no backend is configured, so there is nothing to reach",
                           "configure a route first, or ignore this in capture-only mode")
        try:
            report = client.health()
        except Exception as error:
            return Finding("backend-connectivity", FAIL, f"unreachable: {error}",
                           "start the backend, then re-run with --synthetic-probe to "
                           "exercise the routes",
                           {"error": str(error)[:300]})
        # Liveness alone would let the report say "backend available" while the pinned
        # revision it depends on routes none of the work. What matters is the set.
        try:
            observed = client.negotiate(probe_routes=True).as_dict()
        except Exception as error:
            return Finding("backend-connectivity", WARN,
                           f"the backend answered but its capabilities are unconfirmed: "
                           f"{error}"[:400],
                           "re-run with the backend reachable; a capability the running "
                           "build does not route makes a stage refuse work rather than "
                           "fail at request time",
                           {"reported_version": report.get("version"),
                            "error": str(error)[:300]})
        missing = list(observed["unsupported"]) + list(observed["mismatches"])
        supported = set(observed["supported"])
        # "supported" is a claim about the routes a backend serves. Read alone it promises a
        # working answer, and this installation's own admission ledger has already seen ones
        # that do not work, so the two are reported together or the report contradicts the
        # `inference` check in the same document.
        outcomes = {name: str(row["outcome"]) for name, row in self.route_outcomes().items()
                    if name in supported}
        stalled = sorted(name for name, outcome in outcomes.items() if outcome != "succeeded")
        detail = "the backend answered"
        if missing:
            detail += f", with {len(missing)} pinned capability/capabilities it does not route"
        if stalled:
            detail += (", and " + ", ".join(f"{name} last answered {outcomes[name]}"
                                           for name in stalled)
                       + " on its newest dispatch")
        remedy = None
        if missing:
            remedy = ("the pinned version and the running build disagree; check the "
                      "backend's tag before enabling a stage that needs the missing route")
        if stalled:
            remedy = ((remedy + "; ") if remedy else "") + (
                "a served route is not a working one: the model behind it answers the "
                "dispatch, so look at what runs that route before enabling a stage that "
                "depends on its answer")
        return Finding("backend-connectivity", WARN if (missing or stalled) else OK,
                       detail, remedy,
                       {"reported_version": report.get("version"),
                        "observed_via": observed["observed_via"],
                        "pinned_to": observed["pinned_to"],
                        "supported": sorted(supported),
                        "not_routed": missing,
                        "last_observed": outcomes,
                        "observed_failing": stalled,
                        "observed_at": now()})

    def backend_synthetic_round_trip(self, *, reads: int = PROBE_SETTLE_READS,
                                     wait_s: float = PROBE_SETTLE_WAIT_S,
                                     sleeper: Callable[[float], None] = time.sleep) -> Finding:
        """One bounded, synthetic write and read, in a bank of its own.

        The text is a fixed sentence and the bank is named for the probe, so nothing private is
        sent and nothing of the owner's is overwritten. This is the only doctor action that
        talks to a model or stores anything outside the canonical record set, and it happens
        only when it was asked for by name.

        Three readings, because three parts can be broken and they need different remedies: the
        write was accepted, the backend derived a memory unit from it, and the search finds that
        unit when asked in its own words. The derivation is read back by document id rather than
        inferred from recall, because a recall that answers nothing looks exactly like one that
        was never searchable, and the two mean different things about a different component.
        """
        client = self.client()
        if client is None:
            return Finding("synthetic-probe", WARN, "no backend to probe",
                           "configure a route first")
        attempts = max(1, int(reads))
        waited = 0.0
        units = 0
        returned = 0
        failed = None
        left_behind = None
        try:
            client.retain(document_id=PROBE_DOCUMENT_ID, content=PROBE_TEXT,
                          metadata={"probe": "doctor"})
            for attempt in range(1, attempts + 1):
                state = client.document_state(PROBE_DOCUMENT_ID)
                units = int(state.get("count") or 0)
                if units or attempt == attempts:
                    break
                sleeper(wait_s)
                waited += wait_s
            if units:
                returned = len(client.recall(PROBE_QUERY, max_tokens=64).results)
        except Exception as error:
            failed = str(error)[:300]
        finally:
            # In a finally rather than on the success path: a probe that leaves a document
            # behind is a leak, and it is one whether the round trip worked or not.
            try:
                client.delete_document(PROBE_DOCUMENT_ID)
            except Exception as error:
                # The probe's own cleanup failing is a fact, not a detail: the synthetic
                # document stays in the probe bank, and a report that swallowed this would
                # leave a stranger's probe text sitting there with nobody to say so.
                left_behind = str(error)[:200]
        evidence = {"units": units, "returned": returned, "bank": self.probe_bank,
                    "reads": attempts, "waited_seconds": waited}
        if left_behind is not None:
            evidence["not_deleted"] = left_behind
        if failed is not None:
            evidence["error"] = failed
            return Finding("synthetic-probe", FAIL, f"the round trip failed: {failed}",
                           "check the backend's own logs; the framework wrote nothing "
                           "about this to the canonical store",
                           evidence)
        settling = f" ({waited:g}s spent waiting for it)" if waited else ""
        if not units:
            return Finding("synthetic-probe", FAIL,
                           f"the backend took the text and derived no memory unit from "
                           f"it{settling}",
                           "a retain that produces nothing is the model behind extraction, not "
                           "the index: ask which route the backend's retains are served by. "
                           "Nothing was derived, so recall has nothing to find",
                           evidence)
        if not returned:
            return Finding("synthetic-probe", FAIL,
                           f"the backend derived {units} unit(s) from the text and cannot "
                           f"find any of them{settling}",
                           "the write path answers and the search path does not, which points "
                           "at the embeddings model the bank is indexed with rather than at "
                           "the retain",
                           evidence)
        return Finding("synthetic-probe", OK,
                       "a synthetic document survived retain, derivation and recall"
                       + ("" if not waited else f", after {waited:g}s of waiting"),
                       evidence=evidence)


def _severity_for(state: str) -> str:
    """Map a status stage state onto a doctor severity, in one place.

    Only two stage states are inherently newsworthy: a degraded one is a fault, and a
    paused one is somebody's decision that an operator should know is in force.
    "Nothing is configured here" is not a finding on its own — a capture-only
    installation is a supported shape — so a check that genuinely should warn about an
    absent dependency says so itself rather than through this table.
    """
    if state == DEGRADED:
        return FAIL
    return WARN if state == PAUSED else OK


def unreachable_store_report(path: Any, reason: str) -> dict[str, Any]:
    """The report for a store that cannot be opened at all.

    It uses the same envelope as a full examination, so a reader never has to handle a
    second shape, and it carries one failure with the action that follows from it.
    """
    finding = Finding("layout", FAIL, f"{path}: {reason}",
                      "run `hermes-memory init` to create the canonical store")
    return {
        "checked_at": now(), "profile": "default", "severity": FAIL, "ok": False,
        "exit_code": 1,
        "probes": {"connectivity": False, "synthetic": False},
        "findings": [finding.as_dict()], "actions": [str(finding.remedy)],
    }


def _world_readable(path: Any) -> bool:
    try:
        return bool(os.stat(path).st_mode & 0o077)
    except OSError:
        return False


def _size_of(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
