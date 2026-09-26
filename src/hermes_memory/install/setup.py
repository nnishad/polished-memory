"""The setup transaction: eleven steps, each named before it runs and resumable after.

An installer that does everything at once is an installer that has to be right the first
time. So each step here answers three questions on its own — what it looked at, what it
would change, and whether anything stops it — and a step is only re-done when the inputs
it read have actually moved. An interrupted run resumes; it does not start over,
re-issue a credential, or write the same service file twice.

Two rules hold across all of them. Nothing reaches outside this machine unless the
caller handed in an executor, and the host commands are built here from fixed arguments
rather than parsed out of anybody's string. And nothing the owner changed is overwritten
in passing: a conflict is reported with its path and digest, not silently resolved.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import util as _import_util
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import DEFAULT_ENV_FILENAME, env_file_values
from ..ids import content_digest, digest, now
from .inventory import conflicts, provider_selection, survey
from .profiles import InstallationError, ProfileRegistry, STATE_FILENAME
from .services import plan as service_plan
from .uninstall import record_provider_selection

__all__ = ["STEPS", "plan", "run", "SetupError", "CANARY_SOURCE"]


class SetupError(ValueError):
    """The transaction cannot proceed, and says which step and why."""


# The order is the plan's §10.4. Steps 1 and 2 read, 3 to 9 write, 10 proves, 11 reports.
STEPS: tuple[str, ...] = ("inventory", "plan", "stage", "configure", "initialize",
                          "register-plugin", "preflight", "activate", "services",
                          "canary", "finish")

PLAN_VERSION = "setup-plan-v1"
INPUT_VERSION = "setup-step-v1"
_HEX = re.compile(r"[0-9a-f]{40}")
CANARY_SOURCE = "setup-canary"
# A fixed moment: the canary's identity is the same record every time, so a rerun of a
# completed step is recognised as a rerun instead of a new approval.
CANARY_MOMENT = "2000-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class Context:
    """Everything a step is allowed to look at."""

    settings: Any
    hermes_home: Path
    profile: str
    actor: str
    registry: ProfileRegistry
    runner: Callable[[Sequence[str]], tuple[int, str]] | None = None
    ref: str | None = None
    environ: dict[str, str] | None = None
    start_services: bool = False

    def profile_settings(self):
        return self.registry.profile(self.profile).scoped(self.settings)

    def command(self, argv: Sequence[str]) -> tuple[int, str]:
        if self.runner is None:
            raise SetupError("this step runs a host command and no executor was provided: "
                             + " ".join(str(item) for item in argv))
        return self.runner([str(item) for item in argv])


def plan(settings, *, hermes_home: str | Path, profile: str | None = None,
         registry: ProfileRegistry | None = None, ref: str | None = None,
         environ: dict[str, str] | None = None,
         runner: Callable | None = None, actor: str | None = None,
         start_services: bool = False) -> dict[str, Any]:
    """What each of the eleven steps would do, and which of them is already done."""
    ctx = _context(settings, hermes_home=hermes_home, profile=profile, registry=registry,
                   ref=ref, environ=environ, runner=runner, actor=actor,
                   start_services=start_services)
    entries = []
    try:
        for step in STEPS:
            result = _STEPS[step](ctx, apply=False)
            stamp = digest([INPUT_VERSION, step, result.get("inputs", {})])
            entries.append({"step": step, "actions": list(result["actions"]),
                            "blocking": list(result.get("blocking", ())),
                            "advisory": list(result.get("advisory", ())),
                            "inputs_digest": stamp,
                            "state": _state(ctx, step, result, stamp)})
    finally:
        if registry is None:
            ctx.registry.db.close()
    blocked = [f"{item['step']}: {line}" for item in entries for line in item["blocking"]]
    proposal = {"profile": ctx.profile, "hermes_home": str(ctx.hermes_home),
                "steps": [{"step": item["step"], "state": item["state"],
                           "inputs_digest": item["inputs_digest"],
                           "actions": item["actions"]} for item in entries]}
    return {"profile": ctx.profile, "hermes_home": str(ctx.hermes_home),
            "actor": ctx.actor, "steps": entries,
            "todo": [item["step"] for item in entries if item["state"] == "pending"],
            "resumed": [item["step"] for item in entries if item["state"] == "resumed"],
            "blocked": blocked, "host_commands": host_commands(ctx),
            "review_digest": digest([PLAN_VERSION, proposal])}


def run(settings, *, hermes_home: str | Path, actor: str, review: str,
        profile: str | None = None, ref: str | None = None,
        environ: dict[str, str] | None = None,
        runner: Callable | None = None, start_services: bool = False) -> dict[str, Any]:
    """Do the plan that was shown, from the first step that is not already finished.

    The digest has to match the plan computed now, so an approval cannot be spent on a
    transaction whose shape changed underneath it. A step that blocks stops the run and
    keeps its place, so the next invocation starts from the same question.
    """
    if not isinstance(actor, str) or not actor.strip():
        # Asked before anything is read: a plan computed for an unnamed approver is a
        # plan nobody can approve, and reading the machine to find that out would have
        # been the expensive way to say it.
        raise SetupError("an actor must be named: setup is the owner's decision")
    proposal = plan(settings, hermes_home=hermes_home, profile=profile, ref=ref,
                    environ=environ, runner=runner, actor=actor,
                    start_services=start_services)
    if actor.strip() != settings.owner_principal:
        raise SetupError(
            f"setup may only be approved by the owner principal the configuration names "
            f"({settings.owner_principal!r}); {actor!r} is not that person")
    if review != proposal["review_digest"]:
        raise SetupError(
            "the review digest does not match what setup would do now. Run "
            "`hermes-memory setup` again and approve the plan it prints.")
    ctx = _context(settings, hermes_home=hermes_home, profile=profile, ref=ref,
                   environ=environ, runner=runner, actor=actor,
                   start_services=start_services,
                   registry=ProfileRegistry.open(settings))
    done: list[str] = []
    receipts: list[dict[str, Any]] = []
    try:
        for entry in proposal["steps"]:
            if entry["state"] == "resumed":
                continue
            result = _STEPS[entry["step"]](ctx, apply=True)
            if result.get("blocking"):
                raise SetupError(f"{entry['step']}: " + "; ".join(result["blocking"]))
            _record(ctx, entry["step"], result, review=review)
            done.append(entry["step"])
            receipts.append({"step": entry["step"], "actions": list(result["actions"])})
        report = {"ok": True, "profile": ctx.profile, "done": done,
                  "resumed": proposal["resumed"], "receipts": receipts,
                  "review_digest": review, "readiness": readiness(ctx),
                  "actor": ctx.actor}
    finally:
        ctx.registry.db.close()
    return report


# -- the steps ---------------------------------------------------------------

def _inventory(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """Read the machine. This step writes nothing and asks no model anything."""
    report = survey(ctx.settings, hermes_home=ctx.hermes_home, environ=ctx.environ)
    said = conflicts(report)
    blocking = [line for line in said if _collides(line)]
    return {
        "actions": [f"{len(report['endpoints']['listening_ports'])} port(s) listening, "
                    f"{len(report['capture_owners']['canonical_stores'])} canonical "
                    f"store(s) beside the homes searched, provider reads as "
                    f"{report['host']['memory_provider']!r}"],
        "advisory": [line for line in said if line not in blocking],
        "blocking": blocking,
        "inputs": {"ports": report["endpoints"]["listening_ports"],
                   "wanted": report["endpoints"]["wanted"],
                   "stores": report["capture_owners"]["canonical_stores"],
                   "spools": report["capture_owners"]["capture_spools"],
                   "provider": report["host"]["memory_provider"],
                   # A byte count of free space is not an input to any decision here:
                   # it moves between two calls of the same command, and a plan that
                   # changed digest every second would make an approval meaningless.
                   "disk_readable": report["disk"]["free_bytes"] is not None},
    }


def _collides(line: str) -> bool:
    """A collision with something real stops the run; a messiness in the file does not.

    A held port and a second capture owner mean two installations want one thing. A
    loose permission or a credential at rest is the owner's to fix and does not make
    this installation wrong, so it is said out loud and does not stop anything.
    """
    return any(marker in line for marker in ("already listening", "capture owner",
                                             "no hermes executable", "no owner principal"))


def _plan_step(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """The two diffs this transaction is made of: the profile map and the service units."""
    proposal = ctx.registry.plan(ctx.profile, ctx.hermes_home)
    units = service_plan(ctx.settings)
    actions = [f"enroll {ctx.profile}: data dir {proposal['data_dir']}, bank "
               f"{proposal['bank_id']}, credential scope {proposal['credential_scope']}"]
    actions += [f"{entry['unit']}: {entry['state']}" for entry in units["units"]
                if entry["state"] not in ("unchanged", "not-wanted")]
    blocking = [f"{unit} already exists and was not written by this installation"
                for unit in units["blocked"]]
    return {"actions": actions, "blocking": blocking,
            "inputs": {"enroll": {key: proposal[key] for key in
                                  ("profile", "hermes_home", "data_dir", "bank_id",
                                   "credential_scope")},
                       "services": [entry["state"] for entry in units["units"]]}}


def _stage(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """Prove the release is laid out and importable. Nothing here installs anything.

    §10.3 makes the environments a packaging artefact; this is the check that one
    arrived. Fetching a package on the way past would answer a question nobody asked, so
    a missing release is reported with the path it was expected at.
    """
    release = Path((ctx.environ or {}).get("HERMES_MEMORY_RELEASE")
                   or Path(ctx.settings.home) / "runtime" / "current")
    backend_configured = bool(ctx.settings.hindsight_url)
    wanted = [release / "bin" / "hermes-memory"]
    if backend_configured:
        wanted.append(release / "hindsight" / "bin" / "hindsight-api")
    missing = [str(path) for path in wanted if not path.is_file()]
    importable = {"hermes_memory": _import_util.find_spec("hermes_memory") is not None,
                  "hindsight_client": _import_util.find_spec("hindsight_client") is not None}
    actions = [f"release {release}: " + ", ".join(
        f"{name} {'importable' if found else 'absent'}"
        for name, found in sorted(importable.items()))]
    blocking = [f"{path} is not staged; unpack the pinned release there or point "
                "HERMES_MEMORY_RELEASE at it" for path in missing]
    if backend_configured and not importable["hindsight_client"]:
        blocking.append("a backend route is configured and hindsight_client is not "
                        "importable; install the backend extra rather than the metapackage")
    return {"actions": actions, "blocking": blocking,
            "inputs": {"release": str(release), "missing": missing,
                       "importable": importable, "backend": backend_configured}}


def _configure(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """Enroll this profile and create its own directory: once, and never in passing.

    The map is written here because it is the step the rest of the transaction reads
    paths from, and it is a reviewed decision rather than a side effect - the ledger
    records who approved which mapping, and an existing one is not quietly redone.
    """
    proposal = ctx.registry.plan(ctx.profile, ctx.hermes_home)
    data_dir = Path(proposal["data_dir"])
    env_file = ctx.hermes_home / DEFAULT_ENV_FILENAME
    blocking: list[str] = []
    if env_file.is_file():
        at_rest = sorted(key for key in env_file_values(env_file)
                         if key.endswith("_API_KEY"))
        if at_rest:
            blocking.append(f"{env_file} holds a credential value ({', '.join(at_rest)}); "
                            "it should name the variable that holds it")
        state = "already written by the provider's own setup"
    else:
        state = "absent; the provider writes it when Hermes selects this memory"
    occupied = _other_owner(ctx)
    if occupied:
        blocking.append(occupied)
    enrolled = _enrolled(ctx)
    actions = [f"profile {ctx.profile} -> bank {proposal['bank_id']}, scope "
               f"{proposal['credential_scope']}"
               + (" (already enrolled exactly so)" if enrolled else " (would be enrolled)"),
               f"data dir {data_dir} " + ("exists" if data_dir.is_dir() else "would be created"),
               f"{env_file}: {state}"]
    if apply and not blocking:
        if not enrolled:
            ctx.registry.enroll(ctx.profile, ctx.hermes_home, actor=ctx.actor,
                                review_digest=proposal["review_digest"])
        data_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    return {"actions": actions, "blocking": blocking,
            "inputs": {"env_file": _digest_of(env_file), "data_dir": str(data_dir),
                       "mapping": {key: proposal[key] for key in
                                   ("hermes_home", "bank_id", "credential_scope")}}}


def _other_owner(ctx: Context) -> str | None:
    """Another profile already answers for this Hermes home, if one does.

    Two capture owners for one conversation is the failure the ledger exists to prevent,
    and a UNIQUE index would only say so after the transaction had already written a
    service and migrated a store. The home is compared resolved, because that is how the
    ledger stores it.
    """
    row = ctx.registry.db.execute(
        "SELECT profile, state FROM profiles WHERE hermes_home=?",
        (str(Path(ctx.hermes_home).expanduser().resolve()),)).fetchone()
    if row is None or row["profile"] == ctx.profile:
        return None
    return (f"{ctx.hermes_home} is already served by profile {row['profile']!r} "
            f"({row['state']}); one home has one capture owner")


def _store_path(ctx: Context) -> Path | None:
    """Where this profile's canonical store will be, or None while it is not enrolled.

    Planning runs the whole step list before anything is enrolled, so a step that reads
    the profile's paths has to say "this waits for configure" rather than raise on a
    lookup that is only missing because the transaction has not reached it yet.
    """
    from .profiles import InstallationError

    try:
        return Path(ctx.profile_settings().db_path)
    except InstallationError:
        return None


def _enrolled(ctx: Context) -> bool:
    from .profiles import InstallationError

    try:
        return ctx.registry.profile(ctx.profile).state == "enrolled"
    except InstallationError:
        return False


def _initialize(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """Migrate the profile's own empty store. Never a shared one, and never a reset."""
    path = _store_path(ctx)
    if path is None:
        return {"actions": ["no profile is enrolled yet, so there is no store to create"],
                "blocking": ["the configure step must enroll this profile first"],
                "inputs": {"store": "no profile enrolled"}}
    if path.exists():
        return {"actions": [f"{path} already exists and is left exactly as it is"],
                "inputs": {"store": str(path)}}
    actions = [f"would create and migrate {path}"]
    if apply:
        from ..storage.evidence import EvidenceStore

        scoped = ctx.profile_settings()
        # The store and the blob directory beside it are one piece of layout: a
        # canonical database with nowhere to put an attachment is a doctor failure
        # waiting for the first image.
        Path(scoped.blob_dir).mkdir(parents=True, mode=0o700, exist_ok=True)
        with EvidenceStore(path):
            pass
    return {"actions": actions, "inputs": {"store": str(path)}}


def _register_plugin(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """The host's own install path, with the reviewed commit pinned.

    The arguments are built here rather than accepted from anywhere: the one thing this
    step must not do is hand a shell string to a process that can write into the host's
    configuration, and an unpinned ref would be a moving pointer wearing a release's
    clothes.
    """
    commands = [argv for argv, name in host_commands(ctx) if name != "activate"]
    actions = [" ".join(str(part) for part in argv) for argv in commands]
    actions.append(f"plugin files at the source tree digest {_tree_digest()}")
    blocking = []
    if not (ctx.ref and _HEX.fullmatch(ctx.ref)):
        blocking.append("--ref must be the 40-character commit this release was cut at; "
                        "setup will not install a moving pointer")
    if ctx.runner is None:
        blocking.append("no host command executor was provided, so the plugin cannot be "
                        "registered from here")
    if apply and not blocking:
        for argv in commands:
            code, output = ctx.command(argv)
            if code != 0:
                raise SetupError(
                    f"`{' '.join(str(part) for part in argv)}` failed ({code}): "
                    f"{output.strip()[:300]}")
    return {"actions": actions, "blocking": blocking,
            "inputs": {"ref": ctx.ref or "unset", "tree": _tree_digest(),
                       "commands": actions[:len(commands)],
                       "executor": ctx.runner is not None}}


def _preflight(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """Ask the installation about itself. No model, no network, no private ingestion."""
    from ..operations.doctor import Doctor
    from ..storage.evidence import EvidenceStore

    path = _store_path(ctx)
    if path is None or not path.is_file():
        return {"actions": ["the canonical store is not initialised yet, so preflight has "
                            "nothing to read"],
                "blocking": ["the canonical store must exist before preflight can examine it"],
                "inputs": {"store": str(path) if path else "no profile enrolled"}}
    with EvidenceStore(path) as store:
        report = Doctor(store, settings=ctx.profile_settings()).examine()
    findings = report["findings"]
    return {"actions": [f"doctor: {report['severity']}, {len(findings)} finding(s)"],
            "blocking": [item["detail"] for item in findings if item["severity"] == "fail"],
            "advisory": [item["detail"] for item in findings if item["severity"] == "warn"],
            "inputs": {"severity": report["severity"],
                       "findings": sorted(f"{item['check']}={item['severity']}"
                                          for item in findings)}}


def _activate(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """Select this provider in the host's configuration, and prove nothing else moved.

    Only the memory provider is ours to set. Everything else in that file belongs to the
    owner, so the model block is digested before and after and a change there is a
    failure with the snapshot to restore from - not a rollback this module performs on a
    file it does not own.
    """
    config = ctx.hermes_home / "config.yaml"
    argv = [cmd for cmd, name in host_commands(ctx) if name == "activate"]
    if not config.is_file():
        return {"actions": [f"{config} does not exist, so there is nothing to activate"],
                "blocking": [f"no host configuration at {config}"],
                "inputs": {"config": "absent"}}
    before = _model_block(config)
    prior = _provider(config)
    if prior == "hermes-memory":
        # Said out loud rather than performed again: re-running a completed activation
        # would take a second snapshot of a file nothing changed and ask the host to
        # write the value it already holds. The inputs are the same facts the activation
        # itself records, so a rerun after a finished setup resumes this step instead of
        # finding work for it.
        return {"actions": [f"memory.provider already reads {prior!r}, so the host is "
                            "already selecting this memory"],
                "inputs": {"config": _digest_of(config), "model": before,
                           "provider": prior, "executor": True}}
    actions = [" ".join(str(part) for part in argv[0]),
               f"model block digest {before[:12]}",
               f"memory.provider currently reads {prior!r}"]
    if ctx.runner is None:
        return {"actions": actions + ["not run: no host command executor was provided"],
                "blocking": ["no host command executor was provided, so the provider "
                             "cannot be selected from here"],
                "inputs": {"config": _digest_of(config), "model": before,
                           "executor": False}}
    if apply:
        snapshot = Path(ctx.settings.home) / f"config.yaml.before-{_digest_of(config)[:12]}"
        snapshot.write_bytes(config.read_bytes())
        snapshot.chmod(0o600)
        # Recorded before the write, because afterwards the file answers a different
        # question: §10.7 restores the prior selection only if it can say what it was.
        receipt = record_provider_selection(hermes_home=ctx.hermes_home,
                                            settings=ctx.settings, prior=prior,
                                            actor=ctx.actor)
        code, output = ctx.command(argv[0])
        if code != 0:
            raise SetupError(f"`{' '.join(str(part) for part in argv[0])}` failed "
                             f"({code}): {output.strip()[:300]}")
        actions.insert(0, f"previous file kept at {snapshot}")
        if _model_block(config) != before:
            raise SetupError(
                "the host writer changed the model block, which is not ours to touch; "
                f"restore {snapshot} ({before[:12]} -> {_model_block(config)[:12]})")
        if _provider(config) != "hermes-memory":
            raise SetupError(f"memory.provider still reads {_provider(config)!r} after "
                             "activation; the host did not take the change")
        actions.append(f"prior selection {prior!r} recorded at {receipt['path']}")
    return {"actions": actions,
            "inputs": {"config": _digest_of(config), "model": before,
                       # The value the step leaves behind, not the one it found: the
                       # receipt has to say "this is finished", and the only way a rerun
                       # can recognise its own result is to compare against the state the
                       # result produced.
                       "provider": _provider(config), "executor": True}}


def _services(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """Install the owned units, reload only if one changed, and start only if asked."""
    from .services import Services, apply as write_units

    proposal = service_plan(ctx.settings)
    actions = [f"{entry['unit']}: {entry['state']}" for entry in proposal["units"]]
    if proposal["reload_needed"]:
        actions.append("systemctl --user daemon-reload")
    if ctx.start_services:
        actions.append("systemctl --user start " + " ".join(proposal["start_order"]))
    blocking = [f"{unit} already exists and was not written by this installation"
                for unit in proposal["blocked"]]
    if (proposal["reload_needed"] or ctx.start_services) and ctx.runner is None:
        # Said at plan time, because "we will run systemctl" is a promise the
        # transaction can only keep with an executor, and a plan must not show a step
        # it has no way to finish.
        blocking.append("the units or the manager need telling and no host command "
                        "executor was provided")
    if apply and not blocking:
        # A start is an action against a running manager, so it needs the executor the
        # rest of the transaction uses. Without one the units are still written - they
        # are files - but nothing is asked of the machine, and the plan says which is
        # which rather than letting a quiet failure look like a started service.
        if proposal["would_change"]:
            receipt = write_units(ctx.settings, actor=ctx.actor,
                                  review=proposal["review_digest"])
            actions.append(f"wrote {', '.join(receipt['units_written']) or 'nothing'}")
            if receipt["reload_needed"]:
                Services(ctx.settings, runner=ctx.runner).daemon_reload()
        if ctx.start_services:
            Services(ctx.settings, runner=ctx.runner).start()
    return {"actions": actions, "blocking": blocking,
            "inputs": {"states": [entry["state"] for entry in proposal["units"]],
                       "reload": proposal["reload_needed"],
                       "start": ctx.start_services}}



def _canary(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """Prove the loop closes on a synthetic message that outlives nothing.

    Capture, recall and erasure are each checked against the same record, and the record
    is then forgotten - so a green canary says the fences work rather than that the
    tables exist. No model is asked anything, which is the point of doing this during
    installation rather than after it.
    """
    from ..lifecycle.erasure import ErasureManager
    from ..storage.evidence import EvidenceStore

    path = _store_path(ctx)
    envelope = {
        "source": CANARY_SOURCE, "source_id": "canary-1", "revision": "1",
        "kind": "note",
        "text": "Synthetic installation canary. This line is not anybody's memory.",
        "observed_at": CANARY_MOMENT, "occurred_at": CANARY_MOMENT,
        "occurred_precision": "second",
        "metadata": {"synthetic": True, "profile": ctx.profile},
    }
    inputs = {"store": str(path) if path else "no profile enrolled",
              "envelope": digest(envelope)}
    if path is None or not path.is_file():
        return {"actions": ["the store is not initialised, so no canary can be written"],
                "blocking": ["initialise the canonical store before the canary step"],
                "inputs": inputs}
    if not apply:
        return {"actions": [f"would write, retrieve and forget one synthetic record in {path}"],
                "inputs": inputs}
    with EvidenceStore(path) as store:
        committed = store.commit(envelope)
        record = store.get(committed["id"])
        found = [item.id for item in store.search("synthetic canary")]
        manager = ErasureManager(store, owner_principal=ctx.actor)
        preview = manager.preview(record_ids=[committed["id"]], actor=ctx.actor,
                                 reason="installation canary")
        manager.confirm(intent_id=preview["intent_id"],
                        preview_digest=preview["preview_digest"], actor=ctx.actor)
        gone = manager.forgotten(committed["id"])
        visible = store.live_and_visible(committed["id"])
    if record is None or committed["id"] not in found:
        raise SetupError("the canary was written but could not be retrieved")
    if not gone or visible:
        raise SetupError("the canary outlived its own erasure, so the fences do not hold")
    return {"actions": [f"canary {committed['id'][:12]}: written, retrieved by search, "
                        "erased, and verified gone"],
            "inputs": inputs}


def _finish(ctx: Context, *, apply: bool) -> dict[str, Any]:
    """The last step is an account of the installation, not another action."""
    ready = readiness(ctx)
    return {"actions": [", ".join(f"{name}={state}" for name, state in sorted(ready.items()))],
            "inputs": {"readiness": ready}}


def readiness(ctx: Context) -> dict[str, str]:
    """Per component, in its own words. "Configured" is never written as "operational"."""
    store = _store_path(ctx)
    present = bool(store) and store.is_file()
    return {
        "profile": "enrolled" if _enrolled(ctx) else "not enrolled",
        "store": "present" if present else "absent",
        "capture": "ready" if present else "unconfigured",
        "inference": "configured" if ctx.settings.inference_enabled else "off",
        "delivery": "configured" if ctx.settings.delivery_enabled else "off",
        "units": "ledger present" if (Path(ctx.settings.home) / STATE_FILENAME).exists()
                 else "no ledger yet",
        "real_sources": "disabled until the owner enables one",
    }


_STEPS: dict[str, Callable] = {
    "inventory": _inventory, "plan": _plan_step, "stage": _stage,
    "configure": _configure, "initialize": _initialize,
    "register-plugin": _register_plugin, "preflight": _preflight, "activate": _activate,
    "services": _services, "canary": _canary, "finish": _finish,
}


# -- the machinery around the steps ------------------------------------------

def _context(settings, *, hermes_home, profile=None, registry=None, ref=None,
             environ=None, runner=None, actor=None, start_services=False) -> Context:
    home = Path(hermes_home).expanduser()
    if not home.is_absolute():
        raise SetupError("the Hermes home must be an absolute path")
    owner = (actor or settings.owner_principal or "").strip()
    if not owner:
        raise SetupError(
            "no owner principal is configured, so there is nobody whose approval setup "
            "could act on (set HERMES_MEMORY_OWNER_PRINCIPAL)")
    return Context(settings=settings, hermes_home=home,
                   profile=profile or profile_name(home), actor=owner,
                   registry=registry or ProfileRegistry.reading(settings),
                   runner=runner, ref=ref, environ=dict(environ or {}),
                   start_services=start_services)


def profile_name(home: Path) -> str:
    """The home's own directory name, or ``default`` for a bare account home."""
    name = Path(home).name.strip().lower()
    return name if name and name not in {"home", ".hermes", "hermes"} else "default"


def host_commands(ctx: Context) -> list[tuple[list[str], str]]:
    """Every host command this transaction can run, so a reviewer sees them together."""
    root = _source_root()
    return [
        (["hermes", "plugins", "install",
          f"file://{root}#integrations/hermes-memory", "--ref", ctx.ref or "",
          "--no-enable"], "install"),
        (["hermes", "plugins", "enable", "hermes-memory",
          "--no-allow-tool-override"], "enable"),
        (["hermes", "config", "set", "memory.provider", "hermes-memory"], "activate"),
    ]


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tree_digest() -> str:
    root = _source_root()
    files = sorted((root / "integrations" / "hermes-memory").glob("*.py"))
    return digest([[path.name, content_digest(path.read_bytes())] for path in files])


def _provider(config: Path) -> str:
    return provider_selection(config) if Path(config).is_file() else "no config.yaml to read"


def _model_block(config: Path) -> str:
    """A digest of the ``model:`` block, so a change there is visible and not interpreted.

    One dotted key is ours to set; the only honest check on the rest is that what the
    owner wrote is unchanged, and a digest of the block says that without pretending to
    parse a file belonging to another program.
    """
    try:
        lines = Path(config).read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unreadable"
    collected: list[str] = []
    inside = False
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, separator, _ = line.strip().partition(":")
        if indent == 0:
            inside = bool(separator and key == "model")
            if inside:
                collected.append(line)
            continue
        if inside:
            collected.append(line)
    return digest(collected)


def _digest_of(path: Path) -> str:
    try:
        return content_digest(Path(path).read_bytes())
    except OSError:
        return "absent"


def _state(ctx: Context, step: str, result: dict[str, Any], stamp: str) -> str:
    """Resumed when the inputs have not moved, blocked when something collides, else to do.

    The receipt is matched on the inputs rather than on the step name alone: a step that
    would now do something different is not a step that has already been done, and
    pretending otherwise is how an interrupted run ends up half-applied twice.
    """
    if result.get("blocking"):
        return "blocked"
    row = ctx.registry.db.execute(
        "SELECT inputs_digest FROM setup_steps WHERE profile=? AND step=?",
        (ctx.profile, step)).fetchone()
    return "resumed" if row and row[0] == stamp else "pending"


def _record(ctx: Context, step: str, result: dict[str, Any], *, review: str) -> None:
    stamp = digest([INPUT_VERSION, step, result.get("inputs", {})])
    ctx.registry.db.execute(
        "INSERT INTO setup_steps(profile, step, inputs_digest, actions, actor, "
        "review_digest, at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(profile, step) DO UPDATE SET "
        "inputs_digest=excluded.inputs_digest, actions=excluded.actions, "
        "actor=excluded.actor, review_digest=excluded.review_digest, at=excluded.at",
        (ctx.profile, step, stamp, json.dumps(list(result["actions"])), ctx.actor,
         review, now()))
