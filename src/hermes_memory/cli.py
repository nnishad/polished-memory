"""Operator CLI.

Readiness is reported per stage; 'configured' is never reported as 'operational'. Every
reading here opens the canonical store in a mode that cannot change it, so running the
tool against a live installation is not an action.
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import SettingError, load_settings
from .install.services import Services, subprocess_runner
from .storage.evidence import EvidenceError, EvidenceStore, ReadOnlyStore

__all__ = ["main"]


def _configuration(settings) -> dict[str, Any]:
    return {
        "data_dir": str(settings.data_dir),
        "database": str(settings.db_path),
        "database_present": settings.db_path.exists(),
        "inference_enabled": settings.inference_enabled,
        "capture_only": settings.capture_only,
        "hindsight_route": settings.hindsight_url or "unset",
        "approved_inference_hosts": sorted(settings.allowed_inference_hosts),
        "background_budget_tokens": settings.background_budget_tokens,
        "foreground_deadline_s": settings.foreground_deadline_s,
        "owner_principal": settings.owner_principal or "unset",
        # Deliberately no "is the engine's SDK importable here": this framework's bridge is
        # HTTP and imports nothing from the backend, so that fact said nothing about whether
        # capture or recall were degraded, and a missing package in the gate's own venv is a
        # correct installation rather than a diminished one.
    }


def _capabilities(settings, store_present: bool) -> dict[str, bool]:
    """Which jobs this installation can do, said one at a time.

    A store that only captures is working, and one whose formation is switched off is
    not broken; a single readiness flag made both of those look like failures.
    """
    return {
        "capture": store_present,
        "local_recall": store_present,
        "formation": store_present and not settings.capture_only,
        # Nothing in this installation drains the queue on its own. Formation is a
        # bounded pass an operator runs (`hermes-memory form`), so a store that *can*
        # form observations still forms none until somebody asks and approves the list.
        # §8.2 keeps the always-on worker behind that decision rather than pretending it
        # is already running.
        "formation_unattended": False,
        "forgetting": store_present and bool(settings.owner_principal),
        # Delivery is not a capability an installation has merely because somebody is
        # named: it needs an owner, an explicit switch and one concrete destination.
        "delivery": bool(store_present and settings.owner_principal
                         and settings.delivery_enabled and settings.delivery_target),
    }


def main(argv: list[str] | None = None) -> int:
    from .processing.maintenance import SECTIONS

    parser = argparse.ArgumentParser(prog="hermes-memory")
    parser.add_argument("--env-file",
                        help="owned env file (default: $HERMES_MEMORY_HOME/hermes-memory.env)")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status",
                            help="print configuration and every stage's own account")
    status.add_argument("--hermes-home",
                        help="report the memory enrolled for this Hermes profile home")
    init = sub.add_parser("init", help="create and migrate the canonical store")
    init.add_argument("--dry-run", action="store_true")
    init.add_argument("--hermes-home",
                      help="create the store of the profile enrolled for this Hermes home; "
                           "without it, this installation's own")
    doctor = sub.add_parser("doctor", help="read-only checks; performs no inference")
    doctor.add_argument("--hermes-home",
                        help="examine the memory enrolled for this Hermes profile home")
    doctor.add_argument("--probe", action="store_true",
                        help="ask the configured backend whether it is answering")
    doctor.add_argument("--synthetic-probe", action="store_true",
                        help="run one bounded, synthetic retain and recall round trip")

    compatibility = sub.add_parser(
        "compatibility",
        help="compare this build against the manifest that states what it is compatible with")
    compatibility.add_argument("--write", action="store_true",
                               help="regenerate the manifest from this tree; a packaging step, "
                                    "not something a running installation does")
    compatibility.add_argument("--digests", action="store_true",
                               help="also compare the digests that name which code this is, "
                                    "which is what a release build has to do")

    setup = sub.add_parser("setup",
                           help="the installation transaction; run without --review to "
                                "see every step first")
    setup.add_argument("--hermes-home", required=True,
                       help="the Hermes profile home to set this memory up for")
    setup.add_argument("--profile", help="short name; defaults to the home's directory")
    setup.add_argument("--ref", help="the 40-character release commit to pin the plugin to")
    setup.add_argument("--actor")
    setup.add_argument("--review", metavar="DIGEST",
                       help="the digest of the plan that was actually shown")
    setup.add_argument("--start", action="store_true",
                       help="start the owned services once the units are installed")

    profiles = sub.add_parser("profiles",
                              help="list the Hermes profiles this installation serves, and "
                                   "what has been decided about them")
    profiles.add_argument("--profile", help="report the decisions about one profile only")
    profiles.add_argument("--review",
                          help="the review digest of a decision already approved, to ask "
                               "what it actually did")
    inventory = sub.add_parser("inventory",
                               help="read what is already here; opens no socket and runs "
                                    "no model")
    inventory.add_argument("--hermes-home",
                           help="the Hermes profile home to report the host facts of")
    inventory.add_argument("--conflicts", action="store_true",
                           help="print only the reasons a setup run would be a bad idea")
    enroll = sub.add_parser("enroll",
                            help="map one Hermes profile to its own memory; run without "
                                 "--review to see what would change")
    enroll.add_argument("--hermes-home", required=True,
                        help="the profile home Hermes passes to an activity")
    enroll.add_argument("--profile", help="short name; defaults to the home's directory")
    enroll.add_argument("--actor", help="the owner principal approving this")
    enroll.add_argument("--review", metavar="DIGEST",
                        help="the digest of the plan that was actually shown")
    retire = sub.add_parser("retire",
                            help="unlink a profile; its evidence stays exactly where it is")
    retire.add_argument("--profile", required=True)
    retire.add_argument("--actor")
    retire.add_argument("--reason", required=True)

    audit = sub.add_parser("audit", help="read the ledger of what has already happened")
    audit.add_argument("--action")
    audit.add_argument("--object", dest="object_id")
    audit.add_argument("--actor")
    audit.add_argument("--source", help="one connector's own history instead of the ledger")
    audit.add_argument("--actions", action="store_true",
                       help="what kinds of thing have happened here, and how often")
    audit.add_argument("--actors", action="store_true",
                       help="who has been acting, counted apart from what they did")
    audit.add_argument("--decisions", metavar="CATEGORY",
                       choices=("identity", "identity-decisions", "goals", "lessons",
                                "erasure"),
                       help="owner-only decisions in one category, counted by decider")
    audit.add_argument("--timeline", metavar="RECORD",
                       help="everything this store can say about one record")
    audit.add_argument("--include-private", action="store_true",
                       help="quote the record's own text, redacted, in a timeline")
    audit.add_argument("--limit", type=int, default=50)
    audit.add_argument("--hermes-home", help="read the memory enrolled for this Hermes profile home; without it a single profile is assumed and more than one is refused")

    explain = sub.add_parser("explain", help="why an item came back, or why nothing did")
    target = explain.add_mutually_exclusive_group(required=True)
    target.add_argument("--record")
    target.add_argument("--artifact")
    target.add_argument("--goal")
    target.add_argument("--lesson",
                        help="every version of a habit, what it cites and what was scored "
                             "against it")
    target.add_argument("--summary",
                        help="why a reading of a scope is on the table, or why it is withheld")
    target.add_argument("--not-told", metavar="TOPIC",
                        help="list the reasons this topic did not interrupt the owner")
    explain.add_argument("--at", help="the moment to judge quiet hours against")
    explain.add_argument("--include-private", action="store_true",
                         help="include payload and goal text, still redacted")
    explain.add_argument("--limit", type=int, default=20)
    explain.add_argument("--hermes-home", help="read the memory enrolled for this Hermes profile home; without it a single profile is assumed and more than one is refused")

    measure = sub.add_parser("measure",
                             help="read a typed measurement series the store holds; no model "
                                  "is consulted")
    measure.add_argument("--what", help="the measure to read, as the source named it")
    measure.add_argument("--list", dest="listing", action="store_true",
                        help="what can be asked: every measure, device and unit the store "
                             "holds samples for, with its span")
    measure.add_argument("--device", help="restrict the reading to one device or sensor")
    measure.add_argument("--source", help="restrict the reading to one registered source")
    measure.add_argument("--unit", help="the unit to read in; required when the stored "
                                        "samples disagree")
    measure.add_argument("--since", help="inclusive lower bound, with a timezone")
    measure.add_argument("--until", help="inclusive upper bound, with a timezone")
    measure.add_argument("--hermes-home",
                         help="read the memory enrolled for this Hermes profile home")

    sub.add_parser("serve", help="run the admission endpoint the runtime unit expects")
    services = sub.add_parser("services",
                              help="show the owned user units; --install writes the plan "
                                   "that was shown")
    services.add_argument("--install", metavar="DIGEST",
                          help="the digest of the plan to write")
    services.add_argument("--autostart", choices=("enable", "disable"),
                          help="decide whether these units start at login; a separate "
                               "decision from starting them now")
    services.add_argument("--actor")
    sub.add_parser("start", help="start the owned services, gate before backend before worker")
    stop = sub.add_parser("stop", help="persist the pause, then stop the owned services")
    stop.add_argument("--actor")
    stop.add_argument("--reason", default="the operator stopped the services")
    pause = sub.add_parser("pause", help="hold a stage, and keep holding it across a restart")
    pause.add_argument("--scope", required=True,
                       choices=("inference", "delivery", "capture"))
    pause.add_argument("--source",
                       help="with --scope capture, which connector stops reading; every "
                            "memory that registers it is held")
    pause.add_argument("--resume", action="store_true",
                       help="lift a hold a previous pause set")
    pause.add_argument("--actor")
    pause.add_argument("--reason")

    backup = sub.add_parser("backup",
                            help="copy each profile's store to a snapshot that can be "
                                 "identified and verified")
    backup.add_argument("--profile", help="one enrolled profile; every of them by default")
    backup.add_argument("--reason", default="operator backup")
    backup.add_argument("--actor")
    backup.add_argument("--list", action="store_true",
                        help="report the backups that exist and take none")
    backup.add_argument("--keep", type=int, metavar="N",
                        help="after copying, retain only this many snapshots per profile, "
                             "newest first")

    restore = sub.add_parser(
        "restore", help="take one profile's store back to a snapshot that verifies, "
                        "keeping every decision made since; run without --review to see "
                        "the snapshot's own facts first")
    restore.add_argument("--snapshot", required=True)
    restore.add_argument("--profile", help="whose store; required once one is enrolled")
    restore.add_argument("--actor")
    restore.add_argument("--review", metavar="DIGEST",
                         help="the digest of the restore that was actually shown")

    owner = sub.add_parser(
        "owner", help="the decisions this installation reserves to a human: what awaits "
                      "confirmation, and the confirmation itself")
    owner.add_argument("--list", action="store_true",
                       help="the intents and candidates awaiting a decision, counted "
                            "rather than quoted")
    owner.add_argument("--confirm-forgetting", metavar="INTENT",
                       help="apply the fence an earlier preview described")
    owner.add_argument("--confirm-identity", metavar="CANDIDATE",
                       help="say these two accounts are the same person")
    owner.add_argument("--reject-identity", metavar="CANDIDATE",
                       help="say they are not, durably")
    owner.add_argument("--revoke-edge", metavar="EDGE",
                       help="withdraw an identity that was confirmed and no longer holds")
    owner.add_argument("--confirm-assertion", metavar="ASSERTION",
                       help="let the archive stand behind a claim a pattern produced")
    owner.add_argument("--retract-assertion", metavar="ASSERTION",
                       help="stop asserting a claim that no longer holds")
    owner.add_argument("--activate-lesson", metavar="LESSON",
                       help="make a proposed habit a rule this memory applies")
    owner.add_argument("--retract-lesson", metavar="LESSON",
                       help="stop applying a lesson, now, without waiting for a review")
    owner.add_argument("--confirm-lesson", metavar="LESSON",
                       help="report, as the owner, that applying this lesson worked")
    owner.add_argument("--contradict-lesson", metavar="LESSON",
                       help="report, as the owner, that this lesson was wrong in a case "
                            "you checked")
    owner.add_argument("--version", type=int, metavar="N",
                       help="which version of a lesson the decision is about; the id may "
                            "also carry it as name@N")
    owner.add_argument("--evidence", action="append", metavar="RECORD",
                       help="a record that bears on the outcome being reported")
    owner.add_argument("--digest", metavar="PREVIEW_DIGEST",
                       help="the digest of the forgetting preview being confirmed")
    owner.add_argument("--reason")
    owner.add_argument("--valid-from")
    owner.add_argument("--valid-until")
    owner.add_argument("--profile", help="whose memory; required once one is enrolled")
    owner.add_argument("--actor")

    release = sub.add_parser("release",
                             help="stage the two environments a release is made of. It "
                                  "plans first and applying needs the digest it printed")
    release.add_argument("--into", metavar="DIR",
                         help="where to stage; defaults to <home>/runtime/current")
    release.add_argument("--source",
                         help="the checkout whose deployment/ and integrations/ are carried")
    release.add_argument("--wheel", help="a prebuilt wheel, rather than building one now")
    release.add_argument("--without-backend", action="store_true",
                         help="skip the backend environment; setup then refuses to stage "
                              "while a backend route is configured")
    release.add_argument("--apply", action="store_true")
    release.add_argument("--actor")
    release.add_argument("--review", metavar="DIGEST",
                         help="the digest of the plan that was actually shown")

    upgrade = sub.add_parser("upgrade",
                             help="plan a switch to a staged release. It is read only: "
                                  "there is no flag here that performs one")
    upgrade.add_argument("--version", metavar="RELEASE-TREE",
                         help="the staged release tree to switch to")
    upgrade.add_argument("--hermes-home",
                         help="the Hermes profile home whose host facts are reported")

    uninstall = sub.add_parser("uninstall",
                               help="remove this installation and keep every byte of memory")
    uninstall.add_argument("--keep-data", action="store_true",
                           help="required: data-preserving is the only uninstall there is")
    uninstall.add_argument("--hermes-home",
                           help="the profile home whose provider selection setup changed")
    uninstall.add_argument("--actor")
    uninstall.add_argument("--review", metavar="DIGEST",
                           help="the digest of the list that was actually shown")

    sources = sub.add_parser("sources", help="what the connectors know, including what "
                                             "they did not get")
    sources.add_argument("action", choices=("list", "reconfigure"))
    sources.add_argument("--gaps", action="store_true", help="include each source's open gaps")
    sources.add_argument("--limit", type=int, default=20)
    sources.add_argument("--hermes-home",
                         help="read the connectors one profile's memory knows about")
    sources.add_argument("--source",
                         help="with reconfigure, the connector to start a new generation of")
    sources.add_argument("--policy", default="local-only",
                         choices=("local-only", "private-api", "disabled"),
                         help="with reconfigure, the ingestion scope the connector is "
                              "re-declared under")
    sources.add_argument("--actor")
    sources.add_argument("--reason")
    sources.add_argument("--review", metavar="DIGEST",
                         help="with reconfigure, the digest of the plan that was actually "
                              "shown")

    importing = sub.add_parser("import",
                               help="read one export directory into the memory that owns it")
    importing.add_argument("--source", required=True, help="the connector id to record it under")
    importing.add_argument("--path", required=True,
                           help="a directory of exports the owner pointed at")
    importing.add_argument("--profile", help="whose memory it goes into")
    importing.add_argument("--policy", default="local-only",
                           choices=("local-only", "private-api", "disabled"),
                           help="the scope declared for a connector new to this store")
    importing.add_argument("--dry-run", action="store_true",
                           help="ask whether the export can be read, and read nothing else")
    importing.add_argument("--granularity", choices=("sample", "series"), default=None,
                           help="what a sample fixture keeps: every row the source wrote "
                                "(the shape `measure` reads), or one described summary per "
                                "device+measure+unit group, which is smaller and can never "
                                "be re-windowed later")

    form = sub.add_parser("form",
                          help="project evidence into the derived backend; run without "
                               "--review to see the bounded list and perform nothing")
    form.add_argument("--limit", type=int, default=None,
                      help="the most records one pass may select")
    form.add_argument("--max-jobs", type=int, default=None,
                      help="the most queued jobs to attempt in this invocation")
    form.add_argument("--actor")
    form.add_argument("--hermes-home",
                      help="form the memory enrolled for this Hermes profile home")
    form.add_argument("--review", metavar="DIGEST",
                      help="the digest of the list that was actually shown")

    summarize = sub.add_parser(
        "summarize",
        help="reflect one scope into a summary; run without --review to see the window and "
             "perform nothing")
    summarize.add_argument("--scope", required=True,
                           help="project:<name>, thread:<name>, account:<id>, source:<name>, "
                                "day:<date> or week:<date>")
    summarize.add_argument("--kind", choices=("thread", "day", "week", "project",
                                              "mental_model"),
                           help="the shape of the claim (default: from the scope)")
    summarize.add_argument("--limit", type=int, default=None,
                           help="the most records the window may hold")
    summarize.add_argument("--since", metavar="INSTANT",
                           help="ignore evidence older than this instant")
    summarize.add_argument("--title", help="headlines the summary instead of the derived one")
    summarize.add_argument("--actor")
    summarize.add_argument("--hermes-home",
                           help="summarize the memory enrolled for this Hermes profile home")
    summarize.add_argument("--review", metavar="DIGEST",
                           help="the digest of the window that was actually shown")

    evaluate = sub.add_parser(
        "evaluate",
        help="run one lesson's fixture suite through the authorized evaluator, or report "
             "what the last run said")
    evaluate.add_argument("--lesson", required=True, help="the lesson to evaluate")
    evaluate.add_argument("--version", type=int,
                          help="which version of it; the current one by default")
    evaluate.add_argument("--suite", metavar="FILE",
                          help="the JSON fixture suite to run; without this, report only")
    evaluate.add_argument("--model-version",
                          help="what the run was about; a verdict with no version beside it "
                               "can never be found stale")
    evaluate.add_argument("--code-version", help="defaults to this hermes-memory release")
    evaluate.add_argument("--no-promote", action="store_true",
                          help="record the verdict without letting it promote the lesson")
    evaluate.add_argument("--hermes-home",
                          help="evaluate the memory enrolled for this Hermes profile home")

    maintain = sub.add_parser(
        "maintain",
        help="one bounded pass of the background work: due reminders, invalidated summaries, "
             "expired identity candidates, the queue and what is still owed")
    maintain.add_argument("--hermes-home",
                          help="maintain the memory enrolled for this Hermes profile home")
    maintain.add_argument("--limit", type=int, default=None,
                          help="the most items each section may take in this pass")
    maintain.add_argument("--section", action="append", choices=SECTIONS, metavar="NAME",
                          dest="sections",
                          help="run only this section; repeatable (default: all of them)")
    maintain.add_argument("--at", metavar="INSTANT",
                          help="speak for a fixed instant instead for now; for a backfill or "
                               "a rehearsal, never for a live alert")

    cancel = sub.add_parser(
        "cancel",
        help="stop work: a queued job and the backend operation behind it, or say what is "
             "still owed after a restart")
    what = cancel.add_mutually_exclusive_group(required=True)
    what.add_argument("--job", metavar="ID", help="the queued job to stop")
    what.add_argument("--operation", metavar="ID",
                      help="the backend operation to ask about")
    what.add_argument("--list", action="store_true", dest="listing",
                      help="cancellations on file with no answer yet; a reading")
    cancel.add_argument("--actor", help="who decided; defaults to the owner principal")
    cancel.add_argument("--reason",
                        help="why; a cancellation without a stated reason is refused")
    cancel.add_argument("--hermes-home",
                        help="cancel within the memory enrolled for this Hermes profile home")

    goal = sub.add_parser(
        "goal",
        help="the owner's prospective memory: what is owed, and its revision-checked "
             "transitions")
    goal.add_argument("--list", action="store_true", dest="listing",
                      help="the goals still live, candidates included")
    goal.add_argument("--due-now", action="store_true", dest="due_now",
                      help="what is due at this instant, with each promise's conditions "
                           "answered rather than assumed")
    goal.add_argument("--id", metavar="GOAL", help="the goal to settle, revise or snooze")
    move = goal.add_mutually_exclusive_group()
    move.add_argument("--activate", action="store_true",
                      help="adopt a candidate the owner recognises as their own")
    move.add_argument("--complete", action="store_true", help="the thing was done")
    move.add_argument("--cancel", action="store_true", help="it is not happening after all")
    move.add_argument("--snooze", metavar="UNTIL",
                      help="hold the reminders until this instant; the goal stays as owed")
    move.add_argument("--revise", action="store_true",
                      help="move or restate it, which opens a new revision (--due, "
                           "--statement)")
    goal.add_argument("--due", metavar="INSTANT",
                      help="with --revise: the new due time, a wall time in the goal's zone")
    goal.add_argument("--statement", help="with --revise: what the owner now wants said")
    goal.add_argument("--actor", help="who decided; defaults to the owner principal")
    goal.add_argument("--reason", help="why; every transition is written down with it")
    goal.add_argument("--history", action="store_true",
                      help="with --id: every revision, oldest first")
    goal.add_argument("--hermes-home",
                      help="answer for the memory enrolled for this Hermes profile home")

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.env_file)
    except SettingError as error:
        print(f"configuration refused: {error}", file=sys.stderr)
        return 2

    if args.command == "status":
        return _status_command(settings, args)
    if args.command == "init":
        return _init_command(settings, args)
    if args.command == "doctor":
        return _doctor_command(settings, args)
    if args.command == "compatibility":
        return _compatibility_command(args)
    if args.command == "audit":
        return _audit_command(settings, args)
    if args.command == "measure":
        return _measure_command(settings, args)
    if args.command == "profiles":
        return _profiles_command(settings, args)
    if args.command == "inventory":
        return _inventory_command(settings, args)
    if args.command == "setup":
        return _setup_command(settings, args)
    if args.command == "enroll":
        return _enroll_command(settings, args)
    if args.command == "retire":
        return _retire_command(settings, args)
    if args.command == "serve":
        return _serve_command(settings)
    if args.command == "services":
        return _services_command(settings, args)
    if args.command == "start":
        return _start_command(settings)
    if args.command == "stop":
        return _stop_command(settings, args)
    if args.command == "pause":
        return _pause_command(settings, args)
    if args.command == "backup":
        return _backup_command(settings, args)
    if args.command == "restore":
        return _restore_command(settings, args)
    if args.command == "owner":
        return _owner_command(settings, args)
    if args.command == "release":
        return _release_command(settings, args)
    if args.command == "upgrade":
        return _upgrade_command(settings, args)
    if args.command == "uninstall":
        return _uninstall_command(settings, args)
    if args.command == "sources":
        return _sources_command(settings, args)
    if args.command == "import":
        return _import_command(settings, args)
    if args.command == "form":
        return _form_command(settings, args)
    if args.command == "summarize":
        return _summarize_command(settings, args)
    if args.command == "evaluate":
        return _evaluate_command(settings, args)
    if args.command == "maintain":
        return _maintenance_command(settings, args)
    if args.command == "cancel":
        return _cancel_command(settings, args)
    if args.command == "goal":
        return _goal_command(settings, args)
    return _explain_command(settings, args)


# -- the profile map -----------------------------------------------------------

def _registry(settings, *, create: bool = True):
    """The instance ledger. A read never brings one into being."""
    from .install.profiles import ProfileRegistry

    return ProfileRegistry.open(settings) if create else ProfileRegistry.reading(settings)


def _profiles_command(settings, args) -> int:
    """The profiles, and the decisions that made them that way.

    The ledger is the instance's own: an enrollment is a review digest and an actor, and
    an operator who approved one is entitled to ask later what it did and who recorded it.
    """
    registry = _registry(settings, create=False)
    try:
        if args.review:
            return _emit({"review_digest": args.review,
                          "applied": registry.receipt_for(args.review)})
        enrolled = [item.as_dict(private=True) for item in registry.profiles()]
        retired = [item.as_dict(private=True) for item in registry.profiles(include_retired=True)
                   if item.state != "enrolled"]
        stores = {item.profile: item.db_path.exists() for item in registry.profiles()}
        changes = registry.receipts(args.profile) if args.profile else registry.receipts()
    finally:
        registry.db.close()
    return _emit({"instance_home": str(settings.home), "profiles": enrolled,
                  "retired": retired,
                  "store_present": stores, "changes": changes,
                  "changes_for": args.profile,
                  "note": "each profile has its own store, bank and credential scope; "
                          "nothing here is shared but the machine"})


def _enroll_command(settings, args) -> int:
    """Two steps, because an enrollment is a decision and not a side effect.

    Without ``--review`` this prints what would change and nothing else. With it, the
    digest has to be the one that was shown, and the actor has to be the owner the
    configuration names — a caller that can name a directory does not thereby own it.
    """
    from .install.profiles import InstallationError

    name = args.profile or _profile_name(args.hermes_home)
    reading = _registry(settings, create=False)
    try:
        proposal = reading.plan(name, args.hermes_home)
    finally:
        reading.db.close()
    if not args.review:
        return _emit({**proposal,
                      "next": f"hermes-memory enroll --profile {name} "
                              f"--hermes-home {args.hermes_home} "
                              f"--review {proposal['review_digest']}"})
    registry = _registry(settings)
    try:
        actor = args.actor or settings.owner_principal
        if not actor:
            print("refused: no owner principal is configured, so nobody may approve an "
                  "enrollment (set HERMES_MEMORY_OWNER_PRINCIPAL)", file=sys.stderr)
            return 2
        try:
            result = registry.enroll(name, args.hermes_home, actor=actor,
                                     review_digest=args.review)
        except InstallationError as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        return _emit({**result,
                      "store": str(Path(result["data_dir"]) / "canonical.db")})
    finally:
        registry.db.close()


def _retire_command(settings, args) -> int:
    from .install.profiles import InstallationError

    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: retiring a profile is the owner's decision and no owner is "
              "configured", file=sys.stderr)
        return 2
    registry = _registry(settings)
    try:
        try:
            return _emit(registry.retire(args.profile, actor=actor, reason=args.reason))
        except InstallationError as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
    finally:
        registry.db.close()


def _setup_command(settings, args) -> int:
    """The transaction, from the operator's chair.

    Without ``--review`` this only reads: the eleven steps, their actions, and the host
    commands any of them would run. With it, the digest must be the one that was shown,
    the actor must be the owner the configuration names, and the host commands go through
    the one executor this process uses.
    """
    from .install.profiles import InstallationError
    from .install.setup import SetupError, plan

    arguments = {"hermes_home": args.hermes_home, "profile": args.profile, "ref": args.ref,
                 "runner": _HOST_RUNNER, "actor": args.actor, "start_services": args.start}
    try:
        proposal = plan(settings, **arguments)
    except (SetupError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not args.review:
        return _emit({**proposal,
                      "next": f"hermes-memory setup --hermes-home {args.hermes_home} "
                              f"--ref {args.ref or '<tested-release-commit>'} "
                              f"--review {proposal['review_digest']}"})
    from .install.setup import run as apply_transaction

    try:
        return _emit(apply_transaction(settings, review=args.review,
                                       **{**arguments,
                                          "actor": args.actor
                                          or settings.owner_principal or ""}))
    except (SetupError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


def _inventory_command(settings, args) -> int:
    """What is on this machine, read without changing it.

    ``--conflicts`` exists because the useful question is not everything an inventory
    can see but whether any of it should stop an install, and a script that gates on it
    should not have to re-derive the answer from a 60-key report.
    """
    from .install.inventory import conflicts, survey

    report = survey(settings, hermes_home=args.hermes_home)
    if args.conflicts:
        blocking = conflicts(report)
        print(json.dumps(blocking, indent=2, sort_keys=True))
        return 1 if blocking else 0
    return _emit({**report, "conflicts": conflicts(report)})


def _profile_name(hermes_home: str) -> str:
    """The home's own directory name, or ``default`` for a bare account home."""
    name = Path(hermes_home).expanduser().name.strip().lower()
    return name if name and name not in {"home", ".hermes", "hermes"} else "default"


# -- the services and the fence ------------------------------------------------

def _serve_command(settings) -> int:
    """The process the runtime unit starts. It refuses before it binds.

    A gate that cannot name its address, or that would bind somewhere other than
    loopback, exits non-zero here rather than starting a service that quietly listens
    on a network.
    """
    from .config import SettingError
    from .service import serve

    try:
        return serve(settings)
    except SettingError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


def _service_plan(settings) -> dict | None:
    """The owned units as this installation's layout describes them, or a refusal.

    A wheel installed on its own ships no ``deployment/systemd`` templates, so the plan
    cannot be built at all. That is a missing precondition, not a crash: the operator has
    to be told which door to bring the templates to.
    """
    from .install.profiles import InstallationError
    from .install.services import plan

    try:
        return plan(settings)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return None


def _services_command(settings, args) -> int:
    from .install.profiles import InstallationError
    from .install.services import apply

    proposal = _service_plan(settings)
    if proposal is None:
        return 2
    if args.autostart:
        # A separate verb, because "start it now" and "start it at every login" are
        # different decisions and only one of them is reversible by a reboot.
        try:
            units = _controller(settings).autostart(
                enable=args.autostart == "enable")
        except InstallationError as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        return _emit({"autostart": args.autostart, "units": units,
                      "note": "a start never lifts a pause, so an enabled unit that finds "
                              "inference or delivery held comes up holding it"})
    if not args.install:
        return _emit({**proposal,
                      "next": (f"hermes-memory services --install {proposal['review_digest']}"
                               if proposal["would_change"] else
                               "nothing to write: every owned unit already says this")})
    try:
        receipt = apply(settings, actor=args.actor or settings.owner_principal or "",
                        review=args.install)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({**receipt,
                  "next": "systemctl --user daemon-reload, then hermes-memory start"
                          if receipt["reload_needed"] else "no reload needed"})


def _controller(settings) -> Services:
    """The service controller, with whatever host command executor this process uses."""
    return Services(settings, runner=_HOST_RUNNER)


# Every command that reaches outside this process - systemctl, the host's plugin and
# configuration writers - goes through one executor, so a test can watch the ordering
# and refusal claims instead of starting real services on the machine running them.
_HOST_RUNNER: Callable[[Sequence[str]], tuple[int, str]] = subprocess_runner


def _start_command(settings) -> int:
    from .install.profiles import InstallationError

    stale = _service_plan(settings)
    if stale is None:
        return 2
    if stale["blocked"]:
        print("refused: these units exist but were not written by this installation, so "
              f"they are not ours to start: {', '.join(stale['blocked'])}", file=sys.stderr)
        return 2
    try:
        started = _controller(settings).start()
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({"ok": True, "started": started,
                  "note": "a pause the owner set is still in force; starting does not "
                          "resume formation or delivery",
                  "units_not_as_described": stale["would_change"]})


def _stop_command(settings, args) -> int:
    """Pause first, then stop. The order is the shutdown contract, not a preference.

    A worker killed mid-retain with no recorded pause looks like an idle installation
    to the next boot, which resumes exactly the formation the owner was trying to halt.
    """
    from .install.profiles import InstallationError
    from .processing.instance_gate import instance_gate
    from .storage.evidence import EvidenceStore

    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: stopping is the owner's decision and no owner principal is "
              "configured", file=sys.stderr)
        return 2
    reason = args.reason
    paused = []
    held_for = []

    def hold() -> None:
        # Inference is held once, for the whole installation: the models are shared
        # hardware, and a stop that only paused the default profile would leave the
        # second one dispatching against a machine its owner just stopped.
        with instance_gate(settings) as gate:
            gate.pause(actor=actor, reason=reason)
        for scoped in _archive_targets(settings, None):
            if not scoped.db_path.is_file():
                continue
            with EvidenceStore(scoped.db_path) as store:
                store.set_control("global", "delivery", "paused", actor=actor,
                                  reason=reason, policy_version="operator-stop")
            held_for.append(scoped.profile)
        paused.extend(["inference", "delivery"])

    try:
        stopped = _controller(settings).stop(pause=hold)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({"ok": True, "stopped": stopped, "paused": paused, "held_for": held_for,
                  "actor": actor,
                  "resumes_with": "hermes-memory pause --scope inference --resume"
                                  " (and --scope delivery --resume)"})


def _pause_command(settings, args) -> int:
    """A hold that outlives the process that set it.

    Both are written to a database rather than to the unit, because a restart is exactly
    the event that must not lift it — and an autostart that resumed formation would be the
    machine deciding against its owner. Inference goes in the instance admission ledger,
    which is the one place every profile's queue can see; delivery is a per-profile
    outbox, so it is held on each profile's own archive. Capture is a per-source stage in
    that archive, and a source may be registered in more than one profile's memory.
    """
    from .processing.instance_gate import instance_gate
    from .storage.evidence import EvidenceStore

    if args.source and args.scope != "capture":
        print(f"refused: --source names a connector, and --scope {args.scope} holds the "
              "whole installation or a profile's outbox, not one source; use "
              "--scope capture to stop reading a single source", file=sys.stderr)
        return 2
    if args.scope == "capture" and not args.source:
        print("refused: --scope capture has to say which connector stops reading; "
              "`hermes-memory sources list` names what is registered", file=sys.stderr)
        return 2
    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: a pause is attributed, and with no owner principal configured "
              "there is nobody to attribute it to", file=sys.stderr)
        return 2
    reason = args.reason or ("the owner lifted the hold" if args.resume
                             else "the owner held this stage")
    state = "active" if args.resume else "paused"
    written_to: list[str] = []
    if args.scope == "inference":
        with instance_gate(settings) as gate:
            (gate.resume if args.resume else gate.pause)(actor=actor, reason=reason)
    elif args.scope == "delivery":
        for scoped in _archive_targets(settings, None):
            if not scoped.db_path.is_file():
                continue
            with EvidenceStore(scoped.db_path) as store:
                store.set_control("global", "delivery", state, actor=actor,
                                  reason=reason, policy_version="operator-pause")
            written_to.append(scoped.profile)
    else:
        written_to = _hold_capture(settings, source=args.source, state=state,
                                   actor=actor, reason=reason)
        if not written_to:
            print(f"refused: no memory on this installation registers connector "
                  f"{args.source!r}, so there is nothing to hold; `hermes-memory sources "
                  "list` says what is", file=sys.stderr)
            return 2
    return _emit({"ok": True, "scope": args.scope, "state": state, "actor": actor,
                  "source": args.source, "reason": reason, "held": _holds(settings),
                  "written_to": written_to, "survives_a_restart": True})


def _hold_capture(settings, *, source: str, state: str, actor: str,
                  reason: str) -> list[str]:
    """Hold one connector's ingestion in every memory that registers it.

    A hold that stopped one profile polling a mailbox and left the second reading it
    would be worse than no hold: the operator would believe the source was shut off.
    Only the capture stage is written, so a source already held from formation stays
    held from formation.
    """
    from .sources.sync import SyncController
    from .storage.evidence import EvidenceStore

    held: list[str] = []
    for scoped in _archive_targets(settings, None):
        if not scoped.db_path.is_file():
            continue
        with EvidenceStore(scoped.db_path) as store:
            if store.db.execute("SELECT 1 FROM connectors WHERE source=?",
                                (source,)).fetchone() is None:
                continue
            sync = SyncController(store)
            method = sync.resume_capture if state == "active" else sync.pause_capture
            method(source, actor=actor, reason=reason, policy_version="operator-pause")
        held.append(scoped.profile)
    return held


def _holds(settings) -> dict[str, Any]:
    """What is being held right now, read from both ledgers that hold anything.

    Reported rather than echoed from the command that was just run: an operator asking
    "is it stopped" wants the machine's answer, not a restatement of their own request.
    Capture is per source, so it is a list of holds rather than one flag, and the
    inference hold carries the actor and reason the instance ledger recorded.
    """
    from .processing.instance_gate import instance_gate

    with instance_gate(settings) as gate:
        inference = gate.paused
        # The hold itself is a flag; this is the part that makes it somebody's decision.
        inference_hold = gate.hold() if inference else None
    delivery = False
    capture: list[dict[str, Any]] = []
    for scoped in _archive_targets(settings, None):
        if not scoped.db_path.is_file():
            continue
        with ReadOnlyStore(scoped.db_path) as store:
            delivery = delivery or store.stage_is_paused("global", "delivery")
            rows = store.db.execute(
                "SELECT scope FROM runtime_controls WHERE stage='capture' AND "
                "state='paused' ORDER BY scope").fetchall()
        capture.extend({"profile": scoped.profile, "source": row["scope"]} for row in rows)
    return {"inference": inference, "inference_hold": inference_hold,
            "delivery": delivery, "capture": capture}


# -- the archive, the release and the sources ----------------------------------

def _backup_command(settings, args) -> int:
    """A copy of the archive that can be named, verified and taken back.

    Every enrolled profile is backed up unless one is named, because the operator who
    types `backup` means "this installation is now recoverable" — and one store copied
    out of three would make that sentence false.
    """
    from .lifecycle.snapshots import Snapshots
    from .storage.evidence import EvidenceStore

    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: a backup records who took it, and no owner principal is configured",
              file=sys.stderr)
        return 2
    try:
        targets = _archive_targets(settings, args.profile)
    except Exception as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if args.keep is not None and args.keep < 1:
        print("refused: --keep is how many snapshots to retain, and at least one must stay",
              file=sys.stderr)
        return 2
    taken, present, trimmed = [], [], []
    try:
        for scoped in targets:
            directory = Path(scoped.data_dir) / "snapshots"
            if not scoped.db_path.is_file():
                present.append({"profile": scoped.profile, "snapshot": None,
                                "directory": str(directory),
                                "reason": f"no store at {scoped.db_path} to copy"})
                continue
            with EvidenceStore(scoped.db_path) as store:
                snapshots = Snapshots(store, directory=directory)
                if args.list:
                    present.append({"profile": scoped.profile, "directory": str(directory),
                                    "snapshots": [item.as_dict() for item in
                                                  snapshots.list(limit=20)]})
                    continue
                made = snapshots.create(reason=args.reason, actor=actor)["snapshot"]
                entry = {"profile": scoped.profile, "snapshot": made.as_dict(),
                         "verified": snapshots.verify(made.id)["ok"]}
                taken.append(entry)
                if args.keep is not None:
                    # Retention happens after the copy and never instead of it: the point of
                    # `--keep` is a bounded directory, not a shorter history.
                    pruned = snapshots.prune(keep=args.keep, actor=actor)
                    pruned["profile"] = scoped.profile
                    trimmed.append(pruned)
                    entry["retained"] = pruned["kept"]
    except EvidenceError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if args.list:
        return _emit({"listing": True, "profiles": present})
    return _emit({"ok": True, "actor": actor, "backups": taken, "skipped": present,
                  "pruned": trimmed,
                  "note": "a snapshot is one consistent copy of one store, taken while "
                          "the store stayed open; take it back with `hermes-memory restore`"})


# The facts a restore is approved against. The live epoch is in it on purpose: a store
# another process is still writing has moved since the operator read it, and the answer to
# that is a refusal and `hermes-memory stop`, never a rollback of a store nobody looked at.
RESTORE_PLAN_VERSION = "restore-plan-v1"


def _restore_command(settings, args) -> int:
    """Show the snapshot's own facts, then take back the store that was shown.

    A restore destroys every record written after the snapshot, so it is the one command
    here that a digest has to gate: the operator sees which snapshot, what it holds, and
    how many forgetting decisions the ledger will re-apply, and approves exactly that
    reading. The carried rows are counted rather than printed — a plan that dumped record
    IDs would put private evidence in the shell's scrollback.
    """
    from .ids import digest
    from .install.profiles import InstallationError
    from .lifecycle.recovery import Recovery
    from .lifecycle.snapshots import Snapshots

    try:
        targets = _archive_targets(settings, args.profile)
    except (InstallationError, EvidenceError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not args.profile and len(targets) != 1:
        print(f"refused: a restore takes back one store and {len(targets)} profiles are "
              "enrolled; name one with --profile", file=sys.stderr)
        return 2
    scoped = targets[0]
    actor = args.actor or scoped.owner_principal
    if not actor:
        print("refused: a restore is confirmed by the owner principal and none is "
              "configured, so nothing here could be sure who asked", file=sys.stderr)
        return 2
    if not scoped.db_path.is_file():
        print(f"refused: no store at {scoped.db_path} to take back", file=sys.stderr)
        return 2
    try:
        with EvidenceStore(scoped.db_path) as store:
            snapshots = Snapshots(store, directory=Path(scoped.data_dir) / "snapshots")
            checked = snapshots.verify(args.snapshot)
            if not checked["ok"]:
                print("refused: " + "; ".join(checked["problems"]), file=sys.stderr)
                return 2
            snapshot = snapshots.resolve(args.snapshot)
            carried = Recovery(store, snapshots=snapshots,
                               owner_principal=scoped.owner_principal).carry()
            proposal = {"profile": scoped.profile, "store": str(scoped.db_path),
                        "snapshot": snapshot.as_dict(), "notes": checked["notes"],
                        "live_epoch": store.epoch(),
                        "decisions_kept": {name: len(rows)
                                           for name, rows in carried.tables.items()}}
            review = digest([RESTORE_PLAN_VERSION, proposal])
            if args.review is None:
                return _emit({**proposal, "review_digest": review,
                              "next": f"hermes-memory restore --snapshot {snapshot.id} "
                                      f"--profile {scoped.profile} --actor {actor} "
                                      f"--review {review}"})
            if args.review != review:
                print("refused: the store has moved since that plan was shown, so what "
                      "would be destroyed is not what was approved. This is the reading "
                      f"now: {review}", file=sys.stderr)
                return 2
            report = Recovery(store, snapshots=snapshots,
                              owner_principal=scoped.owner_principal).restore(
                                  args.snapshot, actor=actor)
    except (EvidenceError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({"ok": True, "profile": scoped.profile, **report})


def _owner_command(settings, args) -> int:
    """The decisions that are not an agent's to make, reachable by the owner.

    Every other command here proposes, plans or reports. These are the ones §6.6 reserves
    to a human: a forgetting, an identity, and the withdrawal of one. The library behind
    each already refuses a caller who is not the named owner principal — what was missing
    was a door, and a list of what is standing behind it waiting for somebody with one.
    """
    from .install.profiles import InstallationError
    from .knowledge.assertions import AssertionStore
    from .lifecycle.erasure import ErasureManager
    from .storage.identity import IdentityStore

    decisions = {"forgetting": args.confirm_forgetting,
                 "identity": args.confirm_identity,
                 "identity-rejection": args.reject_identity,
                 "edge-revocation": args.revoke_edge,
                 "assertion": args.confirm_assertion,
                 "assertion-retraction": args.retract_assertion,
                 "lesson-activation": args.activate_lesson,
                 "lesson-retraction": args.retract_lesson,
                 "lesson-confirmation": args.confirm_lesson,
                 "lesson-contradiction": args.contradict_lesson}
    chosen = [name for name, value in decisions.items() if value]
    if args.list and chosen:
        print("refused: --list reads and a decision writes; ask for one or the other",
              file=sys.stderr)
        return 2
    if not args.list and not chosen:
        print("refused: nothing was asked. `hermes-memory owner --list` shows what awaits "
              "a decision", file=sys.stderr)
        return 2
    try:
        targets = _archive_targets(settings, args.profile)
    except (InstallationError, EvidenceError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if chosen and not args.profile and len(targets) != 1:
        print(f"refused: a decision belongs to one memory, and {len(targets)} profiles "
              "are enrolled; name one with --profile", file=sys.stderr)
        return 2
    if args.list:
        return _emit({"awaiting": _awaiting(targets)})
    name = chosen[0]
    scoped = targets[0]
    if not scoped.db_path.is_file():
        print(f"refused: no store at {scoped.db_path}", file=sys.stderr)
        return 2
    actor = args.actor or scoped.owner_principal
    if not actor:
        print("refused: no owner principal is configured, so this decision could not be "
              "attributed to anybody; set HERMES_MEMORY_OWNER_PRINCIPAL", file=sys.stderr)
        return 2
    if name == "forgetting" and not args.digest:
        print("refused: a forgetting is confirmed against the digest of the preview that "
              "was shown; `owner --list` prints it", file=sys.stderr)
        return 2
    if name != "forgetting" and not (args.reason or "").strip():
        print("refused: a decision that changes what the archive stands behind has to say "
              "why, because it outlives this conversation", file=sys.stderr)
        return 2
    chosen_id = {"forgetting": args.confirm_forgetting, "identity": args.confirm_identity,
                 "identity-rejection": args.reject_identity,
                 "edge-revocation": args.revoke_edge,
                 "assertion": args.confirm_assertion,
                 "assertion-retraction": args.retract_assertion,
                 "lesson-activation": args.activate_lesson,
                 "lesson-retraction": args.retract_lesson,
                 "lesson-confirmation": args.confirm_lesson,
                 "lesson-contradiction": args.contradict_lesson}[name]
    try:
        with EvidenceStore(scoped.db_path) as store:
            if name == "forgetting":
                outcome = ErasureManager(store, owner_principal=scoped.owner_principal)\
                    .confirm(intent_id=chosen_id, preview_digest=args.digest, actor=actor)
            elif name.startswith("assertion"):
                claims = AssertionStore(store, owner_principal=scoped.owner_principal)
                decide = claims.confirm if name == "assertion" else claims.retract
                outcome = decide(assertion_id=chosen_id, actor=actor, reason=args.reason)
            elif name.startswith("lesson"):
                from .learning.lessons import LessonStore
                from .learning.outcomes import OutcomeLog

                lessons = LessonStore(store,
                                      outcomes=OutcomeLog(store, owner_principal=scoped
                                                          .owner_principal),
                                      owner_principal=scoped.owner_principal)
                identifier, number = _lesson_ref(chosen_id, args.version)
                if number is None and name != "lesson-retraction":
                    raise EvidenceError(
                        "a lesson decision names the version it is about — `chase-invoice@3` "
                        "or --version 3 — because its versions disagree with each other by "
                        "construction, and withdrawing the newest is a different act from "
                        "withdrawing the one that was taught")
                if name == "lesson-activation":
                    outcome = lessons.activate(lesson_id=identifier, version=number,
                                               actor=actor, reason=args.reason)
                elif name == "lesson-retraction":
                    outcome = lessons.retract(lesson_id=identifier, version=number,
                                              actor=actor, reason=args.reason)
                elif name == "lesson-confirmation":
                    outcome = lessons.record_confirmation(lesson_id=identifier,
                                                          version=number, note=args.reason,
                                                          actor=actor,
                                                          evidence=args.evidence or ())
                else:
                    outcome = lessons.record_contradiction(lesson_id=identifier,
                                                           version=number, note=args.reason,
                                                           actor=actor,
                                                           evidence=args.evidence or ())
            else:
                identities = IdentityStore(store, owner_principal=scoped.owner_principal)
                if name == "identity":
                    outcome = identities.confirm(candidate_id=chosen_id, actor=actor,
                                                 reason=args.reason,
                                                 valid_from=args.valid_from,
                                                 valid_until=args.valid_until)
                elif name == "identity-rejection":
                    outcome = identities.reject(candidate_id=chosen_id, actor=actor,
                                                reason=args.reason)
                else:
                    outcome = identities.revoke(edge_id=chosen_id, actor=actor,
                                                reason=args.reason)
    except (EvidenceError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({"ok": True, "decision": name, "profile": scoped.profile,
                  "actor": actor, **outcome})


def _lesson_ref(value: str, version: int | None) -> tuple[str, int | None]:
    """`name@3` and `--version 3` are one statement; two different numbers are a refusal."""
    named, _, suffix = str(value).partition("@")
    if suffix and version is not None and suffix != str(version):
        raise EvidenceError(f"{value} names version {suffix} while --version names "
                            f"{version}; a decision cannot be about two revisions")
    if suffix and not suffix.isdigit():
        raise EvidenceError(f"{value!r} is not a lesson reference; expected a name, or "
                            "name@N")
    number = int(suffix) if suffix.isdigit() and version is None else version
    return named, number


def _awaiting(targets) -> list[dict[str, Any]]:
    """What each enrolled memory is waiting on, counted rather than quoted.

    A record ID or an account's identifiers are private evidence; a listing that dumped
    them would put them in the scrollback of whatever terminal the owner was reading in.
    The claim, the cost and the digest are what a decision actually needs — the record
    behind it is read with `explain`, by somebody who is allowed to look.
    """
    from .knowledge.assertions import AssertionStore
    from .learning.lessons import LessonStore
    from .learning.outcomes import OutcomeLog
    from .lifecycle.erasure import ErasureManager
    from .storage.identity import IdentityStore

    found = []
    for scoped in targets:
        entry = {"profile": scoped.profile, "store": str(scoped.db_path),
                 "awaiting_forgetting": [], "identity_candidates": [],
                 "candidate_assertions": [], "confirmed_identities": 0,
                 "lesson_candidates": [], "lessons_for_review": [],
                 "goal_candidates": []}
        found.append(entry)
        if not scoped.db_path.is_file():
            entry["reason"] = "no store yet"
            continue
        with ReadOnlyStore(scoped.db_path) as store:
            manager = ErasureManager(store, owner_principal=scoped.owner_principal)
            entry["awaiting_forgetting"] = manager.awaiting()
            identities = IdentityStore(store, owner_principal=scoped.owner_principal)
            entry["identity_candidates"] = [
                {"candidate_id": row["id"], "rule": row["rule"], "basis": row["basis"],
                 "proposed_by": row["proposed_by"], "proposed_kind": row["proposed_kind"],
                 "accounts": [row["account_a"], row["account_b"]],
                 "proposed_at": row["proposed_at"]}
                for row in identities.pending()]
            entry["confirmed_identities"] = int(store.db.execute(
                "SELECT count(*) FROM identity_edges WHERE state='active'").fetchone()[0])
            claims = AssertionStore(store, owner_principal=scoped.owner_principal)
            entry["candidate_assertions"] = [
                {"assertion_id": item.id, "subject": item.subject,
                 "predicate": item.predicate, "value": item.value, "kind": item.kind,
                 "unit": item.unit, "valid_from": item.valid_from,
                 "valid_to": item.valid_to}
                for item in claims.current(include_candidates=True)
                if item.status == "candidate"]
            # A proposed habit is quoted in full, unlike a forgetting preview: the sentence
            # *is* the decision, and a promotion cannot be read off a digest. What is left
            # out is the evidence behind it, which `explain` opens for somebody allowed to.
            habits = LessonStore(store,
                                 outcomes=OutcomeLog(store,
                                                     owner_principal=scoped.owner_principal),
                                 owner_principal=scoped.owner_principal)
            candidates = []
            for row in store.db.execute(
                    "SELECT id, version, text, applicability, created_by, created_kind "
                    "FROM lessons WHERE status='candidate' ORDER BY created_at, id LIMIT 8"
            ).fetchall():
                lesson = habits.get(str(row["id"]), version=int(row["version"]))
                candidates.append({"lesson": f"{row['id']}@{row['version']}",
                                   "text": row["text"],
                                   "applicability": json.loads(str(row["applicability"]
                                                                   or "[]")),
                                   "proposed_by": row["created_by"],
                                   "proposed_kind": row["created_kind"],
                                   "support": lesson.support, "against": lesson.against})
            entry["lesson_candidates"] = candidates
            entry["lessons_for_review"] = habits.needs_review(limit=8)
            # A proposed reminder is quoted for the same reason a proposed habit is: the
            # sentence is the decision, and adopting it is not something a count can express.
            entry["goal_candidates"] = [
                {"goal": row["id"], "title": row["title"], "statement": row["statement"],
                 "proposed_by": row["created_by"], "proposed_kind": row["created_kind"],
                 "proposed_at": row["created_at"],
                 "wants_a_due_time": row["due_at"] is not None}
                for row in store.db.execute(
                    "SELECT id, title, statement, created_by, created_kind, created_at, "
                    "due_at FROM goals WHERE status='candidate' "
                    "ORDER BY created_at, id LIMIT 8").fetchall()]
    return found


def _archive_targets(settings, profile: str | None) -> list:
    """The configurations whose stores a backup covers."""

    from .install.profiles import InstallationError, ProfileRegistry

    if not (Path(settings.home) / "installation.db").is_file():
        if profile:
            raise InstallationError(
                f"no instance ledger exists, so profile {profile!r} is not enrolled")
        return [settings]
    registry = ProfileRegistry.reading(settings)
    try:
        if profile:
            return [registry.profile(profile).scoped(settings)]
        enrolled = [item.scoped(settings) for item in registry.profiles()]
    finally:
        registry.db.close()
    return enrolled or [settings]


def _release_command(settings, args) -> int:
    """Stage the environments the units name. Plan mode reads like `setup`: nothing moves.

    This is the door §10.3 always assumed existed and never named: without it, every
    installation stalled at `stage` with a path to fill in by hand, and the hand-filled
    path was checked by nothing until a unit failed to start.
    """
    from pathlib import Path

    from .install.profiles import InstallationError
    from .install.release import ReleaseError
    from .install.release import apply as stage, plan

    into = Path(args.into).expanduser() if args.into \
        else Path(settings.home) / "runtime" / "current"
    backend = not args.without_backend
    try:
        if not args.apply:
            report = plan(settings=settings, into=into, source=args.source,
                          wheel=args.wheel, backend=backend)
            return _emit({**report,
                          "next": "re-run with --apply --actor <login> --review <the digest "
                                  "above>. Nothing here moves runtime/current once it exists "
                                  "— that switch is `upgrade`, and it waits for a quiesced "
                                  "machine"})
        if not args.review:
            raise ReleaseError("staging fetches packages and writes two interpreters; the "
                               "approval is the digest the plan printed, and none was given")
        report = stage(settings=settings, into=into, source=args.source, wheel=args.wheel,
                       actor=args.actor or "", review=args.review, backend=backend)
    except (ReleaseError, InstallationError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not report.get("complete", True):
        report["warning"] = ("the tree was staged but does not satisfy what the units ask of "
                             "it; read `missing` before pointing anything at it")
    return _emit(report)


def _upgrade_command(settings, args) -> int:
    """The plan of §10.6, and deliberately not the switch.

    There is no ``--apply`` here to approve. Performing the switch needs a quiesced
    machine, a rehearsal on a restored copy and a validated final state, and an operator
    who has those would not want a flag that guesses at them.
    """
    from .install.profiles import InstallationError
    from .install.upgrade import UpgradeError, upgrade_plan

    try:
        report = upgrade_plan(settings, release=args.version, hermes_home=args.hermes_home)
    except (UpgradeError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({**report,
                  "next": "this plan performs nothing. Read the blockers, take a backup, "
                          "rehearse the migrations on a restored copy, and switch the "
                          "release pointer only when the rehearsal passed"})


def _uninstall_command(settings, args) -> int:
    """Two steps again: the list first, then the removal of exactly that list."""
    from .install.profiles import InstallationError
    from .install.uninstall import UninstallError, uninstall_apply, uninstall_plan

    try:
        proposal = uninstall_plan(settings, hermes_home=args.hermes_home)
    except (UninstallError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not args.review:
        return _emit({**proposal, "next": proposal["next"].replace(
            "--actor <owner-principal>",
            f"--actor {args.actor or settings.owner_principal or '<owner-principal>'}")})
    try:
        return _emit(uninstall_apply(
            settings, actor=args.actor or settings.owner_principal or "",
            review=args.review, keep_data=args.keep_data,
            hermes_home=args.hermes_home, runner=_HOST_RUNNER))
    except (UninstallError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


def _sources_command(settings, args) -> int:
    """Each connector's own account: where it stopped, and what it could not get."""
    from .sources.sync import SyncController

    if args.action == "reconfigure":
        return _reconfigure_command(settings, args)
    if args.source or args.review:
        print("refused: --source and --review belong to `sources reconfigure`; a listing "
              "reports every connector, and naming one would make it a different answer",
              file=sys.stderr)
        return 2

    def read(_scoped, store):
        sync = SyncController(store)
        listed = []
        for row in store.db.execute("SELECT source FROM connectors ORDER BY source"):
            state = sync.state(row["source"])
            entry = {"source": row["source"], "generation": state["generation"],
                     "policy": state["policy_version"], "cursor": state["cursor"],
                     "coverage": state["coverage_state"],
                     "last_success_at": state["last_success_at"],
                     "paused_stages": sync.paused_stages(row["source"]),
                     "lease": "held" if state["lease_active"] else "free"}
            if args.gaps:
                entry["open_gaps"] = sync.gaps(row["source"], limit=args.limit)
            listed.append(entry)
        return {"sources": listed,
                "registered": len(listed),
                "note": "coverage is what the source says it has; a gap is what it could "
                        "not hand over, and neither is a claim about what is missing"}

    return _with_store(settings, read, args.hermes_home)


# The facts a reconfigure is approved against, so an old digest cannot authorise a plan
# this command no longer shows.
RECONFIGURE_PLAN_VERSION = "connector-reconfigure-v1"


def _reconfigure_command(settings, args) -> int:
    """Start one connector's generation again, under a declaration the owner signed.

    A reconfigure strands the cursor, so the next read is a full re-read of the source,
    and it re-records the ingestion scope the connector is filed under. Both are felt
    longest by whoever comes after this conversation, which is why the command shows the
    connectors it found, what each one currently holds, and refuses to write without the
    digest of that exact showing. Nothing here touches the archive: records already stored
    stay stored, and a re-read of one arrives as a duplicate.
    """
    from .ids import digest
    from .install.profiles import InstallationError
    from .sources.sync import SyncController
    from .storage.evidence import EvidenceStore

    if args.gaps:
        print("refused: --gaps reads and reconfigure writes; ask for one or the other",
              file=sys.stderr)
        return 2
    if not args.source:
        print("refused: reconfigure has to say which connector starts over; `sources "
              "list` names what is registered", file=sys.stderr)
        return 2
    actor = args.actor or settings.owner_principal
    if not actor:
        print("refused: a reconfigure changes what a connector is recorded as allowed to "
              "ingest, and with no owner principal configured there is nobody to attribute "
              "it to", file=sys.stderr)
        return 2
    if not (args.reason or "").strip():
        print("refused: re-declaring a connector has to say why — a dropped cursor means a "
              "full re-read of a source that may hold other people's messages",
              file=sys.stderr)
        return 2
    try:
        targets = _archive_targets(settings, None)
    except (InstallationError, EvidenceError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    found: list[dict[str, Any]] = []
    try:
        for scoped in targets:
            if not scoped.db_path.is_file():
                continue
            with ReadOnlyStore(scoped.db_path) as store:
                if store.db.execute("SELECT 1 FROM connectors WHERE source=?",
                                    (args.source,)).fetchone() is None:
                    continue
                sync = SyncController(store)
                state = sync.state(args.source)
                found.append({"profile": scoped.profile, "store": str(scoped.db_path),
                              "generation_now": state["generation"],
                              "policy_now": state["policy_version"],
                              "policy_next": args.policy,
                              "cursor_held": bool(state["cursor"]),
                              "coverage_now": state["coverage_state"],
                              "lease": "held" if state["lease_active"] else "free",
                              "paused_stages": sync.paused_stages(args.source)})
    except EvidenceError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not found:
        print(f"refused: no memory on this installation registers connector "
              f"{args.source!r}, so there is no connector to start over", file=sys.stderr)
        return 2
    will = ["a new connector generation, so any writer still holding the old fence is "
            "refused rather than committed",
            "the cursor dropped: the next read starts from the beginning of the source",
            f"the ingestion scope re-recorded as {args.policy!r}",
            "coverage reset to 'unknown' until a run says otherwise"]
    review = digest([RECONFIGURE_PLAN_VERSION, args.source, actor, args.reason, found, will])
    if not args.review:
        return _emit({"source": args.source, "actor": actor, "reason": args.reason,
                      "connectors": found, "will": will, "review_digest": review,
                      "note": "nothing was written. Approve this exact plan by re-running "
                              "with --review <digest>",
                      "archive": "records already stored are not touched by this"})
    if args.review != review:
        print("refused: the digest is not the one this plan carries, so what is being "
              "approved is not what was shown; run without --review and read the output",
              file=sys.stderr)
        return 2
    written: list[dict[str, Any]] = []
    try:
        for plan in found:
            with EvidenceStore(plan["store"]) as store:
                generation = SyncController(store).reconfigure(
                    args.source, policy_version=args.policy, actor=actor,
                    reason=args.reason)
            written.append({"profile": plan["profile"], "generation": generation})
    except EvidenceError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit({"ok": True, "source": args.source, "actor": actor, "reason": args.reason,
                  "reconfigured": written, "will": will, "review_digest": review,
                  "policy": args.policy,
                  "held": _holds(settings),
                  "next": "the next connector run re-reads from the beginning; until it "
                          "finishes, coverage says 'unknown' and that is the honest answer"})


def _import_command(settings, args) -> int:
    """Read a directory of exports the owner named into the store that owns it.

    The adapter is chosen from a fixed map of ids to file readers: no path here is ever
    turned into a command, a URL or an import of somebody's code. A connector already
    registered keeps its own policy version, because re-declaring a scope from a shell
    command is how a private source starts looking public.
    """
    from .sources.runtime import ConnectorRuntime
    from .sources.sync import SyncController
    from .storage.evidence import EvidenceStore

    adapter_class = _export_readers().get(args.source)
    if adapter_class is None:
        print("refused: no export reader for source "
              f"{args.source!r}; this command reads {', '.join(sorted(_export_readers()))} "
              "from a directory the owner points at", file=sys.stderr)
        return 2
    try:
        targets = _archive_targets(settings, args.profile)
    except Exception as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if len(targets) != 1:
        print("refused: --profile must name which memory the export is being read into",
              file=sys.stderr)
        return 2
    scoped = targets[0]
    if not scoped.db_path.is_file():
        print(f"refused: no canonical store at {scoped.db_path}; an import writes into a "
              "memory that exists, so run `hermes-memory init` (or enroll) first",
              file=sys.stderr)
        return 2
    offered = getattr(adapter_class, "granularities", ())
    if args.granularity is None and len(offered) > 1:
        print("refused: this export can be kept two ways, and the choice decides what can "
              "be asked of it later. --granularity sample keeps every row the source wrote "
              f"(the shape `hermes-memory measure` reads); --granularity {offered[-1]} keeps "
              "one described summary per group and gives the rows up", file=sys.stderr)
        return 2
    if args.granularity and args.granularity not in offered:
        print(f"refused: --granularity {args.granularity!r} is not a choice the reader for "
              f"source {args.source!r} has; it states one record per item", file=sys.stderr)
        return 2
    options = ({"per_sample": args.granularity == "sample"} if args.granularity else {})
    adapter = adapter_class(args.path, source=args.source, **options)
    if not args.dry_run:
        try:
            with EvidenceStore(scoped.db_path) as store:
                sync = SyncController(store)
                try:
                    declared = sync.state(args.source)["policy_version"]
                except EvidenceError:
                    declared = args.policy
                    sync.register(args.source, policy_version=declared)
                runtime = ConnectorRuntime(store, sync,
                                           holder=f"cli import from {args.path}")
                outcome = runtime.run(adapter)
        except EvidenceError as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        return _emit({**outcome.as_dict(), "profile": scoped.profile, "policy": declared,
                      "store": str(scoped.db_path),
                      "recheck": "a run that stopped for a reason says so; the source's "
                                 "coverage claim is its own"})
    return _emit({"dry_run": True, "source": args.source, "path": str(args.path),
                  "profile": scoped.profile, "reachable": adapter.check(),
                  "note": "nothing was read; run the same command without --dry-run to "
                          "record what is in the export"})


def _form_command(settings, args) -> int:
    """The one place memory-originated inference is dispatched from a shell.

    Planning is a reading. Performing is a write to the queue, a slot in a device the
    whole machine shares, and a spend from a daily budget, so it requires the digest of
    the list that was actually shown and the name of the person who authorised it. There
    is no ``--yes`` and no default: an unattended pass is a worker that has not been
    commissioned yet, not a flag on this command.
    """
    from .install.profiles import InstallationError
    from .processing.formation import (DEFAULT_BATCH, MAX_JOBS, FormationError,
                                       formation_apply, formation_plan)

    try:
        settings = _memory_for_home(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    limit = DEFAULT_BATCH if args.limit is None else args.limit
    max_jobs = MAX_JOBS if args.max_jobs is None else args.max_jobs
    try:
        proposal = formation_plan(settings, limit=limit)
    except FormationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not args.review:
        return _emit({**proposal, "next": "nothing left this process. Approve this exact "
                                          "list with --review <digest> and an --actor; a "
                                          "pass that costs tokens names who authorised it"})
    try:
        return _emit(formation_apply(
            settings, review=args.review,
            actor=args.actor or settings.owner_principal or "",
            limit=limit, max_jobs=max_jobs))
    except FormationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


def _summarize_command(settings, args) -> int:
    """One reflection, one window, one named author.

    The scope is what the summary will be a claim *about*, and it is listed before the
    model is asked: approving a digest that says 12 records from one project is not
    approving a paragraph about whatever else that backend happens to hold. A
    ``mental_model`` reaches past the evidence, so only the owner principal may write one.
    """
    from .install.profiles import InstallationError
    from .processing.summarization import (DEFAULT_BATCH, SummarizeError,
                                           summarize_apply, summarize_plan)

    try:
        settings = _memory_for_home(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    limit = DEFAULT_BATCH if args.limit is None else args.limit
    actor = args.actor or settings.owner_principal or ""
    try:
        proposal = summarize_plan(settings, scope=args.scope, kind=args.kind,
                                 limit=limit, since=args.since)
    except SummarizeError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not args.review:
        return _emit({**proposal, "next": "no request was sent. Approve this exact window "
                                          "with --review <digest> and an --actor; a summary "
                                          "costs tokens and is attributed to whoever asked"})
    try:
        return _emit(summarize_apply(settings, scope=args.scope, kind=args.kind,
                                     review=args.review, actor=actor, limit=limit,
                                     since=args.since, title=args.title))
    except SummarizeError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


def _evaluate_command(settings, args) -> int:
    """The one door that can turn a run into a rule.

    No agent credential reaches this: the evaluator program is configuration the owner
    wrote, and its answer is weighed case by case by the ledger, which promotes nothing
    that was scored by the same hand that proposed the lesson. Without ``--suite`` the
    command only reports what the archive already remembers, and opens no socket.
    """
    from .install.profiles import InstallationError
    from .learning.evaluator import EvaluationError, evaluate, report_on

    try:
        settings = _memory_for_home(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    try:
        if not args.suite:
            return _emit(report_on(settings, lesson=args.lesson, version=args.version))
        return _emit(evaluate(settings, lesson=args.lesson, suite_path=args.suite,
                              model_version=args.model_version or "",
                              code_version=args.code_version, promote=not args.no_promote))
    except (EvaluationError, EvidenceError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


# The background door. Nothing here needs an --actor or a --review, because nothing here
# spends anything: the pass writes only what is already promised, and the first thing it
# refuses to do is ask a model. That is what lets it sit on a timer; the moment a run
# wanted tokens it would belong to `form`, which names who authorised the spend.
def _maintenance_command(settings, args) -> int:
    from .install.profiles import InstallationError
    from .processing.maintenance import DEFAULT_LIMIT, MAX_LIMIT, run

    try:
        settings = _memory_for_home(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    limit = DEFAULT_LIMIT if args.limit is None else args.limit
    if not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        print(f"refused: --limit must be between 1 and {MAX_LIMIT}", file=sys.stderr)
        return 2
    try:
        report = run(settings, limit=limit, at=args.at, sections=tuple(args.sections or ()))
    except (EvidenceError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not report.get("ok"):
        return _emit(report, 2)
    return _emit(report)


# The stop door. Unlike the background pass this one names a person: a cancelled job is a
# decision somebody made, and the ledger keeps it beside their name. It never claims the
# backend stopped — the report says what was asked, what was answered and what is unknown.
def _cancel_command(settings, args) -> int:
    from .install.profiles import InstallationError
    from .processing.cancellation import outstanding, run

    try:
        settings = _memory_for_home(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    actor = args.actor or settings.owner_principal
    if args.listing:
        return _emit(outstanding(settings))
    try:
        report = run(settings, job=args.job, operation=args.operation,
                     actor=actor, reason=args.reason or "")
    except (EvidenceError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not report.get("ok"):
        return _emit(report, 2)
    return _emit(report)


# The owner's own prospective memory. Every transition here is already owner-gated in the
# store, so this door's job is to name the goal, carry the reason into the history row, and
# refuse the ones that would silently destroy a promise: a snooze that suppresses nothing, a
# revision of something already settled.
def _goal_command(settings, args) -> int:
    from .install.profiles import InstallationError
    from .prospective.due_events import DueEventLog
    from .prospective.goals import GoalStore
    from .storage.evidence import EvidenceStore

    try:
        settings = _memory_for_home(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not settings.db_path.exists():
        print(f"refused: no canonical store at {settings.db_path}; run `hermes-memory init` "
              "first", file=sys.stderr)
        return 2
    actor = args.actor or settings.owner_principal
    readings = (args.listing or args.due_now or args.history)
    moving = (args.activate or args.complete or args.cancel or args.snooze
              or args.revise)
    if not readings and not moving:
        print("refused: name --list, --due-now, --history, or one of --activate/--complete/"
              "--cancel/--snooze/--revise", file=sys.stderr)
        return 2
    if moving and not args.id:
        print("refused: a transition names the goal it applies to with --id", file=sys.stderr)
        return 2
    if readings and moving:
        print("refused: a reading does not also settle something; run them separately",
              file=sys.stderr)
        return 2
    if len([item for item in (args.listing, args.due_now, args.history) if item]) > 1:
        print("refused: one reading at a time — each answers a different question",
              file=sys.stderr)
        return 2
    try:
        with EvidenceStore(settings.db_path) as store:
            goals = GoalStore(store, events=DueEventLog(store),
                              owner_principal=settings.owner_principal)
            if args.listing:
                return _emit({"goals": [item.as_dict() for item in
                                        goals.open(include_candidates=True)],
                              "note": "a candidate is an agent's proposal the owner has not "
                                      "adopted; nothing reminds for it until --activate"})
            if args.due_now:
                # Each condition is answered here rather than assumed: a promise with a
                # predicate is not late or on time, it is waiting on something the store can
                # say, or cannot.
                return _emit({"due": goals.due_now(),
                              "note": "ready means every condition was satisfied; unknown "
                                      "names the conditions the store could not answer, which "
                                      "are held rather than fired or destroyed"})
            if args.history:
                return _emit({"goal_id": args.id, "history": goals.history(args.id)})
            if not (args.reason or "").strip():
                print("refused: every goal transition is written down with a reason, because "
                      "it outlives the conversation that made it", file=sys.stderr)
                return 2
            if args.activate:
                answer = goals.activate(goal_id=args.id, actor=actor, reason=args.reason)
            elif args.complete:
                answer = goals.complete(goal_id=args.id, actor=actor, reason=args.reason)
            elif args.cancel:
                answer = goals.cancel(goal_id=args.id, actor=actor, reason=args.reason)
            elif args.snooze:
                answer = goals.snooze(goal_id=args.id, actor=actor, until=args.snooze,
                                      reason=args.reason)
            else:
                if not args.due and not args.statement:
                    print("refused: --revise needs --due or --statement, otherwise it opens a "
                          "revision that changes nothing", file=sys.stderr)
                    return 2
                answer = goals.revise(goal_id=args.id, actor=actor, reason=args.reason,
                                      due=args.due, statement=args.statement)
    except (EvidenceError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit(answer)


# The file readers only. A live mailbox, an MCP server or the host's own event spool
# each need an authorisation this command cannot carry, so they are not in the map.
def _export_readers() -> dict[str, Any]:
    from .sources.email import EmailSource
    from .sources.files import FileSource
    from .sources.structured import StructuredSource
    from .sources.whatsapp_export import WhatsAppExport

    return {"email": EmailSource, "files": FileSource, "structured": StructuredSource,
            "whatsapp": WhatsAppExport}


def _memory_for_home(settings, hermes_home: str | None):
    """One enrolled profile's configuration, or this installation's own when none is named.

    The lookup is the ledger's, so an unenrolled or retired home is refused rather than
    answered from the default profile. A report about one person that quietly came out of
    another person's memory is the exact failure the profile map exists to prevent.
    """
    if not hermes_home:
        return settings
    from .install.profiles import ProfileRegistry

    reading = ProfileRegistry.reading(settings)
    try:
        return reading.resolve(hermes_home).scoped(settings)
    finally:
        reading.db.close()


# -- the commands ------------------------------------------------------------

def _status_command(settings, args) -> int:
    from .install.profiles import InstallationError
    from .operations.status import StatusReporter

    try:
        settings = _reading_settings(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    present = settings.db_path.exists()
    report = {**_configuration(settings),
              "profile": settings.profile,
              "capabilities": _capabilities(settings, present),
              "stages": None}
    if not present:
        report["note"] = "no canonical store yet; run `hermes-memory init`"
        return _emit(report)
    try:
        with ReadOnlyStore(settings.db_path) as store:
            report["stages"] = StatusReporter(store, settings=settings).report(
                profile=settings.profile)
    except EvidenceError as error:
        report["note"] = str(error)
        return _emit(report, 1)
    return _emit(report)


def _init_command(settings, args) -> int:
    from .install.profiles import InstallationError

    try:
        settings = _memory_for_home(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        return _emit({"would_create": [str(settings.data_dir), str(settings.blob_dir)],
                      "store": str(settings.db_path), "profile": settings.profile,
                      "migrations": "applied when the store is opened for writing"})
    settings.data_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    settings.blob_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    with EvidenceStore(settings.db_path) as store:
        store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return _emit({"initialized": str(settings.db_path), "profile": settings.profile})


def _doctor_command(settings, args) -> int:
    from .install.profiles import InstallationError
    from .operations.doctor import Doctor, unreachable_store_report

    try:
        settings = _reading_settings(settings, args.hermes_home)
    except InstallationError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not settings.db_path.exists():
        return _emit(unreachable_store_report(
            settings.db_path, "the canonical store does not exist yet"), 1)
    try:
        with ReadOnlyStore(settings.db_path) as store:
            report = Doctor(store, settings=settings).examine(
                connectivity=args.probe, synthetic=args.synthetic_probe,
                profile=settings.profile)
    except EvidenceError as error:
        return _emit(unreachable_store_report(settings.db_path, str(error)), 1)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return int(report["exit_code"])


def _compatibility_command(args) -> int:
    """The manifest door: reading it is safe anywhere, writing it belongs to packaging.

    ``--write`` is deliberately not a step the installer performs. A setup that regenerated
    its own compatibility claims would be a build grading its own homework; the check is only
    worth anything because the file was written by a different act than the one that reads it.
    """
    from .install import compatibility

    if args.write:
        written = compatibility.write()
        return _emit({"written": str(written), "facts": compatibility.facts(),
                      "note": "ship this file with the release it describes"})
    checked = compatibility.verify(digests=args.digests)
    if not checked["ok"] and not checked.get("absent"):
        checked["next"] = ("hermes-memory compatibility --write, and ship what it prints "
                           "beside the code that produced it")
    return _emit(checked, 0 if checked["ok"] else 1)


def _audit_command(settings, args) -> int:
    from .operations.audit import AuditTrail

    def read(_scoped, store):
        trail = AuditTrail(store)
        if args.timeline:
            return trail.timeline(args.timeline, include_text=args.include_private,
                                 limit=args.limit)
        if args.actions:
            return {"actions": trail.actions()}
        if args.actors:
            return {"actors": trail.by_actor()}
        if args.decisions:
            return {"category": args.decisions, "decided_by": trail.decided_by(args.decisions)}
        if args.source:
            return trail.source_history(args.source, limit=args.limit)
        return trail.recent(action=args.action, object_id=args.object_id,
                            actor=args.actor, limit=args.limit)

    return _with_store(settings, read, args.hermes_home)


def _explain_command(settings, args) -> int:
    from .operations.explanations import Explanations

    def read(scoped, store):
        explained = Explanations(store, settings=scoped)
        if args.record:
            return explained.retrieval(args.record, limit=args.limit)
        if args.artifact:
            return explained.notification(args.artifact, include_private=args.include_private)
        if args.goal:
            return explained.goal(args.goal, include_private=args.include_private)
        if args.lesson:
            return explained.lesson(args.lesson, include_private=args.include_private)
        if args.summary:
            return explained.summary(args.summary, include_private=args.include_private)
        return explained.suppressed(topic=args.not_told, at=args.at, limit=args.limit)

    return _with_store(settings, read, args.hermes_home)


def _measure_command(settings, args) -> int:
    """Typed measurements, computed from the samples rather than recalled about them.

    A number that does not say what it measured, in what unit, from which device, and
    when, is not a memory — so the reading carries all four, reports the samples it
    could not place in time or could not read, and refuses to average across a unit
    disagreement instead of producing a plausible wrong figure.
    """
    from .storage.measurements import Measurements

    asked = (args.what or args.device or args.source or args.unit or args.since
             or args.until)
    if args.listing and asked:
        print("refused: --list answers 'what can I ask?'; name the measure alone to ask it",
              file=sys.stderr)
        return 2

    def read(_scoped, store):
        subject = Measurements(store)
        if args.listing:
            return {"series": subject.available(),
                    "ask": "hermes-memory measure --what <measure> [--device D] "
                           "[--since <timestamp> --until <timestamp>]"}
        if not args.what:
            raise EvidenceError("measure has to say what to read; --list names what is held")
        return subject.series(args.what, device=args.device, source=args.source,
                              unit=args.unit, since=args.since, until=args.until)

    return _with_store(settings, read, args.hermes_home)


def _with_store(settings, read: Callable[[Any], Any], hermes_home: str | None = None) -> int:
    """Open one memory read-only and answer from it, refusing rather than guessing.

    ``hermes_home`` decides when it is given. Otherwise a single enrolled profile is
    unambiguous — that store is the one the imports have been writing into — and more
    than one is a question for the operator, because a reading about one person that came
    quietly out of another person's memory is the failure the profile map exists to stop.
    """
    from .install.profiles import InstallationError

    try:
        scoped = _reading_settings(settings, hermes_home)
        with ReadOnlyStore(scoped.db_path) as store:
            payload = read(scoped, store)
    except (EvidenceError, ValueError, InstallationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return _emit(payload)


def _reading_settings(settings, hermes_home: str | None = None):
    from .install.profiles import InstallationError

    if hermes_home:
        return _memory_for_home(settings, hermes_home)
    targets = _archive_targets(settings, None)
    if len(targets) == 1:
        return targets[0]
    raise InstallationError(
        f"{len(targets)} memories are enrolled in this installation, so a reading has to "
        "name one of them; --profile or --hermes-home says whose store is being read")


def _emit(payload, code: int = 0) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
