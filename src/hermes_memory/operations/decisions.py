"""The owner's decisions as one function with two handles.

The CLI door and a reply from the owner's own chat must not be two implementations of the
same act. Two implementations means two sets of rules, and the weaker one always wins: a
forgetting confirmed over a transport that skipped one check the command line makes is the
bug this module exists to prevent. So every archive-scoped owner act is written once here,
both handles call it, and the reply path can only name an act listed in ``ANSWERABLE``.

Three things are deliberately *not* here. The delivery switch and a standing grant over the
shared models are decisions about the machine rather than about one memory, and they are
answered from the instance ledger. And nothing in this module decides whether a caller *may*
call it: the stores behind each act refuse a caller who is not the named owner principal,
which is the rule that has to hold no matter how many handles a door grows.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..storage.evidence import EvidenceError

__all__ = ["ACTS", "ANSWERABLE", "NEEDS_DIGEST", "lesson_ref", "settle", "summarise"]

#: Every owner act reachable through one archive. The names are the ones `owner --list`
#: prints, so a question, a listing and a command all call the same thing the same name.
ACTS = ("forgetting", "identity", "identity-rejection", "edge-revocation", "assertion",
        "assertion-retraction", "lesson-activation", "lesson-retraction",
        "lesson-confirmation", "lesson-contradiction", "goal-activation", "goal-completion",
        "goal-cancellation")

#: What a reply from the owner's channel may settle. Everything above: the answer carries
#: the digest it was shown, and the fence below refuses it if the archive has moved.
ANSWERABLE = ACTS

#: A forgetting is confirmed against the digest of the impact preview, never against a
#: sentence about it. Everything else needs the owner's reason in their own words.
NEEDS_DIGEST = frozenset({"forgetting"})
#: The two sentences the door says when a decision arrives without what it needs. They live
#: here because both handles — the command line and a reply from the owner's channel — have
#: to refuse in the same words, or the rule has quietly become two rules.
REASON_REQUIRED = ("a decision that changes what the archive stands behind has to say why, "
                   "because it outlives this conversation")
DIGEST_REQUIRED = ("a forgetting is confirmed against the digest of the preview that was "
                   "shown; `owner --list` prints it")

#: A lesson is versioned, and its versions disagree with each other by construction:
#: withdrawing the newest is a different act from withdrawing the one that was taught.
LESSON_ACTS = frozenset({"lesson-activation", "lesson-retraction", "lesson-confirmation",
                         "lesson-contradiction"})
#: Acts whose subject is a goal candidate or a live goal rather than a claim.
GOAL_ACTS = frozenset({"goal-activation", "goal-completion", "goal-cancellation"})


def lesson_ref(value: str, version: int | None = None) -> tuple[str, int | None]:
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


def settle(store, *, owner_principal: str | None, name: str, subject_id: str,
           actor: str, reason: str | None = None, preview_digest: str | None = None,
           version: int | None = None, valid_from: str | None = None,
           valid_until: str | None = None,
           evidence: Sequence[str] = ()) -> dict[str, Any]:
    """Take one owner decision, from whichever handle arrived.

    The refusals here are the ones the command line made first: an act that needs the digest
    of what was shown cannot be settled by a sentence, and an act that changes what the
    archive stands behind has to say why, because it outlives the conversation that asked.
    """
    if name not in ACTS:
        raise EvidenceError(f"unknown owner act {name!r}; the acts are {list(ACTS)}")
    if name == "delivery":
        raise EvidenceError("the delivery switch is a decision about this machine, not about "
                            "one memory; it is answered from the instance ledger")
    if name in NEEDS_DIGEST:
        if not preview_digest:
            raise EvidenceError(DIGEST_REQUIRED)
    elif not (reason or "").strip():
        raise EvidenceError(REASON_REQUIRED)
    if name in LESSON_ACTS and version is None and "@" not in str(subject_id) \
            and name != "lesson-retraction":
        raise EvidenceError(
            "a lesson decision names the version it is about — `chase-invoice@3` or "
            "--version 3 — because its versions disagree with each other by construction")
    actor = (actor or "").strip()
    if not actor:
        raise EvidenceError("no owner principal is configured, so this decision could not be "
                            "attributed to anybody")
    subject = str(subject_id or "").strip()
    if not subject:
        raise EvidenceError(f"{name} needs the id of the thing it decides about")

    if name == "forgetting":
        from ..lifecycle.erasure import ErasureManager

        return ErasureManager(store, owner_principal=owner_principal).confirm(
            intent_id=subject, preview_digest=str(preview_digest), actor=actor)
    if name.startswith("assertion"):
        from ..knowledge.assertions import AssertionStore

        claims = AssertionStore(store, owner_principal=owner_principal)
        decide = claims.confirm if name == "assertion" else claims.retract
        return decide(assertion_id=subject, actor=actor, reason=reason)
    if name in LESSON_ACTS:
        from ..learning.lessons import LessonStore
        from ..learning.outcomes import OutcomeLog

        lessons = LessonStore(store, outcomes=OutcomeLog(store, owner_principal=owner_principal),
                              owner_principal=owner_principal)
        identifier, number = lesson_ref(subject, version)
        if name == "lesson-activation":
            return lessons.activate(lesson_id=identifier, version=number, actor=actor,
                                    reason=reason)
        if name == "lesson-retraction":
            return lessons.retract(lesson_id=identifier, version=number, actor=actor,
                                   reason=reason)
        if name == "lesson-confirmation":
            return lessons.record_confirmation(lesson_id=identifier, version=number,
                                               note=reason, actor=actor, evidence=evidence)
        return lessons.record_contradiction(lesson_id=identifier, version=number,
                                            note=reason, actor=actor, evidence=evidence)
    if name in GOAL_ACTS:
        from ..prospective.due_events import DueEventLog
        from ..prospective.goals import GoalStore

        goals = GoalStore(store, events=DueEventLog(store), owner_principal=owner_principal)
        if name == "goal-activation":
            return goals.activate(goal_id=subject, actor=actor, reason=reason)
        if name == "goal-completion":
            return goals.complete(goal_id=subject, actor=actor, reason=reason)
        return goals.cancel(goal_id=subject, actor=actor, reason=reason)

    from ..storage.identity import IdentityStore

    identities = IdentityStore(store, owner_principal=owner_principal)
    if name == "identity":
        return identities.confirm(candidate_id=subject, actor=actor, reason=reason,
                                  valid_from=valid_from, valid_until=valid_until)
    if name == "identity-rejection":
        return identities.reject(candidate_id=subject, actor=actor, reason=reason)
    return identities.revoke(edge_id=subject, actor=actor, reason=reason)


def summarise(name: str, outcome: dict[str, Any]) -> str:
    """One line about what a decision did, for a listing or a receipt.

    The stored answer is JSON; a reader of `owner --list` or of a reply receipt wants the
    verb and the state, not a dict.
    """
    state = outcome.get("status") or outcome.get("state") or outcome.get("action") or ""
    changed = outcome.get("changed")
    words = {"forgetting": "the forgetting was confirmed and the erasure began",
             "identity": "the two accounts are now held to be one person",
             "identity-rejection": "the proposed join was refused",
             "edge-revocation": "that identity link is withdrawn",
             "assertion": "the archive now stands behind this claim",
             "assertion-retraction": "the claim is withdrawn",
             "lesson-activation": "the lesson is being taught",
             "lesson-retraction": "the lesson is withdrawn",
             "lesson-confirmation": "the outcome was recorded in its favour",
             "lesson-contradiction": "the outcome was recorded against it",
             "goal-activation": "the reminder is live",
             "goal-completion": "the thing is done",
             "goal-cancellation": "it is not happening after all"}
    line = words.get(name, f"{name} decided")
    if state:
        line += f" ({state})"
    if changed is False:
        line += "; nothing changed, it already stood there"
    return line
