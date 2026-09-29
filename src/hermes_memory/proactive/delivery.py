"""The handover: one artifact, to the one destination the owner named.

The outbox records what may be said; this module is what says it. The consent rules here
are the framework's — they read the owner's configured destination and the named owner
principal, not anything a plugin decides — while the thing that puts bytes in front of a
person stays outside, handed in as a sink. A caller can be the `deliver` command, the
host's scheduler or a test; the state machine is the same for all of them.

The interesting part is what happens when something goes wrong. A failure before the
bytes leave is a release: nothing was delivered. A failure after the handover began is
``uncertain``, which is not put back in the queue, because the third copy of a
notification is a worse outcome than the missing first one. And a transport that says
"sent" without a digest to check it against gets ``accepted_unverified``, not
confirmation: that is a report of the transport's own intent, not of the owner's inbox.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from ..ids import digest, now
from ..storage.evidence import EvidenceError
from .outbox import Outbox
from .policy import AttentionPolicy

__all__ = ["DeliveryPolicy", "bounded_drain", "command_sink", "correlation",
           "deliver_once", "deliver_ready", "drain_installed", "drain_line", "last_drain",
           "local_sink", "sink_for"]

# What a transport program is allowed to see. `hermes send` finds the gateway's bot
# credential through HOME, and nothing in this list is a secret the memory process holds.
_TRANSPORT_ENV = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONIOENCODING")

# Destinations that are not a private channel to one person. ``bot-chat`` is refused on
# the plan's own instruction: it creates an inbound agent turn, so "delivering" to it
# would spend model calls the feature is supposed to avoid.
_PUBLIC = frozenset({"bot-chat", "all", "everyone", "broadcast", "public", "any"})
_KINDS = ("next_turn", "digest", "notify_owner")


@dataclass(frozen=True)
class DeliveryPolicy:
    """Whether this installation may hand anything to a transport, and to which one."""

    enabled: bool
    destination: str | None
    owner_principal: str | None
    kinds: tuple[str, ...] = _KINDS

    @classmethod
    def from_settings(cls, settings: Any) -> "DeliveryPolicy":
        return cls(enabled=bool(getattr(settings, "delivery_enabled", False)),
                   destination=getattr(settings, "delivery_target", None),
                   owner_principal=getattr(settings, "owner_principal", None))

    def refusal(self) -> str | None:
        """Why nothing may be delivered, or None when it may.

        Checked on every call rather than once at import: the answer is a property of
        the owner's decision, and an installation can lose it (by retiring the owner
        principal) without restarting anything.
        """
        if not self.enabled:
            return "provider-initiated delivery is disabled by default"
        target = (self.destination or "").strip()
        if not target:
            return ("no approved destination is configured "
                    "(set HERMES_MEMORY_DELIVERY_TARGET)")
        scheme, separator, address = target.partition(":")
        if not separator or not address.strip():
            return (f"{target!r} is not a concrete destination; a delivery target names "
                    "one private channel, like 'signal:1234' or 'email:me@example'")
        if scheme.strip().lower() in _PUBLIC:
            return f"{scheme!r} is not a private destination for one owner"
        if not self.owner_principal:
            return ("an owner principal is required before anything is delivered on "
                    "someone's behalf")
        return None

    def accepts(self, artifact: Any) -> str | None:
        """Why this artifact may not go out, or None when it may.

        The recipient check is the load-bearing one. Proactivity in this framework is
        notify-and-draft: everything in the outbox is addressed to the owner, and a
        transport that would happily message a third party is a different feature with
        different consent, not this one with a different string in it.
        """
        if artifact.kind not in self.kinds:
            return (f"a {artifact.kind!r} artifact is drafted for the owner to act on, "
                    "not for a transport to send")
        if str(artifact.recipient or "").strip() != (self.owner_principal or "").strip():
            return ("the artifact is addressed to someone other than the owner; this "
                    "installation notifies its owner and drafts for their approval, it "
                    "does not message anybody else")
        return None


def bounded_drain(limit: Any) -> int:
    """Check a drain's bound before anything is opened.

    A caller that asks for an unbounded run is a bug in the caller, and finding out after
    the profile has been resolved would make it a bug that touches somebody's archive.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 25:
        raise ValueError("limit must be between 1 and 25 artifacts")
    return limit


def correlation(artifact: Any) -> str:
    """The opaque marker a transport can echo back and be believed.

    It carries the artifact id and the prefix of the payload digest the outbox already
    holds, so a receipt can be matched to the thing that was sent rather than to the
    time it happened to arrive.
    """
    return f"hermes-memory {artifact.id} {artifact.payload_digest[:12]}"


def deliver_once(outbox: Any, *, policy: DeliveryPolicy,
                 sink: Callable[[str], Any] | None = None,
                 holder: str = "hermes-memory-delivery", out: TextIO | None = None,
                 at: float | None = None) -> dict[str, Any]:
    """Try to deliver one ready artifact. Returns a report about the attempt.

    ``sink`` is the transport: it receives the rendered text and may return a receipt
    (a dict with ``digest``, or ``sent``), or None for "the script finished and that is
    all we know". With no sink the artifact is written to ``out`` instead, which is the
    contract-tests path and still records the same states.

    A caller that names no transport at all gets a ``ValueError``: that is a bug in the
    caller, not a fact about the queue.
    """
    if sink is None and out is None:
        # Delivering into nowhere and then recording a receipt would be the one
        # report worse than no delivery.
        raise ValueError("name a transport: pass a sink or a stream to write to")
    blocked = policy.refusal()
    if blocked:
        return {"ok": True, "delivered": False, "attempted": False, "reason": blocked}
    claim = outbox.lease(holder=holder, at=at)
    if claim is None:
        return {"ok": True, "delivered": False, "attempted": False,
                "reason": "nothing is ready to send right now"}
    artifact = claim.artifact
    mismatch = policy.accepts(artifact)
    if mismatch:
        # The lease already revalidated the artifact against the world; this is the
        # transport's own consent check, and it releases rather than suppresses
        # because the artifact itself is fine — it is this destination that is not.
        outbox.release(artifact_id=artifact.id, token=claim.token, reason=mismatch)
        return {"ok": False, "delivered": False, "attempted": False,
                "artifact": artifact.id, "reason": mismatch}

    body = f"{correlation(artifact)}\n{artifact.payload}"
    try:
        outbox.attempt(artifact_id=artifact.id, token=claim.token)
    except Exception as error:
        # The handover never began, so nothing can have left. Say so and stop: this
        # is the only path that may report a refusal without an uncertain state.
        return {"ok": False, "delivered": False, "attempted": False,
                "artifact": artifact.id,
                "reason": f"the attempt could not be recorded: {str(error)[:200]}"}

    try:
        if sink is None:
            out.write(body + "\n")
            out.flush()
            receipt: Any = None
        else:
            receipt = sink(body)
    except Exception as error:
        # Past the attempt we cannot prove the bytes stayed here, and a sink that
        # dies mid-write is exactly the crash this state machine was built for.
        settled = outbox.uncertain(artifact_id=artifact.id, token=claim.token,
                                   reason=f"the handover began and the transport then "
                                          f"failed: {str(error)[:200]}")
        return {"ok": False, "delivered": False, "attempted": True,
                "artifact": artifact.id, "state": settled.state,
                "reason": "the outcome of that send is not established; it will not be "
                          "replayed blind"}

    settled = outbox.confirm(artifact_id=artifact.id, token=claim.token,
                             proof=_proof(receipt))
    return {"ok": settled.state == "confirmed",
            "delivered": settled.state in ("confirmed", "accepted_unverified"),
            "attempted": True, "artifact": settled.id, "state": settled.state,
            "reason": settled.reason, "correlation": correlation(artifact)}


def deliver_ready(store, *, policy: DeliveryPolicy,
                  sink: Callable[[str], Any] | None = None, out: TextIO | None = None,
                  limit: int = 1, holder: str = "hermes-memory-delivery",
                  held: str | None = None, at: float | None = None) -> dict[str, Any]:
    """Drain up to ``limit`` of one memory's ready artifacts through one transport.

    ``store`` is an open store for one profile's archive; the caller owns it, because a
    drain is one act inside an installation rather than a reason to open a second one.
    ``held`` is an operator's standing instruction, passed as the refusal text so the
    report can say who is holding delivery. Bounding the run matters: a drain that
    empties an unbounded queue is a drain that can be interrupted half-sent.
    """
    bounded_drain(limit)
    for blocked in (policy.refusal(), held):
        if blocked:
            return {"ok": True, "delivered": 0, "asked": 0, "reason": blocked,
                    "reports": []}
    outbox = Outbox(store, policy=AttentionPolicy(store,
                                                  owner_principal=policy.owner_principal),
                    owner_principal=policy.owner_principal)
    reports = []
    for _ in range(limit):
        report = deliver_once(outbox, policy=policy, sink=sink, holder=holder,
                              out=out, at=at)
        reports.append(report)
        # Anything that did not reach the handover will not change by trying again in
        # the same run: the queue is empty, or this one is refused.
        if not report.get("attempted"):
            break
    sent = sum(1 for item in reports if item.get("delivered"))
    # The last report explains why the run stopped, which on a productive drain is "the queue
    # is now empty" — and a record that led with that would read as though nothing had been
    # sent at all. Say what went out first, then why it stopped.
    tail = reports[-1].get("reason") if reports else "nothing was tried"
    reason = f"{sent} artifact(s) reached the transport; {tail}" if sent else tail
    questions = _ask_queued(store, policy=policy, sink=sink, out=out, holder=holder,
                            limit=limit)
    # Questions ride the same hand as reminders and spend the same budget, so one drain can
    # report both halves of what the owner was interrupted with. A run that found nothing
    # being asked adds nothing to the sentence: the quiet drains are the ones whose
    # unchanged reason is what tells the operator the loop is still alive.
    if questions.get("sent") or questions.get("attempted"):
        reason += f"; {questions['reason']}"
    return {"ok": all(item.get("ok") for item in reports)
                    and bool(questions.get("ok", True)),
            "delivered": sent, "asked": int(questions.get("sent") or 0),
            "questions": questions, "reason": reason, "reports": reports}


def _ask_queued(store, *, policy: DeliveryPolicy, sink: Callable[[str], Any] | None,
                out: TextIO | None, holder: str, limit: int) -> dict[str, Any]:
    """Ask the owner every question this archive is holding that may now be asked.

    Living beside the outbox rather than inside it is what keeps a question from having to
    be a reminder: an inquiry is about a decision the archive already awaits, it has no due
    time, and answering it settles that decision through the same door the command line
    uses. What it shares with a reminder is the transport, the destination and the owner's
    daily limit on being interrupted.
    """
    from .inquiries import InquiryStore

    writer = sink if sink is not None else _stream_sink(out)
    if writer is None:
        return {"ok": True, "sent": 0, "attempted": False, "reports": [],
                "reason": "no transport was named to ask through"}
    inquiries = InquiryStore(store, owner_principal=policy.owner_principal)
    report = inquiries.send_next(sink=writer, destination=str(policy.destination),
                                 holder=holder, limit=limit)
    return {"ok": report.get("ok", True), "sent": int(report.get("sent") or 0),
            "attempted": any(item.get("attempted") for item in report.get("reports") or []),
            "reason": str(report.get("reason") or ""), "reports": report.get("reports") or []}


def _stream_sink(out: TextIO | None) -> Callable[[str], Any] | None:
    """A transport that prints the message, for a drain run with no program attached."""
    if out is None:
        return None

    def send(body: str) -> None:
        out.write(body + "\n")
        out.flush()

    return send


def local_sink(hermes_home: str | Path, *, directory: str = "memory/delivered"):
    """A transport that writes the message into the owner's own Hermes home.

    This is the sink for an installation with no messaging platform: the artifact lands
    as a readable file and nothing more. It reports what it did — the path, and the
    digest of the bytes that came back off disk — and not more than that, so the outbox
    settles it as ``accepted_unverified``. A local file is not a carrier's receipt, and
    calling it a confirmation is the failure this state machine exists to prevent.
    """
    root = Path(hermes_home).expanduser() / directory

    def send(body: str) -> dict[str, Any]:
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%f")
        path = root / f"{stamp}-{digest(body)[:12]}.md"
        path.write_text(body + "\n", encoding="utf-8")
        if path.read_text(encoding="utf-8") != body + "\n":
            # The write is done and did not come back as written. Raise rather than
            # report a message the owner may never actually be able to read.
            raise OSError(f"{path} did not read back as written")
        return {"sent": True, "path": str(path), "bytes": len(body.encode("utf-8")),
                "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}

    return send


def command_sink(command: str | Sequence[str], *, destination: str | None = None,
                 timeout_s: float = 90.0,
                 env_extra: Sequence[str] = ()) -> Callable[[str], Any]:
    """A transport that is somebody else's program: pipe the body in, read its answer out.

    `hermes send -t telegram:<chat-id> --json` is what this exists for. The gateway already
    holds the bot credential, and a second copy of it in the memory process would be a second
    way to leak it, so this hands the bytes to a argv — no shell, a scrubbed environment — and
    then believes only what that program reports back.

    The reported chat is checked against the approved destination. A transport that answers
    success for somewhere else has messaged a stranger on this installation's name, which is
    not a delivery to retry quietly; the raise below leaves the artifact uncertain on purpose.

    A command that names no program on disk is refused here, before anything is leased,
    because a misspelled transport would otherwise mark every reminder the owner ever asked
    for as uncertain without a single byte going out.
    """
    argv = shlex.split(command) if isinstance(command, str) else [str(item) for item in command]
    argv = [item for item in argv if item.strip()]
    if not argv:
        raise ValueError("a delivery command names a program to run")
    if shutil.which(argv[0]) is None and not Path(argv[0]).expanduser().is_file():
        raise ValueError(f"{argv[0]!r} is not a program this installation can run")
    _scheme, _separator, address = (destination or "").partition(":")
    address = address.strip()

    def send(body: str) -> dict[str, Any]:
        correlation_line = body.splitlines()[0] if body.strip() else ""
        finished = subprocess.run(argv, input=body.encode("utf-8"), capture_output=True,
                                  timeout=timeout_s,
                                  env={key: os.environ[key] for key in _TRANSPORT_ENV
                                       if key in os.environ
                                       and isinstance(os.environ.get(key), str)}
                                  | {name: os.environ[name] for name in env_extra
                                     if name.isidentifier()
                                     and isinstance(os.environ.get(name), str)})
        text = finished.stdout.decode("utf-8", "replace")
        if finished.returncode != 0:
            raise RuntimeError(f"{Path(argv[0]).name} exited {finished.returncode}: "
                               f"{finished.stderr.decode('utf-8', 'replace').strip()[:200]}")
        receipt: dict[str, Any] = {"sent": True}
        answered = _answered(text)
        if isinstance(answered, dict):
            reported = str(answered.get("chat_id") or "").strip()
            if address and reported and reported != address:
                raise RuntimeError(f"{Path(argv[0]).name} reported sending to {reported!r}, "
                                   f"which is not the approved destination {address!r}")
            for key in ("platform", "chat_id", "message_id"):
                if answered.get(key) is not None:
                    receipt[key] = answered[key]
            if not answered.get("success", True):
                raise RuntimeError(f"{Path(argv[0]).name} said it did not send: "
                                   f"{str(answered.get('error'))[:200]}")
        if address and not receipt.get("chat_id"):
            # Nothing names where it went, so the destination is an assumption. Say that
            # rather than recording a send to a chat nobody confirmed.
            receipt["destination_unconfirmed"] = True
        if correlation_line and correlation_line in text:
            receipt["correlation"] = correlation_line
            receipt["verified"] = True
        return receipt

    return send


def sink_for(settings, hermes_home: str | Path) -> tuple[Callable | None, str | None]:
    """The transport this installation's owner configured, and how to name it in a report.

    Returns ``(None, None)`` when the destination names a transport nobody provided — and
    the caller must then refuse rather than fall back to writing a file, because a local
    artifact is not the message the owner asked to be sent.
    """
    destination = str(getattr(settings, "delivery_target", None) or "")
    command = tuple(getattr(settings, "delivery_command", ()) or ())
    if command:
        return command_sink(command, destination=destination), " ".join(command)
    if destination.lower().startswith("local:"):
        home = Path(hermes_home)
        return local_sink(home), str(home / "memory" / "delivered")
    return None, None


def drain_installed(settings, *, limit: int = 10,
                    holder: str = "hermes-memory-runtime") -> dict[str, Any]:
    """Drain every enrolled profile's outbox through the transport its owner named.

    This is what makes a delivered reminder arrive without anybody typing a command: the
    runtime's own timer, not a section of the background pass — the pass takes the rule
    that it touches no transport, and starting a transport from inside it would be the
    quiet way to break that rule.

    One profile's failure is kept to that profile. A store that cannot be opened, or a
    transport that has gone missing since it was configured, must not stop the other
    memories on the machine from telling their owner things.
    """
    from ..install.profiles import InstallationError, ProfileRegistry
    from ..storage.evidence import EvidenceStore

    per_profile, delivering = [], 0
    try:
        registry = ProfileRegistry.reading(settings)
    except (InstallationError, OSError) as error:
        return {"ok": False, "at": now(), "delivered": 0, "profiles": [],
                "reason": str(error)[:200]}
    try:
        profiles = list(registry.profiles())
    finally:
        registry.db.close()
    for item in profiles:
        scoped = item.scoped(settings)
        policy = DeliveryPolicy.from_settings(scoped)
        summary = {"profile": item.profile, "delivered": 0, "transport": None}
        if policy.refusal() is None:
            try:
                sink, transport = sink_for(scoped, item.hermes_home)
                with EvidenceStore(scoped.db_path) as store:
                    if sink is None:
                        # The owner approved an address this installation has no way to
                        # reach. Not sending is right; saying so once, in the archive it
                        # belongs to, is what makes it findable.
                        summary["reason"] = (f"{policy.destination!r} names no local "
                                             "destination and no transport program is "
                                             "configured")
                    else:
                        held = ("delivery is paused for this installation by the owner"
                                if store.stage_is_paused("global", "delivery") else None)
                        outcome = deliver_ready(store, policy=policy, sink=sink,
                                                limit=limit, held=held, holder=holder)
                        summary.update({"delivered": outcome["delivered"],
                                        "transport": transport,
                                        "reason": outcome["reason"]})
                        delivering += outcome["delivered"]
                    _note(store, summary)
            except (EvidenceError, InstallationError, ValueError, OSError) as error:
                # A drain that cannot be started is a fact about this installation, and the
                # loop that found it is still healthy: keep going, but say who else is waiting.
                summary["reason"] = f"{type(error).__name__}: {str(error)[:180]}"
        else:
            summary["reason"] = policy.refusal()
        per_profile.append(summary)
    return {"ok": True, "at": now(), "delivered": delivering, "profiles": per_profile}


def _changed(line: dict[str, Any], previous: dict[str, Any] | None) -> bool:
    """Whether this drain has anything the last recorded one did not already say."""
    if previous is None:
        return True
    key = ("delivered", "reason")
    return any(line.get(name) != previous.get(name) for name in key)


def _note(store, summary: dict[str, Any]) -> None:
    """Write the drain down, but only when it changed the story.

    `status` and `doctor` read this from another process, so without a line the difference
    between a loop that is running and one that stopped is invisible. Writing every period
    would bury the audit in no-ops — a minute-long poll is 1440 identical lines a day — so
    what gets recorded is each *change*: the first quiet drain, then the first one that sent
    something, then the first that met an error. A loop that has nothing to say since
    Tuesday is told by the backlog reading, which counts prepared artifacts rather than
    trusting this one.
    """
    line = {"profile": summary.get("profile"), "delivered": summary.get("delivered"),
            "transport": summary.get("transport"),
            "reason": str(summary.get("reason") or "")[:200]}
    previous = last_drain(store)
    if not _changed(line, previous):
        return
    store._audit("delivery_drain", f"delivery:{line['profile']}", line)


def last_drain(store) -> dict[str, Any] | None:
    """The last thing this archive's drain loop had to say, or None if it never said one."""
    row = store.db.execute(
        "SELECT created_at, metadata FROM audit WHERE action=? ORDER BY created_at DESC, "
        "rowid DESC LIMIT 1", ("delivery_drain",)).fetchone()
    if row is None:
        return None
    return {"created_at": str(row["created_at"]), **json.loads(str(row["metadata"] or "{}"))}


def drain_line(report: dict[str, Any]) -> dict[str, Any]:
    """The line the runtime keeps in memory for one drain, shaped for `state()`."""
    return {"ok": bool(report.get("ok")), "at": report.get("at"),
            "delivered": report.get("delivered"),
            "profiles": [{"profile": item.get("profile"),
                          "delivered": item.get("delivered"),
                          "reason": str(item.get("reason") or "")[:200]}
                         for item in report.get("profiles") or []]}


def _answered(text: str) -> Any:
    """The one JSON object a transport program answered with, if it answered with one.

    Read the whole stdout first: `hermes send --json` pretty-prints, so the last line is a
    closing brace, and a parser that only looks there finds no receipt and records a real
    send as though the transport had said nothing at all.
    """
    candidate = text.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    start, stop = candidate.find("{"), candidate.rfind("}")
    if start >= 0 and stop > start:
        try:
            return json.loads(candidate[start:stop + 1])
        except ValueError:
            return None
    return None


def _proof(receipt: Any) -> Any:
    if receipt is None:
        return None
    if isinstance(receipt, dict):
        return receipt
    if isinstance(receipt, str) and receipt.strip().startswith("{"):
        try:
            return json.loads(receipt)
        except ValueError:
            return {"sent": True, "echoed": receipt[:200]}
    return {"sent": True, "echoed": str(receipt)[:200]}
