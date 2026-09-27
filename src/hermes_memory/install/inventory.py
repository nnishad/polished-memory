"""What is actually on this machine, read without changing anything.

Setup, upgrade and uninstall all need the same answer to the same question first: what
is already here? An installer that guesses creates the two failures this design is
built to avoid — a second service on a port something else is holding, and a second
capture owner for a profile that already has one.

So every fact here is read, never asked for. Nothing connects to a socket, nothing runs
a model, nothing probes an endpoint and nothing ingests private data. Where a fact
cannot be determined it says ``unknown`` rather than guessing, because a plan built on a
guessed fact is an interruption waiting to happen at the worst possible step.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..config import DEFAULT_ENV_FILENAME, env_file_values

__all__ = ["survey", "listening_ports", "provider_selection", "conflicts"]

ENV_PREFIX = "HERMES_MEMORY_"
_HEX_PORT = re.compile(r"^[0-9A-Fa-f]{4}$")
LISTEN = "0A"
# Where `backup` writes its copies, relative to a profile's data directory. Named here because
# the search for a second capture owner has to look straight through it.
SNAPSHOT_DIR = "snapshots"
# A scalar the host would have written starts with none of these.
_NOT_SCALAR = frozenset("[{&*|>-?")
# The answers an inventory gives when the fact is simply not there to be read. They
# are listed so a plan can refuse to treat one of them as a value it can act on.
UNANSWERABLE = frozenset({"unknown", "unreadable", "no config.yaml to read"})


def survey(settings, *, hermes_home: str | Path | None = None,
           environ: dict[str, str] | None = None,
           proc: Path | None = Path("/proc")) -> dict[str, Any]:
    """The installation, the host, the ports and the disk, as one plain report."""
    environment = os.environ if environ is None else environ
    home = Path(hermes_home).expanduser() if hermes_home else None
    installation = _installation(settings, environment)
    host = _host(home, environment)
    endpoints = _endpoints(settings, proc=proc)
    disk = _disk(settings)
    unknowns = []
    if host["memory_provider"] in UNANSWERABLE:
        unknowns.append(f"the host's memory provider selection ({host['memory_provider']})")
    if host["hermes_version"] == "unknown":
        unknowns.append("no installed Hermes distribution to read a version from")
    if not endpoints["proc_readable"]:
        unknowns.append("the listening-port table, so no port collision has been ruled out")
    if endpoints["unattributed"]:
        unknowns.append("which process holds the "
                        f"{', '.join(endpoints['unattributed'])} port(s), so a stranger's "
                        "listener cannot be told apart from this installation's own")
    if disk["free_bytes"] is None:
        unknowns.append("free disk space beside " + disk["measured_against"])
    return {"installation": installation, "release": _release(settings, environment),
            "host": host,
            "endpoints": endpoints, "disk": disk,
            "capture_owners": _capture_owners(settings, home=home), "unknowns": unknowns}


def _installation(settings, environment: dict[str, str]) -> dict[str, Any]:
    env_file = Path(settings.home) / DEFAULT_ENV_FILENAME
    ledger = Path(settings.home) / "installation.db"
    values = env_file_values(env_file)
    # A key in the *process* environment is the designed arrangement: the config file
    # names the variable and the secret lives outside the file. A key inside the config
    # file is a copy of a secret at rest, which is what the naming convention exists to
    # prevent.
    at_rest = sorted(key for key in values if key.endswith("_API_KEY"))
    return {
        "home": str(settings.home),
        "config_file": str(env_file),
        "config_present": env_file.is_file(),
        "config_mode": _mode(env_file),
        "world_readable": bool(env_file.stat().st_mode & 0o077) if env_file.is_file() else False,
        "data_dir": str(settings.data_dir),
        "data_dir_present": settings.data_dir.is_dir(),
        "store_present": settings.db_path.is_file(),
        "blobs_present": settings.blob_dir.is_dir(),
        "instance_ledger": str(ledger),
        # File existence only. Counting rows would mean opening the ledger, and an
        # inventory that reads a ledger counts it as present, not as understood —
        # `hermes-memory profiles` is the command that answers how many.
        "ledger_present": ledger.is_file(),
        "inference_enabled": settings.inference_enabled,
        "capture_only": settings.capture_only,
        "background_budget_tokens": settings.background_budget_tokens,
        "owner_principal": settings.owner_principal or "unset",
        "delivery": "enabled" if settings.delivery_enabled else "disabled",
        "delivery_target": _target_shape(settings.delivery_target),
        "credential_variables_named": sorted(
            value for value in [settings.hindsight_api_key_env] if value),
        "secrets_in_environment": sorted(
            key for key in environment
            if key.startswith(ENV_PREFIX) and key.endswith("_API_KEY")),
        "secrets_at_rest_in_config": at_rest,
    }


def _target_shape(target: str | None) -> str:
    """The kind of destination, never the whole address.

    A status report is not the place to print a person's phone number; the scheme is
    enough to tell an operator whether the right channel was chosen.
    """
    if not target:
        return "unset"
    scheme, separator, address = target.partition(":")
    return f"{scheme}:{'set' if address.strip() else 'EMPTY'}" if separator else "malformed"


def _release(settings, environment: dict[str, str]) -> dict[str, Any]:
    from .compatibility import tree_root

    tree = tree_root()
    return {
        "framework_version": _version("hermes-memory"),
        # Named as an absence when there is no tree: an installed build whose plugin is not
        # beside it is a fact the operator needs, not a path invented from parent directories.
        "package_source": str(tree) if tree is not None else "no tree beside this install",
        # The units name this path; whether the code answering is the code they start is
        # a question an operator asks at exactly the moment it is expensive to get wrong.
        "release_root": environment.get("HERMES_MEMORY_RELEASE", "not set"),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "uv": shutil.which("uv") or "not found",
        "hindsight_client": _version("hindsight-client") or "not importable",
        "runtime_root": str(Path(settings.home) / "runtime"),
    }


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _host(home: Path | None, environment: dict[str, str]) -> dict[str, Any]:
    """What can be said about Hermes without running it.

    The version comes from installed metadata, not ``hermes --version``: shelling out
    from an inventory would make a read-only report depend on whatever the host
    happens to do when asked about itself.
    """
    root = home or Path(environment.get("HERMES_HOME", Path.home() / ".hermes"))
    config = root / "config.yaml"
    stamp = root / ".install_method"
    provider_config = root / "hermes-memory" / "config.json"
    return {
        "hermes_home": str(root),
        "hermes_executable": shutil.which("hermes") or "not found",
        "hermes_version": _version("hermes-agent") or _version("hermes") or "unknown",
        "install_method": stamp.read_text(encoding="utf-8").strip() if stamp.is_file() else "unknown",
        "config_file": str(config),
        "config_present": config.is_file(),
        "memory_provider": provider_selection(config) if config.is_file() else "no config.yaml to read",
        "plugin_config": str(provider_config),
        "plugin_config_present": provider_config.is_file(),
        "env_file": str(root / ".env"),
        # The host gap of §9.5: a queued body can still be sent after the framework
        # has revalidated it, so unattended live alerts are not claimable here.
        "guarded_delivery_supported": False,
    }


def provider_selection(config_path: Path) -> str:
    """The ``memory.provider`` value from a host config file, read narrowly.

    A whole YAML parser is not a dependency of this framework, and loading one to
    report one dotted key would mean evaluating a file owned by another program. So
    this recognises the one shape the host writes and says ``unknown`` for anything
    else — an uncertain answer is better than a confident wrong one, and the writer
    this informs is the host's own, never a rewrite of the file.
    """
    try:
        lines = Path(config_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unreadable"
    inside = False
    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, separator, value = line.strip().partition(":")
        if indent == 0:
            inside = bool(separator and key.strip() == "memory" and not value.strip())
            continue
        if inside and indent == 2 and key.strip() == "provider" and separator:
            found = value.strip().strip('"').strip("'")
            if not found:
                return "unset"
            # A value that opens with a YAML structural character is not the plain
            # scalar the host writes, and pretending otherwise would have an installer
            # act on a fragment of a collection.
            if found[0] in _NOT_SCALAR or value.strip()[:1] in ("|", ">", "&", "*"):
                return "unknown"
            return found
    return "unknown"


def _endpoints(settings, *, proc: Path | None) -> dict[str, Any]:
    """Configured bind points against what is already listening. Local reads only.

    `held_by_ours` names the wanted ports whose every listener is a process of this
    installation itself — the difference between restarting this installation and arguing with a
    stranger over one port, and the fact that lets a re-run against a running machine be a
    restart rather than a refusal. It lives outside `wanted`, whose shape the approval digest is
    taken over: which process happened to be up when the plan was read is not a change to what
    was approved.
    """
    sockets = _socket_table(proc)
    wanted = _configured_ports(settings)
    ours: dict[str, Any] = {}
    unnamed: list[str] = []
    for name, port in wanted.items():
        if not port or port not in sockets:
            continue
        holders = [_listener(proc, inode, home=str(settings.home))
                   for inode in sockets[port]]
        # Every socket on the port has to be ours to call it ours: a service listening on
        # v4 does not speak for whatever is on v6, and a plan that asked only "who is on
        # this port" would restart into a bind failure it had just approved.
        if all(holder is not None and holder["ours"] for holder in holders):
            ours[name] = {"pid": holders[0]["pid"], "program": holders[0]["program"]}
        elif any(holder is None for holder in holders):
            unnamed.append(name)
    return {
        "listening_ports": sorted(sockets),
        "proc_readable": bool(sockets),
        "wanted": {str(name): {"port": port,
                               "in_use": port in sockets if port else None}
                   for name, port in wanted.items()},
        "held_by_ours": ours,
        "unattributed": sorted(unnamed),
        "routes": [{"work": name, "base_url": route.base_url, "resource": route.resource,
                    "model": route.model or "engine default"}
                   for name, route in (("text", settings.text_route),
                                       ("vision", settings.vision_route),
                                       ("embeddings", settings.embeddings_route))
                   if route is not None],
        "probed": False,
    }


def _configured_ports(settings) -> dict[str, int | None]:
    """The bind points this installation names, whether or not anything is on them.

    Both belong to the installation rather than to any one Hermes profile, and both are
    read from its own configuration file: the shell running the report is a different
    system's opinion of what is configured.
    """
    return {"hindsight": _port_of(settings.hindsight_url),
            "admission": _port_of(admission_url(settings))}


def admission_url(settings) -> str | None:
    values = env_file_values(Path(settings.home) / DEFAULT_ENV_FILENAME)
    return values.get("HERMES_MEMORY_ADMISSION_URL")


def _port_of(url: str | None) -> int | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if parsed.port:
            return int(parsed.port)
        return 443 if parsed.scheme == "https" else 80
    except ValueError:
        return None


def listening_ports(proc: Path | None) -> set[int]:
    """Ports this machine is listening on, read from /proc without opening a socket."""
    return set(_socket_table(proc))


def _socket_table(proc: Path | None) -> dict[int, list[str]]:
    """Every listening socket, as port to the inodes that name their owners. A /proc read.

    Parsing hex is dull and deliberate: binding a probe port to check for a collision is
    exactly how an installer takes a port away from something that was using it. The inodes
    are kept because a port says only that something is there, while they say what — and one
    port can have more than one something, one per address family.
    """
    found: dict[int, list[str]] = {}
    if proc is None:
        return found
    for table in ("net/tcp", "net/tcp6"):
        path = Path(proc) / table
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            columns = line.split()
            if len(columns) < 4 or columns[3] != LISTEN:
                continue
            _, separator, port = columns[1].rpartition(":")
            if _HEX_PORT.fullmatch(port):
                found.setdefault(int(port, 16), []).append(
                    columns[9] if len(columns) > 9 else "")
    return found


def _listener(proc: Path | None, inode: str, *, home: str) -> dict[str, Any] | None:
    """The process holding a listening socket, and whether it is this installation's own.

    A process's `cmdline` is readable for every process on the machine and carries no secret, so
    the owner is found by a file read rather than by asking the service manager to run anything —
    the inventory runs no commands. An installation's own units carry their home in their command
    line, and that is the comparison that matters: a stranger on the same port is not a listener
    this run may replace. When nothing can be named, the port is still somebody's.
    """
    if proc is None or not inode:
        return None
    wanted = f"socket:[{inode}]"
    try:
        pids = sorted((path for path in Path(proc).iterdir() if path.name.isdigit()),
                      key=lambda path: int(path.name))
    except OSError:
        return None
    for pid in pids:
        try:
            descriptors = list((pid / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                held = os.readlink(descriptor)
            except OSError:
                continue
            if held != wanted:
                continue
            argv = _arguments(pid)
            command = " ".join(argv)
            return {"pid": pid.name, "program": _program(argv),
                    "ours": bool(command) and home in command}
    return None


def _arguments(pid: Path) -> list[str]:
    try:
        raw = (pid / "cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


# A shebang makes the kernel report the interpreter first, and naming it answers nothing about
# who owns a port: the unit says which program it was written to run, one argument further along.
_INTERPRETERS = frozenset({"python", "python3", "pythonw", "pypy3", "sh", "bash", "zsh", "dash"})


def _program(argv: list[str]) -> str:
    for argument in argv:
        name = argument.rsplit("/", 1)[-1]
        if argument.startswith("/") and name not in _INTERPRETERS:
            return name
    return argv[0].rsplit("/", 1)[-1] if argv else "unknown"


def _capture_owners(settings, *, home: Path | None) -> dict[str, Any]:
    """Every spool and canonical store visible under the homes this installation owns.

    Two capture owners for one profile is the failure that silently duplicates or
    loses turns, and it is nearly always the residue of an earlier install. Only the
    homes this report was asked about are searched: an inventory that walked the whole
    filesystem would be both slow and nosy.
    """
    # Bounded on purpose: the provider's own two locations, not the whole tree. A
    # recursive walk of a profile home would make an inventory the slowest thing on the
    # machine, and the residue it is looking for always lands in one of these two.
    searched: list[Path] = [Path(settings.data_dir)]
    if home is not None:
        searched.append(Path(home) / "hermes-memory")
    seen: dict[str, list[str]] = {"canonical.db": [], "capture-spool.db": []}
    for root in searched:
        if not root.is_dir():
            continue
        for name in seen:
            for path in sorted(root.rglob(name)):
                # A snapshot is a copy this installation took of a store it already counted, and
                # the pre-restore guard is a copy of one of those. Counting them as owners made
                # `backup` look like a second capture owner, and made the next backup expire the
                # approval of a plan whose steps had not moved.
                if SNAPSHOT_DIR in path.relative_to(root).parts[:-1]:
                    continue
                seen[name].append(str(path))
    return {"searched": [str(root) for root in searched],
            "canonical_stores": sorted(set(seen["canonical.db"])),
            "capture_spools": sorted(set(seen["capture-spool.db"]))}


def _disk(settings) -> dict[str, Any]:
    target = Path(settings.data_dir).parent
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return {"measured_against": str(target), "free_bytes": None, "total_bytes": None}
    return {"measured_against": str(target), "free_bytes": usage.free,
            "total_bytes": usage.total}


def _mode(path: Path) -> str:
    try:
        return oct(path.stat().st_mode & 0o777)
    except OSError:
        return "unreadable"


def conflicts(report: dict[str, Any]) -> list[str]:
    """The reasons a setup run would be a bad idea right now, in plain sentences.

    Ordered deliberately: a port or a second capture owner blocks the run, a loose
    permission on the configuration file is a warning the operator should fix now, and
    a missing owner principal only blocks forgetting.
    """
    say: list[str] = []
    ours = report["endpoints"].get("held_by_ours") or {}
    for name, entry in sorted(report["endpoints"]["wanted"].items()):
        if entry["port"] and entry["in_use"]:
            if name in ours:
                hold = ours[name]
                say.append(f"the {name} endpoint's port {entry['port']} is held by this "
                           f"installation's own {hold['program']} (PID {hold['pid']}); a setup "
                           "run here replaces that listener rather than colliding with it")
            else:
                say.append(f"the {name} endpoint wants port {entry['port']}, which something "
                           "is already listening on")
    stores = [path for path in report["capture_owners"]["canonical_stores"]]
    if len(stores) > 1:
        say.append(f"{len(stores)} canonical stores are visible under the searched homes "
                   f"({', '.join(stores)}); one profile has one capture owner")
    if report["installation"]["world_readable"]:
        say.append(f"{report['installation']['config_file']} is readable beyond its owner "
                   "and holds the route configuration")
    if report["installation"]["secrets_at_rest_in_config"]:
        say.append(f"{report['installation']['config_file']} holds a credential value "
                   f"({', '.join(report['installation']['secrets_at_rest_in_config'])}) "
                   "where it should name the variable that holds it")
    if report["host"]["hermes_executable"] == "not found":
        say.append("no hermes executable is on PATH, so plugin registration and "
                   "activation cannot be verified from here")
    if report["installation"]["owner_principal"] == "unset":
        say.append("no owner principal is configured, so forgetting can be previewed "
                   "but never confirmed")
    if (report["installation"]["delivery"] == "enabled"
            and report["installation"]["delivery_target"] in ("unset", "malformed")):
        say.append("delivery is switched on with no usable destination to send to")
    return say
