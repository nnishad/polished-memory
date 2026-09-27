"""C14: the release's own answer to "what was this composed against" (§10.3).

A compatibility manifest is only worth shipping if it cannot drift. So every value here is
read from the code or the artefact that actually enforces it — the pinned version from the
capability table, the schema version from the migration list, the plugin's minimum host from
its own manifest, the checkpoint API from the provider that has to satisfy it.

`verify()` compares the shipped `deployment/compatibility.json` against what this build
claims. It checks the *compatibility claims* only, and not the source digests: a tree under
development changes every minute, and a doctor that failed on every edit would be a doctor
nobody runs. The digests are still written, still reported, and compared by
`verify(digests=True)`, which is what a packaging step calls before it cuts a release.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ..backend.capabilities import CAPABILITIES, PINNED_VERSION
from ..ids import content_digest
from ..storage.migrations import MIGRATIONS

__all__ = ["MANIFEST_VERSION", "CompatibilityUnavailable", "facts", "manifest_path",
           "read_shipped", "source_checkout", "tree_for", "tree_root", "verify", "write"]

MANIFEST_VERSION = "compatibility-v2"


class CompatibilityUnavailable(ValueError):
    """This build is installed with no tree beside it, so there is nothing to claim.

    Distinct from a divergence: a divergence is two statements that disagree, and this is the
    absence of one. A bare wheel install is the ordinary case, and it has to be said plainly
    rather than raised as a missing file three directories above a site-packages parent.
    """


REPO = Path(__file__).resolve().parents[3]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
SHIP_AT = REPO / "deployment" / "compatibility.json"
PLUGIN_DIR = REPO / "integrations" / "hermes-memory"
RELATIVE = Path("deployment") / "compatibility.json"
#: What makes a directory the tree this build belongs to. A source checkout has both, a
#: staged release carries both beside the ``bin/``, and a bare wheel carries neither — which
#: is a real answer, not a failure to find the right one.
TREE_MARKERS = (RELATIVE, Path("integrations") / "hermes-memory" / "provider.py")

_TOP_LEVEL = re.compile(r"^([a-z_]+):\s*(.*)$")
_NUMBERED = re.compile(r"^([A-Z_]+)\s*=\s*(\d+)\s*$", re.MULTILINE)

# The claims a running build can be asked about. The digests are excluded on purpose: they
# name *which* code this is, and a doctor that went red on every save tells nobody anything.
_CLAIMS = ("manifest_version", "package", "hindsight", "schema", "plugin")
_DIGESTS = ("framework_digest",)


def _scalar(path: Path, key: str) -> Any:
    """One top-level ``key: value`` from a flat manifest, quotes and all stripped."""
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _TOP_LEVEL.match(line)
        if match and match.group(1) == key:
            value = match.group(2).strip().strip("\"'")
            return [] if value == "[]" else value
    raise ValueError(f"{path.name} declares no top-level {key!r}")


def _package(root: Path) -> dict[str, Any]:
    """The framework's own name, version and Python floor.

    Read from ``pyproject.toml`` in a checkout, where that file is the truth being packaged,
    and from the installed distribution's metadata otherwise — an installed tree has no
    ``pyproject.toml`` at all, and the parents of its ``site-packages`` are not a repository.
    """
    declaration = root / "pyproject.toml"
    if declaration.is_file():
        project = declaration.read_text(encoding="utf-8").split("[project]", 1)[1] \
            .split("[", 1)[0]
        values = dict(re.findall(r'(?m)^(\w[\w-]*)\s*=\s*"([^"]*)"', project))
        return {"package": values["name"], "version": values["version"],
                "requires_python": values["requires-python"]}
    from importlib import metadata

    dist = metadata.distribution("hermes-memory")
    return {"package": dist.metadata["Name"], "version": dist.version,
            "requires_python": (dist.metadata["Requires-Python"] or "").strip()}


def _plugin(root: Path) -> dict[str, Any]:
    """What the plugin claims about the host, plus the code that has to keep the claim.

    Read out of the tree this build belongs to, because the half that registers with the host
    travels beside the half that answers: a release whose ``integrations/`` disagrees with its
    ``site-packages/`` is the drift this section exists to catch.
    """
    directory = root / "integrations" / "hermes-memory"
    provider = (directory / "provider.py").read_text(encoding="utf-8")
    numbers = dict(_NUMBERED.findall(provider))
    manifest = directory / "plugin.yaml"
    return {"name": _scalar(manifest, "name"), "version": _scalar(manifest, "version"),
            "kind": _scalar(manifest, "kind"),
            "requires_hermes": _scalar(manifest, "requires_hermes"),
            "python_dependencies": _scalar(manifest, "python_dependencies"),
            # Read from the provider rather than repeated here: a manifest that said "2"
            # while the code said "3" is the second source of truth this file exists to
            # remove.
            "checkpoint_api_version": int(numbers["CHECKPOINT_API_VERSION"]),
            "files": {path.name: content_digest(path.read_bytes())
                      for path in sorted(directory.glob("*.py"))}}


def _hindsight() -> dict[str, Any]:
    return {"engine_pinned": PINNED_VERSION,
            "client_floor": "hindsight-client>=0.6.1",
            "operations": [{"name": cap.name, "method": cap.method, "path": cap.path,
                            "since": cap.since} for cap in CAPABILITIES]}


def _schema() -> dict[str, Any]:
    """What this build can read, in both directions.

    Older stores are migrated forward; a store written by a newer build is refused rather
    than read with an older understanding of it. Both halves are stated because a range that
    names only the current number implies the other one.
    """
    from ..processing.instance_gate import GATE_SCHEMA_VERSION

    return {"evidence_migrations": len(MIGRATIONS),
            "evidence_older": "migrated forward on open",
            "evidence_newer": "refused",
            "gate_ledger_version": GATE_SCHEMA_VERSION}


def _framework_digest() -> str:
    """One digest over the framework's own modules, name and content both.

    Two trees that share it are the same code, and nothing else about a release has to be
    believed. It is taken from the imported package rather than from ``<root>/src`` so that a
    checkout and the release built from it answer with the same value: the path each module
    reports is the path it is imported by, which survives packaging. A digest that only
    computed in a working tree would make every installed build look like a different one.
    """
    sources = sorted(PACKAGE_DIR.rglob("*.py"))
    return content_digest(json.dumps([[path.relative_to(PACKAGE_DIR).as_posix(),
                                       content_digest(path.read_bytes())]
                                      for path in sources]).encode())


def tree_root(*, origin: Path | None = None) -> Path | None:
    """The tree this build belongs to: a source checkout, or the release that carries it.

    Answered by walking up from the imported module rather than by counting parents, because
    the depth differs between the two layouts — ``src/hermes_memory/install`` in a checkout and
    ``lib/python3.12/site-packages/hermes_memory/install`` in a venv — and a hardcoded count is
    how an installed build ends up reading ``lib/python3.12/pyproject.toml`` and crashing.
    """
    checkout = source_checkout()
    if checkout is not None:
        return checkout
    start = (origin or PACKAGE_DIR).resolve()
    for parent in start.parents:
        if all((parent / marker).is_file() for marker in TREE_MARKERS):
            return parent
    return None


def facts(*, root: Path | None = None) -> dict[str, Any]:
    """The manifest's contents, computed from the tree it describes."""
    resolved = root if root is not None else tree_root()
    if resolved is None:
        raise CompatibilityUnavailable(
            "this build is installed without a tree beside it: no deployment manifest and no "
            "plugin files were shipped, so there is nothing here to state compatibility with")
    return {"manifest_version": MANIFEST_VERSION, "package": _package(resolved),
            "hindsight": _hindsight(), "schema": _schema(),
            "plugin": _plugin(resolved),
            "framework_digest": _framework_digest(),
            "not_claimed": ["no live Hindsight request was made to build this",
                            "no model was called, and no route quality is asserted here",
                            "host compatibility is the plugin's own floor, not a tested list"]}


def source_checkout() -> Path | None:
    """The working tree this module was imported from, when it is one.

    A package installed into a venv has no checkout either side of it: the two
    directories that would have to be present are what makes a tree a source tree
    rather than a site-packages parent.
    """
    return REPO if all((REPO / part).exists() for part in ("pyproject.toml",
                                                           "src/hermes_memory")) else None


def manifest_path(settings=None, environ: dict[str, str] | None = None) -> Path | None:
    """Where this release's manifest lives: the release tree, then this checkout.

    An installed tree carries ``deployment/compatibility.json`` beside the ``bin/`` it was
    named by, which is ``HERMES_MEMORY_RELEASE`` or the ``runtime/current`` pointer under the
    instance home — the same two answers the service layout reads. Falling back to the tree
    this module was imported from keeps a source run honest without pretending to be a
    release: a release tree that ships no manifest of its own is not vouched for by whichever
    checkout happens to be on the disk. ``None`` is an answer — this build carries no
    statement of what it is compatible with — and the checks say that rather than naming a
    path that never existed.
    """
    source = os.environ if environ is None else environ
    roots = []
    release = str(source.get("HERMES_MEMORY_RELEASE") or "")
    if release:
        roots.append(Path(release))
    if settings is not None:
        roots.append(Path(settings.home) / "runtime" / "current")
    for root in roots:
        candidate = root / RELATIVE
        if candidate.is_file():
            return candidate
    checkout = source_checkout()
    return checkout / RELATIVE if checkout else None


def write(*, path: Path | None = None, settings=None,
          environ: dict[str, str] | None = None) -> Path:
    """Regenerate the manifest, for packaging.

    The target is chosen by convention, not by which files already exist: writing a manifest
    is how one comes to exist. ``HERMES_MEMORY_RELEASE`` names the release being packed, the
    instance home's ``runtime/current`` names it otherwise, and a source run falls to the
    checkout — which is the only place a manifest may be invented rather than copied.
    """
    source = os.environ if environ is None else environ
    target = path
    if target is None:
        release = str(source.get("HERMES_MEMORY_RELEASE") or "")
        pointer = Path(settings.home) / "runtime" / "current" if settings is not None else None
        if release:
            target = Path(release) / RELATIVE
        elif pointer is not None and (pointer / "deployment").is_dir():
            target = pointer / RELATIVE
        else:
            target = SHIP_AT
    root = tree_for(target)
    if root is None:
        raise CompatibilityUnavailable(
            "`compatibility --write` needs a tree to describe: this build is installed with "
            "neither a checkout nor a release carrying its plugin and units")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(facts(root=root), indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


def read_shipped(*, path: Path | None = None, settings=None) -> dict[str, Any]:
    target = path or manifest_path(settings)
    if target is None or not target.is_file():
        raise ValueError(f"no compatibility manifest at {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def tree_for(manifest: Path | None = None) -> Path | None:
    """Which tree a check is about: the one a named manifest shipped inside, else this build's.

    Two different questions arrive here. A release being verified asks "does the file beside
    this runtime still describe it", and the answer must come from that file's own tree even
    when the checking process was started somewhere else. A named path that is not inside any
    tree — a copy of a manifest on its own — is not a tree, and inventing one from its
    grandparents would compare a file against a directory that was never a release.
    """
    if manifest is not None:
        beside = Path(manifest).resolve().parent.parent
        if all((beside / marker).is_file() for marker in TREE_MARKERS):
            return beside
    return tree_root()


def verify(*, path: Path | None = None, settings=None,
          digests: bool = False) -> dict[str, Any]:
    """Does the shipped file still claim what this build claims?"""
    target = path or manifest_path(settings)
    remedy = ("regenerate deployment/compatibility.json with `hermes-memory compatibility "
              "--write` as part of the release, because the code and its stated "
              "compatibility have already diverged")
    if target is None:
        # Not a divergence: this build says nothing about what it is compatible with, and
        # no fix on this machine can make it say so. A wheel installed on its own is the
        # ordinary case, so the remedy names the release tree rather than a write command.
        return {"ok": False, "absent": True, "checked": None,
                "differences": ["no compatibility manifest is carried by this installation, "
                                "so nothing here states what it is compatible with"],
                "remedy": ("run from a release tree, or name one with HERMES_MEMORY_RELEASE "
                           "(or <instance home>/runtime/current); `compatibility --write` "
                           "belongs to packaging, where the source is"),
                "digests": None}
    if not target.is_file():
        # Named and missing: this tree was supposed to carry the file and does not, which
        # is the one case where regenerating it is the operator's fix.
        return {"ok": False, "absent": False, "checked": str(target),
                "differences": [f"{target} does not exist"], "remedy": remedy,
                "digests": None}
    try:
        shipped = read_shipped(path=target)
    except ValueError as error:
        return {"ok": False, "checked": str(target),
                "differences": [f"{target} is not readable JSON: {str(error)[:160]}"],
                "remedy": remedy, "digests": None}
    try:
        current = facts(root=tree_for(target))
    except CompatibilityUnavailable as error:
        return {"ok": False, "absent": False, "checked": str(target),
                "differences": [str(error)],
                "remedy": ("stage the release rather than installing the wheel alone; "
                           "`hermes-memory release` carries the plugin and the units beside "
                           "the runtime, which is what makes a claim checkable"),
                "digests": None}
    compared = (*_CLAIMS, *_DIGESTS) if digests else _CLAIMS
    shipped_plugin = {key: value for key, value in (shipped.get("plugin") or {}).items()
                      if digests or key != "files"}
    current_plugin = {key: value for key, value in current["plugin"].items()
                      if digests or key != "files"}
    differences = []
    for key in compared:
        wanted = current_plugin if key == "plugin" else current[key]
        actual = shipped_plugin if key == "plugin" else shipped.get(key)
        if actual != wanted:
            differences.append(f"{key}: shipped {actual!r}, this build {wanted!r}")
    differences += [f"the shipped manifest claims compatibility this build does not: {key}"
                    for key in sorted(set(shipped) - set(current) - {"not_claimed"})]
    return {"ok": not differences, "checked": str(target), "differences": differences,
            "remedy": remedy if differences else None,
            "digests": {"framework": current["framework_digest"],
                        "shipped": shipped.get("framework_digest"),
                        "agree": current["framework_digest"] == shipped.get("framework_digest")}}
