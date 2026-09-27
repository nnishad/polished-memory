"""§10.3: staging a release is an operation with a door, not a runbook in someone's head.

The installer has always refused to invent an environment: `setup`'s `stage` step reports a
missing release rather than fetching one, and `upgrade` says outright that nothing there
unpacks a tree. Both were correct about the boundary and silent about who crosses it, which
left the whole installation gated on a manual sequence that existed in no test and could not
be checked by anything afterwards. This is that step: two environments built from one wheel,
from one revision, with the artefacts the runtime and the host both have to agree about
carried alongside them.

Three things are deliberate:

* **The pin is not repeated here.** The backend distribution is named by
  `backend.capabilities.PINNED_VERSION`, the same constant that decides which routes this
  build will send requests to. A release door that pinned separately could stage an engine
  the framework then refuses to talk to — or worse, stage one it talks to without having
  verified the contract.
* **It plans before it writes**, and applying needs the digest the plan printed. Pulling a
  couple of hundred packages off an index and dropping two interpreters onto a disk is not
  reversible by Ctrl-C, so the approval is for a stated tree, at a stated revision, with a
  stated set of packages.
* **It does not flip the pointer.** Pointing the units at a staged tree is the switch §10.6
  describes, and that one needs services paused and in-flight work reconciled first. This
  door produces a tree and proves it is a release; `runtime/current` keeps pointing where it
  pointed until the owner says otherwise.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping

from ..backend.capabilities import PINNED_VERSION
from ..ids import content_digest, digest
from .compatibility import source_checkout
from .services import executables, render

__all__ = ["ReleaseError", "plan", "apply", "verify", "carry", "BACKEND_SPEC",
           "MANIFEST", "CARRIED", "release_root"]

#: The backend this build is written against, as one spec string.
BACKEND_SPEC = f"hindsight-api-slim[embedded-db]=={PINNED_VERSION}"
#: What the staged tree names itself by, so a later reader can ask which revision this is.
MANIFEST = "RELEASE.json"
#: Directories copied out of the source tree. The units read one and the host registers the
#: other, so a release that carried only the Python would separate the two halves of one
#: contract revision — which is exactly what §10.3 forbids.
CARRIED = ("deployment", "integrations")


class ReleaseError(ValueError):
    """Refused rather than half-done."""


def release_root(settings) -> Path:
    """Where this instance's staged releases live."""
    return Path(settings.home) / "runtime"


def _python(into: Path, *, backend: bool = False) -> Path:
    return into / ("hindsight/bin/python" if backend else "bin/python")


def plan(*, settings, into: Path | str, source: Path | str | None = None,
         wheel: Path | str | None = None, environ: Mapping[str, str] | None = None,
         backend: bool = True) -> dict[str, Any]:
    """The actions this staging would take, and the digest an approval is taken against.

    Nothing is written and nothing is fetched: the wheel is located, not built, and a build
    happens only when the caller did not name one, so the plan can be shown before a
    compiler is asked to run.
    """
    target = Path(into).expanduser()
    root = Path(source).expanduser() if source else source_checkout()
    blocking: list[str] = []
    actions: list[str] = []

    if root is None or not (root / "pyproject.toml").is_file():
        blocking.append("no source checkout was found and none was named; `--source` has to "
                        "point at the tree whose deployment/ and integrations/ this release "
                        "carries")
        tree: dict[str, list[str]] = {}
        fingerprints: list[list[str]] = []
    else:
        tree = {name: sorted(str(path.relative_to(root))
                             for path in (root / name).rglob("*")
                             if path.is_file() and "__pycache__" not in path.parts)
                for name in CARRIED if (root / name).is_dir()}
        # Bytes, not names: the plugin and the service templates are the half of this release
        # the host and the units read, and an approval that survived editing them would be a
        # signature over a list of filenames.
        fingerprints = [[name, relative,
                         content_digest((root / relative).read_bytes())]
                        for name, files in sorted(tree.items()) for relative in files]
        missing = [name for name, files in tree.items() if not files]
        blocking += [f"{root / name} carries no files; this release would ship a runtime "
                     "whose service templates or plugin are missing" for name in missing]

    if target.exists() and any(target.iterdir()):
        blocking.append(f"{target} exists and is not empty; a release is staged side by side "
                        "and then switched, never merged into the one that is running")
    if target.is_symlink():
        blocking.append(f"{target} is a pointer, not a tree; staging through it would write "
                        "into whatever it happens to point at")

    distribution = Path(wheel).expanduser() if wheel else None
    if distribution is not None and not distribution.is_file():
        blocking.append(f"{distribution} is not a file")
    commit, dirty = _revision(root) if root is not None else (None, False)
    if commit:
        actions.append(f"stage commit {commit[:12]} of {root}")
    named = re.fullmatch(r"[0-9a-f]{40}", target.name)
    if named and commit and target.name != commit:
        blocking.append(f"{target} is named for commit {target.name[:12]}, but the source at "
                        f"{root} is at {commit[:12]}; a release tree that claims a revision it "
                        "does not hold is worse than one that claims none")
    if commit and dirty:
        blocking.append(f"{root} has uncommitted changes; the wheel would be built from bytes "
                        "no revision names, and the release would record a commit it is not. "
                        "Commit them, or stage from a clean checkout")
    actions.append(f"build the wheel from {root}" if distribution is None
                   else f"stage from {distribution}")
    actions.append(f"create the runtime environment at {target}")
    if backend:
        actions.append(f"create the backend environment at {target / 'hindsight'}")
        actions.append(f"install {BACKEND_SPEC} into it")
    else:
        actions.append("backend environment skipped on request: `setup` will then refuse to "
                       "stage as long as a backend route is configured")
    installed = _version()
    actions.append(f"install hermes-memory{'' if installed is None else '==' + installed} "
                   "into both, so the worker and the gate run the same code")
    for name, files in sorted(tree.items()):
        actions.append(f"carry {len(files)} file(s) of {name}/")
    actions.append(f"write {target / MANIFEST}")

    # The wheel's bytes belong in the approval as much as the carried ones do: naming a
    # file is not vouching for the file that is still there when the work starts.
    wheel_digest = content_digest(distribution.read_bytes()) \
        if distribution is not None and distribution.is_file() else None
    review = digest([str(target), str(root or ""), commit or "unrevisioned",
                     "dirty" if dirty else "clean",
                     wheel_digest or "build-at-apply",
                     BACKEND_SPEC if backend else "no-backend",
                     json.dumps(fingerprints, sort_keys=True)])
    return {"into": str(target), "source": str(root or ""), "wheel": str(distribution or ""),
            "source_commit": commit or "", "source_dirty": bool(dirty),
            "backend": backend, "actions": actions, "carried": tree,
            "fingerprints": fingerprints, "blocking": blocking, "review_digest": review}


def apply(*, settings, into: Path | str, source: Path | str | None = None,
          wheel: Path | str | None = None, actor: str, review: str,
          environ: Mapping[str, str] | None = None, backend: bool = True,
          runner: Callable | None = None, verbose: bool = False) -> dict[str, Any]:
    """Build the tree. Refused unless the digest still describes the plan that was shown.

    ``runner`` defaults to ``subprocess.run``; the tests hand in a recorder, because a
    staging step that fetched a hundred and eighty-eight packages to prove it could not
    finish a sentence would make the suite a function of the network.
    """
    if not actor or not actor.strip():
        raise ReleaseError("staging names who approved it; an anonymous release is a "
                           "dependency nobody accepted")
    staged = plan(settings=settings, into=into, source=source, wheel=wheel,
                  environ=environ, backend=backend)
    if staged["blocking"]:
        raise ReleaseError("; ".join(staged["blocking"]))
    if staged["review_digest"] != review:
        raise ReleaseError(f"the plan changed since it was shown (asked for {review}, this "
                           f"tree would be {staged['review_digest']}); re-run without "
                           "--apply and read the new digest")

    run = runner or subprocess.run
    target = Path(staged["into"])
    root = Path(staged["source"])
    target.mkdir(parents=True)
    # The wheel is compiled outside the tree it is staging a release of: `uv venv` refuses a
    # directory that already has contents, so a build placed inside the half-made tree would
    # make the very first command fail. It is copied in afterwards, which is what lets the
    # record name an artefact that only this release holds.
    scratch = Path(tempfile.mkdtemp(prefix="hermes-memory-wheel-"))
    try:
        distribution = Path(staged["wheel"])
        if not distribution.is_file():
            distribution = _build_wheel(root, into=scratch, run=run)
        _run(run, ["uv", "venv", str(target), "--quiet"], stage="runtime venv")
        environments = [target]
        if backend:
            _run(run, ["uv", "venv", str(target / "hindsight"), "--quiet"],
                 stage="backend venv")
            _run(run, ["uv", "pip", "install", "--python", str(_python(target, backend=True)),
                       "--quiet", BACKEND_SPEC], stage="backend packages")
            environments.append(target / "hindsight")
        for environment in environments:
            _run(run, ["uv", "pip", "install", "--python", str(_python(environment)),
                       "--quiet", str(distribution)], stage="hermes-memory")
        if distribution.parent != target / "wheel":
            kept = target / "wheel" / distribution.name
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(distribution, kept)
            distribution = kept
        carry(source=root, into=target, names=tuple(staged["carried"]))
        manifest = {**{key: staged[key] for key in
                       ("into", "source", "source_commit", "source_dirty", "backend",
                        "review_digest")},
                    # The wheel that was actually installed, and its bytes: a plan that
                    # named none built one, and a release record that could not say which
                    # artefact the environments hold would answer no question at all.
                    "wheel": str(distribution),
                    "wheel_digest": content_digest(distribution.read_bytes()),
                    "backend_spec": BACKEND_SPEC if backend else None,
                    "framework_version": _version(),
                    "staged_by": actor.strip(), "python": sys.version.split()[0],
                    "carried": {name: len(files) for name, files in staged["carried"].items()}}
        (target / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    except Exception:
        # A half-staged tree answers to nothing, and a pointer could be flipped to it by a
        # later command that only checks the directory exists. It came from this call, so
        # taking it away again is not destroying anyone's work.
        shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    report = {**staged, "performed": True, "staged": str(target),
              "wheel": str(distribution), **verify(settings=settings, into=target,
                                                   environ=environ)}
    if verbose:
        report["note"] = "the pointer was not moved; see `upgrade --release` for the switch"
    return report


def verify(*, settings, into: Path | str,
           environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Does this tree satisfy what the units and the host will actually ask of it?

    The executables are read out of the service templates this release carries, so adding a
    unit or changing an ``ExecStart`` cannot leave the check behind — the same rule `setup`'s
    staging step follows, applied at the moment the tree is built rather than months later.
    """
    target = Path(into).expanduser()
    missing: list[str] = []
    wanted: list[str] = []
    # The process environment is the starting point because that is what the installer is
    # handed at run time; only the release pointer is overridden, and the units are then
    # rendered as they would be written *into this tree*.
    environment = dict(os.environ if environ is None else environ)
    environment["HERMES_MEMORY_RELEASE"] = str(target)
    try:
        wanted = [str(path) for path in executables(render(settings, environ=environment))]
    except Exception as error:  # templates unreadable is a finding, not a traceback
        missing.append(f"the service templates could not be rendered: {error}")
    missing += [path for path in wanted if not Path(path).is_file()]
    for pair in (Path("deployment/compatibility.json"), Path("deployment/systemd"),
                 Path("integrations/hermes-memory/provider.py"), Path(MANIFEST)):
        if not (target / pair).exists():
            missing.append(str(target / pair))
    plugin = sorted((target / "integrations" / "hermes-memory").glob("*.py"))
    return {"complete": not missing, "starts": wanted,
            "plugin_files": len(plugin),
            "plugin_digest": digest([[p.name, content_digest(p.read_bytes())] for p in plugin])
            if plugin else None,
            "missing": missing}


def carry(*, source: Path, into: Path, names: tuple[str, ...] = CARRIED) -> list[str]:
    """Copy the halves of a checkout that a release has to carry beside its runtime.

    One implementation, because a staged tree that carries one half and not the other is the
    exact drift the compatibility manifest and `register-plugin` exist to catch.
    """
    copied: list[str] = []
    for name in names:
        origin = Path(source) / name
        if not origin.is_dir():
            continue
        shutil.copytree(origin, Path(into) / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                        dirs_exist_ok=True)
        copied.append(name)
    return copied


def _version() -> str | None:
    """The version of the code doing the staging, if it is installed as a distribution.

    A source run has no installed version, and saying so is better than inventing one: the
    manifest is read later by something asking "which build is this".
    """
    try:
        return metadata.version("hermes-memory")
    except metadata.PackageNotFoundError:
        return None


def _revision(root: Path) -> tuple[str | None, bool]:
    """Which commit this checkout is at, and whether its working tree differs from it.

    ``(None, False)`` when the source carries no repository of its own — an unpacked tree is
    entitled to say nothing rather than to be refused, and a ``git`` run from a directory
    *inside* somebody else's repository would report that one as if it were this. The
    upward search is what the ``.git`` guard is for.
    """
    if not (root / ".git").exists():
        return None, False

    def ask(*arguments: str) -> str | None:
        try:
            answer = subprocess.run(["git", "-C", str(root), *arguments],
                                    capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return answer.stdout.strip() if answer.returncode == 0 else None

    commit = ask("rev-parse", "HEAD")
    if commit is None or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None, False
    return commit, bool(ask("status", "--porcelain", "--untracked-files=all"))


def _build_wheel(root: Path, *, into: Path, run: Callable) -> Path:
    """Compile the distribution into a directory of its own, and insist on one answer.

    A checkout's ``dist/`` holds every build that machine has ever run, so choosing the
    newest wheel there can install something this call never compiled. The scratch directory
    this builds into belongs to the one staging that asked for it.
    """
    into.mkdir(parents=True, exist_ok=True)
    _run(run, ["uv", "build", "--quiet", "--out-dir", str(into), str(root)], stage="wheel")
    built = sorted(into.glob("hermes_memory-*.whl"))
    if len(built) != 1:
        raise ReleaseError(f"the build left {len(built)} wheels under {into}; a release "
                           "installs exactly one")
    return built[0]


def _run(run: Callable, argv: list[str], *, stage: str) -> None:
    result = run(argv, capture_output=True, text=True)
    if getattr(result, "returncode", 1):
        detail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()
        raise ReleaseError(f"{stage} failed: {' '.join(argv)[:160]} — {detail[:300]}")
