"""Owner-reviewed local reranker configuration; no upstream package patches."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

from ..config import env_file_values, load_settings
from ..ids import content_digest, digest
from ..models.reranker import MODEL


def plan(settings):
    paths = [settings.home / "hermes-memory.env", settings.home / "hindsight.env"]
    for path in paths:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("configuration must be owned regular private files")
    if not settings.admission_url or settings.admission_url.rstrip("/") != "http://127.0.0.1:8123":
        raise ValueError("requires the owned loopback admission endpoint")
    proposal = {"files": [str(p) for p in paths],
                "fingerprints": [content_digest(p.read_bytes()) for p in paths],
                "model": MODEL, "upstream": "http://127.0.0.1:8185",
                "gate": "http://127.0.0.1:8123/v1/rerank", "physical_resource": "local-gpu",
                "candidate_caps": {"low": 16, "mid": 32, "high": 64},
                "fallback": "none; errors are not silently reported as learned ranking"}
    return {**proposal, "review_digest": digest(["local-reranking-v1", proposal])}


def _render(path, updates):
    lines = []
    for line in path.read_text().splitlines():
        if line.strip().split("=", 1)[0] not in updates:
            lines.append(line)
    lines.extend(f"{key}={value}" for key, value in updates.items())
    return "\n".join(lines) + "\n"


def _publish(path, text):
    fd, name = tempfile.mkstemp(prefix=".reranking-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def apply(settings, *, actor, review, runner=subprocess.run):
    if not settings.owner_principal or actor != settings.owner_principal:
        raise ValueError("only the configured owner may change reranker routing")
    proposal = plan(settings)
    if review != proposal["review_digest"]:
        raise ValueError("configuration changed since review")
    for unit in ("hermes-memory.service", "hermes-memory-hindsight.service", "hermes-memory-worker.service"):
        state = runner(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=10)
        if state.stdout.strip() not in {"inactive", "failed"}:
            raise ValueError("stop owned memory services before changing configuration")
    paths = list(map(Path, proposal["files"]))
    values = env_file_values(paths[0])
    credential = values.get("HERMES_MEMORY_ROUTE_CREDENTIAL_RERANK") or secrets.token_urlsafe(32)
    if credential in [v for k, v in values.items() if k.startswith("HERMES_MEMORY_ROUTE_CREDENTIAL_")
                      and k != "HERMES_MEMORY_ROUTE_CREDENTIAL_RERANK"]:
        raise ValueError("rerank credential must be distinct")
    framework = {"HERMES_MEMORY_RERANKER_BASE_URL": proposal["upstream"],
                 "HERMES_MEMORY_RERANKER_RESOURCE": "local-gpu", "HERMES_MEMORY_RERANKER_MODEL": MODEL,
                 "HERMES_MEMORY_ROUTE_CREDENTIAL_RERANK": credential}
    native = {"HINDSIGHT_API_RERANKER_PROVIDER": "cohere", "HINDSIGHT_API_ENABLE_RERANKING": "true",
              "HINDSIGHT_API_RERANKER_COHERE_API_KEY": credential,
              "HINDSIGHT_API_RERANKER_COHERE_BASE_URL": proposal["gate"],
              "HINDSIGHT_API_RERANKER_COHERE_MODEL": MODEL, "HINDSIGHT_API_RERANKER_COHERE_TIMEOUT": "3",
              "HINDSIGHT_API_RERANKER_MAX_CANDIDATES": "64",
              **{f"HINDSIGHT_API_RERANKER_MAX_CANDIDATES_{k.upper()}": str(v)
                 for k, v in proposal["candidate_caps"].items()}}
    existing = env_file_values(paths[1])
    if any(k.startswith("HINDSIGHT_API_RERANKER_1_") for k in existing):
        raise ValueError("review existing fallback chain before changing primary reranker")
    backups = settings.home / "activation-backups"
    backups.mkdir(mode=0o700, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="reranking-", dir=backups))
    for path in paths:
        shutil.copy2(path, backup / path.name)
    try:
        for path, updates in zip(paths, (framework, native), strict=True):
            _publish(path, _render(path, updates))
    except BaseException:
        for path in paths:
            _publish(path, (backup / path.name).read_text())
        raise
    return {"state": "configured-services-stopped", "backup": str(backup), "model": MODEL,
            "provider": "cohere-compatible-local-only", "review_digest": review}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--actor")
    parser.add_argument("--review")
    args = parser.parse_args()
    settings = load_settings()
    print(json.dumps(apply(settings, actor=args.actor, review=args.review) if args.apply else plan(settings)))


if __name__ == "__main__":
    main()
