"""Runtime configuration.

Everything comes from an explicit, owned env file plus process environment.
There is deliberately **no** ambient provider discovery: the retired
framework could pick up a cloud credential from an unrelated host
``auth.json`` and silently send evidence to it. Here an unapproved route is a
startup error, not a fallback.
"""
from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

__all__ = ["Settings", "SettingError", "load_settings", "validate_inference_route"]


class SettingError(RuntimeError):
    """Configuration is absent, malformed, or points at an unapproved route."""


DEFAULT_ENV_FILENAME = "hermes-memory.env"


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=value`` env file. No shell evaluation, ever."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _approved_hosts_setting(value: str | None) -> set[str]:
    hosts: set[str] = set()
    for item in (value or "").split(","):
        item = item.strip().lower()
        if item:
            hosts.add(item)
    return hosts


_LAN_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)


def _host_is_local_or_lan(host: str) -> bool:
    """Loopback plus the explicit RFC1918/ULA ranges only.

    ``ipaddress.is_private`` cannot be used as the test: it reports True for
    link-local 169.254.0.0/16, which contains the cloud instance metadata
    address 169.254.169.254. A listed host is still not trusted if it is not
    actually on the LAN, and 'private-looking' is not 'approved'.

    A hostname that merely resolves into a private range is also rejected:
    resolution can be redirected, so only literal addresses are accepted.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return any(address in network for network in _LAN_NETWORKS)


@dataclass(frozen=True)
class Settings:
    home: Path
    data_dir: Path
    db_path: Path
    blob_dir: Path
    hindsight_url: str | None
    hindsight_api_key_env: str | None
    allowed_inference_hosts: frozenset[str] = field(default_factory=frozenset)
    inference_enabled: bool = False
    background_budget_tokens: int = 0
    foreground_deadline_s: float = 8.0

    @property
    def capture_only(self) -> bool:
        """No inference route or budget is configured, so formation stays off."""
        return not (
            self.inference_enabled
            and self.hindsight_url
            and self.background_budget_tokens > 0
        )


def validate_inference_route(url: str, allowed: frozenset[str]) -> str:
    """Return the host of *url*, or raise if the route is not explicitly approved."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SettingError(f"inference route has no http(s) host: {url!r}")
    host = parsed.hostname.lower()
    if host not in allowed:
        raise SettingError(
            f"inference route host {host!r} is not in the approved allowlist; "
            "add HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS to the owned env file"
        )
    if not _host_is_local_or_lan(host):
        raise SettingError(
            f"approved host {host!r} is neither loopback nor a literal private "
            "address; local/LAN-only inference policy refuses it"
        )
    return host


def load_settings(env_file: str | os.PathLike[str] | None = None) -> Settings:
    """Load settings from the owned env file, then process environment.

    Process env wins so a systemd drop-in can override the file, but the
    allowlist check still runs on whatever route ends up configured.
    """
    home = Path(os.environ.get("HERMES_MEMORY_HOME") or Path.home() / "data" / "hermes-memory")
    if env_file is None:
        env_file = home / DEFAULT_ENV_FILENAME
    values = _read_env_file(Path(env_file))

    def get(key: str, default: str | None = None) -> str | None:
        return os.environ.get(f"HERMES_MEMORY_{key}", values.get(f"HERMES_MEMORY_{key}", default))

    def flag(key: str) -> bool:
        return (get(key, "false") or "").strip().lower() in {"1", "true", "yes", "on"}

    data_dir = Path(get("DATA_DIR", str(home / "data")) or str(home / "data"))
    if not data_dir.is_absolute():
        data_dir = home / data_dir
    url = get("HINDSIGHT_URL")
    allowed = frozenset(_approved_hosts_setting(get("ALLOWED_INFERENCE_HOSTS")))
    inference_enabled = flag("INFERENCE_ENABLED")

    if url and allowed:
        validate_inference_route(url, allowed)
    elif url and inference_enabled:
        raise SettingError(
            "inference is enabled with a configured route but no "
            "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS allowlist"
        )

    return Settings(
        home=home,
        data_dir=data_dir,
        db_path=data_dir / "canonical.db",
        blob_dir=data_dir / "blobs",
        hindsight_url=url,
        hindsight_api_key_env=get("HINDSIGHT_API_KEY_ENV"),
        allowed_inference_hosts=allowed,
        inference_enabled=inference_enabled,
        background_budget_tokens=int(get("BACKGROUND_BUDGET_TOKENS", "0") or 0),
        foreground_deadline_s=float(get("FOREGROUND_DEADLINE_S", "8") or 8),
    )
