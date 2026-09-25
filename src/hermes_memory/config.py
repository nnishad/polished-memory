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

# Hermes abandons an external prefetch after this many seconds. A foreground
# deadline at or above it is a number the host will never honour, so it is
# refused at startup rather than discovered on the first slow turn.
HOST_PREFETCH_STOP_S = 8.0
DEFAULT_FOREGROUND_DEADLINE_S = 4.0


def _deadline(value: str | None) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise SettingError(
            f"HERMES_MEMORY_FOREGROUND_DEADLINE_S must be a number of seconds, got {value!r}"
        ) from None
    if not 0 < seconds < HOST_PREFETCH_STOP_S:
        raise SettingError(
            f"HERMES_MEMORY_FOREGROUND_DEADLINE_S={value} is outside the admissible range: "
            f"Hermes abandons an external prefetch after {HOST_PREFETCH_STOP_S}s, so a "
            "deadline at or above that is never honoured, only truncated by the host"
        )
    return seconds


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
class ModelRoute:
    """One approved upstream plus the physical resource it occupies."""

    base_url: str
    resource: str
    model: str | None = None


@dataclass(frozen=True)
class OutputCaps:
    """Explicit completion caps per generation path.

    Every path needs one. A cap that only some operations set is no cap at all
    on the operations that omit it, and the engine's own default permitted
    unbounded completions.
    """

    retain: int = 2048
    consolidate: int = 2048
    reflect: int = 2048
    foreground: int = 1024


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
    foreground_deadline_s: float = DEFAULT_FOREGROUND_DEADLINE_S
    owner_principal: str | None = None
    text_route: ModelRoute | None = None
    vision_route: ModelRoute | None = None
    embeddings_route: ModelRoute | None = None
    max_output_tokens: OutputCaps = field(default_factory=OutputCaps)
    route_credentials: dict[str, str] = field(default_factory=dict)
    gate_token: str | None = None

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

    foreground = _deadline(get("FOREGROUND_DEADLINE_S", str(DEFAULT_FOREGROUND_DEADLINE_S)))

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
        foreground_deadline_s=foreground,
        # Unnamed by default: with no owner principal, forgetting can be
        # requested and previewed but never confirmed, which fails closed
        # instead of accepting any caller that claims to be the owner.
        owner_principal=(get("OWNER_PRINCIPAL") or "").strip() or None,
        text_route=_route(get, allowed, "TEXT", default_resource="remote-9b"),
        vision_route=_route(get, allowed, "VISION", default_resource="local-gpu"),
        embeddings_route=_route(get, allowed, "EMBEDDINGS", default_resource="local-gpu"),
        max_output_tokens=OutputCaps(
            retain=_cap(get, "RETAIN"), consolidate=_cap(get, "CONSOLIDATE"),
            reflect=_cap(get, "REFLECT"), foreground=_cap(get, "FOREGROUND")),
        route_credentials=_credentials(values),
        gate_token=(get("GATE_TOKEN") or "").strip() or None,
    )


def _route(get, allowed: frozenset[str], key: str, *, default_resource: str):
    base_url = (get(f"{key}_BASE_URL") or "").strip()
    if not base_url:
        return None
    validate_inference_route(base_url, allowed)
    resource = (get(f"{key}_RESOURCE") or default_resource).strip()
    if resource not in {"remote-9b", "local-gpu"}:
        raise SettingError(
            f"{key}_RESOURCE must name a physical resource (remote-9b or local-gpu), "
            f"not {resource!r}: slots are per device, not per endpoint")
    return ModelRoute(base_url=base_url, resource=resource,
                      model=(get(f"{key}_MODEL") or "").strip() or None)


def _cap(get, key: str) -> int:
    raw = (get(f"MAX_OUTPUT_TOKENS_{key}") or "").strip()
    if not raw:
        return getattr(OutputCaps(), key.lower())
    value = int(raw)
    if not 1 <= value <= 32_768:
        raise SettingError(f"MAX_OUTPUT_TOKENS_{key} must be between 1 and 32768")
    return value


def _credentials(file_values: dict[str, str]) -> dict[str, str]:
    """Route credentials come from the owned env file or the process environment.

    They select a route; they are not upstream API keys. Upstream credentials
    stay at the gate, so a caller that obtains a route credential still cannot
    reach a model server directly.
    """
    prefix = "HERMES_MEMORY_ROUTE_CREDENTIAL_"
    found = {}
    for source in (file_values, os.environ):
        for name, value in source.items():
            if name.startswith(prefix) and str(value).strip():
                found[name[len(prefix):].lower()] = str(value).strip()
    return found
