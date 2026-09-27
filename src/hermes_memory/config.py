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
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

__all__ = ["Settings", "SettingError", "load_settings", "validate_inference_route",
           "scoped_settings", "scoped_secret", "env_file_values", "DEFAULT_PROFILE",
           "DEFAULT_BANK"]


class SettingError(RuntimeError):
    """Configuration is absent, malformed, or points at an unapproved route."""


DEFAULT_ENV_FILENAME = "hermes-memory.env"

# The name the single-profile installation has always used, so an installation that
# never enrolled a profile keeps its existing bank and secret variable names.
DEFAULT_PROFILE = "default"
DEFAULT_BANK = "hermes"

# Hermes abandons an external prefetch after this many seconds. A foreground
# deadline at or above it is a number the host will never honour, so it is
# refused at startup rather than discovered on the first slow turn.
HOST_PREFETCH_STOP_S = 8.0
DEFAULT_FOREGROUND_DEADLINE_S = 4.0
# How often the runtime process takes a background pass. A day is the ceiling because a
# slower loop is not a loop; zero is the owner saying "I will run it myself".
# An evaluation is a claim about a version. Without one to compare against, a passed run
# could never be found stale, which is the hole C11 exists to close.
DEFAULT_EVALUATOR_TIMEOUT_S = 120.0
DEFAULT_MAINTENANCE_INTERVAL_S = 900
MAX_MAINTENANCE_INTERVAL_S = 86_400


def _maintenance_interval(value: str | None) -> int:
    """The scheduler's own period, or zero for "nothing runs by itself here"."""
    try:
        seconds = int(str(value).strip())
    except (TypeError, ValueError):
        raise SettingError(
            "HERMES_MEMORY_MAINTENANCE_INTERVAL_S must be a whole number of seconds, "
            f"got {value!r}"
        ) from None
    if not 0 <= seconds <= MAX_MAINTENANCE_INTERVAL_S:
        raise SettingError(
            f"HERMES_MEMORY_MAINTENANCE_INTERVAL_S={value} is outside the admissible range "
            f"(0 to {MAX_MAINTENANCE_INTERVAL_S}); 0 means the pass never runs unattended"
        )
    return seconds


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


def env_file_values(path: str | os.PathLike[str]) -> dict[str, str]:
    """The values an owned env file holds, without evaluating anything.

    Public because the installer needs to see the same raw view ``load_settings`` sees,
    and a second parser would eventually disagree with the first about quoting.
    """
    return _read_env_file(Path(path))


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
    # Which enrolled profile these paths belong to. A configuration with no profile
    # name is the installation's own default; a resolved activity always carries one.
    profile: str = DEFAULT_PROFILE
    bank_id: str = DEFAULT_BANK
    credential_scope: str = "profile-default"
    # Provider-initiated delivery is off until an owner names a concrete private
    # destination. Nothing else can authorise it, and no default is "on".
    delivery_enabled: bool = False
    delivery_target: str | None = None
    # The runtime unit owns the background pass, per §4: proactive state and scheduling
    # live in the framework process, not in a fourth unit. Zero hands the timer back to
    # the owner, who then runs `hermes-memory maintain` from wherever they choose.
    maintenance_interval_s: int = DEFAULT_MAINTENANCE_INTERVAL_S
    # The program that is allowed to decide whether a lesson's fixtures passed. Named by
    # the owner, run with no shell and a scrubbed environment. Unset means procedural
    # learning can be proposed, contradicted and retracted but never promoted by a run.
    evaluator_command: tuple[str, ...] = ()
    evaluator_timeout_s: float = DEFAULT_EVALUATOR_TIMEOUT_S
    # Environment names the evaluator is additionally allowed to see. The scrubbed set is
    # fixed; this is the owner's short list, read from the owned file like every other knob
    # rather than from whatever the invoking shell happened to export.
    evaluator_env: tuple[str, ...] = ()

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
    interval = _maintenance_interval(get("MAINTENANCE_INTERVAL_S",
                                         str(DEFAULT_MAINTENANCE_INTERVAL_S)))
    delivery_target = (get("DELIVERY_TARGET") or "").strip() or None
    delivery_enabled = flag("DELIVERY_ENABLED")
    if delivery_enabled and not delivery_target:
        raise SettingError(
            "HERMES_MEMORY_DELIVERY_ENABLED needs HERMES_MEMORY_DELIVERY_TARGET: an "
            "approved destination is what makes delivery an owner's decision rather "
            "than a capability the runtime has on its own"
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
        delivery_enabled=delivery_enabled,
        delivery_target=delivery_target,
        maintenance_interval_s=interval,
        evaluator_command=_command(get, "EVALUATOR_COMMAND"),
        evaluator_timeout_s=_seconds(get, "EVALUATOR_TIMEOUT_S", DEFAULT_EVALUATOR_TIMEOUT_S),
        evaluator_env=_names(get, "EVALUATOR_ENV"),
    )


def scoped_settings(settings: Settings, *, profile: str, data_dir: str | os.PathLike[str],
                    bank_id: str, credential_scope: str) -> Settings:
    """The instance configuration, pointed at one enrolled profile's memory.

    Only the paths and the scope change. Routes, budgets and the allowlist stay as the
    operator set them, because another profile is another *person's* memory rather than
    another policy — and because the ledger, not an env file, decides where a profile's
    evidence lives, so a profile's own file cannot point two profiles at one store.
    """
    for name, value in (("profile", profile), ("bank_id", bank_id),
                        ("credential_scope", credential_scope)):
        if not isinstance(value, str) or not value.strip():
            raise SettingError(f"{name} must be nonempty text to scope a configuration")
    directory = Path(data_dir)
    if not directory.is_absolute():
        raise SettingError(f"profile {profile!r} data_dir must be absolute, got {directory}")
    return replace(settings, profile=profile.strip(), data_dir=directory,
                   db_path=directory / "canonical.db", blob_dir=directory / "blobs",
                   bank_id=bank_id.strip(), credential_scope=credential_scope.strip())


def scoped_secret(settings: Settings, name: str | None) -> str | None:
    """Read one credential *for this profile's scope*, with no fallback for the others.

    A single gateway process carries every profile's environment, so an unscoped lookup
    would let the second profile sign its requests with the first one's key. The scope is
    part of the variable name; a profile without its own scoped secret has none, and its
    route stays unavailable rather than borrowing. Only the default profile — the one
    installation that never had a scope to distinguish — may use the bare name.
    """
    if not name or not isinstance(name, str):
        return None
    wanted = name.strip()
    if not wanted:
        return None
    prefix = settings.credential_scope.strip().upper().replace("-", "_")
    scoped = os.environ.get(f"{prefix}_{wanted}") if prefix else None
    if scoped and scoped.strip():
        return scoped.strip()
    if settings.profile != DEFAULT_PROFILE:
        return None
    bare = os.environ.get(wanted)
    return bare.strip() if bare and bare.strip() else None


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


def _command(get, key: str) -> tuple[str, ...]:
    """The evaluator program, split into argv. A shell is never involved."""
    raw = (get(key) or "").strip()
    if not raw:
        return ()
    try:
        parts = shlex.split(raw)
    except ValueError as error:
        raise SettingError(f"{key} cannot be parsed: {error}") from None
    if not parts or not str(parts[0]).startswith("/"):
        raise SettingError(
            f"{key} must name an absolute path: an evaluation runs whatever it names, and "
            "a bare word resolves from whichever directory PATH happens to offer")
    return tuple(parts)


def _names(get, key: str) -> tuple[str, ...]:
    """A comma-separated list of environment names, validated at load time.

    Checked here rather than where they are used: an entry that is not a variable name can
    never be allowed through, and finding that out mid-evaluation wastes a run.
    """
    raw = (get(key) or "").strip()
    if not raw:
        return ()
    items = [item.strip() for item in raw.split(",")]
    for item in items:
        if not item.isidentifier():
            raise SettingError(f"{key} must list environment names, not {item!r}")
    return tuple(dict.fromkeys(items))


def _seconds(get, key: str, default: float) -> float:
    raw = (get(key) or "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        raise SettingError(f"{key} must be a number of seconds") from None
    if not 1 <= value <= 3600:
        raise SettingError(f"{key} must be between 1 and 3600 seconds")
    return value


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
