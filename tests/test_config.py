"""Configuration must be explicit: owned env file, allowlisted routes, no ambient cloud."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_memory.config import SettingError, load_settings, validate_inference_route

APPROVED_LAN = "192.168.68.65"


def write_env(home: Path, values: dict[str, str]) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "hermes-memory.env"
    path.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8")
    return path


def test_missing_env_file_yields_capture_only_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "home"))
    settings = load_settings()
    assert settings.capture_only is True
    assert settings.inference_enabled is False
    assert settings.hindsight_url is None


def test_route_from_owned_env_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {
        "HERMES_MEMORY_HINDSIGHT_URL": f"http://{APPROVED_LAN}:8080/v1",
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS": APPROVED_LAN,
        "HERMES_MEMORY_INFERENCE_ENABLED": "true",
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS": "50000",
    })
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    settings = load_settings()
    assert settings.hindsight_url == f"http://{APPROVED_LAN}:8080/v1"
    assert settings.capture_only is False
    assert settings.background_budget_tokens == 50000


def test_a_cloud_route_is_refused_even_if_explicitly_configured(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {
        "HERMES_MEMORY_HINDSIGHT_URL": "https://api.hindsight.vectorize.io",
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS": "api.hindsight.vectorize.io",
        "HERMES_MEMORY_INFERENCE_ENABLED": "true",
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS": "10",
    })
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match="neither loopback nor a literal private address"):
        load_settings()


def test_an_unlisted_host_is_refused(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {
        "HERMES_MEMORY_HINDSIGHT_URL": "http://10.0.0.9:8080/v1",
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS": APPROVED_LAN,
        "HERMES_MEMORY_INFERENCE_ENABLED": "true",
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS": "10",
    })
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match="not in the approved allowlist"):
        load_settings()


def test_enabling_inference_without_an_allowlist_fails_closed(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {
        "HERMES_MEMORY_HINDSIGHT_URL": f"http://{APPROVED_LAN}:8080/v1",
        "HERMES_MEMORY_INFERENCE_ENABLED": "true",
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS": "10",
    })
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match="allowlist"):
        load_settings()


def test_zero_budget_keeps_formation_disabled(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {
        "HERMES_MEMORY_HINDSIGHT_URL": f"http://{APPROVED_LAN}:8080/v1",
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS": APPROVED_LAN,
        "HERMES_MEMORY_INFERENCE_ENABLED": "true",
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS": "0",
    })
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    assert load_settings().capture_only is True


def test_ambient_credentials_cannot_select_a_route(tmp_path, monkeypatch):
    """The retired framework could inherit an unrelated host auth.json provider."""
    (tmp_path / "auth.json").write_text(json.dumps({"providers": {"nous": {"key": "x"}}}))
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path))
    for noise in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NOUS_API_KEY", "HINDSIGHT_API_KEY"):
        monkeypatch.setenv(noise, "ambient-secret")
    settings = load_settings()
    assert settings.hindsight_url is None
    assert settings.capture_only is True


def test_process_environment_overrides_the_owned_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS": "10.1.1.1"})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS", APPROVED_LAN)
    assert load_settings().allowed_inference_hosts == frozenset({APPROVED_LAN})


def test_env_file_values_are_never_shell_evaluated(tmp_path, monkeypatch):
    """A config value is data, not a command line."""
    home = tmp_path / "home"
    literal = "`whoami`/$(id -u)/$HOME"
    path = write_env(home, {"HERMES_MEMORY_DATA_DIR": literal})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.delenv("HERMES_MEMORY_DATA_DIR", raising=False)

    settings = load_settings(path)
    assert str(settings.data_dir) == str(home / literal)
    # Relative configured paths resolve under the owned home, never the cwd.
    assert settings.data_dir.is_absolute() and settings.data_dir.is_relative_to(home)


def test_unrelated_cwd_does_not_move_the_data_directory(tmp_path, monkeypatch, tmp_path_factory):
    home = tmp_path / "home"
    path = write_env(home, {"HERMES_MEMORY_DATA_DIR": "data"})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    assert load_settings(path).data_dir == home / "data"


@pytest.mark.parametrize(
    "url",
    ["", "ftp://127.0.0.1/x", "http:///v1", "https://evil.example/v1", "http://169.254.169.254/"],
)
def test_validate_inference_route_rejects_unusable_or_cloud_targets(url):
    with pytest.raises(SettingError):
        validate_inference_route(url, frozenset({APPROVED_LAN, "127.0.0.1"}))


def test_validate_inference_route_accepts_loopback_and_literal_lan():
    hosts = frozenset({"127.0.0.1", APPROVED_LAN})
    assert validate_inference_route("http://127.0.0.1:8080/v1", hosts) == "127.0.0.1"
    assert validate_inference_route(f"http://{APPROVED_LAN}:8080/v1", hosts) == APPROVED_LAN


@pytest.mark.parametrize("dangerous", [
    "169.254.169.254",        # cloud instance metadata (link-local, is_private==True)
    "169.254.1.1",            # any link-local
    "100.64.0.1",             # CGNAT / not explicitly a LAN range
    "0.0.0.0",
    "fe80::1",                # IPv6 link-local
])
def test_a_listed_but_unsafe_address_is_still_refused(dangerous):
    """Being on the allowlist is necessary, not sufficient: the target must be LAN."""
    with pytest.raises(SettingError, match="neither loopback nor a literal private"):
        validate_inference_route(_url(dangerous), frozenset({dangerous}))


@pytest.mark.parametrize("safe", ["10.0.0.5", "172.16.9.9", "192.168.1.50", "fd12::3", "127.0.0.1", "::1"])
def test_explicit_lan_and_loopback_ranges_are_accepted(safe):
    assert validate_inference_route(_url(safe), frozenset({safe})) == safe


def _url(host: str) -> str:
    """IPv6 literals need brackets in a URL; the parsed hostname comes back bare."""
    authority = f"[{host}]" if ":" in host else host
    return f"http://{authority}:8080/v1"


def test_a_hostname_is_refused_even_if_it_resolves_privately():
    # DNS can be redirected; only literal addresses are admissible.
    with pytest.raises(SettingError, match="neither loopback nor a literal private"):
        validate_inference_route("http://internal.example/v1", frozenset({"internal.example"}))
