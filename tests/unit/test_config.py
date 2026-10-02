"""Configuration must be explicit: owned env file, allowlisted routes, no ambient cloud."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from hermes_memory import config
from hermes_memory.config import (SettingError, endpoint_is_private, load_settings,
                                  scoped_secret, scoped_settings, validate_inference_route)

APPROVED_LAN = "192.168.68.65"
REPO = Path(config.__file__).resolve().parents[2]
TEMPLATE = REPO / "deployment" / "env" / "hermes-memory.env.example"

# One credential per configured route, named after it, so the family is documented by its
# prefix rather than by a list that would go stale the day a route is added.
CREDENTIAL_PREFIX = "HERMES_MEMORY_ROUTE_CREDENTIAL_"
_KEY_LINE = re.compile(r"^#?\s*(HERMES_MEMORY_[A-Z][A-Z0-9_]*)=", re.MULTILINE)
# The owned file names the variable that holds a secret; the value belongs to the
# credential file, so a name on the right of one of these lines counts as documented.
_SECRET_HOLDER = re.compile(r"^#?\s*HERMES_MEMORY_[A-Z0-9_]*_API_KEY_ENV="
                            r"(HERMES_MEMORY_[A-Z][A-Z0-9_]*)$", re.MULTILINE)
_NAME = re.compile(r"HERMES_MEMORY_[A-Z][A-Z0-9_]*")


def write_env(home: Path, values: dict[str, str]) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "hermes-memory.env"
    path.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8")
    return path


def test_the_shipped_env_template_names_every_setting_the_code_reads(monkeypatch, tmp_path):
    """A knob that exists only in code is a knob nobody finds, and one nobody sets.

    The template is the only place an operator is told these exist, so it is checked
    against what the loader actually asks the environment for rather than against
    somebody's memory: a setting added without a line here fails, and a line left behind
    after the setting goes fails the same way.
    """
    template = TEMPLATE.read_text(encoding="utf-8")
    documented = {name for name in _KEY_LINE.findall(template)
                  if not name.startswith(CREDENTIAL_PREFIX)}
    # A secret is named here but never set here: the owned file carries the variable that
    # *holds* it, and the value lives in the credential file the installer writes.
    documented |= {name for name in _SECRET_HOLDER.findall(template)}

    asked = set()
    real_get = os.environ.get

    def spy(key, default=None):
        asked.add(str(key))
        return real_get(key, default)

    monkeypatch.setattr(os.environ, "get", spy, raising=False)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "home"))
    load_settings()
    monkeypatch.undo()
    named = {name for name in asked if name.startswith("HERMES_MEMORY_")}
    for root in (REPO / "src" / "hermes_memory", REPO / "integrations"):
        for path in sorted(root.rglob("*.py")):
            named |= {name for name in _NAME.findall(path.read_text(encoding="utf-8"))
                      if not name.startswith(CREDENTIAL_PREFIX)}
    # A route's device and model are only asked about once its base URL is set, and an
    # output ceiling only once its operation is capped: a template that named the one and
    # not the others would look complete while hiding most of the configuration.
    for route in ("TEXT", "VISION", "EMBEDDINGS", "RERANKER"):
        named |= {f"HERMES_MEMORY_{route}_{suffix}"
                  for suffix in ("BASE_URL", "RESOURCE", "MODEL")}
    named |= {f"HERMES_MEMORY_MAX_OUTPUT_TOKENS_{operation}"
              for operation in ("RETAIN", "CONSOLIDATE", "REFLECT", "FOREGROUND")}
    assert named - documented == set(), "a setting the code knows about is not in the template"
    assert documented - named == set(), "the template names a setting no code reads"
    assert CREDENTIAL_PREFIX in template, "one credential per route, named by the route"


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


@pytest.mark.parametrize("url", ["http://127.0.0.1:8888", "http://[::1]:8080/v1",
                                 f"http://{APPROVED_LAN}:8080/v1", "http://10.0.0.5/v1",
                                 "https://172.16.9.9:80/v1", "http://[fd12::3]:8080/v1"])
def test_an_endpoint_on_this_machine_or_the_private_lan_is_private(url):
    """The question a backend's own environment is asked: can this reach the internet?"""
    assert endpoint_is_private(url), url


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "http://localhost:11434/v1",
                                 "http://8.8.8.8/v1", "http://169.254.169.254/latest",
                                 "http://100.64.0.1/v1", "http://[fe80::1]:8080/v1",
                                 "http://internal.example:8080/v1", "ftp://127.0.0.1/x",
                                 "127.0.0.1:8888", ""])
def test_a_name_or_a_public_number_is_not_a_private_endpoint(url):
    """`localhost` is a name, and a name is a record somebody else controls.

    A hosted endpoint is what the whole no-cloud policy exists to keep out, and it does not
    become acceptable because the process reaching it is the backend rather than this one.
    """
    assert not endpoint_is_private(url), url


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


def _routes(tmp_path, monkeypatch, *, backend: str, admission: str):
    """An owned file naming a backend route and, optionally, this framework's own listener.

    The process environment is cleared for both first: these are settings a running
    installation exports, and a suite that inherited them would be testing the machine.
    """
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_HINDSIGHT_URL": backend,
                     "HERMES_MEMORY_ADMISSION_URL": admission,
                     "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS": "127.0.0.1"})
    for key in ("HINDSIGHT_URL", "ADMISSION_URL"):
        monkeypatch.delenv(f"HERMES_MEMORY_{key}", raising=False)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))


def test_a_backend_route_that_points_at_this_frameworks_own_listener_is_refused(tmp_path,
                                                                                monkeypatch):
    """The admission endpoint answers by asking the backend, so that route is a call to itself.

    A shipped template once documented both services on one port; the recursion it would have
    started is a hang rather than an error, which is why the loader refuses it rather than
    leaving it for the first retain to discover.
    """
    _routes(tmp_path, monkeypatch, backend="http://127.0.0.1:8123",
            admission="http://127.0.0.1:8123")
    with pytest.raises(SettingError, match="ADMISSION_URL"):
        load_settings()


@pytest.mark.parametrize("admission", ["http://127.0.0.1", "http://127.0.0.1:80/",
                                       "HTTP://127.0.0.1:80"])
def test_the_implicit_port_and_the_named_one_are_the_same_listener(admission, tmp_path,
                                                                   monkeypatch):
    _routes(tmp_path, monkeypatch, backend="http://127.0.0.1:80/v1", admission=admission)
    with pytest.raises(SettingError, match="ADMISSION_URL"):
        load_settings()


def test_two_listeners_on_one_host_are_only_the_same_thing_when_they_share_a_port(tmp_path,
                                                                                  monkeypatch):
    """The real shape of an installation: its own gate beside the engine, one address each."""
    _routes(tmp_path, monkeypatch, backend="http://127.0.0.1:8888",
            admission="http://127.0.0.1:8123")
    assert load_settings().hindsight_url == "http://127.0.0.1:8888"


def test_a_configured_backend_with_no_admission_endpoint_is_not_argued_with(tmp_path,
                                                                            monkeypatch):
    """An installation whose runtime unit is not installed has nothing to collide with."""
    _routes(tmp_path, monkeypatch, backend="http://127.0.0.1:8888", admission="")
    assert load_settings().hindsight_url == "http://127.0.0.1:8888"


@pytest.mark.parametrize("backend,admission", [
    ("http://127.0.0.1:8888", "http://127.0.0.1:everywhere"),
    ("http://127.0.0.1:nearby", "http://127.0.0.1:everywhere"),
])
def test_an_admission_endpoint_that_is_not_a_listener_is_not_blamed_for_the_route(
        backend, admission, tmp_path, monkeypatch):
    """A port that is not a number is that line's own problem, not evidence of a collision.

    Reporting one as a clash would send the operator to edit the other setting, and two
    unparsable ports are not proof that they name one place to listen.
    """
    _routes(tmp_path, monkeypatch, backend=backend, admission=admission)
    assert load_settings().hindsight_url == backend


@pytest.mark.parametrize("too_long", ["8", "8.0", "30", "0", "-1", "soon", ""])
def test_a_foreground_deadline_the_host_will_not_honour_is_refused(too_long, tmp_path,
                                                                    monkeypatch):
    """Hermes abandons a prefetch at 8s; a longer deadline is a number nobody obeys."""
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_FOREGROUND_DEADLINE_S": too_long})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match="FOREGROUND_DEADLINE_S"):
        load_settings()


def test_the_default_foreground_deadline_leaves_room_inside_the_host_stop(tmp_path,
                                                                          monkeypatch):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    settings = load_settings()
    assert 0 < settings.foreground_deadline_s < 8.0


@pytest.mark.parametrize("nonsense", ["soon", "", "1,2"])
def test_a_queue_that_is_not_a_number_of_seconds_is_refused(nonsense, tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_GATE_QUEUE_S": nonsense})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match="GATE_QUEUE_S must be a number"):
        load_settings()


@pytest.mark.parametrize("outside", ["-1", "901", "3600"])
def test_a_queue_outside_the_admissible_range_is_refused(outside, tmp_path, monkeypatch):
    """A wait measured in hours is a formation queue that never drains, not patience."""
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_GATE_QUEUE_S": outside})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match="GATE_QUEUE_S"):
        load_settings()


def test_zero_queue_is_the_owner_asking_for_the_old_immediate_refusal(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_GATE_QUEUE_S": "0"})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    assert load_settings().gate_queue_s == 0.0


def test_the_default_queue_outlives_the_operation_it_is_waiting_behind(tmp_path, monkeypatch):
    """The engine's retain call holds a device for tens of seconds; a shorter patience
    would put the installation back where the refusal did."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    assert load_settings().gate_queue_s > 60.0


# -- per-profile configuration -------------------------------------------------

def instance(tmp_path, monkeypatch, extra=None):
    home = tmp_path / "instance"
    values = {"HERMES_MEMORY_DATA_DIR": tmp_path / "data",
              "HERMES_MEMORY_INFERENCE_ENABLED": "false"}
    values.update(extra or {})
    write_env(home, values)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.delenv("HERMES_MEMORY_DATA_DIR", raising=False)
    return load_settings()


def test_scoping_a_configuration_moves_only_the_memory_it_names(tmp_path, monkeypatch):
    settings = instance(tmp_path, monkeypatch)
    scoped = scoped_settings(settings, profile="work", data_dir=tmp_path / "w",
                             bank_id="hermes-work", credential_scope="profile-work")
    assert scoped.db_path == tmp_path / "w" / "canonical.db"
    assert scoped.blob_dir == tmp_path / "w" / "blobs"
    assert (scoped.profile, scoped.bank_id, scoped.credential_scope) == \
           ("work", "hermes-work", "profile-work")
    # The policy stays the operator's: another profile is another person's memory,
    # not another permission to reach a model or spend a budget.
    for field in ("hindsight_url", "allowed_inference_hosts", "inference_enabled",
                  "background_budget_tokens", "owner_principal", "text_route",
                  "max_output_tokens", "foreground_deadline_s", "gate_token", "home"):
        assert getattr(scoped, field) == getattr(settings, field), field


@pytest.mark.parametrize("field", ["profile", "bank_id", "credential_scope"])
@pytest.mark.parametrize("junk", ["", "   ", None, 6])
def test_a_scoped_configuration_must_name_everything_it_scopes(tmp_path, monkeypatch,
                                                               field, junk):
    settings = instance(tmp_path, monkeypatch)
    arguments = {"profile": "work", "data_dir": tmp_path / "w", "bank_id": "hermes-work",
                 "credential_scope": "profile-work"}
    arguments[field] = junk
    with pytest.raises(SettingError, match=f"{field} must be nonempty text"):
        scoped_settings(settings, **arguments)


def test_a_relative_data_directory_cannot_be_smuggled_in(tmp_path, monkeypatch):
    settings = instance(tmp_path, monkeypatch)
    with pytest.raises(SettingError, match="must be absolute"):
        scoped_settings(settings, profile="work", data_dir="elsewhere",
                        bank_id="hermes-work", credential_scope="profile-work")


def test_a_scoped_profile_never_reads_the_default_profiles_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_KEY", "default-key")
    monkeypatch.setenv("PROFILE_WORK_HINDSIGHT_API_KEY", "work-key")
    settings = instance(tmp_path, monkeypatch)
    work = scoped_settings(settings, profile="work", data_dir=tmp_path / "w",
                           bank_id="hermes-work", credential_scope="profile-work")
    assert scoped_secret(work, "HINDSIGHT_API_KEY") == "work-key"
    assert scoped_secret(settings, "HINDSIGHT_API_KEY") == "default-key"


def test_a_profile_without_its_own_secret_has_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_KEY", "default-key")
    monkeypatch.delenv("PROFILE_WORK_HINDSIGHT_API_KEY", raising=False)
    work = scoped_settings(instance(tmp_path, monkeypatch), profile="work",
                           data_dir=tmp_path / "w", bank_id="hermes-work",
                           credential_scope="profile-work")
    assert scoped_secret(work, "HINDSIGHT_API_KEY") is None, \
        "falling back would sign one profile's requests with another's key"


@pytest.mark.parametrize("name", [None, "", "   ", 7])
def test_a_secret_with_no_name_is_no_secret(tmp_path, monkeypatch, name):
    monkeypatch.setenv("HINDSIGHT_API_KEY", "default-key")
    assert scoped_secret(instance(tmp_path, monkeypatch), name) is None


# -- the background pass -------------------------------------------------------

def test_the_pass_is_scheduled_by_default_and_the_period_is_the_operators_number(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_MAINTENANCE_INTERVAL_S": "300"})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    assert load_settings().maintenance_interval_s == 300


def test_a_store_with_no_such_setting_still_gets_a_period(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "home"))
    assert load_settings().maintenance_interval_s == 900


def test_zero_hands_the_timer_back_to_the_owner(tmp_path, monkeypatch):
    """Not an error and not a default: an installation that will be run by hand."""
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_MAINTENANCE_INTERVAL_S": "0"})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    assert load_settings().maintenance_interval_s == 0


@pytest.mark.parametrize("value, message", [
    ("-1", "outside the admissible range"),
    ("99999999", "outside the admissible range"),
    ("soon", "whole number of seconds"),
    ("", "whole number of seconds"),
])
def test_a_period_that_cannot_be_meant_is_refused_rather_than_guessed(
        tmp_path, monkeypatch, value, message):
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_MAINTENANCE_INTERVAL_S": value})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match=message):
        load_settings()


def test_the_refusal_names_the_way_to_turn_the_loop_off(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_MAINTENANCE_INTERVAL_S": "-5"})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError) as error:
        load_settings()
    assert "0 means the pass never runs unattended" in str(error.value)


# -- the evaluator door ---------------------------------------------------------

def test_no_evaluator_is_configured_until_a_program_is_named(tmp_path, monkeypatch):
    """A default that could promote a rule would be a rule promoted by nobody."""
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "home"))
    settings = load_settings()
    assert settings.evaluator_command == ()
    assert settings.evaluator_env == ()
    assert settings.evaluator_timeout_s == 120.0


def test_the_evaluator_is_a_program_and_its_arguments(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_EVALUATOR_COMMAND": "/usr/bin/pytest -q suite",
                     "HERMES_MEMORY_EVALUATOR_TIMEOUT_S": "30"})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    settings = load_settings()
    assert settings.evaluator_command == ("/usr/bin/pytest", "-q", "suite")
    assert settings.evaluator_timeout_s == 30.0


def test_a_bare_evaluator_word_is_refused_because_path_is_somebody_elses(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_EVALUATOR_COMMAND": "pytest -q"})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match="absolute path"):
        load_settings()


@pytest.mark.parametrize("value", ["0", "7200", "soon"])
def test_an_evaluator_timeout_that_cannot_be_meant_is_refused(tmp_path, monkeypatch, value):
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_EVALUATOR_TIMEOUT_S": value})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match="EVALUATOR_TIMEOUT_S"):
        load_settings()


def test_the_environment_an_evaluator_may_see_is_a_list_of_names(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_EVALUATOR_ENV": " FIXTURE_MODE , OTHER, FIXTURE_MODE "})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    assert load_settings().evaluator_env == ("FIXTURE_MODE", "OTHER")


def test_an_entry_that_is_not_a_variable_name_is_refused_at_load(tmp_path, monkeypatch):
    """Checked on the way in, so a typo cannot widen what a scored run gets to read."""
    home = tmp_path / "home"
    write_env(home, {"HERMES_MEMORY_EVALUATOR_ENV": "FIXTURE_MODE,/etc/passwd"})
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match="EVALUATOR_ENV"):
        load_settings()
