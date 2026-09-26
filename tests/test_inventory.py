"""The inventory: what is here, read without touching anything.

These tests are mostly about two things — that a collision is *seen*, and that a fact
that cannot be read is reported as unreadable rather than invented.
"""
from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path

import pytest

from hermes_memory.install import inventory as inventory_module
from hermes_memory.install.inventory import (conflicts, listening_ports,
                                            provider_selection, survey)
from hermes_memory.storage.evidence import EvidenceStore

OWNER = "jugaadu"


class FakePath:
    """A PATH with known contents.

    Whether Hermes happens to be installed on the machine running these tests is not a
    fact about the installation under test, and an inventory that read the real PATH
    would pass here and fail anywhere else.
    """

    def __init__(self, found):
        self._found = dict(found)

    def which(self, name):
        return self._found.get(name)

    def disk_usage(self, target):
        return shutil.disk_usage(target)


class FakeMetadata:
    """An installed-distribution table with known contents, for the same reason."""

    class PackageNotFoundError(Exception):
        pass

    def __init__(self, versions):
        self._versions = dict(versions)

    def version(self, name):
        if name not in self._versions:
            raise self.PackageNotFoundError(name)
        return self._versions[name]


@pytest.fixture(autouse=True)
def a_host_that_is_installed(monkeypatch):
    monkeypatch.setattr(inventory_module, "shutil",
                        FakePath({"hermes": "/opt/hermes/bin/hermes", "uv": "/usr/bin/uv"}))
    monkeypatch.setattr(inventory_module, "metadata",
                        FakeMetadata({"hermes-agent": "3.1.0", "hermes-memory": "0.9.2"}))


def config(home, **extra):
    lines = [f"HERMES_MEMORY_DATA_DIR={home / 'data'}",
             "HERMES_MEMORY_INFERENCE_ENABLED=false"]
    lines += [f"HERMES_MEMORY_{key}={value}" for key, value in extra.items()]
    return "\n".join(lines) + "\n"


@pytest.fixture()
def installation(tmp_path, monkeypatch):
    from hermes_memory.config import load_settings

    home = tmp_path / "instance"
    home.mkdir()
    (home / "hermes-memory.env").write_text(config(home, OWNER_PRINCIPAL=OWNER),
                                            encoding="utf-8")
    (home / "hermes-memory.env").chmod(0o600)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.delenv("HERMES_MEMORY_HINDSIGHT_API_KEY", raising=False)
    return home, load_settings()


@pytest.fixture()
def host_home(tmp_path):
    """A Hermes profile home with a memory block that selects another provider."""
    root = tmp_path / "hermes"
    (root / "memory").mkdir(parents=True)
    (root / "config.yaml").write_text(
        "model:\n  provider: \"auto\"\n\n"
        "memory:\n"
        "  # Agent's personal notes\n"
        "  memory_enabled: true\n"
        "  provider: honcho\n",
        encoding="utf-8")
    (root / ".install_method").write_text("git-checkout\n", encoding="utf-8")
    return root


def probe(installation, host_home, **kwargs):
    _, settings = installation
    return survey(settings, hermes_home=host_home, environ={},
                  proc=kwargs.pop("proc", None), **kwargs)


# -- it reads, it does not do --------------------------------------------------

def test_the_inventory_writes_nothing_and_opens_no_socket(installation, host_home, tmp_path):
    before = {path for path in tmp_path.rglob("*")}
    monkeypatch_socket = []
    real_socket = socket.socket

    def refusing(*args, **kwargs):
        monkeypatch_socket.append("socket")
        raise AssertionError("an inventory must not open a socket")

    socket.socket = refusing
    try:
        report = probe(installation, host_home)
    finally:
        socket.socket = real_socket
    assert {path for path in tmp_path.rglob("*")} == before
    assert report["endpoints"]["probed"] is False
    assert "socket" not in json.dumps(monkeypatch_socket)


def test_the_report_is_plain_data(installation, host_home):
    json.dumps(probe(installation, host_home), default=str)


def test_what_cannot_be_read_is_said_so(installation, tmp_path):
    missing = tmp_path / "no-hermes-here"
    report = probe(installation, missing)
    assert report["host"]["memory_provider"] == "no config.yaml to read"
    assert any("memory provider selection" in item for item in report["unknowns"])
    assert any("listening-port table" in item for item in report["unknowns"]), \
        "an unreadable port table cannot rule out a collision"


# -- the installation's own facts ---------------------------------------------

def test_a_store_on_disk_is_reported_as_present(installation):
    home, settings = installation
    settings.data_dir.mkdir(parents=True, mode=0o700)
    with EvidenceStore(settings.db_path):
        pass
    report = probe(installation, home)
    assert report["installation"]["store_present"] is True
    assert report["capture_owners"]["canonical_stores"] == [str(settings.db_path)]


def test_delivery_needs_a_destination_and_says_which_kind_it_has(installation, tmp_path):
    home, settings = installation
    (home / "hermes-memory.env").write_text(
        config(home, OWNER_PRINCIPAL=OWNER, DELIVERY_ENABLED="true",
               DELIVERY_TARGET="signal:+4915112345678"), encoding="utf-8")
    from hermes_memory.config import load_settings

    report = survey(load_settings(), hermes_home=tmp_path, environ={}, proc=None)
    assert report["installation"]["delivery"] == "enabled"
    assert "4915112345678" not in json.dumps(report), \
        "a destination address has no business in a status report"
    assert report["installation"]["delivery_target"] == "signal:set"


def test_a_credential_in_the_environment_is_a_name_not_a_value(installation, tmp_path):
    """The designed arrangement: the file names the variable, the variable holds it."""
    home, _ = installation
    (home / "hermes-memory.env").write_text(
        config(home, OWNER_PRINCIPAL=OWNER, HINDSIGHT_API_KEY_ENV="HERMES_MEMORY_HINDSIGHT_API_KEY"),
        encoding="utf-8")
    from hermes_memory.config import load_settings

    report = survey(load_settings(), hermes_home=tmp_path, environ={}, proc=None)
    assert report["installation"]["secrets_in_environment"] == []
    assert report["installation"]["credential_variables_named"] == \
           ["HERMES_MEMORY_HINDSIGHT_API_KEY"]


def test_a_credential_written_into_the_config_file_is_reported_without_being_echoed(
        installation, tmp_path):
    home, _ = installation
    (home / "hermes-memory.env").write_text(
        config(home, OWNER_PRINCIPAL=OWNER) + "HERMES_MEMORY_HINDSIGHT_API_KEY=hunter2\n",
        encoding="utf-8")
    from hermes_memory.config import load_settings

    report = survey(load_settings(), hermes_home=tmp_path, environ={}, proc=None)
    assert report["installation"]["secrets_at_rest_in_config"] == \
           ["HERMES_MEMORY_HINDSIGHT_API_KEY"]
    assert "hunter2" not in json.dumps(report)
    assert any("holds a credential value" in line for line in conflicts(report))


def test_the_host_selection_is_reported_without_being_trusted(installation, host_home):
    report = probe(installation, host_home)
    assert report["host"]["memory_provider"] == "honcho"
    assert report["host"]["install_method"] == "git-checkout"
    assert report["host"]["guarded_delivery_supported"] is False, \
        "H1 is a host gap; claiming it here would promise a guarantee nobody enforces"


# -- the host config file ------------------------------------------------------

def test_the_provider_line_is_found_through_the_comments_a_real_file_has(host_home):
    assert provider_selection(host_home / "config.yaml") == "honcho"


def test_a_model_provider_is_not_mistaken_for_the_memory_one(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text('model:\n  provider: "auto"\n', encoding="utf-8")
    assert provider_selection(path) == "unknown"


@pytest.mark.parametrize("body,expected", [
    ("memory:\n  provider: hermes-memory\n", "hermes-memory"),
    ("memory:\n  provider: 'hermes-memory'  # chosen at setup\n", "hermes-memory"),
    ("memory:\n  provider:\n", "unset"),
    ("memory:\n    provider: over-indented\n", "unknown"),
    ("provider: top-level\n", "unknown"),
    ("memory:\n  provider: [\n", "unknown"),
    ("memory:\n\tprovider: tabs\n", "unknown"),
])
def test_only_the_shape_the_host_writes_is_read(tmp_path, body, expected):
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    assert provider_selection(path) == expected


def test_an_unreadable_config_says_so(tmp_path):
    assert provider_selection(tmp_path / "absent.yaml") == "unreadable"


# -- ports ---------------------------------------------------------------------

def test_only_listening_sockets_are_counted(tmp_path):
    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    (proc / "net" / "tcp").write_text(
        "  sl  local_address rem_address   st\n"
        "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000\n"   # 127.0.0.1:8080
        "   1: 00000000:1F91 0100007F:C000 01 00000000:00000000\n"   # established: 8081
        "   2: 00000000:ZZZZ 00000000:0000 0A 00000000:00000000\n"   # malformed
        "   3: 00000000:1F92\n",                                     # truncated
        encoding="utf-8")
    (proc / "net" / "tcp6").write_text(
        "  sl  local_address rem_address   st\n"
        "   0: 00000000000000000000000001000000:1F93 0000000000000000"
        "0000000000000000:0000 0A 00\n", encoding="utf-8")
    assert listening_ports(proc) == {8080, 8083}


def test_a_missing_proc_is_no_ports_at_all(tmp_path):
    assert listening_ports(tmp_path / "nowhere") == set()
    assert listening_ports(None) == set()


def test_a_wanted_port_already_in_use_is_a_conflict(installation, tmp_path):
    home, settings = installation
    (home / "hermes-memory.env").write_text(
        config(home, OWNER_PRINCIPAL=OWNER, HINDSIGHT_URL="http://127.0.0.1:8080"),
        encoding="utf-8")
    from hermes_memory.config import load_settings

    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    (proc / "net" / "tcp").write_text(
        "  sl  local_address rem_address   st\n"
        "   0: 0100007F:1F90 00000000:0000 0A 00\n", encoding="utf-8")
    report = survey(load_settings(), hermes_home=tmp_path, environ={}, proc=proc)
    assert report["endpoints"]["wanted"]["hindsight"] == {"port": 8080, "in_use": True}
    assert any("8080" in line and "listening" in line for line in conflicts(report))


def test_a_free_port_is_not_reported_as_a_problem(installation, tmp_path):
    home, settings = installation
    (home / "hermes-memory.env").write_text(
        config(home, OWNER_PRINCIPAL=OWNER, HINDSIGHT_URL="http://127.0.0.1:8080"),
        encoding="utf-8")
    from hermes_memory.config import load_settings

    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    (proc / "net" / "tcp").write_text("  sl  local_address rem_address   st\n",
                                      encoding="utf-8")
    report = survey(load_settings(), hermes_home=tmp_path, environ={}, proc=proc)
    assert report["endpoints"]["wanted"]["hindsight"]["in_use"] is False
    assert conflicts(report) == []


# -- two capture owners --------------------------------------------------------

def test_a_second_spool_under_the_same_profile_home_is_named(installation, tmp_path):
    home, settings = installation
    spool = home / "hermes-memory"
    spool.mkdir(parents=True)
    (spool / "capture-spool.db").write_text("", encoding="utf-8")
    settings.data_dir.mkdir(parents=True, mode=0o700)
    with EvidenceStore(settings.db_path):
        pass
    report = probe(installation, home)
    assert report["capture_owners"]["capture_spools"] == [str(spool / "capture-spool.db")]
    # One store and one spool is a normal installation; two stores is not.
    assert conflicts(report) == []
    (settings.data_dir / "stray").mkdir()
    (settings.data_dir / "stray" / "canonical.db").write_text("", encoding="utf-8")
    assert any("capture owner" in line for line in conflicts(probe(installation, home)))


def test_a_loose_permission_on_the_configuration_is_a_problem(installation, host_home):
    home, _ = installation
    (home / "hermes-memory.env").chmod(0o644)
    report = probe(installation, host_home)
    assert report["installation"]["world_readable"] is True
    assert any("readable beyond its owner" in line for line in conflicts(report))


def test_a_host_that_is_not_on_the_path_stops_a_setup_run(installation, host_home,
                                                         monkeypatch):
    """Registration cannot be verified from here, so the run that claims it should not start."""
    monkeypatch.setattr(inventory_module, "shutil", FakePath({"uv": "/usr/bin/uv"}))
    report = probe(installation, host_home)
    assert report["host"]["hermes_executable"] == "not found"
    assert any("on PATH" in line for line in conflicts(report))


def test_no_owner_principal_blocks_forgetting_and_says_so(tmp_path, monkeypatch):
    from hermes_memory.config import load_settings

    home = tmp_path / "no-owner"
    home.mkdir()
    (home / "hermes-memory.env").write_text(config(home), encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    report = survey(load_settings(), hermes_home=tmp_path, environ={}, proc=None)
    assert report["installation"]["owner_principal"] == "unset"
    assert any("previewed" in line and "never confirmed" in line for line in conflicts(report))


def test_a_switch_with_no_destination_is_refused_at_startup(tmp_path, monkeypatch):
    """The inventory would rather not have to catch this, and does anyway."""
    from hermes_memory.config import SettingError, load_settings

    home = tmp_path / "loose"
    home.mkdir()
    (home / "hermes-memory.env").write_text(
        config(home, DELIVERY_ENABLED="true"), encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    with pytest.raises(SettingError, match="DELIVERY_TARGET"):
        load_settings()


def test_the_admission_endpoint_is_read_from_the_file_that_names_it(installation, tmp_path):
    home, _ = installation
    (home / "hermes-memory.env").write_text(
        config(home, OWNER_PRINCIPAL=OWNER, ADMISSION_URL="http://127.0.0.1:8090"),
        encoding="utf-8")
    from hermes_memory.config import load_settings

    report = survey(load_settings(), hermes_home=tmp_path, environ={}, proc=None)
    assert report["endpoints"]["wanted"]["admission"]["port"] == 8090
    assert report["endpoints"]["probed"] is False


def test_a_malformed_destination_is_reported_as_one(installation, tmp_path):
    home, _ = installation
    (home / "hermes-memory.env").write_text(
        config(home, OWNER_PRINCIPAL=OWNER, DELIVERY_ENABLED="true",
               DELIVERY_TARGET="signal-owner-1234"), encoding="utf-8")
    from hermes_memory.config import load_settings

    report = survey(load_settings(), hermes_home=tmp_path, environ={}, proc=None)
    assert report["installation"]["delivery_target"] == "malformed"
    assert any("no usable destination" in line for line in conflicts(report))


def test_the_release_is_said_as_a_version_and_a_tree(installation, tmp_path):
    report = probe(installation, tmp_path)
    assert report["release"]["framework_version"]
    assert report["release"]["python"]
    assert Path(report["release"]["package_source"]).is_dir()


def test_the_toolchain_is_named_by_where_it_was_found(installation, tmp_path):
    report = probe(installation, tmp_path)
    assert report["release"]["uv"] == "/usr/bin/uv"
    assert report["release"]["python_executable"] == inventory_module.sys.executable
    assert report["host"]["hermes_executable"] == "/opt/hermes/bin/hermes"


def test_a_host_with_nothing_installed_says_so_instead_of_guessing(installation, tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(inventory_module, "metadata", FakeMetadata({}))
    report = probe(installation, tmp_path)
    assert report["host"]["hermes_version"] == "unknown"
    assert report["release"]["framework_version"] is None
    assert any("version" in line for line in report["unknowns"])


def test_a_release_root_the_units_never_set_is_said_as_not_set(installation, tmp_path):
    report = probe(installation, tmp_path)
    assert report["release"]["release_root"] == "not set"
    _, settings = installation
    named = survey(settings, hermes_home=tmp_path,
                   environ={"HERMES_MEMORY_RELEASE": "/srv/hermes-memory/current"},
                   proc=None)
    assert named["release"]["release_root"] == "/srv/hermes-memory/current"


def test_the_ledger_is_reported_present_only_when_it_is_actually_there(installation,
                                                                       host_home):
    _, settings = installation
    assert probe(installation, host_home)["installation"]["ledger_present"] is False
    (settings.home / "installation.db").write_bytes(b"")
    assert probe(installation, host_home)["installation"]["ledger_present"] is True
