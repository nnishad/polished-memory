"""The inventory: what is here, read without touching anything.

These tests are mostly about two things — that a collision is *seen*, and that a fact
that cannot be read is reported as unreadable rather than invented.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import ssl
from pathlib import Path

import pytest

from hermes_memory.install import inventory as inventory_module
from hermes_memory.install.inventory import (BLOCKING_MARKERS, blocking, blocks_setup,
                                            conflicts, listening_ports, provider_selection,
                                            survey)
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
    """A subclass that refuses to be built, rather than a function wearing its place.

    The claim is about construction: no socket is opened. Replacing `socket.socket` with a
    plain function also makes any module that subclasses it unimportable, which turned this
    into a test of whichever import order the suite happened to hit — a read-only report may
    import a client library; it may not connect through one.
    """
    before = {path for path in tmp_path.rglob("*")}
    opened = []
    real_socket = socket.socket
    assert real_socket in ssl.SSLSocket.__mro__, \
        "the TLS class has to be built before anything is swapped out"

    class Refusing(real_socket):
        def __init__(self, *args, **kwargs):
            opened.append("socket")
            raise AssertionError("an inventory must not open a socket")

    socket.socket = Refusing
    try:
        report = probe(installation, host_home)
    finally:
        socket.socket = real_socket
    assert {path for path in tmp_path.rglob("*")} == before
    assert report["endpoints"]["probed"] is False
    assert opened == []


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


def test_a_backup_of_the_store_is_not_a_second_capture_owner(installation):
    """`backup` copies the store into the tree beside it, which is one owner and one copy.

    Counting the copy made a routine backup block the next setup run — "one profile has one
    capture owner" was said about a profile with exactly one live store — and put the snapshot
    list inside the inputs of a plan whose steps had not moved, so every backup taken expired
    an approval nobody had reconsidered.
    """
    home, settings = installation
    settings.data_dir.mkdir(parents=True, mode=0o700)
    with EvidenceStore(settings.db_path):
        pass
    taken = settings.data_dir / "snapshots" / "20260927T172823Z-snap_fc9076"
    (taken / "pre-restore").mkdir(parents=True)
    (taken / "canonical.db").write_bytes(b"")
    (taken / "pre-restore" / "canonical.db").write_bytes(b"")

    report = probe(installation, home)
    assert report["capture_owners"]["canonical_stores"] == [str(settings.db_path)]
    assert not any("capture owner" in line for line in conflicts(report)), conflicts(report)


def test_a_store_under_a_directory_named_snapshots_is_still_a_live_owner(installation,
                                                                        tmp_path):
    """The exclusion is this tree's backups, not the word appearing somewhere in a path.

    A profile whose data directory happens to sit beneath a directory called `snapshots` is a
    capture owner like any other, and an inventory blind to it would let a second one be
    enrolled over a profile that was already serving turns.
    """
    from dataclasses import replace

    home, settings = installation
    living = tmp_path / "snapshots" / "work"
    living.mkdir(parents=True)
    (living / "canonical.db").write_text("", encoding="utf-8")
    settings = replace(settings, data_dir=living, db_path=living / "canonical.db",
                       blob_dir=living / "blobs")

    report = survey(settings, hermes_home=home, environ={}, proc=None)
    assert report["capture_owners"]["canonical_stores"] == [str(living / "canonical.db")]


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


def a_holder(proc, pid, *, inode, owner, program="hindsight", interpreter=None):
    """A process that holds one listening socket and names one installation home."""
    descriptors = proc / pid / "fd"
    descriptors.mkdir(parents=True)
    os.symlink(f"socket:[{inode}]", descriptors / "3")
    argv = ([f"/usr/bin/{interpreter}"] if interpreter else []) + [
        f"/usr/bin/{program}", "--home", str(owner)]
    (proc / pid / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")


def a_port_8080_in_use(tmp_path, *, owner=None, pid="4242", program="hindsight",
                      interpreter=None, inode="9001", mentioner=None, unreadable=None,
                      other_stack=None):
    """A /proc with one listener on the backend's port, written by hand.

    A real /proc cannot be used: the holder has to be exactly the process the test is about,
    and never whatever else happened to be listening while the test ran. The inode is the
    whole mechanism — a port says something is there, and the inode says which process.

    `owner` is the installation home the holder's command line names, so `None` leaves the
    port held by an unnamed process; `mentioner` adds a lower-numbered process that names a
    home in its command line without holding the socket, `unreadable` one with no descriptor
    table to walk at all, and `other_stack` a `(inode, owner)` pair listening on v6 as well.
    All three are what a real /proc is mostly made of.
    """
    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    (proc / "net" / "tcp").write_text(
        "  sl  local_address rem_address   st\n"
        f"   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 00:00000000 "
        f"00000000  1000        0 {inode}\n", encoding="utf-8")
    if other_stack is not None:
        family_inode, family_owner = other_stack
        (proc / "net" / "tcp6").write_text(
            "  sl  local_address rem_address   st\n"
            "   0: 00000000000000000000000000000000:1F90 "
            "00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 "
            f"00000000  1000        0 {family_inode}\n", encoding="utf-8")
        if family_owner is not None:
            a_holder(proc, "5555", inode=family_inode, owner=family_owner)
    if mentioner is not None:
        (proc / "1111" / "fd").mkdir(parents=True)
        os.symlink("/dev/null", proc / "1111" / "fd" / "0")
        (proc / "1111" / "cmdline").write_bytes(
            b"\0".join([b"/usr/bin/cat", f"--read {mentioner}/hermes-memory.env".encode()])
            + b"\0")
    if unreadable is not None:
        # Somebody else's process: no directory to walk at all.
        (proc / unreadable).mkdir(parents=True)
    if owner is not None:
        a_holder(proc, pid, inode=inode, owner=owner, program=program,
                 interpreter=interpreter)
    return proc


def backend_on_8080(home):
    """The installation's own config, pointed at a port the caller has put a listener on."""
    (home / "hermes-memory.env").write_text(
        config(home, OWNER_PRINCIPAL=OWNER, HINDSIGHT_URL="http://127.0.0.1:8080"),
        encoding="utf-8")
    from hermes_memory.config import load_settings

    return load_settings()


def test_a_port_held_by_this_installation_s_own_process_is_not_a_collision(installation,
                                                                           tmp_path):
    """Re-running setup against a running installation is a restart, not a fight over a port.

    The port is wanted, something is listening, and the listener's own command line names this
    installation's home. Stopping that run from proceeding would make `stop, then setup` the only
    way through, for no collision at all.
    """
    home, _ = installation
    settings = backend_on_8080(home)
    proc = a_port_8080_in_use(tmp_path, owner=home, mentioner=home)

    report = survey(settings, hermes_home=tmp_path, environ={}, proc=proc)
    assert report["endpoints"]["held_by_ours"] == {
        "hindsight": {"pid": "4242", "program": "hindsight"}}
    said = conflicts(report)
    assert any("4242" in line and "replaces" in line for line in said), said
    assert not any("already listening" in line for line in said), \
        "this installation's own listener must not carry the blocking sentence"
    assert blocking(report) == [], \
        "the sentences worth saying before a run are not the ones that stop it"


def test_a_process_whose_descriptors_cannot_be_read_does_not_end_the_search(installation,
                                                                            tmp_path):
    """Almost every process on a machine belongs to somebody else, and has no table to walk.

    One unreadable directory has to be stepped over: stopping there would report this
    installation's own listener as an unknown, which is a collision the operator would be
    told to go and clear by hand.
    """
    home, _ = installation
    settings = backend_on_8080(home)
    proc = a_port_8080_in_use(tmp_path, owner=home, unreadable="999")

    report = survey(settings, hermes_home=tmp_path, environ={}, proc=proc)
    assert report["endpoints"]["held_by_ours"] == {
        "hindsight": {"pid": "4242", "program": "hindsight"}}
    assert report["endpoints"]["unattributed"] == []


def test_the_gate_closes_on_collisions_and_on_nothing_else(installation, tmp_path, monkeypatch):
    """Eight reasons to say something, of which exactly five are reasons to stop.

    This is the whole content of `inventory --conflicts`'s exit status. A stranger on a wanted
    port, a second capture owner, no hermes to register against, no principal who can confirm a
    forgetting and a host registered at somebody else's commit each mean a run here would act
    on something that is not this installation's to act on. A loose permission, a credential
    left at rest and a delivery switch with no destination are this installation's own business,
    worth fixing and no reason to refuse the run. Count the partition rather than spot-checking
    it: a sentence that quietly moved from one side to the other is the difference between a
    healthy installation reading as blocked and a broken one reading as fine (§10.4).
    """
    home, _ = installation
    (home / "hermes-memory.env").write_text(
        config(home, HINDSIGHT_URL="http://127.0.0.1:8080",
               HINDSIGHT_API_KEY="a-secret-left-in-a-config-file",
               DELIVERY_ENABLED="true",
               DELIVERY_TARGET="nobody-has-agreed-to-this"), encoding="utf-8")
    (home / "hermes-memory.env").chmod(0o644)
    (home / "data" / "live").mkdir(parents=True)
    (home / "data" / "live" / "canonical.db").write_text("", encoding="utf-8")
    (home / "data" / "stray").mkdir(parents=True)
    (home / "data" / "stray" / "canonical.db").write_text("", encoding="utf-8")
    a_host_registered_at(tmp_path, "c" * 40)
    staged = a_release_cut_at(tmp_path / "staged", "d" * 40)
    monkeypatch.setattr(inventory_module, "shutil", FakePath({}))

    from hermes_memory.config import load_settings

    report = survey(load_settings(), hermes_home=tmp_path,
                    environ={"HERMES_MEMORY_RELEASE": str(staged)},
                    proc=a_port_8080_in_use(tmp_path))
    said = conflicts(report)
    assert len(said) == 8, said

    stopping = blocking(report)
    assert len(stopping) == 5, stopping
    assert all(blocks_setup(line) for line in stopping)
    for marker in BLOCKING_MARKERS:
        assert any(marker in line for line in stopping), marker
    assert not any("readable beyond its owner" in line for line in stopping)
    assert not any("holds a credential value" in line for line in stopping)
    assert not any("no usable destination" in line for line in stopping)


def test_a_port_held_by_a_stranger_on_another_stack_is_still_a_collision(installation,
                                                                        tmp_path):
    """This installation's own v4 listener does not speak for whoever holds the v6 socket.

    A port has one answer per address family, and a plan that read only the first would call
    the port ours, restart into a bind failure, and have the owner's approval on the strength
    of a half-read machine.
    """
    home, _ = installation
    settings = backend_on_8080(home)

    shared = survey(settings, hermes_home=tmp_path, environ={}, proc=a_port_8080_in_use(
        tmp_path, owner=home, other_stack=("7777", tmp_path / "other-installation")))
    assert shared["endpoints"]["held_by_ours"] == {}
    assert any("already listening" in line for line in conflicts(shared)), conflicts(shared)
    assert any("already listening" in line for line in blocking(shared)), \
        "somebody else's listener is the case that does stop the run"

    both = survey(settings, hermes_home=tmp_path, environ={}, proc=a_port_8080_in_use(
        tmp_path / "ours", owner=home, other_stack=("7777", home)))
    assert both["endpoints"]["held_by_ours"] == {
        "hindsight": {"pid": "4242", "program": "hindsight"}}


def test_a_port_held_by_another_installation_is_still_a_collision(installation, tmp_path):
    """A listener whose command line names some other home is a stranger on our port.

    Same program, same port, different installation: only the owner of the listener may be
    replaced by a run here, and a second installation on one port is the failure the whole
    inventory exists to catch.
    """
    home, _ = installation
    settings = backend_on_8080(home)
    proc = a_port_8080_in_use(tmp_path, owner=tmp_path / "other-installation")

    report = survey(settings, hermes_home=tmp_path, environ={}, proc=proc)
    assert report["endpoints"]["held_by_ours"] == {}
    assert report["endpoints"]["unattributed"] == []
    assert any("already listening" in line for line in conflicts(report))


def test_a_port_held_by_a_process_that_cannot_be_named_says_what_is_unknown(installation,
                                                                           tmp_path):
    """Not finding the owner is not the same as finding a stranger, and neither is a licence.

    The port keeps blocking either way; what the report owes the operator is which fact was
    missing, because that is the difference between a machine to re-check and a machine to
    clear by hand.
    """
    home, _ = installation
    settings = backend_on_8080(home)
    proc = a_port_8080_in_use(tmp_path)

    report = survey(settings, hermes_home=tmp_path, environ={}, proc=proc)
    assert report["endpoints"]["held_by_ours"] == {}
    assert report["endpoints"]["unattributed"] == ["hindsight"]
    assert any("already listening" in line for line in conflicts(report))
    assert any("cannot be told apart" in line for line in report["unknowns"]), report["unknowns"]


def test_a_listener_is_named_by_its_program_rather_than_by_what_ran_it(installation, tmp_path):
    """A shebang makes the kernel report `python /…/bin/hindsight-api --home …`.

    "Held by this installation's own python" answers nothing an operator asked: which unit is
    sitting on the port is in the second word, and the units are written with that interpreter
    first.
    """
    home, _ = installation
    settings = backend_on_8080(home)
    proc = a_port_8080_in_use(tmp_path, owner=home,
                              interpreter="python", program="hindsight-api")

    report = survey(settings, hermes_home=tmp_path, environ={}, proc=proc)
    assert report["endpoints"]["held_by_ours"]["hindsight"]["program"] == "hindsight-api"
    assert "python" not in " ".join(conflicts(report)), conflicts(report)


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


# -- the host's plugin and the runtime's release --------------------------------

def a_host_registered_at(home, revision):
    """The record the host keeps for its own plugin registration."""
    (home / "plugins").mkdir(parents=True, exist_ok=True)
    (home / "plugins" / ".install-metadata.json").write_text(
        json.dumps({"hermes-memory": {"revision": revision}}), encoding="utf-8")
    return home


def a_release_cut_at(root, commit):
    """A staged release, with the record the staging door leaves beside the ``bin/``."""
    (root / "deployment").mkdir(parents=True, exist_ok=True)
    (root / "RELEASE.json").write_text(json.dumps({"source_commit": commit}), encoding="utf-8")
    return root


def test_a_host_on_another_commit_than_its_runtime_is_reported_as_split(installation, tmp_path):
    """The plugin answering the host and the code answering the plugin are one contract.

    §10.3 pairs the wheel and the plugin from one revision; a re-release that left the host's
    registration pointing at the old commit splits that contract in half, and every other
    number here still reads as healthy. Only the host's own --force may close the gap, so the
    inventory names both commits rather than waiting for a setup run to refuse.
    """
    home, settings = installation
    registered = a_host_registered_at(tmp_path / "host", "c" * 40)
    staged = a_release_cut_at(tmp_path / "release", "d" * 40)

    report = survey(settings, hermes_home=registered,
                    environ={"HERMES_MEMORY_RELEASE": str(staged)}, proc=None)
    assert report["host"]["plugin_registered_revision"] == "c" * 40
    assert report["release"]["source_commit"] == "d" * 40
    said = conflicts(report)
    assert any("two different revisions" in line for line in said), said
    assert any("cccccccccccc" in line and "dddddddddddd" in line for line in blocking(report)), \
        blocking(report)

    # The units do not always name the release: an installation that only has the pointer is
    # the ordinary one, and the split has to be visible from that too.
    a_release_cut_at(home / "runtime" / "current", "f" * 40)
    fell_back = survey(settings, hermes_home=registered, environ={}, proc=None)
    assert fell_back["release"]["source_commit"] == "f" * 40
    assert any("two different revisions" in line for line in conflicts(fell_back)), \
        conflicts(fell_back)


def test_a_registration_that_cannot_be_compared_is_not_called_a_split(installation, tmp_path):
    """Two records that are simply absent are not evidence of a mismatch.

    A plugin installed by hand, or a tree that was never staged by the release door, has no
    commit to compare. The honest answer is that this installation says nothing about it —
    not a blocking sentence that sends somebody off to fix a thing that is not broken.
    """
    home, settings = installation
    unrecorded = a_host_registered_at(tmp_path / "handmade", "c" * 40)
    bare = tmp_path / "unstaged"
    (bare / "deployment").mkdir(parents=True)

    report = survey(settings, hermes_home=unrecorded,
                    environ={"HERMES_MEMORY_RELEASE": str(bare)}, proc=None)
    assert report["release"]["source_commit"] == "not recorded"
    assert not any("two different revisions" in line for line in conflicts(report)), \
        conflicts(report)

    aligned = survey(settings, hermes_home=a_host_registered_at(tmp_path / "same", "e" * 40),
                     environ={"HERMES_MEMORY_RELEASE":
                              str(a_release_cut_at(tmp_path / "cut", "e" * 40))}, proc=None)
    assert not any("two different revisions" in line
                   for line in conflicts(aligned)), conflicts(aligned)


def test_the_ledger_is_reported_present_only_when_it_is_actually_there(installation,
                                                                       host_home):
    _, settings = installation
    assert probe(installation, host_home)["installation"]["ledger_present"] is False
    (settings.home / "installation.db").write_bytes(b"")
    assert probe(installation, host_home)["installation"]["ledger_present"] is True
