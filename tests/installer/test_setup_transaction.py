"""C14 setup transaction: eleven steps, reviewed once, resumable, and never surprising.

What these tests hold the installer to: a plan writes nothing; an approval cannot be
spent on a different plan; an interrupted run resumes rather than repeating; a step that
would overwrite something the owner wrote says so instead of deciding; the only host
commands that can run are the three built here with fixed arguments; and the canary
proves erasure, not merely that tables exist.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
from pathlib import Path

import pytest

from hermes_memory.config import load_settings
from hermes_memory.install import setup as transaction
from hermes_memory.install.profiles import ProfileRegistry
from hermes_memory.install.inventory import listening_ports
from hermes_memory.install.services import unit_directory
from hermes_memory.install.setup import STEPS, SetupError, plan, profile_name, run
from hermes_memory.install.release import carry
from hermes_memory.storage.evidence import EvidenceStore

OWNER = "jugaadu"
REF = "a" * 40


def a_free_port() -> int:
    """A port this machine is not holding, asked of the kernel rather than guessed.

    The transaction's first step refuses to plan over a port somebody else listens on, so
    naming the real backend's port here made these tests answer a question about whatever
    the developer was running: on a machine where the memory stack is up, a plan about a
    temporary instance was blocked by the installation beside it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# Decided once for the module: every test that wants a backend endpoint wants one that is
# free, and the plan's review digest has to stay reproducible across a resumed run.
BACKEND_PORT = a_free_port()
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"


def host(home, **extra):
    """A release tree that has actually been staged, and an installation beside it."""
    release = home / "release"
    (release / "bin").mkdir(parents=True, exist_ok=True)
    (release / "bin" / "hermes-memory").write_text("#!/bin/sh\n", encoding="utf-8")
    # A staged release carries its plugin beside its runtime — `release` refuses to stage one
    # that does not — and `register-plugin` now checks the two halves of the contract against
    # each other, so the fixture has to be the tree the door would actually have produced.
    carry(source=Path(__file__).resolve().parents[2], into=release, names=("integrations",))
    lines = [f"HERMES_MEMORY_DATA_DIR={home / 'data'}",
             "HERMES_MEMORY_INFERENCE_ENABLED=false",
             f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}"]
    lines += [f"HERMES_MEMORY_{key}={value}" for key, value in extra.items()]
    (home / "hermes-memory.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (home / "hermes-memory.env").chmod(0o600)
    return release


def activity(tmp_path, *, provider=None, model="deepseek-chat"):
    """A Hermes profile home with the host's own configuration file."""
    home = tmp_path / "homes" / "work"
    home.mkdir(parents=True)
    body = f"model:\n  provider: openai\n  model: {model}\n\nmemory:\n  memory_enabled: true\n"
    if provider is not None:
        body += f"  provider: {provider}\n"
    (home / "config.yaml").write_text(body, encoding="utf-8")
    return home


class Heremes:
    """The host commands, faked. Nothing in these tests touches a real installation.

    It obeys the arguments it is given rather than doing one fixed thing, because a
    fake that always wrote the answer we wanted would let the transaction build any
    command at all and still look correct.
    """

    def __init__(self, config=None, *, break_model=False, forget_provider=False,
                 fail=False):
        self.config = config
        self.calls: list[list[str]] = []
        self.break_model = break_model
        self.forget_provider = forget_provider
        self.fail = fail

    def __call__(self, argv):
        self.calls.append(list(argv))
        if self.fail:
            return 1, "host said no"
        if argv[1:3] == ["config", "set"] and self.config is not None:
            self._set(argv[3], argv[4])
        return 0, ""

    def _set(self, key, value):
        if self.forget_provider:
            return  # a host writer that accepted the command and did nothing with it
        text = self.config.read_text(encoding="utf-8")
        block, _, leaf = key.rpartition(".")
        lines = text.splitlines()
        inside = False
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not line.startswith(" ") and stripped.endswith(":"):
                inside = stripped[:-1] == block
                if inside:
                    lines.insert(index + 1, f"  {leaf}: {value}")
                    break
                continue
            if inside and stripped.startswith(f"{leaf}:"):
                lines[index] = f"  {leaf}: {value}"
                break
        else:
            lines += ["", f"{block}:", f"  {leaf}: {value}"]
        self.config.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if self.break_model:
            self.config.write_text(
                self.config.read_text(encoding="utf-8").replace("model: deepseek-chat",
                                                                "model: gpt-9"))

    @property
    def verbs(self):
        return [" ".join(call[1:3]) for call in self.calls]


@pytest.fixture()
def installation(tmp_path, monkeypatch):
    home = tmp_path / "instance"
    home.mkdir()
    release = host(home)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("HERMES_MEMORY_ADMISSION_URL", raising=False)
    settings = load_settings()
    environ = {"HERMES_MEMORY_RELEASE": str(release)}
    return settings, environ


def approve(settings, home, environ, *, expect=None, **kwargs):
    """Approve the plan as shown and run it.

    ``expect`` names steps whose blockers must be present *before* anything is approved:
    a blocker that an earlier step removes on its own is the transaction's normal shape,
    not a reason to refuse to start.
    """
    arguments = {"environ": environ, **kwargs}
    proposal = plan(settings, hermes_home=home, **arguments)
    for wanted in expect or []:
        assert any(wanted in line for line in proposal["blocked"]), proposal["blocked"]
    return run(settings, hermes_home=home, actor=settings.owner_principal,
               review=proposal["review_digest"], **arguments)


# -- the shape of the transaction --------------------------------------------

def test_the_transaction_has_the_eleven_steps_in_the_plans_order():
    assert STEPS == ("inventory", "plan", "stage", "configure", "initialize",
                     "register-plugin", "preflight", "activate", "services", "canary",
                     "finish")


def test_a_plan_writes_nothing_at_all(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    before = {path for path in tmp_path.rglob("*")}
    proposal = plan(settings, hermes_home=home, environ=environ)
    assert {path for path in tmp_path.rglob("*")} == before
    assert not (settings.home / "installation.db").exists()
    assert [item["step"] for item in proposal["steps"]] == list(STEPS)


def test_a_plan_says_what_it_could_not_do_and_which_steps_would_do_it(installation,
                                                                      tmp_path):
    settings, environ = installation
    proposal = plan(settings, hermes_home=activity(tmp_path), environ=environ)
    states = {item["step"]: item["state"] for item in proposal["steps"]}
    assert states["configure"] == "pending"
    assert states["initialize"] == "blocked", "nothing is enrolled yet, and it says so"
    assert states["activate"] == "blocked", "the provider cannot be selected without a runner"
    assert "register-plugin" in " ".join(proposal["blocked"])


def test_setup_refuses_a_relative_home(installation):
    settings, environ = installation
    with pytest.raises(SetupError, match="absolute"):
        plan(settings, hermes_home="homes/work", environ=environ)


# -- what an approval is actually about ---------------------------------------

def proc_table(tmp_path, name: str, *ports: int, owned_by: Path | None = None,
               pid: str = "4242", program: str = "hindsight", inode: str = "9001") -> Path:
    """A machine's socket table as written by the kernel, so a test can say what is on it.

    `owned_by` puts this installation's own process behind the last port, the way /proc does:
    the table names an inode, the process has that socket open, and its command line carries
    this installation's home. Every other port keeps no inode, so it is somebody unreadable's.
    """
    root = tmp_path / name
    (root / "net").mkdir(parents=True)
    rows = []
    for index, port in enumerate(ports):
        ours = owned_by is not None and port == ports[-1]
        rows.append(f"   {index}: 0100007F:{port:04X} 00000000:0000 0A 00000000:00000000"
                    + (f" 00:00000000 00000000  1000        0 {inode}\n" if ours else "\n"))
    (root / "net" / "tcp").write_text("  sl  local_address rem_address   st\n"
                                      + "".join(rows), encoding="utf-8")
    if owned_by is not None:
        descriptors = root / pid / "fd"
        descriptors.mkdir(parents=True)
        os.symlink(f"socket:[{inode}]", descriptors / "7")
        (root / pid / "cmdline").write_bytes(
            b"\0".join([f"/usr/bin/{program}".encode(), b"--config",
                        str(Path(owned_by) / "hermes-memory.env").encode()]) + b"\0")
    return root


def test_an_approval_does_not_expire_because_a_stranger_started_listening(installation,
                                                                          tmp_path):
    """The socket table is a reading of the machine, not a decision anybody approved.

    A plan that carried the whole port list in its digest expired the moment an unrelated
    program opened a listener, so the owner's approval was worth as long as the quietest
    interval on the host — which is how a re-run came to fail between showing the digest and
    spending it.
    """
    settings, environ = installation
    quiet = proc_table(tmp_path, "proc-quiet", 9999)
    busy = proc_table(tmp_path, "proc-busy", 9999, 4242, 6379)
    home = Path(settings.home)
    first = plan(settings, hermes_home=home, environ=environ, proc=quiet)
    second = plan(settings, hermes_home=home, environ=environ, proc=busy)
    assert first["review_digest"] == second["review_digest"], (
        [item["actions"] for item in first["steps"]][:1],
        [item["actions"] for item in second["steps"]][:1])


def test_a_port_this_installation_wants_being_held_is_said_and_changes_the_plan(tmp_path,
                                                                               monkeypatch):
    """Excluding the machine's own noise must not blind the inventory to a real collision."""
    home = tmp_path / "instance"
    home.mkdir()
    release = host(home, HINDSIGHT_URL="http://127.0.0.1:8888")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    settings = load_settings()
    environ = {"HERMES_MEMORY_RELEASE": str(release)}
    free = plan(settings, hermes_home=home, environ=environ,
                proc=proc_table(tmp_path, "free", 9999))
    held = plan(settings, hermes_home=home, environ=environ,
                proc=proc_table(tmp_path, "held", 9999, 8888))
    assert free["review_digest"] != held["review_digest"]
    assert any("already listening" in line for line in held["blocked"] if "hindsight" in line), \
        held["blocked"]
    said = " ".join(held["steps"][0]["actions"])
    assert "1 of 2 endpoint(s)" in said and "hindsight" in said, held["steps"][0]["actions"]
    assert "9999" not in said, "a stranger's listener is not this installation's to report"


def test_a_running_installation_is_replaced_rather_than_refused(tmp_path, monkeypatch):
    """`stop, setup, start` was the only way through, for no collision at all.

    The port this installation wants is held by a process whose own command line names this
    installation's home, read out of /proc. A plan that called that a collision would be
    refusing to restart itself, and asking the service manager instead would have made a
    read-only step run host commands.
    """
    home = tmp_path / "instance"
    home.mkdir()
    release = host(home, HINDSIGHT_URL="http://127.0.0.1:8888")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    settings = load_settings()
    environ = {"HERMES_MEMORY_RELEASE": str(release)}
    proc = proc_table(tmp_path, "held", 9999, 8888)

    running = plan(settings, hermes_home=home, environ=environ,
                   proc=proc_table(tmp_path, "owned", 9999, 8888, owned_by=home))
    assert not any("already listening" in line for line in running["blocked"]), \
        running["blocked"]
    said = " ".join(running["steps"][0]["advisory"])
    assert "4242" in said and "replaces" in said, running["steps"][0]["advisory"]

    stranger = plan(settings, hermes_home=home, environ=environ, proc=proc)
    assert any("already listening" in line for line in stranger["blocked"]), stranger["blocked"]
    # Attribution changes the verdict on the port and nothing else: the steps, and so the
    # work an owner is being asked to approve, are identical either way.
    assert ({item["step"]: item["actions"] for item in stranger["steps"]}
            == {item["step"]: item["actions"] for item in running["steps"]}), (
        [item["actions"] for item in running["steps"]][:1],
        [item["actions"] for item in stranger["steps"]][:1])
    assert [item["state"] for item in running["steps"]][0] == "pending"
    assert [item["state"] for item in stranger["steps"]][0] == "blocked"


def test_an_approval_survives_its_own_service_being_restarted(tmp_path, monkeypatch):
    """Which process happened to hold the port is a reading of the machine, not a decision.

    A restart changes the PID and nothing else, so a plan that carried the holder's name or
    number into what was approved would expire an owner's approval every time a unit was
    bounced — including by the setup run the approval was for.
    """
    home = tmp_path / "instance"
    home.mkdir()
    release = host(home, HINDSIGHT_URL="http://127.0.0.1:8888")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    settings = load_settings()
    environ = {"HERMES_MEMORY_RELEASE": str(release)}

    before = plan(settings, hermes_home=home, environ=environ,
                  proc=proc_table(tmp_path, "first", 8888, owned_by=home, pid="4242"))
    after = plan(settings, hermes_home=home, environ=environ,
                 proc=proc_table(tmp_path, "second", 8888, owned_by=home, pid="7777"))
    assert [item["advisory"] for item in before["steps"]][0] != \
           [item["advisory"] for item in after["steps"]][0], "the two machines do differ"
    assert before["review_digest"] == after["review_digest"]


def test_an_installation_with_no_owner_named_has_nobody_whose_approval_counts(tmp_path,
                                                                             monkeypatch):
    home = tmp_path / "unowned"
    home.mkdir()
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={home / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    with pytest.raises(SetupError, match="OWNER_PRINCIPAL"):
        plan(load_settings(), hermes_home=activity(tmp_path), environ={})


# -- the approval -------------------------------------------------------------

def test_an_approval_is_spent_on_the_plan_that_was_shown_and_on_no_other(installation,
                                                                         tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    proposal = plan(settings, hermes_home=home, environ=environ)
    with pytest.raises(SetupError, match="does not match"):
        run(settings, hermes_home=home, actor=OWNER, review="0" * 64, environ=environ)
    assert not (settings.home / "installation.db").exists()
    assert proposal["review_digest"]


def test_an_approval_cannot_be_spent_on_a_different_plan(installation, tmp_path):
    """The digest is of the work, not of the word "setup".

    A plan that changed after it was shown - another service turned up on the port, a
    unit appeared in the directory - is a different decision, and the run has to ask
    again rather than carry an approval forward.
    """
    settings, environ = installation
    home = activity(tmp_path)
    proposal = plan(settings, hermes_home=home, environ=environ, ref=REF,
                    runner=Heremes(home / "config.yaml"))
    directory = unit_directory()
    directory.mkdir(parents=True, mode=0o700)
    (directory / "hermes-memory.service").write_text("[Service]\nExecStart=/opt/theirs\n",
                                                     encoding="utf-8")
    with pytest.raises(SetupError, match="does not match"):
        run(settings, hermes_home=home, actor=OWNER, review=proposal["review_digest"],
            environ=environ, ref=REF, runner=Heremes(home / "config.yaml"))
    assert "/opt/theirs" in (directory / "hermes-memory.service").read_text(
        encoding="utf-8")


def test_an_agent_credential_cannot_approve_an_installation(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    proposal = plan(settings, hermes_home=home, environ=environ)
    with pytest.raises(SetupError, match="owner principal"):
        run(settings, hermes_home=home, actor="hermes-agent",
            review=proposal["review_digest"], environ=environ)
    assert not (settings.home / "installation.db").exists()


def test_a_moving_plan_is_a_different_approval(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    first = plan(settings, hermes_home=home, environ=environ)
    other = activity(tmp_path.parent / "other", provider="honcho")
    second = plan(settings, hermes_home=other, environ=environ)
    assert first["review_digest"] != second["review_digest"]


# -- running it ---------------------------------------------------------------

def test_a_complete_run_enrolls_creates_installs_and_leaves_the_host_model_alone(
        installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    runner = Heremes(home / "config.yaml")
    receipt = approve(settings, home, environ, runner=runner, ref=REF)
    assert receipt["done"] == list(STEPS)
    assert receipt["resumed"] == []

    with ProfileRegistry.open(settings) as ledger:
        profile = ledger.profile("work")
    assert profile.bank_id == "hermes-work" and profile.credential_scope == "profile-work"
    assert profile.db_path.is_file(), "the profile's own store was migrated"
    assert not (settings.home / "data" / "canonical.db").exists()
    # the instance store is not where a profile's evidence goes

    units = unit_directory()
    assert (units / "hermes-memory.service").is_file()
    assert not (units / "hermes-memory-worker.service").exists()
    # no backend is configured, so no backend unit is installed

    assert runner.verbs == ["plugins install", "plugins enable", "config set",
                            "--user daemon-reload"]
    install = runner.calls[0]
    assert f"--ref={REF}"[-len(REF) - 5:] and REF in install
    assert "--no-enable" in install
    assert install[install.index("install") + 1].startswith("file:///")
    assert "--no-allow-tool-override" in runner.calls[1]

    text = (home / "config.yaml").read_text(encoding="utf-8")
    assert "model: deepseek-chat" in text, "the main model is not this transaction's to set"
    assert "provider: hermes-memory" in text
    assert list((settings.home).glob("config.yaml.before-*")), "the previous file was kept"


def test_the_canary_proves_erasure_and_leaves_nothing_behind(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    approve(settings, home, environ, runner=Heremes(home / "config.yaml"), ref=REF)
    with ProfileRegistry.open(settings) as ledger:
        store_path = ledger.profile("work").db_path
    with EvidenceStore(store_path) as store:
        rows = store.db.execute("SELECT source, deleted FROM records").fetchall()
        assert [row["source"] for row in rows] == [transaction.CANARY_SOURCE]
        assert rows[0]["deleted"] == 1, "the canary was not left in the archive"
        assert store.db.execute("SELECT count(*) FROM erasure_ledger").fetchone()[0] == 1


def test_a_second_run_resumes_every_step_and_asks_the_host_for_nothing(installation,
                                                                       tmp_path):
    """The whole point of receipts: an interrupted setup finishes, it does not repeat.

    Re-installing a plugin, re-rotating a credential or re-writing a bank on a rerun is
    how an installation ends up with two owners of one conversation.
    """
    settings, environ = installation
    home = activity(tmp_path)
    first = Heremes(home / "config.yaml")
    receipt = approve(settings, home, environ, runner=first, ref=REF)
    assert len(first.calls) == 4, "install, enable, config set, daemon-reload"

    proposal = plan(settings, hermes_home=home, environ=environ, runner=first, ref=REF)
    states = {item["step"]: item["state"] for item in proposal["steps"]}
    # The steps that merely read something re-read it, because setup changed what is
    # there. The steps that reach outside - the host commands, the store, the units -
    # are all settled, and that is the property an interrupted run depends on.
    assert sorted(proposal["todo"]) == ["inventory", "plan", "services"]
    for written in ("configure", "initialize", "register-plugin", "preflight",
                    "activate", "canary", "services"):
        assert states[written] in ("resumed", "pending"), written
    assert states["register-plugin"] == "resumed"
    assert states["configure"] == "resumed"
    again = run(settings, hermes_home=home, actor=OWNER,
                review=proposal["review_digest"], environ=environ, runner=first, ref=REF)
    assert sorted(again["done"]) == ["inventory", "plan", "services"]
    assert len(again["resumed"]) == len(STEPS) - 3
    assert len(first.calls) == 4, "the host was asked to do it again"
    settled = run(settings, hermes_home=home, actor=OWNER,
                  review=plan(settings, hermes_home=home, environ=environ, runner=first,
                              ref=REF)["review_digest"],
                  environ=environ, runner=first, ref=REF)
    assert settled["done"] == [], "the third run has nothing left to do"
    assert len(first.calls) == 4
    with ProfileRegistry.open(settings) as ledger:
        assert ledger.db.execute("SELECT count(*) FROM enrollment_receipts"
                                 ).fetchone()[0] == 1
        assert ledger.db.execute("SELECT count(*) FROM setup_steps").fetchone()[0] == \
            len(STEPS), "a step is recorded once per profile, not once per run"


def test_a_changed_input_replans_only_the_step_that_depends_on_it(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    approve(settings, home, environ, runner=Heremes(home / "config.yaml"), ref=REF)
    (settings.home / "hermes-memory.env").write_text(
        (settings.home / "hermes-memory.env").read_text(encoding="utf-8")
        + "HERMES_MEMORY_ADMISSION_URL=http://127.0.0.1:8124\n", encoding="utf-8")
    proposal = plan(settings, hermes_home=home, environ=environ,
                    runner=Heremes(home / "config.yaml"), ref=REF)
    states = {item["step"]: item["state"] for item in proposal["steps"]}
    assert states["inventory"] == "pending", "a new endpoint is a fact about the machine"
    assert states["canary"] == "resumed", "the step that does not read it is not re-run"
    assert states["initialize"] == "resumed"
    assert states["finish"] == "resumed"


def test_activation_refuses_a_host_writer_that_touched_the_model_block(installation,
                                                                      tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    runner = Heremes(home / "config.yaml", break_model=True)
    with pytest.raises(SetupError, match="model block"):
        approve(settings, home, environ, runner=runner, ref=REF)
    assert "model: gpt-9" in (home / "config.yaml").read_text(encoding="utf-8")
    kept = list(settings.home.glob("config.yaml.before-*"))
    assert "model: deepseek-chat" in kept[0].read_text(encoding="utf-8")
    # the owner's file is recoverable from the snapshot setup took


def test_activation_refuses_a_writer_that_changed_nothing(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    with pytest.raises(SetupError, match="still reads"):
        approve(settings, home, environ,
                runner=Heremes(home / "config.yaml", forget_provider=True), ref=REF)


def test_a_failing_host_command_stops_the_transaction(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    with pytest.raises(SetupError, match="host said no"):
        approve(settings, home, environ, runner=Heremes(home / "config.yaml", fail=True),
                ref=REF)
    proposal = plan(settings, hermes_home=home, environ=environ,
                    runner=Heremes(home / "config.yaml"), ref=REF)
    states = {item["step"]: item["state"] for item in proposal["steps"]}
    assert states["configure"] == "resumed" and states["register-plugin"] == "pending"
    # the rerun starts from the step that failed, not from the beginning


def test_a_moving_pointer_is_not_a_release(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    for ref in (None, "main", "deadbeef", REF + "0"):
        proposal = plan(settings, hermes_home=home, environ=environ, ref=ref,
                        runner=Heremes(home / "config.yaml"))
        assert any("--ref" in line for line in proposal["blocked"])


def test_no_command_runs_without_an_executor_and_the_plan_says_which(installation,
                                                                     tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    proposal = plan(settings, hermes_home=home, environ=environ, ref=REF)
    refused = [item["step"] for item in proposal["steps"]
               if any("executor" in line for line in item["blocking"])]
    # a step the transaction cannot finish must not be offered as one it can
    assert refused == ["register-plugin", "activate", "services"]
    assert proposal["host_commands"], "the commands are shown whether or not they can run"


def test_an_approval_needs_a_name_behind_it(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    proposal = plan(settings, hermes_home=home, environ=environ, ref=REF)
    for blank in ("", "   "):
        with pytest.raises(SetupError, match="actor must be named"):
            run(settings, hermes_home=home, actor=blank,
                review=proposal["review_digest"], environ=environ, ref=REF,
                runner=Heremes(home / "config.yaml"))


def test_the_snapshot_of_the_owners_file_is_nobody_elses_to_read(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    approve(settings, home, environ, runner=Heremes(home / "config.yaml"), ref=REF)
    kept = list(settings.home.glob("config.yaml.before-*"))
    assert len(kept) == 1
    assert kept[0].stat().st_mode & 0o077 == 0


def test_a_unit_somebody_else_installed_is_not_touched_by_this_transaction(
        installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    directory = unit_directory()
    directory.mkdir(parents=True, mode=0o700)
    (directory / "hermes-memory.service").write_text("[Service]\nExecStart=/opt/theirs\n",
                                                     encoding="utf-8")
    proposal = plan(settings, hermes_home=home, environ=environ, ref=REF,
                    runner=Heremes(home / "config.yaml"))
    entry = {item["step"]: item for item in proposal["steps"]}["services"]
    assert "not written by this installation" in " ".join(entry["blocking"])
    assert "theirs" in (directory / "hermes-memory.service").read_text(encoding="utf-8")


def test_setup_over_an_installation_that_already_has_memory_leaves_it_alone(
        installation, tmp_path):
    """An idempotent installer is one that never resets a bank it did not create."""
    settings, environ = installation
    home = activity(tmp_path)
    first = Heremes(home / "config.yaml")
    approve(settings, home, environ, runner=first, ref=REF)
    with ProfileRegistry.open(settings) as ledger:
        store_path = ledger.profile("work").db_path
    with EvidenceStore(store_path) as store:
        store.commit({"source": "gmail", "source_id": "keep-me", "revision": "1",
                      "kind": "email", "text": "A real message from before the rerun.",
                      "observed_at": "2026-09-01T00:00:00+00:00",
                      "occurred_at": "2026-09-01T00:00:00+00:00",
                      "occurred_precision": "second", "metadata": {}})
    approve(settings, home, environ, runner=first, ref=REF)
    with EvidenceStore(store_path) as store:
        rows = store.db.execute("SELECT source FROM records ORDER BY source").fetchall()
    assert [row["source"] for row in rows] == ["gmail", transaction.CANARY_SOURCE]


def test_a_restarted_installation_does_not_migrate_over_the_records_it_lost_the_receipt_of(
        installation, tmp_path):
    """The one window where `initialize` is applied to a store that already has memories.

    A run that died after writing the archive and before recording its receipt comes back to
    an existing file, and this step is what decides whether that means "leave it" or "start
    over". Starting over is how an installation loses a person's memory on the second try, so
    the claim has to be in the receipt this run writes, not only in the plan it was approved
    from.
    """
    settings, environ = installation
    home = activity(tmp_path)
    runner = Heremes(home / "config.yaml")
    approve(settings, home, environ, runner=runner, ref=REF)
    with ProfileRegistry.open(settings) as ledger:
        store_path = ledger.profile("work").db_path
        with EvidenceStore(store_path) as store:
            kept = store.commit({"source": "gmail", "source_id": "keep-me", "revision": "1",
                                 "kind": "email",
                                 "text": "A real message from before the crash.",
                                 "observed_at": "2026-09-01T00:00:00+00:00",
                                 "occurred_at": "2026-09-01T00:00:00+00:00",
                                 "occurred_precision": "second", "metadata": {}})["id"]
        # The crash: the store is on disk, its receipt is not.
        ledger.db.execute("DELETE FROM setup_steps WHERE step='initialize'")
        ledger.db.commit()

    proposal = plan(settings, hermes_home=home, environ=environ, runner=runner, ref=REF)
    entry = {item["step"]: item for item in proposal["steps"]}["initialize"]
    assert entry["state"] == "pending"
    assert any("left exactly as it is" in action for action in entry["actions"]), entry
    run(settings, hermes_home=home, actor=OWNER, review=proposal["review_digest"],
        environ=environ, runner=runner, ref=REF)
    with ProfileRegistry.open(settings) as ledger:
        recorded = json.loads(ledger.db.execute(
            "SELECT actions FROM setup_steps WHERE step='initialize'").fetchone()[0])
        with EvidenceStore(store_path) as store:
            assert store.get(kept) is not None, "the rerun reset the archive"
    assert any("left exactly as it is" in action for action in recorded), recorded


def test_a_canary_that_cannot_be_found_is_a_failed_installation(installation, tmp_path,
                                                                monkeypatch):
    """The canary is the proof, so a proof that cannot be read has to stop the run."""
    settings, environ = installation
    home = activity(tmp_path)
    monkeypatch.setattr(EvidenceStore, "search",
                        lambda self, match, *, limit=20: [])
    with pytest.raises(SetupError, match="could not be retrieved"):
        approve(settings, home, environ, runner=Heremes(home / "config.yaml"), ref=REF)


def test_a_unit_is_written_only_for_an_executable_that_was_staged(tmp_path, monkeypatch):
    """Every program the owned units ExecStart is checked, read off the units themselves.

    The worker no longer starts the distribution's own script — our launcher has to, so the
    wrapper can record what is about to run. That makes the backend environment's
    interpreter an executable this installation depends on, and a list hard-coded in the
    staging step is a check a template edit can walk away from.
    """
    home = tmp_path / "instance"
    home.mkdir()
    release = host(home, HINDSIGHT_URL=BACKEND_URL,
                   ALLOWED_INFERENCE_HOSTS="127.0.0.1")
    (release / "hindsight" / "bin").mkdir(parents=True)
    (release / "hindsight" / "bin" / "hindsight-api").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.delenv("HERMES_MEMORY_RELEASE", raising=False)
    settings = load_settings()
    arguments = {"hermes_home": activity(tmp_path),
                 "environ": {"HERMES_MEMORY_RELEASE": str(release)}}
    proposal = plan(settings, **arguments)
    assert any("hindsight/bin/python is not staged" in line
               for line in proposal["blocked"])
    assert not any("hindsight-api is not staged" in line for line in proposal["blocked"])
    (release / "hindsight" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    staged = plan(settings, **arguments)
    assert not any("is not staged" in line for line in staged["blocked"]), staged["blocked"]


def test_a_capture_only_installation_is_checked_only_for_its_own_unit(tmp_path, monkeypatch):
    """No backend route means no backend processes to start, so none to demand either."""
    home = tmp_path / "instance"
    home.mkdir()
    release = host(home)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.delenv("HERMES_MEMORY_RELEASE", raising=False)
    arguments = {"hermes_home": activity(tmp_path),
                 "environ": {"HERMES_MEMORY_RELEASE": str(release)}}
    assert not any("is not staged" in line
                   for line in plan(load_settings(), **arguments)["blocked"])
    (release / "bin" / "hermes-memory").unlink()
    gone = plan(load_settings(), **arguments)
    assert any("bin/hermes-memory is not staged" in line for line in gone["blocked"])


def test_a_release_whose_compatibility_manifest_lies_is_not_installed(installation, tmp_path):
    """Preflight asks the installation about itself, and one of its answers is a file.

    The release tree's own ``deployment/compatibility.json`` names the backend this code was
    composed against. Reading it is the only place in the transaction where a machine can
    notice that the release it is holding is not the release it believes, and a lie there has
    to stop the install rather than be logged and carried on.
    """
    from hermes_memory.install import compatibility

    settings, environ = installation
    home = activity(tmp_path)
    pointer = Path(settings.home) / "runtime" / "current"
    path = compatibility.write(path=pointer / "deployment" / "compatibility.json")
    claims = json.loads(path.read_text(encoding="utf-8"))
    claims["hindsight"]["engine_pinned"] = "0.9.0"
    path.write_text(json.dumps(claims), encoding="utf-8")

    executor = Heremes(home / "config.yaml")
    with pytest.raises(SetupError, match="compatibility.json"):
        approve(settings, home, environ, runner=executor, ref=REF)


def test_an_unstaged_release_stops_before_anything_is_written(installation, tmp_path,
                                                             monkeypatch):
    settings, environ = installation
    home = activity(tmp_path)
    proposal = plan(settings, hermes_home=home,
                    environ={"HERMES_MEMORY_RELEASE": str(tmp_path / "nowhere")},
                    ref=REF, runner=Heremes(home / "config.yaml"))
    assert any("not staged" in line for line in proposal["blocked"])
    missing = {"HERMES_MEMORY_RELEASE": str(tmp_path / "nowhere")}
    executor = Heremes(home / "config.yaml")
    with pytest.raises(SetupError, match="not staged"):
        run(settings, hermes_home=home, actor=OWNER, review=proposal["review_digest"],
            environ=missing, ref=REF, runner=executor)
    # A refused run may open its own journal, and the read-only steps that did finish
    # belong in it; what must not happen is that anybody is enrolled, a store is made,
    # or a unit is installed.
    with ProfileRegistry.open(settings) as ledger:
        assert ledger.names() == []
        recorded = {row[0] for row in ledger.db.execute("SELECT step FROM setup_steps")}
        assert recorded <= {"inventory", "plan"}, recorded
    assert not unit_directory().exists()


# -- the things it must not step on -------------------------------------------

def test_a_credential_written_into_the_configuration_stops_the_configure_step(
        installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_HINDSIGHT_URL={BACKEND_URL}\n"
        "HERMES_MEMORY_HINDSIGHT_API_KEY=sk-never-in-a-config-file\n", encoding="utf-8")
    proposal = plan(settings, hermes_home=home, environ=environ, ref=REF)
    entry = {item["step"]: item for item in proposal["steps"]}["configure"]
    assert entry["state"] == "blocked"
    assert "name the variable" in " ".join(entry["blocking"])


def test_a_second_capture_owner_for_one_home_is_refused_before_anything_is_written(
        installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    with ProfileRegistry.open(settings) as ledger:
        proposal = ledger.plan("elsewhere", home)
        ledger.enroll("elsewhere", home, actor=OWNER,
                      review_digest=proposal["review_digest"])
    after = plan(settings, hermes_home=home, environ=environ, ref=REF)
    assert any("hermes_home" in line or "already" in line for line in after["blocked"])


def test_a_port_somebody_else_holds_is_said_before_the_services_are_written(installation,
                                                                           tmp_path,
                                                                           monkeypatch):
    settings, environ = installation
    home = activity(tmp_path)
    taken = sorted(listening_ports(Path("/proc")))
    if not taken:
        pytest.skip("nothing is listening on this machine to collide with")
    (settings.home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={settings.data_dir}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n"
        f"HERMES_MEMORY_OWNER_PRINCIPAL={OWNER}\n"
        f"HERMES_MEMORY_HINDSIGHT_URL=http://127.0.0.1:{taken[0]}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(settings.home))
    proposal = plan(load_settings(), hermes_home=home, environ=environ, ref=REF)
    ports = [line for line in proposal["blocked"] if "listening on" in line]
    assert ports, f"port {taken[0]} is held by something on this machine"


def test_a_canary_that_cannot_be_forgotten_fails_the_installation(installation, tmp_path,
                                                                 monkeypatch):
    """An erasure that did not happen is the one finding a canary exists to catch."""
    settings, environ = installation
    home = activity(tmp_path)
    from hermes_memory.lifecycle import erasure

    monkeypatch.setattr(erasure.ErasureManager, "forgotten",
                        lambda self, record_pk: False)
    with pytest.raises(SetupError, match="fences do not hold"):
        approve(settings, home, environ, runner=Heremes(home / "config.yaml"), ref=REF)


def test_a_profile_configuration_the_owner_wrote_is_left_word_for_word(installation,
                                                                       tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    written = (f"HERMES_MEMORY_HINDSIGHT_URL={BACKEND_URL}\n"
               "HERMES_MEMORY_OWNER_PRINCIPAL=jugaadu\n"
               "# a line the owner added, and a reason not to rewrite this file\n")
    (home / "hermes-memory.env").write_text(written, encoding="utf-8")
    approve(settings, home, environ, runner=Heremes(home / "config.yaml"), ref=REF)
    assert (home / "hermes-memory.env").read_text(encoding="utf-8") == written


def test_the_host_profile_is_named_from_the_home_and_the_home_alone(tmp_path):
    assert profile_name(tmp_path / "homes" / "Work-2") == "work-2"
    assert profile_name(tmp_path / "home") == "default"
    assert profile_name(tmp_path / ".hermes") == "default"
    assert profile_name(tmp_path / "profiles" / "deep" / "nested") == "nested"


def test_readiness_never_writes_configured_up_as_operational(installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path)
    receipt = approve(settings, home, environ, runner=Heremes(home / "config.yaml"),
                      ref=REF)
    assert receipt["readiness"]["inference"] == "off"
    assert receipt["readiness"]["real_sources"] == "disabled until the owner enables one"
    assert receipt["readiness"]["delivery"] == "off"


# -- the receipt that lets an uninstall give the selection back ------------------

def test_activation_records_the_selection_it_replaced(installation, tmp_path):
    """§10.7 restores the prior provider, and that is a fact only for one moment.

    After the write the file answers a different question, so the answer has to be kept
    while it is still true.
    """
    settings, environ = installation
    home = activity(tmp_path, provider="pg0-memory")
    approve(settings, home, environ, runner=Heremes(home / "config.yaml"), ref=REF)
    written = json.loads((settings.home / "provider-selection.json").read_text(
        encoding="utf-8"))
    assert written["prior"] == "pg0-memory"
    assert written["written"] == "hermes-memory"
    assert written["config"] == str(home / "config.yaml")
    assert written["actor"] == settings.owner_principal


def test_an_already_selected_provider_is_not_blocked_by_a_missing_executor(installation,
                                                                          tmp_path):
    """A Hermes-first installation arrives with the selection already made.

    Refusing the whole transaction then would be the installer claiming it still has to
    do something it has done, and a plan whose blockers are wrong is a plan nobody can
    act on.
    """
    settings, environ = installation
    home = activity(tmp_path, provider="hermes-memory")
    proposal = plan(settings, hermes_home=home, environ=environ, ref=REF)
    entry = {item["step"]: item for item in proposal["steps"]}["activate"]
    assert entry["blocking"] == []
    assert entry["state"] == "pending"
    assert "already reads 'hermes-memory'" in " ".join(entry["actions"])


def test_an_already_selected_provider_says_so_instead_of_taking_another_snapshot(
        installation, tmp_path):
    settings, environ = installation
    home = activity(tmp_path, provider="pg0-memory")
    approve(settings, home, environ, runner=Heremes(home / "config.yaml"), ref=REF)
    before = sorted(path.name for path in settings.home.glob("config.yaml.before-*"))
    assert before, "activation snapshots the file it is about to change"

    second = Heremes(home / "config.yaml")
    approve(settings, home, environ, runner=second, ref=REF)
    assert [call for call in second.calls if call[1:3] == ["config", "set"]] == []
    assert sorted(path.name for path in settings.home.glob("config.yaml.before-*")) == before
    assert (settings.home / "provider-selection.json").read_text(
        encoding="utf-8").count("pg0-memory") == 1, "the prior is recorded once"


# -- whose interpreter is asked -----------------------------------------------

def a_worker_interpreter(release, *, imports, exit_code=0):
    """The engine's venv as a staged release carries it: an interpreter that answers a probe.

    The probe is a subprocess by design — the question is about the interpreter the worker
    unit starts, not about the process doing the installing — so the stand-in has to be a
    program that really runs.
    """
    python = release / "hindsight" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nprintf '%s' '" + json.dumps(imports) + f"'\nexit {exit_code}\n",
                      encoding="utf-8")
    python.chmod(0o755)
    return python


def staged_backend(installation, *, imports, exit_code=0, interpreter=True):
    """An installation with a backend route, and a worker environment shaped as asked for."""
    settings, environ = installation
    home = Path(settings.home)
    release = Path(environ["HERMES_MEMORY_RELEASE"])
    # Naming a backend route is what makes the other two units exist, so the executables
    # they start have to be there for the stage step to be about the probe rather than about
    # a missing binary.
    (release / "hindsight" / "bin").mkdir(parents=True, exist_ok=True)
    (release / "hindsight" / "bin" / "hindsight-api").write_text("#!/bin/sh\n", encoding="utf-8")
    env_file = home / "hermes-memory.env"
    env_file.write_text(env_file.read_text(encoding="utf-8")
                        + f"\nHERMES_MEMORY_HINDSIGHT_URL={BACKEND_URL}\n"
                          "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=127.0.0.1\n",
                        encoding="utf-8")
    if interpreter:
        a_worker_interpreter(release, imports=imports, exit_code=exit_code)
    proposal = plan(load_settings(), hermes_home=home, environ=environ)
    # Only the stage step's blockers: a pytest temporary path contains the word "worker",
    # and a filter that matches prose in someone else's sentence proves nothing.
    return [line for line in proposal["blocked"] if line.startswith("stage:")]


def test_a_gate_that_imports_nothing_from_the_engine_is_a_correct_installation(installation):
    """The bug this replaces blocked a staged release for having its environments separate.

    The framework's bridge is HTTP: no package from the backend is importable in the gate's
    own venv, and that is the design rather than a degraded form of it.
    """
    assert staged_backend(installation, imports={"hindsight_api": True,
                                                 "hermes_memory": True}) == []


def test_a_release_with_no_worker_environment_says_which_interpreter_is_missing(installation):
    blockers = staged_backend(installation, imports={}, interpreter=False)
    assert any("hindsight/bin/python" in line for line in blockers), blockers


def test_an_engine_absent_from_the_worker_environment_is_named_not_guessed(installation):
    blockers = staged_backend(installation, imports={"hindsight_api": False,
                                                     "hermes_memory": True})
    assert any("hindsight_api" in line for line in blockers), blockers


def test_a_worker_interpreter_that_cannot_run_is_reported_as_one_that_cannot_run(installation):
    blockers = staged_backend(installation, imports={"hindsight_api": True,
                                                     "hermes_memory": True}, exit_code=3)
    assert any("not importable" in line or "could not run" in line
               for line in blockers), blockers


# -- what the host is actually pointed at -------------------------------------

def test_the_host_is_given_a_git_tree_because_that_is_all_it_accepts(installation):
    """A staged release has no ``.git``, and the host clones rather than copies.

    Verified against the installed host: `hermes plugins install` names a catalog entry, a Git
    URL or an owner/repo, and `--ref` is a commit inside that repository. The runtime still
    comes from the release, which is why the two are compared below rather than merged.
    """
    settings, environ = installation
    context = transaction._context(settings, hermes_home=Path(settings.home),
                                  environ=environ)
    install = next(argv for argv, name in transaction.host_commands(context)
                   if name == "install")
    url = str(install[3])
    checkout = Path(__file__).resolve().parents[2]
    assert url == f"file://{checkout}#integrations/hermes-memory"
    assert str(environ["HERMES_MEMORY_RELEASE"]) not in url


def test_a_release_whose_plugin_disagrees_with_the_checkout_is_not_registered(installation,
                                                                             monkeypatch):
    """§10.3's pairing, enforced where it can still stop something."""
    settings, environ = installation
    carried = Path(environ["HERMES_MEMORY_RELEASE"]) / "integrations" / "hermes-memory"
    (carried / "provider.py").write_text("# a different contract than the checkout runs\n",
                                         encoding="utf-8")
    proposal = plan(settings, hermes_home=Path(settings.home), environ=environ,
                    ref="0" * 40)
    assert any("does not match the one this release carries" in line
               for line in proposal["blocked"]), proposal["blocked"]


# -- a host action that already happened --------------------------------------

def registered_host(tmp_path, revision):
    """A Hermes home whose own record says the plugin is already installed at `revision`."""
    home = tmp_path / "homes" / "registered"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / ".install-metadata.json").write_text(json.dumps({
        "hermes-memory": {"pinned": True, "revision": revision,
                          "source": "file:///somewhere#integrations/hermes-memory"}}),
        encoding="utf-8")
    (home / "config.yaml").write_text("model:\n  provider: openai\n  model: m\n",
                                      encoding="utf-8")
    return home


def test_a_registration_that_already_happened_is_not_happening_again(installation):
    """`hermes plugins install` refuses a plugin that is there, so a re-run used to fail."""
    settings, environ = installation
    revision = "b" * 40
    activity = registered_host(Path(settings.home).parent, revision)
    runner = Heremes(activity / "config.yaml")

    proposal = plan(settings, hermes_home=activity, environ=environ, ref=revision,
                    runner=runner)
    # Other steps still have their ordering to work through; what must be silent here is the
    # one that would re-run a host action the host has already recorded.
    assert not any(line.startswith("register-plugin:") for line in proposal["blocked"]), \
        proposal["blocked"]
    run(settings, hermes_home=activity, actor=OWNER, review=proposal["review_digest"],
        environ=environ, ref=revision, runner=runner)
    assert not any(call[1:3] == ["plugins", "install"] for call in runner.calls), runner.calls
    assert any(call[1:3] == ["config", "set"] for call in runner.calls), \
        "activating the provider is still this transaction's to do"


def test_a_host_holding_a_different_commit_is_told_so_rather_than_overwritten(installation):
    settings, environ = installation
    activity = registered_host(Path(settings.home).parent, "c" * 40)
    proposal = plan(settings, hermes_home=activity, environ=environ, ref="d" * 40,
                    runner=lambda argv: (0, ""))
    assert any("registers hermes-memory at " in line for line in proposal["blocked"]), \
        proposal["blocked"]


def test_a_backend_directory_that_went_missing_reopens_the_step_that_owns_it(installation,
                                                                             tmp_path):
    """The receipt says the layout was made; the filesystem no longer agrees.

    Resumption is by input digest, so inputs that leave out what the step created let a
    finished transaction hand the manager a unit that cannot spawn — the very failure those
    directories exist to prevent. It arrives as exit 226 at exec time, in a service log, on
    the machine that is supposed to be working.
    """
    from hermes_memory.install.services import layout

    assert staged_backend(installation, imports={"hindsight_api": True,
                                                 "hermes_memory": True}) == []
    settings, environ = load_settings(), installation[1]
    home = activity(tmp_path)
    runner = Heremes(home / "config.yaml")
    approve(settings, home, environ, runner=runner, ref=REF)
    placed = layout(settings, environ=environ)
    assert placed.hindsight_dir.is_dir(), "the step never made the directory it claims"
    placed.hindsight_dir.rmdir()

    proposal = plan(settings, hermes_home=home, environ=environ, runner=runner, ref=REF)
    entry = {item["step"]: item for item in proposal["steps"]}["initialize"]
    assert entry["state"] == "pending", "the layout is gone and the receipt says it is not"
    assert any(str(placed.hindsight_dir) in action for action in entry["actions"]), entry
    run(settings, hermes_home=home, actor=OWNER, review=proposal["review_digest"],
        environ=environ, runner=runner, ref=REF)
    assert placed.hindsight_dir.is_dir()
    settled = {item["step"]: item for item in
               plan(settings, hermes_home=home, environ=environ, runner=runner,
                    ref=REF)["steps"]}["initialize"]
    assert settled["state"] == "resumed", ("the receipt records the absence this pass "
                                           "repaired, so the step is asked again forever")


# -- the engine's own arithmetic ----------------------------------------------

def backend_env(installation):
    """The same installation with a backend route, and the file that unit starts with."""
    from hermes_memory.install.services import layout

    assert staged_backend(installation, imports={"hindsight_api": True,
                                                 "hermes_memory": True}) == []
    settings, environ = load_settings(), installation[1]
    return settings, environ, layout(settings, environ=environ).hindsight_env


def stage_blockers(settings, environ) -> list[str]:
    """The blockers the stage step sees: it is where a release is proved runnable."""
    proposal = plan(settings, hermes_home=Path(settings.home), environ=environ)
    return [line for line in proposal["blocked"] if line.startswith("stage:")]


#: An OpenAI-compatible provider with no base URL is served by the SDK's own host, so every
#: engine environment below has to say where its model is, exactly as the shipped template does.
TEXT_ROUTE = "HINDSIGHT_API_LLM_BASE_URL=http://127.0.0.1:8080/v1\n"
EMBEDDINGS_ROUTE = "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL=http://127.0.0.1:11434/v1\n"


def test_a_backend_environment_the_engine_would_refuse_is_refused_before_it_starts(
        installation):
    """`RETAIN_MAX_COMPLETION_TOKENS` has to exceed `RETAIN_CHUNK_SIZE`, or the engine quits.

    Both units of a pinned backend were crash-looping on this arithmetic on a real machine:
    the cap is ours to set and the chunk default is the engine's, and nothing between the two
    noticed until `systemd` had restarted the service a hundred times.
    """
    settings, environ, env = backend_env(installation)
    env.write_text("HINDSIGHT_API_LLM_PROVIDER=openai\n" + TEXT_ROUTE
                   + "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=2048\n", encoding="utf-8")
    blockers = stage_blockers(settings, environ)
    assert any("RETAIN_CHUNK_SIZE" in line and "RETAIN_MAX_COMPLETION_TOKENS" in line
               for line in blockers), blockers
    env.write_text(env.read_text(encoding="utf-8")
                   + "HINDSIGHT_API_RETAIN_CHUNK_SIZE=1500\n", encoding="utf-8")
    assert stage_blockers(settings, environ) == []
    # The rule is "greater than", so the two meeting exactly is still a refusal.
    env.write_text("HINDSIGHT_API_LLM_PROVIDER=openai\n" + TEXT_ROUTE
                   + "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=1500\n"
                   "HINDSIGHT_API_RETAIN_CHUNK_SIZE=1500\n", encoding="utf-8")
    assert stage_blockers(settings, environ), "an equal budget and chunk cannot answer"


def test_an_engine_that_generates_nothing_is_not_checked_for_a_generation_budget(
        installation):
    """A provider switched off has no output to size.

    Refusing it anyway would be the framework disagreeing with its own backend about what a
    valid installation is, and the engine's validator makes exactly this exception.
    """
    settings, environ, env = backend_env(installation)
    env.write_text("HINDSIGHT_API_LLM_PROVIDER=none\n"
                   "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=1\n", encoding="utf-8")
    assert stage_blockers(settings, environ) == []


def test_a_backend_environment_that_names_no_number_is_named_rather_than_guessed(
        installation):
    settings, environ, env = backend_env(installation)
    env.write_text("HINDSIGHT_API_LLM_PROVIDER=openai\n" + TEXT_ROUTE
                   + "HINDSIGHT_API_RETAIN_CHUNK_SIZE=smallish\n", encoding="utf-8")
    assert any("not a number" in line for line in stage_blockers(settings, environ))


def test_an_embedding_route_the_engine_must_measure_is_refused_before_it_starts(
        installation):
    """A declared width is a number in a file; an undeclared one is a model request.

    The engine discovers an unknown OpenAI-compatible model's vector width by sending a test
    embedding at startup. That is a model call inside a process start — which §10.4 rules out
    for a liveness path, and which costs more than the tokens: the request goes through the
    admission gate, so an installation whose inference the owner is holding cannot start its
    own backend, and the unit's restart budget then leaves it failed.
    """
    settings, environ, env = backend_env(installation)
    env.write_text("HINDSIGHT_API_LLM_PROVIDER=openai\n" + TEXT_ROUTE
                   + "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=2048\n"
                   "HINDSIGHT_API_RETAIN_CHUNK_SIZE=1500\n"
                   "HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai\n" + EMBEDDINGS_ROUTE
                   + "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=qwen3-embedding:0.6b\n",
                   encoding="utf-8")
    blockers = stage_blockers(settings, environ)
    assert any("test embedding" in line for line in blockers), blockers
    env.write_text(env.read_text(encoding="utf-8")
                   + "HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS=1024\n", encoding="utf-8")
    assert stage_blockers(settings, environ) == []


def test_an_embedding_route_that_dials_nothing_needs_no_declared_width(installation):
    """`local` loads a model in this process; it does not ask a URL what shape it is."""
    settings, environ, env = backend_env(installation)
    env.write_text("HINDSIGHT_API_LLM_PROVIDER=openai\n" + TEXT_ROUTE
                   + "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=2048\n"
                   "HINDSIGHT_API_RETAIN_CHUNK_SIZE=1500\n"
                   "HINDSIGHT_API_EMBEDDINGS_PROVIDER=local\n", encoding="utf-8")
    assert stage_blockers(settings, environ) == []


def test_an_engine_environment_that_names_no_text_provider_is_refused(installation):
    """An unset provider is not "nothing configured": the engine's own default is a cloud one.

    `DEFAULT_LLM_PROVIDER = "openai"` and `DEFAULT_LLM_MODEL = "gpt-4o-mini"` in the pinned
    0.10.1 config, so a file that leaves the line out has the memory server reading the
    owner's evidence into somebody else's API while looking, from here, like a local install.
    """
    settings, environ, env = backend_env(installation)
    env.write_text("HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=2048\n"
                   "HINDSIGHT_API_RETAIN_CHUNK_SIZE=1500\n", encoding="utf-8")
    blockers = stage_blockers(settings, environ)
    assert any("HINDSIGHT_API_LLM_PROVIDER" in line for line in blockers), blockers


def test_a_provider_named_without_the_endpoint_it_would_use_is_the_hosted_one(installation):
    """The route is what makes `openai` mean a server in this room rather than a vendor's."""
    settings, environ, env = backend_env(installation)
    env.write_text("HINDSIGHT_API_LLM_PROVIDER=openai\n"
                   "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=2048\n"
                   "HINDSIGHT_API_RETAIN_CHUNK_SIZE=1500\n", encoding="utf-8")
    blockers = stage_blockers(settings, environ)
    assert any("HINDSIGHT_API_LLM_BASE_URL" in line and "api.openai.com" in line
               for line in blockers), blockers
    env.write_text(env.read_text(encoding="utf-8") + TEXT_ROUTE, encoding="utf-8")
    assert stage_blockers(settings, environ) == []


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "http://example.com:8080/v1",
                                 "http://8.8.8.8:8080/v1", "http://localhost:8080/v1",
                                 "http://169.254.169.254:8080/v1"])
def test_an_engine_endpoint_off_this_machine_is_refused(installation, url):
    """The framework's own allowlist does not bind a backend process, so its URLs are checked."""
    settings, environ, env = backend_env(installation)
    env.write_text("HINDSIGHT_API_LLM_PROVIDER=openai\n"
                   f"HINDSIGHT_API_LLM_BASE_URL={url}\n"
                   "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=2048\n"
                   "HINDSIGHT_API_RETAIN_CHUNK_SIZE=1500\n", encoding="utf-8")
    blockers = stage_blockers(settings, environ)
    assert any("HINDSIGHT_API_LLM_BASE_URL" in line for line in blockers), blockers


def test_an_engine_endpoint_written_to_a_private_address_is_not_argued_with(installation):
    """A literal RFC1918 endpoint is the approved shape, and no rule here disputes it."""
    settings, environ, env = backend_env(installation)
    env.write_text("HINDSIGHT_API_LLM_PROVIDER=openai\n"
                   "HINDSIGHT_API_LLM_BASE_URL=http://192.168.68.65:8080/v1\n"
                   "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=2048\n"
                   "HINDSIGHT_API_RETAIN_CHUNK_SIZE=1500\n", encoding="utf-8")
    assert stage_blockers(settings, environ) == []


def test_a_local_provider_needs_no_endpoint_and_a_disabled_engine_needs_none(installation):
    """`llamacpp` runs a model from a path and `mock` answers in code: neither dials out."""
    settings, environ, env = backend_env(installation)
    for provider in ("llamacpp", "mock"):
        env.write_text(f"HINDSIGHT_API_LLM_PROVIDER={provider}\n"
                       "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=2048\n"
                       "HINDSIGHT_API_RETAIN_CHUNK_SIZE=1500\n", encoding="utf-8")
        assert stage_blockers(settings, environ) == [], provider


def test_an_embedded_database_is_not_read_as_a_model_endpoint(installation):
    """`pg0` is where the memory lives, not somewhere a request goes.

    A rule about endpoints that swept every key ending in `_URL` would refuse the one key of
    this installation's that deliberately names no host at all.
    """
    settings, environ, env = backend_env(installation)
    env.write_text("HINDSIGHT_API_LLM_PROVIDER=openai\n" + TEXT_ROUTE
                   + "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=2048\n"
                   "HINDSIGHT_API_RETAIN_CHUNK_SIZE=1500\n"
                   "HINDSIGHT_API_DATABASE_URL=pg0\n", encoding="utf-8")
    assert stage_blockers(settings, environ) == []


def test_the_shipped_backend_environment_satisfies_the_checks_it_will_be_read_by():
    """The template is what an operator copies; a template the engine refuses is a trap."""
    from hermes_memory.config import env_file_values
    from hermes_memory.install.setup import (ENGINE_EMBEDDINGS_DIMENSIONS,
                                             ENGINE_EMBEDDINGS_PROVIDER, ENGINE_RETAIN_CAP,
                                             ENGINE_RETAIN_CHUNK)

    template = (Path(__file__).resolve().parents[2] / "deployment" / "env"
                / "hindsight.env.example")
    values = env_file_values(template)
    assert int(values[ENGINE_RETAIN_CAP]) > int(values[ENGINE_RETAIN_CHUNK]), template
    if values.get(ENGINE_EMBEDDINGS_PROVIDER) == "openai":
        assert int(values[ENGINE_EMBEDDINGS_DIMENSIONS]) > 0, (
            f"{template}: an OpenAI-compatible embedding route with no declared width makes "
            "the engine send a test embedding every time it starts")
    from hermes_memory.config import endpoint_is_private
    from hermes_memory.install.setup import ENGINE_LOCAL_PROVIDERS, ENGINE_ROUTES

    assert values, template
    for _work, provider_key, url_key in ENGINE_ROUTES:
        provider = values.get(provider_key, "").strip().lower()
        if provider and provider not in ENGINE_LOCAL_PROVIDERS:
            assert url_key in values, f"{template}: {provider_key}={provider} needs {url_key}"
    for key, value in values.items():
        if key.endswith("_BASE_URL") and value.strip():
            assert endpoint_is_private(value.strip()), f"{template}: {key}={value}"
