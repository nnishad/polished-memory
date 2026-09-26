"""C14 instance ledger: one machine, many profiles, no shared memory and no fallback.

The guards that matter here are the ones that refuse: an unenrolled home served from the
default profile, a second owner for one conversation, an enrollment approved by somebody
who is not the owner, and an approval that does not match what would change now.
"""
from __future__ import annotations

import sqlite3

import pytest

from hermes_memory.config import load_settings
from hermes_memory.ids import digest
from hermes_memory.install.profiles import (DEFAULT_PROFILE, INSTALLATION_MIGRATIONS,
                                            PROPOSAL_KEYS, InstallationError,
                                            Profile, ProfileRegistry,
                                            apply_installation_migrations, open_installation)

OWNER = "jugaadu"


@pytest.fixture()
def home(tmp_path):
    # Resolved because the ledger stores homes after resolving them; a test comparing
    # an unresolved ``tmp_path`` child would fail on any machine with a symlinked temp.
    root = tmp_path / "hm"
    root.mkdir()
    return root.resolve()


@pytest.fixture()
def registry(home):
    db = open_installation(home / "installation.db")
    yield ProfileRegistry(db, root=home, owner_principal=OWNER, default_home=home / "data")
    db.close()


def bare(**overrides):
    """A registry with no enrollment machinery exercised, for validation-only tests."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    apply_installation_migrations(db)
    arguments = {"root": "/r", "owner_principal": OWNER}
    arguments.update(overrides)
    return ProfileRegistry(db, **arguments)


def enroll(registry, name, hermes_home, *, actor=OWNER):
    proposal = registry.plan(name, hermes_home)
    return registry.enroll(name, hermes_home, actor=actor,
                           review_digest=proposal["review_digest"])


def activities(home, *names):
    return {name: home / "homes" / name for name in names}


# -- the ledger itself --------------------------------------------------------

def test_opening_creates_an_owner_only_ledger_and_is_idempotent(home):
    path = home / "nested" / "installation.db"
    db = open_installation(path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert (home / "nested").stat().st_mode & 0o777 == 0o700
    assert apply_installation_migrations(db) == len(INSTALLATION_MIGRATIONS)
    assert apply_installation_migrations(db) == len(INSTALLATION_MIGRATIONS)
    assert db.execute("SELECT count(*) FROM profiles").fetchone()[0] == 0


def test_a_ledger_written_by_a_stranger_is_resealed_on_open(home):
    path = home / "installation.db"
    open_installation(path).close()
    path.chmod(0o644)
    open_installation(path).close()
    assert path.stat().st_mode & 0o777 == 0o600, \
        "the ledger names where every profile's private evidence lives"


def test_a_ledger_from_a_newer_build_is_not_read_by_this_one(home):
    db = open_installation(home / "installation.db")
    db.execute("INSERT INTO schema_migrations(name, applied_at) VALUES(?, ?)",
               ("0199_time_travel", "2026-01-01T00:00:00+00:00"))
    with pytest.raises(InstallationError, match="upgrade the framework"):
        apply_installation_migrations(db)
    assert db.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 3, \
        "the refusal leaves the ledger exactly as it was"


def test_two_profiles_can_never_share_a_bank(home):
    """The schema, not merely the helper, refuses a second owner of one bank."""
    db = open_installation(home / "installation.db")
    columns = ("profile", "hermes_home", "data_dir", "bank_id", "credential_scope",
               "state", "review_digest", "enrolled_by", "enrolled_at")
    statement = ("INSERT INTO profiles(%s) VALUES(%s)"
                 % (",".join(columns), ",".join("?" * len(columns))))
    db.execute(statement, ("work", "/h/work", "/d/work", "hermes-shared", "profile-work",
                           "enrolled", "d1", OWNER, "2026-01-01T00:00:00+00:00"))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(statement, ("personal", "/h/personal", "/d/personal", "hermes-shared",
                               "profile-personal", "enrolled", "d2", OWNER,
                               "2026-01-01T00:00:00+00:00"))


def test_a_home_cannot_be_claimed_twice_even_by_a_raw_write(home):
    """UNIQUE(hermes_home) is the last line; the helper's refusal is only the first."""
    db = open_installation(home / "installation.db")
    registry = ProfileRegistry(db, root=home, owner_principal=OWNER,
                               default_home=home / "data")
    enroll(registry, "work", home / "homes" / "work")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO profiles(profile, hermes_home, data_dir, bank_id, "
                   "credential_scope, state, review_digest, enrolled_by, enrolled_at) "
                   "VALUES('personal',?, '/d/p', 'hermes-p', 'profile-p', 'enrolled', "
                   "'d', ?, '2026-01-01T00:00:00+00:00')",
                   (str(home / "homes" / "work"), OWNER))


def test_open_derives_the_ledger_location_from_settings(home, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    monkeypatch.setenv("HERMES_MEMORY_OWNER_PRINCIPAL", OWNER)
    monkeypatch.delenv("HERMES_MEMORY_DATA_DIR", raising=False)
    settings = load_settings()
    instance = ProfileRegistry.open(settings)
    assert enroll(instance, DEFAULT_PROFILE, home / "homes" / "default")["profile"] == \
           DEFAULT_PROFILE
    assert instance.profile(DEFAULT_PROFILE).data_dir == settings.data_dir
    assert (home / "installation.db").exists()


# -- resolution ---------------------------------------------------------------

def test_an_unenrolled_home_is_refused_rather_than_served_the_default(registry, home):
    enroll(registry, "work", home / "homes" / "work")
    with pytest.raises(InstallationError, match="not a fallback") as error:
        registry.resolve(home / "homes" / "someone-else")
    assert str(home / "homes" / "someone-else") in str(error.value)
    assert registry.names() == ["work"], "a refused lookup changed nothing"


@pytest.mark.parametrize("junk", ["", "   ", None, 7, ["a"], {}])
def test_a_home_that_is_not_a_path_is_refused_rather_than_guessed(junk):
    with pytest.raises(InstallationError, match="must be a path"):
        bare().resolve(junk)


def test_resolution_follows_each_activity_home_not_the_first_one(registry, home):
    homes = activities(home, "work", "personal")
    enroll(registry, "work", homes["work"])
    enroll(registry, "personal", homes["personal"])
    assert registry.resolve(homes["work"]).profile == "work"
    assert registry.resolve(homes["personal"]).profile == "personal", \
        "the first profile to connect does not own the others"


def test_resolution_is_by_home_so_an_alias_of_the_same_directory_agrees(registry, home):
    enroll(registry, "work", home / "homes" / "work")
    assert registry.resolve(str(home / "homes" / ".." / "homes" / "work")).profile == "work"


def test_moving_a_profile_releases_the_home_it_left(registry, home):
    homes = activities(home, "work", "work-2")
    enroll(registry, "work", homes["work"])
    proposal = registry.plan("work", homes["work-2"])
    registry.enroll("work", homes["work-2"], actor=OWNER,
                    review_digest=proposal["review_digest"])
    assert registry.resolve(homes["work-2"]).profile == "work"
    with pytest.raises(InstallationError, match="not a fallback"):
        registry.resolve(homes["work"])
    assert len(registry.receipts("work")) == 2, "a move is a reviewed change, not a edit"


def test_a_retired_home_stops_resolving_even_though_the_row_remains(registry, home):
    enroll(registry, "work", home / "homes" / "work")
    registry.retire("work", actor=OWNER, reason="laptop sold")
    with pytest.raises(InstallationError, match="was retired for"):
        registry.resolve(home / "homes" / "work")
    assert registry.profile("work").state == "retired", \
        "inventory still says it was retired; only access is gone"
    assert registry.names() == []
    assert [item.profile for item in registry.profiles(include_retired=True)] == ["work"]


def test_an_unknown_profile_has_no_invented_row(registry):
    with pytest.raises(InstallationError, match="is not enrolled"):
        registry.profile("work")


# -- the review ---------------------------------------------------------------

def test_the_plan_writes_nothing_and_names_every_path_it_would(registry, home):
    proposal = registry.plan("work", home / "homes" / "work")
    assert {key: proposal[key] for key in PROPOSAL_KEYS} == {
        "profile": "work", "hermes_home": str(home / "homes" / "work"),
        "data_dir": str(home / "profiles" / "work"), "bank_id": "hermes-work",
        "credential_scope": "profile-work"}
    assert proposal["store"] == "created" and proposal["would_write"] is True
    assert proposal["main_model_unchanged"] is True, "enrollment never selects a provider"
    assert not (home / "profiles").exists()
    assert registry.names() == []


def test_the_default_profile_keeps_the_configured_data_dir_and_bank(registry, home):
    default = registry.plan(DEFAULT_PROFILE, home / "homes" / "default")
    assert default["data_dir"] == str(home / "data")
    assert default["bank_id"] == "hermes", "the single-profile name is the one in use"
    assert default["bank_id"] != registry.plan("work", home / "homes" / "work")["bank_id"]


def test_the_review_digest_pins_the_tagged_envelope(registry, home):
    """An approval is only re-checkable if what it covers is written down somewhere.

    The tag is what makes an old approval meaningless after a format change rather than
    accidentally valid, and the five keys are the whole of what can be approved.
    """
    proposal = registry.plan("work", home / "homes" / "work")
    assert proposal["review_digest"] == digest([
        "profile-enroll-v1",
        {"profile": "work", "hermes_home": str(home / "homes" / "work"),
         "data_dir": str(home / "profiles" / "work"), "bank_id": "hermes-work",
         "credential_scope": "profile-work"}])


def test_a_review_digest_covers_the_proposal_only_so_an_interruption_resumes(registry, home):
    proposal = registry.plan("work", home / "homes" / "work")
    directory = registry._data_dir("work")
    directory.mkdir(parents=True, mode=0o700)
    (directory / "canonical.db").write_text("", encoding="utf-8")
    again = registry.plan("work", home / "homes" / "work")
    assert again["review_digest"] == proposal["review_digest"], \
        "the presence of a store must not invalidate an approval already given"
    assert again["store"] == "existing" and again["would_write"] is True
    result = registry.enroll("work", home / "homes" / "work", actor=OWNER,
                             review_digest=proposal["review_digest"])
    assert result["changed"] is True and result["store"] == "existing"


@pytest.mark.parametrize("approval,expect", [
    ("", "no default approval"), ("   ", "no default approval"),
    (None, "no default approval"), (42, "no default approval"),
    ("0" * 64, "does not match what would change")])
def test_an_approval_that_is_not_the_current_one_enrolls_nothing(registry, home,
                                                                 approval, expect):
    with pytest.raises(InstallationError, match=expect):
        registry.enroll("work", home / "homes" / "work", actor=OWNER,
                        review_digest=approval)
    assert registry.names() == [] and registry.receipts() == []


def test_an_approval_for_one_home_does_not_authorise_another(registry, home):
    work = registry.plan("work", home / "homes" / "work")
    with pytest.raises(InstallationError, match="does not match what would change"):
        registry.enroll("work", home / "homes" / "elsewhere", actor=OWNER,
                        review_digest=work["review_digest"])


@pytest.mark.parametrize("actor", [None, "", "   ", "agent", "hermes", 7])
def test_only_the_owner_principal_may_enroll(registry, home, actor):
    proposal = registry.plan("work", home / "homes" / "work")
    with pytest.raises(InstallationError, match="only the owner"):
        registry.enroll("work", home / "homes" / "work", actor=actor,
                        review_digest=proposal["review_digest"])
    assert registry.names() == []


def test_an_unset_owner_cannot_enroll_anybody(home):
    db = open_installation(home / "installation.db")
    locked = ProfileRegistry(db, root=home, owner_principal=None, default_home=home / "data")
    proposal = locked.plan("work", home / "homes" / "work")
    with pytest.raises(InstallationError, match="HERMES_MEMORY_OWNER_PRINCIPAL"):
        locked.enroll("work", home / "homes" / "work", actor=OWNER,
                      review_digest=proposal["review_digest"])


def test_one_home_cannot_be_claimed_by_a_second_profile(registry, home):
    enroll(registry, "work", home / "homes" / "work")
    with pytest.raises(InstallationError, match="already owns this home or bank"):
        enroll(registry, "personal", home / "homes" / "work")
    assert registry.names() == ["work"]


def test_a_retired_home_still_blocks_a_new_owner(registry, home):
    """Retiring unlinks a profile; it does not hand that conversation to a successor."""
    enroll(registry, "work", home / "homes" / "work")
    registry.retire("work", actor=OWNER, reason="paused")
    with pytest.raises(InstallationError, match="already owns this home or bank"):
        enroll(registry, "successor", home / "homes" / "work")


def test_enrollment_is_one_transaction_with_its_receipt(registry, home, monkeypatch):
    monkeypatch.setattr(ProfileRegistry, "_receipt",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk gone")))
    with pytest.raises(RuntimeError, match="disk gone"):
        enroll(registry, "work", home / "homes" / "work")
    assert registry.names() == [], "a half-written enrollment is an unenrolled profile"
    assert registry.receipts() == []


def test_renrolling_the_same_state_changes_nothing_and_records_no_approval(registry, home):
    first = enroll(registry, "work", home / "homes" / "work")
    assert first["changed"] is True
    again = registry.plan("work", home / "homes" / "work")
    second = registry.enroll("work", home / "homes" / "work", actor=OWNER,
                             review_digest=again["review_digest"])
    assert second["changed"] is False and "receipt_id" not in second, \
        "a no-op rerun must not look like a fresh approval"
    assert len(registry.receipts("work")) == 1
    assert [row["id"] for row in registry.receipts("work")] == [first["receipt_id"]]


# -- retirement ---------------------------------------------------------------

def test_retiring_leaves_every_byte_of_evidence_where_it_is(registry, home):
    result = enroll(registry, "work", home / "homes" / "work")
    retired = registry.retire("work", actor=OWNER, reason="laptop sold")
    assert retired["data_dir"] == result["data_dir"] == str(home / "profiles" / "work")
    assert retired["bank_id"] == result["bank_id"] == "hermes-work"
    assert retired["state"] == "retired" and retired["changed"] is True
    assert (registry.profile("work").review_digest == result["review_digest"]), \
        "retirement records a decision, it does not rewrite the proposal"
    assert not (home / "profiles" / "work").exists(), "nothing was created or deleted"


@pytest.mark.parametrize("actor", [None, "agent", ""])
def test_only_the_owner_may_retire(registry, home, actor):
    enroll(registry, "work", home / "homes" / "work")
    with pytest.raises(InstallationError, match="only the owner"):
        registry.retire("work", actor=actor, reason="whatever")
    assert registry.names() == ["work"]


@pytest.mark.parametrize("reason", ["", "   ", None, 9])
def test_retiring_without_a_reason_is_refused(registry, home, reason):
    enroll(registry, "work", home / "homes" / "work")
    with pytest.raises(InstallationError, match="reason must be nonempty text"):
        registry.retire("work", actor=OWNER, reason=reason)
    assert registry.names() == ["work"]


def test_retiring_twice_is_not_two_decisions(registry, home):
    enroll(registry, "work", home / "homes" / "work")
    registry.retire("work", actor=OWNER, reason="laptop sold")
    again = registry.retire("work", actor=OWNER, reason="still gone")
    assert again["changed"] is False and "receipt_id" not in again
    assert [row["action"] for row in registry.receipts("work")] == ["retire", "enroll"]
    with pytest.raises(InstallationError, match="is not enrolled"):
        registry.retire("ghost", actor=OWNER, reason="nothing here")


def test_a_retired_profile_can_be_enrolled_again_by_the_owner(registry, home):
    enroll(registry, "work", home / "homes" / "work")
    registry.retire("work", actor=OWNER, reason="paused")
    assert enroll(registry, "work", home / "homes" / "work")["changed"] is True
    assert registry.resolve(home / "homes" / "work").profile == "work"
    assert [row["action"] for row in registry.receipts("work")] == \
           ["enroll", "retire", "enroll"]


# -- receipts -----------------------------------------------------------------

def test_receipts_are_newest_first_and_record_what_was_shown(registry, home):
    work = registry.plan("work", home / "homes" / "work")
    enroll(registry, "work", home / "homes" / "work")
    registry.retire("work", actor=OWNER, reason="laptop sold")
    rows = registry.receipts()
    assert [row["action"] for row in rows] == ["retire", "enroll"]
    assert rows[1]["review_digest"] == work["review_digest"]
    assert rows[1]["changes"] == {key: work[key] for key in PROPOSAL_KEYS}, \
        "advisory fields are shown, but the receipt records the state that was approved"
    assert rows[0]["result"] == "laptop sold" and rows[1]["result"] == "enrolled"
    assert all(row["actor"] == OWNER for row in rows)
    assert rows[0]["at"] >= rows[1]["at"]


@pytest.mark.parametrize("limit", [0, -1, 501, "20", None, 1.5])
def test_the_receipt_window_is_bounded(registry, limit):
    with pytest.raises(InstallationError, match="between 1 and 500"):
        registry.receipts(limit=limit)


def test_receipt_lookup_by_digest(registry, home):
    proposal = registry.plan("work", home / "homes" / "work")
    enroll(registry, "work", home / "homes" / "work")
    found = registry.receipt_for(proposal["review_digest"])
    assert found["profile"] == "work" and found["action"] == "enroll"
    assert set(found) == {"id", "profile", "action", "actor", "at"}
    assert registry.receipt_for("0" * 64) is None


@pytest.mark.parametrize("approval", ["", "  ", None, 5])
def test_a_blank_digest_lookup_is_a_refusal_not_a_catch_all(registry, approval):
    with pytest.raises(InstallationError, match="review digest must be nonempty text"):
        registry.receipt_for(approval)


def test_receipts_for_an_unknown_profile_are_empty_not_an_error(registry, home):
    enroll(registry, "work", home / "homes" / "work")
    assert registry.receipts("personal") == []


# -- profile names and shape --------------------------------------------------

@pytest.fixture()
def bare_registry():
    return bare()


@pytest.mark.parametrize("given,expected", [
    ("Work", "work"), ("  work  ", "work"), ("work/laptop", "work-laptop"),
    ("W-2 ", "w-2"), ("Müller", "m-ller"), ("../etc", "etc"), ("a b c", "a-b-c"),
    ("--x--", "x")])
def test_a_profile_name_is_a_slug_not_a_path(bare_registry, given, expected):
    plan = bare_registry.plan(given, "/h/x")
    assert plan["profile"] == expected
    assert plan["bank_id"] == f"hermes-{expected}"
    assert plan["credential_scope"] == f"profile-{expected}"
    assert plan["data_dir"].endswith(f"/profiles/{expected}")


@pytest.mark.parametrize("given", ["", "   ", None, 42])
def test_a_profile_name_cannot_be_blank(bare_registry, given):
    with pytest.raises(InstallationError, match="must be nonempty text"):
        bare_registry.plan(given, "/h/x")


@pytest.mark.parametrize("given", ["x" * 300, "工作", "☃"])
def test_a_profile_name_must_be_mappable_to_a_slug(bare_registry, given):
    with pytest.raises(InstallationError, match="name characters"):
        bare_registry.plan(given, "/h/x")


def test_a_profile_exposes_the_paths_the_store_needs(registry, home):
    enroll(registry, "work", home / "homes" / "work")
    profile = registry.profile("work")
    assert isinstance(profile, Profile)
    assert profile.db_path == home / "profiles" / "work" / "canonical.db"
    assert profile.blob_dir == home / "profiles" / "work" / "blobs"
    assert profile.enrolled_by == OWNER and profile.enrolled_at
    assert profile.state == "enrolled"
    public = profile.as_dict()
    assert "hermes_home" not in public, "a home path is not for a model-facing summary"
    assert profile.as_dict(private=True)["hermes_home"] == str(home / "homes" / "work")


def test_per_profile_state_never_overlaps(registry, home):
    homes = activities(home, "work", "personal", "family", "default")
    for name in ("work", "personal", "family", DEFAULT_PROFILE):
        enroll(registry, name, homes[name])
    rows = [registry.profile(name) for name in registry.names()]
    assert len(rows) == 4
    for field in ("bank_id", "data_dir", "credential_scope", "hermes_home"):
        values = [getattr(row, field) for row in rows]
        assert len(set(values)) == len(values), f"{field} is shared between profiles"


def test_the_registry_holds_no_evidence_of_its_own(registry, home):
    """The ledger maps profiles to memory; it must not become a second memory."""
    enroll(registry, "work", home / "homes" / "work")
    tables = {row[0] for row in registry.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"schema_migrations", "profiles", "enrollment_receipts"}
