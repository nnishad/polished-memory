"""C14 §10.3: the compatibility manifest is generated, and generated things can be checked.

A manifest that is typed in is a second source of truth, and the second one is always the
liar. Each case here changes one claim and requires the difference to be named — including
the claim that the manifest itself makes about which code it describes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_memory.backend.capabilities import PINNED_VERSION
from hermes_memory.install import compatibility
from hermes_memory.storage.migrations import MIGRATIONS


@pytest.fixture()
def shipped(tmp_path):
    """A copy of what this build would write, in a directory the test owns."""
    path = tmp_path / "compatibility.json"
    compatibility.write(path=path)
    return path


def test_the_generated_manifest_says_what_this_build_says(shipped):
    assert compatibility.verify(path=shipped)["ok"] is True


def test_the_claims_are_read_from_the_code_that_enforces_them(shipped):
    """Not copied from it: read, so the two cannot disagree by construction.

    A pinned version typed into a JSON file would keep saying ``0.10.1`` after the table
    moved, which is exactly the drift this file is supposed to make loud.
    """
    facts = json.loads(shipped.read_text(encoding="utf-8"))
    assert facts["hindsight"]["engine_pinned"] == PINNED_VERSION
    assert facts["schema"]["evidence_migrations"] == len(MIGRATIONS)
    assert [item["name"] for item in facts["hindsight"]["operations"]] == [
        cap.name for cap in compatibility.CAPABILITIES]
    assert facts["plugin"]["checkpoint_api_version"] == 2
    assert facts["package"]["package"] == "hermes-memory"


def test_a_manifest_that_pinned_a_different_engine_is_refused(shipped):
    facts = json.loads(shipped.read_text(encoding="utf-8"))
    facts["hindsight"]["engine_pinned"] = "0.11.0"
    shipped.write_text(json.dumps(facts), encoding="utf-8")
    checked = compatibility.verify(path=shipped)
    assert checked["ok"] is False
    assert any("engine_pinned" in line for line in checked["differences"])


def test_a_manifest_that_dropped_a_migration_step_is_refused(shipped):
    """The schema range is a claim about what this build can read, not a number to admire."""
    facts = json.loads(shipped.read_text(encoding="utf-8"))
    facts["schema"]["evidence_migrations"] = len(MIGRATIONS) - 1
    shipped.write_text(json.dumps(facts), encoding="utf-8")
    assert any("schema" in line for line in
               compatibility.verify(path=shipped)["differences"])


def test_a_manifest_that_added_a_claim_this_build_cannot_make_is_refused(shipped):
    facts = json.loads(shipped.read_text(encoding="utf-8"))
    facts["delivery_guarded"] = True
    shipped.write_text(json.dumps(facts), encoding="utf-8")
    checked = compatibility.verify(path=shipped)
    assert checked["ok"] is False
    assert any("claims compatibility this build does not" in line
               for line in checked["differences"])


def test_the_two_directions_of_schema_compatibility_are_both_stated(shipped):
    """Naming only the current version implies the older one is refused too, and it is not."""
    schema = json.loads(shipped.read_text(encoding="utf-8"))["schema"]
    assert schema["evidence_older"] == "migrated forward on open"
    assert schema["evidence_newer"] == "refused"


def test_a_source_edit_changes_the_digest_without_condemning_the_release(shipped, tmp_path,
                                                                       monkeypatch):
    """Development edits code; it does not change what the release is compatible with.

    A doctor that went red on every save would be ignored inside a week, and the claim
    drift it exists to catch would hide in the noise. The digests are therefore checked by
    the packaging step, and only reported by the running installation.
    """
    facts = json.loads(shipped.read_text(encoding="utf-8"))
    facts["framework_digest"] = "0" * 64
    shipped.write_text(json.dumps(facts), encoding="utf-8")
    assert compatibility.verify(path=shipped)["ok"] is True
    stale = compatibility.verify(path=shipped, digests=True)
    assert stale["ok"] is False and stale["digests"]["agree"] is False
    assert any("framework_digest" in line for line in stale["differences"])


def test_a_plugin_file_that_no_longer_matches_the_release_is_caught_by_the_digests(shipped):
    """The host loads these files; a patched one is a different release than the one cut."""
    facts = json.loads(shipped.read_text(encoding="utf-8"))
    tampered = dict(facts["plugin"]["files"])
    tampered["provider.py"] = "0" * 64
    facts["plugin"]["files"] = tampered
    shipped.write_text(json.dumps(facts), encoding="utf-8")
    assert compatibility.verify(path=shipped)["ok"] is True
    strict = compatibility.verify(path=shipped, digests=True)
    assert strict["ok"] is False and any("plugin" in line for line in strict["differences"])


def test_a_missing_manifest_is_a_difference_with_the_command_that_fixes_it(tmp_path):
    checked = compatibility.verify(path=tmp_path / "nothing.json")
    assert checked["ok"] is False and checked["remedy"]
    assert "does not exist" in checked["differences"][0]
    assert "hermes-memory compatibility --write" in checked["remedy"], (
        "the report that names a fault has to name the door that clears it")


def test_an_unreadable_manifest_is_reported_rather_than_raising(tmp_path):
    path = tmp_path / "compatibility.json"
    path.write_text("{not json", encoding="utf-8")
    checked = compatibility.verify(path=path)
    assert checked["ok"] is False and "not readable JSON" in checked["differences"][0]


def test_the_plugin_claims_no_python_dependencies_of_its_own(shipped):
    """The claim that keeps `hermes plugins enable` out of the agent's environment.

    Read as a list rather than as the two characters ``[]``: a manifest nobody can parse
    programmatically is a manifest nobody checks.
    """
    plugin = json.loads(shipped.read_text(encoding="utf-8"))["plugin"]
    assert plugin["python_dependencies"] == []
    assert plugin["kind"] == "exclusive"


def test_a_release_tree_carrying_its_own_manifest_is_the_one_read(tmp_path, monkeypatch):
    """The installed tree answers for itself; this checkout is only the fallback.

    §10.3 ships the manifest beside the code it describes, so a machine running from
    ``HERMES_MEMORY_RELEASE`` has to be read from there. Otherwise a new checkout on the
    same disk would vouch for an older installed release.
    """
    release = tmp_path / "release"
    target = release / "deployment"
    target.mkdir(parents=True)
    other = target / "compatibility.json"
    other.write_text(json.dumps(compatibility.facts() | {"framework_digest": "x" * 64}),
                     encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_RELEASE", str(release))
    assert compatibility.manifest_path() == other
    monkeypatch.delenv("HERMES_MEMORY_RELEASE")
    assert compatibility.manifest_path() == compatibility.SHIP_AT


def test_the_manifest_claims_nothing_it_cannot_support(shipped):
    """What it does not say is part of what it says.

    A compatibility file that read like a test report would be believed exactly as much as
    it is false: no live backend was called, no model ran, and the host floor is the plugin's
    own claim rather than a matrix somebody exercised.
    """
    facts = json.loads(shipped.read_text(encoding="utf-8"))
    assert len(facts["not_claimed"]) == 3
    assert not any(key not in facts for key in ("package", "hindsight", "schema", "plugin"))


# -- the door an operator can open -------------------------------------------

@pytest.fixture()
def release(tmp_path, monkeypatch):
    """A release tree this test owns, standing where the installer would put one."""
    root = tmp_path / "release"
    (root / "deployment").mkdir(parents=True)
    monkeypatch.setattr(compatibility, "SHIP_AT",
                        root / "deployment" / "compatibility.json")
    monkeypatch.setenv("HERMES_MEMORY_RELEASE", str(root))
    return root / "deployment" / "compatibility.json"


@pytest.fixture()
def door(monkeypatch):
    """Run the CLI and hand back its exit code and the document it printed."""
    from hermes_memory.cli import main

    def call(*argv):
        printed = []
        monkeypatch.setattr("builtins.print",
                            lambda *a, **k: printed.append(a[0] if a else ""))
        code = main(list(argv))
        return code, json.loads(printed[-1])

    return call


def test_the_command_writes_the_manifest_and_then_believes_it(release, door):
    code, written = door("compatibility", "--write")

    assert code == 0
    assert Path(written["written"]) == release and release.is_file()
    assert written["facts"]["hindsight"]["engine_pinned"] == PINNED_VERSION

    code, read = door("compatibility", "--digests")

    assert code == 0, read
    assert read["ok"] is True and read["digests"]["agree"] is True


def test_the_command_refuses_a_manifest_that_stopped_describing_the_build(release, door):
    door("compatibility", "--write")
    facts = json.loads(release.read_text(encoding="utf-8"))
    facts["hindsight"]["engine_pinned"] = "0.11.0"
    release.write_text(json.dumps(facts), encoding="utf-8")

    code, read = door("compatibility")

    assert code == 1, "a release that misstates its own pin must not exit clean"
    assert any("engine_pinned" in line for line in read["differences"])
    assert "compatibility --write" in read["next"]


def test_a_patched_tree_passes_the_claims_and_fails_only_the_digests(release, door,
                                                                     monkeypatch):
    """The documented split, from the door rather than from the module.

    Editing code is not lying about compatibility, so the ordinary check stays green and a
    developer's `doctor` keeps meaning something. Shipping is the act that has to notice the
    two have come apart, and that is what the digest comparison is for.
    """
    door("compatibility", "--write")
    monkeypatch.setattr(compatibility, "_framework_digest", lambda: "f" * 64)

    assert door("compatibility")[0] == 0
    code, strict = door("compatibility", "--digests")

    assert code == 1
    assert strict["digests"] == {"agree": False, "framework": "f" * 64,
                                 "shipped": strict["digests"]["shipped"]}
    assert any("framework_digest" in line for line in strict["differences"])
