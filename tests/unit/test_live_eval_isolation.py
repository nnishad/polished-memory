"""Every live test path is checked before a socket or canonical writer exists."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from hermes_memory.config import load_settings

spec = importlib.util.spec_from_file_location("memory_intelligence_eval",
    Path(__file__).resolve().parents[2] / "evals" / "run_memory_intelligence.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_all_canonical_paths_are_rebound_not_only_data_directory(tmp_path):
    original = load_settings(tmp_path / "absent.env")
    root = tmp_path / "scratch"
    scoped = runner.isolated_settings(original, root, "eval-synthetic")
    assert scoped.db_path == root / "canonical.db"
    assert scoped.blob_dir == root / "blobs"
    assert scoped.db_path != original.db_path


@pytest.mark.parametrize("escaped", ["data_dir", "db_path", "blob_dir"])
def test_any_production_path_fails_before_external_work(tmp_path, escaped):
    original = load_settings(tmp_path / "absent.env")
    root = tmp_path / "scratch"
    scoped = runner.isolated_settings(original, root, "eval-synthetic")
    broken = replace(scoped, **{escaped: getattr(original, escaped)})
    with pytest.raises(ValueError, match="isolation refused"):
        runner.assert_isolated(original, broken, root)


def test_the_original_partial_replace_bug_is_reproduced_and_refused(tmp_path):
    original = load_settings(tmp_path / "absent.env")
    root = tmp_path / "scratch"
    with pytest.raises(ValueError, match="db_path"):
        runner.assert_isolated(original, replace(original, data_dir=root, bank_id="eval-synthetic"), root)


def test_production_bank_can_never_be_an_evaluation_target(tmp_path):
    original = load_settings(tmp_path / "absent.env")
    root = tmp_path / "scratch"
    scoped = runner.isolated_settings(original, root, "eval-synthetic")
    with pytest.raises(ValueError, match="evaluation bank"):
        runner.assert_isolated(original, replace(scoped, bank_id=original.bank_id), root)
