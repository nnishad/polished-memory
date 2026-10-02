from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_memory.config import env_file_values, load_settings
from hermes_memory.install.reranking import apply, plan


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path))
    framework = tmp_path / "hermes-memory.env"
    framework.write_text("HERMES_MEMORY_OWNER_PRINCIPAL=owner\nHERMES_MEMORY_ADMISSION_URL=http://127.0.0.1:8123\n")
    native = tmp_path / "hindsight.env"
    native.write_text("HINDSIGHT_API_RERANKER_PROVIDER=rrf\n# preserve\n")
    framework.chmod(0o600)
    native.chmod(0o600)
    return load_settings()


def stopped(*args, **kwargs):
    return SimpleNamespace(stdout="inactive\n", returncode=3)


def test_apply_unique_private_credential_and_local_only(settings):
    proposal = plan(settings)
    result = apply(settings, actor="owner", review=proposal["review_digest"], runner=stopped)
    framework = env_file_values(settings.home / "hermes-memory.env")
    native = env_file_values(settings.home / "hindsight.env")
    assert framework["HERMES_MEMORY_ROUTE_CREDENTIAL_RERANK"] == native["HINDSIGHT_API_RERANKER_COHERE_API_KEY"]
    assert native["HINDSIGHT_API_RERANKER_COHERE_BASE_URL"] == "http://127.0.0.1:8123/v1/rerank"
    assert native["HINDSIGHT_API_RERANKER_PROVIDER"] == "cohere"
    assert "# preserve" in (settings.home / "hindsight.env").read_text()
    assert (settings.home / "hindsight.env").stat().st_mode & 0o077 == 0
    assert Path(result["backup"]).is_dir()
    assert framework["HERMES_MEMORY_ROUTE_CREDENTIAL_RERANK"] not in str(result)


def test_owner_review_and_stopped_services_required(settings):
    proposal = plan(settings)
    with pytest.raises(ValueError, match="owner"):
        apply(settings, actor="other", review=proposal["review_digest"], runner=stopped)
    with pytest.raises(ValueError, match="changed"):
        apply(settings, actor="owner", review="stale", runner=stopped)
    with pytest.raises(ValueError, match="stop"):
        apply(settings, actor="owner", review=proposal["review_digest"],
              runner=lambda *a, **k: SimpleNamespace(stdout="active\n"))


def test_public_configuration_refused(settings):
    (settings.home / "hindsight.env").chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        plan(settings)
