"""Pure benchmark contract tests; no optional ML imports or live inference."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[2] / "evals" / "benchmark_reranker.py"
    spec = importlib.util.spec_from_file_location("reranker_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def test_template_suffix_survives_truncation(benchmark):
    tokenizer = CharacterTokenizer()
    ids = benchmark.encode(tokenizer, "Which drink?", "coffee " * 1000, 512)
    assert len(ids) == 512
    assert bytes(ids).startswith(benchmark.PREFIX.encode())
    assert bytes(ids).endswith(benchmark.SUFFIX.encode())
    assert b"<Query>: Which drink?" in bytes(ids)


def test_short_evidence_not_padded_by_encoder(benchmark):
    ids = benchmark.encode(CharacterTokenizer(), "Drink?", "coffee", 1024)
    assert len(ids) < 1024
    assert b"<Document>: coffee" in bytes(ids)


def test_insufficient_template_budget_rejected(benchmark):
    with pytest.raises(ValueError, match="token limit"):
        benchmark.encode(CharacterTokenizer(), "Drink?", "coffee", 8)


def test_empirical_quantile(benchmark):
    assert benchmark.quantile([5, 1, 3, 2, 4], .5) == 3
    assert benchmark.quantile([5, 1, 3, 2, 4], .95) == 5
