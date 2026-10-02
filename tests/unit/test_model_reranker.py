import concurrent.futures
import threading

import pytest

from hermes_memory.models.reranker import Admission, Unavailable, batches, validate
from hermes_memory.models.reranker import (MODEL, ScoredRows, cohere_request,
                                          cohere_response, validate_cohere_response)


def test_authenticated_adapter_contract_and_actual_usage():
    adapted = cohere_request({"model": MODEL, "query": "coffee?", "documents": ["coffee", "train"],
                              "return_documents": False})
    assert adapted["texts"] == ["coffee", "train"]
    result = cohere_response(ScoredRows([{"index": 0, "score": .9}, {"index": 1, "score": .1}],
                                       input_tokens=150, padded_tokens=200))
    validate_cohere_response(result, 2)
    assert result["usage"]["total_tokens"] == 200


@pytest.mark.parametrize("rows", [[], [{"index": 0, "relevance_score": .5}] * 2,
    [{"index": 0, "relevance_score": float("nan")}, {"index": 1, "relevance_score": .1}],
    [{"index": True, "relevance_score": .1}, {"index": 0, "relevance_score": .5}],
    [{"index": 0, "relevance_score": 2}, {"index": 1, "relevance_score": .1}]])
def test_adapter_rejects_partial_or_corrupt_scores(rows):
    with pytest.raises(ValueError):
        validate_cohere_response({"results": rows, "usage": {"input_tokens": 100, "total_tokens": 100}}, 2)


def test_adapter_requires_pinned_model_and_refuses_extra_fields():
    for extra in ({"model": "other"}, {"return_documents": True}, {"upstream": "http://example.org"}):
        with pytest.raises(ValueError):
            cohere_request({"model": MODEL, "query": "q", "documents": ["d"], **extra})


@pytest.mark.parametrize("usage", [{}, {"input_tokens": 100, "total_tokens": 50},
                                  {"input_tokens": True, "total_tokens": 100},
                                  {"input_tokens": 100, "total_tokens": 2049}])
def test_adapter_requires_bounded_actual_usage(usage):
    with pytest.raises(ValueError):
        validate_cohere_response({"results": [{"index": 0, "relevance_score": .5}], "usage": usage}, 1)


def test_batching_uses_padded_cost_and_preserves_all_indices():
    lengths = [2048, 128, 512, 128, 2048, 500, 2048, 2048]
    chunks = batches(lengths, max_batch=4, token_budget=4096)
    assert sorted(i for chunk in chunks for i in chunk) == list(range(len(lengths)))
    assert all(len(c) <= 4 and len(c) * max(lengths[i] for i in c) <= 4096 for c in chunks)
    assert chunks[0] == [1, 3, 5, 2]


def test_operating_envelope():
    assert all(len(c) <= 8 for c in batches([2048] * 128))
    assert all(len(c) <= 32 for c in batches([512] * 128))
    assert all(len(c) <= 64 for c in batches([128] * 128))


@pytest.mark.parametrize("payload", [None, {}, {"query": "q", "texts": []},
    {"query": "q", "texts": [""]}, {"query": "q", "texts": ["x"] * 129},
    {"query": "q", "texts": ["x"], "return_text": 1},
    {"query": "q", "texts": ["x"], "top_n": True},
    {"query": "q", "texts": ["x"], "model": "wrong"}])
def test_invalid_request(payload):
    with pytest.raises(ValueError):
        validate(payload)


def test_queue_capacity_released_on_completion():
    release = threading.Event()

    class Model:
        def score(self, payload):
            assert release.wait(2)
            return [payload["query"]]

    admission = Admission(Model(), slots=1)
    first = admission.submit({"query": "first", "texts": ["x"]})
    try:
        with pytest.raises(Unavailable, match="queue"):
            admission.submit({"query": "second", "texts": ["x"]})
        release.set()
        assert first.result(timeout=2) == ["first"]
        second = admission.submit({"query": "second", "texts": ["x"]})
        assert second.result(timeout=2) == ["second"]
    finally:
        release.set()
        admission.executor.shutdown(wait=True, cancel_futures=True)


def test_cancelled_queued_request_releases_its_slot():
    release, started = threading.Event(), threading.Event()

    class Model:
        def score(self, payload):
            started.set()
            assert release.wait(2)
            return payload["query"]

    admission = Admission(Model(), slots=2)
    try:
        first = admission.submit({"query": "first", "texts": ["x"]})
        assert started.wait(2)
        second = admission.submit({"query": "second", "texts": ["x"]})
        assert second.cancel()
        third = admission.submit({"query": "third", "texts": ["x"]})
        release.set()
        assert first.result(timeout=2) == "first"
        assert third.result(timeout=2) == "third"
    finally:
        release.set()
        admission.executor.shutdown(wait=True, cancel_futures=True)


def test_failed_request_releases_its_slot():
    class Model:
        def score(self, payload):
            raise ValueError("synthetic invalid input")

    admission = Admission(Model(), slots=1)
    try:
        for _ in range(2):
            with pytest.raises(ValueError, match="synthetic"):
                admission.submit({"query": "q", "texts": ["x"]}).result(timeout=2)
    finally:
        admission.executor.shutdown(wait=True, cancel_futures=True)
