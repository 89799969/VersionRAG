import httpx
import pytest

from app.embedding_api import EmbeddingSettings
from app.reranking import APIReranker, RerankError

HITS = [{"id": "a", "contents": "苹果", "score": 0.03, "rank": 1},
        {"id": "b", "contents": "香蕉", "score": 0.02, "rank": 2}]


def reranker(results):
    return APIReranker(EmbeddingSettings("test-key"), httpx.MockTransport(
        lambda request: httpx.Response(200, json={"results": results})))


def test_rerank_preserves_passage_identity_and_original_rank():
    result = reranker([{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]).rerank(
        "香蕉", HITS, 2)
    assert [row["id"] for row in result] == ["b", "a"]
    assert result[0]["original_rank"] == 2 and result[0]["retrieval_score"] == 0.02
    assert result[0]["rank"] == 1


def test_rerank_cannot_invent_candidate_indices_or_duplicate_results():
    for results in ([{"index": 8, "relevance_score": 0.9}],
                    [{"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]):
        with pytest.raises(RerankError, match="malformed"):
            reranker(results).rerank("香蕉", HITS, len(results))


def test_no_candidates_needs_no_rerank_call():
    assert reranker([]).rerank("香蕉", []) == []
