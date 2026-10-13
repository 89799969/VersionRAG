import json

import httpx
from fastapi.testclient import TestClient

from app.api import create_app
from app.embedding_api import EmbeddingSettings
from app.methods import METHOD_NAMES, build_engines
from app.reranking import APIReranker


class Retriever:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.calls = []

    def search(self, question, top_k, source_id=None, **filters):
        self.calls.append((question, top_k, source_id))
        return [{"id": str(i), "title": "申请", "contents": "报名截止日期2月5日",
                 "metadata": {"source_id": "notice"}, "score": 1 / (i + 1), "rank": i + 1}
                for i in range(top_k)]


def test_five_pipelines_share_reranker_and_equal_rerank_input_budget(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.write_text("same corpus", encoding="utf-8")
    bm25, dense = Retriever(corpus), Retriever(corpus)
    sent = []

    def respond(request):
        data = json.loads(request.content)
        sent.append(data)
        return httpx.Response(200, json={"results": [
            {"index": i, "relevance_score": 1 / (i + 1)} for i in range(data["top_n"])
        ]})

    reranker = APIReranker(EmbeddingSettings("offline-test-key"), httpx.MockTransport(respond))
    engines = build_engines(bm25, dense, reranker=reranker)
    assert tuple(engines) == METHOD_NAMES
    for name in ("bm25_rerank", "hybrid_rerank"):
        assert engines[name].reranker is reranker
        result = engines[name].search("截止日期", 5, "notice")
        assert len(result) == 5 and all(h["metadata"]["source_id"] == "notice" for h in result)
    assert [len(data["documents"]) for data in sent] == [30, 30]
    assert all(data["model"] == APIReranker.model for data in sent)
    assert all(call[1:] == (30, "notice") for call in bm25.calls + dense.calls)


def test_keyword_rerank_registration_does_not_require_dense_or_credentials(tmp_path, monkeypatch):
    def forbidden(cls):
        raise AssertionError("Registration must not read credentials")

    monkeypatch.setattr(EmbeddingSettings, "load", classmethod(forbidden))
    corpus = tmp_path / "corpus"
    corpus.write_text("corpus", encoding="utf-8")
    engines = build_engines(Retriever(corpus))
    assert set(engines) == {"bm25", "bm25_rerank"}
    assert engines["bm25"].search("报名", 1)[0]["id"] == "0"


def test_new_method_is_accepted_by_http_schema(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.write_text("corpus", encoding="utf-8")
    with TestClient(create_app({"bm25_rerank": Retriever(corpus)})) as client:
        response = client.post("/retrieve", json={"question": "报名", "method": "bm25_rerank", "top_k": 5})
        assert response.status_code == 200
        assert response.json()["method"] == "bm25_rerank"
        assert response.json()["knowledge_version"] == "baseline-static"
        assert len(response.json()["hits"]) == 5
