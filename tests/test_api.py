from fastapi.testclient import TestClient

from app.api import create_app
from app.generation import GenerationError


class FakeRetriever:
    def search(self, question, top_k, source_id):
        return [{"id": "1189", "title": "中法学院", "contents": "汉语、法语、英语",
                 "metadata": {"source_id": "159"}, "score": 1.0, "rank": 1}]


def mock_generate(question, hits):
    return {"answer": "汉语、法语、英语", "citations": ["1189"], "insufficient_evidence": False,
            "evidence": [], "model_called": False, "model": "mock-for-tests"}


def test_http_request_validates_schema_and_preserves_trace_id():
    with TestClient(create_app({"bm25": FakeRetriever()}, mock_generate)) as client:
        response = client.post("/retrieve", json={"question": "教学语言？"})
        assert response.status_code == 200
        assert response.json()["hits"][0]["id"] == "1189"
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
        assert client.post("/ask", json={"question": "教学语言？"}).json()["model"] == "mock-for-tests"
        assert client.post("/ask", json={"question": "   "}).status_code == 422
        assert client.post("/ask", json={"question": "语言？", "top_k": 0}).status_code == 422
        assert client.post("/ask", json={"question": "语言？", "unexpected": True}).status_code == 422
        assert client.post("/retrieve", json={"question": "语言？", "method": "dense"}).status_code == 503


def test_model_timeout_has_explicit_http_error_no_fake_success():
    def timeout(question, hits):
        raise GenerationError("Model request timed out.")
    with TestClient(create_app({"bm25": FakeRetriever()}, timeout)) as client:
        response = client.post("/ask", json={"question": "语言？"})
        assert response.status_code == 504
        assert response.headers["X-Request-ID"]
