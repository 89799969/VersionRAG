"""Selected free rerank API; it can only reorder the supplied candidate pool."""
from __future__ import annotations

import math
import time

import httpx

from app.embedding_api import EmbeddingError, EmbeddingSettings


class RerankError(EmbeddingError):
    pass


class APIReranker:
    model = "BAAI/bge-reranker-v2-m3"

    def __init__(self, settings: EmbeddingSettings | None = None,
                 transport: httpx.BaseTransport | None = None):
        self.settings = settings or EmbeddingSettings.load()
        self.transport = transport

    def rerank(self, question: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if not candidates:
            return []
        unique = {}
        for hit in candidates:
            unique.setdefault(str(hit["id"]), hit)
        hits = list(unique.values())
        encoded_query = question.encode("utf-8")
        if not encoded_query or len(encoded_query) > 4500:
            raise RerankError("Rerank question is blank or too long.")
        budget = 8000 - len(encoded_query)
        texts = [hit["contents"].encode("utf-8")[:budget].decode("utf-8", errors="ignore") for hit in hits]
        body = {"model": self.model, "query": question, "documents": texts,
                "top_n": min(top_k, len(hits)), "return_documents": False,
                "max_chunks_per_doc": 8, "overlap_tokens": 50}
        with httpx.Client(timeout=self.settings.timeout, transport=self.transport, trust_env=False) as client:
            for attempt in range(3):
                try:
                    response = client.post("https://api.siliconflow.cn/v1/rerank", json=body,
                                           headers={"Authorization": f"Bearer {self.settings.api_key}"})
                except httpx.TimeoutException:
                    raise RerankError("Rerank request timed out.") from None
                except httpx.RequestError:
                    raise RerankError("Could not connect to the rerank service.") from None
                if response.status_code in {429, 502, 503, 504} and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                if response.is_error:
                    raise RerankError(f"Rerank service returned HTTP {response.status_code}; no paid fallback.")
                break
        try:
            results = response.json()["results"]
            if len(results) != min(top_k, len(hits)):
                raise ValueError("Incomplete rerank result")
            seen, ranked = set(), []
            for item in results:
                index, score = item["index"], float(item["relevance_score"])
                if not isinstance(index, int) or not 0 <= index < len(hits) or index in seen or not math.isfinite(score):
                    raise ValueError("Invalid rerank index or score")
                seen.add(index)
                ranked.append({**hits[index], "retrieval_score": hits[index]["score"],
                               "original_rank": hits[index]["rank"], "score": score,
                               "rerank_model": self.model,
                               "rerank_input_truncated": len(texts[index]) < len(hits[index]["contents"])})
        except (ValueError, TypeError, KeyError):
            raise RerankError("Rerank service returned malformed rankings.") from None
        ranked.sort(key=lambda hit: (-hit["score"], str(hit["id"])))
        return [{**hit, "rank": rank} for rank, hit in enumerate(ranked, 1)]


class RerankRetriever:
    def __init__(self, retriever, reranker: APIReranker | None = None, candidate_k: int = 30):
        if not 1 <= candidate_k <= 100:
            raise ValueError("candidate_k must be between 1 and 100")
        self.retriever = retriever
        self.reranker = reranker or APIReranker()
        self.candidate_k = candidate_k

    def search(self, question: str, top_k: int = 5, source_id: str | None = None):
        candidates = self.retriever.search(question, max(top_k, self.candidate_k), source_id)
        return self.reranker.rerank(question, candidates, top_k)
