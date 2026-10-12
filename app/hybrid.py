"""Rank-based fusion; BM25 scores and cosine values are deliberately not added."""
from __future__ import annotations

from app.retrieval import sha256


def rrf_fuse(channels: dict[str, list[dict]], top_k: int = 5, constant: int = 60) -> list[dict]:
    if constant <= 0 or not 1 <= top_k <= 100:
        raise ValueError("Invalid RRF constant or top_k")
    combined = {}
    for name, hits in channels.items():
        seen = set()
        for rank, hit in enumerate(hits, 1):
            passage_id = str(hit["id"])
            if passage_id in seen:
                continue
            seen.add(passage_id)
            if passage_id not in combined:
                combined[passage_id] = {**hit, "score": 0.0, "channel_ranks": {}}
            previous = combined[passage_id]
            if previous["contents"] != hit["contents"]:
                raise ValueError("Same passage ID has conflicting texts across retrieval channels.")
            previous["score"] += 1.0 / (constant + rank)
            previous["channel_ranks"][name] = rank
    ordered = sorted(combined.values(), key=lambda hit: (-hit["score"], str(hit["id"])))[:top_k]
    return [{**hit, "rank": rank} for rank, hit in enumerate(ordered, 1)]


class HybridRetriever:
    def __init__(self, bm25, dense, candidate_k: int = 30, rrf_constant: int = 60):
        if sha256(bm25.corpus_path) != sha256(dense.corpus_path):
            raise ValueError("Hybrid retrieval requires the same corpus for both channels.")
        if not 1 <= candidate_k <= 100:
            raise ValueError("candidate_k must be between 1 and 100")
        self.bm25, self.dense = bm25, dense
        self.candidate_k, self.rrf_constant = candidate_k, rrf_constant

    def search(self, question: str, top_k: int = 5, source_id: str | None = None,
               *, allowed_ids: set[str] | None = None):
        count = max(top_k, self.candidate_k)
        filters = {} if allowed_ids is None else {"allowed_ids": allowed_ids}
        channels = {"bm25": self.bm25.search(question, count, source_id, **filters),
                    "dense": self.dense.search(question, count, source_id, **filters)}
        return rrf_fuse(channels, top_k, self.rrf_constant)
