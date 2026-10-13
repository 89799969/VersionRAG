"""The five explicit retrieval pipelines; a shared reranker gives equal settings."""
from typing import Literal

from app.hybrid import HybridRetriever
from app.reranking import RerankRetriever

RetrievalMethod = Literal["bm25", "dense", "hybrid", "hybrid_rerank", "bm25_rerank"]
METHOD_NAMES = ("bm25", "dense", "hybrid", "hybrid_rerank", "bm25_rerank")


def build_engines(bm25, dense=None, *, candidate_k=30, rrf_constant=60, reranker=None):
    from app.reranking import APIReranker
    chosen = reranker if reranker is not None else APIReranker()
    engines = {"bm25": bm25, "bm25_rerank": RerankRetriever(bm25, chosen, candidate_k)}
    if dense is not None:
        hybrid = HybridRetriever(bm25, dense, candidate_k, rrf_constant)
        engines.update({"dense": dense, "hybrid": hybrid,
                        "hybrid_rerank": RerankRetriever(hybrid, chosen, candidate_k)})
    return {name: engines[name] for name in METHOD_NAMES if name in engines}
