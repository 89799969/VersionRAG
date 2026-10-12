"""Evidence metrics; these measure retrieval, never answer correctness."""
from __future__ import annotations

import hashlib


def split_for(sample: dict) -> str:
    # Shared source scenarios stay together. Corpus knowledge is available in both
    # splits, but QA labels in holdout must not be used to choose parameters.
    sources = sorted({str(group["source_id"]) for group in sample["required_evidence_groups"]})
    # Source 159 was inspected during data selection and used as the public demo.
    # It cannot be advertised as untouched holdout, even before tuning starts.
    if "159" in sources:
        return "demo"
    key = "|".join(sources)
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10
    return "dev" if bucket < 3 else "holdout"


def evidence_metrics(sample: dict, retrieved_ids: list[str], k: int) -> dict:
    groups = [set(map(str, group["acceptable_corpus_passage_ids"]))
              for group in sample["required_evidence_groups"]]
    if not groups:
        raise ValueError("Evidence metrics require at least one annotated evidence group.")
    ranked = list(map(str, retrieved_ids[:k]))
    present = set(ranked)
    coverage = sum(bool(group & present) for group in groups) / len(groups)
    first_rank = next((i for i, doc_id in enumerate(ranked, 1)
                       if any(doc_id in group for group in groups)), None)
    return {"evidence_group_recall": coverage, "all_evidence_hit": coverage == 1.0,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0}
