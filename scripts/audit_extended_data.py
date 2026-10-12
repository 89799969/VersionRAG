"""Inspect dataset structure without printing long references or any configuration."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "DomainRAG/BCM/labeled_data"


def preview(value, depth=0):
    if isinstance(value, str):
        return value[:160]
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "first": preview(value[0], depth + 1) if value else None}
    if isinstance(value, dict):
        return {key: preview(item, depth + 1) for key, item in value.items()
                if depth < 4}
    return value


for relative in ("multi-doc_qa/multidoc_qa.jsonl", "time-sensitive_qa/time_sensitive.jsonl",
                 "noisy_qa/noisy_qa_ver3.jsonl", "conversation_qa/conversation_qa.jsonl"):
    path = BASE / relative
    with path.open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
    print(json.dumps({"file": relative, "sample": preview(first)}, ensure_ascii=False, indent=2))
