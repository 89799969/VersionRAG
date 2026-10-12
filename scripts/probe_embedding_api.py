"""Small real-model probe before encoding the full corpus; never prints the key."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.embedding_api import APIEncoder, EmbeddingError
from app.retrieval import ROOT


def main():
    encoder = APIEncoder()
    start = time.perf_counter()
    vectors, truncated = encoder.encode(["中法学院采用汉语、法语、英语三语教学。", "物理与化学专业招生信息。"])
    if vectors.shape != (2, 1024):
        raise ValueError("Unexpected model response shape")
    report = {"real_api_called": True, "encoder": encoder.metadata(),
              "shape": list(vectors.shape), "finite": bool(np.all(np.isfinite(vectors))),
              "normalized_lengths": np.linalg.norm(vectors, axis=1).tolist(),
              "truncated_inputs": truncated, "latency_ms": (time.perf_counter() - start) * 1000,
              "scope": "service integration probe; not retrieval quality evaluation"}
    path = ROOT / "artifacts/reports/embedding_api_probe.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except EmbeddingError as exc:
        print(str(exc))
        raise SystemExit(1)
