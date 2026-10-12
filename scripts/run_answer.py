"""One real retrieval+generation example, saved without credentials."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.generation import GenerationError, generate
from app.retrieval import ROOT, ChineseBM25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="中国人民大学中法学院用哪三个语言进行教学？")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    retriever = ChineseBM25()
    retriever.load()
    start = time.perf_counter()
    hits = retriever.search(args.question, args.top_k)
    retrieval_ms = (time.perf_counter() - start) * 1000
    try:
        answer = generate(args.question, hits)
    except GenerationError as exc:
        print(str(exc))
        raise SystemExit(1)
    report = {"question": args.question, "retrieved_ids": [str(hit["id"]) for hit in hits],
              "retrieval_ms": retrieval_ms, "total_ms": (time.perf_counter() - start) * 1000,
              **answer}
    path = ROOT / "artifacts/reports/first_answer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Console output intentionally excludes full prompts, API configuration and key.
    print(json.dumps({key: value for key, value in report.items() if key != "evidence"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
