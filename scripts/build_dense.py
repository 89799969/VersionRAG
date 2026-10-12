"""Build all corpus embeddings locally; progress is checkpointed after each batch."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dense import BGEEncoder, DenseRetriever, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backend", choices=["api", "local"], default="api")
    args = parser.parse_args()
    encoder = BGEEncoder() if args.backend == "local" else None
    index = ROOT / "artifacts/indexes" / ("dense_local" if args.backend == "local" else "dense")
    print(json.dumps(DenseRetriever(index_path=index, encoder=encoder).build(args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
