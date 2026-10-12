"""Run once with the HTTP service stopped; copy the frozen corpus/indexes locally."""
import json

from app.dense import DENSE_INDEX
from app.knowledge import KnowledgeService
from app.retrieval import CORPUS, INDEX, ROOT


def main():
    service = KnowledgeService(ROOT / "artifacts/knowledge")
    result = service.seed(CORPUS, INDEX, DENSE_INDEX)
    report = ROOT / "artifacts/reports/knowledge_seed.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
