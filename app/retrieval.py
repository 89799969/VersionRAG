"""Chinese BM25 adapter using BM25s, also used by upstream FlashRAG.

Gold answers/evidence are intentionally absent from this module.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import bm25s
import jieba

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/domainrag_text_corpus.jsonl"
INDEX = ROOT / "artifacts/indexes/bm25"
jieba.setLogLevel(logging.WARNING)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tokenize(text: str) -> list[str]:
    # The corpus contains spaces between Chinese words. Normalize these without
    # joining English words; apply exactly the same rule to documents and queries.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text.lower())
    return [word.strip() for word in jieba.lcut(text) if word.strip().isalnum()]


class ChineseBM25:
    def __init__(self, corpus_path: Path = CORPUS, index_path: Path = INDEX):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.documents = read_jsonl(corpus_path)
        self.engine = None

    def build(self) -> dict:
        tokens = bm25s.tokenize(
            [" ".join(tokenize(doc["contents"])) for doc in self.documents],
            token_pattern=r"(?u)\b\w+\b", stopwords=[], show_progress=True,
        )
        self.engine = bm25s.BM25(k1=1.5, b=0.75, method="lucene", backend="numpy")
        self.engine.index(tokens)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.engine.save(str(self.index_path))
        manifest = {
            "corpus_sha256": sha256(self.corpus_path),
            "documents": len(self.documents),
            "tokenization": "jieba-0.42.1/chinese-whitespace-v1",
            "bm25s": "0.2.1", "k1": 1.5, "b": 0.75, "method": "lucene",
        }
        (self.index_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def load(self) -> None:
        manifest = json.loads((self.index_path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["corpus_sha256"] != sha256(self.corpus_path):
            raise ValueError("Corpus changed; rebuild the BM25 index before querying.")
        self.engine = bm25s.BM25.load(str(self.index_path), mmap=True)

    def search(self, question: str, top_k: int = 5, source_id: str | None = None,
               *, allowed_ids: set[str] | None = None) -> list[dict]:
        if self.engine is None:
            self.load()
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        token_ids = [self.engine.vocab_dict[word] for word in tokenize(question)
                     if word in self.engine.vocab_dict]
        if not token_ids:
            return []
        # A filter must apply before TopK. Retrieve the full ranking when filtering
        # so relevant documents cannot disappear merely because another source won.
        k = min(len(self.documents), top_k if source_id is None and allowed_ids is None else len(self.documents))
        rows, scores = self.engine.retrieve([token_ids], k=k, show_progress=False)
        result = []
        for row, score in zip(rows[0], scores[0]):
            doc = self.documents[int(row)]
            if float(score) <= 0:
                continue
            if source_id is not None and str(doc["metadata"]["source_id"]) != source_id:
                continue
            if allowed_ids is not None and str(doc["id"]) not in allowed_ids:
                continue
            result.append({**doc, "score": float(score), "rank": len(result) + 1})
            if len(result) == top_k:
                break
        return result
