"""Single-writer BM25 snapshot demo: build first, atomically switch a pointer.

This is deliberately separate from the official corpus and production API.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from app.retrieval import ChineseBM25, sha256


class VersionStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def directory(self, version: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", version):
            raise ValueError("Invalid version name")
        return self.root / version

    def prepare(self, version: str, documents: list[dict]) -> None:
        directory = self.directory(version)
        if directory.exists():
            raise ValueError("Snapshot already exists; use a new version name")
        if not documents or len({str(doc["id"]) for doc in documents}) != len(documents):
            raise ValueError("Snapshot requires nonempty, unique document IDs")
        if any(not doc["contents"].strip() for doc in documents):
            raise ValueError("Snapshot contains empty text")
        directory.mkdir()
        corpus = directory / "corpus.jsonl"
        corpus.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents), encoding="utf-8")
        index = directory / "bm25"
        manifest = ChineseBM25(corpus, index).build()
        # Readiness is written only after the complete index is built and loaded.
        ChineseBM25(corpus, index).load()
        (directory / "ready.json").write_text(json.dumps({"version": version, **manifest}), encoding="utf-8")

    def checked(self, version: str) -> ChineseBM25:
        directory = self.directory(version)
        manifest = json.loads((directory / "ready.json").read_text(encoding="utf-8"))
        corpus = directory / "corpus.jsonl"
        if manifest["corpus_sha256"] != sha256(corpus):
            raise ValueError("Snapshot text changed after indexing")
        engine = ChineseBM25(corpus, directory / "bm25")
        engine.load()
        return engine

    def activate(self, version: str) -> None:
        self.checked(version)  # Any failed validation leaves the old pointer intact.
        pending = self.root / "active.pending.json"
        pending.write_text(json.dumps({"version": version}), encoding="utf-8")
        os.replace(pending, self.root / "active.json")

    def search(self, question: str, top_k: int = 5) -> dict:
        # Read once: the entire request uses one immutable corpus/index generation.
        version = json.loads((self.root / "active.json").read_text(encoding="utf-8"))["version"]
        return {"version": version, "hits": self.checked(version).search(question, top_k)}
