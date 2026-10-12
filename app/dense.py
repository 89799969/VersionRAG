"""Cached API/local embeddings and exact cosine retrieval, with resumable indexing."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from app.retrieval import CORPUS, ROOT, read_jsonl, sha256

MODEL_DIR = ROOT / "artifacts/models/bge-small-zh-v1.5"
DENSE_INDEX = ROOT / "artifacts/indexes/dense"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BGEEncoder:
    def __init__(self, model_dir: Path = MODEL_DIR, max_length: int = 512):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.max_length = max_length
        self.model_dir = model_dir
        self.model_path = model_dir / "onnx/model_quantized.onnx"
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.dimension = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["hidden_size"]

    def metadata(self) -> dict:
        lock = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))
        return {"repo": lock["repo"], "revision": lock["revision"],
                "model_sha256": sha256(self.model_path),
                "tokenizer_sha256": sha256(self.model_dir / "tokenizer.json"),
                "pooling": "CLS", "normalization": "L2", "dimension": self.dimension,
                "max_length": self.max_length, "query_instruction": QUERY_INSTRUCTION,
                "provider": "CPUExecutionProvider", "quantized": True,
                "document_preprocessing": "join-whitespace-between-Chinese-characters-v1"}

    def encode(self, texts: list[str], query: bool = False) -> tuple[np.ndarray, int]:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32), 0
        normalized = [re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text) for text in texts]
        if query:
            normalized = [QUERY_INSTRUCTION + text for text in normalized]
        encoded = self.tokenizer.encode_batch(normalized)
        inputs = {"input_ids": np.array([item.ids for item in encoded], dtype=np.int64),
                  "attention_mask": np.array([item.attention_mask for item in encoded], dtype=np.int64),
                  "token_type_ids": np.array([item.type_ids for item in encoded], dtype=np.int64)}
        hidden = self.session.run(None, {name: value for name, value in inputs.items()
                                         if name in self.input_names})[0]
        if hidden.ndim != 3 or hidden.shape[2] != self.dimension:
            raise ValueError("Unexpected BGE token embedding shape; inspect exported model outputs.")
        vectors = hidden[:, 0, :].astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.all(np.isfinite(vectors)) or np.any(norms == 0):
            raise ValueError("Embedding model produced invalid vectors")
        return vectors / norms, sum(bool(item.overflowing) for item in encoded)


def ranked_rows(vectors: np.ndarray, query: np.ndarray, top_k: int,
                allowed_rows: np.ndarray | None = None) -> list[tuple[int, float]]:
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    rows = np.arange(len(vectors)) if allowed_rows is None else np.asarray(allowed_rows, dtype=np.int64)
    if len(rows) == 0:
        return []
    scores = (vectors if allowed_rows is None else vectors[rows]) @ query
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(rows[index]), float(scores[index])) for index in order]


class DenseRetriever:
    def __init__(self, corpus_path: Path = CORPUS, index_path: Path = DENSE_INDEX,
                 encoder: BGEEncoder | None = None):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.documents = read_jsonl(corpus_path)
        self.encoder = encoder
        self.vectors = None

    def get_encoder(self) -> BGEEncoder:
        if self.encoder is None:
            from app.embedding_api import APIEncoder
            self.encoder = APIEncoder()
        return self.encoder

    def build(self, batch_size: int = 16) -> dict:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        encoder = self.get_encoder()
        signature = {"corpus_sha256": sha256(self.corpus_path), "documents": len(self.documents),
                     "encoder": encoder.metadata()}
        self.index_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.index_path / "building.json"
        partial_path = self.index_path / "vectors.partial.npy"
        start_row, truncated = 0, 0
        if checkpoint_path.exists() and partial_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint["signature"] != signature:
                raise ValueError("Partial index configuration differs; use a different index directory.")
            start_row, truncated = checkpoint["completed"], checkpoint["truncated_documents"]
            vectors = np.lib.format.open_memmap(partial_path, mode="r+")
        else:
            vectors = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32,
                                                shape=(len(self.documents), encoder.dimension))
        start_time = time.perf_counter()
        print(f"Encoding {len(self.documents)} passages; resume at {start_row}", flush=True)
        for begin in range(start_row, len(self.documents), batch_size):
            end = min(begin + batch_size, len(self.documents))
            try:
                batch, count = encoder.encode([doc["contents"] for doc in self.documents[begin:end]])
            except Exception:
                # Release the Windows mmap handle before a caller resumes/rebuilds.
                del vectors
                raise
            vectors[begin:end] = batch
            vectors.flush()
            truncated += count
            state = {"signature": signature, "completed": end, "truncated_documents": truncated}
            checkpoint_tmp = checkpoint_path.with_suffix(".tmp")
            checkpoint_tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            checkpoint_tmp.replace(checkpoint_path)
            if begin == start_row or end % (batch_size * 20) == 0 or end == len(self.documents):
                elapsed = time.perf_counter() - start_time
                print(f"Encoded {end}/{len(self.documents)}; {elapsed:.1f}s in this run", flush=True)
        del vectors
        final_path = self.index_path / "vectors.npy"
        partial_path.replace(final_path)
        manifest = {**signature, "vectors_sha256": sha256(final_path),
                    "truncated_documents": truncated,
                    "search": "exact cosine over normalized float32 vectors",
                    "build_run_seconds": time.perf_counter() - start_time}
        manifest_path = self.index_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        checkpoint_path.unlink()
        self.vectors = np.load(final_path, mmap_mode="r")
        return manifest

    def load(self):
        manifest = json.loads((self.index_path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["corpus_sha256"] != sha256(self.corpus_path):
            raise ValueError("Corpus changed; rebuild the Dense index before querying.")
        if manifest["encoder"] != self.get_encoder().metadata():
            raise ValueError("Embedding model or tokenization changed; rebuild the Dense index.")
        path = self.index_path / "vectors.npy"
        if manifest["vectors_sha256"] != sha256(path):
            raise ValueError("Dense vector index checksum mismatch")
        self.vectors = np.load(path, mmap_mode="r")
        if self.vectors.shape != (len(self.documents), manifest["encoder"]["dimension"]):
            raise ValueError("Dense index shape does not match corpus and model")

    def search(self, question: str, top_k: int = 5, source_id: str | None = None,
               *, allowed_ids: set[str] | None = None) -> list[dict]:
        if not question.strip():
            return []
        if self.vectors is None:
            self.load()
        query, _ = self.get_encoder().encode([question], query=True)
        allowed = None if source_id is None and allowed_ids is None else np.array([
            index for index, doc in enumerate(self.documents)
            if (source_id is None or str(doc["metadata"]["source_id"]) == source_id)
            and (allowed_ids is None or str(doc["id"]) in allowed_ids)
        ], dtype=np.int64)
        ranked = ranked_rows(self.vectors, query[0], top_k, allowed)
        return [{**self.documents[row], "score": score, "rank": rank}
                for rank, (row, score) in enumerate(ranked, 1)]
