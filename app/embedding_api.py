"""Free-model-only SiliconFlow embedding adapter. Never switches to paid models."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import httpx
import numpy as np
from dotenv import load_dotenv

from app.retrieval import ROOT
from app.telemetry import tracked_post


class EmbeddingError(Exception):
    """Sanitized service failure, without key or response-body content."""


@dataclass(frozen=True)
class EmbeddingSettings:
    api_key: str = field(repr=False)
    model: str = "BAAI/bge-m3"
    timeout: float = 60

    @classmethod
    def load(cls):
        load_dotenv(ROOT / ".env", override=False)
        key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        if not key:
            raise EmbeddingError("Configure SILICONFLOW_API_KEY in the local .env file.")
        return cls(key)


class APIEncoder:
    dimension = 1024
    max_input_bytes = 8000  # conservative input bound below the 8192-token API limit

    def __init__(self, settings: EmbeddingSettings | None = None,
                 transport: httpx.BaseTransport | None = None):
        self.settings = settings or EmbeddingSettings.load()
        if self.settings.model != "BAAI/bge-m3":
            raise EmbeddingError("This adapter only permits the selected free BAAI/bge-m3 model.")
        self.transport = transport

    def metadata(self) -> dict:
        return {"provider": "SiliconFlow API", "base_url": "https://api.siliconflow.cn/v1",
                "model": self.settings.model, "model_revision": "not exposed by API",
                "dimension": self.dimension, "normalization": "L2", "query_instruction": "",
                "max_input_utf8_bytes": self.max_input_bytes,
                "document_preprocessing": "join-whitespace-between-Chinese-characters-v1"}

    def encode(self, texts: list[str], query: bool = False) -> tuple[np.ndarray, int]:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32), 0
        inputs, truncated = [], 0
        for text in texts:
            normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
            raw = normalized.encode("utf-8")
            truncated += len(raw) > self.max_input_bytes
            clipped = raw[:self.max_input_bytes].decode("utf-8", errors="ignore")
            if not clipped.strip():
                raise EmbeddingError("Embedding input must not be blank.")
            inputs.append(clipped)
        with httpx.Client(timeout=self.settings.timeout, transport=self.transport, trust_env=False) as client:
            for attempt in range(3):
                try:
                    response = tracked_post(client, "https://api.siliconflow.cn/v1/embeddings",
                                           stage='embedding', model=self.settings.model,
                                           headers={"Authorization": f"Bearer {self.settings.api_key}"},
                                           json={"model": self.settings.model, "input": inputs,
                                                 "encoding_format": "float"})
                except httpx.TimeoutException:
                    raise EmbeddingError("Embedding request timed out; saved index batches can be resumed.") from None
                except httpx.RequestError:
                    raise EmbeddingError("Could not connect to the embedding service.") from None
                if response.status_code in {429, 502, 503, 504} and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                if response.is_error:
                    raise EmbeddingError(f"Embedding service returned HTTP {response.status_code}; no paid fallback.")
                break
        try:
            rows = response.json()["data"]
            if len(rows) != len(inputs) or sorted(row["index"] for row in rows) != list(range(len(inputs))):
                raise ValueError("Incomplete or duplicated embedding indices")
            ordered = sorted(rows, key=lambda row: row["index"])
            vectors = np.array([row["embedding"] for row in ordered], dtype=np.float32)
            if vectors.shape != (len(inputs), self.dimension) or not np.all(np.isfinite(vectors)):
                raise ValueError("Invalid embedding dimensions or values")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("Zero embedding vector")
        except (ValueError, KeyError, TypeError):
            raise EmbeddingError("Embedding service returned malformed vectors.") from None
        return vectors / norms, int(truncated)
