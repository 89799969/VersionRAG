"""Persistent, single-process knowledge lifecycle; immutable dual-index versions.

SQLite owns draft revisions, job snapshots and the active pointer. The worker is
one thread, not a distributed queue. Embeddings are incremental; indexes are
complete snapshots. Existing benchmark modules and data are left unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.dense import DenseRetriever
from app.hybrid import HybridRetriever
from app.retrieval import ChineseBM25, read_jsonl, sha256


class KnowledgeError(Exception):
    pass


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def packed(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def split_document(document_id: str, title: str, text: str, url: str = "") -> list[dict]:
    """Bounded character chunks, preferring paragraph/sentence endings; 100 overlap.

    Offsets refer to the original supplied text. This is a text ingestion policy,
    not PDF/OCR parsing, nor a re-chunking of the frozen DomainRAG benchmark.
    """
    chunks, start = [], 0
    while start < len(text):
        end = min(start + 900, len(text))
        if end < len(text):
            boundaries = [text.rfind(mark, start + 450, end) for mark in ("\n", "。", "！", "？")]
            cut = max(boundaries)
            if cut >= start + 450:
                end = cut + 1
        contents = text[start:end]
        if contents.strip():
            content_hash = digest(contents)
            chunks.append({"id": f"doc-{digest(document_id)[:16]}-{start}-{content_hash[:16]}",
                           "title": title, "contents": contents,
                           "metadata": {"source_id": document_id, "document_id": document_id,
                                        "url": url, "content_sha256": content_hash,
                                        "start": start, "end": end,
                                        "chunk_policy": "chars900-overlap100-boundaries-v1"}})
        if end == len(text):
            break
        start = end - 100
    return chunks


class EmptyRetriever:
    def search(self, question, top_k=5, source_id=None):
        return []


@dataclass(frozen=True)
class Snapshot:
    version: str
    engines: dict


class KnowledgeService:
    def __init__(self, root: Path, encoder=None, max_pending: int = 4):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "knowledge.sqlite3"
        self.encoder = encoder
        self.max_pending = max_pending
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = None
        self._guard = None
        self._cache_lock = threading.RLock()
        self._snapshots = {}
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO settings VALUES ('revision', '0');
                INSERT OR IGNORE INTO settings VALUES ('active_epoch', '0');
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, version TEXT NOT NULL UNIQUE, revision INTEGER NOT NULL,
                    base_epoch INTEGER NOT NULL, state TEXT NOT NULL, auto_publish INTEGER NOT NULL,
                    published INTEGER NOT NULL DEFAULT 0, error TEXT, note TEXT,
                    stats TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT);
                CREATE TABLE IF NOT EXISTS job_chunks (
                    job_id TEXT NOT NULL, ordinal INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (job_id, ordinal));
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY, revision INTEGER NOT NULL, ready_sha256 TEXT NOT NULL,
                    stats TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    key TEXT PRIMARY KEY, vector BLOB NOT NULL, vector_sha256 TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def setting(db, key):
        row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def set_setting(db, key, value):
        db.execute("INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, str(value)))

    def get_encoder(self):
        if self.encoder is None:
            from app.embedding_api import APIEncoder
            self.encoder = APIEncoder()
        return self.encoder

    def upsert(self, document_id, title, text, url=""):
        if not document_id.strip() or not title.strip() or not text.strip():
            raise KnowledgeError("Document ID, title and text must not be blank.")
        chunks = split_document(document_id, title, text, url)
        text_hash = digest(text)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            if old and (old["title"], old["url"], old["text_sha256"]) == (title, url, text_hash):
                return {"changed": False, "draft_revision": int(self.setting(db, "revision"))}
            db.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?)",
                       (document_id, title, url, text_hash))
            db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            db.executemany("INSERT INTO chunks VALUES (?,?,?)",
                           [(c["id"], document_id, packed(c)) for c in chunks])
            revision = int(self.setting(db, "revision")) + 1
            self.set_setting(db, "revision", revision)
        return {"changed": True, "draft_revision": revision, "chunks": len(chunks)}

    def delete(self, document_id):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount > 0
            db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            revision = int(self.setting(db, "revision")) + int(changed)
            self.set_setting(db, "revision", revision)
        return {"changed": changed, "draft_revision": revision}

    def documents(self, limit=50, offset=0):
        with self.connection() as db:
            rows = db.execute("SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.document_id=d.id) AS chunks "
                              "FROM documents d ORDER BY id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            total = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return {"total": total, "items": [dict(row) for row in rows]}

    def submit(self, auto_publish=True):
        job_id = uuid4().hex
        version = f"v-{job_id}"
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            pending = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
            if pending >= self.max_pending:
                raise KnowledgeError("Build queue is full; wait for an existing job.")
            db.execute("INSERT INTO jobs(id,version,revision,base_epoch,state,auto_publish) VALUES (?,?,?,?,?,?)",
                       (job_id, version, int(self.setting(db, "revision")),
                        int(self.setting(db, "active_epoch")), "queued", int(auto_publish)))
            # Same transaction as revision capture: later edits cannot mutate a queued job.
            db.execute("INSERT INTO job_chunks SELECT ?, ROW_NUMBER() OVER (ORDER BY id), payload FROM chunks",
                       (job_id,))
        self._wake.set()
        return self.job(job_id)

    def job(self, job_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KnowledgeError("Job does not exist.")
        result = dict(row)
        result["stats"] = json.loads(result["stats"])
        result["published"] = bool(result["published"])
        return result

    def status(self):
        with self.connection() as db:
            return {"active_version": self.setting(db, "active"),
                    "draft_revision": int(self.setting(db, "revision")),
                    "pending_jobs": db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0],
                    "worker_alive": bool(self._thread and self._thread.is_alive())}

    def versions(self):
        with self.connection() as db:
            active = self.setting(db, "active")
            rows = db.execute("SELECT * FROM versions ORDER BY created_at DESC, rowid DESC").fetchall()
        return [{**dict(row), "stats": json.loads(row["stats"]), "active": row["id"] == active} for row in rows]

    def directory(self, version):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}", version):
            raise KnowledgeError("Invalid version ID.")
        return self.root / "versions" / version

    def _vector_key(self, text):
        return digest(packed(self.get_encoder().metadata()) + "\n" + text)

    def _cache_vectors(self, values):
        with self.connection() as db:
            for key, vector in values:
                raw = np.asarray(vector, dtype="<f4").tobytes()
                db.execute("INSERT OR REPLACE INTO embedding_cache VALUES (?,?,?)",
                           (key, raw, hashlib.sha256(raw).hexdigest()))

    def _vectors(self, documents):
        encoder = self.get_encoder()
        vectors = np.empty((len(documents), encoder.dimension), dtype=np.float32)
        missing, cache_hits, truncated = {}, 0, 0
        with self.connection() as db:
            for index, doc in enumerate(documents):
                key = self._vector_key(doc["contents"])
                row = db.execute("SELECT * FROM embedding_cache WHERE key=?", (key,)).fetchone()
                vector = (None if row is None or len(row["vector"]) != encoder.dimension * 4
                          else np.frombuffer(row["vector"], dtype="<f4"))
                valid = (vector is not None and hashlib.sha256(row["vector"]).hexdigest() == row["vector_sha256"]
                         and vector.shape == (encoder.dimension,) and np.all(np.isfinite(vector))
                         and np.isclose(np.linalg.norm(vector), 1, atol=1e-4))
                if valid:
                    vectors[index] = vector
                    cache_hits += 1
                else:
                    missing.setdefault(key, {"text": doc["contents"], "rows": []})["rows"].append(index)
        items = list(missing.items())
        for begin in range(0, len(items), 16):
            batch = items[begin:begin + 16]
            encoded, count = encoder.encode([item[1]["text"] for item in batch])
            if (encoded.shape != (len(batch), encoder.dimension) or not np.all(np.isfinite(encoded))
                    or not np.allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-4)):
                raise KnowledgeError("Encoder returned invalid or unnormalized vectors.")
            truncated += count
            self._cache_vectors([(key, encoded[i]) for i, (key, _) in enumerate(batch)])
            for i, (_, item) in enumerate(batch):
                vectors[item["rows"]] = encoded[i]
        return vectors, {"total_chunks": len(documents), "cache_hit_chunks": cache_hits,
                         "encoded_unique_texts": len(items), "truncated_new_texts": truncated}

    def _seal(self, version, revision, stats):
        directory = self.directory(version)
        files = {str(path.relative_to(directory)).replace("\\", "/"): sha256(path)
                 for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "ready.json"}
        ready = {"version": version, "revision": revision, "files": files, "stats": stats,
                 "encoder": self.get_encoder().metadata()}
        (directory / "ready.json").write_text(packed(ready), encoding="utf-8")
        return self._load(version), sha256(directory / "ready.json")

    def _load(self, version, ready_hash=None):
        directory = self.directory(version)
        ready_path = directory / "ready.json"
        if ready_hash is not None and sha256(ready_path) != ready_hash:
            raise KnowledgeError("Version manifest checksum mismatch.")
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready["version"] != version or ready["encoder"] != self.get_encoder().metadata():
            raise KnowledgeError("Version identity or embedding configuration mismatch.")
        actual = {str(p.relative_to(directory)).replace("\\", "/"): sha256(p)
                  for p in directory.rglob("*") if p.is_file() and p.name != "ready.json"}
        if actual != ready["files"]:
            raise KnowledgeError("Version files changed or the build is incomplete.")
        corpus = directory / "corpus.jsonl"
        bm25 = ChineseBM25(corpus, directory / "bm25")
        dense = DenseRetriever(corpus, directory / "dense", self.get_encoder())
        if len(bm25.documents) != ready["stats"]["total_chunks"]:
            raise KnowledgeError("Version document count mismatch.")
        dense.load()
        if not bm25.documents:
            empty = EmptyRetriever()
            return Snapshot(version, {name: empty for name in ("bm25", "dense", "hybrid", "hybrid_rerank")})
        bm25.load()
        hybrid = HybridRetriever(bm25, dense)
        from app.reranking import RerankRetriever
        return Snapshot(version, {"bm25": bm25, "dense": dense, "hybrid": hybrid,
                                  "hybrid_rerank": RerankRetriever(hybrid)})

    def pin(self):
        with self.connection() as db:
            version = self.setting(db, "active")  # read once per request
            row = db.execute("SELECT ready_sha256 FROM versions WHERE id=?", (version,)).fetchone()
        if version is None:
            return None
        if row is None:
            raise KnowledgeError("Active version is missing.")
        with self._cache_lock:
            if version not in self._snapshots:
                snapshot = self._load(version, row[0])
                self._snapshots = {version: snapshot}  # requests retain their own previous reference
            return self._snapshots[version]

    def activate(self, version):
        with self.connection() as db:
            row = db.execute("SELECT ready_sha256 FROM versions WHERE id=?", (version,)).fetchone()
        if row is None:
            raise KnowledgeError("Version does not exist or is not ready.")
        snapshot = self._load(version, row[0])  # even a cached rollback must pass validation
        with self._cache_lock, self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self.set_setting(db, "active", version)
            self.set_setting(db, "active_epoch", int(self.setting(db, "active_epoch")) + 1)
            self._snapshots = {version: snapshot}
        return {"active_version": version, "draft_unchanged": True}

    def _build(self, job):
        version = job["version"]
        directory = self.directory(version)
        directory.mkdir(parents=True, exist_ok=False)
        with self.connection() as db:
            documents = [json.loads(r[0]) for r in db.execute(
                "SELECT payload FROM job_chunks WHERE job_id=? ORDER BY ordinal", (job["id"],))]
        corpus = directory / "corpus.jsonl"
        corpus.write_text("".join(packed(doc) + "\n" for doc in documents), encoding="utf-8")
        vectors, stats = self._vectors(documents)
        dense_path = directory / "dense"
        dense_path.mkdir()
        np.save(dense_path / "vectors.npy", vectors)
        dense_manifest = {"corpus_sha256": sha256(corpus), "documents": len(documents),
                          "encoder": self.get_encoder().metadata(),
                          "vectors_sha256": sha256(dense_path / "vectors.npy"),
                          "search": "exact cosine over normalized float32 vectors"}
        (dense_path / "manifest.json").write_text(packed(dense_manifest), encoding="utf-8")
        if documents:
            ChineseBM25(corpus, directory / "bm25").build()
        snapshot, ready_hash = self._seal(version, job["revision"], stats)
        # New edits or a manual rollback while building must not be overwritten.
        with self._cache_lock, self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            unchanged = (int(self.setting(db, "revision")) == job["revision"] and
                         int(self.setting(db, "active_epoch")) == job["base_epoch"])
            publish = bool(job["auto_publish"] and unchanged)
            db.execute("INSERT INTO versions(id,revision,ready_sha256,stats) VALUES (?,?,?,?)",
                       (version, job["revision"], ready_hash, packed(stats)))
            if publish:
                self.set_setting(db, "active", version)
                self.set_setting(db, "active_epoch", job["base_epoch"] + 1)
                self._snapshots = {version: snapshot}
            note = None if publish else "ready; explicit activation required (manual mode or draft/active changed)"
            db.execute("UPDATE jobs SET state='succeeded',published=?,stats=?,note=?,finished_at=CURRENT_TIMESTAMP WHERE id=?",
                       (int(publish), packed(stats), note, job["id"]))
            db.execute("DELETE FROM job_chunks WHERE job_id=?", (job["id"],))

    def run_one(self):
        """Worker unit: atomically claim one immutable queued snapshot."""
        if self._thread and threading.current_thread() is not self._thread:
            raise KnowledgeError("The background worker owns job execution.")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                return False
            job = dict(row)
            db.execute("UPDATE jobs SET state='running' WHERE id=?", (job["id"],))
        try:
            self._build(job)
        except Exception as exc:
            # Only our sanitized provider/validation errors are safe for public task status.
            from app.embedding_api import EmbeddingError
            message = str(exc) if isinstance(exc, (EmbeddingError, KnowledgeError)) else "Index build failed; active version was kept."
            with self.connection() as db:
                db.execute("UPDATE jobs SET state='failed',error=?,finished_at=CURRENT_TIMESTAMP WHERE id=?",
                           (message, job["id"]))
                db.execute("DELETE FROM job_chunks WHERE job_id=?", (job["id"],))
        return True

    def _worker(self):
        while not self._stop.is_set():
            if not self.run_one():
                self._wake.wait(1)
                self._wake.clear()

    def _acquire_guard(self):
        guard = (self.root / "writer.lock").open("a+b")
        if guard.seek(0, os.SEEK_END) == 0:
            guard.write(b"0")
            guard.flush()
        guard.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            guard.close()
            raise KnowledgeError("Knowledge service already has a writer; use one server process.") from None
        return guard

    def start(self):
        if self._thread is not None:
            raise KnowledgeError("Worker already started.")
        guard = self._acquire_guard()
        self._guard = guard
        try:
            with self.connection() as db:
                # A killed process has no live owner. Do not silently replay its running job.
                db.execute("UPDATE jobs SET state='failed',error='Build interrupted by service restart; resubmit.', "
                           "finished_at=CURRENT_TIMESTAMP WHERE state='running'")
                db.execute("DELETE FROM job_chunks WHERE job_id IN (SELECT id FROM jobs WHERE state='failed')")
            self.pin()  # validate active bundle before serving queries
            self._stop.clear()
            self._thread = threading.Thread(target=self._worker, name="knowledge-builder", daemon=True)
            self._thread.start()
        except Exception:
            guard.close()
            self._guard = None
            raise

    def close(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join()  # retain writer ownership until the current bounded API call ends
            self._thread = None
        if self._guard:
            self._guard.close()
            self._guard = None

    def seed(self, corpus: Path, bm25_path: Path, dense_path: Path):
        """Offline first import: copy validated frozen artifacts, no re-embedding."""
        if self._thread:
            raise KnowledgeError("Seed before starting the service.")
        guard = self._acquire_guard()
        try:
            return self._seed(corpus, bm25_path, dense_path)
        finally:
            guard.close()

    def _seed(self, corpus, bm25_path, dense_path):
        with self.connection() as db:
            if db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] or db.execute("SELECT COUNT(*) FROM versions").fetchone()[0]:
                raise KnowledgeError("Seed requires an empty knowledge store.")
        version = f"seed-{uuid4().hex}"
        directory = self.directory(version)
        directory.mkdir(parents=True)
        shutil.copyfile(corpus, directory / "corpus.jsonl")
        shutil.copytree(bm25_path, directory / "bm25")
        shutil.copytree(dense_path, directory / "dense")
        docs = read_jsonl(corpus)
        stats = {"total_chunks": len(docs), "cache_hit_chunks": len(docs), "encoded_unique_texts": 0,
                 "truncated_new_texts": 0, "seed": True}
        snapshot, ready_hash = self._seal(version, 1, stats)
        dense = snapshot.engines["dense"]
        self._cache_vectors([(self._vector_key(doc["contents"]), dense.vectors[i]) for i, doc in enumerate(docs)])
        grouped = {}
        for doc in docs:
            grouped.setdefault(str(doc["metadata"]["source_id"]), []).append(doc)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.executemany("INSERT INTO documents VALUES (?,?,?,?)", [
                (sid, chunks[0]["title"], chunks[0]["metadata"].get("url", ""),
                 digest("\n".join(c["contents"] for c in chunks))) for sid, chunks in grouped.items()])
            db.executemany("INSERT INTO chunks VALUES (?,?,?)", [
                (str(doc["id"]), str(doc["metadata"]["source_id"]), packed(doc)) for doc in docs])
            db.execute("INSERT INTO versions(id,revision,ready_sha256,stats) VALUES (?,?,?,?)",
                       (version, 1, ready_hash, packed(stats)))
            self.set_setting(db, "revision", 1)
            self.set_setting(db, "active", version)
            self.set_setting(db, "active_epoch", 1)
        self._snapshots = {version: snapshot}
        return {"active_version": version, "source_records": len(grouped), **stats}
