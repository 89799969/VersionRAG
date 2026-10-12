"""Reproducible comparison retrieval, exact-equivalent scoring, and quote drafts.

No label text/IDs reach the planner or retrieval. Original mapped labels remain
untouched. Generation uses the same prompt and character budget in all arms.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.comparison import ComparisonRetriever, ExactEvidenceIndex
from app.comparison_generation import bounded_answer, finish_draft, generate_comparison
from app.dense import DenseRetriever
from app.embedding_api import APIEncoder
from app.evaluation import evidence_metrics
from app.generation import Settings
from app.generation import GenerationError
from app.hybrid import HybridRetriever
from app.reranking import APIReranker
from app.retrieval import CORPUS, ROOT, ChineseBM25, read_jsonl, sha256

OUT = ROOT / "artifacts/reports/comparison_2026-10-07"
QA = ROOT / "data/extended_qa_mapped.jsonl"
ARMS = ("original", "scope_dedup", "split_balanced")
FILES = [CORPUS, QA, ROOT / "app/comparison.py", ROOT / "app/comparison_generation.py",
         ROOT / "app/retrieval.py", ROOT / "app/dense.py", ROOT / "app/hybrid.py",
         ROOT / "scripts/evaluate_comparison.py"]


def save(name, payload):
    path = OUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprints():
    return {str(p.relative_to(ROOT)): sha256(p) for p in FILES}


class LoggedTransport(httpx.BaseTransport):
    """Save safe request metadata only; pace SiliconFlow request starts globally."""
    def __init__(self):
        self.inner = httpx.HTTPTransport()
        self.lock, self.last = threading.Lock(), 0.
        self.billing_unavailable = False

    def handle_request(self, request):
        with self.lock:
            if request.url.host == "api.deepseek.com" and self.billing_unavailable:
                raise GenerationError("Model balance unavailable (HTTP 402); remaining model requests skipped.")
        wait = 0.
        if request.url.host == "api.siliconflow.cn":
            with self.lock:
                wait = max(0., 1.5 - (time.monotonic() - self.last))
                if wait:
                    time.sleep(wait)
                self.last = time.monotonic()
        started = time.perf_counter()
        try:
            response = self.inner.handle_request(request)
            status = response.status_code
            if request.url.host == "api.deepseek.com" and status == 402:
                with self.lock:
                    self.billing_unavailable = True
            return response
        finally:
            event = {"host": request.url.host, "path": request.url.path,
                     "status": locals().get("status"), "pacing_ms": wait * 1000,
                     "headers_ms": (time.perf_counter() - started) * 1000}
            with self.lock:
                with (OUT / "service_events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")

    def close(self):
        # httpx clients close their supplied transport. Keep this shared pool open
        # until the experiment process exits; no credentials are serialized.
        pass


class CachedEncoder:
    def __init__(self, transport):
        self.base = APIEncoder(transport=transport)
        self.dimension = self.base.dimension
        self.lock = threading.Lock()
        self.cache = OUT / "query_vectors"
        self.cache.mkdir(exist_ok=True)

    def metadata(self):
        return self.base.metadata()

    def encode(self, texts, query=False):
        rows = []
        for text in texts:
            key = hashlib.sha256(json.dumps([self.metadata(), text, query], ensure_ascii=False,
                                           sort_keys=True).encode()).hexdigest()
            path = self.cache / (key + ".npy")
            with self.lock:
                if path.exists():
                    vector = np.load(path)
                else:
                    vector, _ = self.base.encode([text], query=query)
                    np.save(path, vector)
                    path.with_suffix(".json").write_text(json.dumps({"query": text, "query_flag": query,
                        "encoder": self.metadata(), "sha256": sha256(path)}, ensure_ascii=False), encoding="utf-8")
                meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if sha256(path) != meta["sha256"] or vector.shape != (1, self.dimension):
                    raise ValueError("Query cache checksum/shape mismatch")
                rows.append(vector[0])
        return np.array(rows, dtype=np.float32), 0


def clients():
    transport = LoggedTransport()
    keyword = ChineseBM25()
    keyword.load()
    dense = DenseRetriever(encoder=CachedEncoder(transport))
    dense.load()
    hybrid = HybridRetriever(keyword, dense)
    ranker = APIReranker(transport=transport)
    return ComparisonRetriever(hybrid, ranker, keyword.documents), transport


def score(sample, hits, exact):
    ids = [str(h["id"]) for h in hits]
    revised = copy.deepcopy(sample)
    for group in revised["required_evidence_groups"]:
        group["acceptable_corpus_passage_ids"] = sorted(exact.equivalents(group["acceptable_corpus_passage_ids"]))
    return {"original_labels": evidence_metrics(sample, ids, 5),
            "exact_equivalents": evidence_metrics(revised, ids, 5)}


def summary(rows):
    summaries = []
    for split in sorted({r["split"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows}):
            selected = [r for r in rows if r["split"] == split and r["arm"] == arm]
            successful = [r for r in selected if r["status"] == "ok"]
            item = {"split": split, "arm": arm, "questions": len(selected),
                    "successful": len(successful), "errors": len(selected) - len(successful)}
            for metric in ("original_labels", "exact_equivalents"):
                # Operational failures remain in the denominator and score zero.
                item[metric] = {"all_evidence_hit_count": sum(r[metric]["all_evidence_hit"] for r in successful),
                    "all_evidence_hit_rate": sum(r[metric]["all_evidence_hit"] for r in successful) / len(selected),
                    "evidence_group_recall": sum(r[metric]["evidence_group_recall"] for r in successful) / len(selected),
                    "mrr_at_5": sum(r[metric]["reciprocal_rank"] for r in successful) / len(selected)}
            if successful:
                item["median_pipeline_ms"] = statistics.median(r["elapsed_ms"] for r in successful)
            summaries.append(item)
    return summaries


def retrieval(phase):
    current = fingerprints()
    if phase == "dev":
        if (OUT / "dev_retrieval.jsonl").exists():
            raise RuntimeError("Development records already exist; use --output with a fresh report directory")
        frozen = {"fingerprints": current, "arms": list(ARMS), "candidate_k": 30, "top_k": 5,
            "rrf_constant": 60, "model": APIReranker.model, "context_chars": 16000,
            "rounds": 1, "selection": "No parameter scan; validate fixed scoped two-entity policy on dev first",
            "scope": "Original split preserved. Diagnostic/source overlaps reported separately. Not an untouched new benchmark."}
        save("frozen_policy.json", frozen)
    else:
        old = json.loads((OUT / "frozen_policy.json").read_text(encoding="utf-8"))["fingerprints"]
        revision_path = OUT / "generation_v4_policy.json"
        approved = json.loads(revision_path.read_text(encoding="utf-8"))["fingerprints"] if revision_path.exists() else {}
        changed = {key: value for key, value in current.items() if old.get(key) != value}
        if any(key.replace("\\", "/") not in {"app/comparison_generation.py", "scripts/evaluate_comparison.py"}
               or approved.get(key.replace("\\", "/")) != value for key, value in changed.items()):
            raise RuntimeError("Retrieval code/data changed after development; use a fresh experiment directory")
    engine, _ = clients()
    samples = [r for r in read_jsonl(QA) if r["task"] == "multi_doc" and r["mapping_complete"]
               and ((r["split"] == "dev") if phase == "dev" else r["split"] != "dev")]
    output = OUT / (phase + "_retrieval.jsonl")
    rows = read_jsonl(output) if output.exists() else []
    done = {(r["id"], r["arm"]) for r in rows}
    for sample in samples:
        plan = engine.planner.plan(sample["question"])
        for arm in ARMS:
            if (sample["id"], arm) in done:
                continue
            started = time.perf_counter()
            row = {"id": sample["id"], "split": sample["split"], "arm": arm,
                   "question": sample["question"], "plan": plan.payload() if plan else None}
            try:
                if arm == "original" or plan is None:
                    candidates = engine.hybrid.search(sample["question"], 30)
                    hits = engine.reranker.rerank(sample["question"], candidates, 5)
                    trace = {"policy": "original" if arm == "original" else "standard-fallback",
                             "candidate_ids": [str(h["id"]) for h in candidates]}
                else:
                    hits, trace = engine.retrieve(plan, split=arm == "split_balanced")
                row.update({"status": "ok", "hits": hits, "trace": trace, **score(sample, hits, engine.exact)})
            except Exception as exc:
                # Provider wrappers contain safe messages; do not persist raw httpx requests.
                row.update({"status": "error", "error_type": type(exc).__name__,
                            "error": str(exc) if type(exc).__module__.startswith("app.") else "Experiment error"})
            row["elapsed_ms"] = (time.perf_counter() - started) * 1000
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(f"{phase} {sample['id']} {arm} {row['status']} {len(rows)}/{len(samples)*3}", flush=True)
    save(phase + "_summary.json", summary(rows))
    print(json.dumps(summary(rows), ensure_ascii=False, indent=2), flush=True)


def answers(cohort):
    engine, transport = clients()
    inputs = read_jsonl(OUT / "dev_retrieval.jsonl")
    if cohort == "all":
        inputs += read_jsonl(OUT / "evaluation_retrieval.jsonl")
    samples = {r["id"]: r for r in read_jsonl(QA) if r["task"] == "multi_doc"}
    output = OUT / "answers.jsonl"
    done = {(r["id"], r["arm"]) for r in read_jsonl(output)} if output.exists() else set()
    settings = Settings.load()

    def work(row):
        result = {"id": row["id"], "split": row["split"], "arm": row["arm"], "question": row["question"]}
        plan = engine.planner.plan(row["question"])
        if row["status"] != "ok" or plan is None:
            return [{**result, "status": "not_generated", "reason": "retrieval failed or rule planner declined"}]
        started = time.perf_counter()
        try:
            generate_fn = lambda p, h: generate_comparison(p, h, settings=settings, transport=transport)
            draft = generate_fn(plan, row["hits"])
            result.update({"status": "ok", **finish_draft(draft), "hits": row["hits"],
                           "elapsed_ms": (time.perf_counter() - started) * 1000})
            outcomes = [result]
            if row["arm"] == "split_balanced":
                class InitialReplay:
                    def retrieve(self, p, **kwargs):
                        if kwargs.get("supplemental"):
                            return engine.retrieve(p, **kwargs)
                        return copy.deepcopy(row["hits"]), copy.deepcopy(row["trace"])
                first = [True]
                def cached_first(p, h):
                    if first[0]:
                        first[0] = False
                        return copy.deepcopy(draft)
                    return generate_fn(p, h)
                recovered = bounded_answer(InitialReplay(), plan, generator=cached_first)
                final_hits = recovered["trace"]["final_ids"]
                # All equivalence scoring uses final IDs, not the first attempt's IDs.
                outcomes.append({**result, "arm": "split_recover", **recovered,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    **score(samples[row["id"]], [{"id": pid} for pid in final_hits], engine.exact)})
            return outcomes
        except Exception as exc:
            safe = str(exc) if type(exc).__module__.startswith("app.") else "Experiment error"
            error = {**result, "status": "error", "error_type": type(exc).__name__, "error": safe,
                     "invalid_draft": getattr(exc, "draft", None), "evidence": getattr(exc, "evidence", [])}
            return [error] if row["arm"] != "split_balanced" else [error, {**error, "arm": "split_recover"}]

    selected = [r for r in inputs if (r["id"], r["arm"]) not in done]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(work, row): row for row in selected}
        for future in as_completed(pending):
            results = future.result()
            with output.open("a", encoding="utf-8") as stream:
                for result in results:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            print("answers " + pending[future]["id"] + " " + pending[future]["arm"] + " " + results[0]["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["dev", "evaluation", "answers"], required=True)
    parser.add_argument("--cohort", choices=["dev", "all"], default="dev")
    parser.add_argument("--output", type=Path, default=OUT,
                        help="Fresh report directory under artifacts/reports for a new full run")
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to((ROOT / "artifacts/reports").resolve()):
        raise ValueError("Experiment outputs must stay under artifacts/reports")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "answers":
        answers(args.cohort)
    else:
        retrieval(args.phase)
