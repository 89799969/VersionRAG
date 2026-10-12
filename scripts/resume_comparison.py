"""Resume failed calls without changing frozen algorithms or first-run artifacts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.comparison import ExactEvidenceIndex, TitlePlanner
from app.comparison_generation import Draft, bounded_answer, finish_draft, generate_comparison, validate_draft
from app.generation import Settings
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256
from scripts import evaluate_comparison as trial
from scripts.judge_comparison import PROMPT, Review

OUT = ROOT / "artifacts/reports/comparison_2026-10-07"
RUN = OUT / "resume_run"
QA = trial.QA
ARMS = (*trial.ARMS, "split_recover")
IMMUTABLE = [OUT / name for name in ("answers.jsonl", "dev_retrieval.jsonl",
             "evaluation_retrieval.jsonl", "semantic_reviews.jsonl", "generation_v4_policy.json")]
IMMUTABLE += [CORPUS, QA, ROOT / "app/comparison.py", ROOT / "app/comparison_generation.py",
              ROOT / "scripts/judge_comparison.py", ROOT / "scripts/evaluate_comparison.py",
              ROOT / "app/retrieval.py", ROOT / "app/dense.py", ROOT / "app/hybrid.py"]


def rows(name):
    path = OUT / name
    return read_jsonl(path) if path.exists() else []


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append(name, values):
    with (OUT / name).open("a", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def key(row):
    return row["id"], row["arm"]


def freeze():
    RUN.mkdir(exist_ok=True)
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in IMMUTABLE}
    path = OUT / "resume_policy.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8"))["immutable_sha256"] == hashes
    else:
        revision = json.loads((OUT / "generation_v4_policy.json").read_text(encoding="utf-8"))
        for name, expected in revision["fingerprints"].items():
            assert sha256(ROOT / name) == expected, name
        write("resume_policy.json", {"immutable_sha256": hashes,
              "policy": "Retry original failed generation calls once; keep input hits, model, prompt and budgets unchanged. Paired arms reuse the initial draft. No tuning on evaluation questions.",
              "judge_policy": "Original judge prompt/settings unchanged; reuse judgments only by full input hash.",
              "generation_temperature": "provider default, same as first run; stochastic retries are not algorithm gains",
              "orchestration_sha256": sha256(Path(__file__))})


def merge_answers(original, resumed):
    result = {key(r): copy.deepcopy(r) for r in original}
    for row in resumed:
        assert key(row) in result and result[key(row)]["status"] == "error", key(row)
        assert row["question"] == result[key(row)]["question"]
        assert row["split"] == result[key(row)]["split"]
        if row["status"] == "ok":
            result[key(row)] = copy.deepcopy(row)
    return [result[key(r)] for r in original]


def recover_from_initial(engine, plan, row, draft, generator):
    class Replay:
        def retrieve(self, p, **kwargs):
            if kwargs.get("supplemental"):
                return engine.retrieve(p, **kwargs)
            return copy.deepcopy(row["hits"]), copy.deepcopy(row["trace"])
    first = [True]
    def cached_first(p, hits):
        if first[0]:
            first[0] = False
            return copy.deepcopy(draft)
        return generator(p, hits)
    return bounded_answer(Replay(), plan, generator=cached_first)


def generate():
    original = rows("answers.jsonl")
    done = {key(r) for r in rows("answers_resume.jsonl")}
    failed = {key(r): r for r in original if r["status"] == "error"}
    inputs = rows("dev_retrieval.jsonl") + rows("evaluation_retrieval.jsonl")
    selected = [r for r in inputs if key(r) in failed and key(r) not in done]
    engine, transport = trial.clients()  # reuse original query cache
    trial.OUT = RUN  # separate resumed service events
    settings = Settings.load()
    samples = {r["id"]: r for r in read_jsonl(QA)}

    def work(row):
        plan = engine.planner.plan(row["question"])
        assert plan is not None
        start = time.perf_counter()
        old = failed[key(row)]
        result = {k: row[k] for k in ("id", "split", "arm", "question")}
        result["provenance"] = {"generation_attempt": 2, "source": "answers_resume.jsonl",
            "first_run_status": old["status"], "first_run_error": old["error"],
            "retrieval_hits_sha256": hashlib.sha256(json.dumps(row["hits"], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "policy": "unchanged generation_v4"}
        generator = lambda p, h: generate_comparison(p, h, settings=settings, transport=transport)
        try:
            draft = generator(plan, row["hits"])
            result.update({"status": "ok", **finish_draft(draft), "hits": row["hits"],
                           "elapsed_ms": (time.perf_counter() - start) * 1000})
        except Exception as exc:
            result.update({"status": "error", "error_type": type(exc).__name__,
                "error": str(exc) if type(exc).__module__.startswith("app.") else "Resume experiment error"})
            return [result] if row["arm"] != "split_balanced" else [result, {**result, "arm": "split_recover"}]
        outcomes = [result]
        if row["arm"] == "split_balanced" and (row["id"], "split_recover") not in done:
            # A recovery failure must not erase a successfully generated baseline.
            recovered = {**result, "arm": "split_recover"}
            try:
                final = recover_from_initial(engine, plan, row, draft, generator)
                recovered.update({**final, "elapsed_ms": (time.perf_counter() - start) * 1000,
                    **trial.score(samples[row["id"]], [{"id": p} for p in final["trace"]["final_ids"]], engine.exact)})
            except Exception as exc:
                recovered.update({"status": "error", "error_type": type(exc).__name__,
                    "error": str(exc) if type(exc).__module__.startswith("app.") else "Resume recovery error"})
            outcomes.append(recovered)
        return outcomes

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(work, row): row for row in selected}
        for future in as_completed(futures):
            outcomes = future.result()
            append("answers_resume.jsonl", outcomes)
            print("resume " + futures[future]["id"] + " " + futures[future]["arm"] + " " + str([r["status"] for r in outcomes]), flush=True)
    effective = merge_answers(original, rows("answers_resume.jsonl"))
    (OUT / "answers_effective.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in effective), encoding="utf-8")
    print(json.dumps(dict(Counter(r["status"] for r in effective))), flush=True)


def judge_payload(row, sample):
    flattened = [(a["entity"], c) for a in row.get("aspects", []) for c in a["claims"]]
    flattened += [("comparison", c) for c in row.get("comparisons", [])]
    payload = {"question": row["question"], "reference_answer": sample["reference_answer"],
        "answer": row.get("answer"), "claims": [{"index": i, "entity": e, **c} for i, (e, c) in enumerate(flattened)],
        "evidence": [{k: e[k] for k in ("passage_id", "title", "content")} for e in row.get("evidence", [])]}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return digest, payload


def judge():
    trial.OUT = RUN
    settings, transport = Settings.load(), trial.LoggedTransport()
    answers = rows("answers_effective.jsonl")
    samples = {r["id"]: r for r in read_jsonl(QA)}
    old = rows("semantic_reviews.jsonl") + rows("semantic_reviews_completed.jsonl")
    known = {r["input_sha256"]: r for r in old if r["status"] == "ok"}
    done = {key(r) for r in rows("semantic_reviews_completed.jsonl") if r["status"] in ("ok", "not_judged")}
    unique, aliases = {}, {}
    for row in answers:
        if key(row) in done:
            continue
        base = {k: row[k] for k in ("id", "arm", "split")}
        digest, payload = judge_payload(row, samples[row["id"]])
        if row["status"] != "ok":
            append("semantic_reviews_completed.jsonl", [{**base, "status": "not_judged", "reason": row["status"], "answer_success": False}])
        elif digest in known:
            append("semantic_reviews_completed.jsonl", [{**known[digest], **base, "reused_input_hash": True}])
        else:
            unique.setdefault(digest, payload)
            aliases.setdefault(digest, []).append(base)

    def work(item):
        digest, payload = item
        start = time.perf_counter()
        result = {"input_sha256": digest, "judge_model": settings.model,
            "judge_scope": "single same-provider model judgment; not human accuracy"}
        body = {"model": settings.model, "messages": [{"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "response_format": {"type": "json_object"}, "temperature": 0, "max_tokens": 2000,
            "thinking": {"type": "disabled"}, "stream": False}
        try:
            with httpx.Client(timeout=settings.timeout, transport=transport, trust_env=False) as client:
                response = client.post(settings.base_url + "/chat/completions", json=body,
                    headers={"Authorization": "Bearer " + settings.api_key})
                response.raise_for_status()
                raw = response.json()
            review = Review.model_validate_json(raw["choices"][0]["message"]["content"])
            if sorted(c.index for c in review.claims) != list(range(len(payload["claims"]))):
                raise ValueError("Missing or duplicated claim reviews")
            result.update({"status": "ok", **review.model_dump(), "usage": raw.get("usage", {}),
                "answer_success": bool(payload["claims"]) and review.facts_correct and review.request_complete
                    and all(c.supported for c in review.claims)})
        except Exception as exc:
            result.update({"status": "judge_error", "error_type": type(exc).__name__, "answer_success": False})
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000
        return digest, result

    print(f"judge unique new inputs: {len(unique)}", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(work, item) for item in unique.items()]
        for future in as_completed(futures):
            digest, result = future.result()
            append("semantic_reviews_completed.jsonl", [{**result, **base} for base in aliases[digest]])
            print("judge " + aliases[digest][0]["id"] + " " + aliases[digest][0]["arm"] + " " + result["status"], flush=True)


def audit():
    freeze()
    originals, resumed, answers = rows("answers.jsonl"), rows("answers_resume.jsonl"), rows("answers_effective.jsonl")
    assert answers == merge_answers(originals, resumed)
    assert len(answers) == len({key(r) for r in answers}) == 180
    assert len(resumed) == len({key(r) for r in resumed}) == 41
    documents = read_jsonl(CORPUS)
    exact, planner = ExactEvidenceIndex(documents), TitlePlanner(documents)
    samples = {r["id"]: r for r in read_jsonl(QA)}
    review_map = {key(r): r for r in rows("semantic_reviews_completed.jsonl")}
    assert len(review_map) == 180
    for row in answers:
        review = review_map[key(row)]
        if row["status"] == "ok":
            plan = planner.plan(row["question"])
            validate_draft(Draft.model_validate({"aspects": row["aspects"], "comparisons": row["comparisons"]}), plan, row["evidence"])
            for evidence in row["evidence"]:
                assert evidence["content"] == exact.by_id[evidence["passage_id"]]["contents"][:len(evidence["content"])]
            assert sum(len(e["content"]) + len(f"\n[片段 {e['passage_id']}] 标题：{e['title']}\n") for e in row["evidence"]) <= 16000
            digest, payload = judge_payload(row, samples[row["id"]])
            assert review["status"] == "ok" and review["input_sha256"] == digest
            checked = Review.model_validate({k: review[k] for k in Review.model_fields})
            assert sorted(c.index for c in checked.claims) == list(range(len(payload["claims"])))
            assert review["answer_success"] == (bool(payload["claims"]) and checked.facts_correct and checked.request_complete and all(c.supported for c in checked.claims))
            if row["arm"] == "split_recover":
                assert len(row["trace"]["generation_attempts"]) <= 2
                assert len(row["trace"].get("supplemental", {}).get("steps", [])) <= 2
        else:
            assert row["status"] in ("not_generated", "error") and review["status"] == "not_judged"
    summaries = []
    for split in sorted({r["split"] for r in answers}):
        for arm in ARMS:
            selected = [r for r in answers if r["split"] == split and r["arm"] == arm]
            eligible = [r for r in selected if planner.plan(r["question"]) is not None]
            valid = [r for r in eligible if r["status"] == "ok"]
            judged = [review_map[key(r)] for r in valid]
            item = {"split": split, "arm": arm, "input_questions": len(selected), "eligible_questions": len(eligible),
                "generated": len(valid), "generation_errors": len(eligible) - len(valid),
                "model_judged_success_count": sum(r["answer_success"] for r in judged),
                "model_success_and_no_partial_warning": sum(review_map[key(r)]["answer_success"] and not r["insufficient_evidence"] for r in valid),
                "partial_count": sum(r["insufficient_evidence"] for r in valid),
                "unsupported_claims": sum(not c["supported"] for r in judged for c in r["claims"]),
                "claims": sum(len(r["claims"]) for r in judged),
                "rejected_claims": sum(len(r.get("rejected_claims", [])) for r in valid),
                "first_run_errors": sum(r["status"] == "error" for r in originals if r["split"] == split and r["arm"] == arm),
                "median_generation_recovery_ms": statistics.median(r["elapsed_ms"] for r in valid),
                "mean_prompt_tokens": statistics.mean((r.get("total_usage") or r["usage"]).get("prompt_tokens", 0) for r in valid)}
            if arm == "split_recover":
                item.update({"recovery_triggered": sum(r["trace"]["recovery_triggered"] for r in valid),
                    "context_changed": sum(r["trace"].get("recovery_context_changed", False) for r in valid)})
                base = {r["id"]: r for r in answers if r["split"] == split and r["arm"] == "split_balanced"}
                item["semantic_gains"] = sum(review_map[key(r)]["answer_success"] and not review_map[key(base[r["id"]])]["answer_success"] for r in valid)
                item["semantic_regressions"] = sum(not review_map[key(r)]["answer_success"] and review_map[key(base[r["id"]])]["answer_success"] for r in valid)
            summaries.append(item)
    write("completed_summary.json", summaries)
    write("completed_audit.json", {"status": "passed", "answer_records": len(answers),
        "answer_statuses": dict(Counter(r["status"] for r in answers)), "resumed_records": len(resumed),
        "original_statuses": dict(Counter(r["status"] for r in originals)), "semantic_reviews": len(review_map),
        "checks": ["First-run files and frozen code/data unchanged", "Retakes only replace failed calls in a separate view",
            "Quote anchors and exact supplied corpus prefixes", "Judge input hashes and success definition recomputed", "Bounded recovery"],
        "scope": "Single same-provider model evaluation, eligible two-profession subset; not human accuracy or unbiased production latency",
        "orchestration_sha256_at_audit": sha256(Path(__file__)),
        "orchestration_revision": "Audit now accepts explicitly retained generation failures as scored failures; added separately named conservative no-partial-warning diagnostic. Measured generation/retrieval/judge code unchanged.",
        "effective_answers_sha256": sha256(OUT / "answers_effective.jsonl")})
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["generate", "judge", "audit", "all"], required=True)
    args = parser.parse_args()
    freeze()
    for name, action in (("generate", generate), ("judge", judge), ("audit", audit)):
        if args.phase in (name, "all"):
            action()
