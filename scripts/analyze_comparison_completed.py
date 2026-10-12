"""Read-only paired analysis of completed calls and unmodified judge verdicts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import read_jsonl
from scripts.resume_comparison import ARMS, OUT, QA, key, rows, write


def main():
    answers = {key(r): r for r in rows("answers_effective.jsonl")}
    reviews = {key(r): r for r in rows("semantic_reviews_completed.jsonl")}
    retrievals = {key(r): r for r in rows("dev_retrieval.jsonl") + rows("evaluation_retrieval.jsonl")}
    cases, paired, recovery = [], [], []
    for sample in read_jsonl(QA):
        pid = sample["id"]
        if (pid, "split_balanced") not in answers or answers[pid, "split_balanced"]["status"] == "not_generated":
            continue
        item = {"id": pid, "split": sample["split"], "question": sample["question"],
                "reference_answer": sample["reference_answer"], "arms": {}}
        for arm in ARMS:
            a, j = answers[pid, arm], reviews[pid, arm]
            t = retrievals[pid, "split_balanced" if arm == "split_recover" else arm]
            item["arms"][arm] = {"generation_status": a["status"], "generation_error": a.get("error"),
                "model_judged_success": j["answer_success"], "partial_warning": a.get("insufficient_evidence"),
                "model_success_and_no_partial_warning": j["answer_success"] and not a.get("insufficient_evidence", True),
                "answer": a.get("answer"), "first_ids": [str(h["id"]) for h in t["hits"]],
                "first_titles": [h["title"] for h in t["hits"]],
                "final_ids": a.get("trace", {}).get("final_ids", [str(h["id"]) for h in t["hits"]]),
                "rejected_claims": a.get("rejected_claims", []),
                "review_reason": j.get("reason"), "missing_points": j.get("missing_points", []),
                "unsupported_claims": [c for c in j.get("claims", []) if not c["supported"]],
                "actual_evidence_truncated": [e["passage_id"] for e in a.get("evidence", []) if e["truncated"]],
                "provenance": a.get("provenance", {"source": "answers.jsonl", "generation_attempt": 1})}
        cases.append(item)
        a, b = item["arms"]["original"], item["arms"]["split_balanced"]
        paired.append({"id": pid, "split": sample["split"], "baseline_success": a["model_judged_success"],
                       "split_success": b["model_judged_success"],
                       "transition": "gain" if b["model_judged_success"] and not a["model_judged_success"] else
                           "regression" if a["model_judged_success"] and not b["model_judged_success"] else "unchanged"})
        r = answers[pid, "split_recover"]
        if r["status"] == "ok" and r["trace"]["recovery_triggered"]:
            first, final = b["first_ids"], r["trace"]["final_ids"]
            recovery.append({"id": pid, "split": sample["split"], "first_ids": first, "final_ids": final,
                "ordered_context_changed": first != final, "evidence_id_set_changed": set(first) != set(final),
                "new_ids": sorted(set(final) - set(first)), "removed_ids": sorted(set(first) - set(final)),
                "baseline_success": reviews[pid, "split_balanced"]["answer_success"],
                "recovery_success": reviews[pid, "split_recover"]["answer_success"],
                "generation_attempts": len(r["trace"]["generation_attempts"])})
    write("completed_case_analysis.json", {"scope": "Raw single-model verdicts retained; explanations are reviewable, not human ground truth",
          "cases": cases, "paired_transitions": paired, "recovery": recovery})
    lines = ["# 31道适用比较题逐题结果", "", "原始判定不改写。成功指独立调用的同服务模型评审通过，错误和不完整计失败。",
        "额外列‘无部分告警’用于发现系统与评审判断不一致，属于保守诊断，未替换冻结指标。",
        "", "|题号|分组|原流程|范围去重|拆分平衡|补检|拆分成功且无部分告警|", "|---|---|---|---|---|---|---|"]
    for case in cases:
        marks = ["通过" if case["arms"][arm]["model_judged_success"] else "生成失败" if case["arms"][arm]["generation_status"] == "error" else "未通过" for arm in ARMS]
        lines.append("|" + "|".join([case["id"][-3:], case["split"], *marks,
            "通过" if case["arms"]["split_balanced"]["model_success_and_no_partial_warning"] else "未通过"]) + "|")
    lines += ["", "## 每道拆分失败题的原始判定及引用问题", ""]
    for case in cases:
        arm = case["arms"]["split_balanced"]
        if arm["model_judged_success"]:
            continue
        lines += [f"### {case['id'][-3:]} · {case['split']}", "", case["question"], "",
            "- 初始证据编号：" + ", ".join(arm["first_ids"]),
            "- 标题顺序：" + " / ".join(arm["first_titles"]),
            "- 模型评审理由/生成错误：" + str(arm["generation_error"] or arm["review_reason"])]
        for point in arm["missing_points"]:
            lines.append("- 评审认为缺少：" + point)
        for rejected in arm["rejected_claims"]:
            lines.append("- 校验删除：" + rejected["claim"]["text"] + "；原因：" + rejected["reason"])
        lines += ["", "实际回答、参考回答、全部判定与补检前后编号见同目录`completed_case_analysis.json`；实际送入模型的原文见`answers_effective.jsonl`。", ""]
    (OUT / "逐题结果与失败理由.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(cases)} eligible paired cases and {len(recovery)} triggered recoveries")


if __name__ == "__main__":
    main()
