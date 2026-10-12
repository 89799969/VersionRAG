"""Readable delivery from saved experimental inputs, outputs and judgments."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import ROOT, read_jsonl

OUT = ROOT / "artifacts/reports/gap_recovery_2026-10-08"
OLD = ROOT / "artifacts/reports/comparison_2026-10-07"
LABELS = {"no_recovery": "不补检", "broad_priority": "宽查询＋新结果优先",
          "targeted_priority": "定向查询＋新结果优先", "broad_joint": "宽查询＋统一选择",
          "targeted_joint": "定向查询＋统一选择"}
ARMS = list(LABELS)


def load(name):
    labeled = {"answers.jsonl": "answers_labeled.jsonl", "reviews.jsonl": "reviews_labeled.jsonl"}.get(name)
    if labeled and (OUT / labeled).exists():
        name = labeled
    return read_jsonl(OUT / name) if (OUT / name).exists() else []


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    answers, reviews, gaps, retrieval, events = [load(n) for n in
        ("answers.jsonl", "reviews.jsonl", "gap_plans.jsonl", "supplemental_retrieval.jsonl", "requests.jsonl")]
    key = lambda r: (r["cohort"], r["id"], r["arm"])
    by_review = {key(r): r for r in reviews}
    by_gap = {(r["cohort"], r["id"]): r for r in gaps}
    by_retrieval = {(r["cohort"], r["id"]): r for r in retrieval}
    detail = ["# 逐题实验记录", "", "所有结论来自保存的回答和同服务单轮评审；评分可误判。先展示查询和实际输入，再展示判定。", ""]
    analysis = []
    cases = sorted({(r["cohort"], r["id"]) for r in answers})
    for cohort, qid in cases:
        selected = {r["arm"]: r for r in answers if r["cohort"] == cohort and r["id"] == qid}
        first = selected["no_recovery"]
        item = {"cohort": cohort, "id": qid, "question": first["question"],
                "triggered": first.get("recovery_triggered", False), "arms": {}}
        detail += [f"## {cohort} / {qid}", "", first["question"], "",
                   "首次不足原因：" + json.dumps([(a["entity"], a["missing_reason"]) for a in first.get("aspects", [])
                                              if a["status"] == "insufficient"], ensure_ascii=False), ""]
        gap = by_gap.get((cohort, qid))
        if gap:
            item["gap_plan"] = gap
            detail += ["缺口规划：", "", "```json", json.dumps(gap, ensure_ascii=False, indent=2), "```", ""]
        ret = by_retrieval.get((cohort, qid))
        if ret:
            item["supplemental_retrieval"] = ret
            for kind in ("broad", "targeted"):
                detail += [f"{kind}补检：", ""]
                for step in ret.get(kind + "_trace", {}).get("steps", []):
                    detail += [f"- {step['entity']}：`{step['query']}`；候选{step['candidate_ids']}；重排前五{step['ranked_ids']}", ""]
        detail += ["| 组 | 服务状态 | 评审完整成功 | 无部分告警成功 | 实际模型证据编号 | 新增调用输入token |", "|---|---|---|---|---|---|"]
        for arm in ARMS:
            row = selected[arm]
            review = by_review[key(row)]
            ids = [e["passage_id"] for e in row.get("evidence", [])]
            success = review["answer_success"]
            item["arms"][arm] = {"status": row["status"], "success": success,
                "success_no_partial": success and not row.get("insufficient_evidence", True),
                "actual_context_ids": ids, "selected_ids": row.get("final_ids", row.get("trace", {}).get("final_ids")),
                "answer": row.get("answer"), "review": review,
                "incremental_usage": row.get("incremental_usage", {}),
                "merge_trace": row.get("merge_trace"), "rejected_claims": row.get("rejected_claims", [])}
            detail.append(f"| {LABELS[arm]} | {row['status']} | {success} | {item['arms'][arm]['success_no_partial']} | {', '.join(ids)} | {row.get('incremental_usage', {}).get('prompt_tokens', 0)} |")
        detail.append("")
        for arm in ARMS:
            row, review = selected[arm], by_review[key(selected[arm])]
            detail += [f"### {LABELS[arm]}", "", row.get("answer", row.get("error", "无有效回答")), "",
                "评审：" + review.get("reason", review.get("error_type", review["status"])), "",
                "缺失点：" + json.dumps(review.get("missing_points", []), ensure_ascii=False), "",
                "被校验删除的结论：" + json.dumps(row.get("rejected_claims", []), ensure_ascii=False), ""]
        analysis.append(item)
    (OUT / "逐题记录.md").write_text("\n".join(detail), encoding="utf-8")
    (OUT / "case_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    report = ["# 补检查询与证据合并：完整过程和结果", "", "## 实验回答什么", "",
        "测试模型给出的缺口主题词是否比宽泛专业名更有用，以及统一选择新旧证据是否改善结果。线上代码、生成提示、索引和旧报告保持。方案在调用前固定，未按本轮诊断结果调参。", "",
        "## 每组怎么修改", "",
        "- 不补检：复用相同首次草稿。",
        "- 宽查询＋新结果优先：原补检；自然题复用旧逐步排名及旧生成。",
        "- 定向查询＋新结果优先：增加一次缺口规划，原问题维度和年份加主题词；其他合并方式保持。",
        "- 宽查询＋统一选择：复用宽查询，合并旧五条与补检五条，按原问题统一重排；每专业最多保留一条旧引用锚点；两侧交替选最多五条；正文哈希集合不变则不再生成。",
        "- 定向查询＋统一选择：上述两种修改组合。",
        "", "查询收益可在相同选择方式下对照；统一选择是重排、有限引用保留和停止条件的组合，不能分别归因。引用保护只保证旧字面依据保留，不保证旧结论语义正确。", "",
        "## 数据与评审边界", "",
        "原开发五题全部未触发补检，因此另做五题受控移除第二专业资料的测试。这五题同属政治学/外交学，用于检查机制，不是自然业务准确率。自然诊断使用历史31题，开发5、原扩展留出11、诊断15；原失败题保留。所有历史题均已暴露，不称为新的未见测试。", "",
        "各组同一个首次草稿、最多一次修订、最多五条证据和16000字符上下文。单轮生成与同服务自动评审有随机性和评审偏差，不称为人工准确率。成功要求有事实、事实正确、问题完整、所有输出事实有上下文支持；另列成功且系统无部分告警。", "",
        "## 分组结果", "",
        "| 数据 | 组 | 完整成功 | 成功且无部分告警 | 错误 | 平均总输入token | 平均新增输入token |", "|---|---|---|---|---|---|---|"]
    for s in summary:
        name = s["cohort"] + "/" + s["split"]
        report.append(f"| {name} | {LABELS[s['arm']]} | {s['success_count']}/{s['questions']} | {s['success_no_partial']}/{s['questions']} | {s['errors']} | {s['mean_total_prompt_tokens']:.0f} | {s['mean_incremental_prompt_tokens']:.0f} |")
    report += ["", "## 逐题新增与退步", ""]
    for s in summary:
        if s["arm"] == "no_recovery":
            continue
        report += [f"- {s['cohort']}/{s['split']}，{LABELS[s['arm']]}：新增{[x.rsplit('_',1)[-1] for x in s['gains']]}；退步{[x.rsplit('_',1)[-1] for x in s['regressions']]}。"]
    report += ["", "## 实际调用和成本口径", ""]
    statuses = Counter((e["host"], e["path"], e.get("status")) for e in events)
    calls = [{"host": h, "path": p, "status": s, "calls": n} for (h, p, s), n in statuses.items()]
    actual_usage = Counter()
    for row in load("controlled_initial.jsonl") + gaps:
        actual_usage.update({k: v for k, v in row.get("usage", {}).items() if k in ("prompt_tokens", "completion_tokens", "total_tokens")})
    for row in answers:
        if row.get("revision_called"):
            actual_usage.update({k: v for k, v in row.get("usage", {}).items() if k in ("prompt_tokens", "completion_tokens", "total_tokens")})
    for row in reviews:
        if row["status"] == "ok" and not row.get("reused_input_hash"):
            actual_usage.update({k: v for k, v in row.get("usage", {}).items() if k in ("prompt_tokens", "completion_tokens", "total_tokens")})
    cost = {"service_calls": calls, "saved_successful_actual_model_usage": dict(actual_usage),
        "gap_plans": len(gaps), "gap_fallbacks": sum(r.get("fallback", False) for r in gaps),
        "judge_hash_reuses": sum(r.get("reused_input_hash", False) for r in reviews),
        "scope": "successful saved calls only; no currency price inferred; embedding/rerank usage not in model token total",
        "preflight": "separate sandbox_preflight report retains five connection failures with no HTTP response"}
    (OUT / "cost_summary.json").write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")
    report += ["```json", json.dumps(cost, ensure_ascii=False, indent=2), "```", "",
        "分组成本按每组独立运行的逻辑调用计算，实际实验共享首次草稿、缺口规划和检索，旧评审按完整输入哈希复用。总输入token包含首次生成、该组缺口规划和发生的修订；不含评审费用。实际费用相关请求与成功调用token另列，不臆测人民币价格。", "",
        "执行为串行服务请求，硅基流动按1.5秒起始间隔节流。旧组的历史时间、缓存、网络与新组不同，本轮不宣称生产延迟改善。受限环境首次五次连接失败已另存，成功联网后原方案重跑；不是根据坏答案重抽样。", "",
        "追加的原上下文控制在主结果后单独执行，结果和费用口径见[结论与根因分析](结论与根因分析.md)及[全量调用](全量调用与成本.md)；上面的cost_summary只统计主实验。", "",
        "## 怎么复现", "", "先阅读实验设计和policy.json。冻结目录不允许代码/数据变化后继续。脚本支持分阶段续接，已保存的回答不会重跑。", "",
        "```text", "python scripts/experiment_gap_recovery.py --phase dev",
        "python scripts/experiment_gap_recovery.py --phase evaluation",
        "python scripts/experiment_gap_recovery.py --phase judge",
        "python scripts/audit_gap_recovery_labels.py",
        "python scripts/report_gap_recovery.py", "```", "",
        "## 证据入口", "", "- [逐题记录](逐题记录.md)：每题每组的真实查询、候选编号、上下文、回答、删除事实及评审理由。",
        "- [机器分析](case_analysis.json)、[汇总](summary.json)、[审计](audit.json)、[成本](cost_summary.json)。",
        "- [实验设计](实验设计.md)、[冻结策略](policy.json)、requests.jsonl保存确切请求体，未保存认证头。", "",
        "## 记录修复", "", "审计发现跳过修订时复用首次草稿覆盖了新组名。原始回答/评审不修改；根据串行五组顺序及合并轨迹生成answers_labeled.jsonl、reviews_labeled.jsonl。label_repair.json记录修正，所有实际上下文、回答及评审输入哈希不变。汇总读取修复视图。跳过修订时最终真实输入以evidence字段为准，候选选择编号不代表再次送入模型。", "",
        "不要直接重跑已完成的evaluation阶段：旧标签错误会使该阶段误以为部分实验尚未完成。复核时使用标签审计脚本；新实验应先修复复用草稿的元数据覆盖并使用新目录。", "",
        "本轮是工程实验，不表示学习者已经掌握实现。候选策略未发布到线上。"]
    (OUT / "完整过程与结果.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "cost": cost}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
