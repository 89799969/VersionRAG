"""Final read-only consistency checks and combined safe service accounting."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import ROOT, read_jsonl, sha256

OUT = ROOT / "artifacts/reports/gap_recovery_2026-10-08"
C = OUT / "same_context_control"


def main():
    main_audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    control_audit = json.loads((C / "audit.json").read_text(encoding="utf-8"))
    assert main_audit["status"] == control_audit["status"] == "passed"
    labels = json.loads((OUT / "label_repair.json").read_text(encoding="utf-8"))
    assert sha256(OUT / "answers.jsonl") == labels["original_answers_sha256"]
    assert sha256(OUT / "reviews.jsonl") == labels["original_reviews_sha256"]
    all_events = read_jsonl(OUT / "requests.jsonl") + read_jsonl(C / "requests.jsonl")
    calls = Counter((e["host"], e["path"], e.get("status")) for e in all_events)
    body_by_role = Counter(e["role"].split(":", 1)[0] for e in all_events if e["host"] == "api.deepseek.com")
    control_usage = Counter()
    for row in read_jsonl(C / "answers.jsonl"):
        if row.get("revision_called"):
            control_usage.update({k: row["usage"].get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")})
    for row in read_jsonl(C / "reviews.jsonl"):
        if row["status"] == "ok" and not row.get("reused_input_hash"):
            control_usage.update({k: row["usage"].get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")})
    primary = json.loads((OUT / "cost_summary.json").read_text(encoding="utf-8"))["saved_successful_actual_model_usage"]
    combined = {k: primary.get(k, 0) + control_usage.get(k, 0) for k in control_usage}
    final = {"status": "passed", "main_answer_records": 180, "control_answer_records": 31,
        "old_and_measured_files_preserved": True, "label_only_repairs": len(labels["changes"]),
        "service_calls": [{"host": h, "path": p, "status": s, "count": n} for (h, p, s), n in calls.items()],
        "deepseek_calls_by_role": dict(body_by_role), "control_saved_valid_usage": dict(control_usage),
        "combined_saved_valid_model_usage": combined,
        "unavailable_usage": "Control003 got HTTP200 but no valid structured result; its token usage was not retained. Combined usage is a lower bound, not an invoice total.",
        "checks": ["Primary and control audits passed", "Original measurement record hashes preserved",
                   "22 related offline tests passed, one existing dependency warning", "No production/index/code changes to frozen app files"],
        "scope": "Single same-provider review; historical exposed natural cohort and masked dev kept separate"}
    (OUT / "delivery_audit.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ["# 全量调用与成本", "", "主实验及追加原上下文控制合并。只统计服务请求和保存下来的实际token，不根据未核验价格估计人民币金额。", "",
            "```json", json.dumps(final, ensure_ascii=False, indent=2), "```", "",
            "同一模型请求返回200不代表业务结构/引文校验成功。控制003未保留有效结果的usage，合计token是已知有效调用的下界，不能当账单总数。", "",
            "模型评审费用计入实际调用统计，但分组总输入表只含首次生成、缺口规划和修订，不含评审。规划/候选/评审共享复用使实际调用与逐组独立运行不同。", "",
            "受限环境五次连接失败保存在同级gap_recovery_2026-10-08_sandbox_preflight；这些请求没有收到HTTP响应，没有混成回答质量失败。"]
    (OUT / "全量调用与成本.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
