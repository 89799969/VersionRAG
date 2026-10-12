"""Explain a saved real trial without model keys, network, or new inference."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import ROOT, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="038", help="Official multi-document case number, e.g. 037 or 038")
    args = parser.parse_args()
    case_id = "domainrag_multi_doc_" + args.case.zfill(3)
    folder = ROOT / "artifacts/reports/comparison_2026-10-07"
    retrievals = read_jsonl(folder / "dev_retrieval.jsonl") + read_jsonl(folder / "evaluation_retrieval.jsonl")
    selected = [r for r in retrievals if r["id"] == case_id]
    if not selected:
        raise SystemExit("No saved trial for this case")
    print("这是已保存真实实验的回放，不发起新检索或模型调用。\n")
    print("问题：" + selected[0]["question"])
    names = {"original": "原完整问题检索", "scope_dedup": "限定两个专业，但不拆问题", "split_balanced": "分别检索并平衡合并"}
    for row in selected:
        print("\n" + names[row["arm"]])
        print("前五条：" + "；".join(f"{h['id']}（{h['title']}）" for h in row["hits"]))
        print("全部标注证据命中：" + str(row["original_labels"]["all_evidence_hit"]))
        for step in row["trace"].get("steps", []):
            print("子查询：" + step["query"])
    answers = [r for r in read_jsonl(folder / "answers.jsonl") if r["id"] == case_id and r["arm"] == "split_balanced"]
    if answers and answers[0]["status"] == "ok":
        print("\n已保存的新回答：\n" + answers[0]["answer"])
        print("\n注意：引用字面校验通过不等于结论一定正确；仍需核对原文与问题要求。")
    elif answers:
        print("\n该题生成尚未成功：" + answers[0].get("error", answers[0].get("reason", "未知状态")))


if __name__ == "__main__":
    main()
