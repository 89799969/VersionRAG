"""Map multi-document, historical time and noise evidence; report unmapped cases honestly."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_mapping import EvidenceMapper, compact
from app.evaluation import split_for
from app.retrieval import CORPUS, ROOT, read_jsonl, sha256

TASKS = {"multi_doc": "multi-doc_qa/multidoc_qa.jsonl",
         "time_sensitive": "time-sensitive_qa/time_sensitive.jsonl",
         "noise": "noisy_qa/noisy_qa_ver3.jsonl"}


def references(record):
    value = record.get("positive_references", record.get("positive_reference"))
    return value if isinstance(value, list) else [value]


def main():
    mapper = EvidenceMapper(read_jsonl(CORPUS))
    base_samples = read_jsonl(ROOT / "data/basic_qa_mapped.jsonl")
    families = defaultdict(set)
    for sample in base_samples:
        families[compact(sample["question"])].add(split_for(sample))
    all_rows, inputs = [], {}
    for task, relative in TASKS.items():
        path = ROOT / "DomainRAG/BCM/labeled_data" / relative
        inputs[task] = sha256(path)
        for line, record in enumerate(read_jsonl(path), 1):
            groups = [mapper.map_reference(reference) for reference in references(record)]
            row = {"id": f"domainrag_{task}_{line:03d}", "task": task,
                   "question": record["question"], "required_evidence_groups": groups,
                   "expected_answer_groups": record.get("answers"),
                   "reference_answer": record.get("answer"),
                   "answer_format": "free_form" if task == "multi_doc" else "keypoint_groups",
                   "as_of": record.get("date"), "original_file": str(path.relative_to(ROOT)),
                   "original_line": line, "mapping_complete": all(group["mapping_status"] == "matched" for group in groups)}
            source_splits = {split_for({"required_evidence_groups": [group]}) for group in groups}
            proposed = source_splits.pop() if len(source_splits) == 1 else "diagnostic"
            row["proposed_split"] = proposed
            families[compact(row["question"])].add(proposed)
            if task == "noise":
                # Official noise is an injected-context benchmark, not proof that
                # our actual retriever selected these documents.
                row["noise_evidence_groups"] = [mapper.map_reference(reference)
                                                for reference in record.get("noisy_reference", [])]
            all_rows.append(row)
    for row in all_rows:
        family_splits = families[compact(row["question"])]
        row["split"] = "demo" if "demo" in family_splits else (
            next(iter(family_splits)) if len(family_splits) == 1 else "diagnostic")
        del row["proposed_split"]
    path = ROOT / "data/extended_qa_mapped.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    report = {"corpus_sha256": sha256(CORPUS), "inputs_sha256": inputs,
              "matching_normalization": "decode literal newline escapes and remove whitespace; preserve semantic characters",
              "output_sha256": sha256(path), "tasks": {},
              "split_rule": "preserve frozen base source split; mixed source or question families are diagnostic only",
              "scope": "text containment alignment, not semantic label audit"}
    for task in TASKS:
        rows = [row for row in all_rows if row["task"] == task]
        report["tasks"][task] = {"records": len(rows), "complete_evidence_mapping": sum(row["mapping_complete"] for row in rows),
                                 "split_counts": dict(Counter(row["split"] for row in rows)),
                                 "unmapped_ids": [row["id"] for row in rows if not row["mapping_complete"]]}
    (ROOT / "artifacts/reports/extended_mapping_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
