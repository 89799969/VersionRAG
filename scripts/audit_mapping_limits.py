"""Structural diagnostics only; don't print holdout questions or answers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_mapping import EvidenceMapper, compact
from app.retrieval import CORPUS, ROOT, read_jsonl

mapper = EvidenceMapper(read_jsonl(CORPUS))
rows = read_jsonl(ROOT / "data/extended_qa_mapped.jsonl")
diagnostics = []
for row in rows:
    for group in row["required_evidence_groups"]:
        if group["mapping_status"] == "matched":
            continue
        candidates = mapper.by_source[group["source_id"]]
        needle = compact(group["reference_text"])
        diagnostics.append({"task": row["task"], "source_id": group["source_id"],
                            "reference_chars": len(needle), "source_passages": len(candidates),
                            "contains_html_tags": "<table" in needle or "<p" in needle,
                            "passages_contained_in_reference": sum(text in needle for doc, text in candidates)})
print(json.dumps(diagnostics[:8], ensure_ascii=False, indent=2))

for row in rows:
    if row["split"] != "dev" or row["mapping_complete"]:
        continue
    import difflib
    for group in row["required_evidence_groups"]:
        if group["mapping_status"] == "matched":
            continue
        needle = compact(group["reference_text"])
        best = None
        for doc, text in mapper.by_source[group["source_id"]]:
            common = difflib.SequenceMatcher(None, needle, text, autojunk=False).find_longest_match()
            if best is None or common.size > best[0]:
                best = (common.size, doc, text, common)
        print(json.dumps({"dev_id": row["id"], "source_id": group["source_id"],
                          "reference_preview": group["reference_text"][:260],
                          "best_corpus_id": best[1]["id"] if best else None,
                          "best_corpus_preview": best[1]["contents"][:260] if best else None,
                          "longest_exact_match_chars": best[0] if best else 0}, ensure_ascii=False, indent=2))
    break
