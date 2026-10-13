"""Coordinates in immutable provided passages, never an invented full webpage."""
from __future__ import annotations

from dataclasses import asdict

from app.evidence_alignment import EvidenceSnapshot, make_anchor, text_hash

SOURCE_SCOPE = "upstream_preprocessed_passage"


def normalized_positions(text: str):
    characters, positions = [], []
    index = 0
    while index < len(text):
        if text[index:index + 2] in ("\\n", "\\r", "\\t"):
            index += 2
            continue
        if not text[index].isspace():
            characters.append(text[index])
            positions.append(index)
        index += 1
    return "".join(characters), positions


def locate_reference(text: str, reference: str):
    normalized, positions = normalized_positions(text)
    needle, _ = normalized_positions(reference)
    if not needle:
        return {"status": "blocked", "reason": "empty_reference", "ranges": []}
    ranges, start = [], normalized.find(needle)
    while start >= 0:
        ranges.append([positions[start], positions[start + len(needle) - 1] + 1])
        start = normalized.find(needle, start + 1)
    if len(ranges) != 1:
        return {"status": "blocked", "reason": "reference_absent" if not ranges else "reference_ambiguous",
                "ranges": ranges}
    return {"status": "mapped", "reason": None, "ranges": ranges}


def passage_snapshot(corpus: list[dict], version: str):
    documents, chunks, origins = {}, [], {}
    for row in corpus:
        pid = str(row["id"])
        if pid in origins:
            raise ValueError("Original passage IDs must be unique")
        source_id = str(row["metadata"]["source_id"])
        document_id = f"upstream:{source_id}:{pid}"
        text = row["contents"]
        documents[document_id] = text
        origins[pid] = {"document_id": document_id, "source_id": source_id,
                        "title": row["title"], "url": row["metadata"].get("url"),
                        "source_scope": SOURCE_SCOPE, "document_sha256": text_hash(text)}
        chunks.append({"id": pid, "contents": text, "metadata": {
            **row["metadata"], "document_id": document_id, "start": 0, "end": len(text)}})
    snapshot = EvidenceSnapshot(version, documents, chunks)
    return snapshot, documents, chunks, origins


def align_legacy_case(sample: dict, documents: dict, origins: dict):
    groups, issues = [], []
    for group_index, group in enumerate(sample["required_evidence_groups"]):
        anchors = []
        for pid in map(str, group["acceptable_corpus_passage_ids"]):
            if pid not in origins:
                issues.append({"group": group_index, "passage_id": pid, "reason": "passage_missing"})
                continue
            origin = origins[pid]
            if origin["source_id"] != str(group["source_id"]):
                issues.append({"group": group_index, "passage_id": pid, "reason": "source_identity_wrong"})
                continue
            text = documents[origin["document_id"]]
            located = locate_reference(text, group["reference_text"])
            if located["status"] != "mapped":
                issues.append({"group": group_index, "passage_id": pid, "reason": located["reason"]})
                continue
            start, end = located["ranges"][0]
            anchors.append({**asdict(make_anchor(f"{sample['id']}:g{group_index}:{pid}", origin["document_id"], text, start, end)),
                            "source_id": origin["source_id"], "original_passage_id": pid})
        groups.append(anchors)
    return {"id": sample["id"], "question": sample["question"], "label_status": "upstream_unreviewed",
            "source_scope": SOURCE_SCOPE, "expected_answer_groups": sample.get("expected_answer_groups"),
            "required_evidence_groups": groups, "mapping_status": "blocked" if any(not g for g in groups) else "mapped",
            "mapping_issues": issues, "human_confirmed": False}
