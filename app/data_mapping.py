"""Evidence alignment for offline evaluation only; never imported by retrieval."""
from __future__ import annotations

import re
from collections import defaultdict


def compact(text: str) -> str:
    # Some original multi-doc references contain literal backslash-n/backslash-r
    # while the corpus contains actual line breaks. Decode only these formatting
    # escapes; do not alter punctuation, numbers or semantic content.
    text = text.replace("\\n", "\n").replace("\\r", "\r")
    return re.sub(r"\s+", "", text)


class EvidenceMapper:
    def __init__(self, corpus: list[dict]):
        self.by_source = defaultdict(list)
        for document in corpus:
            self.by_source[str(document["metadata"]["source_id"])].append(
                (document, compact(document["contents"])))

    def map_reference(self, reference: dict) -> dict:
        text = reference.get("contents", "")
        needle = compact(text)
        source_id = str(reference["id"])
        matches = [document for document, normalized in self.by_source[source_id]
                   if needle and needle in normalized]
        return {"source_id": source_id,
                "source_passage_id": reference.get("psg_id", reference.get("pid")),
                "acceptable_corpus_passage_ids": [str(document["id"]) for document in matches],
                "title": reference.get("title"), "url": reference.get("url"),
                "reference_date": reference.get("date"), "reference_text": text,
                "mapping_status": "matched" if matches else "unmapped"}
