"""Offline, exact-text evidence coverage across chunking policies.

This is a separate metric version. It does not change the historical ID-based
benchmark or infer minimal answer evidence from its old whole-passage labels.
All offsets are Python character offsets in the exact, frozen source text.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


METRIC_VERSION = "source-span-union-v1"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceAnchor:
    evidence_id: str
    document_id: str
    document_sha256: str
    start: int
    end: int
    quote: str


def make_anchor(evidence_id: str, document_id: str, text: str,
                start: int, end: int) -> EvidenceAnchor:
    if not evidence_id or not document_id or not 0 <= start < end <= len(text):
        raise ValueError("Evidence needs an identity and a nonempty valid source range.")
    return EvidenceAnchor(evidence_id, str(document_id), text_hash(text),
                          start, end, text[start:end])


def anchor_quote(evidence_id: str, document_id: str, text: str,
                 quote: str) -> EvidenceAnchor:
    """Bootstrap a unique verbatim quote; repeated/missing text needs review."""
    start = text.find(quote)
    if not quote or start < 0 or text.find(quote, start + 1) >= 0:
        raise ValueError("Quote is absent or ambiguous; provide verified source offsets.")
    return make_anchor(evidence_id, document_id, text, start, start + len(quote))


@dataclass(frozen=True)
class ChunkSpan:
    chunk_id: str
    document_id: str
    start: int
    end: int


class EvidenceSnapshot:
    """Validate real chunks against the source snapshot before any scoring."""

    def __init__(self, version: str, documents: dict[str, str], chunks: list[dict]):
        if not version:
            raise ValueError("A knowledge snapshot version is required.")
        self.version = version
        self._documents = {str(key): text for key, text in documents.items()}
        self._hashes = {key: text_hash(text) for key, text in self._documents.items()}
        self._chunks: dict[str, ChunkSpan] = {}
        for chunk in chunks:
            chunk_id = str(chunk["id"])
            meta = chunk["metadata"]
            document_id = str(meta.get("document_id", meta.get("source_id", "")))
            if chunk_id in self._chunks or document_id not in self._documents:
                raise ValueError(f"Duplicate chunk or missing source: {chunk_id}")
            text = self._documents[document_id]
            start, end = meta.get("start"), meta.get("end")
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(text)):
                raise ValueError(f"Unverified source offsets: {chunk_id}")
            if chunk["contents"] != text[start:end]:
                raise ValueError(f"Chunk text does not match source offsets: {chunk_id}")
            if (meta.get("content_sha256") is not None
                    and meta["content_sha256"] != text_hash(chunk["contents"])):
                raise ValueError(f"Chunk content hash differs: {chunk_id}")
            self._chunks[chunk_id] = ChunkSpan(chunk_id, document_id, start, end)
        payload = {"version": version, "documents": self._hashes,
                   "chunks": [asdict(self._chunks[key]) for key in sorted(self._chunks)]}
        self.sha256 = text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _anchor_issue(self, anchor: EvidenceAnchor) -> str | None:
        text = self._documents.get(anchor.document_id)
        if text is None:
            return "source_missing"
        if self._hashes[anchor.document_id] != anchor.document_sha256:
            return "source_text_changed"
        if (not 0 <= anchor.start < anchor.end <= len(text)
                or text[anchor.start:anchor.end] != anchor.quote):
            return "invalid_anchor"
        return None

    @staticmethod
    def _covered(anchor: EvidenceAnchor, chunks: list[ChunkSpan]) -> bool:
        # One shared coordinate system; overlap cannot compensate for a gap.
        intervals = sorted((max(anchor.start, chunk.start), min(anchor.end, chunk.end))
                           for chunk in chunks
                           if chunk.document_id == anchor.document_id
                           and chunk.start < anchor.end and chunk.end > anchor.start)
        cursor = anchor.start
        for start, end in intervals:
            if start > cursor:
                return False
            cursor = max(cursor, end)
            if cursor >= anchor.end:
                return True
        return False

    def map_anchor(self, anchor: EvidenceAnchor) -> dict:
        issue = self._anchor_issue(anchor)
        candidates = [] if issue else [chunk for chunk in self._chunks.values()
                                      if chunk.document_id == anchor.document_id
                                      and chunk.start < anchor.end and chunk.end > anchor.start]
        covered = not issue and self._covered(anchor, candidates)
        return {"evidence_id": anchor.evidence_id,
                "status": "mapped" if covered else "needs_review",
                "reason": issue or (None if covered else "source_range_not_indexed"),
                "overlapping_chunk_ids": sorted(chunk.chunk_id for chunk in candidates)}

    def score(self, groups: list[list[EvidenceAnchor]], retrieved_ids: list[str],
              k: int, *, retrieval_version: str) -> dict:
        """AND across facts, OR across equivalent anchors within each fact.

        A quote split between two chunks is covered only by their interval union.
        Mapping issues block scoring instead of silently shrinking a denominator.
        """
        if retrieval_version != self.version:
            raise ValueError("Retrieval and evidence snapshot versions differ.")
        if type(k) is not int or k < 1 or not groups or any(not group for group in groups):
            raise ValueError("Positive k and nonempty required evidence groups are required.")
        ranked = list(map(str, retrieved_ids[:k]))
        if any(chunk_id not in self._chunks for chunk_id in ranked):
            raise ValueError("Retrieved chunk is missing from the pinned snapshot.")
        mappings = [self.map_anchor(anchor) for group in groups for anchor in group]
        issues = [mapping for mapping in mappings if mapping["status"] != "mapped"]
        result = {"metric_version": METRIC_VERSION, "knowledge_version": self.version,
                  "snapshot_sha256": self.sha256, "required_group_count": len(groups),
                  "retrieved_ids": ranked, "mapping_issues": issues}
        if issues:
            return {**result, "status": "needs_review", "group_hits": None,
                    "evidence_group_recall": None, "all_evidence_hit": None}
        selected = [self._chunks[chunk_id] for chunk_id in ranked]
        hits = [any(self._covered(anchor, selected) for anchor in group) for group in groups]
        return {**result, "status": "ok", "group_hits": hits,
                "evidence_group_recall": sum(hits) / len(hits),
                "all_evidence_hit": all(hits)}
