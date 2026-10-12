"""Bounded two-entity comparisons. Plans use question + corpus titles, never gold.

Exact content equivalence preserves all original IDs and sources. It is separate
from semantic equivalence and does not rewrite the historical benchmark labels.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlparse


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


class ExactEvidenceIndex:
    def __init__(self, documents: list[dict]):
        self.by_id = {str(d["id"]): d for d in documents}
        if len(self.by_id) != len(documents):
            raise ValueError("Corpus passage IDs must be unique")
        self.groups: dict[str, list[str]] = {}
        self.id_hash = {}
        for doc in documents:
            key = hashlib.sha256(doc["contents"].encode("utf-8")).hexdigest()
            self.id_hash[str(doc["id"])] = key
            self.groups.setdefault(key, []).append(str(doc["id"]))

    def equivalents(self, ids) -> set[str]:
        return {alias for pid in ids for alias in self.groups[self.id_hash[str(pid)]]}

    def deduplicate(self, hits: list[dict], allowed_ids: set[str] | None = None) -> list[dict]:
        result, seen = [], set()
        for hit in hits:
            key = self.id_hash[str(hit["id"])]
            if key in seen:
                continue
            seen.add(key)
            aliases = [pid for pid in self.groups[key] if allowed_ids is None or pid in allowed_ids]
            result.append({**hit, "rank": len(result) + 1, "content_sha256": key,
                           "equivalent_passage_ids": aliases,
                           "equivalent_sources": [{"passage_id": pid,
                               "source_id": str(self.by_id[pid]["metadata"]["source_id"]),
                               "url": self.by_id[pid]["metadata"].get("url")}
                               for pid in aliases]})
        return result


@dataclass(frozen=True)
class ComparisonPlan:
    question: str
    entities: tuple[str, str]
    facet: str
    queries: tuple[str, str]
    policy: str = "professional-title-two-entity-v2"

    def payload(self):
        return asdict(self)


class TitlePlanner:
    def __init__(self, documents: list[dict]):
        # Corpus metadata is available at runtime; QA labels are not.
        self.titles = sorted({compact(d["title"]) for d in documents
                              if 3 <= len(compact(d["title"])) <= 40
                              and ("/professional/" in urlparse(d["metadata"].get("url") or "").path
                                   or not d["metadata"].get("url"))},
                             key=lambda t: (-len(t), t))
        self.ids_by_title = {}
        for doc in documents:
            self.ids_by_title.setdefault(compact(doc["title"]), set()).add(str(doc["id"]))

    def plan(self, question: str) -> ComparisonPlan | None:
        text = compact(question)
        if not re.search(r"区别|不同|异同|共同|各自|分别|相比|对比|比较|各.*(?:哪些|什么)", text):
            return None
        found = []
        for title in self.titles:
            start = text.find(title)
            if start < 0:
                continue
            end = start + len(title)
            if any(start < previous_end and end > previous_start for previous_start, previous_end, _ in found):
                continue
            found.append((start, end, title))
        found.sort()
        if len(found) != 2:
            return None
        entities = (found[0][2], found[1][2])
        tail = text[found[-1][1]:].strip("专业与和、在的，,？?。")
        facet = re.sub(r"(?:有哪些)?(?:区别|不同|异同)[？?]?$", "", tail).strip() or "相关资料"
        # Repeat every explicit year rather than adding one from a reference label.
        years = " ".join(dict.fromkeys(re.findall(r"(?:19|20)\d{2}年?", text)))
        queries = tuple(f"{years} 中国人民大学 {entity}专业 {facet}".strip() for entity in entities)
        return ComparisonPlan(question, entities, facet, queries)

    def allowed(self, entity: str, source_id: str | None = None, documents=None):
        ids = set(self.ids_by_title[entity])
        if source_id is not None:
            ids &= {str(d["id"]) for d in documents if str(d["metadata"]["source_id"]) == source_id}
        return ids


def balanced_hits(groups: dict[str, list[dict]], top_k: int = 5) -> list[dict]:
    if not 2 <= top_k <= 20:
        raise ValueError("Comparisons require a final top_k between 2 and 20")
    selected, positions, by_hash = [], {entity: 0 for entity in groups}, {}
    while len(selected) < top_k:
        changed = False
        for entity, hits in groups.items():
            while positions[entity] < len(hits):
                hit = hits[positions[entity]]
                positions[entity] += 1
                key = hashlib.sha256(hit["contents"].encode("utf-8")).hexdigest()
                if key in by_hash:
                    old = by_hash[key]
                    if entity not in old["evidence_entities"]:
                        old["evidence_entities"].append(entity)
                    continue
                item = {**hit, "rank": len(selected) + 1, "evidence_entities": [entity]}
                selected.append(item)
                by_hash[key] = item
                changed = True
                break
            if len(selected) == top_k:
                break
        if not changed:
            break
    return selected


class ComparisonRetriever:
    def __init__(self, hybrid, reranker, documents: list[dict], candidate_k: int = 30):
        self.hybrid, self.reranker, self.documents = hybrid, reranker, documents
        self.planner, self.exact = TitlePlanner(documents), ExactEvidenceIndex(documents)
        if not 1 <= candidate_k <= 100:
            raise ValueError("Invalid candidate budget")
        self.candidate_k = candidate_k

    def retrieve(self, plan: ComparisonPlan, *, split: bool = True, top_k: int = 5,
                 source_id: str | None = None, supplemental: bool = False,
                 only_entities: list[str] | None = None):
        if only_entities is not None and (not only_entities or not set(only_entities) <= set(plan.entities)):
            raise ValueError("Supplemental entities must come from the original plan")
        entities = only_entities or list(plan.entities)
        steps, groups = [], {}
        targets = entities if split else ["joint"]
        for target in targets:
            ids = set().union(*(self.planner.allowed(e, source_id, self.documents)
                               for e in (entities if target == "joint" else [target])))
            if target == "joint":
                query = plan.question
            else:
                query = plan.queries[list(plan.entities).index(target)]
                if supplemental:
                    # Broad corpus-title lookup; no expected facts or gold IDs.
                    query = f"中国人民大学 {target}专业 " + " ".join(
                        dict.fromkeys(re.findall(r"(?:19|20)\d{2}年?", plan.question)))
            start = time.perf_counter()
            pool = self.hybrid.search(query, self.candidate_k, source_id, allowed_ids=ids) if ids else []
            retrieved_ms = (time.perf_counter() - start) * 1000
            unique = self.exact.deduplicate(pool, ids)
            start = time.perf_counter()
            ranked = self.reranker.rerank(query, unique, min(top_k, len(unique))) if unique else []
            rerank_ms = (time.perf_counter() - start) * 1000
            steps.append({"entity": target, "query": query, "supplemental": supplemental,
                          "scope_documents": len(ids), "candidate_ids": [str(h["id"]) for h in pool],
                          "unique_candidate_ids": [str(h["id"]) for h in unique],
                          "ranked_ids": [str(h["id"]) for h in ranked],
                          "ranked_scores": [h["score"] for h in ranked],
                          "retrieval_ms": retrieved_ms, "rerank_ms": rerank_ms})
            groups[target] = ranked
        if split:
            hits = balanced_hits(groups, top_k)
        else:
            hits = [{**h, "evidence_entities": [e for e in entities
                     if str(h["id"]) in self.planner.ids_by_title[e]]} for h in groups["joint"]]
        return hits, {"policy": plan.policy, "plan": plan.payload(), "steps": steps,
                      "final_ids": [str(h["id"]) for h in hits],
                      "candidate_k_per_query": self.candidate_k, "top_k": top_k}
