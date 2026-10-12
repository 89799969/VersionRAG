import pytest

from app.evidence_alignment import EvidenceSnapshot, anchor_quote, make_anchor
from app.knowledge import split_document
from scripts.evaluate_span_evidence import demo_bundles, evaluate_bundle


def chunk(chunk_id, start, end, text, document_id="d"):
    return {"id": chunk_id, "contents": text[start:end],
            "metadata": {"document_id": document_id, "start": start, "end": end}}


def test_rechunk_union_preserves_ground_truth_and_whole_question_result():
    text = "前言：学费六千元，学制四年。结束。"
    anchor = anchor_quote("e", "d", text, "学费六千元，学制四年。")
    old = EvidenceSnapshot("old", {"d": text}, [chunk("125", 0, len(text), text)])
    new = EvidenceSnapshot("new", {"d": text}, [chunk("x", 0, 10, text),
                                               chunk("y", 8, len(text), text)])
    assert old.score([[anchor]], ["125"], 5, retrieval_version="old")["all_evidence_hit"]
    assert not new.score([[anchor]], ["x"], 5, retrieval_version="new")["all_evidence_hit"]
    assert new.score([[anchor]], ["x", "y"], 5, retrieval_version="new")["all_evidence_hit"]
    assert new.map_anchor(anchor)["overlapping_chunk_ids"] == ["x", "y"]


def test_overlap_duplicates_and_gaps_do_not_create_false_coverage():
    text = "0123456789"
    anchor = make_anchor("e", "d", text, 0, 10)
    snapshot = EvidenceSnapshot("v", {"d": text},
                                [chunk("a", 0, 6, text), chunk("b", 0, 6, text),
                                 chunk("c", 7, 10, text), chunk("bridge", 5, 8, text)])
    assert not snapshot.score([[anchor]], ["a", "b", "c"], 5,
                              retrieval_version="v")["all_evidence_hit"]
    assert snapshot.score([[anchor]], ["a", "c", "bridge"], 5,
                          retrieval_version="v")["all_evidence_hit"]


def test_required_facts_are_and_equivalent_evidence_is_or_and_top_k_is_enforced():
    text = "学费六千元。学制四年。学费6000元。"
    fee = anchor_quote("fee", "d", text, "学费六千元。")
    alternative = anchor_quote("fee_alt", "d", text, "学费6000元。")
    years = anchor_quote("years", "d", text, "学制四年。")
    snapshot = EvidenceSnapshot("v", {"d": text},
                                [chunk("a", fee.start, fee.end, text),
                                 chunk("b", years.start, years.end, text),
                                 chunk("alt", alternative.start, alternative.end, text)])
    groups = [[fee, alternative], [years]]
    partial = snapshot.score(groups, ["alt", "b"], 1, retrieval_version="v")
    assert partial["group_hits"] == [True, False]
    assert partial["evidence_group_recall"] == 0.5
    assert snapshot.score(groups, ["alt", "b"], 2, retrieval_version="v")["all_evidence_hit"]


def test_changed_or_deleted_source_blocks_scoring_instead_of_dropping_question():
    old_text = "费用六千元。"
    anchor = anchor_quote("fee", "d", old_text, "费用六千元。")
    for documents, chunks, reason in [
        ({"d": "费用八千元。"}, [chunk("a", 0, 6, "费用八千元。")], "source_text_changed"),
        ({}, [], "source_missing"),
        ({"d": old_text}, [], "source_range_not_indexed"),
    ]:
        result = EvidenceSnapshot("v", documents, chunks).score(
            [[anchor]], [], 5, retrieval_version="v")
        assert result["status"] == "needs_review"
        assert result["all_evidence_hit"] is None
        assert result["mapping_issues"][0]["reason"] == reason


def test_no_cross_document_coverage_and_reused_ids_are_version_checked():
    text = "0123456789"
    anchor = make_anchor("e", "d", text, 0, 10)
    snapshot = EvidenceSnapshot("v", {"d": text, "other": text},
                                [chunk("a", 0, 5, text), chunk("tail", 5, 10, text),
                                 chunk("b", 5, 10, text, "other")])
    assert not snapshot.score([[anchor]], ["a", "b"], 5,
                              retrieval_version="v")["all_evidence_hit"]
    with pytest.raises(ValueError, match="versions differ"):
        snapshot.score([[anchor]], ["a"], 5, retrieval_version="old")
    with pytest.raises(ValueError, match="missing from"):
        snapshot.score([[anchor]], ["unknown"], 5, retrieval_version="v")


def test_bad_offsets_text_and_ambiguous_quotes_fail_explicitly():
    text = "证据。证据。"
    with pytest.raises(ValueError, match="ambiguous"):
        anchor_quote("e", "d", text, "证据。")
    with pytest.raises(ValueError, match="absent"):
        anchor_quote("e", "d", text, "不存在")
    bad = chunk("a", 0, 3, text)
    bad["contents"] = "伪造。"
    with pytest.raises(ValueError, match="does not match"):
        EvidenceSnapshot("v", {"d": text}, [bad])
    bad = chunk("a", 0, 3, text)
    del bad["metadata"]["start"]
    with pytest.raises(ValueError, match="offsets"):
        EvidenceSnapshot("v", {"d": text}, [bad])


def test_existing_ingestion_offsets_work_with_exact_raw_text():
    text = "课程介绍。\r\n" * 190 + "完整证据：四年。" + "后续说明。" * 60
    chunks = split_document("d", "示例资料", text)
    snapshot = EvidenceSnapshot("ingestion", {"d": text}, chunks)
    anchor = anchor_quote("years", "d", text, "完整证据：四年。")
    ids = [item["id"] for item in chunks]
    assert snapshot.score([[anchor]], ids, len(ids),
                          retrieval_version="ingestion")["all_evidence_hit"]


@pytest.mark.parametrize("groups,k", [([], 5), ([[]], 5), ([[None]], 0)])
def test_empty_annotations_and_invalid_budget_cannot_pass(groups, k):
    with pytest.raises(ValueError, match="required evidence"):
        EvidenceSnapshot("v", {}, []).score(groups, [], k, retrieval_version="v")


def test_demo_end_to_end_keeps_labels_fixed_and_blocks_changed_text():
    bundles = demo_bundles()
    labels = bundles["old_complete"]["questions"]
    assert all(bundle["questions"] == labels for bundle in bundles.values())
    reports = {name: evaluate_bundle(bundle) for name, bundle in bundles.items()}
    assert reports["old_complete"]["all_evidence_hit_rate"] == 1
    assert reports["new_partial"]["all_evidence_hit_rate"] == 0
    assert reports["new_partial"]["rows"][0]["evidence_group_recall"] == 0.5
    assert reports["new_complete"]["all_evidence_hit_rate"] == 1
    assert reports["changed_text"]["all_evidence_hit_rate"] is None
    assert reports["changed_text"]["question_count"] == 1
    assert reports["changed_text"]["blocked_question_ids"] == ["demo"]


def test_missing_retrieval_row_cannot_shrink_denominator():
    bundle = demo_bundles()["old_complete"]
    bundle["retrieval"] = []
    with pytest.raises(ValueError, match="exactly one result"):
        evaluate_bundle(bundle)
