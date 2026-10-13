from dataclasses import asdict

from app.evidence_alignment import EvidenceAnchor
from app.knowledge import split_document
from app.source_coordinates import align_legacy_case, locate_reference, passage_snapshot


def test_normalized_match_returns_raw_character_positions_not_normalized_indices():
    text = '标题\r\n学 费：八 千 元。\\n学制四年。'
    result = locate_reference(text, '学费：八千元。')
    start, end = result['ranges'][0]
    assert text[start:end] == '学 费：八 千 元。'
    assert locate_reference(text, '学制 四年。')['status'] == 'mapped'
    assert locate_reference('学费。学费。', '学费。')['reason'] == 'reference_ambiguous'
    assert locate_reference(text, '一万元')['reason'] == 'reference_absent'


def test_fixed_passage_coordinates_survive_rechunking_without_inventing_webpage_offsets():
    corpus = [{'id': '125', 'title': '教学构造', 'contents': '前言。' * 190 + '学费八千元。学制四年。' + '后文。' * 190,
               'metadata': {'source_id': '159', 'url': 'https://example.test/history'}}]
    old, documents, _, origins = passage_snapshot(corpus, 'old')
    sample = {'id': 'q', 'question': '学费和学制？', 'required_evidence_groups': [
        {'source_id': '159', 'acceptable_corpus_passage_ids': ['125'], 'reference_text': '学费 八千元。学制四年。'}]}
    label = align_legacy_case(sample, documents, origins)
    anchor_data = label['required_evidence_groups'][0][0]
    anchor = EvidenceAnchor(**{k: v for k, v in anchor_data.items() if k in EvidenceAnchor.__dataclass_fields__})
    assert anchor_data['source_id'] == '159' and label['source_scope'] == 'upstream_preprocessed_passage'
    assert not label['human_confirmed']
    from app.evidence_alignment import EvidenceSnapshot
    doc_id = origins['125']['document_id']
    new_chunks = split_document(doc_id, '教学构造', documents[doc_id])
    new = EvidenceSnapshot('new', documents, new_chunks)
    assert new.score([[anchor]], [c['id'] for c in new_chunks], len(new_chunks), retrieval_version='new')['all_evidence_hit']


def test_unmapped_case_is_retained_with_explicit_reason():
    row = {'id': '1', 'title': '标题', 'contents': '学费八千元。', 'metadata': {'source_id': 's'}}
    _, docs, _, origins = passage_snapshot([row], 'v')
    sample = {'id': 'q', 'question': '学费？', 'required_evidence_groups': [
        {'source_id': 'other', 'acceptable_corpus_passage_ids': ['1'], 'reference_text': '学费八千元。'}]}
    result = align_legacy_case(sample, docs, origins)
    assert result['mapping_status'] == 'blocked' and result['id'] == 'q'
    assert result['mapping_issues'][0]['reason'] == 'source_identity_wrong'
