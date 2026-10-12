from app.data_mapping import EvidenceMapper


def test_reference_source_id_is_not_confused_with_corpus_passage_id():
    mapper = EvidenceMapper([
        {"id": "159", "contents": "汉语法语英语", "metadata": {"source_id": "109"}},
        {"id": "1189", "contents": "教学采用汉语、法语、英语", "metadata": {"source_id": "159"}},
    ])
    mapped = mapper.map_reference({"id": "159", "contents": "汉语 、 法语 、 英语", "psg_id": 7})
    assert mapped["acceptable_corpus_passage_ids"] == ["1189"]
    assert mapped["source_passage_id"] == 7
    assert mapper.map_reference({"id": "159", "contents": ""})["mapping_status"] == "unmapped"


def test_literal_linebreak_escapes_match_real_linebreaks_without_changing_numbers():
    mapper = EvidenceMapper([
        {"id": "a", "contents": "年份：2023\n三语教学", "metadata": {"source_id": "10"}},
    ])
    assert mapper.map_reference({"id": "10", "contents": "年份：2023\\n三语 教学"})[
        "acceptable_corpus_passage_ids"] == ["a"]
    assert mapper.map_reference({"id": "10", "contents": "年份：2022\\n三语教学"})[
        "mapping_status"] == "unmapped"
