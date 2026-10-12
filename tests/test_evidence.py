from app.evaluation import evidence_metrics, split_for
from app.retrieval import tokenize


def test_equivalent_passages_and_multiple_required_groups():
    sample = {"required_evidence_groups": [
        {"source_id": "10", "acceptable_corpus_passage_ids": ["a", "duplicate-a"]},
        {"source_id": "20", "acceptable_corpus_passage_ids": ["b"]},
    ]}
    partial = evidence_metrics(sample, ["x", "duplicate-a", "b"], 2)
    assert partial == {"evidence_group_recall": 0.5, "all_evidence_hit": False, "reciprocal_rank": 0.5}
    assert evidence_metrics(sample, ["duplicate-a", "b"], 2)["all_evidence_hit"]
    assert evidence_metrics(sample, ["a", "duplicate-a"], 2)["evidence_group_recall"] == 0.5


def test_corpus_spaces_do_not_change_chinese_query_tokenization():
    assert tokenize("汉语 、 法语 、 英语 三语 教学") == tokenize("汉语、法语、英语三语教学")


def test_same_source_scenario_stays_in_same_split():
    def sample(source_ids):
        return {"required_evidence_groups": [{"source_id": sid} for sid in source_ids]}
    assert split_for(sample(["159", "10"])) == split_for(sample(["10", "159", "159"]))
