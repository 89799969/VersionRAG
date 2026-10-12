import json

import pytest

from app.retrieval import ChineseBM25


def test_persistent_index_filters_before_topk_and_detects_changed_corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    documents = [
        {"id": "a", "title": "语言教学", "contents": "汉语 法语 英语三语教学",
         "metadata": {"source_id": "1"}},
        {"id": "b", "title": "语言教学", "contents": "汉语法语英语三语教学",
         "metadata": {"source_id": "2"}},
        {"id": "c", "title": "招生", "contents": "物理化学专业招生",
         "metadata": {"source_id": "3"}},
    ]
    corpus.write_text("\n".join(json.dumps(doc, ensure_ascii=False) for doc in documents), encoding="utf-8")
    retriever = ChineseBM25(corpus, tmp_path / "index")
    retriever.build()
    loaded = ChineseBM25(corpus, tmp_path / "index")
    loaded.load()
    assert loaded.search("三语教学", top_k=1, source_id="2")[0]["id"] == "b"
    assert loaded.search("三语教学", top_k=1, source_id="3") == []
    assert loaded.search("zzzznonexistentword") == []
    corpus.write_text(corpus.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus changed"):
        ChineseBM25(corpus, tmp_path / "index").load()
