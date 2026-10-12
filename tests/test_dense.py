import numpy as np
import json
import pytest

from app.dense import DenseRetriever, ranked_rows


def test_cosine_ranking_applies_source_filter_before_topk():
    vectors = np.array([[1, 0], [0.8, 0.6], [-1, 0]], dtype=np.float32)
    query = np.array([1, 0], dtype=np.float32)
    assert ranked_rows(vectors, query, 2) == [(0, 1.0), (1, float(np.float32(0.8)))]
    assert ranked_rows(vectors, query, 1, np.array([1, 2]))[0][0] == 1
    assert ranked_rows(vectors, query, 1, np.array([], dtype=np.int64)) == []


def test_interrupted_embedding_index_resumes_saved_rows_and_detects_corruption(tmp_path):
    class Encoder:
        dimension = 3

        def __init__(self, fail_second=False):
            self.calls = []
            self.fail_second = fail_second

        def metadata(self):
            return {"dimension": 3, "model": "mock-only-in-tests"}

        def encode(self, texts, query=False):
            self.calls.append(texts)
            if self.fail_second and len(self.calls) == 2:
                raise RuntimeError("simulated provider interruption")
            return np.array([[1, 0, 0] for text in texts], dtype=np.float32), 0

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"id": str(index), "title": "fixture", "contents": text,
                                              "metadata": {"source_id": "1"}})
                                for index, text in enumerate(["a", "b", "c"])), encoding="utf-8")
    index_path = tmp_path / "index"
    with pytest.raises(RuntimeError, match="interruption"):
        DenseRetriever(corpus, index_path, Encoder(True)).build(batch_size=2)
    assert json.loads((index_path / "building.json").read_text())["completed"] == 2
    encoder = Encoder()
    DenseRetriever(corpus, index_path, encoder).build(batch_size=2)
    assert encoder.calls == [["c"]]
    assert not (index_path / "building.json").exists()
    DenseRetriever(corpus, index_path, Encoder()).load()
    with (index_path / "vectors.npy").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        DenseRetriever(corpus, index_path, Encoder()).load()
