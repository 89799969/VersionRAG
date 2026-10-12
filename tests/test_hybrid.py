import pytest

from app.hybrid import rrf_fuse


def hit(passage_id, score):
    return {"id": passage_id, "contents": f"text-{passage_id}", "score": score}


def test_rrf_uses_ranks_not_raw_scores_and_rewards_overlap():
    channels = {"bm25": [hit("a", 1000), hit("b", 500)],
                "dense": [hit("c", 0.9), hit("b", 0.8)]}
    result = rrf_fuse(channels, 3)
    assert result[0]["id"] == "b"
    assert result[0]["channel_ranks"] == {"bm25": 2, "dense": 2}
    channels["bm25"][0]["score"] = -100
    assert [row["id"] for row in rrf_fuse(channels, 3)] == [row["id"] for row in result]


def test_duplicate_from_one_channel_cannot_inflate_fusion_score():
    one = rrf_fuse({"bm25": [hit("a", 1)]}, 1)[0]["score"]
    duplicated = rrf_fuse({"bm25": [hit("a", 1), hit("a", 2)]}, 1)[0]["score"]
    assert one == duplicated
    with pytest.raises(ValueError, match="conflicting"):
        rrf_fuse({"bm25": [hit("a", 1)], "dense": [{"id": "a", "contents": "different"}]})
