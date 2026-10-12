import httpx
import numpy as np
import pytest

from app.embedding_api import APIEncoder, EmbeddingError, EmbeddingSettings


def vector(axis):
    result = [0.0] * 1024
    result[axis] = 2.0
    return result


def test_out_of_order_api_rows_are_mapped_to_inputs_and_normalized():
    def respond(request):
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": vector(1)}, {"index": 0, "embedding": vector(0)},
        ]})
    encoder = APIEncoder(EmbeddingSettings("test-key"), httpx.MockTransport(respond))
    values, count = encoder.encode(["汉语", "法语"])
    assert values.shape == (2, 1024) and count == 0
    assert values[0, 0] == 1 and values[1, 1] == 1
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), [1, 1])


def test_malformed_embedding_count_is_not_written_to_index():
    encoder = APIEncoder(EmbeddingSettings("test-key"), httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": []})))
    with pytest.raises(EmbeddingError, match="malformed"):
        encoder.encode(["汉语"])


def test_paid_model_cannot_be_selected_as_fallback():
    with pytest.raises(EmbeddingError, match="only permits"):
        APIEncoder(EmbeddingSettings("test-key", "Pro/BAAI/bge-m3"))


def test_service_error_does_not_disclose_body_or_credentials():
    encoder = APIEncoder(EmbeddingSettings("test-key"), httpx.MockTransport(
        lambda request: httpx.Response(401, text="secret-provider-debug")))
    with pytest.raises(EmbeddingError, match="HTTP 401") as captured:
        encoder.encode(["汉语"])
    assert "secret-provider-debug" not in str(captured.value)
    assert "test-key" not in str(captured.value)
