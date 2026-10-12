import httpx
import pytest

from app.generation import GenerationError
from scripts import evaluate_comparison as experiment


def test_billing_failure_stops_further_requests_in_the_same_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "OUT", tmp_path)
    calls = []
    transport = experiment.LoggedTransport()
    transport.inner.close()
    def respond(request):
        calls.append(request.url.host)
        return httpx.Response(402)
    transport.inner = httpx.MockTransport(respond)
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    assert transport.handle_request(request).status_code == 402
    with pytest.raises(GenerationError, match="remaining"):
        transport.handle_request(request)
    assert calls == ["api.deepseek.com"]
