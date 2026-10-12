"""Run the frozen test with recorded pacing after observed development 429s."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import sha256
from scripts import optimize_retrieval as experiment


class PacedService(experiment.Service):
    def __init__(self):
        super().__init__()
        self.previous_started = None

    def rerank(self, pool, variant):
        started_wait = time.perf_counter()
        if self.previous_started is not None:
            delay = 1.35 - (time.perf_counter() - self.previous_started)
            if delay > 0:
                time.sleep(delay)
        pacing_wait = (time.perf_counter() - started_wait) * 1000
        self.previous_started = time.perf_counter()
        result = super().rerank(pool, variant)
        result['pacing_wait_ms'] = pacing_wait
        return result


if __name__ == '__main__':
    experiment.protect(experiment.OUT / 'test_runtime_policy.json', {
        'minimum_start_interval_seconds': 1.35,
        'scope': 'All three test variants, including warmups. Pacing before the HTTP timer; wait saved per measured row.',
        'reason': '88 recorded development HTTP 429 events; keep the frozen metric configuration and slow scheduling.',
        'launcher_sha256': sha256(Path(__file__)),
        'selection_sha256': sha256(experiment.OUT / 'selection.json'),
    })
    experiment.Service = PacedService
    experiment.run_phase('test')
