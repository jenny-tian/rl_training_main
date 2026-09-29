import gc
import json
import pytest
from rl_training.online_adaptation.control_latency import ControlLatencyProbe


def test_probe_attributes_only_gc_overlapping_control_and_keeps_other_callbacks(monkeypatch):
    import rl_training.online_adaptation.control_latency as module
    now = [0.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    monkeypatch.setattr(module.time, 'thread_time', lambda: now[0] / 2)
    existing = list(gc.callbacks)
    probe = ControlLatencyProbe()
    try:
        probe._gc_event('start', {'generation':1})
        now[0] = 1.
        probe.start(now[0])
        now[0] = 1.01
        probe._gc_event('stop', {'generation':1})
        probe.mark('base_policy')
        now[0] = 1.02
        result = probe.finish()
        assert result['gc_ms'] == pytest.approx(10.)
        assert result['wall_before_fallback_ms'] == pytest.approx(20.)
        assert result['thread_cpu_ms'] == pytest.approx(10.)
        assert result['gc_generations'] == [1]
        json.dumps(result, allow_nan=False)
        probe.start(now[0])
        assert probe.finish()['gc_ms'] == 0.
    finally:
        probe.close()
        probe.close()
    assert gc.callbacks == existing
