"""The GPU check must reject native physics failures even with a successful child exit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("curriculum_gpu_check", ROOT / "scripts/tools/source_curriculum_check.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


@pytest.mark.parametrize("message", [
    "PhysX error: PxgCudaDeviceMemoryAllocator failed to allocate memory 23110615040 bytes!",
    "Scene state is corrupted. Simulation cannot continue!",
    "CUDA error: an illegal memory access was encountered",
    "CUDA_ERROR_OUT_OF_MEMORY", "GPU buffer overflow", "buffer capacity exceeded",
])
def test_fatal_native_messages_fail_closed(message):
    assert check.runtime_failure(message)


def test_known_visual_warning_is_not_a_physics_failure():
    assert not check.runtime_failure("[Warning] [omni.usd] Unresolved reference prim path /visuals/FR_FOOT")


@pytest.mark.parametrize("message, expected", [("completed\n", True), ("PhysX error: Scene state is corrupted\n", False)])
def test_supervisor_overrules_successful_child_report(tmp_path, monkeypatch, message, expected):
    import json
    output = tmp_path / "result.json"

    class Child:
        stdout = iter([message])
        def __enter__(self):
            output.write_text(json.dumps({"checks": [{"name": "finite_tensors", "passed": True}]}))
            return self
        def __exit__(self, *args):
            pass
        def wait(self, timeout=None):
            return 0
        def terminate(self):
            pass

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(check.subprocess, "check_output", lambda *args, **kwargs: "24000")
    monkeypatch.setattr(check.subprocess, "Popen", lambda *args, **kwargs: Child())
    assert check.supervise(SimpleNamespace(output=output, num_envs=2048)) == (0 if expected else 1)
    report = json.loads(output.read_text())
    assert report["all_passed"] is expected
    assert report["checks"][-1]["passed"] is expected
    assert report["runtime"]["peak_total_gpu_used_mib_sampled"] == 24000
    assert report["runtime"]["gpu_memory_sample_interval_seconds"] == .5
