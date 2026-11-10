"""CPU checks for Flat retention statistics and the command/step boundary."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_training.ea_lpl.retention import COMMAND_CATEGORIES, RetentionMetrics, categorize_commands


COMMAND = torch.tensor([[0., 0., 0.], [-0.2, 0., 0.], [0., 0.3, 0.], [0., 0., -0.4], [0.2, 0.3, 0.4]])


def update_all(metrics, *, repeats=1):
    command = COMMAND.repeat(repeats, 1)
    n = len(command)
    zero = torch.zeros(n, dtype=torch.bool)
    metrics.update(command, command[:, :2] + torch.tensor([3., 4.]), command[:, 2] + 2., zero, zero, zero)


def add_endings(metrics):
    # Contact + orientation on one episode counts once; timeout-only counts in denominator.
    metrics.update(torch.zeros(3, 3), torch.full((3, 2), float("nan")), torch.full((3,), float("nan")),
                   torch.ones(3, dtype=torch.long), torch.tensor([1, 1, 0]), torch.tensor([1, 0, 0]))


def test_command_categories_follow_actual_values_and_signs():
    tiny = torch.tensor([[1e-12, 0., 0.], [0., -1e-12, 0.], [0., 0., 1e-12], [-0., 0., 0.],
                         [1., 0., -1.], [0., 1., 1.], [1., -1., 0.]])
    assert categorize_commands(COMMAND).tolist() == [0, 1, 2, 3, 4]
    assert categorize_commands(tiny).tolist() == [1, 2, 3, 0, 4, 4, 4]
    assert COMMAND_CATEGORIES == ("standing", "only_x", "only_y", "only_ang", "mixed")


@pytest.mark.parametrize("command", [torch.zeros(3), torch.zeros(2, 4), torch.zeros(2, 3, dtype=torch.long),
                                    torch.tensor([[float("nan"), 0., 0.]]),
                                    torch.tensor([[0., float("inf"), 0.]])])
def test_invalid_commands_fail_closed(command):
    with pytest.raises(ValueError):
        categorize_commands(command)


def test_rms_is_per_xy_component_and_resets_are_excluded_from_every_category():
    metrics = RetentionMetrics()
    update_all(metrics, repeats=2)
    add_endings(metrics)
    result = metrics.summary()
    # sqrt(mean([3**2, 4**2])) differs from planar Euclidean RMS of 5.
    assert result["rms_xy_error"] == pytest.approx(math.sqrt(12.5))
    assert result["rms_yaw_error"] == pytest.approx(2.)
    assert result["tracking_samples"] == 10
    assert result["total_env_steps"] == 13
    assert result["excluded_reset_samples"] == 3
    assert result["ended_episodes"] == 3
    assert result["fall_episodes"] == 2
    assert result["falls_per_episode"] == pytest.approx(2 / 3)
    for values in result["categories"].values():
        assert values["tracking_samples"] == 2
        assert values["rms_xy_error"] == pytest.approx(math.sqrt(12.5))
        assert values["rms_yaw_error"] == pytest.approx(2.)


def test_streaming_rms_uses_sample_counts_instead_of_mean_of_step_rms():
    metrics = RetentionMetrics()
    update_all(metrics)
    # One extra standing sample with exact tracking lowers only standing/global errors.
    metrics.update(torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1), *[torch.zeros(1)] * 3)
    add_endings(metrics)
    result = metrics.summary()
    assert result["rms_xy_error"] == pytest.approx(math.sqrt(125 / 12))
    assert result["rms_yaw_error"] == pytest.approx(math.sqrt(20 / 6))
    assert result["categories"]["standing"]["rms_xy_error"] == pytest.approx(2.5)


@pytest.mark.parametrize("case", ["empty", "no_episodes", "missing_category"])
def test_zero_denominator_fails_closed(case):
    metrics = RetentionMetrics()
    if case == "no_episodes":
        update_all(metrics)
    elif case == "missing_category":
        add_endings(metrics)
        metrics.update(COMMAND[:1], torch.zeros(1, 2), torch.zeros(1), *[torch.zeros(1)] * 3)
    with pytest.raises(ValueError, match="denominator is zero"):
        metrics.summary()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_tracking_sample_fails_closed(bad):
    metrics = RetentionMetrics()
    with pytest.raises(FloatingPointError, match="nonfinite tracking"):
        metrics.update(COMMAND[:1], torch.tensor([[bad, 0.]]), torch.zeros(1), *[torch.zeros(1)] * 3)
    assert metrics.total_env_steps == 0


def test_nonfinite_accumulated_error_fails_closed():
    metrics = RetentionMetrics()
    update_all(metrics)
    add_endings(metrics)
    metrics.squared_error[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="nonfinite accumulated"):
        metrics.summary()


@pytest.mark.parametrize("case", ["bad_flag", "bad_shape", "fall_without_done"])
def test_invalid_episode_flags_fail_closed(case):
    done, contact, orientation = [torch.zeros(5) for _ in range(3)]
    if case == "bad_flag":
        done[0] = float("nan")
    elif case == "bad_shape":
        done = done[:, None]
    else:
        contact[0] = 1
    with pytest.raises(ValueError):
        RetentionMetrics().update(COMMAND, torch.zeros(5, 2), torch.zeros(5), done, contact, orientation)


def load_script():
    path = Path(__file__).resolve().parents[2] / "scripts/tools/stage1_flat_retention.py"
    spec = importlib.util.spec_from_file_location("stage1_flat_retention_test", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_collect_snapshots_command_before_step_and_uses_body_velocity(monkeypatch):
    script = load_script()
    monkeypatch.setattr(script, "NUM_ENVS", 6)
    monkeypatch.setattr(script, "STEPS", 2)
    command = torch.cat((COMMAND, COMMAND[:1])).clone()
    done = torch.tensor([0, 0, 0, 0, 0, 1])
    contact = torch.tensor([0, 0, 0, 0, 0, 1])
    observations = dict(policy=torch.zeros(6, 45), critic=torch.zeros(6, 48))
    data = SimpleNamespace(root_lin_vel_b=torch.zeros(6, 3), root_ang_vel_b=torch.zeros(6, 3))
    manager = SimpleNamespace(active_terms=script.EXPECTED_TERMINATIONS,
                              get_term=lambda _: contact)
    base = SimpleNamespace(num_envs=6, device="cpu", termination_manager=manager,
                           observation_manager=SimpleNamespace(group_obs_dim={"policy": (45,), "critic": (48,)}),
                           command_manager=SimpleNamespace(get_command=lambda _: command), scene={"robot": SimpleNamespace(data=data)})

    class Wrapped:
        unwrapped = base

        def get_observations(self):
            return observations

        def step(self, action):
            assert action.shape == (6, 12)
            data.root_lin_vel_b[:, :2] = command[:, :2]
            data.root_ang_vel_b[:, 2] = command[:, 2]
            # Returned command aliases mutate during step; reset state has arbitrary velocity.
            command[:5] *= -2
            data.root_lin_vel_b[5] = 1000
            return observations, torch.zeros(6), done, {}

    result = script.collect(Wrapped(), lambda _: torch.zeros(6, 12), SimpleNamespace(is_running=lambda: True))
    assert result["rms_xy_error"] == 0
    assert result["rms_yaw_error"] == 0
    assert result["tracking_samples"] == 10
    assert result["excluded_reset_samples"] == 2
    assert result["falls_per_episode"] == 1


@pytest.mark.parametrize("gpu_request,expected", [("auto", 1), ("0", 0), ("1", 1)])
def test_gpu_admission_uses_only_zero_and_one(monkeypatch, gpu_request, expected):
    script = load_script()
    monkeypatch.setattr(script.subprocess, "check_output", lambda *a, **k: "0, 12288\n1, 13000\n2, 99999\n")
    assert script.choose_gpu(gpu_request)[0] == expected


def test_gpu_admission_rejects_insufficient_memory(monkeypatch):
    script = load_script()
    monkeypatch.setattr(script.subprocess, "check_output", lambda *a, **k: "0, 12287\n1, 10000\n")
    with pytest.raises(RuntimeError, match="12288 MiB"):
        script.choose_gpu("auto")


def test_gpu_admission_rejects_other_physical_gpus():
    with pytest.raises(ValueError, match="GPU 0 or 1"):
        load_script().choose_gpu("2")


@pytest.mark.parametrize("failure", [None, "native", "deadline", "missing"])
def test_supervisor_checks_native_output_and_bounds_cleanup_without_gpu(monkeypatch, tmp_path, failure):
    script = load_script()
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"CPU supervision fixture")
    output = tmp_path / "retention.json"
    report = dict(status="passed", completed_steps=script.STEPS,
                  metrics={"rms_xy_error": 0., "rms_yaw_error": 0.},
                  cleanup={"environment": "returned", "application": "returned"})
    # This real subprocess only writes JSON/log fixtures; Isaac and CUDA are never imported.
    stub = tmp_path / "cpu_child.py"
    code = "import json, pathlib, time\n"
    if failure != "missing":
        code += f"pathlib.Path({str(output)!r}).write_text({json.dumps(report)!r})\n"
    if failure == "native":
        code += "print('PhysX error: Scene state is corrupted', flush=True)\n"
    elif failure == "deadline":
        state = dict(name="environment_close", started_monotonic=0, limit_seconds=30)
        code += f"pathlib.Path({str(output.with_suffix('.phase.json'))!r}).write_text({json.dumps(state)!r})\n"
        code += "time.sleep(5)\n"
    stub.write_text(code)
    popen = script.subprocess.Popen
    monkeypatch.setattr(script, "choose_gpu", lambda _: (1, 12288))
    monkeypatch.setattr(script.subprocess, "Popen", lambda command, **kwargs: popen(
        [script.sys.executable, "-u", str(stub)], **kwargs))
    args = SimpleNamespace(output=output, checkpoint=checkpoint,
                           sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), gpu="1")
    assert script.supervise(args) == (0 if failure is None else 1)
    result = json.loads(output.read_text())
    assert result["status"] == ("passed" if failure is None else "failed")
    assert result["runtime"]["physical_gpu"] == 1
    if failure == "native":
        assert "Scene state is corrupted" in result["runtime"]["fatal_messages"][0]
    elif failure == "deadline":
        assert result["runtime"]["deadline_failure"] == "environment_close exceeded 30s"
    elif failure == "missing":
        assert result["checkpoint_sha256"] == args.sha256
        assert result["completed_steps"] == 0
