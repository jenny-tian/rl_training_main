import copy
import random

import numpy as np
import pytest
import torch

from rl_training.online_adaptation.agent import (
    AgentConfig,
    OnlineAdaptationAgent,
    _normalize_replay_state_for_load,
)
from rl_training.online_adaptation.checkpointing import (
    load_experiment_checkpoint,
    save_experiment_checkpoint,
)
from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.sac import SACConfig
from rl_training.online_adaptation.safety import SafetyLimits


def make_agent():
    actor = ResidualActor(2, 1, 4, torch.tensor([0.2]))
    critic = TwinCritic(2, 1, 4)
    return OnlineAdaptationAgent(
        num_envs=1,
        actor=actor,
        critic=critic,
        safety_limits=SafetyLimits(residual_limits=torch.tensor([0.2])),
        config=AgentConfig(
            history_length=3,
            frame_dim=2,
            action_dim=1,
            replay_capacity=4,
            batch_size=1,
            residual_gain=0.5,
            n_step=3,
        ),
        sac_config=SACConfig(),
        device="cpu",
    )


def _assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.parametrize("failure", ["legacy_replay", "adaptation_replay", "nstep", "frozen_replay"])
def test_agent_failed_restore_preserves_training_and_control_state(failure):
    agent = make_agent()
    for optimizer in (agent.updater.actor_optimizer, agent.updater.critic_optimizer,
                      agent.updater.alpha_optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    agent.updater.version = 3
    agent.publisher.publish(agent.updater.actor, agent.updater.version)
    agent.history.append(torch.ones(1, 2))
    agent.safety._previous_residual = torch.full((1, 1), 0.05)
    before = copy.deepcopy(agent.state_dict())
    history_before = agent.history.snapshot().clone()
    safety_before = agent.safety._previous_residual.clone()
    published_actor, published_version = agent.publisher.current()
    published_before = copy.deepcopy(published_actor.state_dict())
    candidate = copy.deepcopy(before)
    for name in ("actor", "critic", "target_critic"):
        for tensor in candidate["updater"][name].values():
            tensor.add_(1.0)
    for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
        for state in candidate["updater"][name]["state"].values():
            for tensor in state.values():
                if isinstance(tensor, torch.Tensor):
                    tensor.add_(2.0)
    candidate["updater"]["version"] = 99
    candidate["updater"]["log_alpha"].add_(3.0)
    candidate["replay"]["reward"].fill_(17.0)
    if failure == "legacy_replay":
        candidate["replay"]["schema_version"] = 1
        candidate["replay"].pop("requested_residual_action")
    elif failure == "adaptation_replay":
        candidate["adaptation_replay"]["schema_version"] = 999
    elif failure == "nstep":
        candidate["nstep"]["queues"] = "invalid"
    else:
        candidate["replay"]["frozen"] = True
    with pytest.raises(ValueError):
        agent.load_state_dict(candidate)
    _assert_state_equal(agent.state_dict(), before)
    _assert_state_equal(agent.history.snapshot(), history_before)
    _assert_state_equal(agent.safety._previous_residual, safety_before)
    published_after, version_after = agent.publisher.current()
    assert version_after == published_version
    _assert_state_equal(published_after.state_dict(), published_before)
    _assert_state_equal(published_after.state_dict(), agent.updater.actor.state_dict())


def test_experiment_checkpoint_round_trip_restores_state_and_rng(tmp_path):
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    path = tmp_path / "experiment.pt"
    state = {
        "protocol_sha256": "protocol-hash",
        "updater": {"version": 7, "weight": torch.tensor([1.0, 2.0])},
        "curriculum": {"stage": 2},
    }
    save_experiment_checkpoint(path, state)
    expected_random = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)

    restored = load_experiment_checkpoint(
        path, expected_protocol_sha256="protocol-hash"
    )

    assert restored["updater"]["version"] == 7
    torch.testing.assert_close(
        restored["updater"]["weight"], torch.tensor([1.0, 2.0])
    )
    assert restored["curriculum"] == {"stage": 2}
    assert random.random() == expected_random
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)
    assert (tmp_path / "experiment.pt.sha256").exists()


def test_frozen_evaluation_can_load_checkpoint_without_overwriting_rng(tmp_path):
    path = tmp_path / "experiment.pt"
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    save_experiment_checkpoint(path, {"protocol_sha256": "protocol-hash"})

    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    expected_random = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)

    load_experiment_checkpoint(
        path,
        expected_protocol_sha256="protocol-hash",
        restore_rng=False,
    )

    assert random.random() == expected_random
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)


def test_restore_rng_state_normalizes_cuda_states_to_cpu_byte_tensors(monkeypatch):
    captured = []

    class CudaResidentState:
        def detach(self):
            return self

        def to(self, *, device, dtype=torch.uint8):
            return torch.tensor([1, 2, 3], device=device, dtype=dtype)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda states: captured.extend(states),
    )

    from rl_training.online_adaptation.checkpointing import restore_rng_state

    restore_rng_state(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "cuda_device_count": 1,
            "torch_cuda": (CudaResidentState(),),
        }
    )

    assert len(captured) == 1
    assert captured[0].device.type == "cpu"
    assert captured[0].dtype == torch.uint8


def test_normalize_replay_state_moves_storage_tensors_to_cpu():
    class CudaResidentState:
        def detach(self):
            return self

        def to(self, *, device):
            return torch.ones(2, device=device)

    normalized = _normalize_replay_state_for_load(
        {
            "capacity": 4,
            "history": CudaResidentState(),
            "reward": CudaResidentState(),
        }
    )

    assert normalized["capacity"] == 4
    assert normalized["history"].device.type == "cpu"
    assert normalized["reward"].device.type == "cpu"


def test_checkpoint_rejects_checksum_and_protocol_mismatches(tmp_path):
    path = tmp_path / "experiment.pt"
    save_experiment_checkpoint(path, {"protocol_sha256": "protocol-hash"})

    with pytest.raises(ValueError, match="protocol"):
        load_experiment_checkpoint(path, expected_protocol_sha256="other-hash")

    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_experiment_checkpoint(path, expected_protocol_sha256="protocol-hash")


def test_failed_atomic_save_preserves_previous_good_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "experiment.pt"
    save_experiment_checkpoint(
        path, {"protocol_sha256": "protocol-hash", "updater": {"version": 1}}
    )
    import rl_training.online_adaptation.checkpointing as checkpointing

    original_replace = checkpointing.os.replace

    def fail_before_checkpoint_replace(source, destination):
        if destination == path:
            raise OSError("simulated interrupted save")
        return original_replace(source, destination)

    monkeypatch.setattr(checkpointing.os, "replace", fail_before_checkpoint_replace)
    with pytest.raises(OSError, match="interrupted"):
        save_experiment_checkpoint(
            path,
            {"protocol_sha256": "protocol-hash", "updater": {"version": 2}},
        )
    monkeypatch.setattr(checkpointing.os, "replace", original_replace)

    restored = load_experiment_checkpoint(
        path, expected_protocol_sha256="protocol-hash"
    )
    assert restored["updater"]["version"] == 1


def test_failed_checksum_replace_rolls_back_previous_good_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "experiment.pt"
    save_experiment_checkpoint(
        path, {"protocol_sha256": "protocol-hash", "updater": {"version": 1}}
    )
    import rl_training.online_adaptation.checkpointing as checkpointing

    original_replace = checkpointing.os.replace
    checksum_path = tmp_path / "experiment.pt.sha256"

    def fail_checksum_replace(source, destination):
        if destination == checksum_path:
            raise OSError("simulated checksum replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(checkpointing.os, "replace", fail_checksum_replace)
    with pytest.raises(OSError, match="checksum replace"):
        save_experiment_checkpoint(
            path,
            {"protocol_sha256": "protocol-hash", "updater": {"version": 2}},
        )
    monkeypatch.setattr(checkpointing.os, "replace", original_replace)

    restored = load_experiment_checkpoint(
        path, expected_protocol_sha256="protocol-hash"
    )
    assert restored["updater"]["version"] == 1


def test_agent_state_round_trip_republishes_actor_and_resets_control_state():
    source = make_agent()
    source.updater.version = 4
    with torch.no_grad():
        next(source.updater.actor.parameters()).fill_(0.25)
    source.history.append(torch.ones(1, 2))
    source.safety._previous_residual = torch.ones(1, 1)
    state = source.state_dict()
    restored = make_agent()
    restored.history.append(torch.full((1, 2), 9.0))
    restored.safety._previous_residual = torch.full((1, 1), 9.0)

    restored.load_state_dict(state)

    published_actor, version = restored.publisher.current()
    assert version == 4
    for actual, expected in zip(
        published_actor.parameters(), restored.updater.actor.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(restored.history.snapshot(), torch.zeros(1, 3, 2))
    assert restored.safety._previous_residual is None
    assert restored._pending is None
