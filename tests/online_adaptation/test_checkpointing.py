import random

import numpy as np
import pytest
import torch

from rl_training.online_adaptation.agent import AgentConfig, OnlineAdaptationAgent
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
