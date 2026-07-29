import pytest
import torch
from types import SimpleNamespace

from rl_training.online_adaptation import rsl_rl_compat
from rl_training.online_adaptation.rsl_rl_compat import (
    freeze_rsl_rl_models,
    prepare_rsl_rl_agent_cfg,
)


def test_split_rsl_rl_endings_marks_terminal_rows_without_bootstrap():
    terminated, truncated, bootstrap_mask = rsl_rl_compat.split_rsl_rl_endings(
        torch.tensor([True, False]),
        {"time_outs": torch.tensor([False, False])},
    )

    assert torch.equal(terminated, torch.tensor([[True], [False]]))
    assert torch.equal(truncated, torch.tensor([[False], [False]]))
    assert torch.equal(bootstrap_mask, torch.tensor([[False], [True]]))


def test_split_rsl_rl_endings_marks_timeout_as_bootstrapping_truncation():
    terminated, truncated, bootstrap_mask = rsl_rl_compat.split_rsl_rl_endings(
        torch.tensor([True]),
        {"time_outs": torch.tensor([True])},
    )

    assert torch.equal(terminated, torch.tensor([[False]]))
    assert torch.equal(truncated, torch.tensor([[True]]))
    assert torch.equal(bootstrap_mask, torch.tensor([[True]]))


@pytest.mark.parametrize(
    "extras",
    [
        {},
        {"time_outs": [False]},
        {"time_outs": torch.tensor([False, False])},
        {"time_outs": torch.tensor([0.5])},
        {"time_outs": torch.tensor([float("nan")])},
    ],
)
def test_split_rsl_rl_endings_rejects_missing_or_malformed_time_outs(extras):
    with pytest.raises(RuntimeError, match="time_outs"):
        rsl_rl_compat.split_rsl_rl_endings(torch.tensor([True]), extras)


def test_prepare_rsl_rl_agent_cfg_uses_installed_version_converter():
    calls = []
    agent_cfg = object()
    converted_cfg = object()

    def converter(candidate, installed_version):
        calls.append((candidate, installed_version))
        return converted_cfg

    result = prepare_rsl_rl_agent_cfg(
        agent_cfg,
        installed_version="5.0.1",
        converter=converter,
    )

    assert result is converted_cfg
    assert calls == [(agent_cfg, "5.0.1")]


def test_freeze_rsl_rl_models_supports_separate_actor_and_critic():
    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(2, 1)
    runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic))

    frozen = freeze_rsl_rl_models(runner)

    assert frozen == (actor, critic)
    assert all(not parameter.requires_grad for module in frozen for parameter in module.parameters())
    assert all(not module.training for module in frozen)
