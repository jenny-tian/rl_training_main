import torch
from types import SimpleNamespace

from rl_training.online_adaptation.rsl_rl_compat import freeze_rsl_rl_models, prepare_rsl_rl_agent_cfg


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
