from __future__ import annotations

import hashlib

import pytest
import torch

from rsl_rl.models import MLPModel
from tensordict import TensorDict

from rl_training.source_scan.frozen_actor import FrozenSourceActor
from rl_training.source_scan.source_training import TransitionBudget, load_source_actor_only


def test_transition_budget_rounds_down_and_never_overshoots():
    budget = TransitionBudget(num_envs=120, rollout_length=24, max_transitions=2_097_152)
    assert budget.transitions_per_update == 2_880
    assert budget.max_updates == 728
    assert budget.committed_transitions == 2_096_640
    assert budget.unused_transitions == 512
    assert budget.transitions_after(364) == 1_048_320
    budget.assert_update_allowed(727)
    with pytest.raises(RuntimeError, match="exhausted"):
        budget.assert_update_allowed(728)


def test_transition_budget_rejects_invalid_or_partial_progress():
    with pytest.raises(ValueError):
        TransitionBudget(num_envs=120, rollout_length=24, max_transitions=0)
    budget = TransitionBudget(num_envs=2, rollout_length=3, max_transitions=7)
    assert budget.max_updates == 1
    assert budget.committed_transitions <= budget.max_transitions
    with pytest.raises(ValueError):
        budget.transitions_after(2)
    with pytest.raises(ValueError):
        budget.assert_update_allowed(-1)


def make_actor(activation="elu"):
    obs = TensorDict({"policy": torch.zeros(2, 45)}, batch_size=[2])
    return MLPModel(
        obs, {"actor": ["policy"]}, "actor", 12,
        hidden_dims=[512, 256, 128], activation=activation,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "std_type": "log", "init_std": 0.3,
        },
    )


def checkpoint_fixture(tmp_path, *, nonfinite=False):
    actor = make_actor()
    state = actor.state_dict()
    if nonfinite:
        state["mlp.6.bias"][0] = float("nan")
    path = tmp_path / "fixture.pt"
    torch.save({
        "actor_state_dict": state,
        "critic_state_dict": {"must_not_load": torch.ones(1)},
        "optimizer_state_dict": {"must_not_load": True},
        "iter": 1998, "infos": None,
    }, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), state


def test_actor_initialization_preserves_mean_and_rng_and_remains_trainable(tmp_path):
    path, digest, state = checkpoint_fixture(tmp_path)
    actor = make_actor()
    critic = torch.nn.Linear(45, 1)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=0.001)
    critic_before = {key: value.clone() for key, value in critic.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()
    report = load_source_actor_only(actor, path, expected_sha256=digest, action_std=1.0)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert not optimizer.state
    assert all(torch.equal(value, critic_before[key]) for key, value in critic.state_dict().items())
    assert report["checkpoint_iteration_ignored"] == 1998
    assert torch.equal(actor.distribution.log_std_param, torch.zeros(12))
    obs = TensorDict({"policy": torch.randn(9, 45)}, batch_size=[9])
    expected = FrozenSourceActor(state)(obs)
    assert torch.equal(actor(obs), expected)
    loss = actor(obs).square().mean()
    loss.backward()
    assert actor.mlp[0].weight.grad is not None
    optimizer.step()
    assert not torch.equal(actor.mlp[0].weight, state["mlp.0.weight"])


@pytest.mark.parametrize("failure", ["hash", "nonfinite", "architecture", "activation", "std"])
def test_bad_initialization_rejected_before_actor_mutation(tmp_path, failure):
    path, digest, _ = checkpoint_fixture(tmp_path, nonfinite=failure == "nonfinite")
    actor = make_actor(activation="relu" if failure == "activation" else "elu")
    if failure == "architecture":
        actor.mlp[6] = torch.nn.Linear(128, 11)
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    with pytest.raises((ValueError, FloatingPointError)):
        load_source_actor_only(
            actor, path, expected_sha256="0" * 64 if failure == "hash" else digest,
            action_std=float("nan") if failure == "std" else 1.0,
        )
    assert all(torch.equal(value, before[key]) for key, value in actor.state_dict().items())


def test_adaptation_initialization_preserves_nonuniform_std_actor_and_rng(tmp_path):
    from types import SimpleNamespace as NS
    from rl_training.source_scan.adaptation_profile import (
        SOURCE_ADAPTATION_PROFILE, build_source_adaptation_rewards, initialize_source_adaptation_actor)

    path, _, _ = checkpoint_fixture(tmp_path)
    saved = torch.load(path, weights_only=True)
    saved["actor_state_dict"]["distribution.log_std_param"] = torch.linspace(0.2, 0.45, 12).log()
    torch.save(saved, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    actor = make_actor()
    cfg = NS(source_reward_profile=SOURCE_ADAPTATION_PROFILE, reward_gait_level=1.0,
             rewards=build_source_adaptation_rewards(NS, NS))
    rng = torch.random.get_rng_state().clone()
    report = initialize_source_adaptation_actor(actor, path, cfg, expected_sha256=digest)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(v, saved["actor_state_dict"][k]) for k, v in actor.state_dict().items())
    assert report["action_std_mode"] == "checkpoint" and len(report["action_std"]) == 12
    assert not report["critic_restored"] and not report["optimizer_restored"]
    assert all(p.requires_grad for p in actor.parameters())
    with pytest.raises(ValueError, match="preserves checkpoint"):
        initialize_source_adaptation_actor(actor, path, cfg, action_std=1.0)


@pytest.mark.parametrize("std", [1e-100, 1e100, None])
def test_unrepresentable_exploration_is_rejected_before_actor_mutation(tmp_path, std):
    path, digest, _ = checkpoint_fixture(tmp_path)
    if std is None:
        saved = torch.load(path, weights_only=True)
        saved["actor_state_dict"]["distribution.log_std_param"][0] = 1000.0
        torch.save(saved, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    actor = make_actor()
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    with pytest.raises(FloatingPointError, match="exploration std"):
        load_source_actor_only(actor, path, expected_sha256=digest, action_std=std)
    assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())
