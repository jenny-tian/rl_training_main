import copy

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from rl_training.lite3_joint_conventions import (
    LITE3_SAFE_ACTION_UPPER,
    clip_lite3_policy_action,
)
from rl_training.online_adaptation.lora import inject_lora, mark_only_lora_trainable
from rl_training.online_adaptation.online_session import OnlinePPOSession, SessionCounters
from rl_training.online_adaptation.ppo_context_policy import ContextGatedPolicy, migrate_context_policy
from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater, PPOConfig
from rl_training.online_adaptation.ppo_runtime_config import apply_ppo_runtime_config, read_ppo_runtime_config
from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher
from rl_training.online_adaptation.rsl_rl_compat import RslMLPPolicyAdapter


class GaussianModule(nn.Module):
    def __init__(self, parameter_name="log_std_param"):
        super().__init__()
        value = torch.full((12,), 0.02)
        self.register_parameter(parameter_name, nn.Parameter(value.log() if parameter_name == "log_std_param" else value))
        self._distribution = None

    def update(self, mean):
        std = self.log_std_param.exp() if hasattr(self, "log_std_param") else self.std_param
        self._distribution = Normal(mean, std)


class RslActor(nn.Module):
    def __init__(self, std_name="log_std_param"):
        super().__init__()
        self.obs_groups = ["policy"]
        self.distribution = GaussianModule(std_name)
        self.mlp = nn.Sequential(nn.Linear(45, 16), nn.ELU(), nn.Linear(16, 12))
        self.observed_widths = []

    def forward(self, observations, stochastic_output=False):
        value = observations["policy"]
        self.observed_widths.append(value.shape[-1])
        return self.mlp(value)


def base_policy(std_name="log_std_param"):
    model = RslActor(std_name)
    inject_lora(model, rank=2)
    mark_only_lora_trainable(model)
    model.eval()
    return RslMLPPolicyAdapter(model, action_transform=clip_lite3_policy_action)


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            equal_tree(value, reference)
    else:
        assert actual == expected


def legacy_session(*, warmup=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(92)
        policy = base_policy()
        # The migration must not assume any critic input width or architecture.
        updater = OnlinePPOUpdater(policy, nn.Linear(7, 1), PPOConfig(target_kl=None), policy_version=5)
        updater.actor_optimizer = torch.optim.Adam(
            [p for p in policy.parameters() if p.requires_grad], lr=1.5e-4, amsgrad=True,
        )
        obs = torch.randn(8, 45)

        def actor_step():
            distribution = policy.get_distribution(obs)
            actions = distribution.mean.detach() + 0.01
            updater.actor_optimizer.zero_grad()
            (-distribution.log_prob(actions).sum(-1).mean()).backward()
            updater.actor_optimizer.step()

        if warmup:
            actor_step()
        session = OnlinePPOSession(updater, rollout_capacity=16)
        if warmup:
            actor_step()
        updater.policy_version = 6
        session.counters = SessionCounters(
            transitions_collected=32, update_attempts=3, accepted_updates=2, rejected_updates=1,
        )
        state = session.state_dict()
        state["has_unverified_candidate"] = True
        state["audit_metadata"] = {"keep": [2, 9]}
        return state


@pytest.mark.parametrize("std_name", ["log_std_param", "std_param"])
def test_initial_head_preserves_base_mean_likelihood_entropy_and_single_std(std_name):
    base = base_policy(std_name)
    wrapped = ContextGatedPolicy(base)
    observations = torch.randn(9, 45)
    context = torch.randn(9, 3) * 100
    extended = torch.cat((observations, context), dim=-1)
    before = observations.clone()
    with torch.no_grad():
        old_distribution = base.get_distribution(observations)
        actions = old_distribution.sample()
        old_log_prob, old_entropy = base.evaluate_actions(observations, actions)
        distribution = wrapped.get_distribution(extended)
        new_log_prob, new_entropy = wrapped.evaluate_actions(extended, actions)
        torch.testing.assert_close(wrapped(extended), old_distribution.mean, rtol=0, atol=0)
        torch.testing.assert_close(distribution.mean, old_distribution.mean, rtol=0, atol=0)
        torch.testing.assert_close(distribution.scale, old_distribution.scale, rtol=0, atol=0)
        torch.testing.assert_close(new_log_prob, old_log_prob, rtol=0, atol=0)
        torch.testing.assert_close(new_entropy, old_entropy, rtol=0, atol=0)
    assert wrapped.base_policy is base
    assert set(base.model.observed_widths) == {45}
    torch.testing.assert_close(observations, before, rtol=0, atol=0)
    std_names = [name for name, _ in wrapped.named_parameters() if name.rsplit(".", 1)[-1] in {"log_std_param", "std_param"}]
    assert std_names == ["base_policy.model.distribution." + std_name]
    trainables = [name for name, p in wrapped.named_parameters() if p.requires_grad]
    assert trainables[-2:] == ["context_A.weight", "context_B.weight"]
    assert all(name.startswith("base_policy.") for name in trainables[:-2])
    assert torch.count_nonzero(wrapped.context_A.weight) > 0
    assert torch.count_nonzero(wrapped.context_B.weight) == 0


def test_three_context_branches_sum_and_apply_canonical_action_limits():
    policy = ContextGatedPolicy(base_policy())
    observations = torch.zeros(2, 45)
    observations[:, 0] = 1
    with torch.no_grad():
        policy.context_A.weight.zero_()
        policy.context_A.weight[0, 0] = 1
        policy.context_B.weight.zero_()
        policy.context_B.weight[:12, 0] = 1
        policy.context_B.weight[12:24, 0] = 2
        policy.context_B.weight[24:, 0] = -1
        base_mean = policy.base_policy.mean_action(observations)
        context = torch.tensor([[0.1, 0.2, 0.3], [10000.0, 0.0, 0.0]])
        mean = policy.mean_action(torch.cat((observations, context), dim=-1))
    torch.testing.assert_close(mean[0], base_mean[0] + 0.2)
    torch.testing.assert_close(mean[1], torch.tensor(LITE3_SAFE_ACTION_UPPER), rtol=0, atol=0)


def test_nonzero_context_teaches_head_and_zero_context_anchor_has_zero_head_gradient():
    policy = ContextGatedPolicy(base_policy())
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-3)
    observations = torch.randn(8, 45)
    context = torch.tensor([[0.3, -0.2, 0.7]]).expand(8, -1)
    extended = torch.cat((observations, context), dim=-1)
    target = policy.base_policy.mean_action(observations).detach() + 0.2
    (policy.mean_action(extended) - target).square().mean().backward()
    assert policy.context_B.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(policy.context_A.weight.grad) == 0  # B=0 at initialization.
    optimizer.step()
    assert policy.context_B.weight.detach().abs().sum() > 0
    optimizer.zero_grad()
    (policy.mean_action(extended) - target).square().mean().backward()
    assert policy.context_A.weight.grad.abs().sum() > 0
    optimizer.zero_grad()
    zero_context = torch.cat((observations, torch.zeros_like(context)), dim=-1)
    mean = policy.mean_action(zero_context)
    torch.testing.assert_close(mean, policy.base_policy.mean_action(observations), rtol=0, atol=0)
    (mean - target).square().mean().backward()
    assert torch.count_nonzero(policy.context_A.weight.grad) == 0
    assert torch.count_nonzero(policy.context_B.weight.grad) == 0


def test_migration_preserves_both_legacy_policies_and_adam_with_same_zero_head():
    original = legacy_session()
    before = copy.deepcopy(original)
    target = ContextGatedPolicy(base_policy())
    migrated = migrate_context_policy(original, target_policy=target)
    observations = torch.randn(6, 45)
    extended = torch.cat((observations, torch.randn(6, 3)), dim=-1)
    base_names = [name for name, p in target.base_policy.named_parameters() if p.requires_grad]

    for branch_name in ("updater", "stable_state"):
        old, new = original[branch_name], migrated[branch_name]
        restored_base = base_policy()
        restored_base.load_state_dict(old["policy"])
        restored = ContextGatedPolicy(base_policy())
        restored.load_state_dict(new["policy"])
        with torch.no_grad():
            old_dist = restored_base.get_distribution(observations)
            actions = old_dist.sample()
            new_dist = restored.get_distribution(extended)
            torch.testing.assert_close(new_dist.mean, old_dist.mean, rtol=0, atol=0)
            torch.testing.assert_close(new_dist.log_prob(actions), old_dist.log_prob(actions), rtol=0, atol=0)
        for name, value in old["policy"].items():
            torch.testing.assert_close(new["policy"]["base_policy." + name], value, rtol=0, atol=0)
        assert torch.count_nonzero(new["policy"]["context_B.weight"]) == 0
        old_ids = old["actor_optimizer"]["param_groups"][0]["params"]
        new_ids = new["actor_optimizer"]["param_groups"][0]["params"]
        assert len(old_ids) == len(base_names)
        assert new_ids == old_ids + [max(old_ids) + 1, max(old_ids) + 2]
        equal_tree(new["actor_optimizer"]["state"], old["actor_optimizer"]["state"])
        for name in ("critic", "critic_optimizer", "policy_version"):
            equal_tree(new[name], old[name])
        assert new["actor_optimizer"]["param_groups"][0]["lr"] == 1.5e-4
    torch.testing.assert_close(
        migrated["updater"]["policy"]["context_A.weight"], migrated["stable_state"]["policy"]["context_A.weight"],
        rtol=0, atol=0,
    )
    assert migrated["updater"]["policy"]["context_A.weight"].data_ptr() != migrated["stable_state"]["policy"]["context_A.weight"].data_ptr()
    assert migrated["counters"] == before["counters"] and migrated["has_unverified_candidate"] is True
    migrated["audit_metadata"]["keep"].append(10)
    migrated["updater"]["policy"]["context_A.weight"].zero_()
    equal_tree(original, before)


def test_restored_adam_steps_old_and_new_parameters_and_trained_head_migration_is_idempotent():
    original = legacy_session()
    policy = ContextGatedPolicy(base_policy())
    migrated = migrate_context_policy(original, target_policy=policy)
    updater = OnlinePPOUpdater(policy, nn.Linear(7, 1), PPOConfig(target_kl=None))
    session = OnlinePPOSession(updater, rollout_capacity=16)
    session.load_state_dict(migrated)
    first_base = next(p for p in policy.base_policy.parameters() if p.requires_grad)
    old_step = updater.actor_optimizer.state[first_base]["step"].item()
    observations = torch.randn(8, 48)
    distribution = policy.get_distribution(observations)
    updater.actor_optimizer.zero_grad()
    (-distribution.log_prob(distribution.mean.detach() + 0.01).sum(-1).mean()).backward()
    updater.actor_optimizer.step()
    assert updater.actor_optimizer.state[first_base]["step"].item() == old_step + 1
    assert updater.actor_optimizer.state[policy.context_A.weight]["step"].item() == 1
    assert updater.actor_optimizer.state[policy.context_B.weight]["step"].item() == 1
    assert torch.count_nonzero(policy.context_B.weight) > 0
    trained = session.state_dict()
    restored = migrate_context_policy(trained, target_policy=ContextGatedPolicy(base_policy()))
    equal_tree(restored, trained)  # Nonzero learned B is retained, not reinitialized.
    assert restored is not trained
    effective = read_ppo_runtime_config(session)
    assert effective["action_std"] == pytest.approx(0.02)
    before_head = policy.context_B.weight.detach().clone()
    changed = apply_ppo_runtime_config(session, action_std=0.03, actor_lr=2e-4, critic_lr=4e-4)
    assert changed["action_std"] == pytest.approx(0.03)
    torch.testing.assert_close(policy.context_B.weight, before_head, rtol=0, atol=0)


def test_uninitialized_adam_and_nonzero_parameter_ids_migrate_without_inventing_moments():
    original = legacy_session(warmup=False)
    for branch_name in ("updater", "stable_state"):
        optimizer = original[branch_name]["actor_optimizer"]
        group = optimizer["param_groups"][0]
        group["params"] = [value + 70 for value in group["params"]]
        group["param_names"] = [name for name in original[branch_name]["policy"] if name.endswith((".lora_A", ".lora_B"))]
    migrated = migrate_context_policy(original, target_policy=ContextGatedPolicy(base_policy()))
    for branch_name in ("updater", "stable_state"):
        optimizer = migrated[branch_name]["actor_optimizer"]
        assert optimizer["state"] == {}
        group = optimizer["param_groups"][0]
        assert group["params"] == [70, 71, 72, 73, 74, 75]
        assert group["param_names"][-2:] == ["context_A.weight", "context_B.weight"]
        assert all(name.startswith("base_policy.") for name in group["param_names"][:-2])


def test_publisher_supports_frozen_context_policy_and_atomic_mean_changes():
    policy = ContextGatedPolicy(base_policy())
    observations = torch.randn(5, 48)
    with torch.no_grad():
        policy.context_B.weight.fill_(0.01)
        expected = policy.mean_action(observations).clone()
        policy.get_distribution(observations)
    publisher = DoubleBufferedActorPublisher(policy)
    publisher.publish(policy, 7)
    published, version = publisher.current()
    assert version == 7
    assert not any(p.requires_grad for p in published.parameters())
    with torch.no_grad():
        torch.testing.assert_close(published.mean_action(observations), expected, rtol=0, atol=0)
        policy.context_B.weight.add_(0.05)
        torch.testing.assert_close(published.mean_action(observations), expected, rtol=0, atol=0)
        publisher.publish(policy, 8)
        updated, version = publisher.current()
        assert version == 8
        torch.testing.assert_close(updated.mean_action(observations), policy.mean_action(observations), rtol=0, atol=0)
        assert not torch.equal(updated.mean_action(observations), expected)


@pytest.mark.parametrize("shape", [(48,), (2, 45), (2, 49)])
def test_context_policy_rejects_wrong_input_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        ContextGatedPolicy(base_policy()).mean_action(torch.zeros(shape))


@pytest.mark.parametrize("damage", [
    "pending", "extra_policy_key", "bad_policy_shape", "nonfinite_policy", "extra_optimizer_param",
    "optimizer_order", "wrong_moment", "multiple_groups", "wrong_param_names", "source_parameter_order",
])
def test_migration_rejects_damaged_or_ambiguous_checkpoints_without_mutating_input(damage):
    original = legacy_session()
    branch = original["updater"]
    optimizer = branch["actor_optimizer"]
    group = optimizer["param_groups"][0]
    if damage == "pending":
        original["pending_rollout"] = {}
    elif damage == "extra_policy_key":
        branch["policy"]["unknown.weight"] = torch.zeros(2, 2)
    elif damage == "bad_policy_shape":
        branch["policy"]["model.mlp.0.lora_A"] = torch.zeros(2, 46)
    elif damage == "nonfinite_policy":
        branch["policy"]["model.distribution.log_std_param"][0] = float("nan")
    elif damage == "extra_optimizer_param":
        group["params"].append(99)
    elif damage == "optimizer_order":
        group["params"][0], group["params"][1] = group["params"][1], group["params"][0]
    elif damage == "wrong_moment":
        optimizer["state"][group["params"][0]]["exp_avg"] = torch.zeros(2, 46)
    elif damage == "multiple_groups":
        optimizer["param_groups"].append(copy.deepcopy(group))
    elif damage == "wrong_param_names":
        group["param_names"] = ["wrong"] * len(group["params"])
    else:
        branch["policy"] = dict(reversed(list(branch["policy"].items())))
    before = copy.deepcopy(original)
    with pytest.raises(ValueError):
        migrate_context_policy(original, target_policy=ContextGatedPolicy(base_policy()))
    if damage != "nonfinite_policy":
        equal_tree(original, before)
    else:
        assert torch.isnan(original["updater"]["policy"]["model.distribution.log_std_param"][0])


def test_migration_rejects_non_lora_base_trainables():
    target = ContextGatedPolicy(base_policy())
    target.base_policy.model.distribution.log_std_param.requires_grad_(True)
    with pytest.raises(ValueError, match="only.*LoRA"):
        migrate_context_policy(legacy_session(), target_policy=target)
