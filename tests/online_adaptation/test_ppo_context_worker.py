import copy

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.lora import inject_lora, mark_only_lora_trainable
from rl_training.online_adaptation.online_session import OnlinePPOSession
from rl_training.online_adaptation.ppo_context_policy import ContextGatedPolicy, CONTEXT_POLICY_OBSERVATION_SCHEMA
from rl_training.online_adaptation.ppo_critic_context import build_critic_observation
from rl_training.online_adaptation.ppo_odometry import PlanarProprioOdometry
from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater
from rl_training.online_adaptation.rsl_rl_compat import RslMLPPolicyAdapter
from scripts.reinforcement_learning.online_adaptation.train_lora_ppo import (
    BASE_POLICY_OBSERVATION_SCHEMA,
    CONTEXT_CRITIC_OBSERVATION_SCHEMA,
    CRITIC_OBSERVATION_SCHEMA,
    _append_odometry_context,
    _prepare_checkpoint_session,
)


class _Gaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_std_param = nn.Parameter(torch.full((12,), -3.0))

    def update(self, mean):
        self._distribution = Normal(mean, self.log_std_param.exp())


class _Actor(nn.Module):
    obs_groups = ["policy"]

    def __init__(self):
        super().__init__()
        self.distribution = _Gaussian()
        self.mlp = nn.Sequential(nn.Linear(45, 16), nn.ELU(), nn.Linear(16, 12))

    def forward(self, observations, stochastic_output=False):
        return self.mlp(observations["policy"])


def _policy():
    model = _Actor()
    inject_lora(model, rank=2)
    mark_only_lora_trainable(model)
    return RslMLPPolicyAdapter(model, action_transform=clip_lite3_policy_action)


def _critic(width):
    return nn.Sequential(nn.Linear(width, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))


def _checkpoint(width=48):
    old_policy, old_critic = _policy(), _critic(width)
    session = OnlinePPOSession(OnlinePPOUpdater(old_policy, old_critic), rollout_capacity=16)
    return {
        "session": session.state_dict(),
        **({"critic_observation_schema": CRITIC_OBSERVATION_SCHEMA} if width == 48 else {}),
    }, old_policy, old_critic


@pytest.mark.parametrize("width", [45, 48])
def test_worker_migration_preserves_actor_and_value_with_new_nonzero_odometry(width):
    checkpoint, old_policy, old_critic = _checkpoint(width)
    target = ContextGatedPolicy(copy.deepcopy(old_policy))
    state, record = _prepare_checkpoint_session(
        checkpoint, policy=target, base_obs_dim=45, proprioceptive_context=True,
    )
    session = OnlinePPOSession(OnlinePPOUpdater(target, _critic(51)), rollout_capacity=16)
    session.load_state_dict(state)
    base = torch.randn(5, 45)
    velocity = torch.randn(5, 2)
    context = torch.tensor([[0.3, -0.4, 0.7]]).expand(5, -1)
    critic48 = build_critic_observation(base, velocity, step=4, max_steps=10)
    with torch.no_grad():
        torch.testing.assert_close(
            target.mean_action(_append_odometry_context(base, context)), old_policy.mean_action(base),
            atol=0, rtol=0,
        )
        torch.testing.assert_close(
            session.updater.critic(_append_odometry_context(critic48, context)),
            old_critic(critic48 if width == 48 else base), atol=1e-7, rtol=1e-6,
        )
    assert record["critic_input_from"] == width and record["critic_input_to"] == 51
    assert record["actor_observation_from"] == BASE_POLICY_OBSERVATION_SCHEMA
    assert len(state["updater"]["actor_optimizer"]["param_groups"][0]["params"]) == 6
    assert checkpoint["session"]["updater"]["critic"]["0.weight"].shape[1] == width

    # A saved trained contextual head must be preserved, including on frozen probes.
    with torch.no_grad():
        target.context_B.weight.fill_(0.01)
    saved = {
        "session": session.state_dict(), "actor_observation_schema": CONTEXT_POLICY_OBSERVATION_SCHEMA,
        "critic_observation_schema": CONTEXT_CRITIC_OBSERVATION_SCHEMA,
    }
    restored, _ = _prepare_checkpoint_session(saved, policy=target, base_obs_dim=45, proprioceptive_context=True)
    torch.testing.assert_close(restored["updater"]["policy"]["context_B.weight"], target.context_B.weight)
    with pytest.raises(ValueError, match="requires --proprioceptive-context"):
        _prepare_checkpoint_session(saved, policy=old_policy, base_obs_dim=45, proprioceptive_context=False)


def test_checkpoint_observation_semantics_are_checked_before_migration():
    checkpoint, policy, _ = _checkpoint()
    checkpoint["critic_observation_schema"] = "unrelated_48dim_observation"
    before = checkpoint["session"]["updater"]["critic"]["0.weight"].clone()
    with pytest.raises(ValueError, match="schema"):
        _prepare_checkpoint_session(checkpoint, policy=ContextGatedPolicy(policy), base_obs_dim=45,
                                    proprioceptive_context=True)
    torch.testing.assert_close(before, checkpoint["session"]["updater"]["critic"]["0.weight"], atol=0, rtol=0)


def test_actor_current_and_critic_terminal_inputs_keep_distinct_odometry_across_reset():
    odometry = PlanarProprioOdometry(1, "cpu")
    current_base = torch.ones(1, 45)
    terminal_base = torch.full((1, 45), 2.0)
    current = _append_odometry_context(current_base, odometry.context())
    measured_velocity = torch.tensor([[0.3, 0.1]])
    terminal_context = odometry.observe(measured_velocity, torch.tensor([[0.2]]), step_dt=0.02)
    terminal_critic = _append_odometry_context(
        build_critic_observation(terminal_base, measured_velocity, step=1, max_steps=2000), terminal_context,
    )
    odometry.reset(torch.tensor([True]))

    assert current.shape == (1, 48) and terminal_critic.shape == (1, 51)
    assert torch.count_nonzero(current[:, 45:]) == 0
    assert torch.count_nonzero(terminal_critic[:, 48:]) > 0
    torch.testing.assert_close(terminal_critic[:, :45], terminal_base, atol=0, rtol=0)
    torch.testing.assert_close(terminal_critic[:, 45:47], measured_velocity, atol=0, rtol=0)
    assert terminal_critic[0, 47] == pytest.approx(1 / 2000)
    assert torch.count_nonzero(odometry.context()) == 0
    assert torch.count_nonzero(terminal_context) > 0
    assert torch.equal(current_base, torch.ones_like(current_base))


@pytest.mark.parametrize("context", [torch.zeros(2, 3), torch.zeros(1, 4), torch.full((1, 3), float("nan"))])
def test_invalid_odometry_cannot_enter_actor_or_critic_rollouts(context):
    with pytest.raises(ValueError, match="context"):
        _append_odometry_context(torch.zeros(1, 45), context)
