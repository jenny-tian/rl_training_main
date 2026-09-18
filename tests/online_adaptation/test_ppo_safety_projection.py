"""PPO learns requests through the fixed safety filter, including projections."""

import math

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from rl_training.online_adaptation.online_session import OnlinePPOSession
from rl_training.online_adaptation.ppo_online import OnlinePPOUpdater, PPOConfig
from rl_training.online_adaptation.ppo_rollout import PPORollout
from rl_training.online_adaptation.ppo_safety import requested_action_rollout
from rl_training.online_adaptation.publisher import DoubleBufferedActorPublisher
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState


class GaussianPolicy(nn.Module):
    """A small actor using the same requested-action interface as online PPO."""

    def __init__(self, action_dim=12, mean=0.1, std=0.01):
        super().__init__()
        self.mean_coefficient = nn.Parameter(torch.full((action_dim,), mean))
        self.register_buffer("std", torch.full((action_dim,), std))

    def get_distribution(self, observations):
        return Normal(observations[:, :1] * self.mean_coefficient, self.std)

    def evaluate_actions(self, observations, actions):
        distribution = self.get_distribution(observations)
        return distribution.log_prob(actions).sum(-1), distribution.entropy().sum(-1)


def safe_state():
    return SafetyState(
        projected_gravity_xy=torch.zeros(1, 2),
        base_height=torch.full((1, 1), 0.35),
        joint_pos=torch.zeros(1, 12),
        joint_vel=torch.zeros(1, 12),
        joint_lower=torch.full((1, 12), -1.0),
        joint_upper=torch.full((1, 12), 1.0),
        default_joint_pos=torch.zeros(1, 12),
        action_scale=torch.full((1, 12), 0.25),
        nonfoot_contact=torch.zeros(1, 1, dtype=torch.bool),
        control_timed_out=torch.zeros(1, 1, dtype=torch.bool),
    )


def safety_filter():
    return SafetyFilter(SafetyLimits(
        residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4),
        residual_rate_limit=0.03,
    ))


def zero_critic():
    critic = nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight.zero_()
        critic.bias.zero_()
    return critic


def test_projected_request_keeps_sampling_log_probability_and_old_policy_ratio_one():
    policy = GaussianPolicy()
    critic = zero_critic()
    publisher = DoubleBufferedActorPublisher(policy)
    publisher.publish(policy, version=11)
    control_policy, version = publisher.current()
    observations = torch.tensor([[1.0, 0.0]], requires_grad=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        distribution = control_policy.get_distribution(observations)
        requested = distribution.sample()
    old_log_prob = distribution.log_prob(requested).sum(-1, keepdim=True)
    sampled_action = requested.detach().clone()
    sampled_log_prob = old_log_prob.detach().clone()

    decision = safety_filter().project(torch.zeros_like(requested), requested, safe_state())
    assert decision.projected.item()
    assert not decision.paused.item()
    assert not torch.equal(decision.action, sampled_action)
    torch.testing.assert_close(requested, sampled_action, rtol=0, atol=0)

    rollout = requested_action_rollout(
        observations=observations,
        requested_actions=requested,
        old_log_probs=old_log_prob,
        old_values=critic(observations),
        rewards=torch.zeros(1, 1),
        terminated=torch.zeros(1, 1, dtype=torch.bool),
        truncated=torch.zeros(1, 1, dtype=torch.bool),
        next_values=critic(observations),
        policy_version=version,
        safety_decision=decision,
    )

    torch.testing.assert_close(rollout.actions, sampled_action, rtol=0, atol=0)
    torch.testing.assert_close(rollout.old_log_probs, sampled_log_prob, rtol=0, atol=0)
    new_log_prob, _ = policy.evaluate_actions(rollout.observations, rollout.actions)
    ratio = (new_log_prob.unsqueeze(-1) - rollout.old_log_probs).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=0)
    executed_log_prob, _ = policy.evaluate_actions(rollout.observations, decision.action)
    assert not torch.allclose(executed_log_prob.unsqueeze(-1), sampled_log_prob)
    assert rollout.version_tensor().tolist() == [11]
    for name in ("observations", "actions", "old_log_probs", "old_values", "next_values"):
        assert not getattr(rollout, name).requires_grad


def test_all_rate_projected_requests_train_actor_and_critic_with_terminal_returns():
    length = 6
    gamma = 0.9
    terminal_penalty = -8.0
    policy = GaussianPolicy()
    critic = zero_critic()
    publisher = DoubleBufferedActorPublisher(policy)
    updater = OnlinePPOUpdater(policy, critic, PPOConfig(
        gamma=gamma,
        gae_lambda=1.0,
        epochs=1,
        minibatch_size=length,
        actor_lr=1e-4,
        critic_lr=1e-3,
        target_kl=None,
        normalize_advantages=False,
    ))
    session = OnlinePPOSession(updater, rollout_capacity=length, publisher=publisher)
    control_policy, version = publisher.current()
    filt = safety_filter()
    previous_executed = torch.zeros(1, 12)
    sampled_actions = []

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72)
        for step in range(length):
            # Alternating means require rate projection while remaining well
            # within the amplitude and physical joint-position limits.
            observations = torch.tensor([[(-1.0) ** step, step / 10.0]])
            with torch.no_grad():
                distribution = control_policy.get_distribution(observations)
                requested = distribution.sample()
                old_log_prob = distribution.log_prob(requested).sum(-1, keepdim=True)
                decision = filt.project(torch.zeros_like(requested), requested, safe_state())
            assert (requested.abs() < filt.limits.residual_limits).all()
            assert ((requested - previous_executed).abs() > 0.03).all()
            assert decision.projected.item()
            assert not decision.paused.item()
            assert decision.reasons == ()
            torch.testing.assert_close(
                (decision.action - previous_executed).abs(),
                torch.full_like(requested, 0.03),
                rtol=0,
                atol=1e-7,
            )
            previous_executed = decision.action.detach().clone()
            sampled_actions.append(requested.detach().clone())
            session.append(requested_action_rollout(
                observations=observations,
                requested_actions=requested,
                old_log_probs=old_log_prob,
                old_values=critic(observations),
                rewards=torch.tensor([[terminal_penalty if step == length - 1 else 0.0]]),
                terminated=torch.tensor([[step == length - 1]]),
                truncated=torch.zeros(1, 1, dtype=torch.bool),
                next_values=torch.zeros(1, 1),
                policy_version=version,
                safety_decision=decision,
            ))

    # Use the public checkpoint representation to inspect exactly what the
    # session accumulated, including the terminal row and every projection.
    rollout = PPORollout(**session.state_dict()["pending_rollout"])
    expected_returns = terminal_penalty * gamma ** torch.arange(length - 1, -1, -1).float().unsqueeze(-1)
    advantages, returns = rollout.compute_returns(gamma=gamma, gae_lambda=1.0)
    torch.testing.assert_close(advantages, expected_returns)
    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(rollout.actions, torch.cat(sampled_actions), rtol=0, atol=0)
    assert session.ready

    actor_before = policy.mean_coefficient.detach().clone()
    critic_before = {name: parameter.detach().clone() for name, parameter in critic.named_parameters()}
    critic_inputs = []
    hook = critic.register_forward_pre_hook(
        lambda module, inputs: critic_inputs.append(inputs[0].detach().clone())
    )
    try:
        stats = session.update()
    finally:
        hook.remove()

    assert stats is not None and stats.accepted
    assert stats.num_samples == length
    assert stats.policy_version_before == version
    assert stats.policy_version_after == version + 1
    assert stats.actor_loss == pytest.approx(-expected_returns.mean().item())
    expected_value_loss = 0.5 * updater.config.value_coef * expected_returns.square().mean().item()
    assert stats.value_loss == pytest.approx(expected_value_loss)
    assert not torch.equal(policy.mean_coefficient.detach(), actor_before)
    assert any(not torch.equal(parameter.detach(), critic_before[name])
               for name, parameter in critic.named_parameters())
    seen_observations = torch.cat(critic_inputs)
    assert seen_observations.shape[0] == length
    torch.testing.assert_close(
        seen_observations[seen_observations[:, 1].argsort()], rollout.observations,
        rtol=0, atol=0,
    )
    assert session.pending_transitions == 0
    assert session.counters.transitions_collected == length
    assert session.counters.accepted_updates == 1
    assert session.counters.discarded_transitions == 0


def test_clipped_bandit_request_gradient_matches_integral_but_projection_selection_is_biased():
    # Standardized one-step environment: u ~ Normal(mu, 1), execution/reward
    # clip(u, -c, c). At mu=0 the baseline is zero and dE[reward]/dmu=P(|u|<c).
    # The grid includes both clipping boundaries exactly; quadrature has no
    # random sampling variance and the omitted normal tails are negligible.
    bound = 0.5
    requests = torch.linspace(-8.0, 8.0, 32769, dtype=torch.float64).unsqueeze(-1)
    observations = torch.ones_like(requests)
    policy = GaussianPolicy(action_dim=1, mean=0.0, std=1.0).double()
    old_log_prob, _ = policy.evaluate_actions(observations, requests)
    old_log_prob = old_log_prob.detach()
    density = old_log_prob.exp()
    reward = requests[:, 0].clamp(-bound, bound)

    log_prob, _ = policy.evaluate_actions(observations, requests)
    ratio = (log_prob - old_log_prob).exp()
    full_objective = torch.trapezoid(ratio * reward * density, requests[:, 0])
    full_gradient = torch.autograd.grad(full_objective, policy.mean_coefficient)[0].item()
    analytic_gradient = math.erf(bound / math.sqrt(2.0))
    assert full_gradient == pytest.approx(analytic_gradient, rel=2e-6, abs=2e-8)

    # The old mask conditions on the sampled action. Even with the exact
    # baseline and old-policy ratio=1, its population gradient is different.
    unprojected = requests[:, 0].abs() <= bound
    log_prob, _ = policy.evaluate_actions(observations, requests)
    ratio = (log_prob - old_log_prob).exp()
    retained_probability = torch.trapezoid(density[unprojected], requests[unprojected, 0])
    selected_objective = torch.trapezoid(
        (ratio * reward * density)[unprojected], requests[unprojected, 0]
    ) / retained_probability
    selected_gradient = torch.autograd.grad(selected_objective, policy.mean_coefficient)[0].item()
    boundary_density = math.exp(-0.5 * bound ** 2) / math.sqrt(2.0 * math.pi)
    analytic_selected_gradient = (analytic_gradient - 2.0 * bound * boundary_density) / analytic_gradient
    assert selected_gradient == pytest.approx(analytic_selected_gradient, rel=2e-6, abs=2e-8)
    assert full_gradient - selected_gradient > 0.25
