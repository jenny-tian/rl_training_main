import copy

import torch

from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.sac import SACConfig, SACUpdater
from rl_training.online_adaptation.types import TransitionBatch


def success_batch():
    z = torch.zeros(128, 1)
    b = TransitionBatch(torch.zeros(128, 25, 49), torch.zeros(128, 49), torch.zeros(128, 12),
        torch.zeros(128, 12), torch.full_like(z, 1.5), torch.zeros(128, 25, 49), torch.zeros(128, 49),
        torch.zeros(128, 12), z.clone(), z.clone(), torch.ones_like(z), torch.full_like(z, .99),
        torch.ones_like(z), z.long(), z.long())
    b.reward[0] = 305
    b.terminated[0] = 1
    b.bootstrap_mask[0] = 0
    return b


def test_huber_accepts_single_legal_success_without_changing_actor_or_temperature():
    torch.manual_seed(7)
    actor, critic = ResidualActor(49, 12), TwinCritic(49, 12)
    results = []
    for delta in (None, 1.):
        updater = SACUpdater(copy.deepcopy(actor), copy.deepcopy(critic), SACConfig(
            initial_alpha=.01, max_critic_loss=800, max_mean_td_error=10,
            max_critic_grad_norm=3000, critic_huber_delta=delta))
        alpha = updater.log_alpha.detach().clone()
        torch.manual_seed(23)
        result = updater.update(success_batch(), update_actor=False, residual_gain=.5)
        results.append(result)
        assert torch.equal(alpha, updater.log_alpha)
        assert not updater.actor_optimizer.state and not updater.alpha_optimizer.state
        assert all(torch.equal(v, updater.actor.state_dict()[k]) for k, v in actor.state_dict().items())
    assert results[0].rejection_reason == 'critic_loss_limit'
    assert results[1].accepted and results[1].critic_updated and not results[1].actor_updated
    assert results[0].mean_unclipped_target_q == results[1].mean_unclipped_target_q
    assert results[1].target_q_clip_fraction == 0


def test_huber_keeps_td_guard_and_rolls_back_rejected_update():
    torch.manual_seed(7)
    updater = SACUpdater(ResidualActor(49, 12), TwinCritic(49, 12), SACConfig(
        initial_alpha=.01, max_critic_loss=800, max_mean_td_error=10,
        max_critic_grad_norm=3000, critic_huber_delta=1.))
    initial = copy.deepcopy(updater.critic.state_dict())
    b = success_batch()
    b.reward.fill_(100)
    result = updater.update(b, update_actor=False, residual_gain=.5)
    assert not result.accepted and result.rejection_reason == 'mean_td_error_limit'
    assert not updater.critic_optimizer.state
    assert all(torch.equal(v, updater.critic.state_dict()[k]) for k, v in initial.items())
