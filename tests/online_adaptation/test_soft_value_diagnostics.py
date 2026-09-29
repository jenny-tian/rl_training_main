import copy
from dataclasses import replace

import pytest
import torch

from rl_training.online_adaptation.networks import ResidualActor, TwinCritic
from rl_training.online_adaptation.soft_value_diagnostics import fixed_soft_rows, summarize_soft_rows
from rl_training.online_adaptation.types import TransitionBatch


def batch():
    frame = torch.ones(4, 3)
    history = frame.unsqueeze(1).repeat(1, 2, 1)
    action = torch.zeros(4, 2)
    z = torch.zeros(4, 1)
    return TransitionBatch(history, frame, action, action, torch.ones_like(z), history, frame,
                           action, z.clone(), z.clone(), torch.ones_like(z), torch.full_like(z, .9),
                           torch.ones_like(z), z.clone(), z.clone())


class FixedActor:
    def sample(self, history, frame, base):
        return torch.ones_like(base), torch.full((len(frame), 1), -2.), torch.zeros_like(base)


class ActionCritic:
    def __call__(self, history, frame, base, request):
        q = 3 + request.sum(dim=1, keepdim=True)
        return q, q+1


def test_terminal_masks_remove_both_bootstrap_q_and_entropy():
    b = batch()
    b.bootstrap_mask[0] = 0
    b.terminated[0] = 1
    rows = fixed_soft_rows(FixedActor(), ActionCritic(), ActionCritic(), b,
                          alpha=.2, residual_gain=.25, draws=4, batch_size=3)
    # two .25 actions; next min-Q=3.5, entropy=.4. Terminal target is reward alone.
    torch.testing.assert_close(rows['target_mean'], torch.tensor([[1.], [4.51], [4.51], [4.51]]))
    torch.testing.assert_close(rows['effective_entropy_bonus'], torch.tensor([[0.], [.36], [.36], [.36]]))
    torch.testing.assert_close(rows['sampled_minus_zero_next_q'], torch.full((4, 1), .5))
    assert rows['target_sample_std'].max() == 0
    assert summarize_soft_rows(rows, b.terminated.bool())['samples'] == 1
    assert summarize_soft_rows(rows, b.truncated.bool()) == {'samples': 0}


def test_current_q_uses_request_not_executed_residual():
    b = replace(batch(), requested_residual_action=torch.full((4, 2), .75))
    rows = fixed_soft_rows(FixedActor(), ActionCritic(), ActionCritic(), b, alpha=.2, residual_gain=.25)
    torch.testing.assert_close(rows['q1'], torch.full((4, 1), 4.5))


def test_real_network_diagnostic_preserves_parameters_and_rng_and_repeats():
    torch.manual_seed(71)
    actor, critic = ResidualActor(3, 2, action_limits=torch.tensor([.2, .4])), TwinCritic(3, 2)
    target = copy.deepcopy(critic)
    saved = [copy.deepcopy(m.state_dict()) for m in (actor, critic, target)]
    rng = torch.get_rng_state().clone()
    outputs = [fixed_soft_rows(actor, critic, target, batch(), alpha=.01, residual_gain=.5,
                              draws=3, batch_size=4, seed=23) for _ in range(2)]
    assert torch.equal(torch.get_rng_state(), rng)
    for key in outputs[0]:
        assert torch.equal(outputs[0][key], outputs[1][key])
        assert not outputs[0][key].requires_grad
    for model, initial in zip((actor, critic, target), saved):
        for key in initial:
            assert torch.equal(initial[key], model.state_dict()[key])


def test_normalized_entropy_is_independent_of_action_limit_scaling():
    actor = ResidualActor(3, 2, action_limits=torch.ones(2))
    b = batch()
    torch.manual_seed(5)
    action, log_prob, _ = actor.sample(b.history, b.current_frame, b.base_action)
    actor.action_limits.mul_(.25)
    torch.manual_seed(5)
    scaled, scaled_log_prob, _ = actor.sample(b.history, b.current_frame, b.base_action)
    torch.testing.assert_close(scaled, action*.25)
    assert torch.equal(log_prob, scaled_log_prob)


@pytest.mark.parametrize('mask', [float('nan'), .5])
def test_invalid_masks_rejected(mask):
    b = batch()
    b.bootstrap_mask[0] = mask
    with pytest.raises(ValueError):
        fixed_soft_rows(FixedActor(), ActionCritic(), ActionCritic(), b, alpha=.2, residual_gain=.25)
