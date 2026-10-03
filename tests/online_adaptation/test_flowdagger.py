import copy

import pytest
import torch
from torch import nn

from rl_training.online_adaptation.flowdagger import (
    ConditionalFlow, DualBuffer, NoisePolicy, balanced_batch, invert_euler,
)


class AffineField(ConditionalFlow):
    def __init__(self, slope=-.5):
        super().__init__(observation_dim=2, action_dim=2, hidden=(8,), steps=8)
        self.slope = slope

    def velocity(self, obs, state, time):
        return self.slope * state + obs + float(time)


def test_inverse_reverses_discrete_map_not_reverse_time_approximation():
    flow = AffineField().double()
    obs = torch.tensor([[.3, -.2], [.1, .4]], dtype=torch.float64)
    noise = torch.tensor([[1., -1.], [.5, .3]], dtype=torch.float64)
    target = flow(obs, noise)
    saved = copy.deepcopy(flow.state_dict())
    inverse = invert_euler(flow, obs, target, tolerance=1.e-9)
    assert inverse.accepted.all()
    torch.testing.assert_close(inverse.noise, noise, atol=1.e-8, rtol=0)
    assert all(torch.equal(v, flow.state_dict()[k]) for k, v in saved.items())
    assert all(p.grad is None for p in flow.parameters())


def test_noncontracting_inverse_is_rejected_not_fabricated():
    flow = AffineField(slope=10.)
    obs = torch.ones(2, 2)
    inverse = invert_euler(flow, obs, torch.ones_like(obs), iterations=3)
    assert not inverse.accepted.any()


def test_out_of_bound_inverse_label_is_not_clamped():
    flow = AffineField()
    obs = torch.zeros(1, 2)
    inverse = invert_euler(flow, obs, flow(obs, torch.full_like(obs, 8.)), noise_bound=6.)
    assert not inverse.accepted.any()
    assert inverse.noise.min() > 7.9


def test_dual_buffer_filters_failures_invalid_labels_and_duplicates():
    buffer = DualBuffer()
    obs, noise = torch.arange(12.).reshape(4, 3), torch.arange(8.).reshape(4, 2)
    flags = torch.tensor([True, True, False, False])
    valid = torch.tensor([True, False, True, True])
    noise[1] = float("nan")
    args = dict(observations=obs, noises=noise, interventions=flags, valid=valid)
    buffer.add_episode("failed", successful=False, **args)
    assert buffer.counts() == {"correction": 0, "autonomous": 0}
    buffer.add_episode("success", successful=True, **args)
    assert buffer.counts() == {"correction": 1, "autonomous": 2}
    with pytest.raises(ValueError, match="already"):
        buffer.add_episode("success", successful=True, **args)
    batch_o, batch_z = balanced_batch(buffer.tensors("cpu"), 100, torch.Generator().manual_seed(8))
    assert (batch_o[:50] == obs[0]).all()
    assert (batch_o[50:, 0] >= obs[2, 0]).all()
    assert torch.isfinite(batch_z).all()


def test_empty_buffer_cannot_silently_change_mixture():
    with pytest.raises(ValueError, match="both"):
        DualBuffer().tensors("cpu")


def test_initial_noise_zero_and_learning_does_not_mutate_frozen_flow():
    flow = ConditionalFlow(3, 2, hidden=(16,), steps=4).requires_grad_(False)
    actor = NoisePolicy(3, 2)
    obs = torch.randn(20, 3)
    assert torch.equal(actor(obs), torch.zeros(20, 2))
    saved = copy.deepcopy(flow.state_dict())
    opt = torch.optim.Adam(actor.parameters(), lr=.01)
    for _ in range(10):
        loss = (actor(obs) - 1.).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss < .5
    assert all(torch.equal(v, flow.state_dict()[k]) for k, v in saved.items())
    assert torch.isfinite(flow(obs, actor(obs))).all()
