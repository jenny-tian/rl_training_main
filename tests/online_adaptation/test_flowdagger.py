import copy

import pytest
import torch
from torch import nn

from rl_training.online_adaptation.flowdagger import (
    ConditionalFlow, DualBuffer, NoisePolicy, balanced_batch, invert_euler,
    qualify_paired_correction,
    preservation_loss,
    save_adapter_checkpoint,
    widen_noise_policy,
    correction_region_mask,
    lookahead_teacher_observation,
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


def episode(success=True, rms=.1, peak=.3, steps=1300, **extra):
    return dict(success=success, failure=not success, deadline=False,
                body_lateral_rms_m=rms, max_body_lateral_m=peak, steps=steps, **extra)


def test_paired_correction_requires_real_task_benefit():
    reference = episode()
    common = dict(prefix_error=0., intervention_count=10)
    assert qualify_paired_correction(reference, episode(rms=.11), **common) == (False, "insufficient_rms_gain")
    assert qualify_paired_correction(reference, episode(rms=.08, peak=.31), **common) == (False, "peak_regression")
    assert qualify_paired_correction(reference, episode(rms=.08, steps=1400), **common) == (False, "duration_regression")
    assert qualify_paired_correction(reference, episode(rms=.08, peak=.28), **common) == (True, "lower_drift")


def test_failed_teacher_and_unpaired_prefix_never_provide_labels():
    assert qualify_paired_correction(episode(False), episode(False), prefix_error=0., intervention_count=5) == (False, "corrected_episode_failed")
    assert qualify_paired_correction(episode(False), episode(), prefix_error=.01, intervention_count=5) == (False, "prefix_mismatch")
    assert qualify_paired_correction(episode(False), episode(), prefix_error=0., intervention_count=0) == (False, "no_intervention")


def test_recovery_requires_complete_reference_and_successful_correction():
    common = dict(prefix_error=0., intervention_count=5)
    assert qualify_paired_correction(episode(False), episode(), **common) == (True, "completion_recovery")
    assert qualify_paired_correction(episode(False, collector_cutoff=True), episode(), **common) == (False, "reference_incomplete")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1])
def test_nonfinite_or_negative_pairing_errors_rejected(value):
    with pytest.raises(ValueError):
        qualify_paired_correction(episode(), episode(), prefix_error=value, intervention_count=1)


def test_preservation_uses_autonomous_half_and_keeps_decoder_frozen():
    torch.manual_seed(17)
    flow = ConditionalFlow(3, 12, hidden=(16,), steps=2).requires_grad_(False)
    actor = NoisePolicy(3, 12)
    observations = torch.randn(8, 3)
    target = torch.ones(8, 12)
    frozen = copy.deepcopy(flow.state_dict())
    loss, noise, action = preservation_loss(flow, actor, observations, target, weight=25.)
    assert action > 0 and loss > noise
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert all(p.grad is None for p in flow.parameters())
    assert all(torch.equal(v, flow.state_dict()[k]) for k, v in frozen.items())
    target[:4] *= -100  # Correction labels must not change autonomous preservation.
    _, _, second_action = preservation_loss(flow, actor, observations, target, weight=25.)
    torch.testing.assert_close(action, second_action)


def test_zero_preservation_is_original_regression_and_unfrozen_decoder_rejected():
    flow = ConditionalFlow(3, 12, hidden=(8,), steps=2)
    actor = NoisePolicy(3, 12)
    obs, target = torch.randn(4, 3), torch.randn(4, 12)
    loss, noise, action = preservation_loss(flow, actor, obs, target)
    torch.testing.assert_close(loss, (actor(obs)-target).square().mean())
    assert action == 0
    with pytest.raises(ValueError, match="frozen"):
        preservation_loss(flow, actor, obs, target, weight=1.)


def test_skipped_update_still_preserves_qualified_replay_and_rng(tmp_path):
    actor = NoisePolicy(3, 12)
    optimizer = torch.optim.Adam(actor.parameters())
    buffer = DualBuffer()
    buffer.add_episode("qualified", successful=True, observations=torch.ones(2, 3), noises=torch.ones(2, 12),
        interventions=torch.tensor([True, False]), valid=torch.ones(2, dtype=torch.bool))
    generator = torch.Generator().manual_seed(17)
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(path, actor, optimizer, buffer, generator, round_id=0, updates=0,
                            protocol_sha256="test", skipped_reason="insufficient_data")
    saved = torch.load(path, weights_only=False)
    assert saved["updates"] == 0 and saved["skipped_reason"] == "insufficient_data"
    assert saved["episode_ids"] == ["qualified"]
    assert len(saved["buffer"]["correction"][0][0]) == 1
    assert saved["optimizer"]["state"] == {}
    torch.testing.assert_close(saved["generator_state"], generator.get_state())


def test_widening_preserves_function_without_sharing_parameters():
    torch.manual_seed(13)
    source = NoisePolicy(3, 12, hidden=(16, 8))
    with torch.no_grad():
        source.mlp[-1].weight.normal_(0., .1)
    before = copy.deepcopy(source.state_dict())
    wider = widen_noise_policy(source, 4, seed=17)
    observations = torch.randn(64, 3)*2
    torch.testing.assert_close(wider(observations), source(observations), atol=1.e-6, rtol=1.e-5)
    optimizer = torch.optim.SGD(wider.parameters(), lr=.1)
    loss = (wider(observations)-1).square().mean()
    loss.backward(); optimizer.step()
    assert not torch.equal(wider.mlp[2].weight[0], wider.mlp[2].weight[1])
    assert all(torch.equal(v, source.state_dict()[k]) for k,v in before.items())
    clone = NoisePolicy(3, 12, hidden=wider.hidden)
    clone.load_state_dict(wider.state_dict())
    torch.testing.assert_close(clone(observations), wider(observations))


def test_corrective_and_preservation_regions_are_disjoint_and_catch_inversion():
    import math
    obs = torch.zeros(5, 48); obs[:, 5] = -1; obs[:, 11] = 1
    obs[1, 9] = .11/1.65
    obs[2, 11] = math.cos(math.radians(6))
    obs[3, 5] = -math.cos(math.radians(11))
    obs[4, 5] = 1.  # Upside down must not count as low-risk preservation.
    region = correction_region_mask(obs)
    assert region.tolist() == [False, True, True, True, True]
    assert not (region & ~region).any()
    duplicate = obs.repeat(2, 1)
    torch.testing.assert_close(correction_region_mask(duplicate)[:5], region)
    with pytest.raises(ValueError):
        correction_region_mask(obs, lateral_m=-1)


def test_lookahead_teacher_uses_centerline_heading_without_mutating_sensor_input():
    obs = torch.zeros(3,48); obs[:,5] = -1; obs[:,11] = 1
    obs[0,9] = .2/1.65; obs[1,9] = -.2/1.65
    original = obs.clone()
    query = lookahead_teacher_observation(obs, lookahead_m=1.)
    torch.testing.assert_close(obs, original)
    assert query[0,10] > 0 and query[1,10] < 0
    torch.testing.assert_close(query[2], obs[2])
    torch.testing.assert_close(query[:,:10], obs[:,:10])
    torch.testing.assert_close(query[:,12:], obs[:,12:])
    torch.testing.assert_close(query[:,10].square()+query[:,11].square(), torch.ones(3))
    with pytest.raises(ValueError):
        lookahead_teacher_observation(obs, lookahead_m=0.)
