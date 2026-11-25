"""抬腿反射和 SARA 式残差里不依赖仿真的部分。方案见 docs/lite3-online-baselines-20261125/PLAN.md。"""
import pytest
import torch

from rl_training.ea_lpl import sara
from rl_training.ea_lpl.swing_reflex import SwingReflex

ROUGH = torch.ones(1)


def run(rule, stances, feet):
    reflex, out = SwingReflex(rule, 1), []
    for s, x in zip(stances, feet):
        out.append(reflex.step(torch.tensor([[not s, True, True, True]]), torch.tensor([[x, 0., 0., 0.]]), ROUGH)[0, 2].item())
    return out, reflex


def test_reflex_fires_only_for_an_early_stall_and_ends_at_touchdown():
    # FL 摆动 5 步：第 1 步足端没有前进（受阻），之后正常前进；然后着地。
    swing = [False, True, True, True, True, True, False, False]
    feet = [0., 0., 0., .01, .02, .03, .04, .04]
    out, reflex = run('same', swing, feet)
    assert out == pytest.approx([0, 0, .2, .2, .2, .2, 0, 0])
    assert int(reflex.triggers[0]) == 1
    # 同样的摆动，足端一直在前进：不触发。
    out, _ = run('same', swing, [0., 0., .01, .02, .03, .04, .05, .05])
    assert out == [0.] * 8
    # 离地 2 步之后才受阻：不触发。
    out, _ = run('same', swing, [0., 0., .01, .02, .02, .02, .02, .02])
    assert out == [0.] * 8


def test_next_rule_carries_the_reflex_through_one_more_swing_only():
    swing = [False, True, True, True, False, True, True, False, True, True, False]
    feet = [0., 0., 0., .01, .02, .02, .03, .04, .04, .05, .06]
    out, _ = run('next', swing, feet)
    assert out == pytest.approx([0, 0, .2, .2, 0, .2, .2, 0, 0, 0, 0])
    out, _ = run('same', swing, feet)
    assert out == pytest.approx([0, 0, .2, .2, 0, 0, 0, 0, 0, 0, 0])


def test_reflex_scales_with_roughness_and_reset_clears_state():
    reflex = SwingReflex('next', 1)
    swing = torch.tensor([[False, True, True, True]])
    for x in (0., 0.): reflex.step(swing, torch.full((1, 4), x), ROUGH)
    request = reflex.step(swing, torch.zeros(1, 4), torch.tensor([.5]))
    assert request[0, 1].item() == pytest.approx(-.05) and request[0, 2].item() == pytest.approx(.1)
    assert request[0, 3:].abs().sum() == 0
    reflex.reset(torch.tensor([0]))
    assert reflex.step(swing, torch.zeros(1, 4), ROUGH).abs().sum() == 0
    with pytest.raises(ValueError): SwingReflex('other', 1)


def test_gate_attenuates_opposing_components_and_respects_bounds():
    limits = torch.tensor([.24, .20, .20] * 4)
    residual, nominal = torch.full((1, 12), .1), torch.ones(1, 12)
    nominal[0, 0] = -1.
    u = sara.gate(residual, nominal, torch.ones(1), torch.ones(1, 12), limits)
    assert u[0, 0].item() == pytest.approx(.025) and u[0, 1].item() == pytest.approx(.1)
    u = sara.gate(torch.full((1, 12), 5.), torch.ones(1, 12), torch.ones(1), torch.ones(1, 12), limits)
    assert (u <= limits + 1e-6).all() and torch.linalg.vector_norm(u).item() == pytest.approx(sara.NORM_BOUND)


def test_sara_starts_at_zero_learns_from_tracking_error_and_keeps_copies_apart():
    limits = [.24, .20, .20] * 4
    learner = sara.Sara('sa2', 2, 2, 6, limits, learn=True, seed=0)
    active, nominal = torch.tensor([True, True, False, False]), torch.ones(4, 12)
    obs = torch.randn(4, 6, generator=torch.Generator().manual_seed(1))
    assert learner.act(obs, nominal, active).abs().sum() == 0
    before = learner.digest(1)
    error = torch.ones(4, 12)
    for k in range(20):
        learner.observe(error, torch.zeros(4), torch.full((4,), .5), active)
        out = learner.act(obs * (1 + .1 * k), nominal, active)
    assert learner.updates.tolist() == [20, 0] and learner.digest(1) == before
    assert out[:2].abs().sum() > 0 and out[2:].abs().sum() == 0
    saved = learner.snapshot(0)
    other = sara.Sara('sa2', 2, 1, 6, limits, learn=False, seed=0)
    other.load(1, saved)
    assert other.digest(1) == learner.digest(0)
    other.act(obs[:2], nominal[:2], active[:2]); other.observe(error[:2], torch.zeros(2), torch.ones(2), active[:2])
    assert other.digest(1) == learner.digest(0)  # 评估时权重不变


def test_gamma_rises_when_performance_drops_and_reset_restores_it():
    learner = sara.Sara('sa1', 1, 1, 6, [.2] * 12, learn=False, seed=0)
    active = torch.tensor([True])
    learner.act(torch.zeros(1, 6), torch.ones(1, 12), active)
    for _ in range(100): learner.observe(torch.full((1, 12), 2.), torch.ones(1), torch.zeros(1), active)
    assert learner.gamma.item() > .8 and learner.beta.max().item() == pytest.approx(2.) and learner.boost.item() > 2.
    learner.reset(torch.tensor([0]))
    assert learner.gamma.item() == pytest.approx(sara.GAMMA_MIN) and learner.beta.max().item() == 1.


def test_tracking_error_sign():
    e = sara.tracking_error(torch.tensor([1.]), torch.tensor([.5]), torch.tensor([5.]), torch.tensor([.25]))
    assert e.item() == pytest.approx((.5 - .1) / .25)
