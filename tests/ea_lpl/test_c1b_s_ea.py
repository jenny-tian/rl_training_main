import numpy as np
import pytest
import torch

from rl_training.ea_lpl.c1b_s import ParallelSession, residual_request
from rl_training.ea_lpl.c1b_s_ea import GRID, ParallelSessionEA, SlotEA, ea_variance, execution_sums, response

ISO = dict(executed_requested=[.8, .8], requested_squared=[1., 1.], steps=100)  # 两个方向响应相同，回退到固定标准差


def play(group, noises, returns, execution=ISO):
    for k in range(4): group.begin(k, noises[k])
    return [group.finish(k, returns[k], **execution) for k in range(4)]


def test_isotropic_response_reproduces_the_fixed_covariance_learner():
    rng = np.random.default_rng(4)
    ea, plain = ParallelSessionEA(True), ParallelSession(True, active=('front_swing', 'rear_swing'))
    for _ in range(6):
        noises, returns = rng.standard_normal((4, 4)), rng.standard_normal(4).tolist()
        noises[:, 3] = noises[:, 2]  # 两个摆动参数始终相等，响应各向同性，标准差保持 0.7
        codes = [ea.begin(k, noises[k]) for k in range(4)]; plain_codes = [plain.begin(k, noises[k]) for k in range(4)]
        assert np.allclose(codes, plain_codes)
        stats = [ea.finish(k, returns[k], **ISO) for k in range(4)]; [plain.finish(k, returns[k]) for k in range(4)]
        assert stats[-1]['covariance_status'] == 'fallback_isotropic'
    assert torch.equal(ea.w, plain.w) and ea.w[2:].any() and not ea.w[:2].any()
    assert torch.allclose(ea.std, torch.tensor([0., 0., .7, .7], dtype=torch.float64))


def test_execution_sums_recover_the_executed_share_of_each_leg_group():
    weights = torch.tensor([[0., 0., 1.2, .4], [0., 0., -.5, 2.]])
    stance = torch.tensor([[True, False, False, True], [False, False, True, False]])
    request = residual_request(weights, stance, torch.tensor([1., .5]))
    scale = torch.tensor([.6, .6, .3, .3]).repeat_interleave(3)  # 前腿执行 60%，后腿执行 30%
    done, squared = execution_sums(request, request * scale)
    assert done.shape == squared.shape == (2, 2) and np.allclose(done / squared, [[.6, .3], [.6, .3]])
    z = np.tanh(weights.numpy())
    assert np.allclose(squared, [[z[0, 2]**2, z[0, 3]**2], [2 * (.5 * z[1, 2])**2, (.5 * z[1, 3])**2]])  # 请求系数就是摆动腿的幅度
    hip_x = torch.zeros(2, 12); hip_x[:, 0::3] = .2  # HipX 不在 sag 方向上，不计入
    assert np.allclose(execution_sums(request, request * scale + hip_x)[0], done)
    zero, none = execution_sums(torch.zeros(1, 12), torch.ones(1, 12))
    assert not zero.any() and not none.any()
    with pytest.raises(ValueError): execution_sums(torch.zeros(2, 12), torch.zeros(2, 6))


def test_variance_solution_matches_the_unconstrained_formula():
    # 目标对每个方向可分：s = 0.49 + (m - 0.5 * low) / 2，再取到网格上。
    variance, status = ea_variance([.30, .10])
    assert status == 'solved' and np.allclose(variance, [.64, .29], atol=.006)
    assert ea_variance([.5, .49])[1] == 'fallback_isotropic' and ea_variance([0., 0.])[1] == 'fallback_no_response'
    both, status = ea_variance([1., .6])  # 总量约束 s1 + s2 <= 0.98 生效
    assert status == 'solved' and both.sum() <= .98 + 1e-9 and both[0] > both[1]
    assert np.allclose(response([.6, .5], [0., 0.]), [.36, .25]) and len(GRID) == 97


def test_new_std_applies_from_the_next_round_and_scales_the_step():
    group = ParallelSessionEA(True)
    execution = dict(executed_requested=[.6, .3], requested_squared=[1., 1.], steps=100)
    noises = np.ones((4, 4)); returns = [2., 0., 0., 0.]
    stats = play(group, noises, returns, execution)
    assert stats[-1]['covariance_status'] == 'solved' and stats[-1]['std'] == [0., 0., .7, .7]
    first = group.w.clone()
    assert torch.allclose(first[2:], torch.full((2,), .6 * .7 * (2 / 3 * 1 - 2 / 9 * 3) / 4 * 4, dtype=torch.float64), atol=1) and first[2] == first[3]
    std = group.std.clone()
    assert std[2] > std[3] and not std[:2].any()
    assert np.allclose(group.begin(0, noises[0]), (first + std * 1.).numpy())
    for k in range(1, 4): group.begin(k, noises[k])
    for k in range(4): group.finish(k, returns[k], **execution)
    step = group.w - first
    assert step[2] / step[3] == pytest.approx(float(std[2] / std[3]))


def test_frozen_mean_stays_zero_while_covariance_adapts():
    frozen = ParallelSessionEA(False)
    stats = play(frozen, np.ones((4, 4)), [1., 0., 2., 3.], dict(executed_requested=[.6, .3], requested_squared=[1., 1.], steps=50))
    assert not frozen.w.any() and frozen.updates == 0 and stats[-1]['covariance_status'] == 'solved' and frozen.std[2] != .7


def test_small_requests_and_missing_statistics():
    group = ParallelSessionEA(True)
    stats = play(group, np.ones((4, 4)), [1., 0., 2., 3.], dict(executed_requested=[0., 0.], requested_squared=[1e-3, 1.], steps=100))
    assert stats[-1]['covariance_status'] == 'fallback_small_request' and torch.allclose(group.std[2:], torch.full((2,), .7, dtype=torch.float64))
    group.begin(0, np.zeros(4))
    with pytest.raises(ValueError): group.finish(0, 1.)
    idle = ParallelSessionEA(True); idle.begin(0, np.ones(4)); idle.finish(0, 1., learn=False)
    assert not idle.batch and not idle.w.any()


def test_slot_passes_execution_statistics_and_snapshot_has_std():
    group = ParallelSessionEA(True); slots = [SlotEA(group, k) for k in range(4)]
    for s in slots: s.begin(np.ones(4))
    for s, value in zip(slots, [1., 0., .5, 2.]): s.finish(value, **ISO)
    snap = slots[0].snapshot()
    assert slots[1].updates == 1 and snap['w'].shape == (4,) and snap['std'].shape == (4,)
    evaluator = ParallelSession(False, replicas=1); evaluator.load_evaluation_mean(snap)
    assert torch.equal(evaluator.w, group.w)
