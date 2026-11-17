import numpy as np
import pytest
import torch

from rl_training.ea_lpl.c1a import Session
from rl_training.ea_lpl.c1a_ea import SessionEA, ea_variance, response, LATENT_BUDGET, VARIANCE_MIN, VARIANCE_MAX

FULL = dict(executed_requested=[100., 100.], requested_squared=[100., 100.], steps=200)


def run(head, noises, returns, **execution):
    result = []
    for noise, value in zip(noises, returns):
        head.begin(noise)
        result.append(head.finish(value, **(execution or FULL)))
    return result


def test_response_is_ratio_times_tanh_slope_squared():
    assert np.allclose(response([1., .5], [0., 0.]), [1., .25])
    assert np.allclose(response([1., 1.], [2., 0.]), [(1 - np.tanh(2.)**2)**2, 1.])
    with pytest.raises(ValueError): response([1.], [0., 0.])


@pytest.mark.parametrize('m,status', [([.7, .7], 'fallback_isotropic'), ([.72, .66], 'fallback_isotropic'),
    ([0., 0.], 'fallback_no_response'), ([1., 1.e-5], 'fallback_condition'), ([np.nan, 1.], 'fallback_invalid')])
def test_fallbacks_return_fixed_covariance(m, status):
    variance, reason = ea_variance(m)
    assert reason == status and np.array_equal(variance, [.36, .36])


def test_solution_respects_constraints_and_matches_closed_form():
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = rng.uniform(0., 1., 2)
        variance, reason = ea_variance(m)
        assert (variance >= VARIANCE_MIN - 1e-12).all() and (variance <= VARIANCE_MAX + 1e-12).all()
        assert variance.sum() <= LATENT_BUDGET + 1e-9
        if reason == 'solved': assert variance[np.argmax(m)] >= variance[np.argmin(m)]
    # 一维饱和、一维正常：g = (0.0036 - 0.5, 0.72)，预算约束生效，解为 0.36 ∓ (g2 - g1)/4。
    variance, reason = ea_variance([.0036, .72])
    assert reason == 'solved' and np.allclose(variance, [.36 - 1.2164 / 4, .36 + 1.2164 / 4], atol=.006)


def test_first_batch_and_isotropic_response_reproduce_fixed_covariance_learner():
    rng = np.random.default_rng(3)
    plain, ea = Session(True), SessionEA(True)
    for _ in range(6):
        noises, returns = rng.normal(size=(4, 2)), rng.normal(size=4)
        for noise, value in zip(noises, returns):
            assert np.array_equal(plain.begin(noise), ea.begin(noise))
            plain.finish(value); stats = ea.finish(value, executed_requested=[85., 85.], requested_squared=[100., 100.], steps=300)
        # 把均值放回原点附近，使两维斜率相同，协方差保持回退值。
        plain.b = plain.b * 0; ea.b = ea.b * 0
        assert torch.equal(plain.m, ea.m) and torch.equal(plain.v, ea.v)
        assert stats['std_after'] == [.6, .6]


def test_covariance_changes_only_between_batches_and_scales_noise_and_gradient():
    head = SessionEA(True)
    head.b = torch.tensor([2., 0.], dtype=torch.float64)
    stats = run(head, [[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]], [0., 0., 0., 0.])
    assert all('std_after' not in s for s in stats[:3]) and stats[-1]['covariance_status'] == 'solved'
    assert stats[-1]['std'] == [.6, .6]
    std = head.std.clone()
    assert std[0] < .6 < std[1] and torch.equal(head.b, torch.tensor([2., 0.], dtype=torch.float64))
    code = head.begin([1., 1.])
    assert np.allclose(code, [2. + std[0], std[1]])
    head.finish(1., **FULL)
    stats = run(head, [[-1., 1.], [1., -1.], [-1., -1.]], [0., 0., 0.])[-1]
    noises = torch.tensor([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]], dtype=torch.float64)
    adv = torch.tensor([1., -1 / 3, -1 / 3, -1 / 3], dtype=torch.float64)
    assert np.allclose(stats['gradient'], (-(adv[:, None] * noises).mean(0) / std).numpy())


def test_frozen_adapts_covariance_without_moving_mean_and_small_request_falls_back():
    head = SessionEA(False)
    digest = head.digest()
    stats = run(head, np.zeros((4, 2)), [0., 1., 0., 1.], executed_requested=[0., 0.], requested_squared=[1e-9, 1e-9], steps=100)[-1]
    assert stats['covariance_status'] == 'fallback_small_request' and head.digest() == digest and head.updates == 0
    stats = run(head, np.ones((4, 2)), [0., 1., 0., 1.], executed_requested=[10., 90.], requested_squared=[100., 100.], steps=300)[-1]
    assert stats['covariance_status'] == 'solved' and head.std[0] < head.std[1] and head.digest() == digest
    snapshot = head.snapshot()
    assert torch.equal(snapshot['std'], head.std) and not snapshot['execution']


def test_learning_requires_execution_statistics_and_evaluation_does_not():
    head = SessionEA(True)
    head.begin([0., 0.])
    with pytest.raises(ValueError): head.finish(1.)
    head.finish(1., learn=False)
    assert not head.batch and head.updates == 0
