"""不学习的对照：有方向误差的目标残差直接作为下一步的请求。方案见 docs/lite3-online-adapter2-20261126/PLAN.md 第 12 节。"""
import pytest
import torch

from rl_training.ea_lpl import adapter, directed_rule

from test_adapter import EXTRA, learner

torch.set_default_dtype(torch.float32)


def test_the_request_is_zero_until_the_first_step_has_been_observed():
    rule = directed_rule.DirectedRule(3)
    assert torch.count_nonzero(rule.act(torch.ones(3, dtype=torch.bool))) == 0


def test_the_request_is_the_target_the_small_model_regresses_onto_one_step_later():
    # 同一串误差和受阻信号分别喂给规则和学习器：规则在第 t+1 步请求的，就是学习器在第 t 步之后拿来回归的目标。
    g = torch.Generator().manual_seed(3)
    L = learner('residual', 's')
    rule = directed_rule.DirectedRule(4)
    on = torch.ones(4, dtype=torch.bool)
    for step in range(30):
        error = torch.randn(4, 12, generator=g)
        stalled = torch.rand(4, 4, generator=g) < .3
        expect = L.target(error, stalled, on)
        rule.observe(error, stalled, on)
        assert torch.allclose(rule.act(on), expect, atol=1e-7), step


def test_a_new_attempt_starts_from_a_zero_request_and_a_fresh_slow_filter():
    rule = directed_rule.DirectedRule(2)
    on, none = torch.ones(2, dtype=torch.bool), torch.zeros(2, 4, dtype=torch.bool)
    rule.observe(torch.zeros(2, 12), none, on)
    rule.observe(torch.full((2, 12), .5), none, on)
    assert torch.allclose(rule.act(on), torch.full((2, 12), .2 * .5 * .98))
    rule.reset(torch.tensor([1]))
    request = rule.act(on)
    assert torch.count_nonzero(request[1]) == 0 and torch.count_nonzero(request[0]) == 12
    rule.observe(torch.full((2, 12), 3.), none, on)  # 环境 1 的慢滤波从这一步的误差重新开始，所以目标是 0
    assert torch.count_nonzero(rule.act(on)[1]) == 0 and torch.count_nonzero(rule.act(on)[0]) == 12


def test_environments_outside_the_policy_phase_request_nothing_and_keep_their_state():
    rule = directed_rule.DirectedRule(2)
    none = torch.zeros(2, 4, dtype=torch.bool)
    both, first = torch.ones(2, dtype=torch.bool), torch.tensor([True, False])
    rule.observe(torch.zeros(2, 12), none, both)
    rule.observe(torch.full((2, 12), 1.), none, first)
    assert torch.count_nonzero(rule.act(both)[1]) == 0  # 这一步环境 1 不在策略阶段，没有新的目标
    assert torch.count_nonzero(rule.act(first)[1]) == 0 and torch.count_nonzero(rule.act(first)[0]) == 12
    rule.observe(torch.full((2, 12), 1.), none, both)
    # 环境 1 的慢滤波在被跳过的那一步没有动：这一步从 0 走到 0.02，目标是 0.2 × 0.98；如果被推动过，会是 0.2 × 0.9604
    assert torch.allclose(rule.act(both)[1], torch.full((12,), .2 * .98))


def test_the_request_respects_the_amplitude_limits_and_lifts_stalled_swing_legs():
    wide = [2 * x for x in adapter.RESIDUAL_LIMITS]
    on = torch.ones(1, dtype=torch.bool)
    for limits, expect in ((None, [.24, .20, .20]), (wide, [.48, .40, .40])):
        rule = directed_rule.DirectedRule(1, limits=limits)
        rule.observe(torch.zeros(1, 12), torch.zeros(1, 4, dtype=torch.bool), on)
        rule.observe(torch.full((1, 12), 9.), torch.zeros(1, 4, dtype=torch.bool), on)
        assert rule.act(on)[0, :3].tolist() == pytest.approx(expect)
    rule = directed_rule.DirectedRule(1)
    stalled = torch.tensor([[False, False, True, False]])
    rule.observe(torch.zeros(1, 12), stalled, on)
    request = rule.act(on)[0]
    assert request[7] == pytest.approx(-.10) and request[8] == pytest.approx(.20) and torch.count_nonzero(request) == 2


def test_wrong_shapes_are_rejected():
    rule = directed_rule.DirectedRule(2)
    with pytest.raises(ValueError):
        rule.observe(torch.zeros(3, 12), torch.zeros(3, 4, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError):
        rule.observe(torch.zeros(2, 12), torch.zeros(2, 4), torch.ones(2, dtype=torch.bool))
