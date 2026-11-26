"""S1 小模型里不依赖仿真的部分：两种结构、奖励、目标残差、三种更新。方案见 docs/lite3-online-adapter-20261126/PLAN.md。"""
import math

import pytest
import torch

from rl_training.ea_lpl import adapter

torch.set_default_dtype(torch.float32)
EXTRA = 20


def state(sizes=(45, 512, 256, 128, 12), seed=0, std=None):
    g = torch.Generator().manual_seed(seed)
    result = {}
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        result[f'mlp.{2 * k}.weight'] = torch.randn(b, a, generator=g) / a**.5
        result[f'mlp.{2 * k}.bias'] = torch.randn(b, generator=g) * .1
    if std: result['distribution.log_std_param'] = torch.full((sizes[-1],), math.log(std))
    return result


def base_forward(s, x, upto=None):
    layers = sorted({int(k.split('.')[1]) for k in s if k.startswith('mlp.')})[:upto]
    for k in layers:
        x = x @ s[f'mlp.{k}.weight'].T + s[f'mlp.{k}.bias']
        if k != 6: x = torch.nn.functional.elu(x)
    return x


def make(kind, copies=3, seed=1):
    return adapter.StackedAdapter(state(), copies, kind, extra=EXTRA, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize('kind,count', [('film', 5564), ('residual', 5004)])
def test_every_copy_starts_as_the_base_policy_with_the_registered_size(kind, count):
    net = make(kind)
    obs, extra = torch.randn(3, 4, 45), torch.randn(3, 4, EXTRA)
    action, base = net(obs, extra)
    assert torch.equal(action, base) and torch.allclose(base, base_forward(state(), obs), atol=1e-5)
    assert sum(p[0].numel() for p in net.parameters()) == count
    assert not any(k.startswith('mlp') for k, _ in net.named_parameters())  # 基础策略的权重不是可训练参数


def test_film_modulates_the_last_hidden_layer_and_maps_the_action():
    net = make('film')
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in net.parameters(): p.copy_(torch.randn(p.shape, generator=g) * .3)
    obs, extra = torch.randn(3, 2, 45), torch.randn(3, 2, EXTRA)
    action, base = net(obs, extra)
    s, c = state(), 1
    P = {k: v[c] for k, v in net.named_parameters()}
    z = torch.tanh(torch.cat((obs[c], extra[c]), -1) @ P['G1'].T + P['g1'])
    gb = z @ P['G2'].T + P['g2']
    h = base_forward(s, obs[c], upto=3)
    h = (1 + .5 * torch.tanh(gb[:, :128])) * h + .5 * torch.tanh(gb[:, 128:])
    a0 = h @ s['mlp.6.weight'].T + s['mlp.6.bias']
    assert torch.allclose(action[c], a0 + a0 @ P['D'].T + P['d'], atol=1e-5)
    assert torch.allclose(base[c], base_forward(s, obs[c]), atol=1e-5)


def test_residual_network_adds_to_the_base_action():
    net = make('residual')
    with torch.no_grad():
        for p in net.parameters(): p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(5)) * .3)
    obs, extra = torch.randn(3, 2, 45), torch.randn(3, 2, EXTRA)
    action, base = net(obs, extra)
    P = {k: v[2] for k, v in net.named_parameters()}
    z = torch.tanh(torch.cat((obs[2], extra[2]), -1) @ P['R1'].T + P['r1'])
    assert torch.allclose(action[2] - base[2], z @ P['R2'].T + P['r2'], atol=1e-5)


@pytest.mark.parametrize('kind', ['film', 'residual'])
def test_copies_do_not_share_gradients(kind):
    net = make(kind)
    with torch.no_grad():
        for p in net.parameters(): p.add_(.05)
    action, _ = net(torch.randn(3, 4, 45), torch.randn(3, 4, EXTRA))
    action[0].sum().backward()
    for p in net.parameters():
        assert p.grad[0].abs().sum() > 0 and p.grad[1:].abs().sum() == 0


def test_reward_follows_the_registered_terms():
    swing = torch.tensor([[True, True, False, False], [False] * 4, [True, False, False, False]])
    stalled = torch.tensor([[True, False, False, False], [False] * 4, [False] * 4])
    correction = torch.zeros(3, 12); correction[2, 0] = .5
    r = adapter.reward(torch.tensor([.4, .4, .15]), swing, stalled, correction, torch.tensor([False, False, True]))
    assert r[0] == pytest.approx(.02 * (1 + .5 * .5)) and r[1] == pytest.approx(.02 * 1.5)
    assert r[2] == pytest.approx(.02 * (math.exp(-1.) + .5 - 2 * .25) - 1.)


def test_directed_target_combines_tracking_error_and_the_lift_of_stalled_swing_legs():
    error = torch.zeros(2, 12); error[0, 0] = 1.
    stalled = torch.tensor([[False, True, False, False], [False] * 4])
    target = adapter.directed_target(error, stalled)
    expect = torch.zeros(2, 12); expect[0, 0] = .2; expect[0, 4], expect[0, 5] = -.10, .20
    assert torch.allclose(target, expect)


def learner(kind, update, copies=2, replicas=2, learn=True, **kw):
    return adapter.Learner(state(std=.3), state((393, 16, 16, 16, 1), seed=3), kind=kind, update=update, copies=copies, replicas=replicas,
                           learn=learn, seed=7, device='cpu', extra=EXTRA, **kw)


def rollout(L, steps, active=None, error=None, stalled=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = L.copies * L.replicas
    active = torch.ones(n, dtype=torch.bool) if active is None else active
    for _ in range(steps):
        L.act(torch.randn(n, 45, generator=g), torch.randn(n, 393, generator=g), torch.randn(n, EXTRA, generator=g), active)
        L.observe(torch.rand(n, generator=g) * .02, torch.randn(n, 393, generator=g), torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool),
                  torch.randn(n, 12, generator=g) if error is None else error, torch.zeros(n, 4, dtype=torch.bool) if stalled is None else stalled)


def test_the_slow_part_of_the_tracking_error_is_removed_and_restarts_with_each_attempt():
    L = learner('film', 'b')
    n = 4
    a = L.target(torch.full((n, 12), .57), torch.zeros(n, 4, dtype=torch.bool), torch.ones(n, dtype=torch.bool))
    assert torch.count_nonzero(a) == 0  # 恒定的误差不产生目标
    b = L.target(torch.full((n, 12), 1.57), torch.zeros(n, 4, dtype=torch.bool), torch.ones(n, dtype=torch.bool))
    assert torch.allclose(b, torch.full((n, 12), .2 * .98))
    L.reset(torch.tensor([1]))
    c = L.target(torch.full((n, 12), 3.), torch.zeros(n, 4, dtype=torch.bool), torch.ones(n, dtype=torch.bool))
    assert torch.count_nonzero(c[1]) == 0 and torch.count_nonzero(c[0]) == 12


@pytest.mark.parametrize('kind', ['film', 'residual'])
def test_supervised_update_moves_the_residual_towards_the_target_every_step(kind):
    L = learner(kind, 'b')
    n, g = 4, torch.Generator().manual_seed(0)
    obs, cobs, extra = torch.randn(n, 45, generator=g), torch.randn(n, 393, generator=g), torch.randn(n, EXTRA, generator=g)
    stalled = torch.zeros(n, 4, dtype=torch.bool); stalled[:, 0] = True
    active = torch.tensor([True, True, False, False])
    before = L.digest(1)
    gap = []
    for _ in range(200):
        action = L.act(obs, cobs, extra, active)
        _, base = L.adapter(obs.view(2, 2, 45), extra.view(2, 2, EXTRA))
        gap.append(float((action.view(2, 2, 12) - base.detach())[0, :, 1].mean()))
        L.observe(torch.zeros(n), cobs, torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), torch.zeros(n, 12), stalled)
    assert gap[0] == 0 and gap[-1] < -.05  # 前左腿髋关节的残差朝抬腿方向（-0.10）移动
    assert L.updates.tolist() == [200, 0] and L.digest(1) == before  # 没有在走的会话不更新
    assert L.learning_rate() == pytest.approx(1e-3 / (1 + 200 / 5000))


def test_ppo_calibrates_the_critic_before_the_adapter_changes():
    L = learner('film', 'a', horizon=10, minimum=4, calibration=20)
    adapter_before = [p.detach().clone() for p in L.adapter.parameters()]
    critic_before = [p.detach().clone() for p in L.critic.parameters()]
    n = 4
    first = L.act(torch.ones(n, 45), torch.ones(n, 393), torch.ones(n, EXTRA), torch.ones(n, dtype=torch.bool))
    again = L.act(torch.ones(n, 45), torch.ones(n, 393), torch.ones(n, EXTRA), torch.ones(n, dtype=torch.bool))
    assert torch.equal(first, again)  # 校准阶段不加噪声
    L.pending = None
    rollout(L, 20)
    assert all(torch.equal(a, b) for a, b in zip(adapter_before, L.adapter.parameters())) and L.updates.tolist() == [0, 0]
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, L.critic.parameters()))
    a1 = L.act(torch.ones(n, 45), torch.ones(n, 393), torch.ones(n, EXTRA), torch.ones(n, dtype=torch.bool))
    L.pending = None
    a2 = L.act(torch.ones(n, 45), torch.ones(n, 393), torch.ones(n, EXTRA), torch.ones(n, dtype=torch.bool))
    L.pending = None
    assert not torch.equal(a1, a2) and float((a1 - a2).std()) == pytest.approx(.1 * 2**.5, rel=.5)  # 之后加标准差 0.1 的噪声
    rollout(L, 10)
    assert L.updates.tolist() == [1, 1] and any(not torch.equal(a, b) for a, b in zip(adapter_before, L.adapter.parameters()))


def test_combined_update_weights_the_supervised_term_by_a_halving_schedule():
    assert adapter.beta(0) == 1. and adapter.beta(10) == .5 and adapter.beta(20) == .25
    L = learner('residual', 'c', horizon=10, minimum=4, calibration=0)
    stalled = torch.zeros(4, 4, dtype=torch.bool); stalled[:, 0] = True
    rollout(L, 100, error=torch.zeros(4, 12), stalled=stalled)
    assert L.updates.tolist() == [10, 10] and 'supervised' in L.log[-1]
    A = learner('residual', 'a', horizon=10, minimum=4, calibration=0)
    rollout(A, 100, error=torch.zeros(4, 12), stalled=stalled)
    g = torch.Generator().manual_seed(11)
    obs, extra = torch.randn(2, 8, 45, generator=g), torch.randn(2, 8, EXTRA, generator=g)
    lift = lambda M: float((lambda a, b: (a - b)[..., 1].mean())(*M.adapter(obs, extra)))
    assert lift(L) < lift(A) - .005  # 加了有方向的误差之后，受阻腿的髋关节残差更靠近抬腿方向


@pytest.mark.parametrize('update', ['a', 'b', 'c'])
def test_snapshot_round_trip_and_frozen_evaluation(update):
    L = learner('film', update, horizon=10, minimum=4, calibration=0)
    rollout(L, 20)
    E = learner('film', update, copies=3, replicas=1, learn=False)
    zero = E.digest(0)
    E.load(2, L.snapshot(1))
    obs, cobs, extra = torch.randn(3, 45), torch.randn(3, 393), torch.randn(3, EXTRA)
    out = E.act(obs, cobs, extra, torch.ones(3, dtype=torch.bool))
    assert torch.equal(out, E.act(obs, cobs, extra, torch.ones(3, dtype=torch.bool)))  # 评估不加噪声
    expect, _ = L.adapter(obs[2].expand(2, 1, 45), extra[2].expand(2, 1, EXTRA))
    assert torch.allclose(out[2], expect[1, 0], atol=1e-6) and torch.allclose(out[0], base_forward(state(), obs[0]), atol=1e-5)
    assert E.digest(2) != zero and E.digest(0) == zero
    E.observe(torch.zeros(3), cobs, torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool), torch.zeros(3, 12), torch.zeros(3, 4, dtype=torch.bool))
    assert E.updates.tolist() == [0, 0, 0]


def test_unknown_structure_or_update_is_rejected():
    with pytest.raises(ValueError): make('lora')
    with pytest.raises(ValueError): learner('film', 'd')


def test_learning_rate_rule_can_go_one_decade_below_the_base_training_floor():
    lr = torch.tensor([2e-6, 1e-3, 1e-3])
    assert torch.allclose(adapter.adapt_lr(lr, torch.tensor([1., .001, .01])), torch.tensor([2e-6 / 1.5, 1.5e-3, 1e-3]))
    assert adapter.adapt_lr(torch.tensor([1e-6]), torch.tensor([1.])) == pytest.approx(1e-6)


def test_combined_update_applies_the_directed_step_every_control_step_after_calibration():
    L = learner('film', 'c', horizon=10, minimum=4, calibration=10)
    before = [p.detach().clone() for p in L.adapter.parameters()]
    stalled = torch.zeros(4, 4, dtype=torch.bool); stalled[:, 0] = True
    rollout(L, 10, error=torch.zeros(4, 12), stalled=stalled)
    assert all(torch.equal(a, b) for a, b in zip(before, L.adapter.parameters()))  # 校准阶段不动适配器
    rollout(L, 3, error=torch.zeros(4, 12), stalled=stalled)
    assert any(not torch.equal(a, b) for a, b in zip(before, L.adapter.parameters())) and L.updates.tolist() == [0, 0]  # 还没到 PPO 更新，已经在按误差修正
    assert L.directed.lr[0] == pytest.approx(1e-3 / (1 + 13 / 5000))


def test_directed_target_stays_inside_the_residual_limits():
    error = torch.full((1, 12), 5.); error[0, 3:6] = -5.
    target = adapter.directed_target(error, torch.ones(1, 4, dtype=torch.bool))
    assert target[0, :6].tolist() == pytest.approx([.24, .20, .20, -.24, -.20, -.20])


# ---------------------------------------------------------------- S2′：更新方式 s（有方向的误差加权重的滑动平均）和可设的幅度上限
def stalled_front_left(n=4):
    stalled = torch.zeros(n, 4, dtype=torch.bool); stalled[:, 0] = True
    return stalled


def test_averaged_update_decays_the_learning_rate_over_1000_steps():
    L = learner('residual', 's')
    assert adapter.DECAY_STEPS_S == 1000 and adapter.EMA == .996
    assert L.learning_rate() == pytest.approx(1e-3)
    rollout(L, 50)
    assert L.learning_rate() == pytest.approx(1e-3 / (1 + 50 / 1000))


def test_averaged_update_acts_with_the_running_average_of_the_trained_weights():
    L = learner('residual', 's')
    n, g = 4, torch.Generator().manual_seed(0)
    obs, cobs, extra = torch.randn(n, 45, generator=g), torch.randn(n, 393, generator=g), torch.randn(n, EXTRA, generator=g)
    active = torch.tensor([True, True, False, False])
    start = {k: v.detach().clone() for k, v in L.adapter.named_parameters()}
    expect = {k: v[0].clone() for k, v in start.items()}  # 会话 0 的慢权重，按定义逐步手算
    for _ in range(30):
        L.act(obs, cobs, extra, active)
        L.observe(torch.zeros(n), cobs, torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), torch.zeros(n, 12), stalled_front_left())
        for k, v in L.adapter.named_parameters(): expect[k] = .996 * expect[k] + .004 * v[0].detach()
    fast, slow = dict(L.adapter.named_parameters()), dict(L.average.named_parameters())
    for k in expect:
        assert torch.allclose(slow[k][0], expect[k], atol=1e-7)
        assert torch.equal(slow[k][1], start[k][1]) and torch.equal(fast[k][1], start[k][1])  # 没有在走的会话两套权重都不变
    assert not torch.allclose(fast['R2'][0], slow['R2'][0]) and L.updates.tolist() == [30, 0]
    action = L.act(obs, cobs, extra, active); L.pending = None
    by_slow, _ = L.average(obs.view(2, 2, 45), extra.view(2, 2, EXTRA))
    by_fast, _ = L.adapter(obs.view(2, 2, 45), extra.view(2, 2, EXTRA))
    assert torch.equal(action.view(2, 2, 12), by_slow) and not torch.allclose(by_slow[0], by_fast[0])  # 作用在机器人上的是慢权重


def test_averaged_snapshot_carries_the_acting_weights_into_evaluation():
    L = learner('residual', 's')
    rollout(L, 40, error=torch.zeros(4, 12), stalled=stalled_front_left())
    obs, cobs, extra = torch.randn(4, 45), torch.randn(4, 393), torch.randn(4, EXTRA)
    during = L.act(obs, cobs, extra, torch.ones(4, dtype=torch.bool)); L.pending = None
    E = learner('residual', 's', copies=4, replicas=1, learn=False)
    for i in range(4): E.load(i, L.snapshot(i // 2))
    after = E.act(obs, cobs, extra, torch.ones(4, dtype=torch.bool))
    assert torch.allclose(after, during, atol=1e-6)  # 评估用的就是训练中作用在机器人上的权重
    assert float((after - base_forward(state(), obs)).abs().max()) > 1e-4
    assert E.digest(0) == L.digest(0) and E.digest(2) == L.digest(1) and E.digest(0) != E.digest(2)


def test_plain_directed_update_still_acts_with_the_trained_weights():
    L = learner('residual', 'b')
    obs, cobs, extra = torch.randn(4, 45), torch.randn(4, 393), torch.randn(4, EXTRA)
    L.act(obs, cobs, extra, torch.ones(4, dtype=torch.bool))
    L.observe(torch.zeros(4), cobs, torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), torch.zeros(4, 12), stalled_front_left())
    action = L.act(obs, cobs, extra, torch.ones(4, dtype=torch.bool)); L.pending = None
    by_trained, base = L.adapter(obs.view(2, 2, 45), extra.view(2, 2, EXTRA))
    assert torch.equal(action.view(2, 2, 12), by_trained) and not torch.equal(by_trained, base)
    assert L.learning_rate() == pytest.approx(1e-3 / (1 + 1 / 5000))


def test_the_amplitude_limit_of_the_directed_target_can_be_doubled():
    wide = [2 * x for x in adapter.RESIDUAL_LIMITS]
    error = torch.full((1, 12), 5.); error[0, 3:6] = -5.
    target = adapter.directed_target(error, torch.ones(1, 4, dtype=torch.bool), limits=wide)
    assert target[0, :6].tolist() == pytest.approx([.48, .40, .40, -.48, -.40, -.40])
    for limits, expect in ((wide, [.48, .40, .40]), (None, [.24, .20, .20])):
        L = learner('residual', 's', **(dict(limits=limits) if limits else {}))
        none, on = torch.zeros(4, 4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)
        L.target(torch.zeros(4, 12), none, on)
        assert L.target(torch.full((4, 12), 5.), none, on)[0, :3].tolist() == pytest.approx(expect)
