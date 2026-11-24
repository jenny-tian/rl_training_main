"""直接微调基础策略的对照（全参数与秩 1 LoRA）里不依赖仿真的部分。方案见 docs/lite3-online-finetune-20261124/PLAN.md。"""
import numpy as np
import pytest
import torch

from rl_training.ea_lpl import finetune

torch.set_default_dtype(torch.float32)


def reference_state(sizes=(5, 8, 6, 3), seed=0):
    g = torch.Generator().manual_seed(seed)
    state = {}
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        state[f'mlp.{2 * k}.weight'] = torch.randn(b, a, generator=g) * .5
        state[f'mlp.{2 * k}.bias'] = torch.randn(b, generator=g) * .1
    return state


def reference_forward(state, x):
    layers = sorted({int(k.split('.')[1]) for k in state})
    for n, k in enumerate(layers):
        x = x @ state[f'mlp.{k}.weight'].T + state[f'mlp.{k}.bias']
        if n < len(layers) - 1: x = torch.nn.functional.elu(x)
    return x


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_every_copy_starts_as_the_base_network(mode):
    state = reference_state()
    net = finetune.StackedMLP(state, copies=3, mode=mode, generator=torch.Generator().manual_seed(1))
    x = torch.randn(3, 4, 5)
    assert torch.allclose(net(x), reference_forward(state, x), atol=1e-6)


def test_lora_adds_a_rank_one_branch_before_the_activation_and_trains_only_the_branch():
    state = reference_state()
    net = finetune.StackedMLP(state, copies=2, mode='lora', rank=1, alpha=1., generator=torch.Generator().manual_seed(1))
    names = [n for n, p in net.named_parameters() if p.requires_grad]
    assert names and all(n.startswith(('A.', 'B.')) for n in names)
    with torch.no_grad():
        for b in net.B: b.normal_(generator=torch.Generator().manual_seed(2))
    x = torch.randn(2, 4, 5)
    for s in range(2):
        merged = {}
        for k, layer in enumerate((0, 2, 4)):
            delta = net.B[k][s] @ net.A[k][s]
            assert torch.linalg.matrix_rank(delta) == 1
            merged[f'mlp.{layer}.weight'] = state[f'mlp.{layer}.weight'] + delta
            merged[f'mlp.{layer}.bias'] = state[f'mlp.{layer}.bias']
        assert torch.allclose(net(x)[s], reference_forward(merged, x[s]), atol=1e-5)


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_copies_do_not_share_gradients(mode):
    net = finetune.StackedMLP(reference_state(), copies=2, mode=mode, generator=torch.Generator().manual_seed(1))
    if mode == 'lora':
        with torch.no_grad():
            for b in net.B: b.fill_(.1)
    net(torch.randn(2, 4, 5))[0].sum().backward()
    for p in net.parameters():
        if p.requires_grad:
            assert p.grad[0].abs().sum() > 0 and p.grad[1].abs().sum() == 0


def test_stacked_adam_equals_one_adam_per_copy_and_leaves_inactive_copies_alone():
    g = torch.Generator().manual_seed(0)
    start = torch.randn(3, 4, 2, generator=g)
    stacked = torch.nn.Parameter(start.clone())
    lrs = torch.tensor([1e-2, 3e-3, 1e-1])
    optimizer = finetune.StackedAdam([stacked], lrs)
    singles = [torch.nn.Parameter(start[s].clone()) for s in range(3)]
    references = [torch.optim.Adam([p], lr=float(lr)) for p, lr in zip(singles, lrs)]
    active = torch.tensor([True, True, False])
    for step in range(5):
        target = torch.randn(3, 4, 2, generator=g)
        optimizer.zero_grad(); ((stacked - target)**2).sum().backward(); optimizer.step(active)
        for s in range(2):
            references[s].zero_grad(); ((singles[s] - target[s])**2).sum().backward(); references[s].step()
    for s in range(2): assert torch.allclose(stacked[s], singles[s], atol=1e-6)
    assert torch.equal(stacked[2].detach(), start[2])


def test_gradient_clipping_is_applied_to_each_copy_separately():
    a, b = torch.nn.Parameter(torch.zeros(2, 3)), torch.nn.Parameter(torch.zeros(2, 4))
    a.grad = torch.tensor([[3., 0., 0.], [.3, 0., 0.]]); b.grad = torch.tensor([[0., 4., 0., 0.], [0., .4, 0., 0.]])
    finetune.clip_per_copy([a, b], 1.)
    assert torch.allclose(a.grad[0], torch.tensor([.6, 0., 0.])) and torch.allclose(b.grad[0], torch.tensor([0., .8, 0., 0.]))
    assert torch.allclose(a.grad[1], torch.tensor([.3, 0., 0.])) and torch.allclose(b.grad[1], torch.tensor([0., .4, 0., 0.]))


def test_learning_rate_follows_the_kl_rule_used_in_training():
    lr = torch.tensor([1e-3, 1e-3, 1e-3, 1e-5, 1e-2])
    kl = torch.tensor([.03, .004, .01, .03, .001])
    assert torch.allclose(finetune.adapt_lr(lr, kl, desired=.01), torch.tensor([1e-3 / 1.5, 1.5e-3, 1e-3, 1e-5, 1e-2]))


def test_gae_bootstraps_at_cutoffs_but_not_at_posture_failures_and_ignores_invalid_steps():
    gamma, lam = .9, .5
    reward = torch.tensor([[1.], [1.], [1.], [1.], [1.]])
    value = torch.tensor([[2.], [2.], [2.], [2.], [2.]])
    next_value = torch.tensor([[2.], [3.], [9.], [2.], [4.]])
    done = torch.tensor([[False], [True], [False], [True], [False]])
    failed = torch.tensor([[False], [False], [False], [True], [False]])
    valid = torch.tensor([[True], [True], [False], [True], [True]])
    returns, advantage = finetune.gae(reward, value, next_value, done, failed, valid, gamma, lam)
    d4 = 1 + gamma * 4 - 2          # 缓冲区结尾：用下一状态的价值自举
    d3 = 1 - 2                      # 姿态终止：不自举
    d1 = 1 + gamma * 3 - 2          # 其他方式结束：自举，但不接后面的回合
    d0 = 1 + gamma * 2 - 2
    expected = torch.tensor([[d0 + gamma * lam * d1], [d1], [0.], [d3], [d4]])
    assert torch.allclose(advantage, expected, atol=1e-6)
    assert torch.allclose(returns, expected + value * valid, atol=1e-6)


def make_tuner(mode, *, learn=True, copies=2, replicas=4, horizon=40, lr=None, seed=0):
    actor = dict(reference_state((5, 16, 16, 3), 1), **{'distribution.log_std_param': torch.full((3,), -1.)})
    critic = reference_state((7, 16, 16, 1), 2)
    return finetune.Tuner(actor, critic, copies=copies, replicas=replicas, mode=mode, learn=learn, seed=seed,
                          device='cpu', lr=lr or (3e-3 if mode == 'full' else 1e-2), horizon=horizon, minimum=16)


def run(tuner, steps, *, active=None, seed=0):
    """奖励是第 0 维动作的值；每 10 步回合结束一次。"""
    g = torch.Generator().manual_seed(seed)
    n = tuner.copies * tuner.replicas
    active = torch.ones(n, dtype=torch.bool) if active is None else active
    for t in range(steps):
        obs, cobs = torch.randn(n, 5, generator=g), torch.randn(n, 7, generator=g)
        action = tuner.act(obs, cobs, active, sample=tuner.learn)
        done = torch.full((n,), t % 10 == 9)
        tuner.record(action[:, 0].clone(), torch.randn(n, 7, generator=g), done, torch.zeros(n, dtype=torch.bool))
        if tuner.ready(): tuner.update()


def mean_action(tuner, seed=5):
    g = torch.Generator().manual_seed(seed)
    n = tuner.copies * tuner.replicas
    obs = torch.randn(64, n, 5, generator=g)
    return torch.stack([tuner.act(o, torch.zeros(n, 7), torch.zeros(n, dtype=torch.bool), sample=False) for o in obs]).mean((0,)).view(tuner.copies, tuner.replicas, 3).mean(1)


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_ppo_updates_move_the_action_towards_higher_reward(mode):
    tuner = make_tuner(mode)
    before = mean_action(tuner)
    run(tuner, 1200)
    after = mean_action(tuner)
    assert (tuner.updates == 30).all()
    assert (after[:, 0] > before[:, 0] + .1).all()


def test_actions_are_the_mean_when_not_sampling_and_the_base_action_before_any_update():
    tuner = make_tuner('lora', learn=False)
    obs = torch.randn(8, 5)
    action = tuner.act(obs, torch.zeros(8, 7), torch.ones(8, dtype=torch.bool), sample=False)
    assert torch.allclose(action, reference_forward(reference_state((5, 16, 16, 3), 1), obs), atol=1e-6)


def test_a_copy_with_too_few_valid_steps_is_not_updated():
    tuner = make_tuner('full')
    active = torch.tensor([True] * 4 + [False] * 4)
    start = tuner.digest(1)
    run(tuner, 80, active=active)
    assert tuner.updates.tolist() == [2, 0] and tuner.digest(1) == start and tuner.digest(0) != tuner.digest(1)


def test_nothing_changes_without_learning():
    tuner = make_tuner('full', learn=False)
    start = [tuner.digest(s) for s in range(2)]
    run(tuner, 80)
    assert [tuner.digest(s) for s in range(2)] == start and tuner.updates.tolist() == [0, 0]


@pytest.mark.parametrize('mode', ['full', 'lora'])
def test_a_snapshot_reproduces_the_tuned_mean_in_an_evaluation_tuner(mode):
    tuner = make_tuner(mode)
    run(tuner, 200)
    saved = [tuner.snapshot(s) for s in range(2)]
    evaluation = make_tuner(mode, learn=False, copies=4, replicas=1, seed=9)
    for k, s in enumerate((0, 1, 1, 0)): evaluation.load(k, saved[s])
    obs = torch.randn(8, 5)
    trained = tuner.act(obs, torch.zeros(8, 7), torch.zeros(8, dtype=torch.bool), sample=False).view(2, 4, 3)
    again = evaluation.act(obs[[0, 4, 5, 1]], torch.zeros(4, 7), torch.zeros(4, dtype=torch.bool), sample=False)
    assert torch.allclose(again, torch.stack([trained[0, 0], trained[1, 0], trained[1, 1], trained[0, 1]]), atol=1e-6)
    assert all(v.dtype == torch.float32 and v.device.type == 'cpu' for v in saved[0]['actor'].values())


def test_frozen_groups_request_nothing_and_updated_groups_expose_their_copy():
    tuner = make_tuner('lora')
    frozen, updated = finetune.Group(None, None), finetune.Group(tuner, 1)
    slot = finetune.Slot(updated, 2)
    assert np.array_equal(slot.begin(None), np.zeros(2)) and np.array_equal(slot.requested_code(), np.zeros(2))
    assert slot.finish(1.5, learn=True) == {} and slot.batch == [] and frozen.updates == 0
    run(tuner, 40)
    assert updated.updates == 1 and slot.updates == 1 and updated.digest() == tuner.digest(1)
    assert frozen.snapshot() == dict(mode='frozen') and frozen.digest() == 'frozen'
    assert updated.snapshot()['mode'] == 'lora' and updated.snapshot()['updates'] == 1
