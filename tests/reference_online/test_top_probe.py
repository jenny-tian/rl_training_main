"""上阶最后一级的脚本化检查里的手工动作。方案见 docs/lite3-online-s2-20261127/PLAN.md 第 5 节。"""
import pytest
import torch

from rl_training.ea_lpl import top_probe

torch.set_default_dtype(torch.float32)
ALL = torch.ones(1, 4, dtype=torch.bool)   # FL FR HL HR 都着地


def test_the_registered_groups_and_their_filter_setting():
    assert set(top_probe.PROBES) == {'none', 'hind_half', 'hind_full', 'all_half', 'h15', 'h15_hind_half'}
    assert {name: top_probe.min_base_height(name) for name in top_probe.PROBES} == dict(
        none=.20, hind_half=.20, hind_full=.20, all_half=.20, h15=.15, h15_hind_half=.15)
    with pytest.raises(ValueError): top_probe.min_base_height('hind')


@pytest.mark.parametrize('name', ['none', 'h15'])
def test_groups_without_an_action_request_nothing(name):
    assert torch.count_nonzero(top_probe.request(name, ALL, torch.ones(1))) == 0


def test_hind_half_extends_only_grounded_hind_legs():
    r = top_probe.request('hind_half', ALL, torch.ones(1))[0].view(4, 3)   # 每条腿是髋侧摆、髋俯仰、膝
    assert torch.count_nonzero(r[:2]) == 0                                 # 前腿不动
    assert r[2].tolist() == pytest.approx([0., .05, -.10]) and r[3].tolist() == pytest.approx([0., .05, -.10])
    swinging = torch.tensor([[True, True, False, True]])                   # 左后腿在摆动
    r = top_probe.request('hind_half', swinging, torch.ones(1))[0].view(4, 3)
    assert torch.count_nonzero(r[2]) == 0 and r[3].tolist() == pytest.approx([0., .05, -.10])


def test_full_amplitude_and_all_legs():
    full = top_probe.request('hind_full', ALL, torch.ones(1))[0].view(4, 3)
    assert full[2].tolist() == pytest.approx([0., .10, -.20]) and torch.count_nonzero(full[:2]) == 0
    everything = top_probe.request('all_half', ALL, torch.ones(1))[0].view(4, 3)
    assert all(leg.tolist() == pytest.approx([0., .05, -.10]) for leg in everything)
    assert torch.equal(top_probe.request('h15_hind_half', ALL, torch.ones(1)), top_probe.request('hind_half', ALL, torch.ones(1)))


def test_the_action_scales_with_the_top_feature_and_vanishes_away_from_the_top():
    stance = ALL.expand(3, 4)
    r = top_probe.request('hind_full', stance, torch.tensor([0., .5, 1.]))
    assert torch.count_nonzero(r[0]) == 0
    assert torch.allclose(r[1], .5 * r[2]) and r[2, 8] == pytest.approx(-.20)


def test_the_extension_is_the_opposite_of_the_swing_lift_and_stays_inside_the_limits():
    from rl_training.ea_lpl import adapter, c1b_s
    lift = c1b_s.residual_request(torch.tensor([[0., 0., 9., 9.]]), torch.zeros(1, 4, dtype=torch.bool), torch.ones(1))[0]  # 四条腿摆动时的满幅抬腿
    extend = top_probe.request('hind_full', ALL, torch.ones(1))[0]
    assert torch.allclose(extend[6:], -lift[6:], atol=1e-4)
    limits = torch.tensor(adapter.RESIDUAL_LIMITS)
    for name in top_probe.PROBES:
        assert (top_probe.request(name, ALL, torch.ones(1))[0].abs() <= limits + 1e-6).all()


def test_wrong_inputs_are_rejected():
    with pytest.raises(ValueError): top_probe.request('hind_half', torch.ones(2, 4), torch.ones(2))        # 不是布尔
    with pytest.raises(ValueError): top_probe.request('hind_half', ALL, torch.ones(2))                      # 形状不符
    with pytest.raises(ValueError): top_probe.request('hind_half', ALL, torch.tensor([1.5]))                # 特征超出 0 到 1
    with pytest.raises(ValueError): top_probe.request('unknown', ALL, torch.ones(1))
