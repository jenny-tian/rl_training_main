"""S2′ 只作描述的统计里不依赖记录文件的部分：失败的回合停在台阶的哪里。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
FRONTS = [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]


@pytest.fixture
def describe(monkeypatch):
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_adapter2_describe', None)
    return importlib.import_module('reference_online_adapter2_describe')


def test_risers_passed_counts_the_risers_behind_each_foot(describe):
    assert describe.risers_passed([3.3, 3.31, 2.9, 2.95], FRONTS).tolist() == [7, 7, 6, 6]
    assert describe.risers_passed([1.0, 1.21, 0.5, 0.6], FRONTS).tolist() == [0, 1, 0, 0]


@pytest.mark.parametrize('feet,place', [
    ([1.0, 1.1, 0.6, 0.7], 'before'),        # 没有一只脚过第一个立面
    ([2.0, 2.2, 1.6, 1.7], 'on_stairs'),
    ([3.3, 3.2, 2.5, 2.9], 'on_stairs'),     # 前腿都到了，一条后腿还差两级以上
    ([3.3, 3.2, 2.9, 2.95], 'last_riser'),   # 前腿都过了全部立面，后腿只差最后一级
    ([3.3, 3.2, 3.1, 2.95], 'last_riser'),
    ([3.4, 3.4, 3.1, 3.05], 'past')])        # 四只脚都过了最后一个立面
def test_failure_place_separates_the_last_riser_from_the_rest_of_the_stairs(describe, feet, place):
    assert describe.failure_place(feet, FRONTS) == place


def test_failure_place_needs_four_feet_and_some_risers(describe):
    with pytest.raises(ValueError): describe.failure_place([1., 2., 3.], FRONTS)
    with pytest.raises(ValueError): describe.failure_place([1., 2., 3., 4.], [])


def test_phase_means_split_the_executed_residual_by_joint_and_by_swing_or_stance(describe):
    # 两步、一个环境：第一步左前腿在摆动，第二步全部着地。
    effect = np.zeros((2, 1, 12)); effect[0, 0, 2] = .2; effect[1, 0, 2] = -.1; effect[:, 0, 5] = .05
    stance = np.ones((2, 1, 4), bool); stance[0, 0, 0] = False
    policy = np.ones((2, 1), bool)
    means = describe.phase_means(effect, stance, policy)
    assert means['swing']['knee'] == pytest.approx(.2) and means['swing']['hipy'] == 0.
    assert means['stance']['knee'] == pytest.approx((.1 + .05 * 2) / 7)  # 支撑的腿步数 7 个：左前 1 个，其余三条腿各 2 个
    none = describe.phase_means(effect, stance, np.zeros((2, 1), bool))
    assert none['swing']['knee'] is None and none['stance']['hipx'] is None


def test_mean_speed_uses_only_policy_steps_at_the_full_command(describe):
    # 三步、两个环境：指令从 0.2 升到 0.4；环境 1 的最后一步不在策略阶段。
    speed = np.array([[.1, .1], [.3, .2], [.5, .9]])
    command = np.array([[.2, .2], [.4, .4], [.4, .4]])
    policy = np.array([[True, True], [True, True], [True, False]])
    assert describe.mean_speed(speed, command, policy) == pytest.approx((.3 + .2 + .5) / 3)
    assert describe.mean_speed(speed, command, np.zeros((3, 2), bool)) is None


def test_retention_counts_split_a_condition_by_map_row_and_arm(describe):
    record = lambda condition, row, arm, outcome: dict(condition=condition, row=row, arm=arm, outcome=outcome)
    records = [record('slope_rough_down', 8, 'Frozen', 'timeout'), record('slope_rough_down', 8, 'Updated', 'success'),
               record('slope_rough_down', 8, 'Frozen', 'success'), record('slope_rough_down', 8, 'Updated', 'success'),
               record('slope_rough_down', 1, 'Frozen', 'success'), record('slope_rough_down', 1, 'Updated', 'stuck'),
               record('stairs_up', 8, 'Frozen', 'posture_failure'), record('stairs_up', 8, 'Updated', 'success')]
    counts = describe.retention_counts(records)
    assert counts['slope_rough_down', 8] == dict(n=2, Frozen=1, Updated=2)
    assert counts['slope_rough_down', 1] == dict(n=1, Frozen=1, Updated=0)
    assert counts['stairs_up', 8] == dict(n=1, Frozen=0, Updated=1)
    with pytest.raises(ValueError):  # 两组的回合数不相等说明记录不完整
        describe.retention_counts(records[:-1])
