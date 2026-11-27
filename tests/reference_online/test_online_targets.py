"""在线入口里超出原有上阶和箱体的目标：坡面目标的参数、下行台阶的高度和射线核对。方案见 docs/lite3-online-adapter2-20261126/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
X = np.linspace(0., 3.8, 3401)


@pytest.fixture
def targets(monkeypatch):
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2_targets', None)
    return importlib.import_module('reference_online_c1a2_targets')


def staircase(height, first=1.2, tread=.3, steps=7):
    """中心线上的台阶剖面：从 first 开始每隔 tread 一个立面，每级高 height（负数是下行）。"""
    return height * (X[:, None] > first + tread * np.arange(steps)).sum(1)


def test_descending_stairs_give_riser_positions_and_the_height_after_each_riser(targets):
    fronts, tops = targets.stair_profile(X, staircase(-.085), -.085)
    assert len(fronts) == 7 and np.allclose(fronts, 1.2 + .3 * np.arange(7), atol=.002)
    assert np.allclose(tops, -.085 * np.arange(1, 8), atol=1e-9)


def test_ascending_stairs_use_the_same_check_with_a_positive_height(targets):
    fronts, tops = targets.stair_profile(X, staircase(.07), .07)
    assert np.allclose(fronts, 1.2 + .3 * np.arange(7), atol=.002) and np.allclose(tops, .07 * np.arange(1, 8))


@pytest.mark.parametrize('profile,height', [(staircase(.085), -.085),   # 方向相反：上行的地形按下行核对
                                            (staircase(-.085), .085),
                                            (staircase(-.09), -.085),   # 每级高度不符
                                            (staircase(-.085, steps=4), -.085),  # 级数不足
                                            (np.zeros_like(X), -.085)])
def test_a_profile_that_does_not_match_the_registered_stairs_is_rejected(targets, profile, height):
    with pytest.raises(ValueError):
        targets.stair_profile(X, profile, height)


def test_slope_and_downstairs_targets_use_the_screened_terrain_parameters(targets, monkeypatch):
    sys.modules.pop('reference_frozen_rebaseline', None)
    screening = importlib.import_module('reference_frozen_rebaseline')
    assert {'slope_rough_up_x11', 'slope_rough_down_x10'} <= set(targets.SLOPE_TARGETS)  # S2 又加了候选，见 test_s2_targets.py
    for name, spec in targets.SLOPE_TARGETS.items():
        assert spec == {k: tuple(v) if isinstance(v, (tuple, list)) else v for k, v in screening.terrain_spec(name).items()}
    # 单地块核对之后粗糙下坡的目标从坡度 0.34 改为 0.40（方案第 11 节）
    assert targets.SLOPE_TARGETS['slope_rough_down_x10'] == dict(source='slope_rough_down', slope_range=(.40, .40), noise_range=(.01, .06))
    assert targets.DOWN_STAIR_HEIGHT['stairs_down_085'] == .085  # S2 又加了 9 cm 和 10 cm，见 test_s2_targets.py
    assert screening.terrain_spec('stairs_down_085') == dict(source='stairs_down', step_height_range=(.085, .085))


def test_stair_targets_are_recognised_by_name(targets):
    assert targets.stairs_target('stairs_down_085') and targets.stairs_target('stairs_up_07') and not targets.stairs_target('slope_rough_down_x10')


def training_map():
    """训练地图里用到的三个子地形的参数（只有数据，和 reference_six_terrain_cfg 里的范围相同）。"""
    from types import SimpleNamespace
    slope = lambda: SimpleNamespace(slope_range=(0., .4), noise_range=(.01, .06), proportion=.05)
    return dict(slope_rough_up=slope(), slope_rough_down=slope(), stairs_down=SimpleNamespace(step_height_range=(.05, .23), proportion=.10))


@pytest.mark.parametrize('name,source,expect', [
    ('slope_rough_up_x11', 'slope_rough_up', dict(slope_range=(.44, .44), noise_range=(.01, .07))),
    ('slope_rough_down_x10', 'slope_rough_down', dict(slope_range=(.40, .40), noise_range=(.01, .06))),
    ('stairs_down_085', 'stairs_down', dict(step_height_range=(.085, .085)))])
def test_a_target_becomes_the_only_sub_terrain_with_its_parameters_fixed(targets, name, source, expect):
    captured = training_map()
    result = targets.target_sub_terrains(name, captured)
    assert list(result) == [name] and result[name] is captured[source] and result[name].proportion == 1.
    for key, value in expect.items(): assert getattr(result[name], key) == value


@pytest.mark.parametrize('name', ['stairs_up_07', 'boxes_07', 'stairs_up_06', 'slope_rough_down_x085'])
def test_targets_handled_elsewhere_are_left_alone(targets, name):
    assert targets.target_sub_terrains(name, training_map()) is None


def test_a_slope_target_refuses_a_map_whose_training_range_changed(targets):
    captured = training_map()
    captured['slope_rough_down'].slope_range = (0., .5)
    with pytest.raises(ValueError):
        targets.target_sub_terrains('slope_rough_down_x10', captured)
