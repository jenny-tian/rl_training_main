"""S0 目标筛选里不依赖仿真的部分。方案见 docs/lite3-online-targets-20261125/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_frozen_rebaseline', None)
    return importlib.import_module('reference_frozen_rebaseline')


def test_candidate_names_give_the_registered_terrain_parameters(tool):
    spec = tool.terrain_spec
    assert spec('rough_12') == dict(source='rough', noise_range=(.01, .12))
    assert spec('slope_up_60') == dict(source='slope_up', slope_range=(.6, .6))
    assert spec('slope_down_70') == dict(source='slope_down', slope_range=(.7, .7))
    assert spec('slope_rough_down_x085') == dict(source='slope_rough_down', slope_range=(.34, .34), noise_range=(.01, .06))
    assert spec('slope_rough_down_x10') == dict(source='slope_rough_down', slope_range=(.4, .4), noise_range=(.01, .06))
    assert spec('slope_rough_down_x11') == dict(source='slope_rough_down', slope_range=(.44, .44), noise_range=(.01, .07))
    assert spec('stairs_down_09') == dict(source='stairs_down', step_height_range=(.09, .09))
    assert spec('friction_lateral_1020') == dict(source='friction', friction_layout='left_right_bands', friction_range=(.1, .2))
    assert spec('friction_longitudinal_0206') == dict(source='friction', friction_layout='front_back_bands', friction_range=(.02, .06))


def test_earlier_candidates_keep_their_parameters(tool):
    spec = tool.terrain_spec
    assert spec('stairs_up_065') == dict(source='stairs_up', step_height_range=(.065, .065))
    assert spec('stairs_down_10') == dict(source='stairs_down', step_height_range=(.1, .1))
    assert spec('boxes_08') == dict(source='boxes', grid_height_range=(.08, .08))
    assert spec('slope_rough_up_x11') == dict(source='slope_rough_up', slope_range=(.44, .44), noise_range=(.01, .07))
    assert spec('slope_rough_up_x13') == dict(source='slope_rough_up', slope_range=(.52, .52), noise_range=(.02, .08))


@pytest.mark.parametrize('name', ['gravel_10', 'slope_up_x11', 'rough_1', 'friction_lateral_2010', 'friction_diagonal_1020', 'slope_rough_up_x99'])
def test_unknown_or_malformed_candidates_are_rejected(tool, name):
    with pytest.raises(ValueError):
        tool.terrain_spec(name)


def test_screening_covers_the_registered_candidates(tool):
    assert tool.SCREEN == ('rough_09', 'rough_12', 'rough_15', 'slope_up_50', 'slope_up_60', 'slope_up_70',
                           'slope_down_50', 'slope_down_60', 'slope_down_70', 'slope_rough_up_x11', 'slope_rough_up_x12',
                           'slope_rough_down_x085', 'slope_rough_down_x10', 'slope_rough_down_x11',
                           'stairs_down_09', 'stairs_down_10', 'stairs_up_07', 'boxes_07')
    assert tool.SCREEN_FRICTION == ('1020', '0512', '0206')
    for name in tool.SCREEN: tool.terrain_spec(name)
    for band in tool.SCREEN_FRICTION:
        run = tool.RUNS[f'screen_friction_{band}']
        assert run['conditions'] == (f'friction_lateral_{band}', f'friction_longitudinal_{band}')
        # 同一张地图里的摩擦分区只能用同一组系数
        assert len({tool.terrain_spec(c)['friction_range'] for c in run['conditions']}) == 1


def test_screening_runs_use_new_starts_and_a_separate_output(tool):
    specs = tool.specs_for('screen')
    assert len(specs) == 18 * 64 and [s['env_id'] for s in specs] == list(range(len(specs)))
    assert {s['namespace'] for s in specs} == {48} and {s['row'] for s in specs} == {0}
    assert [s['condition'] for s in specs[::64]] == list(tool.SCREEN)
    offsets = np.asarray([s['offset'] for s in specs])
    assert (offsets >= [.2, -.1, -.03]).all() and (offsets <= [.3, .1, .03]).all()
    assert len({tuple(o) for o in offsets.tolist()}) == len(specs)
    # 与 11 月 17 日加密筛选（命名空间 42）的起点不同
    assert specs[9 * 64]['condition'] == 'slope_rough_up_x11' and specs[9 * 64]['offset'] != tool.start_offset('slope_rough_up_x11', 0, 42)
    for name in ('screen', 'screen_friction_0512'):
        run = tool.RUNS[name]
        assert (run['layout'], run['profile'], run['heading'], run.get('residual')) == ('fixed', 'nominal', True, None)
        assert run['diagnostics'] and run['tolerate_start_failure'] and run.get('starts', tool.STARTS) == 64
        assert tool.output_of(name).parts[-2:] == ('target_screening_20261125', name)
        smoke = tool.RUNS[name + '_smoke']
        assert smoke['conditions'] == run['conditions'] and smoke['starts'] == 4 and smoke['namespace'] == 49
        assert tool.output_of(name + '_smoke').parts[-2:] == ('target_screening_20261125', name + '_smoke')
    assert len(tool.specs_for('screen_friction_0206')) == 2 * 64


def test_earlier_runs_keep_their_output_and_starts(tool):
    assert tool.output_of('dense_nominal_on').parts[-2:] == ('frozen_rebaseline_20261117', 'dense_nominal_on')
    specs = tool.specs_for('dense_nominal_on')
    assert len(specs) == 6 * 64 and {s['namespace'] for s in specs} == {42}
    assert specs[0]['offset'] == tool.start_offset('stairs_up_065', 0, 42)
    assert not tool.RUNS['dense_nominal_on'].get('diagnostics')
