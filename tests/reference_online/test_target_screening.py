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


# ---------------------------------------------------------------- 每回合的诊断量和入选判定
import torch


@pytest.fixture
def screening(monkeypatch):
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_target_screening', None)
    return importlib.import_module('reference_target_screening')


def feed(d, *, active=(True,), stance=(True,) * 4, feet=((0., 0.),) * 4, force=((0., 0.),) * 4, computed=(0.,) * 12, applied=(0.,) * 12,
         gravity=(0., 0., -1.)):
    n = len(active)
    rows = lambda x, shape, dtype=torch.float32: torch.tensor(x, dtype=dtype).reshape(shape).expand(n, *shape).clone()
    d.step(torch.tensor(active), rows(stance, (4,), torch.bool), rows(feet, (4, 2)), rows(force, (4, 2)),
           rows(computed, (12,)), rows(applied, (12,)), rows(gravity, (3,)))


def test_slip_counts_only_feet_that_stay_on_the_ground_and_move(screening):
    d = screening.EpisodeDiagnostics(1)
    d.start(torch.tensor([[True, True, True, False]]), torch.zeros(1, 4, 2))
    # FL 整步着地、水平移动 4 mm（0.2 m/s）：打滑；FR 移动 1 mm：不算；HL 这一步抬起；HR 这一步才落地。两者都不计入支撑。
    feed(d, stance=(True, True, False, True), feet=((.004, 0.), (0., .001), (.01, 0.), (.02, 0.)))
    s = d.summary(0)
    assert s['steps'] == 1 and s['stance_slip_ratio'] == pytest.approx(.5)
    assert s['stance_foot_speed'] == pytest.approx((.2 + .05) / 2)


def test_swing_stall_uses_the_stance_at_the_start_of_the_step_and_splits_front_and_rear(screening):
    d = screening.EpisodeDiagnostics(1)
    d.start(torch.tensor([[False, False, False, True]]), torch.zeros(1, 4, 2))
    # FL 摆动、前进 1 mm（0.05 m/s）：受阻；FR 摆动、前进 4 mm：不受阻；HL 摆动、后退：受阻；HR 着地：不计。
    feed(d, stance=(False, False, False, True), feet=((.001, 0.), (.004, 0.), (-.002, 0.), (0., 0.)))
    s = d.summary(0)
    assert s['swing_stall_ratio_front'] == pytest.approx(.5) and s['swing_stall_ratio_rear'] == pytest.approx(1.)
    # 第二步用第一步结束时的着地状态：FL 仍在摆动、前进 6 mm，FR 也在摆动、前进 6 mm，前腿累计 4 步里受阻 1 步。
    feed(d, stance=(True, True, True, True), feet=((.007, 0.), (.010, 0.), (-.002, 0.), (0., 0.)))
    s = d.summary(0)
    assert s['steps'] == 2 and s['swing_stall_ratio_front'] == pytest.approx(1 / 4) and s['swing_stall_ratio_rear'] == pytest.approx(1.)


def test_swing_hit_needs_horizontal_force_without_weight_bearing(screening):
    d = screening.EpisodeDiagnostics(1)
    d.start(torch.zeros(1, 4, dtype=torch.bool), torch.zeros(1, 4, 2))
    # FL 没有承重、水平受力 10 N：碰撞；FR 水平 2 N：不算；HL 承重、水平 30 N：不算；HR 悬空。
    feed(d, stance=(False, False, True, False), force=((-10., 0.), (2., 0.), (30., 0.), (0., 0.)))
    assert d.summary(0)['swing_hit_ratio'] == pytest.approx(1 / 3)


def test_torque_limit_and_tilt(screening):
    d = screening.EpisodeDiagnostics(1)
    d.start(torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4, 2))
    feed(d, computed=(0.,) * 11 + (30.,), applied=(0.,) * 11 + (24.,), gravity=(.5, 0., -.8660254))
    feed(d, computed=(5.,) * 12, applied=(5.,) * 12, gravity=(0., .1736482, -.9848078))
    s = d.summary(0)
    assert s['torque_limit_ratio'] == pytest.approx(.5) and s['max_tilt_deg'] == pytest.approx(30., abs=1e-3)


def test_inactive_environments_do_not_accumulate_and_empty_ratios_are_none(screening):
    d = screening.EpisodeDiagnostics(2)
    d.start(torch.ones(2, 4, dtype=torch.bool), torch.zeros(2, 4, 2))
    feed(d, active=(True, False), feet=((.01, 0.),) * 4)
    a, b = d.summary(0), d.summary(1)
    assert a['steps'] == 1 and a['stance_slip_ratio'] == 1. and a['swing_stall_ratio_front'] is None and a['swing_hit_ratio'] is None
    assert b == dict(steps=0, stance_slip_ratio=None, stance_foot_speed=None, swing_stall_ratio_front=None, swing_stall_ratio_rear=None,
                     swing_hit_ratio=None, torque_limit_ratio=None, max_tilt_deg=None)
    with pytest.raises(ValueError): d.step(torch.tensor([True]), *[torch.zeros(2, 4)] * 6)


def records(success=0, posture=0, lateral=0, stuck=0, timeout=0, start=0, progress=.6):
    names = dict(success=success, posture_failure=posture, lateral_exit=lateral, stuck=stuck, timeout=timeout, start_failure=start)
    return [dict(outcome=name, progress=0. if name == 'start_failure' else 1. if name == 'success' else progress) for name, k in names.items() for _ in range(k)]


def test_candidate_stats_and_the_registered_rule(screening):
    stats = screening.candidate_stats(records(success=20, posture=10, lateral=4, stuck=20, timeout=10))
    assert stats['n'] == 64 and stats['success'] == 20 and stats['lateral_exit'] == 4 and stats['start_failure'] == 0
    assert stats['mean_progress'] == pytest.approx((20 + 44 * .6) / 64)
    classify = lambda **k: screening.classify(screening.candidate_stats(records(**k)))
    assert classify(success=6, stuck=58) == 'eligible' and classify(success=45, stuck=19) == 'eligible'
    assert classify(success=5, stuck=59) == 'too_hard' and classify(success=46, stuck=18) == 'too_easy'
    assert classify(success=10, stuck=54, progress=.1) == 'too_hard'            # 平均推进低于 30%
    assert classify(success=10, lateral=28, stuck=26) == 'excluded'              # 横向越界超过失败数的一半
    assert classify(success=10, lateral=27, stuck=27) == 'eligible'
    assert classify(success=10, stuck=47, start=7) == 'excluded' and classify(success=10, stuck=48, start=6) == 'eligible'
    with pytest.raises(ValueError): screening.candidate_stats(records(success=3) + [dict(outcome='unknown', progress=0.)])


def test_selection_takes_the_candidate_closest_to_twenty_successes_or_reports_the_bracket(screening):
    stats = lambda **k: screening.candidate_stats(records(**k))
    chains = dict(a=[['a1', 'a2', 'a3']], b=[['b1', 'b2']], c=[['c1', 'c2'], ['d1', 'd2']], e=[['e1', 'e2']])
    result = screening.select(dict(
        a1=stats(success=50, stuck=14), a2=stats(success=30, stuck=34), a3=stats(success=12, stuck=52),
        b1=stats(success=60, stuck=4), b2=stats(success=1, stuck=63),
        c1=stats(success=64), c2=stats(success=58, stuck=6), d1=stats(success=40, stuck=24), d2=stats(success=0, posture=64),
        e1=stats(success=2, stuck=62), e2=stats(success=0, stuck=64)), chains)
    assert result['a'] == dict(chosen='a3', bracket=None, classes=dict(a1='too_easy', a2='eligible', a3='eligible'))
    assert result['b'] == dict(chosen=None, bracket=('b1', 'b2'), classes=dict(b1='too_easy', b2='too_hard'))
    assert result['c']['chosen'] == 'd1' and result['c']['bracket'] is None
    assert result['e'] == dict(chosen=None, bracket=None, classes=dict(e1='too_hard', e2='too_hard'))


def test_chains_cover_every_screened_candidate_once(tool, screening):
    names = [name for chains in screening.CHAINS.values() for chain in chains for name in chain]
    friction = [c for band in tool.SCREEN_FRICTION for c in tool.RUNS[f'screen_friction_{band}']['conditions']]
    assert sorted(names + list(screening.REFERENCE)) == sorted(list(tool.SCREEN) + friction) and len(set(names)) == len(names)
    assert [chain[0] for chain in screening.CHAINS['friction']] == ['friction_lateral_1020', 'friction_longitudinal_1020']


def test_start_failure_record(screening):
    spec = dict(env_id=3, condition='rough_15', slot=3, row=0, namespace=48, offset=[.2, 0., 0.])
    record = screening.start_failure_record(spec, .25)
    assert record['outcome'] == 'start_failure' and record['progress'] == 0. and record['duration_s'] == 0. and record['final_x'] == .25
    assert record['condition'] == 'rough_15' and record['diagnostics'] is None


# ---------------------------------------------------------------- 地形核对
X = np.linspace(0., 3.8, 381)
ramp = lambda slope: slope * np.clip(X - 1.75, 0., None)


def test_slope_audit_measures_the_slope_beyond_the_centre_platform(tool):
    up = tool.terrain_audit('slope_up_60', X, ramp(.6))
    assert up['measured_slope'] == pytest.approx(.6, abs=1e-6) and up['slope_start_x'] == pytest.approx(1.79, abs=.011)
    assert tool.terrain_audit('slope_down_50', X, ramp(-.5))['measured_slope'] == pytest.approx(-.5, abs=1e-6)
    noise = np.repeat(np.random.default_rng(0).integers(1, 8, 39) / 100., 10)[:381]
    rough = tool.terrain_audit('slope_rough_down_x11', X, ramp(-.44) + noise)
    assert rough['measured_slope'] == pytest.approx(-.44, abs=.066)


@pytest.mark.parametrize('name,profile', [('slope_down_50', ramp(.5)), ('slope_up_60', ramp(.5)), ('slope_up_60', np.zeros(381))])
def test_slope_audit_rejects_the_wrong_direction_or_steepness(tool, name, profile):
    with pytest.raises(ValueError):
        tool.terrain_audit(name, X, profile)


def test_rough_audit_checks_the_height_range_on_the_centre_line(tool):
    heights = np.repeat(np.random.default_rng(1).integers(1, 13, 39) / 100., 10)[:381]
    assert tool.terrain_audit('rough_12', X, heights)['measured_height_range'] == pytest.approx(.11, abs=.011)
    for bad in (np.zeros(381), heights * 2.):
        with pytest.raises(ValueError):
            tool.terrain_audit('rough_12', X, bad)


def test_friction_audit_requires_flat_ground_and_other_terrain_has_no_audit(tool):
    assert tool.terrain_audit('friction_lateral_0512', X, np.zeros(381)) == {}
    with pytest.raises(ValueError):
        tool.terrain_audit('friction_lateral_0512', X, ramp(.1))
    assert tool.terrain_audit('stairs_down_09', X, ramp(-.3)) == {} and tool.terrain_audit('boxes_07', X, ramp(.1)) == {}


def test_path_coefficients_follow_the_band_layout(tool):
    lateral = [dict(bounds_xy_m=[0., 8., 2. * k, 2. * k + 2.], coefficient=c) for k, c in enumerate((.11, .05, .12, .07))]
    assert tool.path_coefficients(lateral, (8., 8.)) == dict(forward_m=[.3, 1.5, 2.5, 3.2], left=[.12] * 4, right=[.05] * 4)
    longitudinal = [dict(bounds_xy_m=[2. * k, 2. * k + 2., 0., 8.], coefficient=c) for k, c in enumerate((.11, .05, .12, .07))]
    assert tool.path_coefficients(longitudinal, (8., 8.)) == dict(forward_m=[.3, 1.5, 2.5, 3.2], left=[.12, .12, .07, .07], right=[.12, .12, .07, .07])
    with pytest.raises(ValueError):
        tool.path_coefficients(longitudinal[:3], (8., 8.))
