"""S2′ 的判定（docs/lite3-online-adapter2-20261126/PLAN.md 第 6 节）：哪个变体看哪几条，每条的门槛。"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
DOWN = ('slope_rough_down_x10', 'stairs_down_085')
OK = dict(target_gain_at_least_10pp=True, ci_lower_above_zero=True, posture_increase_at_most_2pp=True,
          every_family_loss_at_most_5pp=True, flat_velocity_ratio_at_most_1p10=True)


@pytest.fixture
def results(monkeypatch):
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_adapter_results', None)
    return importlib.import_module('reference_online_adapter_results')


def s(delta, low, high):
    return dict(mean_session_delta=delta, ci95=[low, high])


def good(**changes):
    return dict(dict(gain=s(.30, .20, .40), late_minus_early=s(.02, -.06, .10), minus_r2=s(.08, .01, .15), five=OK), **changes)


@pytest.mark.parametrize('variant', ['mrs', 'mrsz'])
@pytest.mark.parametrize('target', DOWN)
def test_downhill_runs_are_judged_on_all_four_criteria(results, variant, target):
    assert results.verdict2(variant, target, **good()) == dict(effective=True, not_worse=True, beats_stall_correction=True, retention=True, passed=True)
    lost = results.verdict2(variant, target, **good(minus_r2=s(.08, -.01, .15)))
    assert lost['beats_stall_correction'] is False and lost['passed'] is False


def test_effective_needs_ten_points_and_an_interval_above_zero(results):
    judge = lambda gain: results.verdict2('mrs', 'stairs_down_085', **good(gain=gain))['effective']
    assert judge(s(.10, .01, .19)) and not judge(s(.099, .01, .19)) and not judge(s(.20, 0., .40)) and not judge(s(.20, -.01, .40))
    # 240 个回合里多 24 个成功正好是 10 个百分点；按会话平均后有浮点误差，仍应算达到
    assert judge(s(float(np.mean([.1] * 60)), .01, .19))


def test_not_worse_allows_a_drop_of_five_points_from_the_first_snapshot(results):
    judge = lambda change: results.verdict2('mrsz', 'stairs_up_07', **good(late_minus_early=change))['not_worse']
    assert judge(s(-.05, -.13, .03)) and judge(s(float(np.mean([-.05] * 60)), -.13, .03)) and judge(s(.04, -.04, .12))
    assert not judge(s(-.054, -.13, .03)) and not judge(s(-.133, -.22, -.04))


def test_stairs_without_stall_correction_leaves_the_comparison_with_it_descriptive(results):
    v = results.verdict2('mrsz', 'stairs_up_07', **good(minus_r2=s(-.10, -.18, -.02)))
    assert v == dict(effective=True, not_worse=True, retention=True, passed=True)


def test_the_wide_control_must_beat_stall_correction_and_keep_improving(results):
    v = results.verdict2('mrsw', 'stairs_up_07', **good(late_minus_early=s(.08, .01, .15)))
    assert v == dict(effective=True, beats_stall_correction=True, retention=True, still_improving=True, passed=True)
    flat = results.verdict2('mrsw', 'stairs_up_07', **good(late_minus_early=s(.02, -.06, .10)))
    assert flat['still_improving'] is False and flat['passed'] is False and 'not_worse' not in flat


@pytest.mark.parametrize('key', ['posture_increase_at_most_2pp', 'every_family_loss_at_most_5pp', 'flat_velocity_ratio_at_most_1p10'])
def test_retention_needs_each_of_the_three_conditions(results, key):
    v = results.verdict2('mrs', 'stairs_down_085', **good(five=dict(OK, **{key: False})))
    assert v['retention'] is False and v['passed'] is False
    # 目标增益的两项属于第 1 条，不重复计入保持
    assert results.verdict2('mrs', 'stairs_down_085', **good(five=dict(OK, target_gain_at_least_10pp=False, ci_lower_above_zero=False)))['retention'] is True


@pytest.mark.parametrize('variant,target', [('mrbz', 'stairs_up_07'), ('r2', 'slope_rough_down_x10'), ('r2', 'stairs_down_085')])
def test_descriptive_runs_have_no_registered_verdict(results, variant, target):
    assert results.verdict2(variant, target, **good(minus_r2=None, five=None)) is None


def test_the_run_table_lists_the_nine_registered_chains(results):
    assert results.STAGE2_RUNS == (('mrsz', 'slope_rough_down_x10'), ('mrsz', 'stairs_down_085'), ('mrsz', 'stairs_up_07'), ('mrsw', 'stairs_up_07'),
                                   ('r2', 'slope_rough_down_x10'), ('r2', 'stairs_down_085'), ('mrs', 'slope_rough_down_x10'), ('mrs', 'stairs_down_085'),
                                   ('mrbz', 'stairs_up_07'))
    assert results.STAGE2_DOCS.name == 'lite3-online-adapter2-20261126'
    with pytest.raises(ValueError): results.verdict2('mrs', 'boxes_07', **good())


def test_a_child_process_summarises_one_run_and_never_spawns_again(results, monkeypatch):
    # 父进程把 --stage2 原样传给子进程；子进程必须直接汇总，不能再派发一轮。
    calls = []
    monkeypatch.setattr(results, 'one', lambda partial: calls.append(partial))

    def refuse(*args, **kwargs): raise AssertionError('a child process started another process')
    monkeypatch.setattr(results.subprocess, 'run', refuse)
    monkeypatch.setenv('ADAPTER_CHILD', '1'); monkeypatch.setenv('ADAPTER_STAGE2', '1')
    monkeypatch.setattr(sys, 'argv', ['x', '--stage2', '--partial'])
    results.main()
    assert calls == [True]


def test_the_stage_two_parent_starts_one_child_per_registered_chain(results, monkeypatch):
    started = []
    monkeypatch.delenv('ADAPTER_CHILD', raising=False)
    monkeypatch.setattr(results.subprocess, 'run', lambda command, env, check: started.append(
        (env['C1A2_VARIANT'], env['C1A2_TARGET'], env['ADAPTER_CHILD'], env['ADAPTER_STAGE2'], check, '--stage2' in command)))
    monkeypatch.setattr(sys, 'argv', ['x', '--stage2'])
    results.main()
    assert [s[:2] for s in started] == list(results.STAGE2_RUNS) and all(s[2:] == ('1', '1', True, True) for s in started)


def test_the_second_stage_always_judges_at_ten_thousand_steps(results):
    # r2 登记的主快照是 2500 步；S2′ 里它作参照时也按 10000 步比较
    assert results.judged_mark(stage2=True, primary=2500) == 10000 and results.judged_mark(stage2=True, primary=10000) == 10000
    assert results.judged_mark(stage2=False, primary=2500) == 2500


def test_training_counts_of_the_frozen_arm_are_labelled_by_what_the_arm_really_ran(results):
    # 审查发现：r2 的训练任务里 Frozen 组每个窗口加参数扰动、只是不更新，它的训练成功数不是冻结策略的数字；
    # 小模型的变体两组都不加参数扰动，Frozen 组就是冻结策略。
    assert results.frozen_training_arm(dict(kind='residual', update='s', level0=False, limit_scale=1.)) == 'base policy'
    assert results.frozen_training_arm(dict(kind='film', update='a', level0=True, limit_scale=1.)) == 'base policy'
    assert results.frozen_training_arm(None) == 'perturbed parameters, not updated'
