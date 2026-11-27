"""S2 里不依赖记录文件的判定：按规则对照探路的结果选目标、脚本化检查选组和确认、覆盖面。方案见 docs/lite3-online-s2-20261127/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import pytest

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')


@pytest.fixture
def s2(monkeypatch):
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_s2', None)
    return importlib.import_module('reference_online_s2')


def test_the_candidates_are_grouped_into_six_classes(s2):
    assert s2.CLASSES == dict(boxes=('boxes_07', 'boxes_08'), rough=('rough_09', 'rough_10', 'rough_11'), slope=('slope_up_55', 'slope_up_57', 'slope_up_60'),
                              slope_rough_up=('slope_rough_up_x11', 'slope_rough_up_x12'), slope_rough_down=('slope_rough_down_x11', 'slope_rough_down_x12'),
                              stairs_down=('stairs_down_09', 'stairs_down_10'))
    assert s2.FALLBACK == dict(slope_rough_down='slope_rough_down_x10', stairs_down='stairs_down_085')


@pytest.mark.parametrize('frozen,rule,expect', [
    (75, 150, True), (168, 204, True), (24, 0, True), (0, 24, True),
    (75, 205, False),    # 规则对照到顶
    (169, 180, False),   # 原步态太容易
    (23, 23, False),     # 两者都接近 0，太难
    (0, 0, False)])
def test_eligibility_follows_the_three_registered_conditions(s2, frozen, rule, expect):
    assert s2.eligible(frozen, rule) is expect


def test_each_class_takes_the_eligible_candidate_closest_to_75_frozen_successes(s2):
    counts = dict(boxes_07=(24, 100), boxes_08=(60, 90),                 # 都符合：取 60（离 75 更近）
                  rough_09=(150, 230), rough_10=(100, 160), rough_11=(10, 5),   # 9 cm 规则到顶，11 cm 太难
                  slope_up_55=(200, 200), slope_up_57=(50, 20), slope_up_60=(5, 3),
                  slope_rough_up_x11=(51, 70), slope_rough_up_x12=(30, 40),
                  slope_rough_down_x11=(4, 230), slope_rough_down_x12=(0, 0),  # 没有符合的：沿用 S2′ 的目标，不再训练
                  stairs_down_09=(20, 240), stairs_down_10=(2, 180))
    chosen = s2.select_targets(counts)
    assert chosen == dict(boxes=dict(target='boxes_08', train=True), rough=dict(target='rough_10', train=True), slope=dict(target='slope_up_57', train=True),
                          slope_rough_up=dict(target='slope_rough_up_x11', train=True), slope_rough_down=dict(target='slope_rough_down_x10', train=False),
                          stairs_down=dict(target='stairs_down_10', train=True))


def test_a_class_without_an_eligible_candidate_and_without_a_fallback_has_no_target(s2):
    counts = {name: (0, 0) for names in s2.CLASSES.values() for name in names}
    chosen = s2.select_targets(counts)
    assert chosen['rough'] == dict(target=None, train=False) and chosen['boxes'] == dict(target=None, train=False)
    assert chosen['stairs_down'] == dict(target='stairs_down_085', train=False)


def test_ties_go_to_the_easier_candidate_listed_first(s2):
    counts = {name: (0, 0) for names in s2.CLASSES.values() for name in names}
    counts.update(boxes_07=(85, 100), boxes_08=(65, 100))  # 都离 75 差 10
    assert s2.select_targets(counts)['boxes']['target'] == 'boxes_07'


def test_missing_candidates_are_an_error(s2):
    with pytest.raises(KeyError): s2.select_targets(dict(boxes_07=(1, 2)))


# ---------------------------------------------------------------- 脚本化检查
def test_the_best_action_group_is_confirmed_and_an_unlowered_one_too_when_the_best_lowers_the_height_limit(s2):
    assert s2.groups_to_confirm(dict(none=101, hind_half=120, hind_full=110, all_half=90, h15=105, h15_hind_half=118)) == ['hind_half']
    assert s2.groups_to_confirm(dict(none=101, hind_half=120, hind_full=110, all_half=90, h15=130, h15_hind_half=125)) == ['h15', 'hind_half']
    assert s2.groups_to_confirm(dict(none=101, hind_half=90, hind_full=95, all_half=95, h15=99, h15_hind_half=80)) == ['h15', 'hind_full']  # 低于 none 也照样确认最高的
    with pytest.raises(ValueError): s2.groups_to_confirm(dict(none=101, hind_half=120))


def test_confirmation_needs_ten_points_over_no_action_with_an_interval_above_zero(s2):
    d = lambda delta, low: dict(mean_session_delta=delta, ci95=[low, delta + .1])
    assert s2.confirmed(d(.10, .01)) and s2.confirmed(d(.25, .15))
    assert not s2.confirmed(d(.099, .01)) and not s2.confirmed(d(.20, 0.)) and not s2.confirmed(d(.20, -.02))


# ---------------------------------------------------------------- 覆盖面
def test_coverage_needs_four_of_six_classes_with_both_rough_slope_directions(s2):
    base = dict(stairs_up=True, stairs_down=True, boxes=True, rough=False, slope=False, slope_rough_up=True, slope_rough_down=True)
    assert s2.coverage(base) == dict(classes=dict(stairs_up=True, stairs_down=True, boxes=True, rough=False, slope=False, slope_rough=True), effective=4, passed=True)
    assert s2.coverage(dict(base, slope_rough_down=False))['passed'] is False and s2.coverage(dict(base, slope_rough_down=False))['effective'] == 3
    assert s2.coverage(dict(base, rough=None))['classes']['rough'] is False  # 没有目标的类算不满足
    with pytest.raises(KeyError): s2.coverage(dict(stairs_up=True))


# ---------------------------------------------------------------- 读规则对照探路的记录
def write_part(directory, outcomes):
    """outcomes：[(Frozen 的结局, Updated 的结局), ...]，每对环境一条。"""
    import json
    directory.mkdir(parents=True)
    records = [dict(arm=arm, outcome=o, mean_progress=1. if o == 'success' else .5) for pair in outcomes for arm, o in zip(('Frozen', 'Updated'), pair)]
    (directory / 'episodes.json').write_text(json.dumps(dict(records=records)))
    (directory / 'report.json').write_text(json.dumps(dict(status='passed')))


def test_probe_counts_add_the_four_parts_of_each_candidate(s2, tmp_path):
    for part, outcomes in enumerate([[('success', 'success'), ('timeout', 'success')], [('timeout', 'timeout')], [('posture_failure', 'success')], [('success', 'stuck')]]):
        write_part(tmp_path / 'boxes_07' / 'directed' / f'target_0_{part:02d}', outcomes)
    row = s2.probe_counts(tmp_path, 'boxes_07')
    assert row == dict(n=5, frozen=2, rule=3, frozen_outcomes=dict(success=2, timeout=2, posture_failure=1), rule_outcomes=dict(success=3, timeout=1, stuck=1),
                       frozen_progress=pytest.approx((2 * 1. + 3 * .5) / 5), rule_progress=pytest.approx((3 * 1. + 2 * .5) / 5))


def test_probe_counts_refuse_an_unfinished_or_failed_candidate(s2, tmp_path):
    import json
    for part in range(3): write_part(tmp_path / 'rough_10' / 'directed' / f'target_0_{part:02d}', [('success', 'success')])
    with pytest.raises(FileNotFoundError): s2.probe_counts(tmp_path, 'rough_10')
    write_part(tmp_path / 'rough_10' / 'directed' / 'target_0_03', [('success', 'success')])
    (tmp_path / 'rough_10' / 'directed' / 'target_0_03' / 'report.json').write_text(json.dumps(dict(status='failed')))
    with pytest.raises(ValueError): s2.probe_counts(tmp_path, 'rough_10')
