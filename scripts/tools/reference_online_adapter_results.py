"""汇总小模型的结果。S1 的判定条件见 docs/lite3-online-adapter-20261126/PLAN.md 第 7 节，S2′ 的见 docs/lite3-online-adapter2-20261126/PLAN.md 第 6 节。

用法：python reference_online_adapter_results.py [--partial] [变体 ...]              S1
      python reference_online_adapter_results.py --stage2 [--partial]               S2′ 的九条运行链
      python reference_online_adapter_results.py --rule                             S2′ 不学习的规则对照（方案第 12 节）
各目标、各变体用子进程分别汇总；参照是同一目标上只有受阻修正的 r2（评估起点相同）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-adapter-20261126'
LOGS = ROOT / 'logs/reference_six'
TARGETS = ('stairs_up_07', 'slope_rough_up_x11')
VARIANTS = ('mfa', 'mfb', 'mfc', 'mra', 'mrb', 'mrc')
count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())

# S2′：九条运行链，按运行顺序。
STAGE2_DOCS = ROOT / 'docs/lite3-online-adapter2-20261126'
STAGE2_RUNS = (('mrsz', 'slope_rough_down_x10'), ('mrsz', 'stairs_down_085'), ('mrsz', 'stairs_up_07'), ('mrsw', 'stairs_up_07'),
               ('r2', 'slope_rough_down_x10'), ('r2', 'stairs_down_085'), ('mrs', 'slope_rough_down_x10'), ('mrs', 'stairs_down_085'),
               ('mrbz', 'stairs_up_07'))
EPSILON = 1e-9  # 门槛正好落在整数个回合上时，按会话平均的浮点误差不应改变判定
RULE_TARGETS = ('stairs_up_07', 'slope_rough_down_x10', 'stairs_down_085')


def frozen_training_arm(adapter):
    """训练任务里 Frozen 组实际跑的是什么。小模型的变体两组都不加参数扰动，Frozen 组就是冻结策略；
    r2 等窗口学习器的 Frozen 组每个窗口加参数扰动、只是不更新，它在训练里的成功数不能当作冻结策略的数字。评估任务不受影响。"""
    return 'base policy' if adapter else 'perturbed parameters, not updated'


def judged_mark(*, stage2, primary):
    """判定所在的快照。S2′ 一律是 10000 步（r2 登记的主快照是 2500 步，作参照时也按 10000 步比较）。"""
    return 10000 if stage2 else primary


def verdict2(variant, target, *, gain, late_minus_early, minus_r2, five):
    """S2′ 登记的判定。gain：10000 步对 Frozen；late_minus_early：10000 步减 1250 步；minus_r2：10000 步减 r2 的 10000 步；
    三个都是带 mean_session_delta 和 ci95 的配对汇总。five 是保持评估的五项。只作描述的运行返回 None。"""
    if (variant, target) not in STAGE2_RUNS:
        raise ValueError(f'{variant} on {target} is not a registered S2′ run')
    if variant in ('mrbz', 'r2'): return None
    result = dict(effective=bool(gain['mean_session_delta'] >= .10 - EPSILON and gain['ci95'][0] > 0))
    if variant != 'mrsw': result['not_worse'] = bool(late_minus_early['mean_session_delta'] >= -.05 - EPSILON)
    if variant == 'mrsw' or target != 'stairs_up_07': result['beats_stall_correction'] = bool(minus_r2['ci95'][0] > 0)
    result['retention'] = bool(five['posture_increase_at_most_2pp'] and five['every_family_loss_at_most_5pp'] and five['flat_velocity_ratio_at_most_1p10'])
    if variant == 'mrsw': result['still_improving'] = bool(late_minus_early['ci95'][0] > 0)
    result['passed'] = all(result.values())
    return result


def rule_verdict(delta):
    """小模型（mrsz 的 10000 步）减规则对照，按会话配对的登记判定：区间下限大于 0 是小模型好于规则，区间上限小于 0 是规则好于小模型。"""
    low, high = delta['ci95']
    return 'model_better' if low > 0 else 'rule_better' if high < 0 else 'no_confirmed_difference'


def rule_one():
    from reference_online_c1a2 import OUTPUT, RULE, RUNS, TARGET, protocol
    from reference_online_c1a_results import paired_summary
    from reference_online_c1aea_results import against_fixed, frozen_reproduced
    from reference_online_select_results import done, evaluation, read, retention
    assert RULE == 'directed'
    if not done(OUTPUT, 0): return print(TARGET, 'rule not finished')
    rows = evaluation(OUTPUT, 0)
    model, stall = evaluation(LOGS / RUNS['mrsz'] / TARGET, 10000), evaluation(LOGS / RUNS['r2'] / TARGET, 10000)
    result = dict(target=TARGET, method='directed rule, no learning', versus_frozen=paired_summary(rows),
                  model_minus_rule=against_fixed(model, rows), rule_minus_r2=against_fixed(rows, stall),
                  frozen_reproduced=frozen_reproduced(rows, model),
                  policy_steps_with_rule=[read(OUTPUT, f'target_0_{p:02d}', 'rule.json')['policy_steps_with_rule'] for p in range(4)])
    result['registered'] = rule_verdict(result['model_minus_rule'])
    if (OUTPUT / 'retention_0_19' / 'report.json').exists():
        result.update(retention(OUTPUT, protocol(), result['versus_frozen']))
    save(STAGE2_DOCS / f'RESULTS_rule_{TARGET}.json', result)
    x, d, e = result['versus_frozen'], result['model_minus_rule'], result['rule_minus_r2']
    print(f"{TARGET} rule: Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)} posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')}"
          f" | mrsz@10000 ({d['D_success']}) minus rule {pp(d)} -> {result['registered']} | rule minus r2@10000 ({e['B_success']}) {pp(e)}"
          f" | frozen identical {result['frozen_reproduced']['identical']}/{result['frozen_reproduced']['n']}")
    if 'five_criteria' in result: print('   five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3),
                                        'families', {f: round(100 * v['mean_session_delta'], 1) for f, v in result['retention_families'].items()})


def one(partial):
    from reference_online_c1a2 import ADAPTER, MARKS, OUTPUT, PRIMARY, RUNS, TARGET, VARIANT, protocol
    from reference_online_c1a_results import paired_summary
    from reference_online_c1aea_results import against_fixed, frozen_reproduced
    from reference_online_select_results import done, evaluation, read, retention
    r2 = LOGS / RUNS['r2'] / TARGET
    marks = [m for m in MARKS[1:] if done(OUTPUT, m)]
    if not marks or (not partial and len(marks) != len(MARKS) - 1): return print(TARGET, VARIANT, 'not finished')
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    result = dict(target=TARGET, method=VARIANT, adapter=ADAPTER, marks=marks, versus_frozen={str(m): paired_summary(rows[m]) for m in marks})
    train = read(OUTPUT, 'train')['records']
    result['training_frozen_arm'] = frozen_training_arm(ADAPTER)
    frozen_label = 'Frozen' if ADAPTER else 'perturbed'
    result['training'] = {str(m): {arm: dict(attempts=sum(r['arm'] == arm and r['control_step'] <= m for r in train) / 60,
        success=sum(r['arm'] == arm and r['control_step'] <= m and r['outcome'] == 'success' for r in train)) for arm in ('Frozen', 'Updated')} for m in marks}
    stage2 = bool(os.environ.get('ADAPTER_STAGE2'))
    judged = judged_mark(stage2=stage2, primary=PRIMARY)
    result['judged_mark'] = judged
    if judged in marks and 1250 in marks:
        result['primary_minus_first'] = against_fixed(rows[judged], rows[1250])
    if VARIANT != 'r2':
        shared = [m for m in marks if done(r2, m)]
        result['minus_r2'] = {str(m): against_fixed(rows[m], evaluation(r2, m)) for m in shared}
        result['frozen_reproduced'] = {str(m): frozen_reproduced(rows[m], evaluation(r2, m)) for m in shared}
    retention_mark = protocol()['primary_mark']
    has_retention = (OUTPUT / f'retention_{retention_mark}_19' / 'report.json').exists()
    if has_retention and not partial:
        result.update(retention(OUTPUT, protocol(), result['versus_frozen'][str(retention_mark)]))
    if stage2 and not partial:
        registered = verdict2(VARIANT, TARGET, gain=result['versus_frozen'][str(judged)], late_minus_early=result['primary_minus_first'],
                              minus_r2=result.get('minus_r2', {}).get(str(judged)), five=result.get('five_criteria'))
        if registered: result['registered'] = registered
    elif not partial and ADAPTER:
        v = result['versus_frozen'][str(PRIMARY)]
        five = result['five_criteria']
        result['registered'] = dict(
            effective=bool(v['mean_session_delta'] >= .10 and v['ci95'][0] > 0),
            still_improving=bool(result['primary_minus_first']['ci95'][0] > 0),
            adds_to_stall_correction=bool(str(PRIMARY) in result.get('minus_r2', {}) and result['minus_r2'][str(PRIMARY)]['ci95'][0] > 0),
            retention=bool(five['posture_increase_at_most_2pp'] and five['every_family_loss_at_most_5pp'] and five['flat_velocity_ratio_at_most_1p10']))
    if not partial: save((STAGE2_DOCS if stage2 else DOCS) / f'RESULTS_{VARIANT}_{TARGET}.json', result)
    print(TARGET, VARIANT)
    for m in marks:
        x, t = result['versus_frozen'][str(m)], result['training'][str(m)]
        line = f" {m:>5}: Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)} posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')} | train success {frozen_label}/Updated {t['Frozen']['success']}/{t['Updated']['success']} attempts {t['Updated']['attempts']:.1f}"
        if str(m) in result.get('minus_r2', {}):
            d = result['minus_r2'][str(m)]
            line += f" | minus r2 ({d['B_success']}) {pp(d)} frozen identical {result['frozen_reproduced'][str(m)]['identical']}/{result['frozen_reproduced'][str(m)]['n']}"
        print(line)
    if 'primary_minus_first' in result: print(f'   {judged} minus 1250:', pp(result['primary_minus_first']))
    if 'five_criteria' in result: print('   five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3),
                                        'families', {f: round(100 * x['mean_session_delta'], 1) for f, x in result['retention_families'].items()})
    if 'registered' in result: print('   registered', result['registered'])


def compare():
    """只作描述的配对比较（10000 步，同一目标、相同评估起点）：两种结构之间、三种更新方式之间，以及各快照的成功数。"""
    from reference_online_c1a2 import RUNS
    from reference_online_c1aea_results import against_fixed
    from reference_online_select_results import done, evaluation
    result = {}
    for target in TARGETS:
        base = lambda v: LOGS / RUNS[v] / target
        rows = {v: evaluation(base(v), 10000) for v in VARIANTS}
        pairs = [(f'mf{u}', f'mr{u}') for u in 'abc'] + [(f'm{k}{x}', f'm{k}{y}') for k in 'fr' for x, y in (('b', 'a'), ('c', 'a'), ('c', 'b'))]
        result[target] = dict(pairs={f'{a}-{b}': against_fixed(rows[a], rows[b]) for a, b in pairs}, success={})
        for v in VARIANTS + ('r2',) + tuple(x for x in ('mrcz',) if done(base(x), 10000)):
            marks = [m for m in (1250, 2500, 5000, 10000, 20000) if done(base(v), m)]
            counts = {m: [sum(r['arm'] == arm and r['outcome'] == 'success' for r in evaluation(base(v), m)) for arm in ('Frozen', 'Updated')] for m in marks}
            result[target]['success'][v] = {str(m): c[1] for m, c in counts.items()}
            result[target]['frozen'] = counts[marks[0]][0]
        print(target, 'Frozen', result[target]['frozen'])
        for v, row in result[target]['success'].items(): print(f'  {v:5s}', ' '.join(f'{m}:{c}' for m, c in row.items()))
        for name, v in result[target]['pairs'].items(): print(f"  {name}: {v['D_success']} - {v['B_success']} = {pp(v)}")
    save(DOCS / 'COMPARISONS.json', result)


def main():
    if os.environ.get('ADAPTER_CHILD'):  # 子进程只汇总自己这一条，不再派发
        return rule_one() if os.environ.get('ADAPTER_RULE') else one('--partial' in sys.argv)
    if '--rule' in sys.argv:
        for target in RULE_TARGETS:
            subprocess.run([sys.executable, __file__, '--rule'], check=True,
                           env=dict(os.environ, ADAPTER_CHILD='1', ADAPTER_RULE='1', C1A2_TARGET=target, C1A2_VARIANT='r2', C1A2_RULE='directed'))
        return
    if '--compare' in sys.argv: return compare()
    if '--stage2' in sys.argv:
        for variant, target in STAGE2_RUNS:
            subprocess.run([sys.executable, __file__, *[a for a in sys.argv[1:] if a.startswith('--')]],
                           env=dict(os.environ, ADAPTER_CHILD='1', ADAPTER_STAGE2='1', C1A2_TARGET=target, C1A2_VARIANT=variant), check='--partial' not in sys.argv)
        return
    arguments = [a for a in sys.argv[1:] if not a.startswith('--')]
    for target in TARGETS:
        for variant in arguments or VARIANTS + ('r2',):
            if variant == 'r2' and target != 'slope_rough_up_x11': continue
            subprocess.run([sys.executable, __file__, *[a for a in sys.argv[1:] if a.startswith('--')]],
                           env=dict(os.environ, ADAPTER_CHILD='1', C1A2_TARGET=target, C1A2_VARIANT=variant), check='--partial' not in sys.argv)


if __name__ == '__main__': main()
