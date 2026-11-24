"""汇总直接微调基础策略的对照（ft、lora，以及只作描述的 ftu、lorau）。判定条件见 docs/lite3-online-finetune-20261124/PLAN.md。

用法：python reference_online_finetune_results.py [--partial]
我们的方法用已有的评估：h1（受阻修正加持续选择）、r2（只有受阻修正）、d4（有参数库的持续选择），都在 10000 步、上限 0.04 下，评估起点相同。
各变体和目标用子进程分别汇总，因为入口按环境变量决定变体和目标。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-finetune-20261124'
LOGS = ROOT / 'logs/reference_six'
TARGETS = ('stairs_up_07', 'boxes_07', 'stairs_up_075', 'boxes_08')
RUNS_TO_DO = [(v, t) for t in TARGETS for v in ('ft', 'lora')] + [(v, t) for t in TARGETS[:2] for v in ('ftu', 'lorau')]
OURS, OURS_MARK = ('h1', 'r2', 'd4'), 10000
count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())


def retention_at(base, mark, p):
    from reference_online_c1a_results import FAMILY, paired_summary
    from reference_online_c1aea_results import read
    if not all((base / f'retention_{mark}_{q:02d}' / 'report.json').exists() for q in range(20)): return None
    retained = [r for q in range(20) for r in read(base, f'retention_{mark}_{q:02d}')['records']]
    families = {f: paired_summary([r for r in retained if FAMILY[r['condition']] == f], repeats=2 * sum(v == f for v in FAMILY.values())) for f in dict.fromkeys(FAMILY.values())}
    ratio = families['flat']['Updated']['xy_velocity_rmse'] / families['flat']['Frozen']['xy_velocity_rmse']
    row = dict(families=families, flat_velocity_rmse_ratio=ratio,
               every_family_loss_at_most_5pp=all(v['mean_session_delta'] >= -p['family_loss_max'] for v in families.values()),
               flat_velocity_ratio_at_most_1p10=ratio <= 1.10)
    row['passed'] = bool(row['every_family_loss_at_most_5pp'] and row['flat_velocity_ratio_at_most_1p10'])
    return row


def one_run(partial):
    from reference_online_c1a2 import MARKS, OUTPUT, PRIMARY, RETENTION_MARKS, RUNS, TARGET, UNFILTERED, VARIANT, protocol
    from reference_online_c1a_results import paired_summary
    from reference_online_c1aea_results import against_fixed, frozen_reproduced, read
    from reference_online_select_results import done, evaluation
    p = protocol()
    marks = [m for m in MARKS[1:] if done(OUTPUT, m)] if partial else list(MARKS[1:])
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    ours = {name: evaluation(LOGS / RUNS[name] / TARGET, OURS_MARK) for name in OURS}
    result = dict(variant=VARIANT, target=TARGET, primary_mark=PRIMARY, marks=marks, versus_frozen={str(m): paired_summary(rows[m]) for m in marks},
                  ours_versus_frozen={name: paired_summary(ours[name]) for name in OURS})
    for name in OURS:  # 我们的方法（10000 步）减去这一组在各快照的结果
        result[f'{name}_minus_this'] = {str(m): against_fixed(ours[name], rows[m]) for m in marks}
        result[f'frozen_reproduced_{name}'] = {str(m): frozen_reproduced(rows[m], ours[name]) for m in marks}
        reached = [m for m in marks if count(result['versus_frozen'][str(m)], 'Updated') >= count(result['ours_versus_frozen'][name], 'Updated')]
        result[f'first_mark_reaching_{name}'] = reached[0] if reached else None
    if (OUTPUT / 'train/episodes.json').exists():
        train = read(OUTPUT, 'train')['records']
        log = json.loads((OUTPUT / 'train/finetune.json').read_text())
        horizon, learning = log['settings']['horizon'], {}
        for m in marks:
            attempts = np.zeros(60)
            for r in train:
                if r['arm'] == 'Updated' and r['control_step'] <= m: attempts[r['session']] += 1
            entry = log['log'][m // horizon - 1]
            learning[str(m)] = dict(mean_attempts_per_session=float(attempts.mean()), updates=m // horizon,
                                    **{k: float(np.mean(entry[k])) for k in ('lr', 'kl', 'value_loss', 'std', 'mean_reward', 'valid')})
        result['learning'] = learning
        result['train_success'] = {arm: float(np.mean([r['outcome'] == 'success' for r in train if r['arm'] == arm])) for arm in ('Frozen', 'Updated')}
        result['train_posture'] = {arm: float(np.mean([r['outcome'] == 'posture_failure' for r in train if r['arm'] == arm])) for arm in ('Frozen', 'Updated')}
    result['retention'] = {str(m): retention_at(OUTPUT, m, p) for m in RETENTION_MARKS if not UNFILTERED}
    if not partial:
        assert UNFILTERED or all(result['retention'].values())
        primary = result['versus_frozen'][str(PRIMARY)]
        result['registered'] = dict(
            h1_better_at_primary=bool(result['h1_minus_this'][str(PRIMARY)]['ci95'][0] > 0),
            r2_better_at_primary=bool(result['r2_minus_this'][str(PRIMARY)]['ci95'][0] > 0),
            posture_increase_at_most_2pp={str(m): bool(result['versus_frozen'][str(m)]['posture_delta'] <= p['safety_increase_max']) for m in RETENTION_MARKS},
            retention_passed={m: bool(v['passed']) for m, v in result['retention'].items()},
            gain_at_primary_at_least_10pp=bool(primary['mean_session_delta'] >= p['success_gain_min'] and primary['ci95'][0] > 0))
        save(DOCS / f'RESULTS_{VARIANT}_{TARGET}.json', result)
    print(VARIANT, TARGET, 'partial' if partial else 'final', '| ours at 10000:', {name: count(result['ours_versus_frozen'][name], 'Updated') for name in OURS})
    for m in marks:
        x, l = result['versus_frozen'][str(m)], result.get('learning', {}).get(str(m), {})
        print(f" {m:>5}: attempts {l.get('mean_attempts_per_session', 0):.1f} lr {l.get('lr', 0):.1e} kl {l.get('kl', 0):.4f} std {l.get('std', 0):.3f}"
              f" | Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)} posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')}"
              f" | h1-this {pp(result['h1_minus_this'][str(m)])} r2-this {pp(result['r2_minus_this'][str(m)])} d4-this {pp(result['d4_minus_this'][str(m)])}"
              f" | frozen identical {result['frozen_reproduced_h1'][str(m)]['identical']}")
    for m, v in result['retention'].items():
        if v: print(f'   retention {m}:', {f: (count(x, 'Frozen'), count(x, 'Updated'), pp(x)[0]) for f, x in v['families'].items()}, 'flat ratio', round(v['flat_velocity_rmse_ratio'], 3), 'passed', v['passed'])
    if not partial: print('   registered', result['registered'], '| first mark reaching', {name: result[f'first_mark_reaching_{name}'] for name in OURS})


def main():
    partial = '--partial' in sys.argv
    if os.environ.get('FINETUNE_RUN'): return one_run(partial)
    for variant, target in RUNS_TO_DO:
        subprocess.run([sys.executable, __file__, *sys.argv[1:]], env=dict(os.environ, FINETUNE_RUN='1', C1A2_VARIANT=variant, C1A2_TARGET=target), check=not partial)


if __name__ == '__main__': main()
