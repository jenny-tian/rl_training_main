"""汇总一个目标的加快学习 B（C1-b-W）结果。判定条件见 docs/lite3-online-window-20261121/PLAN.md。

用法：C1A2_VARIANT=w2 C1A2_TARGET=<目标> python reference_online_window_results.py [--partial]
--partial：只汇总已经跑完的快照，不做保持评估和判定，用于训练后先看学习速度。
"""
from __future__ import annotations

import sys

import numpy as np

from reference_online_c1a2 import OUTPUT, TARGET, MARKS, PRIMARY, TERRAINS, VARIANT, RUNS, jobs, protocol
from reference_online_c1a_results import FAMILY, paired_summary
from reference_online_c1aea_results import against_fixed, frozen_reproduced, read
from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-window-20261121'
LOGS = ROOT / 'logs/reference_six'
BOXES = TARGET == 'boxes_06'
# 对照：上阶是做法二在相同快照上的评估；箱体是不学习的迁移结果（T3，平均参数）和本地学习（bs，40000 步）。
S2N = LOGS / RUNS['s2n'] / TARGET
T3 = LOGS / 'online_transfer_20261120/boxes_06/from_s2n_stairs_up_06_60000_pooled'
NATIVE = LOGS / RUNS['bs'] / 'boxes_06'


def evaluation(base, mark):
    return [r for p in range(4) for r in read(base, f'target_{mark}_{p:02d}')['records']]


def done(mark):
    return all((OUTPUT / f'target_{mark}_{p:02d}' / 'report.json').exists() for p in range(4))


def learning(train, windows):
    """各快照时每个会话已经完成的尝试数、更新次数和参数。"""
    import torch
    result = {}
    for mark in MARKS[1:]:
        attempts = np.zeros(60)
        for r in train:
            if r['arm'] == 'Updated' and r['control_step'] <= mark: attempts[r['session']] += 1
        heads = torch.load(OUTPUT / 'train' / f'at_{mark}.pt', map_location='cpu', weights_only=True)['heads']
        w = np.asarray([np.asarray(h['w']) for h in heads[1::2]]); z = np.tanh(w)
        result[str(mark)] = dict(mean_attempts_per_session=float(attempts.mean()), mean_updates_per_session=float(np.mean([h['updates'] for h in heads[1::2]])),
            mean_tanh=z.mean(0).tolist(), std_tanh=z.std(0).tolist(), swing_both_above_0p5=float((z[:, 2:] > .5).all(1).mean()))
    logs = [e for key, log in windows.items() if key.endswith('Updated') for e in log]
    steps = np.asarray([e['step'] for e in logs if 'step' in e])
    result['windows'] = dict(total=len(logs), updated=sum(e['updated'] for e in logs), mean_used=float(np.mean([len(e['used']) for e in logs])),
        step_mean=steps.mean(0).tolist(), step_std=steps.std(0).tolist(), share_positive=(steps > 0).mean(0).tolist())
    return result


def main():
    assert VARIANT == 'w2'
    partial = '--partial' in sys.argv
    p = protocol()
    marks = [m for m in MARKS[1:] if done(m)] if partial else list(MARKS[1:])
    train = read(OUTPUT, 'train')['records']
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    curves = {str(m): paired_summary(rows[m]) for m in marks}
    result = dict(target=TARGET, primary_mark=PRIMARY, marks=marks, versus_frozen=curves,
                  learning=learning(train, read(OUTPUT, 'train', 'windows.json')), train_episodes=len(train),
                  train_success={arm: float(np.mean([r['outcome'] == 'success' for r in train if r['arm'] == arm])) for arm in ('Frozen', 'Updated')},
                  train_posture={arm: float(np.mean([r['outcome'] == 'posture_failure' for r in train if r['arm'] == arm])) for arm in ('Frozen', 'Updated')})
    if BOXES:
        transfer, native = evaluation(T3, 60000), evaluation(NATIVE, 40000)
        result.update(versus_transfer={str(m): against_fixed(rows[m], transfer) for m in marks}, transfer_versus_frozen=paired_summary(transfer),
                      versus_native={str(m): against_fixed(rows[m], native) for m in marks}, native_versus_frozen=paired_summary(native),
                      frozen_reproduced={str(m): frozen_reproduced(rows[m], transfer) for m in marks})
    else:
        shared = [m for m in marks if (S2N / f'target_{m}_00' / 'episodes.json').exists()]
        fixed = {m: evaluation(S2N, m) for m in shared}
        result.update(versus_s2n={str(m): against_fixed(rows[m], fixed[m]) for m in shared}, s2n_versus_frozen={str(m): paired_summary(fixed[m]) for m in shared},
                      frozen_reproduced={str(m): frozen_reproduced(rows[m], fixed[m]) for m in shared})
    if not partial:
        schedule = [j for j in jobs()[1:] if j['kind'] in ('train', 'target', 'retention')]
        reports = {j['name']: read(OUTPUT, j['name'], 'report.json') for j in schedule}
        assert all(r['status'] == 'passed' for r in reports.values())
        retained = [r for j in schedule if j['kind'] == 'retention' for r in read(OUTPUT, j['name'])['records']]
        families = {f: paired_summary([r for r in retained if FAMILY[r['condition']] == f],
                    repeats=2 * sum(v == f for v in FAMILY.values())) for f in dict.fromkeys(FAMILY.values())}
        target, flat = curves[str(PRIMARY)], families['flat']
        ratio = flat['Updated']['xy_velocity_rmse'] / flat['Frozen']['xy_velocity_rmse']
        five = dict(target_gain_at_least_10pp=target['mean_session_delta'] >= p['success_gain_min'],
            ci_lower_above_zero=target['ci95'][0] > p['ci_lower_min'],
            posture_increase_at_most_2pp=target['posture_delta'] <= p['safety_increase_max'],
            every_family_loss_at_most_5pp=all(v['mean_session_delta'] >= -p['family_loss_max'] for v in families.values()),
            flat_velocity_ratio_at_most_1p10=ratio <= 1.10)
        registered = result['versus_transfer'][str(PRIMARY)]['ci95'][0] > 0 if BOXES else five['target_gain_at_least_10pp'] and five['ci_lower_above_zero']
        result.update(five_criteria=five, all_five_met=all(five.values()), registered_test_passed=bool(registered),
            retention_families=families, retention_subtypes={c: paired_summary([r for r in retained if r['condition'] == c], repeats=2) for c in TERRAINS},
            flat_velocity_rmse_ratio=ratio, interactions=sum(r['interactions'] for r in reports.values()))
        save(DOCS / f'RESULTS_{TARGET}.json', result)
    count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
    pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())
    print(TARGET, 'partial' if partial else 'final')
    for m in marks:
        v, l = curves[str(m)], result['learning'][str(m)]
        line = f" {m:>5}: attempts {l['mean_attempts_per_session']:.1f} updates {l['mean_updates_per_session']:.1f} tanh {np.round(l['mean_tanh'], 2).tolist()} | Frozen {count(v, 'Frozen')} -> W {count(v, 'Updated')} gain {pp(v)} posture {v['Frozen']['outcomes'].get('posture_failure', 0)} -> {v['Updated']['outcomes'].get('posture_failure', 0)}"
        for name in ('versus_s2n', 'versus_transfer', 'versus_native'):
            if str(m) in result.get(name, {}):
                x = result[name][str(m)]; line += f" | {name} {x['D_success']} vs {x['B_success']} {pp(x)}"
        print(line)
    print(' frozen identical:', {m: (v['identical'], v['n']) for m, v in result['frozen_reproduced'].items()})
    print(' windows:', {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 2)) for k, v in result['learning']['windows'].items()})
    print(' train success', result['train_success'], 'posture', result['train_posture'])
    if not partial: print(' registered test passed:', result['registered_test_passed'], 'five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3))


if __name__ == '__main__': main()
