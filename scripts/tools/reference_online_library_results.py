"""汇总一个新地形上加快学习 C 的结果：C0（库里的参数直接用）、C1（c3：选择加修正）、C2（w2：从 0 学）。
判定条件见 docs/lite3-online-library-20261122/PLAN.md。

用法：C1A2_VARIANT=c3 C1A2_TARGET=<目标> python reference_online_library_results.py [--partial]
--partial：只汇总已经跑完的部分，不做保持评估和判定，也不写结果文件。
"""
from __future__ import annotations

import json
import sys

import numpy as np

from reference_online_c1a2 import LIBRARY_SOURCES, MARKS, OUTPUT, PRIMARY, RUNS, TARGET, TERRAINS, VARIANT, jobs, protocol
from reference_online_c1a_results import FAMILY, paired_summary
from reference_online_c1aea_results import against_fixed, frozen_reproduced, read
from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-library-20261122'
LOGS = ROOT / 'logs/reference_six'
CONTROL = LOGS / RUNS['w2'] / TARGET  # C2
ENTRIES = [f'{v}:{t}:{m}' for v, t, m in LIBRARY_SOURCES]
TRANSFER = [LOGS / 'online_library_transfer_20261122' / TARGET / f'from_{v}_{t}_{m}_pooled' for v, t, m in LIBRARY_SOURCES]  # C0


def evaluation(base, mark):
    return [r for p in range(4) for r in read(base, f'target_{mark}_{p:02d}')['records']]


def done(base, mark):
    return all((base / f'target_{mark}_{p:02d}' / 'report.json').exists() for p in range(4))


def learning(base, marks):
    """各快照时每个会话已经完成的尝试数（4 个并行环境合计）、更新次数和参数。"""
    import torch
    train, result = read(base, 'train')['records'], {}
    for mark in marks:
        attempts = np.zeros(60)
        for r in train:
            if r['arm'] == 'Updated' and r['control_step'] <= mark: attempts[r['session']] += 1
        heads = torch.load(base / 'train' / f'at_{mark}.pt', map_location='cpu', weights_only=True)['heads']
        z = np.tanh(np.asarray([np.asarray(h['w']) for h in heads[1::2]]))
        result[str(mark)] = dict(mean_attempts_per_session=float(attempts.mean()), mean_updates_per_session=float(np.mean([h['updates'] for h in heads[1::2]])),
                                 mean_tanh=z.mean(0).tolist(), std_tanh=z.std(0).tolist(),
                                 sessions_selected=sum(h.get('selection') is not None for h in heads[1::2]))
    result['train_success'] = {arm: float(np.mean([r['outcome'] == 'success' for r in train if r['arm'] == arm])) for arm in ('Frozen', 'Updated')}
    result['train_posture'] = {arm: float(np.mean([r['outcome'] == 'posture_failure' for r in train if r['arm'] == arm])) for arm in ('Frozen', 'Updated')}
    return result


def selection(transfer):
    """每个会话选中了哪一组；按整回合回报来选会选出哪一组；选中的那一组不再修正时在评估起点上的成功数。"""
    chosen = json.loads((OUTPUT / 'train/selection.json').read_text())
    library, sessions = np.asarray(chosen['library']), chosen['sessions']
    picked = np.asarray([s['entry'] for s in sessions])
    returns, outcome = np.full((60, len(library), 2), np.nan), np.zeros((60, len(library), 2), bool)
    steps = np.zeros(60)
    for r in read(OUTPUT, 'train')['records']:
        if r['arm'] != 'Updated' or r['episode'] > 1: continue
        entry = (r['repeat'] + r['episode']) % len(library)
        assert np.allclose(r['code'], library[entry], atol=1e-9)
        returns[r['session'], entry, r['episode']] = r['episode_return']; outcome[r['session'], entry, r['episode']] = r['outcome'] == 'success'
        steps[r['session']] = max(steps[r['session']], r['control_step'])
    assert np.isfinite(returns).all()
    by_return = returns.mean(2)
    best = by_return.max(1, keepdims=True)
    result = dict(library_tanh=np.tanh(library).tolist(), entries=ENTRIES + ['zero'],
        chosen_counts=np.bincount(picked, minlength=len(library)).tolist(),
        mean_score_by_entry=np.mean([[x if x is not None else np.nan for x in s['scores']] for s in sessions], 0).tolist(),
        mean_windows_by_entry=np.mean([s['windows'] for s in sessions], 0).tolist(),
        selection_trial_success_by_entry=outcome.mean((0, 2)).tolist(),
        return_choice_counts=np.bincount(by_return.argmax(1), minlength=len(library)).tolist(),
        sessions_where_window_choice_has_the_best_return=int((np.take_along_axis(by_return, picked[:, None], 1) >= best - 1e-12).sum()),
        sessions_with_tied_best_return=int(((by_return >= best - 1e-12).sum(1) > 1).sum()),
        mean_control_steps_until_selected=float(steps.mean()), max_control_steps_until_selected=float(steps.max()))
    if transfer:  # 选中后不修正：每个会话取它选中的那一组在 C0 里的结局（全 0 取 Frozen 组）
        success = np.zeros((60, len(library)))
        for k, rows in enumerate(transfer):
            for r in rows:
                if r['arm'] == 'Updated': success[r['session'], k] += r['outcome'] == 'success'
        for r in transfer[0]:
            if r['arm'] == 'Frozen': success[r['session'], len(library) - 1] += r['outcome'] == 'success'
        frozen = success[:, -1]
        result.update(success_by_entry=success.sum(0).tolist(), chosen_without_refinement=float(np.take_along_axis(success, picked[:, None], 1).sum()),
                      best_entry_per_session=float(success.max(1).sum()), frozen=float(frozen.sum()))
    return result


def main():
    assert VARIANT == 'c3'
    partial = '--partial' in sys.argv
    p = protocol()
    marks = [m for m in MARKS[1:] if done(OUTPUT, m)] if partial else list(MARKS[1:])
    control_marks = [m for m in marks if done(CONTROL, m)]
    rows = {m: evaluation(OUTPUT, m) for m in marks}
    control = {m: evaluation(CONTROL, m) for m in control_marks}
    transfer = [evaluation(base, mark) for base, (_, _, mark) in zip(TRANSFER, LIBRARY_SOURCES) if done(base, mark)]
    assert partial or (len(transfer) == len(TRANSFER) and control_marks == marks)
    result = dict(target=TARGET, primary_mark=PRIMARY, marks=marks,
        C1_versus_frozen={str(m): paired_summary(rows[m]) for m in marks}, C2_versus_frozen={str(m): paired_summary(control[m]) for m in control_marks},
        C1_versus_C2={str(m): against_fixed(rows[m], control[m]) for m in control_marks},
        C0_versus_frozen={name: paired_summary(t) for name, t in zip(ENTRIES, transfer)},
        C1_versus_C0={name: {str(m): against_fixed(rows[m], t) for m in marks} for name, t in zip(ENTRIES, transfer)},
        frozen_reproduced=dict(C2={str(m): frozen_reproduced(rows[m], control[m]) for m in control_marks},
                               C0={name: frozen_reproduced(rows[marks[0]], t) for name, t in zip(ENTRIES, transfer)} if marks else {}),
        C1_learning=learning(OUTPUT, marks), C2_learning=learning(CONTROL, control_marks) if control_marks else {},
        selection=selection(transfer if len(transfer) == len(TRANSFER) else None))
    if not partial:
        schedule = [j for j in jobs()[1:] if j['kind'] in ('train', 'target', 'retention')]
        reports = {j['name']: read(OUTPUT, j['name'], 'report.json') for j in schedule}
        assert all(r['status'] == 'passed' for r in reports.values())
        retained = [r for j in schedule if j['kind'] == 'retention' for r in read(OUTPUT, j['name'])['records']]
        families = {f: paired_summary([r for r in retained if FAMILY[r['condition']] == f],
                    repeats=2 * sum(v == f for v in FAMILY.values())) for f in dict.fromkeys(FAMILY.values())}
        target, flat = result['C1_versus_frozen'][str(PRIMARY)], families['flat']
        ratio = flat['Updated']['xy_velocity_rmse'] / flat['Frozen']['xy_velocity_rmse']
        five = dict(target_gain_at_least_10pp=target['mean_session_delta'] >= p['success_gain_min'],
            ci_lower_above_zero=target['ci95'][0] > p['ci_lower_min'],
            posture_increase_at_most_2pp=target['posture_delta'] <= p['safety_increase_max'],
            every_family_loss_at_most_5pp=all(v['mean_session_delta'] >= -p['family_loss_max'] for v in families.values()),
            flat_velocity_ratio_at_most_1p10=ratio <= 1.10)
        result.update(five_criteria=five, all_five_met=all(five.values()),
            registered_test_passed=bool(five['target_gain_at_least_10pp'] and five['ci_lower_above_zero']),
            retention_families=families, retention_subtypes={c: paired_summary([r for r in retained if r['condition'] == c], repeats=2) for c in TERRAINS},
            flat_velocity_rmse_ratio=ratio, interactions=sum(r['interactions'] for r in reports.values()))
        save(DOCS / f'RESULTS_{TARGET}.json', result)
    count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
    posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
    pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())
    print(TARGET, 'partial' if partial else 'final')
    for name, v in result['C0_versus_frozen'].items():
        print(f" C0 {name:26s} Frozen {count(v, 'Frozen')} -> {count(v, 'Updated')} gain {pp(v)} posture {posture(v, 'Frozen')} -> {posture(v, 'Updated')}")
    for m in marks:
        v, l = result['C1_versus_frozen'][str(m)], result['C1_learning'][str(m)]
        line = f" {m:>5}: C1 attempts {l['mean_attempts_per_session']:.1f} updates {l['mean_updates_per_session']:.1f} selected {l['sessions_selected']} tanh {np.round(l['mean_tanh'], 2).tolist()} | Frozen {count(v, 'Frozen')} -> C1 {count(v, 'Updated')} gain {pp(v)} posture {posture(v, 'Frozen')} -> {posture(v, 'Updated')}"
        if str(m) in result['C2_versus_frozen']:
            c, x, l2 = result['C2_versus_frozen'][str(m)], result['C1_versus_C2'][str(m)], result['C2_learning'][str(m)]
            line += f" | C2 {count(c, 'Updated')} gain {pp(c)} attempts {l2['mean_attempts_per_session']:.1f} tanh {np.round(l2['mean_tanh'][2:], 2).tolist()} | C1-C2 {pp(x)}"
        print(line)
        for name, by_mark in result['C1_versus_C0'].items():
            x = by_mark[str(m)]; print(f"        C1 vs C0 {name:26s} {x['D_success']} vs {x['B_success']} {pp(x)}")
    print(' frozen identical:', {k: ({m: (x['identical'], x['n']) for m, x in v.items()}) for k, v in result['frozen_reproduced'].items()})
    print(' selection:', {k: (np.round(v, 3).tolist() if isinstance(v, list) and not isinstance(v[0], str) else v) for k, v in result['selection'].items()})
    print(' C1 train', result['C1_learning']['train_success'], result['C1_learning']['train_posture'])
    if not partial: print(' registered test passed:', result['registered_test_passed'], 'five:', result['five_criteria'], 'flat ratio', round(result['flat_velocity_rmse_ratio'], 3),
                          'families', {f: round(100 * v['mean_session_delta'], 1) for f, v in result['retention_families'].items()})


if __name__ == '__main__': main()
