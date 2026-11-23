"""汇总一个新地形上加快学习 D 的结果：D1（d1：持续选择）、D2（d2：持续选择加候选）、D0（候选参数直接用）。
对照用阶段 C 已有的评估：C1（c3）、C2（w2）、C0（库里的参数直接用）。判定条件见 docs/lite3-online-select-20261123/PLAN.md。

用法：C1A2_VARIANT=d1 C1A2_TARGET=<目标> python reference_online_select_results.py [--partial]
--partial：只汇总已经跑完的部分，不做保持评估和判定，也不写结果文件。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from reference_online_c1a2 import LIBRARY_SOURCES, MARKS, PRIMARY, RUNS, TARGET, TERRAINS, VARIANT, jobs, protocol
from reference_online_c1a_results import FAMILY, paired_summary
from reference_online_c1aea_results import against_fixed, frozen_reproduced, read
from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-select-20261123'
LOGS = ROOT / 'logs/reference_six'
BASE = {v: LOGS / RUNS[v] / TARGET for v in ('d1', 'd2', 'c3', 'w2')}
ENTRIES = [f'{v}:{t}:{m}' for v, t, m in LIBRARY_SOURCES]
TRANSFER = [LOGS / 'online_library_transfer_20261122' / TARGET / f'from_{v}_{t}_{m}_pooled' for v, t, m in LIBRARY_SOURCES]
CANDIDATES = ('01m', '02m', '12m', '01u', '01d', '10u', '10d', '20u', '20d')
CANDIDATE_BASE = LOGS / 'online_select_candidates_20261123' / TARGET
SIMULATED = dict(zip(MARKS[1:], ('4', '8', '16', '36')))  # 快照对应离线模拟里的尝试数


def evaluation(base, mark):
    return [r for p in range(4) for r in read(base, f'target_{mark}_{p:02d}')['records']]


def done(base, mark):
    return all((base / f'target_{mark}_{p:02d}' / 'report.json').exists() for p in range(4))


def learning(base, marks, library_success):
    """各快照时每个会话已完成的尝试数，以及各会话领先的是哪一组；library_success[会话, 库编号] 是这组参数在该会话评估起点上的成功数。"""
    train, result = read(base, 'train')['records'], {}
    for mark in marks:
        attempts = np.zeros(60)
        for r in train:
            if r['arm'] == 'Updated' and r['control_step'] <= mark: attempts[r['session']] += 1
        heads = torch.load(base / 'train' / f'at_{mark}.pt', map_location='cpu', weights_only=True)['heads'][1::2]
        leader = np.asarray([h['selection']['entry'] for h in heads])
        arms = max(len(h['selection']['arms']) for h in heads)
        row = dict(mean_attempts_per_session=float(attempts.mean()), mean_rounds=float(np.mean([h['selection']['rounds'] for h in heads])),
                   leader_counts=np.bincount(leader, minlength=arms).tolist(), sessions_led_by_a_candidate=int((leader >= len(ENTRIES) + 1).sum()),
                   mean_tanh=np.tanh(np.asarray([np.asarray(h['w']) for h in heads])).mean(0).tolist())
        if library_success is not None:
            inside = leader <= len(ENTRIES)
            row['expected_from_C0_for_library_leaders'] = float(library_success[np.flatnonzero(inside), leader[inside]].sum())
            row['sessions_with_library_leader'] = int(inside.sum())
        result[str(mark)] = row
    final = json.loads((base / 'train/selection.json').read_text())['sessions']
    arms = max(len(s['arms']) for s in final)
    pad = lambda rows: np.asarray([list(r) + [0] * (arms - len(r)) for r in rows])
    tries, wins = pad([s['tries'] for s in final]).sum(0), pad([s['wins'] for s in final]).sum(0)
    result.update(train_tries_by_arm=tries.tolist(), train_wins_by_arm=wins.tolist(), sessions_that_added_candidates=sum(bool(s['added']) for s in final),
                  train_success={arm: float(np.mean([r['outcome'] == 'success' for r in train if r['arm'] == arm])) for arm in ('Frozen', 'Updated')},
                  train_posture={arm: float(np.mean([r['outcome'] == 'posture_failure' for r in train if r['arm'] == arm])) for arm in ('Frozen', 'Updated')})
    return result


def retention(base, p, target):
    schedule = [j for j in jobs()[1:] if j['kind'] in ('train', 'target', 'retention')]
    reports = {j['name']: read(base, j['name'], 'report.json') for j in schedule}
    assert all(r['status'] == 'passed' for r in reports.values())
    retained = [r for j in schedule if j['kind'] == 'retention' for r in read(base, j['name'])['records']]
    families = {f: paired_summary([r for r in retained if FAMILY[r['condition']] == f], repeats=2 * sum(v == f for v in FAMILY.values())) for f in dict.fromkeys(FAMILY.values())}
    flat = families['flat']
    ratio = flat['Updated']['xy_velocity_rmse'] / flat['Frozen']['xy_velocity_rmse']
    five = dict(target_gain_at_least_10pp=target['mean_session_delta'] >= p['success_gain_min'], ci_lower_above_zero=target['ci95'][0] > p['ci_lower_min'],
        posture_increase_at_most_2pp=target['posture_delta'] <= p['safety_increase_max'],
        every_family_loss_at_most_5pp=all(v['mean_session_delta'] >= -p['family_loss_max'] for v in families.values()),
        flat_velocity_ratio_at_most_1p10=ratio <= 1.10)
    return dict(five_criteria=five, all_five_met=all(five.values()), retention_families=families,
        retention_subtypes={c: paired_summary([r for r in retained if r['condition'] == c], repeats=2) for c in TERRAINS},
        flat_velocity_rmse_ratio=ratio, interactions=sum(r['interactions'] for r in reports.values()))


def main():
    assert VARIANT == 'd1'
    partial = '--partial' in sys.argv
    p = protocol()
    transfer = [evaluation(base, mark) for base, (_, _, mark) in zip(TRANSFER, LIBRARY_SOURCES)]
    library_success = np.zeros((60, len(ENTRIES) + 1))
    for k, rows in enumerate(transfer):
        for r in rows:
            if r['arm'] == 'Updated': library_success[r['session'], k] += r['outcome'] == 'success'
    for r in transfer[0]:
        if r['arm'] == 'Frozen': library_success[r['session'], -1] += r['outcome'] == 'success'
    best = int(library_success.sum(0)[:-1].argmax())
    offline = json.loads((DOCS / 'OFFLINE.json').read_text())[TARGET]['rules']['thompson:success+w']
    result = dict(target=TARGET, primary_mark=PRIMARY, marks=list(MARKS[1:]), library_success=library_success.sum(0).tolist(), best_library_entry=ENTRIES[best])
    rows = {}
    for v in ('d1', 'd2'):
        marks = [m for m in MARKS[1:] if done(BASE[v], m)] if partial else list(MARKS[1:])
        rows[v] = {m: evaluation(BASE[v], m) for m in marks}
        c1, c2 = {m: evaluation(BASE['c3'], m) for m in marks}, {m: evaluation(BASE['w2'], m) for m in marks}
        result[v] = dict(marks=marks, versus_frozen={str(m): paired_summary(rows[v][m]) for m in marks},
            versus_C1={str(m): against_fixed(rows[v][m], c1[m]) for m in marks}, versus_C2={str(m): against_fixed(rows[v][m], c2[m]) for m in marks},
            versus_best_C0={str(m): against_fixed(rows[v][m], transfer[best]) for m in marks},
            frozen_reproduced={str(m): frozen_reproduced(rows[v][m], c1[m]) for m in marks},
            learning=learning(BASE[v], marks, library_success) if (BASE[v] / 'train/selection.json').exists() else {},
            simulated_success={str(m): offline[SIMULATED[m]]['expected_success_of_240'] for m in marks})
        if not partial: result[v].update(retention(BASE[v], p, result[v]['versus_frozen'][str(PRIMARY)]))
    result['D2_versus_D1'] = {str(m): against_fixed(rows['d2'][m], rows['d1'][m]) for m in result['d1']['marks'] if m in rows['d2']}
    result['C1_versus_frozen'] = {str(m): paired_summary(evaluation(BASE['c3'], m)) for m in MARKS[1:]}
    result['D0'] = {}
    for name in CANDIDATES:
        if done(CANDIDATE_BASE / name, 0):
            records = evaluation(CANDIDATE_BASE / name, 0)
            result['D0'][name] = dict(versus_frozen=paired_summary(records), versus_best_C0=against_fixed(records, transfer[best]),
                                      frozen_reproduced=frozen_reproduced(records, transfer[0]))
    if not partial:
        assert len(result['D0']) == len(CANDIDATES)
        d1 = result['d1']
        result['registered'] = dict(
            primary_gain_over_frozen=bool(d1['five_criteria']['target_gain_at_least_10pp'] and d1['five_criteria']['ci_lower_above_zero']),
            primary_better_than_C1=bool(d1['versus_C1'][str(PRIMARY)]['ci95'][0] > 0),
            candidates_help=bool(result['D2_versus_D1'][str(PRIMARY)]['ci95'][0] > 0))
        save(DOCS / f'RESULTS_{TARGET}.json', result)
    count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
    posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
    pp = lambda v: (round(100 * v['mean_session_delta'], 1), np.round(np.multiply(v['ci95'], 100), 1).tolist())
    print(TARGET, 'partial' if partial else 'final', '| library', result['library_success'], 'best', result['best_library_entry'])
    for v in ('d1', 'd2'):
        for m in result[v]['marks']:
            x, l = result[v]['versus_frozen'][str(m)], result[v]['learning'].get(str(m), {})
            print(f" {v} {m:>5}: attempts {l.get('mean_attempts_per_session', 0):.1f} leaders {l.get('leader_counts')} | Frozen {count(x, 'Frozen')} -> {count(x, 'Updated')} gain {pp(x)}"
                  f" posture {posture(x, 'Frozen')} -> {posture(x, 'Updated')} | simulated {result[v]['simulated_success'][str(m)]:.1f} from C0 {l.get('expected_from_C0_for_library_leaders')}"
                  f" | vs C1 {result[v]['versus_C1'][str(m)]['B_success']} {pp(result[v]['versus_C1'][str(m)])} | vs C2 {result[v]['versus_C2'][str(m)]['B_success']} {pp(result[v]['versus_C2'][str(m)])}"
                  f" | vs best C0 {pp(result[v]['versus_best_C0'][str(m)])} | frozen identical {result[v]['frozen_reproduced'][str(m)]['identical']}")
        l = result[v]['learning']
        if l: print(f"    train tries {l['train_tries_by_arm']} wins {l['train_wins_by_arm']} added in {l['sessions_that_added_candidates']} sessions | train success {l['train_success']}")
        if not partial: print('    five:', result[v]['five_criteria'], 'flat ratio', round(result[v]['flat_velocity_rmse_ratio'], 3),
                              'families', {f: round(100 * x['mean_session_delta'], 1) for f, x in result[v]['retention_families'].items()})
    for m, x in result['D2_versus_D1'].items(): print(f" D2 - D1 at {m}: {x['D_success']} vs {x['B_success']} {pp(x)}")
    for name, x in result['D0'].items():
        v = x['versus_frozen']; print(f" D0 {name}: {count(v, 'Updated')} gain {pp(v)} posture {posture(v, 'Frozen')} -> {posture(v, 'Updated')} | vs best C0 {pp(x['versus_best_C0'])} | frozen identical {x['frozen_reproduced']['identical']}")
    if not partial: print(' registered:', result['registered'])


if __name__ == '__main__': main()
