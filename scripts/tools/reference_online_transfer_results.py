"""汇总参数迁移评估。方案见 docs/lite3-online-transfer-20261120/PLAN.md。

用法：python reference_online_transfer_results.py（不需要环境变量；逐个读取 8 组评估，缺的跳过）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reference_online_c1a_results import paired_summary
from reference_online_c1aea_results import against_fixed, frozen_reproduced
from reference_stairs_eval import ROOT, save

LOGS, DOCS = ROOT / 'logs/reference_six', ROOT / 'docs/lite3-online-transfer-20261120'
# 目标 -> 它原来那一轮评估的目录、判定时刻、已评估的快照
NATIVE = dict(stairs_up_06=('online_c1b_s2n_20261119', 60000, (5000, 10000, 20000, 40000, 60000)),
              stairs_up_065=('online_c1b_s2n_20261119', 60000, (5000, 10000, 20000, 40000, 60000)),
              boxes_06=('online_c1b_swing_20261118', 40000, (20000, 40000, 60000)))
CASES = [('T1', 'stairs_up_065', 's2n:stairs_up_06:60000'), ('T2', 'stairs_up_06', 's2n:stairs_up_065:60000'),
         ('T3', 'boxes_06', 's2n:stairs_up_06:60000'), ('T4', 'stairs_up_06', 'bs:boxes_06:40000')]


def records(base, mark):
    paths = [base / f'target_{mark}_{p:02d}' for p in range(4)]
    if not all((p / 'report.json').exists() and json.loads((p / 'report.json').read_text())['status'] == 'passed' for p in paths): return None
    return [r for p in paths for r in json.loads((p / 'episodes.json').read_text())['records']]


def parameters(rows):
    w = np.asarray([r['mean'] for r in rows if r['arm'] == 'Updated'])
    return dict(mean_tanh=np.tanh(w).mean(0).tolist(), distinct=len({tuple(x) for x in w.tolist()}))


def main():
    result = {}
    for label, target, source in CASES:
        folder, primary, marks = NATIVE[target]
        native = {m: records(LOGS / folder / target, m) for m in marks}
        assert all(native.values()), f'native evaluation missing for {target}'
        for mode in ('', ':pooled'):
            rows = records(LOGS / 'online_transfer_20261120' / target / ('from_' + (source + mode).replace(':', '_')), int(source.split(':')[2]))
            if rows is None: continue
            summary = paired_summary(rows)
            entry = dict(target=target, source=source, mode='pooled' if mode else 'per_session', transfer_versus_frozen=summary,
                frozen_reproduced=frozen_reproduced(rows, native[primary]), parameters=parameters(rows),
                native_versus_frozen={str(m): paired_summary(native[m]) for m in marks},
                transfer_versus_native={str(m): against_fixed(rows, native[m]) for m in marks})
            entry['share_of_native_gain'] = summary['mean_session_delta'] / entry['native_versus_frozen'][str(primary)]['mean_session_delta']
            result[label + ('_pooled' if mode else '')] = entry
            pct = lambda v: f"{100 * v['mean_session_delta']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]"
            count = lambda v, arm: v[arm]['outcomes'].get('success', 0)
            posture = lambda v, arm: v[arm]['outcomes'].get('posture_failure', 0)
            print(f"{label}{' 平均' if mode else ' 逐会话'} {target} <- {source}: Frozen {count(summary, 'Frozen')} -> 迁移 {count(summary, 'Updated')} / 240, {pct(summary)}, "
                  f"姿态终止 {posture(summary, 'Frozen')} -> {posture(summary, 'Updated')}, Frozen 一致 {entry['frozen_reproduced']['identical']}/{entry['frozen_reproduced']['n']}, "
                  f"tanh 均值 {np.round(entry['parameters']['mean_tanh'], 2).tolist()}")
            for m in marks:
                n, v = entry['native_versus_frozen'][str(m)], entry['transfer_versus_native'][str(m)]
                print(f"    本地学习 {m:>5} 步: {count(n, 'Updated')} ({pct(n)})；迁移 − 本地 {pct(v)}")
            print(f"    迁移增益占本地学习（{primary} 步）增益的 {100 * entry['share_of_native_gain']:.0f}%")
    save(DOCS / 'RESULTS.json', result)


if __name__ == '__main__': main()
