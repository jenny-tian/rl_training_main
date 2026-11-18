"""汇总 C1-b-H 训练前的脚本化检查和保持地形特征检查。方案见 docs/lite3-online-c1b-height-20261118/PLAN.md 第 5 节。

用法：C1A2_VARIANT=bh C1A2_TARGET=stairs_up_06 python reference_online_c1bh_script_results.py
"""
from __future__ import annotations

import collections
import json

import numpy as np

from reference_online_c1a2 import OUTPUT, TARGET, VARIANT
from reference_stairs_eval import ROOT, save

assert VARIANT == 'bh'
DOCS = ROOT / 'docs/lite3-online-c1b-height-20261118'
NAMES = ('rough', 'entry', 'middle', 'top')
SECTIONS = ((0., .5), (.5, 1.2), (1.2, 2.85), (2.85, 3.3), (3.3, 9.))


def trace(name):
    index = json.loads((OUTPUT / name / 'trace_index.json').read_text())
    parts = [np.load(OUTPUT / name / c['file']) for c in index['chunks']]
    return {k: np.concatenate([p[k] for p in parts]) for k in ('features', 'root', 'phase', 'code')}


def scripted():
    records = json.loads((OUTPUT / 'script' / 'episodes.json').read_text())['records']
    by = {(r['session'], r['repeat'], r['arm']): r for r in records}
    assert len(by) == len(records) == 720
    groups = {}
    for repeat, name in enumerate('BCD'):
        zero = [by[s, repeat, 'Frozen'] for s in range(120)]
        code = [by[s, repeat, 'Updated'] for s in range(120)]
        assert all(r['script'] == name for r in code) and all(r['script'] == 'A' for r in zero)
        gain = np.array([(c['outcome'] == 'success') - (z['outcome'] == 'success') for z, c in zip(zero, code)], float)
        rng = np.random.default_rng(0)
        boot = gain[rng.integers(0, 120, (10000, 120))].mean(1)
        groups[name] = dict(zero=collections.Counter(r['outcome'] for r in zero), scripted=collections.Counter(r['outcome'] for r in code),
            success_gain=float(gain.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
            posture_increase=sum(r['outcome'] == 'posture_failure' for r in code) - sum(r['outcome'] == 'posture_failure' for r in zero),
            scripted_failure_final_x=np.quantile([r['final_root'][0] for r in code if r['outcome'] != 'success'] or [np.nan], [.1, .5, .9]).tolist())
    # 同一起点的零码环境重复三次，记录三次结局是否一致。
    zero = np.array([[by[s, k, 'Frozen']['outcome'] == 'success' for k in range(3)] for s in range(120)])
    passed = {k: bool(v['success_gain'] >= .10 and v['posture_increase'] <= 3) for k, v in groups.items()}
    return dict(groups=groups, zero_success_by_copy=zero.mean(0).tolist(), zero_copies_agree=float((zero.min(1) == zero.max(1)).mean()),
                group_meets_condition=passed, check_passed=bool(passed['C'] or passed['D']))


def sections(name, conditions=None):
    data = trace(name)
    policy, x, f = data['phase'] == 1, data['root'][:, :, 0], data['features']
    result = {}
    for label, ids in (conditions or {TARGET: np.arange(x.shape[1])}).items():
        ids = np.asarray(ids)
        if conditions:
            m = policy[:, ids]
            result[label] = dict(steps=int(m.sum()), mean=f[:, ids][m].mean(0).tolist(), max=f[:, ids][m].max(0).tolist(),
                                 share_nonzero=(f[:, ids][m] > 0).mean(0).tolist(), code_abs_max=float(np.abs(data['code'][:, ids][m]).max()))
        else:
            for low, high in SECTIONS:
                m = policy & (x >= low) & (x < high)
                result[f'x {low}-{high}'] = dict(steps=int(m.sum()), mean=f[m].mean(0).tolist() if m.any() else None)
    return result


def main():
    result = dict(target=TARGET, feature_names=NAMES, scripted=scripted(), stair_sections=sections('script'),
                  missing_rays={k: json.loads((OUTPUT / k / 'report.json').read_text())['missing_rays'] for k in ('script', 'script_retention')})
    specs = json.loads((OUTPUT / 'script_retention' / 'config.json').read_text())['specs']
    groups = collections.defaultdict(list)
    for s in specs: groups[s['condition']].append(s['env_id'])
    result['retention_features'] = sections('script_retention', groups)
    save(DOCS / 'SCRIPT_CHECK.json', json.loads(json.dumps(result)))
    for k, v in result['scripted']['groups'].items():
        print(k, dict(v['zero']), '->', dict(v['scripted']), 'gain', round(v['success_gain'], 3), np.round(v['ci95'], 3).tolist(), 'posture', v['posture_increase'])
    print('zero copies', result['scripted']['zero_success_by_copy'], 'agree', result['scripted']['zero_copies_agree'], 'passed', result['scripted']['check_passed'])
    for k, v in result['stair_sections'].items(): print(k, v['steps'], None if v['mean'] is None else np.round(v['mean'], 2).tolist())
    for k, v in result['retention_features'].items(): print(k, v['steps'], 'mean', np.round(v['mean'], 2).tolist(), 'max', np.round(v['max'], 2).tolist(), 'nonzero', np.round(v['share_nonzero'], 2).tolist())


if __name__ == '__main__': main()
