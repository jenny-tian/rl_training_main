"""S2′ 里只作描述的统计（docs/lite3-online-adapter2-20261126/PLAN.md 第 6 节末尾）：失败的回合停在哪里、实际执行的残差、
跨地形的保持结果。只读已有的记录，不跑仿真。

用法：python reference_online_adapter2_describe.py [变体:目标 ...]     缺省是 S2′ 的九条运行链加 7 cm 上阶上已有的 r2
"""
from __future__ import annotations

import collections
import glob
import json
import sys

import numpy as np

JOINTS = ('hipx', 'hipy', 'knee')
MARK = 10000


def risers_passed(feet_x, fronts):
    """每只脚已经越过了几个立面。feet_x (4,) 四只脚的前向位置，fronts 是各立面的位置（递增）。"""
    feet_x, fronts = np.asarray(feet_x, float), np.asarray(fronts, float)
    if feet_x.shape != (4,) or fronts.ndim != 1 or len(fronts) == 0:
        raise ValueError('four feet and at least one riser required')
    return (feet_x[:, None] > fronts[None, :]).sum(1)


def failure_place(feet_x, fronts):
    """回合结束时停在台阶的哪里。脚的顺序是 FL FR HL HR。

    before      没有一只脚过第一个立面
    last_riser  两条前腿都过了全部立面，后腿最多只差最后一级
    past        四只脚都过了最后一个立面
    on_stairs   其余：停在途中
    """
    passed, n = risers_passed(feet_x, fronts), len(fronts)
    if (passed == n).all(): return 'past'
    if (passed[:2] == n).all() and passed[2:].min() >= n - 1: return 'last_riser'
    if passed.max() == 0: return 'before'
    return 'on_stairs'


def phase_means(effect, stance, policy):
    """实际执行的残差按关节和摆动、支撑分开的平均绝对值。effect (T, n, 12)，stance (T, n, 4)，policy (T, n) 是否在策略阶段。"""
    effect, stance, policy = np.asarray(effect), np.asarray(stance, bool), np.asarray(policy, bool)
    result = {}
    for name, legs in (('swing', ~stance), ('stance', stance)):
        mask = legs & policy[:, :, None]
        result[name] = {joint: float(np.abs(effect[:, :, k::3])[mask].mean()) if mask.any() else None for k, joint in enumerate(JOINTS)}
    return result


def describe(variant, target):
    from reference_online_c1a2 import RUNS  # 只用到各变体的输出目录名
    from reference_stairs_eval import ROOT
    base = ROOT / 'logs/reference_six' / RUNS[variant] / target
    if not all((base / f'target_{MARK}_{p:02d}' / 'report.json').exists() for p in range(4)):
        return print(variant, target, 'not finished')
    outcomes = {arm: collections.Counter() for arm in ('Frozen', 'Updated')}
    places = {arm: collections.Counter() for arm in ('Frozen', 'Updated')}
    steps = {arm: [] for arm in ('Frozen', 'Updated')}
    sums = {arm: {phase: {j: [0., 0] for j in JOINTS} for phase in ('swing', 'stance')} for arm in ('Frozen', 'Updated')}
    fronts = None
    for part in range(4):
        d = base / f'target_{MARK}_{part:02d}'
        geometry = json.loads((d / 'geometry.json').read_text())[f'{target}:0']
        fronts = geometry['fronts']
        records = json.loads((d / 'episodes.json').read_text())['records']
        keys = ('learned_effect', 'stance', 'phase', 'feet')
        trace = {k: [] for k in keys}
        for f in sorted(glob.glob(str(d / 'trace_*.npz'))):
            z = np.load(f)
            for k in keys: trace[k].append(z[k])
        trace = {k: np.concatenate(v) for k, v in trace.items()}
        policy = trace['phase'] == 1
        for arm, offset in (('Frozen', 0), ('Updated', 1)):
            legs = trace['stance'][:, offset::2]
            for phase, mask in (('swing', ~legs), ('stance', legs)):
                mask = mask & policy[:, offset::2, None]
                for k, joint in enumerate(JOINTS):
                    values = np.abs(trace['learned_effect'][:, offset::2, k::3])[mask]
                    sums[arm][phase][joint][0] += float(values.sum()); sums[arm][phase][joint][1] += values.size
        for r in records:
            outcomes[r['arm']][r['outcome']] += 1
            steps[r['arm']].append(r['policy_steps'])
            if fronts and r['outcome'] in ('stuck', 'timeout'):
                last = np.flatnonzero(policy[:, r['env_id']])[-1]
                places[r['arm']][failure_place(trace['feet'][last, r['env_id'], :, 0], fronts)] += 1
    print(f'{variant} {target} @ {MARK}')
    for arm in ('Frozen', 'Updated'):
        mean = lambda phase: ' '.join(f"{j} {sums[arm][phase][j][0] / max(sums[arm][phase][j][1], 1):.3f}" for j in JOINTS)
        print(f"  {arm:7s} {dict(outcomes[arm])} mean steps {np.mean(steps[arm]):.0f}" + (f" | stuck or timeout by place {dict(places[arm])}" if fronts else '')
              + (f" | executed residual swing {mean('swing')} stance {mean('stance')}" if arm == 'Updated' else ''))


def main():
    from reference_online_adapter_results import STAGE2_RUNS
    runs = [tuple(a.split(':')) for a in sys.argv[1:]] or [('r2', 'stairs_up_07'), *STAGE2_RUNS]
    for variant, target in runs: describe(variant, target)


if __name__ == '__main__': main()
