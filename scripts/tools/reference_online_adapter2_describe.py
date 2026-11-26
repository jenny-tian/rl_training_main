"""S2′ 里只作描述的统计（docs/lite3-online-adapter2-20261126/PLAN.md 第 6 节末尾）：失败的回合停在哪里、实际执行的残差、
跨地形的保持结果。只读已有的记录，不跑仿真。

用法：python reference_online_adapter2_describe.py [变体:目标 ...]              缺省是 S2′ 的九条运行链加 7 cm 上阶上已有的 r2
      python reference_online_adapter2_describe.py --retention [变体:目标 ...]  保持地图上按行拆开的成功数（跨地形的结果）
"""
from __future__ import annotations

import collections
import glob
import json
import sys

import numpy as np

JOINTS = ('hipx', 'hipy', 'knee')
MARK = 10000
FULL_COMMAND = .4
SLIP_SPEED, DT = .10, .02  # 与 S0 筛选的支撑脚滑动定义相同


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


def merged_outcomes(outcomes):
    """到时间还没到终点的失败合成一类（not_reached）。坡面目标没有终点坐标，采集循环在这类目标上不会给出 stuck，
    全部记为 timeout；合并之后台阶和坡面的失败形式才能比较。"""
    result = {k: v for k, v in outcomes.items() if k not in ('stuck', 'timeout')}
    late = outcomes.get('stuck', 0) + outcomes.get('timeout', 0)
    if late: result['not_reached'] = late
    return result


def slip_counts(feet_xy, stance, policy):
    """（在滑的着地脚步数，着地脚步数）。feet_xy (T, n, 4, 2) 每步结束时的脚的位置；stance (T, n, 4) 每步开始时是否着地；
    policy (T, n)。一步里脚在开始和结束时都着地才算着地；水平位移折合速度超过 SLIP_SPEED 算在滑。"""
    feet_xy, stance, policy = np.asarray(feet_xy, float), np.asarray(stance, bool), np.asarray(policy, bool)
    speed = np.linalg.norm(feet_xy[1:-1] - feet_xy[:-2], axis=-1) / DT
    grounded = stance[1:-1] & stance[2:] & (policy[:-2] & policy[1:-1] & policy[2:])[:, :, None]
    return int((speed[grounded] > SLIP_SPEED).sum()), int(grounded.sum())


def slip_ratio(feet_xy, stance, policy):
    slipping, grounded = slip_counts(feet_xy, stance, policy)
    return slipping / grounded if grounded else None


def mean_speed(speed, command, policy):
    """指令升到 0.4 m/s 之后的策略步里机身前向速度的平均。三个数组都是 (T, n)。没有这样的步时返回 None。"""
    mask = np.asarray(policy, bool) & (np.asarray(command) >= FULL_COMMAND - 1e-3)
    return float(np.asarray(speed)[mask].mean()) if mask.any() else None


def retention_counts(records):
    """保持评估的记录按（地形、地图的行）拆开：每组的回合数和两组各自的成功数。两组回合数不等时报错。"""
    counts = collections.defaultdict(lambda: collections.Counter())
    for r in records:
        key = r['condition'], r['row']
        counts[key][r['arm'] + '_n'] += 1
        counts[key][r['arm']] += r['outcome'] == 'success'
    result = {}
    for key, c in counts.items():
        if c['Frozen_n'] != c['Updated_n']:
            raise ValueError(f'unpaired retention records for {key}')
        result[key] = dict(n=c['Frozen_n'], Frozen=c['Frozen'], Updated=c['Updated'])
    return result


def run_location(variant, target, runs):
    """一条运行的评估记录在哪里：（logs/reference_six 下的目录，用来比较的快照）。rule 是不学习的规则对照，只有快照 0。"""
    if variant == 'rule': return ('online_directed_rule_20261126', target, 'directed'), 0
    return (runs[variant], target), MARK


def describe(variant, target):
    from reference_online_c1a2 import RUNS  # 只用到各变体的输出目录名
    from reference_stairs_eval import ROOT
    parts, mark = run_location(variant, target, RUNS)
    base = ROOT.joinpath('logs/reference_six', *parts)
    if not all((base / f'target_{mark}_{p:02d}' / 'report.json').exists() for p in range(4)):
        return print(variant, target, 'not finished')
    outcomes = {arm: collections.Counter() for arm in ('Frozen', 'Updated')}
    places = {arm: collections.Counter() for arm in ('Frozen', 'Updated')}
    steps = {arm: [] for arm in ('Frozen', 'Updated')}
    sums = {arm: {phase: {j: [0., 0] for j in JOINTS} for phase in ('swing', 'stance')} for arm in ('Frozen', 'Updated')}
    motion = {arm: dict(speed=[], command=[], policy=[]) for arm in ('Frozen', 'Updated')}
    slip = {arm: [0, 0] for arm in ('Frozen', 'Updated')}
    fronts = None
    for part in range(4):
        d = base / f'target_{mark}_{part:02d}'
        geometry = json.loads((d / 'geometry.json').read_text())[f'{target}:0']
        fronts = geometry['fronts']
        records = json.loads((d / 'episodes.json').read_text())['records']
        keys = ('learned_effect', 'stance', 'phase', 'feet', 'body_velocity', 'command')
        trace = {k: [] for k in keys}
        for f in sorted(glob.glob(str(d / 'trace_*.npz'))):
            z = np.load(f)
            for k in keys: trace[k].append(z[k])
        trace = {k: np.concatenate(v) for k, v in trace.items()}
        policy = trace['phase'] == 1
        for arm, offset in (('Frozen', 0), ('Updated', 1)):
            legs = trace['stance'][:, offset::2]
            motion[arm]['speed'].append(trace['body_velocity'][:, offset::2, 0]); motion[arm]['command'].append(trace['command'][:, offset::2, 0])
            motion[arm]['policy'].append(policy[:, offset::2])
            a, b = slip_counts(trace['feet'][:, offset::2, :, :2], legs, policy[:, offset::2]); slip[arm][0] += a; slip[arm][1] += b
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
    print(f'{variant} {target} @ {mark}')
    for arm in ('Frozen', 'Updated'):
        mean = lambda phase: ' '.join(f"{j} {sums[arm][phase][j][0] / max(sums[arm][phase][j][1], 1):.3f}" for j in JOINTS)
        speed = mean_speed(*(np.concatenate(motion[arm][k]) for k in ('speed', 'command', 'policy')))
        print(f"  {arm:7s} {merged_outcomes(outcomes[arm])} mean steps {np.mean(steps[arm]):.0f} mean speed {speed:.3f} stance slip {slip[arm][0] / max(slip[arm][1], 1):.3f}" + (f" | not reached, by place {dict(places[arm])}" if fronts else '')
              + (f" | executed residual swing {mean('swing')} stance {mean('stance')}" if arm == 'Updated' else ''))


def retention_rows(variant, target, mark):
    from reference_online_c1a2 import RUNS
    from reference_stairs_eval import ROOT
    parts, rule_mark = run_location(variant, target, RUNS)
    base, mark = ROOT.joinpath('logs/reference_six', *parts), (rule_mark if variant == 'rule' else mark)
    if not all((base / f'retention_{mark}_{p:02d}' / 'report.json').exists() for p in range(20)):
        return print(variant, target, 'no retention at', mark)
    records = [r for p in range(20) for r in json.loads((base / f'retention_{mark}_{p:02d}' / 'episodes.json').read_text())['records']]
    counts = retention_counts(records)
    show = [('slope_rough_down', 8), ('slope_rough_down', 1), ('stairs_up', 8), ('stairs_up', 1), ('stairs_down', 8), ('slope_rough_up', 8), ('rough', 8), ('slope_down', 8)]
    print(f'{variant:5s} trained on {target:22s} @ {mark:5d} | ' + ' | '.join(f"{c}@{row} {counts[c, row]['Frozen']}->{counts[c, row]['Updated']}/{counts[c, row]['n']}" for c, row in show))


def main():
    from reference_online_adapter_results import STAGE2_RUNS
    if '--retention' in sys.argv:
        runs = [tuple(a.split(':')) for a in sys.argv[1:] if not a.startswith('--')]
        for variant, target in runs or [('r2', 'stairs_up_07'), ('r2', 'slope_rough_up_x11'), ('mfa', 'stairs_up_07'), ('mrb', 'stairs_up_07'), ('mrb', 'slope_rough_up_x11'),
                                        ('mrcz', 'stairs_up_07'), *[r for r in STAGE2_RUNS if r[0] != 'r2']]:
            retention_rows(variant, target, 2500 if variant == 'r2' else MARK)
        return
    runs = [tuple(a.split(':')) for a in sys.argv[1:]] or [('r2', 'stairs_up_07'), *STAGE2_RUNS]
    for variant, target in runs: describe(variant, target)


if __name__ == '__main__': main()
