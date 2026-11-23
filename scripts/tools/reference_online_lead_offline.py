"""提前起抬的离线检查：新地形上的步态节律，以及各提前规则在每步变化上限下能把多少摆动残差送到摆动期。不占 GPU。

用法：python reference_online_lead_offline.py
数据是阶段 C 的不学习评估轨迹（7 cm 上阶用 6 cm 上阶那组参数，7 cm 箱体用 6 cm 箱体那组），只取起伏特征大于 0.5 的步。
残差用一个只含每步变化上限的限幅器在记录的着地序列上重算，没有包含安全过滤的其他保护，也没有考虑残差改变步态。
结果写到 docs/lite3-online-lead-20261123/OFFLINE.json。
"""
from __future__ import annotations

import numpy as np
import torch

from rl_training.ea_lpl.swing_lead import RULES, SwingLead
from reference_stairs_eval import ROOT, save

LOGS = ROOT / 'logs/reference_six/online_library_transfer_20261122'
CASES = (('stairs_up_07', 'from_s2n_stairs_up_06_60000_pooled', 60000), ('boxes_07', 'from_bs_boxes_06_40000_pooled', 40000))
AMPLITUDE, RATE = .18, .02


def limiter(request):
    out, previous = np.zeros_like(request), np.zeros(request.shape[1])
    for t, r in enumerate(request):
        previous = np.clip(r, previous - RATE, previous + RATE); out[t] = previous
    return out


def main():
    result = {}
    for target, source, mark in CASES:
        row = {}
        for arm, name in ((0, 'Frozen'), (1, 'Updated')):
            sums = {rule: np.zeros(4) for rule in RULES}
            swing_steps, stance_steps, periods, recorded = [], [], [], np.zeros(4)
            for part in range(4):
                z = np.load(LOGS / target / source / f'target_{mark}_{part:02d}' / 'trace_0000.npz')
                for env in range(arm, 120, 2):
                    steps = np.flatnonzero(z['phase'][:, env] == 1)
                    steps = steps[z['features'][steps, env, 0] > .5]
                    if len(steps) < 60: continue
                    if (np.diff(steps) != 1).any(): steps = steps[:np.flatnonzero(np.diff(steps) != 1)[0] + 1]
                    if len(steps) < 60: continue
                    stance, base = z['stance'][steps, env], z['base_action'][steps, env]
                    knee = np.arange(2, 12, 3)
                    recorded += [z['residual_request'][steps, env][:, knee][~stance].sum(), z['learned_effect'][steps, env][:, knee][~stance].sum(),
                                 z['learned_effect'][steps, env][:, knee][stance].sum(), 0]
                    for k in range(4):
                        change = np.flatnonzero(np.diff(stance[:, k].astype(int)) != 0) + 1
                        lifts = [s[0] for s in np.split(np.arange(len(steps)), change)[1:-1] if not stance[s[0], k]]
                        for s in np.split(np.arange(len(steps)), change)[1:-1]: (stance_steps if stance[s[0], k] else swing_steps).append(len(s))
                        periods += [b - a for a, b in zip(lifts, lifts[1:])]
                    for rule in RULES:
                        lead = SwingLead(rule, 1)
                        used = np.stack([lead.step(torch.as_tensor(stance[t:t + 1]), torch.as_tensor(base[t:t + 1]))[0].numpy() for t in range(len(steps))])
                        executed = limiter(AMPLITUDE * (~used).astype(float))
                        sums[rule] += [executed[~stance].sum(), (~stance).sum(), executed[stance].sum(), stance.sum()]
            q = lambda x: np.percentile(x, [25, 50, 75]).tolist()
            row[name] = dict(swing_steps_quartiles=q(swing_steps), stance_steps_quartiles=q(stance_steps), period_steps_quartiles=q(periods),
                rules={rule: dict(executed_in_swing=float(s[0] / s[1]), executed_in_stance=float(s[2] / s[3])) for rule, s in sums.items()},
                swing_fraction=float(sums['none'][1] / (sums['none'][1] + sums['none'][3])),
                recorded=dict(requested_knee_in_swing=float(recorded[0] / sums['none'][1]), executed_knee_in_swing=float(recorded[1] / sums['none'][1]),
                              executed_knee_in_stance=float(recorded[2] / sums['none'][3])))
            print(target, name, 'swing', row[name]['swing_steps_quartiles'], 'stance', row[name]['stance_steps_quartiles'], 'period', row[name]['period_steps_quartiles'],
                  '| recorded', {k: round(v, 3) for k, v in row[name]['recorded'].items()})
            for rule, v in row[name]['rules'].items(): print(f"   {rule:8s} swing {v['executed_in_swing']:.3f} stance {v['executed_in_stance']:.3f}")
        result[target] = row
    save(ROOT / 'docs/lite3-online-lead-20261123/OFFLINE.json', dict(amplitude=AMPLITUDE, rate_limit=RATE, targets=result))


if __name__ == '__main__': main()
