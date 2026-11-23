"""摆动受阻信号的离线检查：受阻比例随摆动幅度怎样变化，在平地和起伏地形上各是多少。不占 GPU。

用法：python reference_online_stall_offline.py
受阻的定义与窗口学习器相同：腿在摆动且足端前向速度低于 0.10 m/s。三部分数据：
  1. 做法二在 6 cm、6.5 cm 上阶的训练回合（每个回合的摆动幅度是随机扰动出来的），到达台阶后 4 秒内的受阻比例和成功率，按幅度分组；
  2. 7 cm 上阶和 7 cm 箱体上固定参数的评估（全 0、库参数、摆动幅度乘 0.8 和 1.2），整回合的受阻比例；
  3. 同一批评估里 2 秒窗口的受阻比例，分平地和起伏地形。
结果写到 docs/lite3-online-stall-20261123/OFFLINE.json。
"""
from __future__ import annotations

import numpy as np

from reference_stairs_eval import ROOT, save

LOGS = ROOT / 'logs/reference_six'
EDGES = (-1., -.5, -.2, .2, .5, .7, .85, 1.01)
TRANSFER, CANDIDATE = LOGS / 'online_library_transfer_20261122', LOGS / 'online_select_candidates_20261123'
SOURCE = ('from_s2n_stairs_up_06_60000_pooled', 'from_s2n_stairs_up_065_60000_pooled', 'from_bs_boxes_06_40000_pooled')
MARK = (60000, 60000, 40000)


def dose(target):
    z = np.load(LOGS / 'online_dense_signal_20261120' / f'{target}.npz')
    ok = (z['arrival'] >= 0) & np.isfinite(z['success']) & np.isfinite(z['code'][..., 0])
    success, result = z['success'][ok], {}
    for g, name in enumerate(('front', 'rear')):
        amplitude = z['code'][..., g][ok]
        stall = (z['stall'][:, :, 1, g] / np.maximum(z['swing'][:, :, 1, g], 1))[ok]
        bins = [(amplitude >= lo) & (amplitude < hi) for lo, hi in zip(EDGES, EDGES[1:])]
        high = amplitude > .5
        result[name] = dict(stall_by_bin=[float(stall[m].mean()) for m in bins], success_by_bin=[float(success[m].mean()) for m in bins], episodes_by_bin=[int(m.sum()) for m in bins],
            correlation_amplitude_stall=float(np.corrcoef(amplitude, stall)[0, 1]), correlation_amplitude_stall_above_half=float(np.corrcoef(amplitude[high], stall[high])[0, 1]))
    return dict(episodes=int(ok.sum()), bin_edges=list(EDGES), **result)


def windows(base, mark, arm):
    """arm：0 是 Frozen（全 0），1 是 Updated。返回整回合的前后腿受阻比例，以及平地、起伏地形上各窗口的受阻比例。"""
    swing, stalled, rows = np.zeros(2), np.zeros(2), dict(flat=[], rough=[])
    for part in range(4):
        z = np.load(base / f'target_{mark}_{part:02d}' / 'trace_0000.npz')
        speed = np.zeros(z['stance'].shape); speed[1:] = (z['feet'][1:, :, :, 0] - z['feet'][:-1, :, :, 0]) / .02
        for env in range(arm, 120, 2):
            steps = np.flatnonzero(z['phase'][:, env] == 1)[1:]
            lifted = ~z['stance'][steps, env]; blocked = lifted & (speed[steps, env] < .10); rough = z['features'][steps, env, 0]
            on = rough > .5
            swing += lifted[on].reshape(-1, 2, 2).sum((0, 2)); stalled += blocked[on].reshape(-1, 2, 2).sum((0, 2))
            for w in range(len(steps) // 100):
                part_ = slice(w * 100 + 10, (w + 1) * 100)
                s, t, r = lifted[part_].reshape(-1, 2, 2).sum((0, 2)), blocked[part_].reshape(-1, 2, 2).sum((0, 2)), rough[part_].mean()
                if (s > 0).all() and (r >= .5 or r < .05): rows['rough' if r >= .5 else 'flat'].append(t / s)
    summary = lambda v: dict(windows=len(v), mean=np.mean(v, 0).tolist(), quartiles=np.percentile(v, [25, 50, 75], 0).T.tolist()) if len(v) else dict(windows=0)
    return dict(episode_stall=(stalled / swing).tolist(), flat=summary(rows['flat']), rough=summary(rows['rough']))


def main():
    result = dict(dose={t: dose(t) for t in ('stairs_up_06', 'stairs_up_065')}, fixed={})
    for t, d in result['dose'].items():
        for g in ('front', 'rear'): print(t, g, 'stall', np.round(d[g]['stall_by_bin'], 3).tolist(), 'success', np.round(d[g]['success_by_bin'], 2).tolist(),
                                          'corr', round(d[g]['correlation_amplitude_stall'], 3), 'above 0.5', round(d[g]['correlation_amplitude_stall_above_half'], 3))
    for target in ('stairs_up_07', 'boxes_07'):
        row = dict(zero=windows(TRANSFER / target / SOURCE[0], MARK[0], 0))
        for k in range(3):
            other = '1' if k == 0 else '0'
            row[f'entry{k} x0.8'] = windows(CANDIDATE / target / f'{k}{other}d', 0, 1)
            row[f'entry{k} x1.0'] = windows(TRANSFER / target / SOURCE[k], MARK[k], 1)
            row[f'entry{k} x1.2'] = windows(CANDIDATE / target / f'{k}{other}u', 0, 1)
        result['fixed'][target] = row
        for name, v in row.items(): print(target, f'{name:12s} episode stall', np.round(v['episode_stall'], 3).tolist(), '| rough windows', np.round(v['rough']['mean'], 3).tolist(),
                                          '| flat windows', np.round(v['flat'].get('mean', [np.nan] * 2), 3).tolist(), v['flat']['windows'])
    save(ROOT / 'docs/lite3-online-stall-20261123/OFFLINE.json', result)


if __name__ == '__main__': main()
