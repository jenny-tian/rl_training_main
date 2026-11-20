"""不占 GPU 的分析：用做法二 Frozen 组的训练回合（均值为 0，只有探索扰动），比较几种反馈量各自算出的一轮更新有多大把握走对方向。

每轮是一个会话的 4 个并行回合，按学习器的留一法优势和噪声算出更新量。走对方向指两个摆动幅度的更新量为正。
先运行 reference_online_dense_signal.py <目标> 生成逐回合的窗口统计。结果写到 docs/lite3-online-transfer-20261120/DENSE_SIGNAL.json。
"""
from __future__ import annotations

import numpy as np

from reference_stairs_eval import ROOT, save

LOGS = ROOT / 'logs/reference_six'


def directions(target):
    z = np.load(LOGS / 'online_dense_signal_20261120' / f'{target}.npz', allow_pickle=True)
    progress, alive, stall, swing, ret = (z[k] for k in ('progress', 'alive', 'stall', 'swing', 'ret'))
    table = np.load(LOGS / 'online_c1b_s2n_20261119' / target / 'train/noise_table.npy')
    frozen = np.arange(0, ret.shape[0], 2); session, replica = frozen // 8, (frozen % 8) // 2
    noise = np.stack([table[session, 4 * e + replica, 2:] for e in range(ret.shape[1])], 1)
    assert np.allclose(np.tanh(.7 * noise)[np.isfinite(ret[frozen])], z['code'][frozen][np.isfinite(ret[frozen])], atol=1e-6)
    p, a, st, sw = progress[frozen], alive[frozen], stall[frozen], swing[frozen]
    unimpeded = lambda w: np.where(sw[..., w, :] > 0, 1 - st[..., w, :] / np.maximum(sw[..., w, :], 1), np.nan)  # 前腿、后腿各一个
    signals = {'whole_episode_return': ret[frozen][..., None], 'progress_0_2s': np.where(a[..., 0] > 0, p[..., 0], np.nan)[..., None],
               'progress_0_4s': np.where(a[..., 1] > 0, p[..., 1], np.nan)[..., None], 'progress_0_8s': np.where(a[..., 2] > 0, p[..., 2], np.nan)[..., None],
               'own_legs_unimpeded_swing_0_2s': unimpeded(0), 'own_legs_unimpeded_swing_0_4s': unimpeded(1), 'own_legs_unimpeded_swing_0_8s': unimpeded(2)}
    result = {}
    for name, x in signals.items():
        steps = []
        for s in range(60):
            rows = np.flatnonzero(session == s)
            for e in range(ret.shape[1]):
                v = x[rows, e].astype(float)
                if not np.isfinite(v).all(): continue
                steps.append(((v - (v.sum(0) - v) / 3) * noise[rows, e]).mean(0))
        steps = np.asarray(steps); snr = steps.mean(0) / steps.std(0)
        result[name] = dict(rounds=len(steps), right_direction_front_rear=(steps > 0).mean(0).tolist(), signal_to_noise=snr.tolist(),
                            rounds_for_90_percent=((1.2816 / snr)**2).tolist())
        print(f'{target} {name:32s} rounds {len(steps)} right {np.round((steps > 0).mean(0), 3).tolist()} snr {snr.round(3).tolist()} rounds for 90% {np.round((1.2816 / snr)**2).tolist()}')
    return result


if __name__ == '__main__':
    save(ROOT / 'docs/lite3-online-transfer-20261120/DENSE_SIGNAL.json', {t: directions(t) for t in ('stairs_up_06', 'stairs_up_065')})
