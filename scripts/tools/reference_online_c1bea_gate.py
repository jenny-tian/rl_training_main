"""C1-b-EA 的适用门槛：用做法二（S2-N）全部训练轨迹计算执行与请求不一致、残差被清零的步数比例。

规则与 docs/lite3-online-c1b-swing2-20261119/EA_GATE_PRECHECK.json 的抽样预检相同：只算策略步，排除每个回合的前 10 步；
不一致指执行残差与请求的最大差超过 1e-3，清零指请求超过 1e-3 而执行低于 1e-6。任一比例达到 5% 即适用。
执行比例按 c1b_s_ea.execution_sums 的定义计算（每条腿在 sag 方向上的系数，Σ 执行×请求 / Σ 请求²）。

用法：python reference_online_c1bea_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/rl_training'))
BASE = ROOT / 'logs/reference_six/online_c1b_s2n_20261119'
OUT = ROOT / 'docs/lite3-online-c1b-ea-20261120/EA_GATE.json'
HIPY, KNEE = -.10, .20


def coefficients(x):
    """(T, n, 12) -> (T, n, 4)：每条腿在 sag 方向上的系数。"""
    return (HIPY * x[..., 1::3] + KNEE * x[..., 2::3]) / (HIPY**2 + KNEE**2)


def main():
    result = {}
    for target in ('stairs_up_06', 'stairs_up_065'):
        directory = BASE / target / 'train'
        index = json.loads((directory / 'trace_index.json').read_text())
        assert index['control_steps'] == 60000 and len(index['chunks']) == 60
        n = np.zeros((2, 60)); mismatch = np.zeros((2, 60)); cleared = np.zeros((2, 60))
        num = np.zeros((2, 60, 2)); den = np.zeros((2, 60, 2))  # 组别 × 会话 × (前腿, 后腿)
        for chunk in index['chunks']:
            z = np.load(directory / chunk['file'])
            request, effect, phase, age = z['residual_request'], z['learned_effect'], z['phase'], z['age']
            policy = (phase == 1) & (age >= 10)
            bad = (np.abs(effect - request).max(-1) > 1e-3) & policy
            zero = (np.abs(request).max(-1) > 1e-3) & (np.abs(effect).max(-1) < 1e-6) & policy
            env = np.arange(request.shape[1]); session, arm = env // 8, env % 2
            r, e = coefficients(request), coefficients(effect)
            every = phase == 1  # 执行比例用全部策略步，与在线统计一致
            for a in (0, 1):
                pick = arm == a
                for store, value in ((n, policy), (mismatch, bad), (cleared, zero)):
                    store[a] += np.bincount(session[pick], weights=value[:, pick].sum(0), minlength=60)
                for g, legs in enumerate(((0, 1), (2, 3))):
                    rr = (r[:, pick][:, :, legs] * every[:, pick, None]); ee = e[:, pick][:, :, legs]
                    num[a, :, g] += np.bincount(session[pick], weights=(rr * ee).sum((0, 2)), minlength=60)
                    den[a, :, g] += np.bincount(session[pick], weights=(rr**2).sum((0, 2)), minlength=60)
            print(target, chunk['file'], flush=True)
        entry = {}
        for a, name in ((0, 'Frozen'), (1, 'Updated')):
            m, c = mismatch[a] / n[a], cleared[a] / n[a]
            ratio = num[a] / den[a]
            entry[name] = dict(steps=int(n[a].sum()), mismatch=float(mismatch[a].sum() / n[a].sum()), cleared=float(cleared[a].sum() / n[a].sum()),
                min_session_mismatch=float(m.min()), min_session_cleared=float(c.min()),
                sessions_above_5pct=int((np.maximum(m, c) >= .05).sum()),
                execution_ratio_front_rear=(num[a].sum(0) / den[a].sum(0)).tolist(),
                session_execution_ratio_quantiles_05_50_95=np.quantile(ratio, [.05, .5, .95], axis=0).tolist())
        entry['applicable'] = all(max(entry[name]['mismatch'], entry[name]['cleared']) >= .05 for name in ('Frozen', 'Updated'))
        result[target] = entry
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == '__main__': main()
