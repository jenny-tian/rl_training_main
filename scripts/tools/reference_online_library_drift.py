"""加快学习 C 的补充检查：选择之后的窗口修正让参数朝一致的方向移动，还是只是分散开。不占 GPU。

用法：python reference_online_library_drift.py
每个会话的移动量是各快照的参数减去它选中的那一组库参数（w 空间）；距离是 tanh 后 4 个参数的绝对差之和。
按距离的中位数把会话分成两半，比较两半在该快照的成功率增益。结果写到 docs/lite3-online-library-20261122/REFINEMENT_DRIFT.json。
"""
from __future__ import annotations

import json

import numpy as np
import torch

from reference_stairs_eval import ROOT, save

DOCS = ROOT / 'docs/lite3-online-library-20261122'
LOGS = ROOT / 'logs/reference_six/online_c1b_library_20261122'
MARKS = (2500, 5000, 10000)


def main():
    result = {}
    for target in ('stairs_up_07', 'boxes_07'):
        chosen = json.loads((LOGS / target / 'train/selection.json').read_text())
        start = np.asarray(chosen['library'])[[s['entry'] for s in chosen['sessions']]]
        summary = json.loads((DOCS / f'RESULTS_{target}.json').read_text())
        result[target] = {}
        for mark in MARKS:
            heads = torch.load(LOGS / target / 'train' / f'at_{mark}.pt', map_location='cpu', weights_only=True)['heads'][1::2]
            w = np.asarray([np.asarray(h['w']) for h in heads])
            moved = w - start
            distance = np.abs(np.tanh(w) - np.tanh(start)).sum(1)
            gain = np.asarray(summary['C1_versus_frozen'][str(mark)]['session_deltas'])
            far = distance > np.median(distance)
            row = dict(mean_move=moved.mean(0).tolist(), std_move=moved.std(0).tolist(),
                       mean_move_over_standard_error=(moved.mean(0) / np.maximum(moved.std(0) / np.sqrt(len(moved)), 1e-12)).tolist(),
                       sessions_moved=int((distance > 0).sum()), median_distance=float(np.median(distance)))
            if far.any() and (~far).any():
                row.update(gain_near=float(gain[~far].mean()), gain_far=float(gain[far].mean()), sessions_near=int((~far).sum()),
                           sessions_far=int(far.sum()), correlation_distance_gain=float(np.corrcoef(distance, gain)[0, 1]))
            result[target][str(mark)] = row
            print(target, mark, 'mean', np.round(row['mean_move'], 2).tolist(), 'std', np.round(row['std_move'], 2).tolist(),
                  'near/far gain pp', [round(100 * row[k], 1) for k in ('gain_near', 'gain_far') if k in row])
    save(DOCS / 'REFINEMENT_DRIFT.json', result)


if __name__ == '__main__': main()
