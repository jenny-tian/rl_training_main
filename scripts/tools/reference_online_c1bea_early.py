"""由做法二（s2n）的训练记录重建 5000、10000 步时各会话的参数，写成与训练快照相同格式的文件，供早期快照的目标评估使用。

训练记录里每个回合都记下了回合结束后的参数（b_after）。重建规则：取控制步不晚于该时刻的最后一条记录。
先用同样的规则重建 20000、40000、60000 步，与训练时存下的快照逐个会话核对参数哈希，全部一致才写新文件。

用法：C1A2_VARIANT=s2n C1A2_TARGET=<目标> python reference_online_c1bea_early.py
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from reference_online_c1a2 import EARLY_MARKS, MARKS, OUTPUT, VARIANT
from reference_online_targets import HASHES


def reconstruct(records, mark, sessions=60):
    """返回 60 × (Frozen, Updated) 的参数；Frozen 恒为零。"""
    result = np.zeros((sessions, 2, 4))
    latest = {}
    for r in records:
        if r['arm'] != 'Updated' or r['control_step'] > mark: continue
        key = r['control_step'], r['updates']
        if r['session'] not in latest or key >= latest[r['session']][0]: latest[r['session']] = key, r['b_after']
    for session, (_, value) in latest.items(): result[session, 1] = value
    return result


def main():
    import torch
    assert VARIANT == 's2n' and EARLY_MARKS
    train = OUTPUT / 'train'
    records = json.loads((train / 'episodes.json').read_text())['records']
    digest = lambda w: hashlib.sha256(torch.as_tensor(w, dtype=torch.float64).numpy().tobytes()).hexdigest()
    reference = None
    for mark in MARKS[1:]:
        saved = torch.load(train / f'at_{mark}.pt', map_location='cpu', weights_only=True)
        reference = reference or saved
        rebuilt = reconstruct(records, mark)
        for k, head in enumerate(saved['heads']):
            if digest(rebuilt[k // 2, k % 2]) != head['parameter_sha256']: raise ValueError(f'reconstruction differs at {mark}, head {k}')
    print('reconstruction matches the saved snapshots at', MARKS[1:], flush=True)
    for mark in EARLY_MARKS:
        path = train / f'at_{mark}.pt'
        if path.exists(): raise FileExistsError(path)
        rebuilt = reconstruct(records, mark)
        heads = [dict(w=torch.as_tensor(rebuilt[k // 2, k % 2], dtype=torch.float64), updated=bool(k % 2),
                      parameter_sha256=digest(rebuilt[k // 2, k % 2])) for k in range(120)]
        torch.save(dict(heads=heads, control_steps=mark, base_sha256=HASHES['B'], noise_sha256=reference['noise_sha256'],
                        kind='train', reconstructed_from='episodes.json'), path)
        z = np.tanh(rebuilt[:, 1, 2:])
        print(f'wrote {path.name}: mean tanh swing front/rear {z.mean(0).round(3).tolist()}, sessions with both > 0.5: {int(((z > .5).all(1)).sum())}', flush=True)


if __name__ == '__main__': main()
