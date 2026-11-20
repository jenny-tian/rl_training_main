"""参数迁移用的快照读取：把在另一个地形上学到的参数交给目标评估。方案见 docs/lite3-online-transfer-20261120/PLAN.md。"""
from __future__ import annotations

import hashlib

import torch


def pooled_heads(heads):
    """所有 Updated 会话（奇数位）换成它们参数的平均值；Frozen 会话（偶数位）不动。"""
    mean = torch.stack([torch.as_tensor(h['w'], dtype=torch.float64) for h in heads[1::2]]).mean(0)
    digest = hashlib.sha256(mean.numpy().tobytes()).hexdigest()
    return [dict(h, w=mean.clone(), parameter_sha256=digest) if k % 2 else h for k, h in enumerate(heads)]


def load_source():
    from reference_online_c1a2 import SOURCE_FILE, SOURCE_MARK, SOURCE_POOLED
    checkpoint = torch.load(SOURCE_FILE, map_location='cpu', weights_only=True)
    if checkpoint['control_steps'] != SOURCE_MARK or checkpoint['kind'] != 'train' or len(checkpoint['heads']) != 120:
        raise ValueError('unexpected source snapshot')
    if any(torch.count_nonzero(torch.as_tensor(h['w'])) for h in checkpoint['heads'][::2]):
        raise ValueError('source Frozen parameters are not zero')
    if SOURCE_POOLED: checkpoint['heads'] = pooled_heads(checkpoint['heads'])
    return checkpoint
