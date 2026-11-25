"""参数迁移用的快照读取：把在另一个地形上学到的参数交给目标评估。方案见 docs/lite3-online-transfer-20261120/PLAN.md。"""
from __future__ import annotations

import hashlib

import torch


def pooled_heads(heads):
    """所有 Updated 会话（奇数位）换成它们参数的平均值；Frozen 会话（偶数位）不动。"""
    mean = torch.stack([torch.as_tensor(h['w'], dtype=torch.float64) for h in heads[1::2]]).mean(0)
    digest = hashlib.sha256(mean.numpy().tobytes()).hexdigest()
    return [dict(h, w=mean.clone(), parameter_sha256=digest) if k % 2 else h for k, h in enumerate(heads)]


def library(sources=None):
    """参数库：各来源 60 个 Updated 会话的平均参数，加一组全 0。默认是原来的三组来源，返回 (4, 4)。"""
    from reference_online_c1a2 import LIBRARY_SOURCES, ROOT, RUNS
    rows = []
    for variant, target, mark in sources or LIBRARY_SOURCES:
        checkpoint = torch.load(ROOT / 'logs/reference_six' / RUNS[variant] / target / 'train' / f'at_{mark}.pt', map_location='cpu', weights_only=True)
        if checkpoint['control_steps'] != mark or checkpoint['kind'] != 'train' or len(checkpoint['heads']) != 120:
            raise ValueError('unexpected library snapshot')
        rows.append(torch.stack([torch.as_tensor(h['w'], dtype=torch.float64) for h in checkpoint['heads'][1::2]]).mean(0))
    return torch.stack(rows + [torch.zeros(4, dtype=torch.float64)])


def load_source():
    from reference_online_c1a2 import SOURCE_FILE, SOURCE_MARK, SOURCE_POOLED
    checkpoint = torch.load(SOURCE_FILE, map_location='cpu', weights_only=True)
    if checkpoint['control_steps'] != SOURCE_MARK or checkpoint['kind'] != 'train' or len(checkpoint['heads']) != 120:
        raise ValueError('unexpected source snapshot')
    if any(torch.count_nonzero(torch.as_tensor(h['w'])) for h in checkpoint['heads'][::2]):
        raise ValueError('source Frozen parameters are not zero')
    if SOURCE_POOLED: checkpoint['heads'] = pooled_heads(checkpoint['heads'])
    return checkpoint


def library_heads(entry):
    """120 个会话位，Frozen（偶数位）是全 0，Updated（奇数位）都是库里的一组参数。"""
    w, zero = library()[entry], torch.zeros(4, dtype=torch.float64)
    head = lambda x: dict(w=x.clone(), parameter_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
    return [head(w if k % 2 else zero) for k in range(120)]


def zero_heads():
    """120 个会话位，两组都是全 0。"""
    zero = torch.zeros(4, dtype=torch.float64)
    return [dict(w=zero.clone(), parameter_sha256=hashlib.sha256(zero.numpy().tobytes()).hexdigest()) for _ in range(120)]


def candidate_heads(name):
    """候选参数的不学习评估：120 个会话位，Frozen（偶数位）是全 0，Updated（奇数位）都是名字指定的一组参数。"""
    from rl_training.ea_lpl.c1b_d import candidates
    rows = library()
    w = candidates(rows[int(name[0])], rows[int(name[1])])['mud'.index(name[2])]
    zero = torch.zeros(4, dtype=torch.float64)
    head = lambda x: dict(w=x.clone(), parameter_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
    return [head(w if k % 2 else zero) for k in range(120)]
