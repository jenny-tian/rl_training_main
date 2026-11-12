"""Restore the reviewed final checkpoint without performing an update."""
from __future__ import annotations

import torch

from reference_stairs_eval import TRAIN, sha

FINAL_CHECKPOINT = TRAIN/'model_updates_10000.pt'
FINAL_SHA256 = '5cf4c1160b6f5b7b5102844526b26f3886c27562d98ea9558e3fa09da504f5a1'


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(equal_state(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):
        return type(left)==type(right) and len(left)==len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left==right


def restore_final(runner):
    if sha(FINAL_CHECKPOINT)!=FINAL_SHA256:
        raise ValueError('final checkpoint identity changed')
    saved=torch.load(FINAL_CHECKPOINT,map_location=runner.device,weights_only=True)
    completed=saved['infos']['completed_updates']
    if completed!=10000 or saved['iter']!=9999:
        raise ValueError('unexpected final checkpoint update count')
    optimizer=saved['optimizer_state_dict']
    steps={int(v['step']) for v in optimizer['state'].values()}
    rates={g['lr'] for g in optimizer['param_groups']}
    if steps!={200000} or len(rates)!=1:
        raise ValueError('unexpected final optimizer state')
    runner.alg.load(saved,load_cfg=None,strict=True)
    # Native PPO.load restores Adam groups but not this adaptive-rate scalar.
    runner.alg.learning_rate=next(iter(rates))
    # Checkpoint iter is zero based; the next iteration must start at 10000.
    runner.current_learning_iteration=completed
    restored=runner.alg.save()
    for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):
        if not equal_state(saved[key],restored[key]):
            raise ValueError(f'inexact checkpoint restore: {key}')
    return dict(status='passed',checkpoint=str(FINAL_CHECKPOINT),sha256=FINAL_SHA256,
                completed_updates=completed,next_iteration=completed,saved_iteration=saved['iter'],
                optimizer_steps=200000,learning_rate=runner.alg.learning_rate,
                actor_exact=True,critic_exact=True,optimizer_exact=True,
                environment_state_restored=False,
                note='Weights/Adam/rate restored; terrain/reset/RNG state freshly seeded, not bitwise continuation.')
