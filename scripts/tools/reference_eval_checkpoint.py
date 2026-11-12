"""Register a named, hash-pinned PyTorch policy for frozen evaluation."""
from __future__ import annotations

from pathlib import Path
import re

from reference_stairs_eval import MODELS, TRAIN, sha

BUILTINS = ('update4001', 'update6001', 'update8001', 'update10000')
FINAL_SHA = '5cf4c1160b6f5b7b5102844526b26f3886c27562d98ea9558e3fa09da504f5a1'


def register_checkpoint(model, checkpoint=None, expected_sha=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',model):
        raise ValueError('model label must contain only letters, digits, underscores or hyphens')
    if checkpoint is None:
        if model not in BUILTINS or expected_sha is not None:
            raise ValueError('custom policy requires both checkpoint path and SHA256')
        path = TRAIN/'model_updates_10000.pt' if model=='update10000' else TRAIN/'training'/f'model_{int(model[6:])-1}.pt'
        actual = sha(path)
        if model=='update10000' and actual!=FINAL_SHA:
            raise ValueError('original final checkpoint changed')
    else:
        if model in BUILTINS or model=='reference':
            raise ValueError('custom checkpoint cannot reuse a historical model label')
        if expected_sha is None or not re.fullmatch(r'[a-f0-9]{64}',expected_sha):
            raise ValueError('custom checkpoint requires a lowercase SHA256')
        path=Path(checkpoint).resolve()
        actual=sha(path)
        if actual!=expected_sha:
            raise ValueError('evaluation checkpoint hash mismatch')
    MODELS[model]=path
    return actual
