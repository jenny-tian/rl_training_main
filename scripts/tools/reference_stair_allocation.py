"""Audited, one-time environment allocation migration on the unchanged map."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from reference_stairs_eval import ROOT, sha
from reference_low_stairs_resume import STATE_KEY, validate_curriculum
from reference_reward_resume import equal_state

PARENT = ROOT / 'logs/reference_six/low_stairs_zero_preservation_20261115/train_001_100/model_updates_10100.pt'
DIGEST = '290d94746c467b5cd21ef74c47a908011521c1f145da35ba7b94f1b92d9b7497'
OUTPUT = ROOT / 'logs/reference_six/stair_allocation_20261116'
DOCS = ROOT / 'docs/lite3-stair-allocation-20261116'
GROUPS = {'flat': (0, 4), 'rough': (4, 8), 'slope': (8, 12),
          'stairs_up': (12, 14), 'stairs_down': (14, 16), 'friction': (16, 18), 'slope_rough': (18, 20)}


def parent_state():
    if sha(PARENT) != DIGEST:
        raise ValueError('parent checkpoint changed')
    saved = torch.load(PARENT, map_location='cpu', weights_only=True)
    if saved['iter'] != 10099 or saved['infos'][STATE_KEY]['mode'] != 'exact_curriculum':
        raise ValueError('requires the reviewed exact zero10100 checkpoint')
    return saved


def counts(types, levels=None):
    result = {}
    for name, (lo, hi) in GROUPS.items():
        mask = (types >= lo) & (types < hi)
        result[name] = int(mask.sum()) if levels is None else torch.bincount(levels[mask], minlength=10).tolist()
    return result


def allocate(types, levels, variant):
    if variant not in ('A', 'B'):
        raise ValueError('unknown branch')
    native = torch.div(torch.arange(4096), 4096 / 20, rounding_mode='floor').long()
    if not equal_state(types, native) or levels.shape != (4096,) or levels.dtype != torch.long:
        raise ValueError('requires native 4096 environment allocation')
    if (levels < 0).any() or (levels >= 10).any():
        raise ValueError('invalid difficulty levels')
    result = types.clone()
    changes = []
    if variant == 'B':
        for donor, target in (('flat', 'stairs_up'), ('rough', 'stairs_down')):
            lo, hi = GROUPS[donor]; dest, end = GROUPS[target]
            target_hist = torch.bincount(levels[(types >= dest) & (types < end)], minlength=10)
            numerators = target_hist * 205
            quotas = numerators // int(target_hist.sum())
            remainder = numerators % int(target_hist.sum())
            order = sorted(range(10), key=lambda i: (-int(remainder[i]), i))
            for i in order[:205-int(quotas.sum())]:
                quotas[i] += 1
            for level, amount in enumerate(quotas.tolist()):
                available = torch.where((types >= lo) & (types < hi) & (levels == level))[0]
                if len(available) < amount:
                    raise ValueError('insufficient donors in a required difficulty stratum')
                # Spread deterministic selections across all donor columns; no RNG consumption.
                selected = available[(torch.arange(amount) * len(available) // max(amount, 1)).long()]
                for env_id in selected.tolist():
                    # Balance destination columns within this level, then globally, then by column id.
                    col = min(range(dest, end), key=lambda c: (
                        int(((result == c) & (levels == level)).sum()), int((result == c).sum()), c))
                    changes.append(dict(env_id=env_id, level=level, old_column=int(types[env_id]), new_column=col))
                    result[env_id] = col
    expected = dict(flat=820, rough=819, slope=819, stairs_up=410, stairs_down=409, friction=410, slope_rough=409)
    if variant == 'B':
        expected.update(flat=615, rough=614, stairs_up=615, stairs_down=614)
    if counts(result) != expected or len(changes) != (410 if variant == 'B' else 0):
        raise ValueError('allocation count mismatch')
    report = dict(schema=1, variant=variant, parent_sha256=DIGEST,
        counts_before=counts(types), counts_after=counts(result), changes=changes,
        levels_before=counts(types, levels), levels_after=counts(result, levels),
        global_levels_unchanged=True, map_columns_unchanged=True,
        selection='largest remainder target-stair level quotas; evenly spaced donor IDs; balanced destination columns')
    return result, report


def allocation_contract(original_contract, report):
    value = dict(original_terrain_contract=original_contract, allocation=report)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def migrate(base, saved, original_contract, variant):
    """Validate the unmodified parent before authorizing precisely three state changes.

    The physical map, config, and native later curriculum remain unchanged.
    Runtime assignment happens before the runner's first reset; the new contract
    explicitly includes the complete transfer list, never masquerading as the old one.
    """
    if base.common_step_counter != 0:
        raise ValueError('migration requires an unstepped environment')
    parent = parent_state()
    if not equal_state(saved, parent):
        raise ValueError('migration input is not the reviewed parent')
    state = saved['infos'][STATE_KEY]
    validate_curriculum(base, state, original_contract)
    types, report = allocate(state['terrain_types'], state['terrain_levels'], variant)
    contract = allocation_contract(original_contract, report)
    migrated = copy.deepcopy(saved)
    target = migrated['infos'][STATE_KEY]
    target.update(contract=contract, terrain_types=types,
                  env_origins=state['terrain_origins'][state['terrain_levels'], types].clone())
    allowed = {'contract', 'terrain_types', 'env_origins'}
    if any(not equal_state(state[k], target[k]) for k in state if k not in allowed):
        raise ValueError('migration touched unrelated curriculum fields')
    if not all(equal_state(saved[k], migrated[k]) for k in saved if k != 'infos'):
        raise ValueError('migration touched algorithm state')
    base.scene.terrain.terrain_types.copy_(types.to(base.device))
    validate_curriculum(base, target, contract)
    migrated['infos']['stair_allocation_v1'] = copy.deepcopy(report)
    report.update(original_contract=original_contract, contract=contract,
                  permitted_state_fields=sorted(allowed), algorithm_exact=True,
                  physics_and_rng_restored=False)
    return migrated, target, contract, report
