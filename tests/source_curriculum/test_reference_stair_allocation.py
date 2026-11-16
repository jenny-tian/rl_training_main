import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/tools'))
import reference_stair_allocation as allocation
import reference_low_stairs_resume as resume
from reference_low_stairs_resume import STATE_KEY
from reference_reward_resume import equal_state


def test_reviewed_checkpoint_allocation_preserves_levels_and_has_exact_donors():
    saved = allocation.parent_state()
    state = saved['infos'][STATE_KEY]
    original = copy.deepcopy(state)
    a, audit_a = allocation.allocate(state['terrain_types'], state['terrain_levels'], 'A')
    b, audit_b = allocation.allocate(state['terrain_types'], state['terrain_levels'], 'B')
    b2, audit_b2 = allocation.allocate(state['terrain_types'], state['terrain_levels'], 'B')
    assert equal_state(a, state['terrain_types']) and equal_state(state, original)
    assert equal_state(b, b2) and audit_b == audit_b2
    assert len(audit_a['changes']) == 0 and len(audit_b['changes']) == 410
    assert int(((b >= 12) & (b < 16)).sum()) == 1229
    ids = [r['env_id'] for r in audit_b['changes']]
    assert len(set(ids)) == 410
    for r in audit_b['changes']:
        assert (r['old_column'] < 4 and r['new_column'] in (12, 13)) or (4 <= r['old_column'] < 8 and r['new_column'] in (14, 15))
        assert r['level'] == int(state['terrain_levels'][r['env_id']])
    for name in ('stairs_up', 'stairs_down'):
        before = torch.tensor(audit_b['levels_before'][name])
        addition = torch.tensor(audit_b['levels_after'][name]) - before
        assert int(addition.sum()) == 205
        assert ((addition.double() - before.double() * 205 / before.sum()).abs() < 1).all()
    assert allocation.allocation_contract(state['contract'], audit_a) != allocation.allocation_contract(state['contract'], audit_b)


@pytest.mark.parametrize('damage', ['variant', 'shape', 'types', 'levels', 'dtype'])
def test_bad_allocation_rejected(damage):
    state = allocation.parent_state()['infos'][STATE_KEY]
    types, levels, variant = state['terrain_types'].clone(), state['terrain_levels'].clone(), 'B'
    if damage == 'variant': variant = 'C'
    if damage == 'shape': levels = levels[:100]
    if damage == 'types': types[0] = 1
    if damage == 'levels': levels[0] = 10
    if damage == 'dtype': levels = levels.float()
    with pytest.raises(ValueError): allocation.allocate(types, levels, variant)


@pytest.mark.parametrize('variant', ['A', 'B'])
def test_migration_restores_then_captures_same_branch_and_rejects_mixed_parent(monkeypatch, variant):
    saved = allocation.parent_state()
    original = copy.deepcopy(saved)
    state = saved['infos'][STATE_KEY]
    rewards = NS(gait_level=1., update_gait_level_from_terrain_mean=lambda mean: 1.)
    monkeypatch.setattr(resume, 'native_rewards', lambda: rewards)
    terrain = NS(terrain_levels=torch.zeros(4096, dtype=torch.long),
        terrain_types=state['terrain_types'].clone(), terrain_origins=state['terrain_origins'].clone(),
        env_origins=torch.zeros(4096, 3))
    manager = NS(active_terms=['terrain_levels', 'gait_level'], _curriculum_state={}, compute=lambda ids=None: None)
    base = NS(scene=NS(terrain=terrain), curriculum_manager=manager, num_envs=4096, device='cpu', common_step_counter=0)
    bad = copy.deepcopy(saved); bad['iter'] -= 1
    with pytest.raises(ValueError, match='reviewed parent'):
        allocation.migrate(base, bad, state['contract'], variant)
    migrated, target, contract, audit = allocation.migrate(base, saved, state['contract'], variant)
    assert equal_state(saved, original)
    resume.validate_checkpoint_pair(migrated, target)
    with resume.restored_initial_reset(base, target, contract): manager.compute()
    assert equal_state(resume.capture_curriculum(base, contract), target)
    assert all(equal_state(state[k], target[k]) for k in ('terrain_levels', 'terrain_origins', 'phase_reference', 'gait_level'))
    assert all(equal_state(saved[k], migrated[k]) for k in saved if k != 'infos')
    wrong = copy.deepcopy(target); wrong['contract'] = state['contract']
    with pytest.raises(ValueError): resume.validate_curriculum(base, wrong, contract)
