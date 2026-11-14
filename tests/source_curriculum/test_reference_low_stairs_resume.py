"""Resume invariants: no synthetic startup promotion or silent partial restore."""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/tools'))
import reference_low_stairs_resume as resume
from reference_reward_code_audit import definitions, REWARD


@pytest.fixture
def native(monkeypatch):
    namespace = dict(torch=torch, math=math, gait_level=1.)
    definitions(REWARD, ['update_gait_level_from_terrain_mean'], namespace)

    class Rewards:
        @property
        def gait_level(self):
            return namespace['gait_level']

        @gait_level.setter
        def gait_level(self, value):
            namespace['gait_level'] = value

        update_gait_level_from_terrain_mean = staticmethod(namespace['update_gait_level_from_terrain_mean'])

    module = Rewards()
    monkeypatch.setattr(resume, 'native_rewards', lambda: module)
    return module


class Curriculum:
    active_terms = ['terrain_levels', 'gait_level']

    def __init__(self, terrain, native):
        self._curriculum_state = {}
        self.terrain, self.native = terrain, native
        self.calls = 0

    def compute(self, env_ids=None):
        self.calls += 1
        ids = slice(None) if env_ids is None else env_ids
        self.terrain.terrain_levels[ids] += 1
        self.native.update_gait_level_from_terrain_mean(self.terrain.terrain_levels.float().mean())


def environment(native, n=4):
    origins = torch.arange(10*2*3, dtype=torch.float32).reshape(10, 2, 3)
    levels, types = torch.full((n,), 5), torch.arange(n) % 2
    terrain = NS(terrain_levels=levels, terrain_types=types, terrain_origins=origins,
                 env_origins=origins[levels, types].clone())
    return NS(scene=NS(terrain=terrain), curriculum_manager=Curriculum(terrain, native),
              num_envs=n, device='cpu', common_step_counter=0,
              phase_foot_ref_body=torch.arange(12, dtype=torch.float32).reshape(1, 4, 3))


def test_initial_reset_preserves_levels_origins_scale_and_native_later_resets(native):
    base = environment(native)
    saved = resume.capture_curriculum(base, 'contract')
    base.scene.terrain.terrain_levels.zero_()
    base.scene.terrain.env_origins.zero_()
    native.gait_level = .12
    base.phase_foot_ref_body.zero_()
    with resume.restored_initial_reset(base, saved, 'contract'):
        base.curriculum_manager.compute(torch.arange(4))
    assert torch.equal(base.scene.terrain.terrain_levels, saved['terrain_levels'])
    assert torch.equal(base.scene.terrain.env_origins, saved['env_origins'])
    assert torch.equal(base.phase_foot_ref_body, saved['phase_reference'])
    assert native.gait_level == 1.
    assert base.curriculum_manager.calls == 0
    assert base.curriculum_manager._curriculum_state['gait_level'] == 1.
    base.curriculum_manager.compute(torch.tensor([0]))
    assert base.curriculum_manager.calls == 1 and base.scene.terrain.terrain_levels.tolist() == [6, 5, 5, 5]


@pytest.mark.parametrize('damage', ['contract', 'shape', 'dtype', 'negative', 'overflow', 'types',
                                  'origins', 'mesh_origins', 'scale', 'nan', 'phase', 'schema'])
def test_bad_snapshot_rejected_before_environment_mutation(native, damage):
    base = environment(native)
    saved = resume.capture_curriculum(base, 'contract')
    if damage == 'contract': saved['contract'] = 'wrong'
    if damage == 'shape': saved['terrain_levels'] = saved['terrain_levels'][:3]
    if damage == 'dtype': saved['terrain_levels'] = saved['terrain_levels'].float()
    if damage == 'negative': saved['terrain_levels'][0] = -1
    if damage == 'overflow': saved['terrain_levels'][0] = 10
    if damage == 'types': saved['terrain_types'][0] = 1
    if damage == 'origins': saved['env_origins'][0, 0] += 1
    if damage == 'mesh_origins': saved['terrain_origins'][0, 0, 0] += 1
    if damage == 'scale': saved['gait_level'] = .1
    if damage == 'nan': saved['gait_level'] = float('nan')
    if damage == 'phase': saved['phase_reference'][0, 0, 0] = float('nan')
    if damage == 'schema': saved['schema'] = 0
    before = copy.deepcopy(base.scene.terrain)
    with pytest.raises(ValueError):
        with resume.restored_initial_reset(base, saved, 'contract'):
            pytest.fail('invalid snapshot reached initialization')
    assert torch.equal(base.scene.terrain.terrain_levels, before.terrain_levels)
    assert torch.equal(base.scene.terrain.env_origins, before.env_origins)
    assert native.gait_level == 1.


@pytest.mark.parametrize('action', ['missing_reset', 'double_reset', 'partial_reset', 'exception'])
def test_reset_guard_restores_native_method_on_failure(native, action):
    base = environment(native)
    saved = resume.capture_curriculum(base, 'contract')
    with pytest.raises((ValueError, RuntimeError)):
        with resume.restored_initial_reset(base, saved, 'contract'):
            if action == 'double_reset':
                base.curriculum_manager.compute()
                base.curriculum_manager.compute()
            if action == 'partial_reset': base.curriculum_manager.compute(torch.tensor([0]))
            if action == 'exception': raise RuntimeError('simulated startup error')
    assert 'compute' not in base.curriculum_manager.__dict__
    base.curriculum_manager.compute()
    assert base.curriculum_manager.calls == 1


def test_late_restore_and_unknown_curriculum_fail_closed(native):
    base = environment(native)
    saved = resume.capture_curriculum(base, 'contract')
    base.common_step_counter = 1
    with pytest.raises(ValueError, match='fresh'):
        with resume.restored_initial_reset(base, saved, 'contract'): pass
    base.common_step_counter = 0
    base.curriculum_manager.active_terms = ['terrain_levels', 'gait_level', 'commands']
    with pytest.raises(ValueError, match='curriculum terms'):
        resume.validate_curriculum(base, saved, 'contract')


def test_capture_uses_real_mapping_and_rejects_stale_reward_scale(native):
    base = environment(native)
    base.scene.terrain.terrain_levels.fill_(1)
    with pytest.raises(ValueError, match='inconsistent'):
        resume.capture_curriculum(base, 'contract')
    native.update_gait_level_from_terrain_mean(1.)
    base.scene.terrain.env_origins[:] = base.scene.terrain.terrain_origins[base.scene.terrain.terrain_levels,
                                                                        base.scene.terrain.terrain_types]
    saved = resume.capture_curriculum(base, 'contract')
    assert saved['gait_level'] == pytest.approx(math.exp(-2))
    assert native.gait_level == saved['gait_level']


def test_periodic_and_explicit_native_saves_embed_independent_state(native):
    base = environment(native)
    calls = []
    runner = NS(save=lambda path, infos=None: calls.append((path, infos)))
    resume.attach_curriculum_saving(runner, base, 'contract')
    infos = dict(validation_only=True)
    runner.save('explicit', infos=infos)
    runner.save('periodic')
    assert infos == {'validation_only': True}
    assert all(resume.STATE_KEY in metadata for _, metadata in calls)
    base.scene.terrain.terrain_levels[0] = 6
    assert calls[0][1][resume.STATE_KEY]['terrain_levels'][0] == 5
    with pytest.raises(ValueError, match='already installed'):
        resume.attach_curriculum_saving(runner, base, 'contract')
    with pytest.raises(ValueError, match='override'):
        runner.save('invalid', infos={resume.STATE_KEY: {}})


def test_model_and_snapshot_cannot_be_mixed(native):
    state = resume.capture_curriculum(environment(native), 'contract')
    with pytest.raises(ValueError, match='not embedded'):
        resume.validate_checkpoint_pair(dict(infos=None), state)
    saved = dict(infos={resume.STATE_KEY: copy.deepcopy(state)})
    resume.validate_checkpoint_pair(saved, state)
    saved['infos'][resume.STATE_KEY]['terrain_levels'][0] = 6
    with pytest.raises(ValueError, match='not embedded'):
        resume.validate_checkpoint_pair(saved, state)


def test_legacy_histogram_is_explicit_exact_counts_but_reconstructed_assignment(native):
    base = environment(native, 4096)
    # The real parent has 20 columns; use its native allocation with synthetic geometry.
    terrain = base.scene.terrain
    terrain.terrain_origins = torch.arange(10*20*3, dtype=torch.float32).reshape(10, 20, 3)
    terrain.terrain_types = (torch.arange(4096)/(4096/20)).floor().long()
    before = torch.random.get_rng_state().clone()
    a = resume.legacy_histogram_state(base, resume.legacy_contract())
    b = resume.legacy_histogram_state(base, resume.legacy_contract())
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(a['terrain_levels'], b['terrain_levels'])
    assert torch.bincount(a['terrain_levels'], minlength=10).tolist() == [113, 163, 250, 310, 386, 452, 570, 605, 666, 581]
    assert a['gait_level'] == 1. and a['phase_reference'] is None
    assert a['provenance']['per_environment_assignment_reconstructed']
    assert not a['provenance']['per_terrain_level_distribution_known']
    with pytest.raises(ValueError, match='reviewed low-stair terrain configuration'):
        resume.legacy_histogram_state(base, 'different heights or seed')


@pytest.mark.parametrize('args', [['--mode', 'train'], ['--mode', 'capture', '--gpu', '0'],
                                 ['--mode', 'reload'], ['--mode', 'capture', '--checkpoint', 'unknown.pt']])
def test_validation_cannot_start_training_or_accept_implicit_checkpoint(args, monkeypatch, tmp_path):
    from reference_low_stairs_resume_validate import main
    output = tmp_path/'unstarted'
    monkeypatch.setattr(sys, 'argv', ['validate.py', *args, '--output', str(output)])
    with pytest.raises(SystemExit) as error: main()
    assert error.value.code == 2 and not output.exists()
