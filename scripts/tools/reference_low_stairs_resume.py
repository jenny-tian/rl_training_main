"""Restore curriculum at a fresh episode boundary without editing pinned sources.

Legacy checkpoints have only a global level histogram: their per-environment
assignment cannot be recovered. New checkpoints embed exact curriculum tensors
and the cached phase reference. Neither mode restores physics or rollout/RNG
state, so neither claims bitwise continuation of an interrupted episode.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math

import torch

from reference_reward_resume import equal_state

STATE_KEY = 'low_stairs_curriculum_v1'


def native_rewards():
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards
    return rewards


def terrain_contract(cfg):
    from reference_reward_variants import plain
    value = plain(cfg)
    selected = dict(seed=value['seed'], num_envs=value['scene']['num_envs'],
                    terrain=value['scene']['terrain'], curriculum=value['curriculum'])
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def legacy_contract():
    """Reconstruct the pinned parent's terrain configuration from its audits."""
    from reference_low_stairs_evaluate import TRAINING
    env = json.loads((TRAINING/'configuration_audit.json').read_text())['candidate_environment']
    terrain = copy.deepcopy(env['scene']['terrain'])
    terrain['max_init_terrain_level'] = 0
    for name in ('stairs_up', 'stairs_down'):
        terrain['terrain_generator']['sub_terrains'][name]['step_height_range'] = [.02, .05]
    selected = dict(seed=42, num_envs=4096, terrain=terrain, curriculum=env['curriculum'])
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def _expected_scale(levels):
    rewards = native_rewards()
    previous = rewards.gait_level
    try:
        return rewards.update_gait_level_from_terrain_mean(levels.float().mean())
    finally:
        rewards.gait_level = previous


def capture_curriculum(base, contract):
    terrain = base.scene.terrain
    levels = terrain.terrain_levels.detach().cpu().clone()
    scale = float(native_rewards().gait_level)
    if not math.isfinite(scale) or abs(scale - _expected_scale(levels)) > 1e-7:
        raise ValueError('reward multiplier is inconsistent with terrain levels')
    phase = getattr(base, 'phase_foot_ref_body', None)
    state = dict(schema=1, mode='exact_curriculum', contract=contract,
        terrain_levels=levels, terrain_types=terrain.terrain_types.detach().cpu().clone(),
        terrain_origins=terrain.terrain_origins.detach().cpu().clone(),
        env_origins=terrain.env_origins.detach().cpu().clone(), gait_level=scale,
        phase_reference=None if phase is None else phase.detach().cpu().clone(),
        physics_and_rng_restored=False)
    validate_curriculum(base, state, contract)
    return state


def legacy_histogram_state(base, contract, *, assignment_seed=42):
    """Explicit migration of the pinned 10000-update parent, never a fallback."""
    from reference_low_stairs_evaluate import EXPECTED_SHA, TRAINING
    from reference_low_stairs_trial import source_check
    from reference_stairs_eval import sha
    source_check()
    if contract != legacy_contract():
        raise ValueError('legacy migration requires the reviewed low-stair terrain configuration')
    path = TRAINING / 'report.json'
    report = json.loads(path.read_text())
    if report['output_sha256'] != EXPECTED_SHA or report['completed_updates'] != 10000:
        raise ValueError('legacy histogram is not linked to the reviewed checkpoint')
    counts = torch.tensor(report['final_terrain_level_histogram'], dtype=torch.long)
    terrain = base.scene.terrain
    if counts.shape != (terrain.terrain_origins.shape[0],) or (counts < 0).any():
        raise ValueError('invalid legacy level counts')
    if int(counts.sum()) != base.num_envs:
        raise ValueError('legacy histogram requires the original environment count')
    levels = torch.repeat_interleave(torch.arange(len(counts)), counts)
    # A local CPU generator does not perturb the simulator's random stream.
    levels = levels[torch.randperm(len(levels), generator=torch.Generator().manual_seed(assignment_seed))]
    mean = float(levels.float().mean())
    if abs(mean - report['updates'][-1]['mean_terrain_level']) > 1e-6:
        raise ValueError('histogram disagrees with final recorded terrain mean')
    types = terrain.terrain_types.detach().cpu().clone()
    origins = terrain.terrain_origins.detach().cpu().clone()
    state = dict(schema=1, mode='legacy_global_histogram', contract=contract,
        terrain_levels=levels, terrain_types=types, terrain_origins=origins,
        env_origins=origins[levels, types].clone(), gait_level=_expected_scale(levels),
        phase_reference=None, physics_and_rng_restored=False,
        provenance=dict(checkpoint_sha256=EXPECTED_SHA, report_sha256=sha(path),
            assignment_seed=assignment_seed, global_histogram_exact=True,
            per_environment_assignment_reconstructed=True,
            per_terrain_level_distribution_known=False, phase_reference_reinitialized=True))
    validate_curriculum(base, state, contract)
    return state


def validate_curriculum(base, state, contract):
    if state.get('schema') != 1 or state.get('mode') not in ('exact_curriculum', 'legacy_global_histogram'):
        raise ValueError('unsupported curriculum snapshot')
    if state.get('contract') != contract:
        raise ValueError('terrain configuration differs from curriculum snapshot')
    terrain = base.scene.terrain
    levels, types = state['terrain_levels'], state['terrain_types']
    rows, columns = terrain.terrain_origins.shape[:2]
    for name, tensor, limit in [('levels', levels, rows), ('types', types, columns)]:
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.long
                or tensor.shape != (base.num_envs,) or (tensor < 0).any() or (tensor >= limit).any()):
            raise ValueError(f'invalid terrain {name}')
    if not equal_state(types, terrain.terrain_types):
        raise ValueError('native terrain column allocation differs')
    if not equal_state(state['terrain_origins'], terrain.terrain_origins):
        raise ValueError('generated terrain origins differ')
    expected = state['terrain_origins'][levels, types]
    if not equal_state(state['env_origins'], expected):
        raise ValueError('environment origins disagree with saved levels and columns')
    if not torch.isfinite(expected).all():
        raise ValueError('nonfinite environment origins')
    scale = state['gait_level']
    if not math.isfinite(scale) or abs(scale - _expected_scale(levels)) > 1e-7:
        raise ValueError('saved reward multiplier disagrees with terrain levels')
    phase = state['phase_reference']
    if phase is not None and (not isinstance(phase, torch.Tensor) or phase.shape != (1, 4, 3)
                              or not torch.isfinite(phase).all()):
        raise ValueError('invalid phase reference')
    if set(base.curriculum_manager.active_terms) != {'terrain_levels', 'gait_level'}:
        raise ValueError('unsupported extra or missing curriculum terms')


@contextmanager
def restored_initial_reset(base, state, contract):
    """Wrap exactly one fresh full reset, including RslRlVecEnvWrapper's reset.

    A startup reset is not an episode result and must not promote/demote levels.
    Subsequent native episode resets retain their original curriculum behavior.
    """
    if base.common_step_counter != 0 or getattr(base, '_resume_initial_reset_done', False):
        raise ValueError('curriculum restore requires a fresh, unstepped environment')
    validate_curriculum(base, state, contract)  # reject before mutating the env
    terrain, manager = base.scene.terrain, base.curriculum_manager
    terrain.terrain_levels.copy_(state['terrain_levels'].to(base.device))
    terrain.env_origins.copy_(state['env_origins'].to(base.device))
    native_rewards().gait_level = float(state['gait_level'])
    if state['phase_reference'] is not None:
        base.phase_foot_ref_body = state['phase_reference'].to(base.device).clone()
    elif hasattr(base, 'phase_foot_ref_body'):
        del base.phase_foot_ref_body
    for name in ('_low_stairs_sample', '_low_stairs_wall_sample'):
        if hasattr(base, name):
            delattr(base, name)
    calls = 0
    original_compute = manager.compute
    previous_override = manager.__dict__.get('compute')

    def initial_compute(env_ids=None):
        nonlocal calls
        ids = torch.arange(base.num_envs) if env_ids is None else torch.as_tensor(env_ids).cpu()
        if calls or not torch.equal(ids.sort().values, torch.arange(base.num_envs)):
            raise ValueError('resume initialization requires exactly one full reset')
        calls += 1
        manager._curriculum_state['terrain_levels'] = terrain.terrain_levels.float().mean()
        manager._curriculum_state['gait_level'] = torch.tensor(state['gait_level'], device=base.device)

    manager.compute = initial_compute
    try:
        yield
        if calls != 1:
            raise ValueError('the restored environment was not reset')
        if not equal_state(terrain.terrain_levels, state['terrain_levels']):
            raise ValueError('startup reset changed restored terrain levels')
        if not equal_state(terrain.env_origins, state['env_origins']):
            raise ValueError('startup reset changed restored origins')
        if float(native_rewards().gait_level) != state['gait_level']:
            raise ValueError('startup reset changed restored reward multiplier')
        base._resume_initial_reset_done = True
    finally:
        if previous_override is None:
            del manager.compute
        else:
            manager.compute = original_compute


def restore_algorithm(runner, saved):
    rates = {group['lr'] for group in saved['optimizer_state_dict']['param_groups']}
    if len(rates) != 1 or not all(math.isfinite(r) and r > 0 for r in rates):
        raise ValueError('invalid checkpoint learning rate')
    runner.alg.load(saved, load_cfg=None, strict=True)
    runner.alg.learning_rate = rates.pop()
    runner.current_learning_iteration = saved['iter'] + 1
    actual = runner.alg.save()
    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
        if not equal_state(saved[key], actual[key]):
            raise ValueError(f'inexact native restore: {key}')
    return dict(actor_exact=True, critic_exact=True, optimizer_exact=True,
                learning_rate=runner.alg.learning_rate, next_iteration=runner.current_learning_iteration)


def attach_curriculum_saving(runner, base, contract):
    """Include state in every native save, including periodic saves without infos."""
    if getattr(runner, '_curriculum_save_attached', False):
        raise ValueError('curriculum checkpoint hook already installed')
    native_save = runner.save

    def save(path, infos=None):
        metadata = copy.deepcopy(infos) if infos is not None else {}
        if STATE_KEY in metadata:
            raise ValueError('caller cannot override captured curriculum state')
        metadata[STATE_KEY] = capture_curriculum(base, contract)
        native_save(path, infos=metadata)

    runner.save = save
    runner._curriculum_save_attached = True


def validate_checkpoint_pair(saved, curriculum):
    """Reject accidental combinations of one model and another saved curriculum."""
    if curriculum.get('mode') == 'legacy_global_histogram':
        from reference_low_stairs_trial import checked_state
        parent, _ = checked_state()
        for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter', 'infos'):
            if not equal_state(parent[key], saved[key]):
                raise ValueError('legacy histogram requires the reviewed parent checkpoint')
    elif not equal_state((saved.get('infos') or {}).get(STATE_KEY), curriculum):
        raise ValueError('curriculum snapshot is not embedded in the supplied checkpoint')


def make_resumed_runner(base, agent_cfg, saved, curriculum, contract, *, log_dir=None):
    """Single integration path used by validation and future approved trainers."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    validate_checkpoint_pair(saved, curriculum)
    with restored_initial_reset(base, curriculum, contract):
        env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.get('clip_actions'))
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=base.device)
    audit = restore_algorithm(runner, saved)
    attach_curriculum_saving(runner, base, contract)
    audit.update(curriculum_mode=curriculum['mode'], initial_gait_level=native_rewards().gait_level,
                 initial_mean_terrain_level=float(base.scene.terrain.terrain_levels.float().mean()),
                 first_reset_preserved_levels=True, physics_and_rng_restored=False)
    return runner, env, audit
