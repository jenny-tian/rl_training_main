"""Two-process curriculum restore validation; no PPO updates or training CLI."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import os
from pathlib import Path
import sys
import traceback

from reference_lite3_loader import ROOT, load_reference, resolve_lite3_asset
from reference_stairs_eval import save, sha
from reference_low_stairs_resume import (STATE_KEY, capture_curriculum,
    legacy_histogram_state, make_resumed_runner, native_rewards, terrain_contract)


def collect(args):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from reference_low_stairs_trial import CHECKPOINT, checked_state, source_check
    source = source_check()
    parent, _ = checked_state()
    reference, agent = load_reference()
    from reference_low_stairs_cfg import apply_terrain, apply_rewards
    from reference_six_terrain_cfg import replace_terrain, column_names
    from reference_reward_variants import plain
    from reference_reward_resume import equal_state

    cfg, _ = apply_terrain(replace_terrain(reference))
    cfg, _ = apply_rewards(cfg)
    cfg.seed = agent.seed = 42
    cfg.scene.num_envs = 4096
    cfg.sim.device = agent.device = 'cuda:0'
    resolve_lite3_asset(cfg)
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    contract = terrain_contract(cfg)
    save(args.output / 'configuration.json', dict(environment=plain(cfg), agent=plain(agent), contract=contract))
    base = ManagerBasedRLEnv(cfg=cfg)
    if args.mode == 'capture':
        saved = parent
        curriculum = legacy_histogram_state(base, contract)
    else:
        if sha(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError('validation checkpoint identity differs')
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if saved['infos'].get('validation_only') is not True or STATE_KEY not in saved['infos']:
            raise ValueError('reload requires a validation checkpoint with curriculum state')
        for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
            if not equal_state(parent[key], saved[key]):
                raise ValueError('validation checkpoint contains an optimizer or policy update')
        curriculum = saved['infos'][STATE_KEY]
    runner, env, audit = make_resumed_runner(base, agent.to_dict(), saved, curriculum, contract)
    # Normally learn() initializes this before a save. This validation never
    # calls learn(); with log_dir=None the native initializer creates no writer.
    runner.logger.init_logging_writer()
    initial = capture_curriculum(base, contract)
    for key in ('terrain_levels', 'terrain_types', 'env_origins', 'terrain_origins', 'gait_level', 'phase_reference'):
        assert equal_state(initial[key], curriculum[key]), key
    # Every wrapper startup log must describe restored state, not the initial row.
    assert base.extras['log']['Curriculum/gait_level'] == curriculum['gait_level'] == 1.
    assert abs(base.extras['log']['Curriculum/terrain_levels'] - audit['initial_mean_terrain_level']) < 1e-6
    initial_histogram = torch.bincount(initial['terrain_levels'], minlength=10).tolist()
    phase_before = initial['phase_reference']
    before = copy.deepcopy(runner.alg.save())
    saved_parent_hash = sha(CHECKPOINT)
    policy = runner.get_inference_policy()
    obs = env.get_observations()
    scaled = ('base_height_l2', 'joint_torques_l2', 'joint_power', 'joint_mirror', 'action_rate_l2',
              'contact_forces', 'foot_impact_velocity', 'feet_air_time_lin_xy',
              'feet_air_time_ang_z_lite3', 'phase_foot_trajectory_exp')
    coefficients = []
    resets = 0
    for step in range(64):
        with torch.inference_mode():
            obs, rewards, done, _ = env.step(policy(obs))
        assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(rewards).all()
        assert obs['policy'].shape == (4096, 45) and obs['critic'].shape == (4096, 393)
        resets += int(done.sum())
        coefficients.append(float(native_rewards().gait_level))
    assert min(coefficients) == max(coefficients) == 1.
    # Hold the same physical state and compare effective multipliers directly.
    term_checks = []
    original_scale = native_rewards().gait_level
    try:
        for name in scaled:
            term = base.reward_manager.get_term_cfg(name)
            native_rewards().gait_level = 1.
            unit = term.func(base, **term.params).detach().clone() * term.weight
            native_rewards().gait_level = .13410544
            reduced = term.func(base, **term.params).detach().clone() * term.weight
            error = float((reduced - unit * .13410544).abs().max())
            assert torch.allclose(reduced, unit * .13410544, rtol=1e-5, atol=1e-6), name
            term_checks.append(dict(term=name, configured_weight=term.weight,
                restored_effective_weight=term.weight, old_initial_effective_weight=term.weight*.13410544,
                nonzero_unit_samples=int((unit.abs() > 1e-8).sum()), proportional_error=error))
    finally:
        native_rewards().gait_level = original_scale
    # Actual native reset progression after startup: one promotion and demotion.
    terrain, robot = base.scene.terrain, base.scene['robot']
    middle = torch.nonzero((terrain.terrain_levels > 0) & (terrain.terrain_levels < 9)).flatten()
    ids = middle[:2]
    old_levels = terrain.terrain_levels[ids].clone()
    poses = robot.data.root_link_pose_w[ids].clone()
    poses[:, :3] = terrain.env_origins[ids]
    poses[:, 2] += .32
    poses[0, 0] += cfg.scene.terrain.terrain_generator.size[0] / 2 + 1.
    robot.write_root_pose_to_sim(poses, env_ids=ids)
    commands = base.command_manager.get_term('base_velocity')
    commands.vel_command_b[ids] = commands.vel_command_b.new_tensor([.4, 0., 0.])
    base.reset(env_ids=ids)
    expected = old_levels + old_levels.new_tensor([1, -1])
    assert torch.equal(terrain.terrain_levels[ids], expected)
    assert native_rewards().gait_level == 1.
    assert equal_state(before, runner.alg.save()) and sha(CHECKPOINT) == saved_parent_hash
    if phase_before is not None:
        assert equal_state(phase_before, base.phase_foot_ref_body)
    # Save with the existing parent iteration: zero new optimizer updates.
    checkpoint = args.output / 'validation_checkpoint.pt'
    next_iteration = runner.current_learning_iteration
    try:
        runner.current_learning_iteration = saved['iter']
        runner.save(str(checkpoint), infos=dict(validation_only=True, optimizer_updates=0,
                    parent_sha256=saved_parent_hash, completed_updates=10000))
    finally:
        runner.current_learning_iteration = next_iteration
    output = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert equal_state(output['infos'][STATE_KEY], capture_curriculum(base, contract))
    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
        assert equal_state(output[key], parent[key])
    save(args.output / 'report.json', dict(status='passed', mode=args.mode, **audit,
        environments=4096, steps=64, optimizer_updates=0, native_episode_resets=resets,
        initial_level_histogram=initial_histogram, native_promotion_demotion_verified=True,
        gait_min=min(coefficients), gait_max=max(coefficients), reward_coefficients=term_checks,
        actor_critic_optimizer_unchanged=True, phase_reference_restored=phase_before is not None,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        input_checkpoint=str(CHECKPOINT if args.mode == 'capture' else args.checkpoint),
        input_checkpoint_sha256=saved_parent_hash if args.mode == 'capture' else args.checkpoint_sha256,
        parent_sha256=saved_parent_hash, terrain_columns=column_names(cfg),
        legacy_provenance=curriculum.get('provenance'), source_check=source,
        sources=[dict(path=str(ROOT/'scripts/tools'/name), sha256=sha(ROOT/'scripts/tools'/name))
                 for name in ('reference_low_stairs_resume.py', 'reference_low_stairs_resume_validate.py')],
        note='Fresh episodes; exact new curriculum state, but no restoration of physics/RNG or historical legacy per-env levels. No behavioral improvement claim.'))
    print('CURRICULUM RESTORE VALIDATION PASSED; zero optimizer updates', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('capture', 'reload'), required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--gpu', type=int, choices=(1,), default=1)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    if (args.mode == 'reload') != bool(args.checkpoint and args.checkpoint_sha256):
        p.error('reload requires a checkpoint and its sha256; capture takes neither')
    if args.mode == 'capture' and (args.checkpoint or args.checkpoint_sha256):
        p.error('capture uses only the pinned parent')
    args.output = args.output.resolve()
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=1, output=args.output, timeout=1800)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device='cuda:0',
            kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        import torch
        # Isaac lazily caches sensor/asset tensors during inference. Keep the
        # explicit reset writes in the same context as the frozen rollout.
        with torch.inference_mode():
            collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc()
        save(args.output/'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
