"""先零更新集成检查，再分两段执行最多 200 次已批准更新。"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import traceback

from reference_stairs_eval import save, sha
from reference_lite3_loader import ROOT, load_reference, resolve_lite3_asset, source_manifest
from reference_low_stairs_guarded_bank import OUTPUT, BANK
from reference_low_stairs_clearance_train import source_preflight, check_algorithm
from reference_low_stairs_evaluate import EXPECTED_SHA
from reference_low_stairs_trial import CHECKPOINT, checked_state
from reference_low_stairs_resume import (STATE_KEY, legacy_histogram_state, make_resumed_runner,
    terrain_contract, capture_curriculum, native_rewards, restore_algorithm)
from reference_low_stairs_guarded_ppo import Preserver, attach, attach_saving, KEY, PPO_UPDATE_SHA
from reference_reward_resume import equal_state

VARIANT = 'persistent_clearance_yaw_coverage_preservation'


def directory(phase):
    return OUTPUT / ('integration' if phase == 0 else 'train_001_100' if phase == 1 else 'train_101_200')


def sources():
    names = ('reference_low_stairs_guarded_train.py', 'reference_low_stairs_guarded_ppo.py',
             'reference_low_stairs_guarded_cfg.py', 'reference_low_stairs_guarded_bank.py',
             'reference_low_stairs_preservation.py', 'reference_low_stairs_resume.py',
             'reference_low_stairs_repair.py', 'reference_low_stairs_clearance.py',
             'reference_low_stairs_clearance_cfg.py')
    return [dict(path=str(Path(__file__).with_name(name)), sha256=sha(Path(__file__).with_name(name))) for name in names]


def preflight(phase):
    historical = source_preflight()
    bank = json.loads(BANK.with_suffix('.json').read_text())
    assert bank['status'] == 'passed' and bank['parent_sha256'] == EXPECTED_SHA
    assert bank['bank_sha256'] == sha(BANK)
    for row in bank['source_files']:
        assert sha(row['path']) == row['sha256']
    if phase:
        integration = json.loads((directory(0) / 'report.json').read_text())
        assert integration['status'] == 'passed' and integration['optimizer_updates'] == 0
        assert json.loads((directory(0) / 'execution.json').read_text())['status'] == 'completed'
        assert integration['implementation_sources'] == sources()
    if phase == 2:
        gate = json.loads((OUTPUT / 'gate100.json').read_text())
        prior = json.loads((directory(1) / 'report.json').read_text())
        assert gate['status'] == 'passed' and gate['checkpoint_sha256'] == prior['output_sha256']
        assert prior['status'] == 'passed' and prior['phase_updates'] == 100
    return historical


def worker(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0',
        kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
    import torch
    from tensordict import TensorDict
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from reference_reward_variants import plain
    from reference_low_stairs_preservation_validate import command_audit
    from reference_low_stairs_preservation import joint_target_preservation_loss

    save(args.output / 'source_check.json', preflight(args.phase))
    reference, agent = load_reference()
    from reference_low_stairs_guarded_cfg import configure
    from reference_low_stairs_validate import geometry_audit
    from reference_six_terrain_cfg import column_names
    cfg, audits = configure(reference)
    cfg.seed = agent.seed = 42
    cfg.scene.num_envs = 4096
    cfg.sim.device = agent.device = 'cuda:0'
    agent.max_iterations = 200; agent.logger = 'tensorboard'
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    save(args.output / 'asset_resolution.json', resolve_lite3_asset(cfg))
    contract = terrain_contract(cfg)
    save(args.output / 'configuration.json', dict(environment=plain(cfg), agent=plain(agent), contract=contract))
    save(args.output / 'configuration_audit.json', audits)
    save(args.output / 'implementation_sources.json', sources())
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = False
    base = ManagerBasedRLEnv(cfg=cfg)
    start = 100 if args.phase == 2 else 0
    if start:
        previous = json.loads((directory(1) / 'report.json').read_text())
        parent = Path(previous['output_checkpoint'])
        assert sha(parent) == previous['output_sha256']
        saved = torch.load(parent, map_location='cpu', weights_only=True)
        curriculum = saved['infos'][STATE_KEY]
    else:
        parent = CHECKPOINT
        saved, _ = checked_state()
        curriculum = legacy_histogram_state(base, contract)
    check_algorithm(saved, start)
    runner, env, restored = make_resumed_runner(base, agent.to_dict(), saved, curriculum, contract,
                                               log_dir=str(args.output / 'training'))
    preserver = Preserver(BANK, base.device)
    if start:
        preserver.restore(saved['infos'][KEY])
    assert preserver.calls == start * 20 and runner.current_learning_iteration == 10000 + start
    attach(runner.alg, preserver); attach_saving(runner, preserver)
    save(args.output / 'restore.json', dict(status='passed', **restored, parent=str(parent),
        parent_sha256=sha(parent), auxiliary_state=preserver.state(),
        provenance=curriculum.get('provenance')))
    save(args.output / 'geometry.json', geometry_audit(base, cfg))
    terrain, command = base.scene.terrain, base.command_manager.get_term('base_velocity')
    report = dict(status='running', phase=args.phase, phase_updates=start, max_updates=200,
        original_parent_sha256=EXPECTED_SHA, parent_checkpoint=str(parent), parent_sha256=sha(parent),
        variant=VARIANT, seed=42, physical_gpu=1, num_envs=4096, optimizer_updates=0,
        environment_steps=0, updates=[], checkpoints=[], automatic_extension=False,
        initial_level_histogram=torch.bincount(terrain.terrain_levels, minlength=10).cpu().tolist(),
        terrain_column_names=column_names(cfg), curriculum_contract=contract,
        implementation_sources=sources(), ppo_update_sha256=PPO_UPDATE_SHA,
        bank_sha256=preserver.digest, initial_learning_rate=runner.alg.learning_rate,
        terminated=0, truncated=0)

    def checkpoint(n):
        check_algorithm(runner.alg.save(), n)
        assert preserver.calls == n * 20
        path = args.output / f'model_updates_{10000+n:05d}.pt'
        original_iteration = runner.current_learning_iteration
        try:
            runner.current_learning_iteration = 9999 + n
            runner.save(str(path), infos=dict(phase_updates=n, completed_updates=10000+n,
                trial_budget=200, parent_sha256=EXPECTED_SHA, variant=VARIANT, validation_only=n == 0))
        finally:
            runner.current_learning_iteration = original_iteration
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        assert equal_state(loaded['infos'][STATE_KEY], capture_curriculum(base, contract))
        assert loaded['infos'][KEY] == preserver.state()
        report['checkpoints'].append(dict(phase_updates=n, path=str(path), sha256=sha(path)))
        return loaded

    if args.phase == 0:
        # 同步检查真实 actor 输出；任何验证均不调用 optimizer.step。
        obs, target, groups = preserver.sample()
        with torch.no_grad():
            runner.alg.actor(obs, stochastic_output=True)
            distribution = [p.clone() for p in runner.alg.actor.output_distribution_params]
            prediction = runner.alg.actor(obs)
            error = float((prediction-target).abs().max())
            assert error < 1e-4
            assert joint_target_preservation_loss(prediction, prediction, groups).item() == 0.
            preserver.loss(runner.alg.actor)
            assert all(torch.equal(a,b) for a,b in zip(distribution, runner.alg.actor.output_distribution_params))
        probe = copy.deepcopy(runner.alg.actor)
        with torch.no_grad():
            next(probe.mlp.parameters()).add_(.001)
        loss = preserver.loss(probe)
        loss.backward()
        assert float(loss) > 0 and all(torch.isfinite(p.grad).all() for p in probe.parameters() if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in probe.parameters())
        assert all(p.grad is None for p in runner.alg.actor.parameters())
        assert all(p.grad is None for p in runner.alg.critic.parameters()) and not preserver.target.requires_grad
        preserver.calls = 0; preserver.loss_sum = 0.
        del probe
        sampler = command_audit(plain(reference.commands.base_velocity))
        runner.logger.init_logging_writer()
        loaded = checkpoint(0)
        restored_again = restore_algorithm(runner, loaded)
        duplicate = Preserver(BANK, base.device); duplicate.restore(loaded['infos'][KEY])
        a, b = preserver.sample(), duplicate.sample()
        assert torch.equal(a[0]['policy'], b[0]['policy']) and equal_state(a[1:], b[1:])
        with torch.inference_mode():
            obs = env.get_observations()
            for _ in range(64):
                obs, rewards, done, extra = env.step(runner.alg.actor(obs))
                assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(rewards).all()
                term = base.reward_manager.get_term_cfg('stair_clearance').func
                assert not term.state.active[done.bool()].any()
        for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
            assert equal_state(saved[key], runner.alg.save()[key])
        assert command.changed_left > 0 and command.changed_right > 0
        report.update(status='passed', optimizer_updates=0, environment_steps=64, interactions=64*4096,
            parent_action_max_error=error, perturbed_loss=float(loss.detach()), finite_gradient=True,
            actor_and_critic_parameters_unchanged=True, distribution_cache_unchanged=True,
            restored_again=restored_again, sampler=sampler, initial_gait_level=1.,
            converted_left=command.changed_left, converted_right=command.changed_right)
        runner.logger.stop_logging_writer()
        save(args.output / 'report.json', report)
        print('GUARDED INTEGRATION PASSED: zero optimizer updates', flush=True)
        sys.stdout.flush(); os._exit(0)

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    coverage = torch.zeros(200, dtype=torch.long, device=base.device)
    clearance_frames = torch.zeros_like(coverage)
    command_frames = torch.zeros(3, dtype=torch.long, device=base.device)
    native_reward = base.reward_manager.compute
    def reward_capture(dt):
        value = native_reward(dt)
        slot = terrain.terrain_types * 10 + terrain.terrain_levels
        coverage.add_(torch.bincount(slot, minlength=200))
        term = base.reward_manager.get_term_cfg('stair_clearance').func
        clearance_frames.add_(torch.bincount(slot[term.state.cost.mean(-1) > 0], minlength=200))
        cmd = command.command
        pure = cmd[:, :2].norm(dim=-1) < .1
        command_frames.add_(torch.stack(((pure & (cmd[:, 2].abs() < .1)).sum(),
            (pure & ~command.is_heading_env & (cmd[:, 2] > .1)).sum(),
            (pure & ~command.is_heading_env & (cmd[:, 2] < -.1)).sum())))
        return value
    base.reward_manager.compute = reward_capture
    native_step = env.step
    def step(action):
        if report['environment_steps'] >= 100 * 24:
            raise RuntimeError('本段训练超出 100 次预算')
        result = native_step(action)
        assert torch.isfinite(action).all() and torch.isfinite(result[1]).all()
        assert all(torch.isfinite(v).all() for v in result[0].values())
        for name in ('height_scanner', 'height_scanner_base'):
            assert torch.isfinite(base.scene[name].data.ray_hits_w).all()
        term = base.reward_manager.get_term_cfg('stair_clearance').func
        assert not term.state.active[result[2].bool()].any()
        report['environment_steps'] += 1
        report['terminated'] += int(base.reset_terminated.sum())
        report['truncated'] += int(base.reset_time_outs.sum())
        return result
    env.step = step
    original_init = runner.logger.init_logging_writer
    def initialize():
        original_init(); checkpoint(start)
    runner.logger.init_logging_writer = initialize
    original_log = runner.logger.log
    def record(**kwargs):
        loss = {k: float(v) for k,v in kwargs['loss_dict'].items()}
        assert all(math.isfinite(v) for v in loss.values())
        n = start + len(report['updates']) + 1
        assert kwargs['it'] == 9999+n and n <= 200
        report['updates'].append(dict(update=n, losses=loss, learning_rate=runner.alg.learning_rate,
            collect_s=kwargs['collect_time'], learn_s=kwargs['learn_time'],
            mean_terrain_level=float(terrain.terrain_levels.float().mean()), gait_level=float(native_rewards().gait_level)))
        report.update(phase_updates=n, optimizer_updates=n-start)
        original_log(**kwargs)
        if n % 20 == 0:
            check_algorithm(runner.alg.save(), n)
            save(args.output / 'report.json', report)
        if n == start + 100:
            checkpoint(n)
    runner.logger.log = record
    save(args.output / 'report.json', report)
    runner.learn(num_learning_iterations=100, init_at_random_ep_len=True)
    check_algorithm(runner.alg.save(), start+100)
    assert report['environment_steps'] == 2400 and len(report['updates']) == 100
    assert preserver.calls == (start+100)*20 and sha(CHECKPOINT) == EXPECTED_SHA
    report.update(status='passed', interactions=4096*2400, cumulative_updates=10100+start,
        output_checkpoint=report['checkpoints'][-1]['path'], output_sha256=report['checkpoints'][-1]['sha256'],
        final_learning_rate=runner.alg.learning_rate,
        final_level_histogram=torch.bincount(terrain.terrain_levels, minlength=10).cpu().tolist(),
        coverage_by_column_level=coverage.reshape(20,10).cpu().tolist(),
        clearance_by_column_level=clearance_frames.reshape(20,10).cpu().tolist(),
        standing_fixed_left_fixed_right_frames=command_frames.cpu().tolist(),
        converted_left=command.changed_left, converted_right=command.changed_right, reference_sources=source_manifest())
    save(args.output / 'report.json', report)
    print(f'GUARDED PHASE COMPLETE: {start+100}/200; process exits', flush=True)
    sys.stdout.flush(); os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=int, choices=(0,1,2), required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args(); args.output = directory(args.phase)
    if not args.child:
        preflight(args.phase)
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=1, output=args.output, timeout=7200, minimum_free_mib=12288)
    try:
        worker(args)
    except Exception:
        traceback.print_exc(); save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
