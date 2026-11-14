"""已批准的净空单项续训：固定 500 次，恢复父模型课程，不自动重启或延长。"""
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

from reference_lite3_loader import ROOT, REFERENCE, load_reference, resolve_lite3_asset, source_manifest
from reference_low_stairs_evaluate import EXPECTED_SHA
from reference_low_stairs_trial import CHECKPOINT, checked_state, source_check
from reference_low_stairs_resume import STATE_KEY, capture_curriculum, legacy_histogram_state, make_resumed_runner, native_rewards, terrain_contract
from reference_reward_resume import equal_state
from reference_stairs_eval import save, sha

VARIANT = 'persistent_clearance'
TRAINING = ROOT / 'logs/reference_six/low_stairs_clearance_trial_20261115/clearance_seed42_500'
PREPARATION = ROOT / 'docs/lite3-low-stairs-clearance-repair-20261115'
BUDGET = 500


def source_preflight():
    historical = source_check()
    report = json.loads((PREPARATION / 'verification.json').read_text())
    if report['status'] != 'passed' or report['tests_passed'] != 89 or not report['paired_hashes_equal']:
        raise ValueError('净空准备阶段未通过')
    sources = report['sources'] + json.loads((PREPARATION / 'candidate_report.json').read_text())['sources']
    for row in sources:
        if sha(row['path']) != row['sha256']:
            raise ValueError(f'已验证源码发生变化: {row["path"]}')
    return dict(status='passed', historical=historical, preparation_sha256=sha(PREPARATION / 'verification.json'))


def check_algorithm(state, updates):
    import torch
    if not 0 <= updates <= BUDGET:
        raise ValueError('更新次数超出批准预算')
    if {int(v['step']) for v in state['optimizer_state_dict']['state'].values()} != {200000 + updates * 20}:
        raise ValueError('Adam 步数与更新预算不一致')
    for key in ('actor_state_dict', 'critic_state_dict'):
        if not all(torch.isfinite(v).all() for v in state[key].values()):
            raise FloatingPointError('模型参数出现非有限数值')
    for group in state['optimizer_state_dict']['param_groups']:
        if not math.isfinite(group['lr']) or group['lr'] <= 0:
            raise ValueError('学习率异常')


def worker(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0',
        kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    save(args.output / 'source_check.json', source_preflight())
    saved, _ = checked_state()
    reference, agent = load_reference()
    from reference_six_terrain_cfg import replace_terrain, column_names
    from reference_low_stairs_cfg import apply_terrain, apply_rewards
    from reference_low_stairs_clearance_cfg import apply_clearance
    from reference_low_stairs_clearance import clearance_sample
    from reference_low_stairs_validate import geometry_audit
    from reference_six_train import audit_simulation
    from reference_reward_variants import plain

    cfg, terrain_audit = apply_terrain(replace_terrain(reference))
    cfg, reward_audit = apply_rewards(cfg)
    cfg, clearance_audit = apply_clearance(cfg)
    cfg.seed = agent.seed = 42
    cfg.scene.num_envs = 4096
    cfg.sim.device = agent.device = 'cuda:0'
    agent.max_iterations = BUDGET; agent.logger = 'tensorboard'
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    save(args.output / 'asset_resolution.json', resolve_lite3_asset(cfg))
    contract = terrain_contract(cfg)
    save(args.output / 'configuration.json', dict(environment=plain(cfg), agent=plain(agent), contract=contract))
    save(args.output / 'terrain_audit.json', terrain_audit)
    save(args.output / 'reward_audit.json', reward_audit)
    save(args.output / 'clearance_audit.json', clearance_audit)
    dump_yaml(str(args.output / 'params/env.yaml'), cfg)
    dump_yaml(str(args.output / 'params/agent.yaml'), agent)
    files = [Path(__file__), Path(__file__).with_name('reference_low_stairs_resume.py'),
             Path(__file__).with_name('reference_low_stairs_clearance.py'),
             Path(__file__).with_name('reference_low_stairs_clearance_cfg.py')]
    save(args.output / 'implementation_sources.json', [dict(path=str(p), sha256=sha(p)) for p in files])
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = False
    base = ManagerBasedRLEnv(cfg=cfg)
    curriculum = legacy_histogram_state(base, contract)
    runner, env, restore = make_resumed_runner(base, agent.to_dict(), saved, curriculum, contract,
                                              log_dir=str(args.output / 'training'))
    save(args.output / 'restore.json', dict(status='passed', **restore, provenance=curriculum['provenance']))
    audit_simulation(base, cfg, args.output, make_plot=False)
    save(args.output / 'geometry.json', geometry_audit(base, cfg))
    assert native_rewards().gait_level == 1. and runner.current_learning_iteration == 10000
    initial = copy.deepcopy(runner.alg.save())
    check_algorithm(initial, 0)
    runner.add_git_repo_to_log(__file__)
    runner.add_git_repo_to_log(str(REFERENCE / 'scripts/reinforcement_learning/rsl_rl/train.py'))
    terrain = base.scene.terrain
    names = column_names(cfg)
    report = dict(status='running', seed=42, from_scratch=False, checkpoint_loaded=str(CHECKPOINT),
        initial_sha256=EXPECTED_SHA, initial_iteration=10000, initial_learning_rate=runner.alg.learning_rate,
        reward_variant='low_stairs_persistent_clearance', repair_variant=VARIANT, terrain_profile='low_stairs',
        budget=BUDGET, num_envs=4096, physical_gpu=1, completed_updates=0, environment_steps=0,
        resets=0, terminated=0, truncated=0, updates=[], checkpoints=[], automatic_extension=False,
        terrain_column_names=names, initial_gait_level=1.,
        initial_level_histogram=torch.bincount(terrain.terrain_levels, minlength=10).cpu().tolist(),
        curriculum_mode=curriculum['mode'], curriculum_contract=contract)
    coverage = torch.zeros(200, dtype=torch.long, device=base.device)
    moving = torch.zeros_like(coverage); scored = torch.zeros_like(coverage)
    wall_frames = torch.zeros(4, dtype=torch.long, device=base.device)
    term = base.reward_manager.get_term_cfg('stair_clearance').func
    native_reward = base.reward_manager.compute
    def reward_capture(dt):
        total = native_reward(dt)
        slot = terrain.terrain_types * 10 + terrain.terrain_levels
        coverage.add_(torch.bincount(slot, minlength=200))
        translation = base.command_manager.get_command('base_velocity')[:, :2].norm(dim=-1) > .1
        moving.add_(torch.bincount(slot[translation], minlength=200))
        active = term.state.cost.mean(-1) > 0.
        scored.add_(torch.bincount(slot[active], minlength=200))
        wall_frames.add_(((term.state.cost > 0.) & clearance_sample(base)['wall']).sum(0))
        return total
    base.reward_manager.compute = reward_capture
    native_step = env.step
    def checked_step(action):
        if report['environment_steps'] >= BUDGET * agent.num_steps_per_env:
            raise RuntimeError('环境交互超出批准预算')
        if not torch.isfinite(action).all():
            raise FloatingPointError('动作出现非有限数值')
        obs, rewards, dones, extras = native_step(action)
        if not torch.isfinite(rewards).all() or not all(torch.isfinite(v).all() for v in obs.values()):
            raise FloatingPointError('观测或奖励出现非有限数值')
        for name in ('height_scanner', 'height_scanner_base'):
            if not torch.isfinite(base.scene[name].data.ray_hits_w).all():
                raise FloatingPointError('地形射线出现非有限数值')
        if term.state.active[dones.bool()].any():
            raise RuntimeError('回合重置未清除摆脚目标')
        report['environment_steps'] += 1
        report['resets'] += int(dones.sum())
        report['terminated'] += int(base.reset_terminated.sum())
        report['truncated'] += int(base.reset_time_outs.sum())
        return obs, rewards, dones, extras
    env.step = checked_step

    def checkpoint(updates):
        check_algorithm(runner.alg.save(), updates)
        path = args.output / f'model_updates_{10000 + updates:05d}.pt'
        next_iteration = runner.current_learning_iteration
        try:
            runner.current_learning_iteration = 9999 + updates
            runner.save(str(path), infos=dict(from_scratch=False, completed_updates=10000 + updates,
                phase_updates=updates, trial_budget=BUDGET, terrain_profile='low_stairs',
                reward_variant='low_stairs_persistent_clearance', repair_variant=VARIANT,
                validation_only=updates == 0, parent_sha256=EXPECTED_SHA,
                curriculum_migration=curriculum['provenance']))
        finally:
            runner.current_learning_iteration = next_iteration
        output = torch.load(path, map_location='cpu', weights_only=True)
        assert equal_state(output['infos'][STATE_KEY], capture_curriculum(base, contract))
        report['checkpoints'].append(dict(phase_updates=updates, path=str(path), sha256=sha(path)))

    def record_coverage():
        report['coverage_by_column_level'] = coverage.reshape(20, 10).cpu().tolist()
        report['translation_frames_by_column_level'] = moving.reshape(20, 10).cpu().tolist()
        report['clearance_frames_by_column_level'] = scored.reshape(20, 10).cpu().tolist()
        report['wall_scored_frames_per_foot'] = wall_frames.cpu().tolist()

    # 原生 learn 初始化日志器后、第一步采样前保存启动检查点，避免重复创建日志器。
    native_initialize = runner.logger.init_logging_writer
    def initialize():
        native_initialize()
        checkpoint(0)
        save(args.output / 'report.json', report)
    runner.logger.init_logging_writer = initialize
    native_log = runner.logger.log
    def checked_log(**kwargs):
        losses = {k: float(v) for k, v in kwargs['loss_dict'].items()}
        if not all(math.isfinite(v) for v in losses.values()):
            raise FloatingPointError('PPO 损失出现非有限数值')
        report['completed_updates'] += 1
        n = report['completed_updates']
        if n > BUDGET or kwargs['it'] != 9999 + n:
            raise RuntimeError('训练更新计数异常')
        report['updates'].append(dict(update=n, losses=losses, learning_rate=runner.alg.learning_rate,
            collect_s=kwargs['collect_time'], learn_s=kwargs['learn_time'],
            mean_terrain_level=float(terrain.terrain_levels.float().mean()), gait_level=float(native_rewards().gait_level)))
        native_log(**kwargs)
        if n == 20:
            check_algorithm(runner.alg.save(), n)
            save(args.output / 'first20_check.json', dict(status='passed', updates_included_in_budget=20,
                observations_and_losses_finite=True, adam_steps=200400,
                gait_range=[min(r['gait_level'] for r in report['updates']), max(r['gait_level'] for r in report['updates'])],
                average_seconds_per_update=sum(r['collect_s'] + r['learn_s'] for r in report['updates']) / 20))
        if n in (100, 250, 500):
            checkpoint(n)
        if n % 20 == 0 or n in (100, 250, 500):
            record_coverage()
            save(args.output / 'report.json', report)
            print(f'CLEARANCE TRIAL {n}/{BUDGET}', flush=True)
    runner.logger.log = checked_log
    save(args.output / 'report.json', report)
    runner.learn(num_learning_iterations=BUDGET, init_at_random_ep_len=True)
    current = runner.alg.save(); check_algorithm(current, BUDGET)
    assert report['completed_updates'] == BUDGET and runner.current_learning_iteration == 10499
    assert report['environment_steps'] == BUDGET * agent.num_steps_per_env == 12000
    assert [c['phase_updates'] for c in report['checkpoints']] == [0, 100, 250, 500]
    assert all(any(not torch.equal(initial[key][k], v) for k, v in current[key].items())
               for key in ('actor_state_dict', 'critic_state_dict'))
    assert sha(CHECKPOINT) == EXPECTED_SHA
    record_coverage()
    report.update(status='passed', cumulative_updates=10500, interactions=12000 * 4096,
        output_checkpoint=report['checkpoints'][-1]['path'], output_sha256=report['checkpoints'][-1]['sha256'],
        final_learning_rate=runner.alg.learning_rate, sources=source_manifest(),
        final_terrain_level_histogram=torch.bincount(terrain.terrain_levels, minlength=10).cpu().tolist())
    save(args.output / 'report.json', report)
    print('CLEARANCE TRIAL COMPLETE: 500 updates; no automatic extension', flush=True)
    sys.stdout.flush(); os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, choices=(500,), required=True)
    parser.add_argument('--gpu', type=int, choices=(1,), default=1)
    parser.add_argument('--output', type=Path, default=TRAINING)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args(); args.output = args.output.resolve()
    if not args.child:
        source_preflight()
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=1, output=args.output, timeout=21600)
    try:
        worker(args)
    except Exception:
        traceback.print_exc(); save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
