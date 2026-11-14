"""新旧净空奖励的配对冻结验证，复用已验收的课程恢复，不提供训练模式。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
import traceback

from reference_lite3_loader import ROOT, load_reference, resolve_lite3_asset
from reference_stairs_eval import save, sha


def collect(args):
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from reference_low_stairs_trial import CHECKPOINT, checked_state, source_check
    from reference_low_stairs_resume import legacy_histogram_state, make_resumed_runner, terrain_contract, native_rewards
    from reference_reward_resume import equal_state

    historical = source_check()
    saved, _ = checked_state()
    reference, agent = load_reference()
    from reference_low_stairs_cfg import apply_terrain, apply_rewards
    from reference_low_stairs_clearance_cfg import apply_clearance, StairClearanceReward
    from reference_low_stairs_clearance import clearance_sample, height_cost
    from reference_low_stairs_rewards import sample, stair_clearance, clearance_cost
    from reference_six_terrain_cfg import replace_terrain, column_names, FAMILIES
    from reference_reward_variants import plain

    cfg, _ = apply_terrain(replace_terrain(reference)); cfg, _ = apply_rewards(cfg)
    candidate, audit = apply_clearance(cfg)
    if args.variant == 'candidate':
        cfg = candidate
    cfg.seed = agent.seed = 42
    cfg.scene.num_envs = 4096
    cfg.sim.device = agent.device = 'cuda:0'
    resolve_lite3_asset(cfg)
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    contract = terrain_contract(cfg)
    save(args.output / 'configuration.json', dict(environment=plain(cfg), agent=plain(agent),
         contract=contract, clearance_audit=audit, training=False, optimizer_updates=0))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    base = ManagerBasedRLEnv(cfg=cfg)
    curriculum = legacy_histogram_state(base, contract)
    runner, env, resume_audit = make_resumed_runner(base, agent.to_dict(), saved, curriculum, contract)
    actor_before = copy.deepcopy(runner.alg.save())
    checkpoint_before = sha(CHECKPOINT)
    reward_names = list(base.reward_manager.active_terms)
    index = reward_names.index('stair_clearance')
    if args.variant == 'candidate':
        term = base.reward_manager.get_term_cfg('stair_clearance').func
    else:
        term = StairClearanceReward(candidate.rewards.stair_clearance, base)
        native_reset = base.reward_manager.reset
        def reset(env_ids=None):
            result = native_reset(env_ids=env_ids)
            term.reset(env_ids)
            return result
        base.reward_manager.reset = reset
    names = column_names(cfg)
    families = [names[int(i)] for i in base.scene.terrain.terrain_types.cpu()]
    assert {FAMILIES[n] for n in families} == {'flat', 'rough', 'slope', 'stairs', 'friction', 'slope_rough'}
    stairs = torch.tensor([FAMILIES[n] == 'stairs' for n in families], device=base.device)
    selected = sorted(set(range(0, 4096, 26)) | set(torch.nonzero(stairs).flatten()[::10].cpu().tolist()))
    ids = torch.tensor(selected, device=base.device)
    first_episode = torch.ones(4096, dtype=torch.bool, device=base.device)
    counters = {k: torch.zeros(4, dtype=torch.long, device=base.device) for k in (
        'scored', 'wall_scored', 'wall_old_zero_new_positive', 'slow_or_backward_scored',
        'grounded_exempt', 'remembered_target', 'edge_wall_scored')}
    scope = {k: 0 for k in ('non_stair', 'pure_yaw', 'standing', 'native_resets')}
    errors = []; scales = []; totals = []
    arrays = {k: [] for k in ('bottom', 'ground', 'obstacle', 'target', 'force', 'foot_velocity',
        'direction', 'support', 'wall', 'wall_distance', 'new_per_foot', 'old_per_foot',
        'enabled', 'upright', 'scored', 'active_target', 'first_episode', 'command')}
    hashes = {k: hashlib.sha256() for k in ('trajectory', 'other_rewards', 'candidate_cost', 'old_cost')}

    def digest(key, *values):
        for value in values:
            hashes[key].update(value.contiguous().cpu().numpy().tobytes())

    native_compute = base.reward_manager.compute
    def capture(dt):
        total = native_compute(dt)
        rates = base.reward_manager._step_reward
        x = sample(base); wall = clearance_sample(base)
        old = stair_clearance(base)
        new = term(base)
        # 重复读取不得再次推进状态；与奖励管理器实际使用的值逐项核对。
        state_before = {k: getattr(term.state, k).clone() for k in ('active', 'target', 'dwell', 'cost')}
        assert torch.equal(term(base), new)
        assert all(torch.equal(v, getattr(term.state, k)) for k, v in state_before.items())
        expected = new if args.variant == 'candidate' else old
        error = float((rates[:, index] + expected).abs().max())
        assert error < 1e-6 and torch.allclose(total, rates.sum(-1) * dt, atol=1e-5, rtol=1e-4)
        assert torch.isfinite(rates).all() and new.ge(0.).all() and new.le(1.).all()
        assert new[~x['enabled'] | ~x['upright']].eq(0.).all()
        supported = x['support'] & ~wall['wall']
        assert term.state.cost[supported].eq(0.).all()
        old_per_foot = ((x['obstacle'] + .025 - x['bottom']).clamp_min(0.) / .05).clamp_max(1.).square()
        forward = (x['foot_velocity'][..., :2] * x['direction'][:, None]).sum(-1)
        old_per_foot *= (forward.clamp(0., .5) / .5 * (x['force'].norm(dim=-1) < 1.)
                         * x['upcoming'] * x['upright'][:, None])
        assert torch.allclose(old_per_foot.mean(-1), old, atol=1e-7)
        actual = term.state.cost
        valid = first_episode[:, None]
        positive = (actual > 1e-6) & valid
        checks = dict(scored=positive, wall_scored=positive & wall['wall'],
            wall_old_zero_new_positive=positive & wall['wall'] & old_per_foot.eq(0.),
            slow_or_backward_scored=positive & (forward <= .01),
            grounded_exempt=supported & x['upcoming'] & valid,
            remembered_target=positive & ~x['upcoming'],
            edge_wall_scored=positive & wall['wall'] & ~torch.isfinite(wall['distance'][..., 0]))
        for key, value in checks.items():
            counters[key] += value.sum(0)
        command = base.command_manager.get_command('base_velocity')
        pure = (command[:, :2].norm(dim=-1) <= .1) & (command[:, 2].abs() > .1)
        stand = (command.norm(dim=-1) <= .1)
        assert new[pure | stand | ~stairs].eq(0.).all()
        for key, mask in (('non_stair', ~stairs), ('pure_yaw', pure), ('standing', stand)):
            scope[key] += int((mask & first_episode).sum())
        errors.append(error); scales.append(float(native_rewards().gait_level))
        totals.append([float(old.mean()), float(new.mean())])
        digest('other_rewards', rates[:, [i for i in range(len(reward_names)) if i != index]])
        digest('candidate_cost', actual); digest('old_cost', old_per_foot)
        digest('trajectory', base.scene['robot'].data.root_state_w, x['feet'], x['force'],
               base.action_manager.action, command, base.reset_terminated, base.reset_time_outs)
        values = dict(bottom=x['bottom'], ground=x['ground'], obstacle=x['obstacle'],
            target=term.state.target, force=x['force'], foot_velocity=x['foot_velocity'],
            direction=x['direction'], support=x['support'], wall=wall['wall'],
            wall_distance=torch.where(torch.isfinite(wall['distance']), wall['distance'], -1.),
            new_per_foot=actual, old_per_foot=old_per_foot, enabled=x['enabled'], upright=x['upright'],
            scored=term.state.scored, active_target=term.state.active, first_episode=first_episode, command=command)
        for key, value in values.items():
            assert torch.isfinite(value).all(), key
            arrays[key].append(value[ids].cpu().numpy().copy())
        first_episode.logical_and_(~(base.reset_terminated | base.reset_time_outs))
        return total
    base.reward_manager.compute = capture
    policy = runner.get_inference_policy()
    obs = env.get_observations()
    digest('trajectory', obs['policy'], obs['critic'])
    for step in range(360):
        action = policy(obs)
        obs, rewards, done, _ = env.step(action)
        assert obs['policy'].shape == (4096, 45) and obs['critic'].shape == (4096, 393)
        assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(rewards).all()
        assert not term.state.active[done.bool()].any()
        scope['native_resets'] += int(done.sum())
        digest('trajectory', obs['policy'], obs['critic'])
        if (step + 1) % 120 == 0:
            print(f'FROZEN CLEARANCE {args.variant} {step + 1}/360', flush=True)
    assert equal_state(actor_before, runner.alg.save()) and sha(CHECKPOINT) == checkpoint_before
    assert min(scales) == max(scales) == 1.
    assert all(v > 0 for k, v in scope.items())
    assert counters['wall_old_zero_new_positive'][2:].gt(0).all(), '必须实际覆盖两条后腿碰阶'
    assert counters['slow_or_backward_scored'].gt(0).all()
    assert counters['grounded_exempt'].gt(0).all()
    # 显式选择重置，确认新回合不携带旧目标，未重置的环境保持原状态。
    reset_ids = torch.tensor([0, 1024, 2048, 3072], device=base.device)
    keep = torch.ones(4096, dtype=torch.bool, device=base.device); keep[reset_ids] = False
    before = term.state.active.clone()
    base.reset(env_ids=reset_ids)
    assert not term.state.active[reset_ids].any() and term.state.cost[reset_ids].eq(0.).all()
    assert torch.equal(term.state.active[keep], before[keep])
    assert not hasattr(base, '_persistent_clearance_sample') and not hasattr(base, '_low_stairs_sample')
    np.savez_compressed(args.output / 'trace.npz', env_ids=np.array(selected), **{k: np.asarray(v) for k, v in arrays.items()})
    report = dict(status='passed', variant=args.variant, environments=4096, steps=360,
        optimizer_updates=0, actor_critic_optimizer_unchanged=True, observations=[45, 393],
        checkpoint=str(CHECKPOINT), checkpoint_sha256=checkpoint_before,
        resume_audit=resume_audit, configuration_audit=audit, source_check=historical,
        initial_histogram=torch.bincount(curriculum['terrain_levels'], minlength=10).tolist(),
        gait_range=[min(scales), max(scales)], exact_reward_max_error=max(errors),
        counts_first_episode_per_foot={k: v.cpu().tolist() for k, v in counters.items()},
        scope_counts=scope, mean_cost_old_new=np.mean(totals, axis=0).tolist(),
        hashes={k: v.hexdigest() for k, v in hashes.items()}, terrain_columns=names,
        selective_reset_verified=True, same_step_cache_verified=True,
        trace_environments=len(selected), first_episode_survivors=int(first_episode.sum()),
        sources=[dict(path=str(ROOT / 'scripts/tools' / name), sha256=sha(ROOT / 'scripts/tools' / name))
                 for name in ('reference_low_stairs_clearance.py', 'reference_low_stairs_clearance_cfg.py',
                              'reference_low_stairs_clearance_validate.py', 'reference_low_stairs_resume.py')],
        note='只验证计分与集成。原生随机化、命令和课程保持；旧课程按已记录总数迁移。接触由实际射线和合力联合判定，仍是代理。零 PPO 更新，不能据此宣称上阶或转向能力改善。')
    save(args.output / 'report.json', report)
    print('FROZEN CLEARANCE PASSED; zero optimizer updates', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('baseline', 'candidate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args(); args.output = args.output.resolve()
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=1, output=args.output, timeout=1800)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device='cuda:0',
            kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        import torch
        with torch.inference_mode():
            collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc(); save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
