"""Budgeted physics check of early-recovery/ordinary action-preservation timing."""
import argparse
import json
from pathlib import Path
import signal

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
    from rl_training.vector_step_ledger import VectorStepLedger
    p = json.loads(Path(args.protocol).read_text())
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = {str(f.relative_to(root)): sha256_file(f) for folder in (root/'source/rl_training/rl_training', root/'scripts/reinforcement_learning') for f in folder.rglob('*.py')}
    assert p['max_controls'] == p['steps'] * p['num_envs']
    ledger = VectorStepLedger(output/'ledger.json', num_envs=p['num_envs'], max_transitions=p['max_controls'], metadata={'protocol_sha256': sha256_file(args.protocol)})
    report = dict(status='running', protocol_sha256=sha256_file(args.protocol), source_files=sources,
                  base_sha256=p['base_sha256'], anchor_options=p['anchor_options'])
    write_json_atomic(output/'report.json', report)
    launcher = env = trace = None
    try:
        assert sha256_file(p['base_checkpoint']) == p['base_sha256']
        launcher = AppLauncher(args)
        stop = []
        signal.signal(signal.SIGINT, lambda signum, frame: stop.append(signum))
        signal.signal(signal.SIGTERM, lambda signum, frame: stop.append(signum))
        import numpy as np
        import torch
        import gymnasium as gym
        import rl_training.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rl_training.policy_route import configure_route_training, route_heading_trace
        from rl_training.recovery_teacher import configure_recovery_resets, check_recovery_event_order
        from rl_training.online_adaptation.runner_contract import configure_forward_command, actuator_delay_metadata
        from rl_training.policy_flat_anchor import FrozenFlatMean, flat_anchor_cost
        from rl_training.policy_recovery_anchor import configure_recovery_anchor, prepare_recovery_anchor
        from rl_training.lite3_joint_conventions import clip_lite3_policy_action
        from rl_training.online_adaptation.step_trajectory import StepTrajectory
        torch.set_num_threads(4)
        torch.manual_seed(p['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = parse_env_cfg(p['task'], device=args.device, num_envs=p['num_envs'])
        cfg.seed = p['seed']
        configure_route_training(cfg, p['route_training'])
        velocity_options=p.get('lateral_velocity')
        if velocity_options:
            from rl_training.policy_route import configure_lateral_velocity_training, route_lateral_velocity_trace
            configure_lateral_velocity_training(cfg,velocity_options)
        configure_forward_command(cfg.commands.base_velocity, .3)
        configure_recovery_resets(cfg, p['recovery_resets'])
        configure_recovery_anchor(cfg, p['anchor_options'])
        cfg.rewards.course_lateral.weight = p['recovery_course_lateral_weight']
        env = gym.make(p['task'], cfg=cfg)
        raw = env.unwrapped
        check_recovery_event_order(raw)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        observation = wrapped.get_observations()
        assert observation['policy'].shape == (p['num_envs'], 48)
        assert observation['critic'].shape == (p['num_envs'], 51)
        teacher = FrozenFlatMean(torch.load(p['base_checkpoint'], map_location='cpu', weights_only=False)).to(args.device)
        initial = {k: v.clone() for k, v in teacher.state_dict().items()}
        write_json_atomic(output/'delays.json', actuator_delay_metadata(raw.scene['robot']))
        np.savez_compressed(output/'initial.npz', terrain=raw.scene.terrain.terrain_types.cpu().numpy(), case=raw._recovery_case.cpu().numpy(),
                            root=raw.scene['robot'].data.root_state_w.cpu().numpy(), origins=raw.scene.env_origins.cpu().numpy())
        trace = StepTrajectory(output/'trajectory', metadata={'kind': 'recovery_anchor_smoke', 'step_dt': raw.step_dt}, chunk_steps=128, vector_dtype='float32')
        index = raw.reward_manager.active_terms.index('recovery_policy_anchor')
        expected_age = torch.zeros_like(raw.episode_length_buf)
        observed = dict(ordinary=0, early=0, tail=0, repeated_recovery_resets=0, nonzero_cost_rows=0)
        heading_trace='heading_lookahead_m' in p['route_training']
        for step in range(p['steps']):
            if stop:
                raise InterruptedError('stop before next control')
            obs = observation['policy']
            torch.testing.assert_close(raw.episode_length_buf, expected_age)
            target, mask = prepare_recovery_anchor(raw, teacher, obs, p['anchor_options']['early_steps'])
            offset = p['probe_amplitude'] * (1 if step % 2 else -1)
            requested = teacher(obs) + offset
            emitted = clip_lite3_policy_action(requested)
            cost = flat_anchor_cost(emitted, target, mask)
            ordinary = raw._recovery_case < 0
            early = ~ordinary & (expected_age < p['anchor_options']['early_steps'])
            assert torch.equal(mask, ordinary | early)
            observed['ordinary'] += int(ordinary.sum())
            observed['early'] += int(early.sum())
            observed['tail'] += int((~ordinary & ~early).sum())
            observed['nonzero_cost_rows'] += int((cost > 0).sum())
            observed['repeated_recovery_resets'] += int(((expected_age == 0) & ~ordinary).sum()) if step else 0
            before = trace.snapshot(policy_observation=obs, recovery_case=raw._recovery_case, episode_steps=expected_age,
                recovery_anchor_target=target, recovery_anchor_mask=mask, requested_action=requested, emitted_action=emitted,
                recovery_anchor_cost=cost)
            ledger.reserve()
            if heading_trace:raw._route_heading_snapshot=None
            if velocity_options:raw._route_velocity_snapshot=None
            observation, reward, done, _ = wrapped.step(emitted)
            ledger.confirm()
            component = raw.reward_manager._step_reward[:, index] * raw.step_dt
            torch.testing.assert_close(raw._recovery_anchor_applied_cost, cost, atol=1e-6, rtol=0)
            torch.testing.assert_close(component, -p['anchor_options']['weight']*cost*raw.step_dt, atol=1e-6, rtol=1e-5)
            history = emitted.clone()
            history[done.bool()] = 0
            torch.testing.assert_close(observation['policy'][:, -12:], history, atol=1e-6, rtol=0)
            trace.append(**before, next_policy_observation=observation['policy'], done=done,
                         recovery_anchor_applied_cost=raw._recovery_anchor_applied_cost, recovery_anchor_reward=component, reward=reward,
                         **(route_heading_trace(raw) if heading_trace else {}),
                         **(route_lateral_velocity_trace(raw) if velocity_options else {}))
            expected_age = torch.where(done.bool(), 0, expected_age + 1)
        assert all(value > 0 for value in observed.values()), observed
        assert all(torch.equal(v, teacher.state_dict()[k]) for k, v in initial.items())
        assert all(sha256_file(root/f) == h for f, h in sources.items())
        assert sha256_file(args.protocol) == report['protocol_sha256']
        trace.close(complete=True)
        trace = None
        ledger.finish()
        report.update(status='complete', ledger=ledger.state, observed=observed, step_dt=raw.step_dt,
                      recovery_anchor_timing_and_mask_passed=True, anchor_model_unchanged=True)
        if heading_trace:report.update(route_heading_options=p['route_training'],route_heading_timing_passed=True)
        if velocity_options:report.update(lateral_velocity_options=velocity_options,lateral_velocity_timing_passed=True)
        write_json_atomic(output/'report.json', report)
    except BaseException as error:
        if ledger.state['status'] == 'running':
            ledger.finish(aborted=True)
        if trace is not None:
            trace.close(complete=False)
        report.update(status='failed', error=repr(error), ledger=ledger.state)
        write_json_atomic(output/'report.json', report)
        raise
    finally:
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == '__main__':
    main()
