"""Bounded synthetic SourcePPO updates; no Isaac import or terrain interaction."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'source/rl_training'))

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from rl_training.online_adaptation.types import SafetyState
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.source_training import TransitionBudget, load_source_actor_only
from rl_training.source_scan.training_env import SourceTrainingRuntime, SourceTrainingVecEnv
from rl_training.source_scan.training_lifecycle import SourceRunSeeds
from rl_training.source_scan.training_runner import SourceTrainingRunner, _atomic_json, _state_digest
from scripts.tools.source_training_preflight import runner_config
from scripts.tools.source_training_entrypoint import _input_hashes


class Scene(dict):
    pass


class SyntheticManager:
    def __init__(self, run, device, num_envs):
        self.num_envs, self.device = num_envs, torch.device(device)
        n = num_envs
        zeros = lambda *shape, **kwargs: torch.zeros(*shape, device=device, **kwargs)
        seeds = SourceRunSeeds(run)
        self.cfg = NS(source_training_run=run, source_terrain_level='L1', seed=seeds.ppo,
                      source_parallel_sampler_seed=seeds.sampler,
                      scene=NS(terrain=NS(terrain_generator=NS(seed=seeds.geometry,
                          sub_terrains=dict.fromkeys(SOURCE_RESET_BLOCK_COUNTS)))))
        self.scene = Scene(robot=NS(data=NS(root_link_pos_w=zeros(n, 3))))
        self.scene.terrain = NS(env_origins=zeros(n, 3), terrain_types=torch.arange(n, device=device) % 15)
        self.episode_length_buf = zeros(n, dtype=torch.long)
        self.common_step_counter = 0
        self.executed = zeros(n, 12)
        self.callbacks = []
        self.observation_manager = NS(compute=self.observations)
        self.terms = {}
        self.termination_manager = NS(get_term=lambda name: self.terms[name], active_terms=[
            'source_training_safety', 'source_training_completion', 'source_training_timeout'])
        self.state = SafetyState(
            projected_gravity_xy=zeros(n, 2), base_height=zeros(n, 1) + .4,
            joint_pos=zeros(n, 12), joint_vel=zeros(n, 12),
            joint_lower=zeros(n, 12) - 1., joint_upper=zeros(n, 12) + 1.,
            default_joint_pos=zeros(n, 12), action_scale=zeros(1, 12) + 1.,
            nonfoot_contact=zeros(n, 1, dtype=torch.bool), control_timed_out=zeros(n, 1, dtype=torch.bool))

    def add_pre_reset_callback(self, callback):
        self.callbacks.append(callback)

    def observations(self, update_history=False):
        n, device = self.num_envs, self.device
        policy = torch.zeros(n, 45, device=device)
        policy[:, :1] = self.scene['robot'].data.root_link_pos_w[:, :1]
        policy[:, -12:] = self.executed
        critic = torch.zeros(n, 393, device=device)
        critic[:, :45] = policy
        critic[:, 45] = self.episode_length_buf.float() / 2000.
        return {'policy': policy, 'critic': critic}


class SyntheticRslEnv:
    source_synthetic_fixture = True

    def __init__(self, run=1, device='cpu', num_envs=8):
        self.unwrapped = SyntheticManager(run, device, num_envs)
        self.num_envs, self.num_actions, self.device = num_envs, 12, torch.device(device)
        self.cfg, self.clip_actions = self.unwrapped.cfg, None
        env = self.unwrapped
        adapter = NS(safety_state=lambda timeout: replace(env.state, control_timed_out=timeout),
                     contact_sensor=NS(data=NS(net_forces_w=torch.zeros(num_envs, 17, 3, device=device))))
        self.runtime = SourceTrainingRuntime(env, adapter)
        self.endings = []

    def get_observations(self):
        return TensorDict(self.unwrapped.observations(), batch_size=[self.num_envs])

    def step(self, actions):
        env = self.unwrapped
        env.executed = actions.clone()
        env.common_step_counter += 1
        # One pure timeout and one compound failure+timeout, both after action.
        if env.common_step_counter == 1:
            env.episode_length_buf[0] = 1999
        elif env.common_step_counter == 2:
            env.episode_length_buf[1] = 1999
            env.state.base_height[1] = .1
        env.episode_length_buf += 1
        env.scene['robot'].data.root_link_pos_w[:, 0] += .002
        lifecycle = self.runtime.evaluate()
        env.terms = {'source_training_safety': lifecycle.unsafe.clone(),
                     'source_training_completion': lifecycle.completed.clone(),
                     'source_training_timeout': lifecycle.timeout.clone()}
        env.reset_terminated = lifecycle.unsafe | lifecycle.completed
        env.reset_time_outs = lifecycle.timeout.clone()
        done = env.reset_terminated | env.reset_time_outs
        ids = done.nonzero().flatten()
        if len(ids):
            self.endings.append({'step': env.common_step_counter, 'ids': ids.cpu().tolist(),
                                 'terminated': env.reset_terminated[ids].cpu().tolist(),
                                 'truncated': env.reset_time_outs[ids].cpu().tolist()})
            for callback in env.callbacks:
                callback(ids)
            env.episode_length_buf[ids] = 0
            env.scene.terrain.terrain_types[ids] = 14
            env.scene['robot'].data.root_link_pos_w[ids] = 0.
            env.executed[ids] = 0.
        env.state.base_height.fill_(.4)
        reward = 1. - .01 * actions.square().mean(-1)
        return self.get_observations(), reward, done.long(), {}


def make_runner(*, run=1, device='cpu', num_envs=8, rollout_length=24, compact=False):
    torch.manual_seed(SourceRunSeeds(run).ppo)
    base = SyntheticRslEnv(run, device, num_envs)
    env = SourceTrainingVecEnv(base, runtime=base.runtime)
    config = runner_config()
    config['num_steps_per_env'] = rollout_length
    config['algorithm']['class_name'] = 'rl_training.source_scan.training_ppo:SourcePPO'
    if compact:
        config['actor']['hidden_dims'] = [16, 16]
        config['critic']['hidden_dims'] = [16, 16]
        config['algorithm'].update(num_learning_epochs=1, num_mini_batches=1)
    return OnPolicyRunner(env, config, log_dir=None, device=device)


def check(args):
    if args.output_dir.exists():
        raise FileExistsError('synthetic check output already exists')
    torch.set_num_threads(1)
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU fallback is disabled')
        torch.cuda.set_device(args.device)
        if torch.cuda.mem_get_info()[0] < 1024 ** 3:
            raise RuntimeError('synthetic CUDA check requires 1 GiB free')
        torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'schema': 'lite3_source_synthetic_numerics_v1', 'status': 'running',
              'device': args.device, 'synthetic': True, 'simulation_steps': 0,
              'real_source_training_transitions': 0, 'real_source_ppo_updates': 0,
              'torch': torch.__version__, 'rsl_rl': importlib.metadata.version('rsl-rl-lib'),
              'input_sha256': _input_hashes(),
              'runs': []}
    _atomic_json(args.output_dir / 'summary.json', report)
    try:
        for run in (1, 2, 3):
            runner = make_runner(run=run, device=args.device)
            critic_initial = _state_digest(runner.alg.critic)
            load_report = load_source_actor_only(runner.alg.actor, args.checkpoint)
            actor_initial = _state_digest(runner.alg.actor)
            assert not runner.alg.optimizer.state
            assert critic_initial == _state_digest(runner.alg.critic)
            training = SourceTrainingRunner(runner, run=run, budget=TransitionBudget(8, 24, 384),
                output_dir=args.output_dir / f'run_{run}', checkpoint_sha256=load_report['checkpoint_sha256'],
                checkpoint_nodes=(1, 2))
            result = training.run()
            assert result['completed_updates'] == 2 and result['stored_transitions'] == 384
            assert result['source_backward_checks'] == result['source_optimizer_steps_completed'] == 40
            assert actor_initial != _state_digest(runner.alg.actor)
            assert critic_initial != _state_digest(runner.alg.critic)
            assert sum(row['transitions'] for row in result['exposure'].values()) == 384
            checkpoints = {}
            for update in (1, 2):
                path = training.output_dir / f'actor_update_{update}.pt'
                state = torch.load(path, map_location='cpu', weights_only=True)
                assert all(bool(torch.isfinite(value).all()) for value in state.values())
                checkpoints[str(update)] = hashlib.sha256(path.read_bytes()).hexdigest()
            report['runs'].append({'run': run, 'ppo_seed': SourceRunSeeds(run).ppo,
                'actor_initial_sha256': actor_initial, 'actor_final_sha256': _state_digest(runner.alg.actor),
                'critic_initial_sha256': critic_initial, 'critic_final_sha256': _state_digest(runner.alg.critic),
                'optimizer_state_entries': len(runner.alg.optimizer.state),
                'completed_updates': result['completed_updates'],
                'synthetic_transitions': result['synthetic_transitions'],
                'backward_checks': result['source_backward_checks'],
                'optimizer_steps_completed': result['source_optimizer_steps_completed'],
                'endings': runner.env.env.endings, 'checkpoint_sha256': checkpoints,
                'losses': [row['losses'] for row in result['update_receipts']]})
            _atomic_json(args.output_dir / 'summary.json', report)
            del training, runner
        assert len({row['actor_initial_sha256'] for row in report['runs']}) == 1
        assert len({row['critic_initial_sha256'] for row in report['runs']}) == 3
        assert report['input_sha256'] == _input_hashes(), 'production inputs changed during synthetic check'
        report.update(status='passed', synthetic_ppo_updates=6, synthetic_transitions=1152,
                      elapsed_seconds=time.monotonic() - started)
        if args.device != 'cpu':
            torch.cuda.synchronize()
            report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            report['gpu_name'] = torch.cuda.get_device_name()
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        _atomic_json(args.output_dir / 'summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    check(parser.parse_args())
