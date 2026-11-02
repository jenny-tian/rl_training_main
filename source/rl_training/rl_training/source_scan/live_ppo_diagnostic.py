"""Durable live-PPO evidence, usable with CPU fixtures without importing Isaac."""
from __future__ import annotations

from pathlib import Path
import torch

from .training_runner import _atomic_torch_save
from .moving_diagnostic import digest_file

SCHEMA = 'lite3_source_live_ppo_diagnostic_v1'
TIMEOUT_PROBES = {12: [0], 36: [1, 4, 7]}
CONTROL_STEPS = 48


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('nonfinite PPO evidence tensor')
        return value.detach().cpu().clone()
    if hasattr(value, 'items'):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(cpu_tree(item) for item in value)
    return value


def plain(value):
    value = cpu_tree(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    return value


class LivePPOJournal:
    def __init__(self, writer, report, *, app_running, step_check=None, probes=None):
        self.writer, self.report = writer, report
        self.directory = writer.paths['result'].parent
        self.app_running = app_running
        self.step_check = step_check
        self.probes = TIMEOUT_PROBES if probes is None else probes
        self.last_observations = None

    def save(self, name, payload):
        path = self.directory / name
        if path.exists():
            raise FileExistsError('PPO evidence snapshot already exists')
        _atomic_torch_save(path, cpu_tree(payload))
        self.report.setdefault('evidence_sha256', {})[name] = digest_file(path)
        self.writer.persist(self.report)

    def model(self, training, update):
        algorithm = training.algorithm
        self.save(f'model_{update}.pt', {'actor': algorithm.actor.state_dict(),
            'critic': algorithm.critic.state_dict(), 'optimizer': algorithm.optimizer.state_dict(),
            'completed_updates': update, 'optimizer_steps': algorithm.source_optimizer_steps_completed})

    def initial(self, training, observations):
        self.model(training, 0)

    def before_step(self, training, step):
        if not self.app_running():
            raise RuntimeError('Isaac application stopped before diagnostic completion')
        forced = self.probes.get(step, [])
        if forced:
            training.env.unwrapped.episode_length_buf[forced] = 1999
        self.columns_before = training.env.unwrapped.scene.terrain.terrain_types.clone()

    def after_step(self, training, step, old, observations, actions, rewards, dones, extras):
        if self.step_check is not None:
            self.step_check(step)
        env, algorithm = training.env, training.algorithm
        index = algorithm.storage.step - 1
        terminal = extras['source_terminal']
        terminal_values = None
        if terminal is not None:
            with torch.no_grad():
                terminal_values = algorithm.critic(terminal['observations']).squeeze(-1)
        trace = env.last_trace
        self.writer.append_step({'step': step, 'forced_ids': self.probes.get(step, []),
            'done': plain(dones.long()), 'time_outs': plain(extras['time_outs']),
            'terminal': plain(terminal), 'terminal_values': plain(terminal_values),
            'observations': plain(old), 'next_observations': plain(observations),
            'raw_action': plain(actions), 'base_action': plain(trace.base_action),
            'requested_action': plain(trace.requested_action), 'executed_action': plain(trace.executed_action),
            'projected': plain(trace.decision.projected), 'paused': plain(trace.decision.paused),
            'rewards': plain(rewards.flatten()), 'stored_rewards': plain(algorithm.storage.rewards[index].flatten()),
            'old_log_prob': plain(algorithm.storage.actions_log_prob[index].flatten()),
            'critic_values': plain(algorithm.storage.values[index].flatten()),
            'episode_counters': plain(env.unwrapped.episode_length_buf),
            'terrain_columns_before': plain(self.columns_before),
            'terrain_columns_after': plain(env.unwrapped.scene.terrain.terrain_types)})
        self.last_observations = observations
        self.report.update(simulation_control_steps=step, total_transitions=step*8,
                           exposure=env.exposure.report())
        self.writer.persist(self.report)

    def before_update(self, training, update):
        storage = training.algorithm.storage
        with torch.no_grad():
            last_values = training.algorithm.critic(self.last_observations)
        self.save(f'rollout_{update}.pt', {name: getattr(storage, name) for name in (
            'observations', 'actions', 'actions_log_prob', 'values', 'rewards', 'dones',
            'returns', 'advantages', 'distribution_params')} | {'last_values': last_values})

    def after_update(self, training, update, receipt):
        self.model(training, update)
        self.report.update(ppo_updates=update, training_snapshot=dict(training.report))
        self.writer.persist(self.report)
