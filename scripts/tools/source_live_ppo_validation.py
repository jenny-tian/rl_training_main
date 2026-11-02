"""CPU review of live PPO tensors plus reused motion/contact/reset evidence."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import torch
import torch.nn.functional as F

from rl_training.source_scan.frozen_actor import FrozenSourceActor
from rl_training.source_scan.live_ppo_diagnostic import SCHEMA, TIMEOUT_PROBES
from rl_training.source_scan.moving_diagnostic import EXPECTED_TERMS, digest_file
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.smoke_artifacts import _strict_json, _matrix, artifact_paths
from source_eight_env_moving_validation import review_events, events_digest, require


def close(actual, expected, label):
    actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
    require(actual.shape == expected.shape and bool(torch.isfinite(actual).all())
            and bool(torch.isfinite(expected).all())
            and torch.allclose(actual.float(), expected.float(), atol=1e-4, rtol=1e-4), label)


def critic(state, observations):
    require(set(state) == {f'mlp.{i}.{suffix}' for i in (0, 2, 4, 6) for suffix in ('weight', 'bias')}, 'critic schema')
    value = observations
    for i in (0, 2, 4, 6):
        value = F.linear(value, state[f'mlp.{i}.weight'], state[f'mlp.{i}.bias'])
        if i != 6: value = F.elu(value)
    return value


def finite_tree(value):
    if isinstance(value, torch.Tensor): require(bool(torch.isfinite(value).all()), 'nonfinite model/optimizer')
    elif isinstance(value, dict):
        for item in value.values(): finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite_tree(item)
    elif isinstance(value, float): require(math.isfinite(value), 'nonfinite optimizer scalar')


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode()); digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def controls(rows, report):
    counters = [0]*8; previous = None; ordinary = 0
    exposure = {name: {'episode_assignments': 0, 'transitions': 0} for name in SOURCE_RESET_BLOCK_COUNTS}
    labels = tuple(exposure)
    for step, row in enumerate(rows, 1):
        require(row['step'] == step and row['attempt_id'] == report['attempt_id']
                and row['forced_ids'] == TIMEOUT_PROBES.get(step, []), 'step identity/probe')
        for name in ('raw_action', 'base_action', 'requested_action', 'executed_action'):
            require(_matrix(row[name], 8, 12), 'invalid action journal')
        require(row['requested_action'] == row['base_action'] and row['paused'] == [[False]]*8
                and _matrix(row['projected'], 8, 1, boolean=True), 'filter journal')
        for group, width in (('policy', 45), ('critic', 393)):
            require(_matrix(row['observations'][group], 8, width)
                    and _matrix(row['next_observations'][group], 8, width), 'observation journal')
            if previous is not None: require(previous['next_observations'][group] == row['observations'][group], 'observation sequence')
        require(len(row['done']) == 8 and all(type(x) is int and x in (0, 1) for x in row['done']), 'done mask')
        require(len(row['time_outs']) == 8 and all(type(x) is bool for x in row['time_outs']), 'timeout mask')
        before, after = row['terrain_columns_before'], row['terrain_columns_after']
        require(all(len(v) == 8 and all(type(x) is int and 0 <= x < 15 for x in v) for v in (before, after)), 'terrain columns')
        if previous is not None: require(before == previous['terrain_columns_after'], 'column sequence')
        ended = [i for i, done in enumerate(row['done']) if done]
        require(set(row['forced_ids']).issubset(ended), 'injected timeout missing')
        terminal = row['terminal']
        if ended:
            require(terminal is not None and terminal['ids'] == ended and set(terminal['terms']) == EXPECTED_TERMS, 'terminal IDs/terms')
            require(all(len(v) == len(ended) and all(type(x) is bool for x in v) for v in terminal['terms'].values()), 'terminal term shape')
            require(all(len(terminal[k]) == 8 and all(type(x) is bool for x in terminal[k])
                        for k in ('terminated', 'truncated')), 'terminal masks')
            require(_matrix(terminal['observations']['policy'], 8, 45)
                    and _matrix(terminal['observations']['critic'], 8, 393), 'terminal observation')
            for j, env_id in enumerate(ended):
                active = {name for name, values in terminal['terms'].items() if values[j]}
                require(active and terminal['terminated'][env_id] == bool(active-{'source_training_timeout'})
                        and terminal['truncated'][env_id] == ('source_training_timeout' in active), 'terminal flags')
                ordinary += bool(active-{'source_training_timeout', 'source_training_completion'})
            require(all(not terminal['terminated'][i] and not terminal['truncated'][i] for i in range(8) if i not in ended), 'non-ending terminal flag')
            require(row['time_outs'] == [t and not f for t, f in zip(terminal['truncated'], terminal['terminated'])], 'timeout bootstrap mask')
            require(all(terminal['truncated'][i] for i in row['forced_ids']), 'forced truncation missing')
        else: require(terminal is None and row['time_outs'] == [False]*8, 'unexpected terminal')
        for i in range(8):
            if step == 1: exposure[labels[before[i]]]['episode_assignments'] += 1
            exposure[labels[before[i]]]['transitions'] += 1
            length = (1999 if i in row['forced_ids'] else counters[i])+1
            if i in ended:
                require(terminal['steps'][ended.index(i)] == length, 'terminal episode counter')
                exposure[labels[after[i]]]['episode_assignments'] += 1; counters[i] = 0
            else:
                require(before[i] == after[i], 'unreset column changed'); counters[i] = length
        require(row['episode_counters'] == counters, 'episode counter sequence')
        previous = row
    require(exposure == report['exposure'], 'exposure ledger mismatch')
    return ordinary


def ppo_evidence(folder, report, rows, *, checkpoint, synthetic):
    names = {f'model_{i}.pt' for i in (0, 1, 2)} | {'rollout_1.pt', 'rollout_2.pt'}
    require(set(report['evidence_sha256']) == names, 'PPO snapshot inventory')
    evidence = {}
    for name, digest in report['evidence_sha256'].items():
        require(digest_file(folder/name) == digest, 'PPO snapshot hash mismatch')
        evidence[name] = torch.load(folder/name, map_location='cpu', weights_only=True)
        finite_tree(evidence[name])
    initial = evidence['model_0.pt']
    require(initial['optimizer']['state'] == {} and initial['optimizer_steps'] == 0, 'optimizer not fresh')
    require(torch.equal(initial['actor']['distribution.log_std_param'], torch.zeros(12)), 'initial std not one')
    if not synthetic:
        require(checkpoint is not None and digest_file(checkpoint) == report['checkpoint_sha256'], 'original checkpoint missing/changed')
        original = torch.load(checkpoint, map_location='cpu', weights_only=True)['actor_state_dict']
        require(all(torch.equal(value, original[key]) for key, value in initial['actor'].items()
                    if key != 'distribution.log_std_param'), 'initial actor mean changed')
    for update in (1, 2):
        model, after = evidence[f'model_{update-1}.pt'], evidence[f'model_{update}.pt']
        rollout = evidence[f'rollout_{update}.pt']
        part = rows[(update-1)*24:update*24]
        actor = FrozenSourceActor(model['actor'])
        for group in ('policy', 'critic'):
            expected = torch.tensor([row['observations'][group] for row in part])
            close(rollout['observations'][group], expected, 'stored observations changed')
        actions = torch.tensor([row['raw_action'] for row in part])
        close(rollout['actions'], actions, 'stored sampled action changed')
        with torch.no_grad():
            means = actor(rollout['observations']['policy'].reshape(-1, 45)).reshape(24, 8, 12)
            std = model['actor']['distribution.log_std_param'].exp().expand_as(means)
            expected_logp = torch.distributions.Normal(means, std).log_prob(actions).sum(-1, keepdim=True)
            values = critic(model['critic'], rollout['observations']['critic'])
        close(rollout['actions_log_prob'], expected_logp, 'sampled action log probability mismatch')
        close(rollout['values'], values, 'critic value mismatch')
        require(len(rollout['distribution_params']) == 2, 'distribution parameter inventory')
        for actual, expected in zip(rollout['distribution_params'], (means, std)):
            close(actual, expected, 'old distribution parameters changed')
        for index, row in enumerate(part):
            close(rollout['actions_log_prob'][index].flatten(), row['old_log_prob'], 'step log probability')
            close(rollout['values'][index].flatten(), row['critic_values'], 'step critic value')
            reward = torch.tensor(row['rewards'])
            if row['terminal'] is not None:
                terminal_values = critic(model['critic'], torch.tensor(row['terminal']['observations']['critic'])).flatten()
                close(row['terminal_values'], terminal_values, 'terminal critic value')
                reward = reward + .99 * torch.where(torch.tensor(row['time_outs']), terminal_values, 0.)
            close(row['stored_rewards'], reward, 'terminal bootstrap reward')
            close(rollout['rewards'][index].flatten(), reward, 'stored bootstrap reward')
            close(rollout['dones'][index].flatten(), row['done'], 'stored done mask')
        last = critic(model['critic'], torch.tensor(part[-1]['next_observations']['critic']))
        close(rollout['last_values'], last, 'last critic value')
        returns = torch.zeros_like(rollout['returns']); advantage = 0
        for step in reversed(range(24)):
            next_value = rollout['last_values'] if step == 23 else rollout['values'][step+1]
            continuing = 1. - rollout['dones'][step].float()
            delta = rollout['rewards'][step] + continuing * .99 * next_value - rollout['values'][step]
            advantage = delta + continuing * .99 * .95 * advantage
            returns[step] = advantage + rollout['values'][step]
        close(rollout['returns'], returns, 'GAE returns mismatch')
        advantages = returns - rollout['values']
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
        close(rollout['advantages'], advantages, 'normalized advantages mismatch')
        require(after['completed_updates'] == update and after['optimizer_steps'] == update*20, 'optimizer update counts')
        require(len(after['optimizer']['state']) == 17 and all(int(value['step']) == update*20
                for value in after['optimizer']['state'].values()), 'Adam parameter step counters')
        require(state_digest(model['actor']) != state_digest(after['actor'])
                and state_digest(model['critic']) != state_digest(after['critic']), 'actor/critic unchanged')
        saved = torch.load(folder/f'training/actor_update_{update}.pt', map_location='cpu', weights_only=True)
        require(state_digest(saved) == state_digest(after['actor']), 'saved policy differs from live state')
        receipt = report['training_snapshot']['update_receipts'][update-1]
        require(receipt['actor_sha256'] == state_digest(after['actor']) and all(
            type(value) in (int, float) and torch.isfinite(torch.tensor(value)) for value in receipt['losses'].values()), 'update receipt')


def validate(folder, plan, execution, *, checkpoint=None, allow_synthetic=False):
    folder = Path(folder)
    result = {'schema': 'lite3_source_live_ppo_acceptance_v1', 'accepted': False, 'reasons': [],
              'training_ready': False, 'g3_g4_qualified': False}
    try:
        require(execution.get('returncode') == 0 and execution.get('process_group_empty') is True
                and not execution.get('timed_out') and not execution.get('monitor_errors'), 'process/monitor failure')
        paths = artifact_paths(folder/'result.json')
        report = _strict_json(paths['result'].read_text())
        rows = [_strict_json(line) for line in paths['steps'].read_text().splitlines()]
        cleanup = _strict_json(paths['cleanup'].read_text())
        synthetic = report.get('synthetic_cpu_fixture') is True
        require(not synthetic or allow_synthetic, 'synthetic evidence cannot qualify a live run')
        expected = {'schema': SCHEMA, 'status': 'passed', 'attempt_id': plan['attempt_id'],
            'base_revision': plan['base_revision'], 'checkpoint_sha256': plan['checkpoint_sha256'],
            'num_envs': 8, 'shared_bank_rows': 1, 'simulation_control_steps': 48,
            'total_transitions': 384, 'ppo_updates': 2, 'steps_written': 48, 'critic_width': 393,
            'diagnostic_only': True, 'training_ready': False, 'g3_g4_qualified': False,
            'phase': 'result_before_shutdown'}
        for name, value in expected.items():
            require(type(report.get(name)) is type(value) and report[name] == value, 'report field: '+name)
        require(len(report['body_names']) == len(set(report['body_names'])) == 17, 'body inventory')
        require(len(rows) == 48 and report['steps_file'] == paths['steps'].name
                and report['steps_sha256'] == digest_file(paths['steps']), 'control journal incomplete/changed')
        require(cleanup['attempt_id'] == plan['attempt_id'] and cleanup['environment'] == 'returned'
                and cleanup['application'] in ('closing', 'returned') and cleanup['persistence_errors'] == [], 'cleanup incomplete')
        training = _strict_json((folder/'training/run_report.json').read_text())
        require(training == report['training_snapshot'] and training['status'] == 'completed', 'training report mismatch')
        for name in ('attempted_transition_upper_bound', 'returned_transitions', 'stored_transitions'):
            require(training[name] == 384, 'incomplete training interactions')
        require(training['completed_updates'] == 2 and training['source_optimizer_steps_started']
                == training['source_optimizer_steps_completed'] == training['source_backward_checks'] == 40,
                'incomplete PPO/optimizer updates')
        require(training['data_kind'] == ('synthetic' if synthetic else 'source_simulation')
                and training['real_source_ppo_updates'] == (0 if synthetic else 2)
                and training['real_source_training_transitions'] == (0 if synthetic else 384), 'data provenance')
        ordinary = controls(rows, report)
        require(training['exposure'] == report['exposure'], 'training exposure mismatch')
        ppo_evidence(folder, report, rows, checkpoint=checkpoint, synthetic=synthetic)
        events = (folder/'result.json').with_suffix('.events.jsonl')
        expected_name = events.name
        if not events.exists(): events = events.with_suffix('.jsonl.gz')
        require(report['events_file'] == expected_name and report['events_sha256'] == events_digest(events), 'event journal changed')
        motion = review_events(events, report, rows, control_steps=48)
        result.update(accepted=True, control_steps=48, diagnostic_transitions=384, ppo_updates=2,
            ordinary_failure_episodes=ordinary, event_review=motion, synthetic_cpu_fixture=synthetic,
            scope='bounded live PPO integration' if not synthetic else 'synthetic CPU acceptance fixture only')
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, AttributeError, AssertionError, EOFError) as error:
        result['reasons'].append(type(error).__name__ + ': ' + str(error))
    return result
