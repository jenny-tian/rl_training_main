"""Close a registered collection window and align committed replay with raw steps."""

import argparse
from pathlib import Path

import numpy as np
import torch

from rl_training.online_adaptation.interaction_budget import InteractionBudget
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from scripts.reinforcement_learning.online_adaptation.confirm_value_history import read_json
from scripts.reinforcement_learning.online_adaptation.run_adaptation_session import write_session_summary


def close_window(root, expected_trials):
    with InteractionBudget(Path(root)/'ledger') as ledger:
        if (ledger.state['active'] is not None or ledger.state['pending'] is not None
                or ledger.state['uncertain_steps'] or len(ledger.state['trials']) != expected_trials):
            raise ValueError('collection window incomplete or uncertain; no replacement starts')
        if ledger.state['phase'] == 'adaptation':
            ledger.lock_evaluation()
        elif ledger.state['phase'] != 'evaluation':
            raise ValueError('unexpected collection phase')
        write_session_summary(Path(root), ledger)
        return dict(ledger.state['spent'])


def aligned_replay_fields(arrays, records, gamma):
    """Build n=1 expected fields from pre-action history and post-action targets."""
    n = len(records)
    h = arrays['history'].reshape(n, *arrays['history'].shape[-2:])
    next_frame = arrays['next_frame'].reshape(n, -1)
    terminal = arrays['learning_terminated'].reshape(n, 1)
    truncated = arrays['learning_truncated'].reshape(n, 1)
    bootstrap = arrays['bootstrap_mask'].reshape(n, 1)
    next_history = h.copy()
    next_history[((terminal | truncated).astype(bool) & ~bootstrap.astype(bool)).reshape(-1)] = 0
    next_history = np.concatenate((next_history[:, 1:], next_frame[:, None]), axis=1)
    result = {'history': h, 'next_history': next_history, 'next_frame': next_frame,
              'terminated': terminal, 'truncated': truncated, 'bootstrap_mask': bootstrap,
              'discount': np.full((n, 1), gamma, dtype=np.float32),
              'safety_mask': np.array([r['replay_safety_eligible'] for r in records], dtype=np.float32)[:, None]}
    for name, trace in {'current_frame':'current_frame', 'base_action':'base_action',
                        'residual_action':'executed_residual_action',
                        'requested_residual_action':'requested_residual_action', 'reward':'reward',
                        'next_base_action':'next_base_action', 'episode_id':'episode_id'}.items():
        result[name] = arrays[trace].reshape(n, -1)
    return result


def audit_terrain(root, protocol, seed):
    root = Path(root)
    state = read_json(root/'session_summary.json')
    manifest = read_json(root/'ledger/manifest.json')
    if (manifest['base']['sha256'] != protocol['base_sha256'] or manifest['terrain_seed'] != seed
            or manifest['learner_seed'] != protocol['learner_seed'] or manifest['algorithm'] != 'sac'):
        raise ValueError('wrong registered session')
    if sha256_file(manifest['base']['path']) != protocol['base_sha256']:
        raise ValueError('base weights changed')
    expected, total_steps, previous_updater, episodes = {}, 0, None, []
    hashes = {}
    for index, (trial_id, trial) in enumerate(state['trials'].items()):
        if trial['status'] != 'complete' or trial['reset_seed'] != protocol['reset_seeds'][index]:
            raise ValueError('incomplete or substituted trial')
        path = Path(trial['output'])
        summary = read_json(path/'episode_summary.json')
        trace = read_json(path/'trajectory/manifest.json')
        import json
        records = [json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        chunks = []
        for chunk in trace['chunks']:
            file = path/'trajectory'/chunk['file']
            hashes[str(file)] = sha256_file(file)
            with np.load(file) as data:
                chunks.append({key: np.array(data[key]) for key in data.files})
        arrays = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
        n = len(records)
        if n != summary['steps'] or n != trace['steps'] or n != trial['confirmed_steps'] or trace['status'] != 'complete':
            raise ValueError('trace/log/ledger mismatch')
        if not arrays['learning_terminated'][-1].item() or arrays['collector_cutoff'].any() or arrays['learning_truncated'].any():
            raise ValueError('registered full task episode missing; retain this failed audit')
        np.testing.assert_array_equal(arrays['step'].reshape(-1), np.arange(1, n+1))
        np.testing.assert_array_equal([r['step'] for r in records], np.arange(1, n+1))
        for name in ('requested_action', 'mean_action', 'executed_action'):
            np.testing.assert_allclose(arrays[name], arrays['base_action'], atol=1e-6, rtol=0)
        committed = np.array([r['replay_critic_rows_committed'] for r in records])
        if not np.isin(committed, [0, 1]).all():
            raise ValueError('single-env n=1 commits must be zero or one')
        eligible = np.array([r['replay_safety_eligible'] for r in records])
        np.testing.assert_array_equal(committed.astype(bool), eligible & ~arrays['paused'].reshape(-1).astype(bool))
        checkpoint = torch.load(path/'session.pt', map_location='cpu', weights_only=False)['experiment_state']
        agent, warmup = checkpoint['agent'], checkpoint['target_warmup']
        updater, replay = agent['updater'], agent['replay']
        total_steps += n
        if (warmup['required_steps'] != protocol['target_warmup_steps'] or warmup['collected_steps'] != total_steps
                or warmup['critic_updates'] or updater['version'] or summary['update_attempts']
                or any(updater[k]['state'] for k in ('actor_optimizer', 'critic_optimizer', 'alpha_optimizer'))):
            raise ValueError('collection unexpectedly changed learner state')
        if previous_updater is not None:
            for module in ('actor', 'critic', 'target_critic'):
                for key, value in previous_updater[module].items():
                    assert torch.equal(value, updater[module][key]), (module, key)
            assert torch.equal(previous_updater['log_alpha'], updater['log_alpha'])
        previous_updater = updater
        fields = aligned_replay_fields(arrays, records, checkpoint['training_contract']['sac']['gamma'])
        for key, value in fields.items():
            expected.setdefault(key, []).append(value[committed.astype(bool)])
            wanted = np.concatenate(expected[key]).astype(replay[key].numpy().dtype)
            actual = replay[key][:replay['size']].numpy()
            np.testing.assert_array_equal(actual, wanted, err_msg=f'{trial_id}: {key}')
        if replay['cursor'] != replay['size'] or summary['replay_transitions_written'] != int(committed.sum()):
            raise ValueError('unexpected ring wrap or summary commit total')
        for file in (path/'session.pt', path/'metrics.jsonl', path/'episode_summary.json', path/'trajectory/manifest.json'):
            hashes[str(file)] = sha256_file(file)
        episodes.append({'trial_id':trial_id, 'reset_seed':trial['reset_seed'], 'steps':n,
                         'stop_reason':summary['task_stop_reason'], 'replay_rows_added':int(committed.sum()),
                         'terminal_retained':bool(committed[-1]), 'terminal_bootstrap_mask':float(arrays['bootstrap_mask'][-1].item()),
                         'terminal_safety_eligible':bool(eligible[-1]),
                         'control_timeouts':int(arrays['control_timed_out'].sum()),
                         'max_control_compute_ms':float(arrays['control_compute_ms'].max()),
                         'cumulative_replay_rows':replay['size']})
    return {'episodes':episodes, 'actual_control_transitions':total_steps,
            'replay_rows':replay['size'], 'terminal_rows':int(replay['terminated'][:replay['size']].sum()),
            'truncated_rows':int(replay['truncated'][:replay['size']].sum()),
            'zero_request_rows':int((replay['requested_residual_action'][:replay['size']]==0).all(dim=1).sum()),
            'aligned_fields':list(expected), 'input_sha256':hashes,
            'source_sha256':manifest['worker_configuration']['source_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if protocol['schema'] != 'lite3_sac_boundary_collection_v1':
        raise ValueError('unsupported boundary protocol')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'protocol_sha256':sha256_file(args.protocol), 'terrains':{}}
    write_json_atomic(output/'report.json', report)
    for seed in protocol['terrains']:
        root = Path(protocol['output'])/f'terrain-{seed}'
        spent = close_window(root, protocol['max_trials_per_terrain'])
        assert close_window(root, protocol['max_trials_per_terrain']) == spent
        result = audit_terrain(root, protocol, seed)
        if result['actual_control_transitions'] != sum(spent.values()):
            raise ValueError('collection cost mismatch')
        report['terrains'][str(seed)] = result
        write_json_atomic(output/'report.json', report)
        print(seed, result['episodes'], flush=True)
    total = sum(t['actual_control_transitions'] for t in report['terrains'].values())
    assert total <= protocol['max_actual_control_transitions']
    report.update(status='complete', actual_control_transitions=total,
                  optimizer_updates=0, repeated_lock_added_steps=0, route_qualified=False)
    write_json_atomic(output/'report.json', report)


if __name__ == '__main__':
    main()
