"""Audit registered SAC learning and paired frozen/base phases without refitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def read(path):
    return json.loads(Path(path).read_text())


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal_state(left, right)
    else:
        assert a == b


def trace_arrays(path):
    manifest = read(path/'trajectory/manifest.json')
    chunks = []
    hashes = {}
    for chunk in manifest['chunks']:
        file = path/'trajectory'/chunk['file']
        hashes[str(file)] = sha256_file(file)
        with np.load(file) as data:
            chunks.append({key:np.array(data[key]) for key in data.files})
    return manifest, {key:np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}, hashes


def audit_session(root, protocol, role, terrain):
    state, manifest = read(root/'session_summary.json'), read(root/'ledger/manifest.json')
    assert manifest['base']['sha256'] == protocol['base_sha256']
    assert sha256_file(manifest['base']['path']) == protocol['base_sha256']
    assert manifest['learner_seed'] == protocol['learner_seed'] and manifest['terrain_seed'] == terrain
    assert manifest['worker_configuration']['arguments']['critic_huber_delta'] == 1.
    result = {'spent':state['spent'], 'uncertain_steps':state['uncertain_steps'], 'trials':[],
              'source_sha256':manifest['worker_configuration']['source_sha256'],
              'adaptation_early_stopped':state['adaptation_early_stopped'],
              'adaptation_failure_reason':state['adaptation_failure_reason'], 'warmup_records':0,
              'warmup_gain_counts':{}, 'actor_updates':0, 'critic_updates':0,
              'warmup_control_steps':0, 'post_warmup_control_steps':0,
              'post_warmup_nonzero_request_steps':0, 'post_warmup_nonzero_replay_rows':0,
              'stochastic_control_steps':0, 'input_sha256':{}}
    previous = None
    for trial_id, trial in state['trials'].items():
        path = Path(trial['output'])
        item = {'trial_id':trial_id, 'phase':trial['phase'], 'status':trial['status'],
                'reset_seed':trial['reset_seed'], 'charged_steps':trial['charged_steps']}
        result['trials'].append(item)
        if trial['status'] != 'complete':
            item['reason'] = trial.get('reason')
            continue
        summary = read(path/'episode_summary.json')
        records = [json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        trace, arrays, hashes = trace_arrays(path)
        result['input_sha256'].update(hashes)
        n = len(records)
        assert n == summary['steps'] == trace['steps'] == trial['confirmed_steps']
        assert trace['status'] == 'complete'
        np.testing.assert_array_equal(arrays['step'].reshape(-1), np.arange(1, n+1))
        warmup = np.array([r['target_warmup_collection'] for r in records], dtype=bool)
        requested = arrays['requested_residual_action'].reshape(n, -1)
        nonzero = np.any(requested != 0, axis=1)
        commits = np.array([r['replay_critic_rows_committed'] for r in records])
        if warmup.any():
            for key in ('requested_action', 'mean_action', 'executed_action'):
                np.testing.assert_allclose(arrays[key][warmup], arrays['base_action'][warmup], atol=1e-6, rtol=0)
        snapshot = (torch.load(path/'session.pt', map_location='cpu', weights_only=False)['experiment_state']
                    if trial['phase'] == 'adaptation' else None)
        agent = None if snapshot is None else snapshot['agent']
        item.update(steps=n, stop_reason=summary['task_stop_reason'], completion=summary.get('completion_reached'),
                    progress_m=summary['max_x']-summary['initial_x'], fell=summary['fell'],
                    safety_violation=summary['safety_violation'], actor_updates=summary['actor_updates'],
                    critic_updates=summary['critic_updates'], replay_rows_written=summary['replay_transitions_written'],
                    warmup=summary['target_warmup'], nonzero_request_steps=int(nonzero.sum()),
                    request_rms=float(np.sqrt(np.mean(requested.astype(float)**2))),
                    nonzero_executed_steps=int(np.any(arrays['executed_residual_action'].reshape(n,-1)!=0,axis=1).sum()),
                    control_timeouts=int(arrays['control_timed_out'].sum()),
                    max_control_compute_ms=float(arrays['control_compute_ms'].max()),
                    action_gains=sorted({r['action_residual_gain'] for r in records}),
                    post_warmup_actor_replay_rows=sum(r['replay_actor_rows_committed'] for r, w in zip(records, warmup) if not w),
                    update_rejections=summary['update_rejections'],
                    replay_terminal_rows=(int(agent['replay']['terminated'][:agent['replay']['size']].sum())
                                          if agent is not None else None))
        if trial['phase'] == 'adaptation':
            result['warmup_control_steps'] += int(warmup.sum())
            result['post_warmup_control_steps'] += int((~warmup).sum())
            result['post_warmup_nonzero_request_steps'] += int((nonzero & ~warmup).sum())
            result['post_warmup_nonzero_replay_rows'] += int(commits[nonzero & ~warmup].sum())
            result['stochastic_control_steps'] += sum(int(r['stochastic_control_active']) for r in records)
            result['actor_updates'] += summary['actor_updates']
            result['critic_updates'] += summary['critic_updates']
            for update in summary['warmup_critic_diagnostics']:
                assert update['accepted'] and not update['actor_updated']
                gain = str(update['residual_gain'])
                result['warmup_gain_counts'][gain] = result['warmup_gain_counts'].get(gain, 0)+1
                result['warmup_records'] += 1
            previous = snapshot
        else:
            # Frozen workers deliberately do not serialize a mutable learner checkpoint.
            assert not (path/'session.pt').exists() and summary['experiment_checkpoint_out_sha256'] is None
            before_actor = torch.load(path/'residual_before.pt', map_location='cpu', weights_only=False)
            after_actor = torch.load(path/'residual_after.pt', map_location='cpu', weights_only=False)
            equal_state(before_actor, after_actor)
            assert summary['update_attempts'] == summary['critic_updates'] == summary['actor_updates'] == 0
            assert summary['replay_transitions_written'] == summary['actor_transitions_committed'] == 0
            assert np.all(commits == 0) and not warmup.any()
            assert item['action_gains'] == ([.5] if role=='candidate' else [0.])
            if role == 'candidate':
                assert previous is not None
                equal_state(previous['agent']['updater']['actor'], before_actor)
                assert all(r['replay_size']==previous['agent']['replay']['size'] for r in records)
                assert all(r['adaptation_replay_size']==previous['agent']['adaptation_replay']['size'] for r in records)
                assert summary['experiment_checkpoint_in_sha256'] == trial['input_checkpoint']['sha256']
                assert sha256_file(trial['input_checkpoint']['path']) == trial['input_checkpoint']['sha256']
            else:
                for key in ('requested_action', 'mean_action', 'executed_action'):
                    np.testing.assert_allclose(arrays[key], arrays['base_action'], atol=1e-6, rtol=0)
            item['initial_root_state'] = trace['metadata']['initial_root_state']
            item['frozen_actor_state_equal'] = True
            item['frozen_learner_checkpoint_written'] = False
        if trial['phase'] == 'retention':
            item['flat_retention'] = summary['flat_retention']
        evidence = ([path/'session.pt'] if snapshot is not None else [path/'residual_before.pt', path/'residual_after.pt'])
        for file in (*evidence, path/'episode_summary.json', path/'metrics.jsonl', path/'trajectory/manifest.json'):
            result['input_sha256'][str(file)] = sha256_file(file)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    assert protocol['schema'] == 'lite3_sac_huber_coverage_v1'
    root, output = Path(protocol['output']), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'protocol_sha256':sha256_file(args.protocol), 'terrains':{}}
    write_json_atomic(output/'report.json', report)
    total = 0
    for terrain in protocol['terrains']:
        execution = read(root/f'execution-{terrain}.json')
        assert execution['status'] != 'running'
        results = {role:audit_session(root/f'{role}-{terrain}', protocol, role, terrain) for role in ('candidate','base')}
        for role, result in results.items():
            total += sum(result['spent'].values())
        pairs = {}
        for phase in ('evaluation', 'retention'):
            trials = {role:[t for t in result['trials'] if t['phase']==phase and t['status']=='complete']
                      for role, result in results.items()}
            if all(len(v)==1 for v in trials.values()):
                candidate, base = trials['candidate'][0], trials['base'][0]
                assert candidate['reset_seed'] == base['reset_seed']
                np.testing.assert_allclose(candidate['initial_root_state'], base['initial_root_state'], atol=1e-6, rtol=0)
                pairs[phase] = {'reset_seed':candidate['reset_seed'], 'initial_state_matched':True}
        results['paired_phases'] = pairs
        results['execution_status'] = execution['status']
        report['terrains'][str(terrain)] = results
        write_json_atomic(output/'report.json', report)
        print(terrain, {role:{k:v for k,v in result.items() if k not in ('trials','input_sha256')}
                        for role,result in results.items() if role in ('candidate', 'base')}, flush=True)
    assert total <= protocol['max_actual_control_transitions']
    report.update(status='complete', actual_control_transitions=total, route_qualified=False)
    write_json_atomic(output/'report.json', report)


if __name__ == '__main__':
    main()
