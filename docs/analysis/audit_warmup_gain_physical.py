"""Verify explicit warmup gain, checkpoint continuation, and frozen zero writes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from docs.analysis.audit_sac_huber_coverage import equal_state


def read(path):
    return json.loads(Path(path).read_text())


def audit(root, seed, protocol):
    root = Path(root)
    state, manifest = read(root/'session_summary.json'), read(root/'ledger/manifest.json')
    assert state['active'] is None and state['pending'] is None and state['uncertain_steps']==0
    assert len(state['trials'])==3 and state['spent']['adaptation']==512 and state['spent']['evaluation']==64
    assert manifest['terrain_seed']==seed and manifest['learner_seed']==protocol['learner_seed']
    assert sha256_file(manifest['base']['path'])==protocol['base_sha256']==manifest['base']['sha256']
    report = {'actual_control_transitions':sum(state['spent'].values()), 'trials':[], 'input_sha256':{},
              'source_sha256':manifest['worker_configuration']['source_sha256']}
    previous = None
    for index, (trial_id, trial) in enumerate(state['trials'].items()):
        assert trial['status']=='complete'
        path = Path(trial['output'])
        summary = read(path/'episode_summary.json')
        rows = [json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        trace = read(path/'trajectory/manifest.json')
        assert trace['status']=='complete'
        assert len(rows)==summary['steps']==trial['confirmed_steps']==trace['steps']==(256 if index<2 else 64)
        for chunk in trace['chunks']:
            file = path/'trajectory'/chunk['file']
            with np.load(file) as arrays:
                for key in ('requested_action', 'mean_action', 'executed_action'):
                    np.testing.assert_allclose(arrays[key], arrays['base_action'], atol=1e-6, rtol=0)
            report['input_sha256'][str(file)] = sha256_file(file)
        before_actor = torch.load(path/'residual_before.pt', map_location='cpu', weights_only=False)
        after_actor = torch.load(path/'residual_after.pt', map_location='cpu', weights_only=False)
        equal_state(before_actor, after_actor)
        record = {'trial_id':trial_id, 'phase':trial['phase'], 'steps':summary['steps'],
                  'stop_reason':summary['task_stop_reason'], 'warmup_before':summary['target_warmup_before'],
                  'warmup_after':summary['target_warmup'], 'critic_updates':summary['critic_updates'],
                  'actor_updates':summary['actor_updates'], 'replay_rows_written':summary['replay_transitions_written'],
                  'action_gains':sorted({r['action_residual_gain'] for r in rows}),
                  'max_control_compute_ms':max(r['control_compute_ms'] for r in rows),
                  'control_timeouts':sum('control_timeout' in r['safety_reasons'] for r in rows)}
        assert summary['actor_updates']==0
        updates = summary['warmup_critic_diagnostics']
        if index==0:
            assert len(updates)==1000 and all(u['accepted'] and not u['actor_updated'] for u in updates)
            assert {(u['residual_gain'], u['control_residual_gain']) for u in updates}=={(.5,0.)}
            assert all(r['target_warmup_collection'] for r in rows)
            record['warmup_target_gain'] = .5
            record['warmup_control_gain'] = 0.
        else:
            assert updates==[] and not any(r['target_warmup_collection'] for r in rows)
            assert summary['target_warmup_before']['critic_updates']==1000
        assert summary['target_warmup']['collected_steps']==256 and summary['target_warmup']['critic_updates']==1000
        if index<2:
            snapshot = torch.load(path/'session.pt', map_location='cpu', weights_only=False)['experiment_state']
            agent = snapshot['agent']
            assert not agent['updater']['actor_optimizer']['state'] and not agent['updater']['alpha_optimizer']['state']
            steps = {int(v['step']) for v in agent['updater']['critic_optimizer']['state'].values()}
            assert steps==({1000} if index==0 else {1128})
            assert agent['replay']['size']==256*(index+1)
            assert record['action_gains']==[0.]
            record['critic_optimizer_steps'] = sorted(steps)
            if previous is not None:
                equal_state(previous['agent']['updater']['actor'], agent['updater']['actor'])
                equal_state(previous['agent']['updater']['log_alpha'], agent['updater']['log_alpha'])
                assert trial['input_checkpoint']==state['trials']['trial-00000']['output_checkpoint']
            previous = snapshot
            report['input_sha256'][str(path/'session.pt')] = sha256_file(path/'session.pt')
        else:
            assert record['action_gains']==[.5]
            assert summary['update_attempts']==summary['critic_updates']==summary['replay_transitions_written']==0
            assert all(r['replay_critic_rows_committed']==r['replay_actor_rows_committed']==0 for r in rows)
            assert all(r['replay_size']==previous['agent']['replay']['size'] for r in rows)
            assert not (path/'session.pt').exists()
            equal_state(previous['agent']['updater']['actor'], before_actor)
            assert sha256_file(trial['input_checkpoint']['path'])==trial['input_checkpoint']['sha256']
        for file in (path/'episode_summary.json', path/'metrics.jsonl', path/'residual_before.pt', path/'residual_after.pt', path/'trajectory/manifest.json'):
            report['input_sha256'][str(file)] = sha256_file(file)
        report['trials'].append(record)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    assert protocol['schema']=='lite3_sac_warmup_gain_physical_v1'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'protocol_sha256':sha256_file(args.protocol), 'terrains':{}}
    write_json_atomic(output/'report.json', report)
    for seed in protocol['terrains']:
        result = audit(Path(protocol['output'])/f'terrain-{seed}', seed, protocol)
        repeat = read(Path(protocol['output'])/f'repeat-{seed}.json')
        assert repeat['added_steps']==repeat['added_trials']==0 and repeat['states_equal']
        result['repeat'] = repeat
        report['terrains'][str(seed)] = result
        write_json_atomic(output/'report.json', report)
        print(seed, result['trials'], flush=True)
    report.update(status='complete', actual_control_transitions=sum(t['actual_control_transitions'] for t in report['terrains'].values()),
                  fixed_gain_physical_verified=True, route_qualified=False)
    assert report['actual_control_transitions']<=protocol['max_actual_control_transitions']
    write_json_atomic(output/'report.json', report)


if __name__=='__main__':
    main()
