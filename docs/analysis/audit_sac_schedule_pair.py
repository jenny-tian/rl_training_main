"""Audit the preregistered pair using existing frozen-state and replay checks."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from docs.analysis.audit_sac_huber_coverage import audit_session, read, trace_arrays


def warmup_arrays(session):
    state = read(session/'session_summary.json')
    selected = {key:[] for key in ('root_state','executed_action','reward','learning_terminated')}
    for trial in state['trials'].values():
        if trial['phase']!='adaptation' or trial['status']!='complete':
            continue
        path = Path(trial['output'])
        import json
        rows = [json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        mask = np.array([r['target_warmup_collection'] for r in rows], dtype=bool)
        if mask.any():
            _, arrays, _ = trace_arrays(path)
            for key in selected:
                selected[key].append(arrays[key][mask])
    return {key:np.concatenate(value) for key,value in selected.items()} if all(selected.values()) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    assert p['schema']=='lite3_sac_actor_schedule_pair_v1'
    root, output = Path(p['output']), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'protocol_sha256':sha256_file(args.protocol), 'terrains':{}}
    write_json_atomic(output/'report.json', report)
    total = 0
    for seed in p['terrains']:
        execution = read(root/f'execution-{seed}.json')
        assert execution['status']!='running'
        terrain = {'execution_status':execution['status'], 'arms':{}, 'paired_phases':{}}
        configurations = {}
        warmups = {}
        for arm, rate in p['arms'].items():
            session = root/f'{arm}-{seed}'
            manifest = read(session/'ledger/manifest.json')
            config = dict(manifest['worker_configuration']['arguments'])
            assert config.pop('actor_updates_per_transition')==rate
            configurations[arm] = config
            result = audit_session(session, p, 'candidate', seed)
            state = read(session/'session_summary.json')
            assert state['active'] is None and state['pending'] is None
            assert result['warmup_gain_counts'] in ({'0.5':1000}, {})
            rejections = Counter()
            for trial in result['trials']:
                if trial['status']!='complete':
                    continue
                path = Path(state['trials'][trial['trial_id']]['output'])
                summary = read(path/'episode_summary.json')
                for key in ('update_attempts','rollbacks','safety_rollbacks','performance_rollbacks','actor_version','parameter_changed'):
                    trial[key] = summary[key]
                rejections.update(summary['update_rejections'])
            result['update_rejections'] = dict(rejections)
            result['post_warmup_critic_updates'] = result['critic_updates']-result['warmup_records']
            total += sum(result['spent'].values())
            terrain['arms'][arm] = result
            warmups[arm] = warmup_arrays(session)
        assert configurations['reference']==configurations['actor025'], 'more than the actor quota changed'
        first, second = warmups['reference'], warmups['actor025']
        if first is not None and second is not None:
            terrain['warmup_parity'] = {key:{'same_shape':first[key].shape==second[key].shape,
                'exact_equal':np.array_equal(first[key],second[key]),
                'max_abs':float(np.max(np.abs(first[key].astype(float)-second[key].astype(float))))
                          if first[key].shape==second[key].shape else None} for key in first}
        for phase in ('evaluation','retention'):
            pairs = {arm:[t for t in result['trials'] if t['phase']==phase and t['status']=='complete']
                     for arm,result in terrain['arms'].items()}
            if not all(len(v)==1 for v in pairs.values()):
                terrain['paired_phases'][phase]={'available':False}
                continue
            a,b = pairs['reference'][0],pairs['actor025'][0]
            assert a['reset_seed']==b['reset_seed']==p[phase+'_reset_seed']
            np.testing.assert_allclose(a['initial_root_state'],b['initial_root_state'],atol=1e-6,rtol=0)
            terrain['paired_phases'][phase]={'available':True,'initial_state_matched':True,
                'progress_delta_actor025_minus_reference':b['progress_m']-a['progress_m']}
        report['terrains'][str(seed)] = terrain
        write_json_atomic(output/'report.json', report)
        print(seed, {arm:{key:value for key,value in result.items() if key not in ('trials','input_sha256')}
                     for arm,result in terrain['arms'].items()}, flush=True)
    assert total<=p['max_actual_control_transitions']
    report.update(status='complete', actual_control_transitions=total, route_qualified=False)
    write_json_atomic(output/'report.json', report)


if __name__=='__main__':
    main()
