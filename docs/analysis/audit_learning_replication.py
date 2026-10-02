"""Audit preregistered replication, paired frozen outcomes and descriptive gait."""
import argparse
import json
from pathlib import Path

import numpy as np

from docs.analysis.audit_sac_huber_coverage import audit_session, read
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def retention_pass(candidate, base):
    if not candidate or len(candidate) != len(base):
        return False
    if not all(t['flat_retention']['comparison_eligible'] and not t['fell'] and not t['safety_violation']
               for t in candidate+base):
        return False
    def mean(rows, key):
        return float(np.mean([t['flat_retention'][key] for t in rows]))
    c, b = (mean(rows, 'steady_velocity_xy_mae') for rows in (candidate, base))
    return bool(c <= max(1.1*b, b+.005) and all(
        mean(candidate, f'steady_velocity_{axis}_mae') <= mean(base, f'steady_velocity_{axis}_mae')+.01
        for axis in ('x', 'y')))


def qualify(terrains):
    gains = {k: [s['candidate_completion_rate']-s['base_completion_rate'] for s in t['sessions']]
             for k, t in terrains.items()}
    complete = all(t['complete'] for t in terrains.values())
    retention = all(s['retention_pass'] for t in terrains.values() for s in t['sessions'])
    safe = all(s['evaluation_safe'] for t in terrains.values() for s in t['sessions'])
    gain_ok = any(np.mean(v) >= .2-1e-12 and all(x > 0 for x in v) for v in gains.values())
    no_regression = all(np.mean(v) >= -.2-1e-12 for v in gains.values())
    engineering = complete and retention and safe and gain_ok and no_regression
    stable = engineering and all(
        sum(s['candidate_completions'] for s in t['sessions']) >= 9
        and all(s['candidate_completions'] >= 4 for s in t['sessions']) for t in terrains.values())
    return {'engineering_qualified': bool(engineering), 'stable_development_qualified': bool(stable),
            'goal_complete': False, 'limit': 'New gait and unseen development terrain confirmation still required.'}


def gait(root, trial):
    path = root/trial['trial_id']/'metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # Descriptive only: trajectories and failure lengths differ; no causal gait claim.
    contact = np.asarray([r['foot_contact'] for r in rows], dtype=float)
    height = np.asarray([r['foot_lift_from_contact_m'] for r in rows], dtype=float)
    swing = contact < .5
    return {'samples': len(rows), 'foot_contact_duty_factors': contact.mean(0).tolist(),
            'swing_height_mean_m': [float(height[swing[:,i],i].mean()) if swing[:,i].any() else None for i in range(4)],
            'request_rms': trial['request_rms']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    assert p['schema'] == 'lite3_learning_replication_v1'
    root, out = Path(p['output']), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'protocol_sha256': sha256_file(args.protocol), 'terrains': {},
              'actual_control_transitions': 0, 'uncertain_steps': 0}
    for terrain in p['terrains']:
        execution = read(root/f'execution-{terrain}.json')
        assert execution['status'] != 'running'
        tr = {'sessions': [], 'complete': execution['status'] == 'complete'}
        for seed in p['learner_seeds']:
            cfg = dict(p, learner_seed=seed)
            results = {role: audit_session(root/f'{role}-{seed}-{terrain}', cfg, role, terrain)
                       for role in ('candidate', 'base')}
            item = {'seed': seed, 'roles': results}
            phases = {phase: {role: [t for t in result['trials'] if t['phase']==phase and t['status']=='complete']
                      for role, result in results.items()} for phase in ('evaluation', 'retention')}
            for phase, roles in phases.items():
                expected = p['evaluation_trials'] if phase == 'evaluation' else p['retention_trials']
                tr['complete'] &= all(len(rows) == expected for rows in roles.values())
                for c, b in zip(roles['candidate'], roles['base']):
                    assert c['reset_seed'] == b['reset_seed']
                    np.testing.assert_allclose(c['initial_root_state'], b['initial_root_state'], atol=1e-6, rtol=0)
            for role, result in results.items():
                report['actual_control_transitions'] += sum(result['spent'].values())
                report['uncertain_steps'] += result['uncertain_steps']
                evals = phases['evaluation'][role]
                item[f'{role}_completions'] = sum(t['stop_reason']=='completion' for t in evals)
                item[f'{role}_completion_rate'] = item[f'{role}_completions']/p['evaluation_trials']
                item[f'{role}_gait'] = [gait(root/f'{role}-{seed}-{terrain}', t) for t in evals]
            item['retention_pass'] = retention_pass(phases['retention']['candidate'], phases['retention']['base'])
            item['evaluation_safe'] = all(not t['safety_violation'] for rows in phases['evaluation'].values() for t in rows)
            tr['sessions'].append(item)
        report['terrains'][str(terrain)] = tr
        write_json_atomic(out/'report.json', report)
    assert report['actual_control_transitions'] <= p['max_actual_control_transitions']
    report.update(qualify(report['terrains']))
    report['status'] = 'complete'
    write_json_atomic(out/'report.json', report)
    print({k:v for k,v in report.items() if k != 'terrains'})


if __name__ == '__main__':
    main()
