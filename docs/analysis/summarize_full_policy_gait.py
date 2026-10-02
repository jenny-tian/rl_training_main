"""Describe frozen foot motion; unequal trajectories do not establish a causal gait effect."""
import argparse
import json
from pathlib import Path
import numpy as np
from docs.analysis.audit_sac_huber_coverage import read
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def summarize(rows, step_dt):
    if not rows:
        return {'samples': 0}
    contacts = np.asarray([r['foot_contact'] for r in rows], dtype=float) > .5
    heights = np.asarray([r['foot_lift_from_contact_m'] for r in rows], dtype=float)
    steps = np.asarray([r['step'] for r in rows])
    assert contacts.shape == heights.shape and heights.shape[1] == 4 and np.isfinite(heights).all()
    intervals = []
    for foot in range(4):
        onsets = np.flatnonzero(contacts[1:,foot] & ~contacts[:-1,foot]) + 1
        # Do not infer stride time across omitted samples or gaps in the selected region.
        periods = [float((steps[b]-steps[a])*step_dt) for a,b in zip(onsets[:-1],onsets[1:])
                   if steps[b]-steps[a] == b-a]
        intervals.append({'count':len(periods),'median_seconds':float(np.median(periods)) if periods else None})
    swing = ~contacts
    return {'samples':len(rows),'foot_order':['FL','FR','HL','HR'],
            'contact_duty_factor':contacts.mean(0).tolist(),
            'swing_lift_mean_m':[float(heights[swing[:,i],i].mean()) if swing[:,i].any() else None for i in range(4)],
            'swing_lift_p95_m':[float(np.percentile(heights[swing[:,i],i],95)) if swing[:,i].any() else None for i in range(4)],
            'stance_onset_intervals':intervals,
            'diagonal_contact_fraction':float(((contacts[:,0]&contacts[:,3]&~contacts[:,1]&~contacts[:,2]) |
                                               (contacts[:,1]&contacts[:,2]&~contacts[:,0]&~contacts[:,3])).mean()),
            'all_feet_airborne_fraction':float((~contacts.any(1)).mean())}


def session_specs(p):
    if p['schema']=='lite3_route_transfer_v1':
        return [(terrain,role,run['learning_seed'],f"{role}-{run['learning_seed']}-{terrain}")
                for terrain in p['terrain_seeds'] for run in p['policies'] for role in ('candidate','base')]
    return [(terrain,role,p['seed'],f'{role}-{terrain}')
            for terrain in p['evaluation_terrains'] for role in ('candidate','base')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();p=read(args.protocol);root=Path(p['output'])
    audit_path=root/('final_audit.json' if p['schema']=='lite3_route_transfer_v1' else 'evaluation_audit.json')
    audit=read(audit_path)
    assert audit['status']=='complete' and audit['protocol_sha256']==sha256_file(args.protocol)
    report={'evaluation_audit_sha256':sha256_file(audit_path),'trials':[],
            'limit':'Descriptive only: each trajectory has its own duration, position and failure history. No gait qualification or causal mirror/route effect is inferred.'}
    for terrain,role,seed,directory in session_specs(p):
        state=read(root/directory/'session_summary.json')
        for trial in state['trials'].values():
            if trial['status']!='complete':continue
            path=Path(trial['output']);summary=read(path/'episode_summary.json');trace=read(path/'trajectory/manifest.json')
            rows=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
            assert len(rows)==summary['steps']==trace['steps']
            if trial['phase']=='retention':selected=[r for r in rows if r['step']*trace['metadata']['step_dt']>5.]
            else:selected=[r for r in rows if summary['challenge_start_x']<=r['base_x']<=summary['challenge_end_x']]
            report['trials'].append({'terrain':terrain,'role':role,'paired_learning_seed':seed,'trial':trial['id'],'phase':trial['phase'],
                'terminal':summary['task_stop_reason'],'metrics_sha256':sha256_file(path/'metrics.jsonl'),
                'motion':summarize(selected,trace['metadata']['step_dt'])})
    write_json_atomic(Path(args.output),report)
    print('Described',len(report['trials']),'frozen trajectories')


if __name__=='__main__':main()
