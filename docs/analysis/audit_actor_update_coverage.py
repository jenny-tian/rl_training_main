"""Bound actor updates using existing per-trial samples and recorded credit resets."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def audit(root):
    root = Path(root)
    old = json.loads((root/'audit_validated/report.json').read_text())
    report = {'source_report_sha256':sha256_file(root/'audit_validated/report.json'), 'terrains':{},
              'new_control_steps':0, 'new_optimizer_updates':0,
              'interpretation':'per-trial credit ceiling; other gates can only reduce actual updates'}
    for seed in (101, 107):
        session = root/f'candidate-{seed}'
        state = json.loads((session/'session_summary.json').read_text())
        manifest = json.loads((session/'ledger/manifest.json').read_text())
        rate = Fraction(str(manifest['worker_configuration']['arguments']['actor_updates_per_transition']))
        evidence = old['terrains'][str(seed)]['candidate']['input_sha256']
        trials = []
        for trial_id, trial in state['trials'].items():
            if trial['phase'] != 'adaptation':
                continue
            path = Path(trial['output'])
            for file in (path/'metrics.jsonl', path/'episode_summary.json'):
                if sha256_file(file) != evidence[str(file)]:
                    raise ValueError('original coverage evidence changed')
            rows = [json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
            summary = json.loads((path/'episode_summary.json').read_text())
            post = [row for row in rows if not row['target_warmup_collection']]
            eligible = sum(row['replay_actor_rows_committed'] for row in post)
            upper = int(eligible * rate)
            if summary['actor_updates'] > upper or summary['update_budget_remaining']['actor'] != 0:
                raise ValueError('zero incoming credit/per-trial bound assumption does not hold')
            trials.append({'trial_id':trial_id, 'post_warmup_steps':len(post),
                           'actor_replay_rows':eligible, 'credit_upper_bound':float(eligible*rate),
                           'actor_update_upper_bound':upper, 'actor_updates':summary['actor_updates'],
                           'final_actor_credit':summary['update_budget_remaining']['actor']})
        total_rows = sum(t['actor_replay_rows'] for t in trials)
        report['terrains'][str(seed)] = {'actor_updates_per_transition':float(rate), 'trials':trials,
            'actor_replay_rows':total_rows, 'per_trial_upper_bound':sum(t['actor_update_upper_bound'] for t in trials),
            'actual_actor_updates':sum(t['actor_updates'] for t in trials),
            'pooled_credit_floor_without_trial_resets':int(total_rows*rate),
            'nominal_critic_updates_per_actor_update':float(Fraction(str(manifest['worker_configuration']['arguments']['critic_updates_per_transition']))/rate)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('preserve the existing audit')
    report = audit(args.root)
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
