"""Audit final PPO policy against its own immutable checkpoint and matched base starts."""
import argparse
from pathlib import Path

import numpy as np

from docs.analysis.audit_c0_headroom import audit_child, check_pairs
from docs.analysis.audit_learning_replication import retention_pass
from docs.analysis.audit_sac_huber_coverage import read
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def outcomes(rows):
    valid = [r for r in rows if r['status'] == 'complete']
    assert all(np.isfinite(r['progress_m']) for r in valid)
    return {'completed_trials': len(valid), 'completions': sum(r['completion'] for r in valid),
            'falls': sum(r['fell'] for r in valid),
            'progress_m': [r['progress_m'] for r in valid],
            'safety_violations': sum(r['safety_violation'] for r in valid)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    root = Path(p['output'])
    lock = read(root/'evaluation_lock.json')
    assert lock['protocol_sha256'] == sha256_file(args.protocol)
    assert lock['training_report_sha256'] == sha256_file(root/'train/report.json')
    assert lock['history_audit_sha256'] == sha256_file(root/'action_history_audit.json')
    report = {'status': 'running', 'protocol_sha256': sha256_file(args.protocol), 'terrains': {},
              'training_controls': p['training_control_transitions'],
              'retired_incompatible_training_controls': p['prior_incompatible_training_steps'],
              'evaluation_controls': 0, 'uncertain_steps': 0, 'goal_complete': False,
              'freeze_evidence_limit': 'RSL model files unchanged and frozen execution path; saved before/after tensors cover only the zero residual actor.'}
    sources = set()
    for terrain in p['evaluation_terrains']:
        execution = read(root/f'execution-{terrain}.json')
        assert execution['status'] != 'running'
        item = {'roles': {}, 'outcomes': {}, 'execution_status': execution['status']}
        phases = {}
        for role in ('candidate', 'base'):
            checkpoint = lock[role]
            assert sha256_file(checkpoint['path']) == checkpoint['sha256']
            # The shared audit is used only for its zero-residual frozen path.
            # Each full policy must match its own checkpoint, never the other arm's.
            expected = dict(p, base_sha256=checkpoint['sha256'])
            directory = root/f'{role}-{terrain}'
            child = audit_child(directory, expected, terrain, p['evaluation_seed'], None,
                                p['evaluation_trials']+p['retention_trials'],
                                expected_phases={'evaluation': p['evaluation_trials'], 'retention': p['retention_trials']})
            manifest = read(directory/'ledger/manifest.json')
            assert Path(manifest['base']['path']).resolve() == Path(checkpoint['path']).resolve()
            assert manifest['worker_configuration']['arguments']['stochastic_control'] is False
            for trial in read(directory/'session_summary.json')['trials'].values():
                if trial['status'] == 'complete':
                    summary = read(Path(trial['output'])/'episode_summary.json')
                    assert summary['mode'] == 'base'
                    assert summary['critic_transitions_committed'] == summary['warmup_critic_updates'] == 0
                    assert summary['parameter_changed'] is False
            for row in child['trials']:
                if row['status'] == 'complete':
                    assert row['requested_residual_rms'] == row['executed_residual_rms'] == 0
            sources.add(child['source_sha256'])
            report['evaluation_controls'] += child['spent']
            report['uncertain_steps'] += child['uncertain_steps']
            item['roles'][role] = child
            phases[role] = {phase: [r for r in child['trials'] if r['phase'] == phase]
                            for phase in ('evaluation', 'retention')}
            item['outcomes'][role] = outcomes(phases[role]['evaluation'])
        for phase in ('evaluation', 'retention'):
            check_pairs(phases['candidate'][phase], phases['base'][phase])
        complete = all(c['all_expected_trials_complete'] for c in item['roles'].values())
        flats = [r for arm in phases.values() for r in arm['retention']]
        full_flats = complete and all(r['flat_retention']['sample_count'] == p['retention_steps']
            and r['flat_retention']['steady_sample_count'] == 750 for r in flats)
        item['retention_pass'] = bool(full_flats and retention_pass(phases['candidate']['retention'], phases['base']['retention']))
        item['paired_completion_gain'] = item['outcomes']['candidate']['completions'] - item['outcomes']['base']['completions']
        item['complete'] = complete
        report['terrains'][str(terrain)] = item
    assert len(sources) == 1
    assert report['evaluation_controls'] <= p['max_evaluation_control_transitions']
    report['status'] = 'complete'
    report['pilot_supports_further_work'] = all(t['complete'] and t['retention_pass'] and t['paired_completion_gain'] > 0 for t in report['terrains'].values())
    report['total_controls_including_retired_training'] = report['training_controls'] + report['retired_incompatible_training_controls'] + report['evaluation_controls']
    report['prior_compatible_training_controls'] = p.get('prior_compatible_training_steps', 0)
    report['prior_compatible_evaluation_controls'] = p.get('prior_compatible_evaluation_steps', 0)
    report['full_policy_program_controls'] = report['total_controls_including_retired_training'] + report['prior_compatible_training_controls'] + report['prior_compatible_evaluation_controls']
    write_json_atomic(Path(args.output), report)
    print({k: v for k, v in report.items() if k != 'terrains'})


if __name__ == '__main__':
    main()
