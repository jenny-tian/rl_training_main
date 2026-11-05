"""Audit archived full-episode evidence without simulator access or ledger writes."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'docs/lite3-source-full-episode-comparison-20261105'
RAW = REPORT / 'raw/runtime'
sys.path.insert(0, str(ROOT / 'source/rl_training'))
from rl_training.source_scan.training_budget_ledger import BudgetLedger, canonical_budget_path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def inventory(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}

def main():
    manifest, result, execution = (read(RAW / name) for name in ('manifest.json', 'result.json', 'execution.json'))
    summary = read(REPORT / 'deliverables/summary.json')
    comparison = read(REPORT / 'deliverables/comparison.json')
    common = canonical_budget_path(ROOT).parent.parent
    persistent = common / 'source-delivery-evidence/lite3-training-delivery-20261104-v1/run/fixed_lr_full_episode_20261105'
    copied, original = inventory(RAW), inventory(persistent)
    assert copied and copied == original, 'raw archive differs from persistent runtime'
    revision = manifest['source_revision']
    inputs, current_source_differences = [], []
    for name, expected in manifest['input_sha256'].items():
        path = Path(name)
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else None
        source = relative is not None and relative.parts[0] in ('scripts', 'source')
        if source:
            blob = subprocess.check_output(['git', 'show', revision + ':' + str(relative)], cwd=ROOT)
            actual = hashlib.sha256(blob).hexdigest()
            method = 'git_blob_at_runtime_revision'
            current = digest(path)
            if current != expected:
                current_source_differences.append(dict(path=str(relative),runtime_sha256=expected,current_sha256=current))
        else:
            actual, method = digest(path), 'current_immutable_input_file'
        assert actual == expected, 'runtime input mismatch: ' + name
        inputs.append(dict(path=name,sha256=actual,verification_method=method))
    expected_change = 'source/rl_training/rl_training/source_scan/first_episode_comparison.py'
    assert [r['path'] for r in current_source_differences] == [expected_change]
    ledger_path = canonical_budget_path(ROOT)
    ledger_hash = digest(ledger_path)
    assert ledger_hash == manifest['budget_before'] == execution['budget_before'] == execution['budget_after']
    ledger = BudgetLedger(ledger_path).read()
    charge = sum(e['reserved_transitions'] for e in ledger['entries'])
    assert charge == 271104
    assert result['status'] == 'starting' and len(result['cases']) == 2 and 'comparison' not in result
    assert execution['deadline_failure'] == 'comparison_finalize actor=None exceeded 30s'
    assert not execution['process_group_had_remaining_children']
    assert execution['elapsed_s'] < manifest['wall_limit_s']
    assert summary['runtime_status'] == 'starting' and summary['comparison_computed_offline']
    assert summary['status'] == 'behavior_regressed' and comparison['passed'] is False
    assert summary['execution'] == execution and summary['checks'] == comparison['checks']
    assert result['cases'][0]['initial'] == result['cases'][1]['initial']
    baseline = ROOT / 'docs/lite3-source-reward-live-verification-20261105/raw/runtime'
    prior = ROOT / 'docs/lite3-source-fixed-low-lr-trial-20261105/raw/runtime'
    assert result['cases'][0]['initial'] == read(baseline / 'result.json')['cases'][0]['initial']
    records, total, ends = {}, 0, {}
    for case in result['cases']:
        label = case['label']
        reference = baseline / 'old_mean.npz' if label == 'original' else prior / 'actor_update_2.npz'
        with np.load(RAW / (label + '.npz')) as a, np.load(reference) as ref:
            done = a['done']
            assert done.shape == (case['controls'], 120) and done.any(axis=0).all()
            ends[label] = done.argmax(axis=0) + 1
            assert ends[label].tolist() == [r['steps'] for r in case['first_episodes']]
            assert max(ends[label]) == case['controls'] <= 2000
            assert all(np.isfinite(a[key]).all() for key in a.files)
            prefixes = {key: np.array_equal(a[key][:400, :, :9] if key == 'state' else a[key][:400], ref[key])
                        for key in ('state', 'columns', 'done', 'rewards')}
            assert all(prefixes.values()) and prefixes == case['prefix_matches_short_trial']
            assert np.all(a['gate'] == 1)
        assert case['actor_parameters_unchanged'] and case['input_actions_unchanged']
        assert case['reward_reconstruction_max_abs'] < 1e-5
        assert case['totals'] == summary[label] == comparison[label]
        with (REPORT / ('deliverables/' + label + '_episodes.csv')).open() as stream:
            exported = list(csv.DictReader(stream))
        assert len(exported) == 120
        assert [int(row['steps']) for row in exported] == ends[label].tolist()
        transitions = case['controls'] * 120
        assert transitions == case['diagnostic_transitions']
        total += transitions
        records[label] = dict(controls=case['controls'],first_episodes=120,
            first_endings_match_export=True,prefix_checks=prefixes,totals=case['totals'],
            actor_parameters_unchanged=True,diagnostic_transitions=transitions)
    paired = int(np.maximum(0,np.minimum(ends['original'],ends['candidate'])-100).sum())
    assert paired == summary['paired_controls'] == comparison['paired_controls'] == 141647
    assert total == execution['diagnostic_transitions'] == execution['attempted_diagnostic_transitions'] == 412440
    assert total == result['diagnostic_transitions'] == result['attempted_diagnostic_transitions']
    assert total <= manifest['diagnostic_transition_cap']
    assert all(r[k] == 0 for r in (manifest,result,execution) for k in ('training_transitions','ppo_updates'))
    failures = read(REPORT / 'deliverables/failure_analysis.json')
    assert [r['env_id'] for r in failures['new_failures']] == summary['new_failure_environment_ids'] == [24,31,69,73,75]
    fields = {name:i for i,name in enumerate(result['state_names'])}
    for label in ('original','candidate'):
        with np.load(RAW / (label + '.npz')) as archive:
            state = archive['state']
        for row in failures['new_failures']:
            i, step = row['env_id'], row['comparison_control']
            assert step == ends['candidate'][i] and step <= ends[label][i]
            assert row[label+'_lateral_m'] == float(state[step-1,i,fields['lateral_m']])
            assert row[label+'_progress_m'] == float(state[step-1,i,fields['progress_m']])
    plan = read(REPORT / 'deliverables/next_stage_plan.json')
    assert not plan['user_approved'] and not plan['simulator_started'] and not plan['budget_reserved']
    assert digest(ROOT / plan['actor']['path']) == plan['actor']['sha256']
    assert (ROOT / plan['prefix_validation']['reference']).is_file()
    assert plan['acceptance_gates']['minimum_per_condition_completed'] == [
        row['original_completed'] for row in comparison['conditions']]
    offline_files = ['scripts/tools/source_full_episode_summary.py','scripts/tools/source_full_episode_verify.py',
                     expected_change,'tests/source_scan/test_first_episode_comparison.py']
    report = dict(status='verified_with_preserved_runtime_finalize_failure',
        runtime_source_revision=revision,runtime_archive_matches_persistent=True,
        runtime_archive_files=copied,runtime_inputs=inputs,
        current_source_differences=current_source_differences,
        offline_source_sha256={name:digest(ROOT/name) for name in offline_files},
        offline_provenance='Derived artifacts use these source contents; the delivery Git commit fixes their revision. Runtime inputs are checked against the original runtime revision.',
        runtime_failure_preserved=execution['deadline_failure'],runtime_result_status=result['status'],
        comparison_computed_offline=True,behavioral_gates_passed=False,cases=records,
        actual_diagnostic_transitions=total,paired_transitions_per_actor=paired,
        new_training_transitions=0,new_ppo_updates=0,training_charge=charge,ledger_sha256=ledger_hash,
        ledger_charges_by_run={str(run):BudgetLedger.charged(run,ledger) for run in (1,2,3)},
        next_stage_pending_consent=True)
    target = REPORT / 'raw/artifact_verification.json'
    target.write_text(json.dumps(report,indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ('status','actual_diagnostic_transitions','paired_transitions_per_actor',
        'training_charge','ledger_sha256','runtime_failure_preserved','next_stage_pending_consent')},indent=2))

if __name__ == '__main__':
    main()
