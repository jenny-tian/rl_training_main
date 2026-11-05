"""Verify immutable update1 evaluation inputs, raw copies, and accounting."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'docs/lite3-source-update1-full-episode-20261105'
BASELINE = ROOT/'docs/lite3-source-full-episode-comparison-20261105'
sys.path.insert(0,str(ROOT/'source/rl_training'))
from rl_training.source_scan.training_budget_ledger import BudgetLedger,canonical_budget_path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inventory(path):
    return {str(p.relative_to(path)):digest(p) for p in sorted(path.rglob('*')) if p.is_file()}


def main():
    raw = REPORT/'raw/runtime'
    manifest,result,execution = (read(raw/name) for name in ('manifest.json','result.json','execution.json'))
    summary = read(REPORT/'deliverables/summary.json')
    budget = canonical_budget_path(ROOT)
    persistent = budget.parent.parent/'source-delivery-evidence/lite3-training-delivery-20261104-v1/run/fixed_lr_update1_full_episode_20261105'
    copied = inventory(raw)
    assert copied and copied==inventory(persistent)
    revision = manifest['source_revision']
    inputs = []
    for name,expected in manifest['input_sha256'].items():
        path = Path(name)
        assert digest(path)==expected, 'input changed after run: '+name
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else None
        method = 'immutable_file'
        if relative is not None and relative.parts[0] in ('scripts','source'):
            blob = subprocess.check_output(['git','show',revision+':'+str(relative)],cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest()==expected
            method = 'file_and_git_blob_at_runtime_revision'
        inputs.append(dict(path=name,sha256=expected,verification_method=method))
    for line in (BASELINE/'SHA256SUMS').read_text().splitlines():
        expected,name = line.split('  ',1)
        assert digest(BASELINE/name)==expected, 'previous stage archive changed: '+name
    assert execution['exit_code']==0 and execution['deadline_failure'] is None
    assert not execution['process_group_had_remaining_children']
    assert execution['elapsed_s']<600
    assert result['status']=='passed' and summary['status']=='passed'
    assert summary['execution']==execution and summary['offline_equals_online']
    assert summary['checks']==result['comparison']['checks'] and all(summary['checks'].values())
    assert all(record[key]==0 for record in (manifest,result,execution) for key in ('training_transitions','ppo_updates'))
    assert len(result['cases'])==1
    case = result['cases'][0]
    old = read(BASELINE/'raw/runtime/result.json')
    assert case['initial']==old['cases'][0]['initial']==old['cases'][1]['initial']
    assert case['controls']==1731 and case['totals']['episodes']==120 and case['totals']['completed']==87
    assert all(case['prefix_matches_short_trial'].values())
    assert case['actor_parameters_unchanged'] and case['input_actions_unchanged']
    assert case['reward_reconstruction_max_abs']<1e-5
    assert case['totals']==summary['actors']['update1']
    with np.load(raw/'update1.npz') as a:
        assert all(np.isfinite(a[key]).all() for key in a.files)
        done = a['done']
        assert done.shape==(1731,120) and done.any(axis=0).all()
        ends = done.argmax(axis=0)+1
        assert ends.tolist()==[row['steps'] for row in case['first_episodes']]
        assert int(ends.max())==1731
        reference = ROOT/'docs/lite3-source-fixed-low-lr-trial-20261105/raw/runtime/actor_update_1.npz'
        with np.load(reference) as b:
            assert all(np.array_equal(a[key][:400,:,:9] if key=='state' else a[key][:400],b[key])
                       for key in ('state','columns','done','rewards'))
        assert np.all(a['gate']==1)
    total = case['controls']*120
    assert total==207720==execution['diagnostic_transitions']==execution['attempted_diagnostic_transitions']
    assert total==case['diagnostic_transitions']==result['diagnostic_transitions']==result['attempted_diagnostic_transitions']
    assert total<=manifest['diagnostic_transition_cap']==240000
    old_ends = np.array([row['steps'] for row in old['cases'][0]['first_episodes']])
    update2_ends = np.array([row['steps'] for row in old['cases'][1]['first_episodes']])
    pair_n = int(np.maximum(0,np.minimum(old_ends,ends)-100).sum())
    common_n = int(np.maximum(0,np.minimum(np.minimum(old_ends,ends),update2_ends)-100).sum())
    assert pair_n==summary['paired_original_update1_transitions']==140808
    assert common_n==summary['common_three_actor_transitions']==140021
    assert not summary['changed_original_update1_outcome_environment_ids']
    ledger = BudgetLedger(budget).read()
    charge = sum(row['reserved_transitions'] for row in ledger['entries'])
    assert charge==271104==summary['training_charge']
    ledger_sha = digest(budget)
    assert ledger_sha==manifest['budget_before']==execution['budget_before']==execution['budget_after']
    assert digest(raw/'budget_before.json')==digest(raw/'budget_after.json')==ledger_sha
    phases = [json.loads(line) for line in (raw/'phases.jsonl').read_text().splitlines()]
    times = {r['phase']:r['started_mono'] for r in phases}
    finalize_s = times['application_close']-times['comparison_finalize']
    assert 0<=finalize_s<30
    plan = read(REPORT/'deliverables/next_stage_plan.json')
    assert not plan['user_approved'] and not plan['simulator_started'] and not plan['training_started']
    source_names = ['scripts/tools/source_update1_full_episode.py','scripts/tools/source_update1_full_episode_summary.py',
        'scripts/tools/source_update1_full_episode_verify.py','source/rl_training/rl_training/source_scan/update1_episode_comparison.py',
        'tests/source_scan/test_update1_episode_comparison.py']
    report = dict(status='verified',runtime_source_revision=revision,
        runtime_archive_matches_persistent=True,runtime_archive_files=copied,runtime_inputs=inputs,
        offline_source_sha256={name:digest(ROOT/name) for name in source_names},
        original_and_update2_archives_unchanged=True,first_episode_count=120,
        actual_diagnostic_transitions=total,original_update1_paired_transitions=pair_n,
        common_three_actor_transitions=common_n,online_comparison_finalize_s=finalize_s,
        offline_summary_equals_online=True,actor_parameters_unchanged=True,prefix_matches=True,
        new_training_transitions=0,new_ppo_updates=0,training_charge=charge,ledger_sha256=ledger_sha,
        next_stage_pending_consent=True)
    (REPORT/'raw/artifact_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','actual_diagnostic_transitions','original_update1_paired_transitions',
        'common_three_actor_transitions','online_comparison_finalize_s','training_charge','ledger_sha256')},indent=2))


if __name__=='__main__':
    main()
