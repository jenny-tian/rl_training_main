"""Reconstruct archived paired metrics and hashes without running simulation."""
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(folder):
    folder=folder.resolve();checked=0
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);path=(folder/name).resolve()
        assert path.is_relative_to(folder) and digest(path)==expected,name
        checked+=1
    load=lambda name:json.loads((folder/name).read_text())
    for item in load('raw/runtime_artifacts.json')['files']:
        path=folder/item['repository_path']
        assert path.stat().st_size==item['size_bytes'] and digest(path)==item['sha256']
    acceptance=load('raw/runtime/acceptance.json');execution=load('raw/runtime/execution.json')
    assert acceptance['accepted'] and not acceptance['reasons']
    assert execution['returncode']==0 and execution['process_group_empty']
    assert not execution['timed_out'] and not execution['monitor_errors']
    report=load('raw/runtime/result.json');old,new=report['actors']
    assert report['status']=='passed' and report['simulation_control_steps']==report['finite_controls']==4000
    assert report['completed_diagnostic_transitions']==480000
    assert old['label']=='old' and new['label']=='new' and old['initial']==new['initial']
    steps=[json.loads(line) for line in (folder/'raw/runtime/result.steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in steps]==list(range(1,4001))
    assert all(r['finite'] and r['actor']==('old' if i<2000 else 'new') and r['actor_step']==i%2000+1 for i,r in enumerate(steps))
    with (folder/'deliverables/paired_episodes.csv').open(newline='') as stream:episodes=list(csv.DictReader(stream))
    with (folder/'deliverables/paired_conditions.csv').open(newline='') as stream:conditions=list(csv.DictReader(stream))
    assert len(episodes)==120 and len(conditions)==15
    summary=load('deliverables/summary.json')
    metrics=('forward_progress_m','lateral_peak_m','roll_peak_rad','pitch_peak_rad','velocity_error_rms_mps')
    for side,actor in (('old',old),('new',new)):
        actual=actor['first_episodes']
        assert len(actual)==120 and [r['env_id'] for r in actual]==list(range(120))
        for row,exported in zip(actual,episodes):
            assert row['env_id']==int(exported['env_id']) and row['column']==int(exported['column'])
            assert row['completed']==bool(int(exported[side+'_completed']))
            for metric in metrics:assert row[metric]==float(exported[side+'_'+metric])
        assert sum(r['completed'] for r in actual)==summary[side+'_completed']
        for column,condition in enumerate(conditions):
            group=[r for r in actual if r['column']==column]
            assert int(condition['pairs'])==len(group)
            assert int(condition[side+'_completed'])==sum(r['completed'] for r in group)
            for metric in metrics:
                assert float(condition[side+'_'+metric])==sum(r[metric] for r in group)/len(group)
        for metric in ('completion_fraction',*metrics):
            mean=sum(float(c[side+'_'+metric]) for c in conditions)/15
            assert abs(mean-summary['equal_condition_weighted'][metric][side])<1e-12
    assert summary['old_completed']==87 and summary['new_completed']==0
    assert all(r['termination_reasons']==['source_training_timeout'] for r in new['first_episodes'])
    for prefix in ('original_slots','budget','numeric_acceptance'):
        assert digest(folder/f'raw/{prefix}_before.json')==digest(folder/f'raw/{prefix}_after.json')
    assert not load('raw/numeric_acceptance_after.json')['accepted']
    assert set(load('raw/compare_repair_slots.json')['jobs'])=={'compare'}
    assert summary['new_training_transitions']==summary['ppo_updates']==0
    return dict(archive_integrity_verified=True,hashed_files=checked,paired_diagnostic_accepted=True,
        episode_pairs=120,conditions=15,diagnostic_transitions=480000,old_completed=87,new_completed=0,
        equal_condition_metrics_reconstructed=True,original_slots_and_budget_unchanged=True,
        process_group_empty=True,new_training_transitions=0,formal_selection_qualified=False)


if __name__=='__main__':print(json.dumps(main(Path(__file__).resolve().parent),indent=2))
