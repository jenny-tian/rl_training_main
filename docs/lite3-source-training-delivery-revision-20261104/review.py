"""Verify actual archived outcomes and actor; never grant the failed speed gate."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main(folder):
    folder=folder.resolve();checked=0
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);p=(folder/name).resolve()
        assert p.is_relative_to(folder) and digest(p)==expected,name
        checked+=1
    load=lambda name:json.loads((folder/name).read_text())
    for row in load('raw/runtime_artifacts.json')['files']:
        p=folder/row['repository_path']
        assert p.stat().st_size==row['size_bytes'] and digest(p)==row['sha256'],str(p)
    prefix=load('raw/runtime/prefix_review/report.json')
    assert prefix['scope_qualified'] and prefix['accepted'] is False and prefix['G3_qualified'] is False
    assert prefix['complete_controls']==1884 and prefix['physics_substeps']==7536
    assert prefix['reset_batches']==77 and prefix['deferred_final_recovery_ids']==[]
    assert prefix['observed_cells']==list(range(15)) and prefix['ground_positive_environment_ids']==list(range(120))
    g4=load('raw/runtime/g4/acceptance.json')
    assert g4['accepted'] and g4['scheduled_reset_batches']==6 and g4['scheduled_environment_resets']==120
    assert g4['measured_reset_count']==9 and g4['measured_natural_reset_batches']==3
    assert g4['measured_collection_transitions_per_second']>=600 and g4['reset_p95_s']<=1
    numeric=load('raw/runtime/numerics/acceptance.json')
    assert numeric['accepted'] is False
    assert numeric['reasons']==['end-to-end throughput below approved 600 transitions/s target']
    assert numeric['completed_updates']==90 and numeric['real_transitions']==259200
    assert 0<numeric['measured_end_to_end_transitions_per_second']<600
    for job in ('prefix_review','g4','numerics'):
        execution=load('raw/runtime/'+job+'/execution.json')
        assert execution['returncode']==0 and execution['process_group_empty'] and not execution['timed_out']
    report=load('raw/runtime/numerics/training/run_report.json')
    assert report['status']=='completed' and report['completed_updates']==90
    for name in ('source_backward_checks','source_optimizer_steps_started','source_optimizer_steps_completed'):
        assert report[name]==1800
    assert sum(v['transitions'] for v in report['exposure'].values())==259200
    assert len(report['exposure'])==15 and all(v['transitions']>0 for v in report['exposure'].values())
    curves=[json.loads(line) for line in (folder/'raw/runtime/numerics/training/curves.jsonl').read_text().splitlines()]
    assert [r['update'] for r in curves]==list(range(1,91))
    assert curves[-1]['training_transitions']==259200
    import math
    assert all(math.isfinite(v) for r in curves for v in r['losses'].values())
    budget=load('raw/runtime/numerics/budget_receipt.json')
    assert budget['state']=='failed' and budget['reserved_transitions']==259200 and budget['completed_updates']==90
    assert budget['baseline_charged_transitions']==384 and budget['automatic_refund'] is False
    assert digest(folder/'raw/production_ledger_after.json')=='f171d9fdbe380a20c827453c52bb30ac30c8f6ed431f45c01f796625be303435'
    assert set(load('raw/slots.json')['jobs'])=={'g2','g3','g4','numerics'}
    assert load('raw/numerics_preflight_transport_failure.json')['numerics_slot_consumed'] is False
    assert load('raw/numerics_transport_resolution.json')['gpu_training_launches_before_recovery']==0
    checkpoint=folder/'deliverables/actor_update_90.pt'
    expected='14d11c8b57392c3ac0ac1e08b26de1cc1254ee059d1f653172433df9a05bdf42'
    assert digest(checkpoint)==expected and checkpoint.stat().st_size==760457
    root=folder.parents[1];sys.path[:0]=[str(root/'source/rl_training'),str(root/'scripts/tools')]
    import torch
    from rl_training.source_scan.frozen_actor import FrozenSourceActor
    from source_live_ppo_validation import state_digest
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);actor=FrozenSourceActor(state)
    with torch.inference_mode():value=actor(torch.zeros(1,45))
    assert value.shape==(1,12) and torch.isfinite(value).all()
    assert state_digest(state)==report['update_receipts'][-1]['actor_sha256']!=report['initial_actor_sha256']
    assert report['update_receipts'][-1]['critic_sha256']!=report['initial_critic_sha256']
    delivery=load('deliverables/delivery_summary.json')
    assert delivery['numeric_trial_accepted'] is False and delivery['paired_comparison_status'].startswith('not_started')
    return dict(archive_integrity_verified=True,hashed_files=checked,actual_GPU_jobs=2,
        actual_training_GPU_launches=1,CPU_limited_scope_qualified=True,G4_accepted=True,
        actual_PPO_updates=90,actual_backward_and_Adam_steps=1800,new_training_transitions=259200,
        cumulative_charged_transitions=259584,checkpoint_cpu_load_and_inference=True,
        numeric_trial_accepted=False,numeric_only_rejection='throughput below 600 transitions/s',
        comparison_started=False,formal_selection_qualified=False)


if __name__=='__main__':print(json.dumps(main(Path(__file__).resolve().parent),indent=2))
