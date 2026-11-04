"""CPU verification of the rejected G2 attempt; never reruns physics or changes scope."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def check(condition,message):
    if not condition:raise ValueError(message)


def analyze(raw,source_root):
    sys.path[:0]=[str(source_root/'source/rl_training'),str(source_root/'scripts/tools')]
    import torch
    from rl_training.source_scan.moving_diagnostic import restore_tensor
    from source_parallel_reset_lifecycle import check_transition
    read=lambda name:json.loads((raw/name).read_text())
    plan=read('run_manifest.json');report=read('result.json')
    events=[json.loads(line) for line in (raw/'events.jsonl').read_text().splitlines()]
    counts=Counter(v['phase'] for v in events)
    check(dict(counts)=={name:1 for name in ('scale_initial','contact_runtime','sparse_contact_topology',
        'scale_reset_attempt','scale_reset_after','contact_error','contact_summary')},'unexpected partial event phases')
    attempt=next(v for v in events if v['phase']=='scale_reset_attempt');after_event=next(v for v in events if v['phase']=='scale_reset_after')
    check(attempt['ids']==after_event['ids']==[0] and attempt['sim_step']==after_event['sim_step']==0,'failure boundary differs')
    before={k:restore_tensor(v) for k,v in attempt['before'].items()};after={k:restore_tensor(v) for k,v in after_event['after'].items()}
    check(set(before)==set(after) and len(before)==42,'snapshot inventory differs')
    expected_error='reset changed untouched buffer: sensor/contact_forces/_is_outdated'
    try:check_transition(before,after,[0])
    except AssertionError as error:check(str(error)==expected_error,'boundary rejection differs')
    else:raise ValueError('original reset boundary unexpectedly accepted')
    changes={};selected={}
    for key in before:
        changed=(before[key]!=after[key]).reshape(120,-1).any(dim=1).nonzero().flatten().tolist()
        if 0 in changed:selected[key]=[0]
        others=[i for i in changed if i!=0]
        if others:changes[key]=others
    flag='sensor/contact_forces/_is_outdated';net='cache/contact_forces/net_forces_w';history='cache/contact_forces/net_forces_w_history'
    check(set(changes)=={flag,net,history} and changes[flag]==list(range(1,120))
        and len(changes[net])==26 and changes[net]==changes[history],'nonselected delta profile differs')
    check(before[flag].all() and after[flag][0] and not after[flag][1:].any(),'cache freshness pattern differs')
    check(set(selected)=={'column','origin','root','sensor/height_scanner/_is_outdated','sensor/height_scanner_base/_is_outdated'},'selected delta profile differs')
    sensor_path=next(Path(p) for p in plan['dependency_sha256'] if p.endswith('/sensors/contact_sensor/contact_sensor.py'))
    base_path=next(Path(p) for p in plan['dependency_sha256'] if p.endswith('/sensors/sensor_base.py'))
    for path in (sensor_path,base_path):check(hashlib.sha256(path.read_bytes()).hexdigest()==plan['dependency_sha256'][str(path)],'installed sensor source changed')
    def method(path,cls,name):
        tree=ast.parse(path.read_text());node=next(v for v in tree.body if isinstance(v,ast.ClassDef) and v.name==cls)
        return deepcopy(next(v for v in node.body if isinstance(v,ast.FunctionDef) and v.name==name))
    # Execute only CPU sensor bookkeeping. Backend values are the recorded after-cache,
    # so this checks lazy refresh/reset mechanics and never verifies physical forces.
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
        ast.ClassDef(name='SensorBase',bases=[],keywords=[],decorator_list=[],body=[method(base_path,'SensorBase','reset'),method(base_path,'SensorBase','_update_outdated_buffers')]),
        ast.ClassDef(name='ContactSensor',bases=[ast.Name(id='SensorBase',ctx=ast.Load())],keywords=[],decorator_list=[],body=[method(sensor_path,'ContactSensor','data'),method(sensor_path,'ContactSensor','reset')])],type_ignores=[])
    namespace={'torch':torch};exec(compile(ast.fix_missing_locations(module),str(sensor_path),'exec'),namespace)
    from types import SimpleNamespace as NS
    def sensor():
        obj=namespace['ContactSensor']();obj.cfg=NS(filter_prim_paths_expr=[],track_air_time=False,track_contact_points=False,track_friction_forces=False)
        obj._timestamp=before['sensor/contact_forces/_timestamp'].clone();obj._timestamp_last_update=before['sensor/contact_forces/_timestamp_last_update'].clone()
        obj._is_outdated=before[flag].clone();obj._data=NS(net_forces_w=before[net].clone(),net_forces_w_history=before[history].clone())
        def update(ids):
            obj._data.net_forces_w[ids]=after[net][ids];obj._data.net_forces_w_history[ids]=after[history][ids]
        obj._update_buffers_impl=update;return obj
    obj=sensor();obj.data;obj.reset(torch.tensor([0]))
    check(torch.equal(obj._is_outdated,after[flag]) and torch.equal(obj._data.net_forces_w,after[net])
        and torch.equal(obj._data.net_forces_w_history,after[history]),'CPU lazy-refresh replay differs')
    primed=sensor();primed.data
    baseline=[primed._is_outdated.clone(),primed._data.net_forces_w.clone(),primed._data.net_forces_w_history.clone()]
    primed.data;primed.reset(torch.tensor([0]))
    check(all(torch.equal(a[1:],b[1:]) for a,b in zip(baseline,[primed._is_outdated,primed._data.net_forces_w,primed._data.net_forces_w_history])),
        'CPU primed-baseline comparison still changes unselected cache rows')
    callback_sources={}
    for name in ['source/rl_training/rl_training/online_adaptation/pre_reset_capture_env.py',
        'source/rl_training/rl_training/online_adaptation/isaaclab_adapter.py','source/rl_training/rl_training/source_scan/scale_diagnostic.py']:
        check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==plan['input_sha256'][name],'callback source differs')
        callback_sources[name]=plan['input_sha256'][name]
    check('reset changed untouched buffer: sensor/contact_forces/_is_outdated' in report['error'],'original exception missing')
    return dict(status='rejected_as_frozen_G2',event_counts=dict(counts),snapshot_fields=42,reset_attempts=1,
        original_reset_after_snapshots=1,qualified_reset_batches=0,completed_controls=0,recorded_physics_substeps=0,
        diagnostic_transitions=0,training_transitions=0,ppo_updates=0,original_boundary_error=expected_error,
        changed_nonselected_fields=changes,unchanged_nonselected_fields=sorted(set(before)-set(changes)),
        changed_selected_fields=sorted(selected),actual_observation_history_terms=report['observation_history_terms'],
        source_mechanism='pre-reset adapter callback reads lazy contact data after the before-snapshot',
        callback_source_sha256=callback_sources,sensor_source=str(sensor_path),sensor_base_source=str(base_path),
        CPU_mechanism_replay_matched=True,CPU_replay_backend_uses_recorded_after_cache=True,
        CPU_primed_baseline_preserves_unselected_cache=True,physical_force_correctness_verified=False,
        actual_GPU_retest_performed=False,G2_accepted=False,training_ready=False,g3_g4_qualified=False)


def main(source_root):
    raw=HERE/'raw';read=lambda name:json.loads((raw/name).read_text())
    lines=(HERE/'SHA256SUMS').read_text().splitlines();listed=set()
    for line in lines:
        digest,name=line.split('  ',1);path=(HERE/name).resolve();listed.add(name)
        check(path.is_relative_to(HERE) and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'archive bytes differ: '+name)
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS','review.json') and '__pycache__' not in p.parts}
    check(listed==actual and len(lines)==len(listed),'archive coverage differs')
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,record in provenance['files'].items():
        payload=(HERE/name).read_bytes();check(len(payload)==record['bytes'] and hashlib.sha256(payload).hexdigest()==record['sha256'],'original bytes differ: '+name)
    frozen=read('frozen_proposal.json');plan=read('run_manifest.json');approved=read('approved_manifest.json')
    digest=hashlib.sha256((raw/'frozen_proposal.json').read_bytes()).hexdigest()
    check(digest=='3bc4c8d71beeb78ff7dda3f09b710b978b7aa1a7a47a680ed53cce7c6a1bc9dc','frozen scope changed')
    check(frozen['status']=='proposed' and frozen['user_approved'] is False and plan==approved==frozen|dict(status='approved',user_approved=True),'approval changed scope')
    check(plan['base_revision']==plan['remote_revision_verified']=='48ed6680d16c8fe090ede7db73407db9a135116a','executed revision differs')
    check(read('approval.json')['proposal_sha256']==digest,'approval binding differs')
    for before,after,key,count in [('input_hashes_before.json','input_hashes_after.json','input_sha256',243),('dependencies_before.json','dependencies_after.json','dependency_sha256',51)]:
        check(read(before)==read(after)==plan[key] and len(plan[key])==count,'source/dependency snapshot changed')
    for name,digest in plan['input_sha256'].items():check(hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,'source differs; use --source-root for 48ed668: '+name)
    for name,digest in plan['dependency_sha256'].items():check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'installed dependency differs: '+name)
    before=(raw/'production_ledger_before.json').read_bytes();check(before==(raw/'production_ledger_after.json').read_bytes()
        and hashlib.sha256(before).hexdigest()==plan['budget_ledger']['sha256'],'production budget changed')
    invocation=read('supervisor_execution.json');execution=read('execution.json');post=read('postflight.json')
    check(invocation['single_supervisor_invocation'] is True and invocation['automatic_retry'] is False and invocation['returncode']==1,'supervisor outcome differs')
    check(execution['returncode']==0 and execution['timed_out'] is False and execution['monitor_errors']==[]
        and execution['process_group_empty'] is True and post['process_group_empty'] is True and post['own_worker_pids_absent'] is True,'failure cleanup differs')
    audit=read('attempt_audit.json');check(audit['output_existed_before'] is False and audit['prior_matching_supervisor_or_child_found'] is False,'attempt audit differs')
    sys.path[:0]=[str(source_root/'source/rl_training'),str(source_root/'scripts/tools')]
    from source_scale_reset_validation import validate
    from rl_training.source_scan.scale_reset import validate_predecessor
    import torch
    validate_predecessor(plan['predecessor_qualification'],source_root=source_root)
    acceptance=validate(raw,plan,execution);check(acceptance==read('acceptance.json')==read('independent_acceptance.json') and acceptance['accepted'] is False,'frozen rejection differs')
    check(analyze(raw,source_root)==read('failure_analysis.json'),'independent failure analysis differs')
    cleanup=read('result.cleanup.json');check(cleanup['environment']=='returned' and cleanup['application'] in ('closing','returned') and cleanup['persistence_errors']==[],'cleanup evidence incomplete')
    check((raw/'result.steps.jsonl').read_bytes()==b'','a recovery control unexpectedly completed')
    check(hashlib.sha256(Path(post['checkpoint_path']).read_bytes()).hexdigest()==plan['checkpoint_sha256'],'original checkpoint changed')
    check(not torch.cuda.is_initialized(),'CPU review initialized CUDA')
    return dict(status='verified_rejected_120_env_G2_failure_archive',archive_review_passed=True,G2_accepted=False,
        hashed_archive_files=len(listed),executed_revision=plan['base_revision'],production_inputs=243,installed_dependencies=51,
        runtime_GPU_launches=1,review_GPU_launches=0,reset_attempts=1,qualified_reset_batches=0,actual_diagnostic_transitions=0,
        actual_recorded_physics_substeps=0,real_source_training_transitions=0,real_source_ppo_updates=0,
        callback_chain_source_hashes_verified=True,sensor_lazy_refresh_bookkeeping_replay_verified=True,production_budget_unchanged=True,
        training_ready=False,g3_g4_qualified=False,ground_positive_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,default=HERE.parents[1]);args=parser.parse_args()
    result=main(args.source_root.resolve());(HERE/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
