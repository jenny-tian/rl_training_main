"""Bound the user-approved training delivery stage and its single-use job slots."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path

STAGE='lite3-training-delivery-20261104-v1'
BASE='513e65e7ba94017115fad7a58ef772eded2f610d'
PROPOSAL='docs/lite3-source-training-delivery-proposal-20261104.md'
LIMITS={'locate':dict(controls=600,wall_s=900,training_transitions=0),
        'g1':dict(controls=1,wall_s=240,training_transitions=0),
        'g2':dict(controls=4,wall_s=240,training_transitions=0),
        'g3':dict(controls=2001,wall_s=1800,training_transitions=0),
        'g4':dict(controls=288,wall_s=900,training_transitions=0),
        'numerics':dict(controls=2160,wall_s=900,training_transitions=259200),
        'compare':dict(controls=4000,wall_s=1800,training_transitions=0)}


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def reference(path,kind):
    path=Path(path).resolve()
    return dict(schema='lite3_delivery_artifact_reference_v1',kind=kind,path=str(path),sha256=file_digest(path))


def read_reference(ref,kind):
    if (not isinstance(ref,dict) or set(ref)!={'schema','kind','path','sha256'}
            or ref['schema']!='lite3_delivery_artifact_reference_v1' or ref['kind']!=kind):
        raise ValueError('wrong delivery evidence reference: '+kind)
    path=Path(ref['path'])
    if not path.is_absolute() or file_digest(path)!=ref['sha256']:raise ValueError('delivery evidence changed: '+kind)
    return json.loads(path.read_text())


def validate_grant(ref):
    import subprocess
    grant=read_reference(ref,'stage_approval')
    root=Path(__file__).resolve().parents[4]
    original=subprocess.check_output(['git','show',BASE+':'+PROPOSAL],cwd=root)
    expected=dict(schema='lite3_training_delivery_approval_v1',stage_id=STAGE,user_approved=True,
        user_message='同意',proposal_revision=BASE,proposal_sha256=hashlib.sha256(original).hexdigest(),
        job_limits=LIMITS,minimum_free_vram_mib=12288,run=1,max_updates=90,max_training_transitions=259200)
    if any(type(grant.get(k)) is not type(v) or grant[k]!=v for k,v in expected.items()):
        raise PermissionError('training delivery grant differs from the approved scope')
    if not Path(grant.get('slot_ledger','')).is_absolute():raise ValueError('delivery slot ledger must be absolute')
    return grant


def claim_slot(grant_ref,kind,manifest):
    from .smoke_artifacts import atomic_json
    grant=validate_grant(grant_ref)
    if kind not in LIMITS:raise ValueError('job outside approved stage')
    plan=json.loads(Path(manifest).read_text())
    path=slot_ledger(plan);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(path.read_text()) if path.exists() else dict(stage_id=STAGE,jobs={})
        if ledger['stage_id']!=STAGE or kind in ledger['jobs']:raise PermissionError('delivery slot already consumed: '+kind)
        ledger['jobs'][kind]=dict(state='claimed',manifest=reference(manifest,'run_manifest'),
            claimed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),retry=False)
        atomic_json(path,ledger)


def validate_slot(plan):
    grant=validate_grant(plan['delivery_stage']);kind=plan['delivery_job']
    ledger=json.loads(slot_ledger(plan).read_text())
    row=ledger['jobs'].get(kind,{})
    frozen=read_reference(row.get('manifest'),'run_manifest')
    if row.get('state')!='claimed' or frozen!=plan:raise PermissionError('delivery launch lacks its unique claimed slot')


def validate_receipt(ref,kind,*,inputs=None):
    receipt=read_reference(ref,kind)
    if (receipt.get('schema')!='lite3_delivery_job_receipt_v1' or receipt.get('kind')!=kind
            or receipt.get('accepted') is not True or receipt.get('stage_id')!=STAGE):
        raise ValueError('delivery prerequisite did not pass: '+kind)
    plan=read_reference(receipt['manifest'],'run_manifest')
    acceptance=read_reference(receipt['acceptance'],'acceptance')
    execution=read_reference(receipt['execution'],'execution')
    report=read_reference(receipt['report'],'report')
    validate_grant(plan['delivery_stage'])
    if (plan.get('delivery_job')!=kind or acceptance.get('accepted') is not True
            or execution.get('process_group_empty') is not True or execution.get('timed_out') is not False
            or execution.get('monitor_errors') or execution.get('returncode')!=0
            or report.get('synthetic_cpu_fixture') or acceptance.get('cpu_evidence_validated')
            or report.get('simulation_control_steps')!=LIMITS[kind]['controls']
            or report.get('input_sha256')!=plan.get('input_sha256')):
        raise ValueError('delivery prerequisite runtime/identity differs: '+kind)
    if inputs is not None and plan['input_sha256']!=inputs:
        if kind!='g2':raise ValueError('delivery prerequisite source differs: '+kind)
        g2_compatible_inputs(plan,inputs)
    if kind=='g3' and (execution.get('validation_returncode')!=0
            or not execution.get('validation_process_group_empty')):
        raise ValueError('G3 independent CPU validation did not finish')
    previous={'g2':'g1','g3':'g2','g4':'g3','numerics':'g4','compare':'numerics'}.get(kind)
    if kind=='g2' and plan.get('startup_covered_by_g2') is True:
        from .scale_reset import validate_predecessor
        validate_predecessor(plan['predecessor_qualification'])
        if acceptance.get('startup_and_first_zero_control_qualified') is not True:
            raise ValueError('combined startup/reset validation did not cover startup')
    elif kind=='g4' and plan.get('stage_revision'):
        from .delivery_revision import validate_revision,validate_limited_receipt
        validate_revision(plan['stage_revision'],plan['delivery_stage'])
        validate_limited_receipt(plan['predecessor_qualification'],revision_ref=plan['stage_revision'],inputs=plan['input_sha256'])
    elif previous:validate_receipt(plan['predecessor_qualification'],previous,inputs=plan['input_sha256'])
    return receipt


def job_contract(kind,plan):
    from .scale_startup import contract_for as g1
    from .scale_reset import contract_for as g2
    from .scale_motion import contract_for as g3
    if kind not in LIMITS:raise ValueError('unknown delivery job')
    validate_grant(plan['delivery_stage'])
    revision=plan.get('stage_revision')
    if revision:
        from .delivery_revision import validate_revision
        approval,_=validate_revision(revision,plan['delivery_stage'])
        if kind not in approval['remaining_jobs']:raise PermissionError('job outside revised remaining scope')
    base={'g1':g1,'g2':g2,'g3':g3,'locate':g3}.get(kind,g1)(120)
    count=LIMITS[kind]['controls']
    result=base|dict(attempt_id=STAGE+'-'+kind,purpose='source_delivery_'+kind,num_envs=120,control_steps=count,
        physics_substeps=4*count,diagnostic_transitions=120*count,wall_timeout_s=LIMITS[kind]['wall_s'],
        minimum_free_disk_bytes=32*1024**3,delivery_stage=plan['delivery_stage'],delivery_job=kind,
        scanner_footprint_margin_m=.1)
    if kind in ('g2','g3','g4','numerics','compare'):
        prior={'g2':'g1','g3':'g2','g4':'g3','numerics':'g4','compare':'numerics'}[kind]
        ref=plan.get('predecessor_qualification')
        if kind=='g2' and plan.get('startup_covered_by_g2') is True:
            from .scale_reset import PREDECESSOR_120,validate_predecessor
            if ref!=PREDECESSOR_120:raise ValueError('combined G2 must retain the real fixed 120-env startup anchor')
            validate_predecessor(ref)
            result['startup_covered_by_g2']=True
        elif kind=='g4' and revision:
            from .delivery_revision import validate_limited_receipt
            validate_limited_receipt(ref,revision_ref=revision,inputs=plan['input_sha256'])
        else:validate_receipt(ref,prior,inputs=plan.get('input_sha256'))
        result['predecessor_qualification']=ref
    if kind in ('g3','locate'):
        result.update(schema='lite3_source_shared_scale_G3_delivery_v2',validation_timeout_s=1800,
            main_control_steps=2000 if kind=='g3' else 600,recovery_control_steps=1 if kind=='g3' else 0,
            events_encoding='concatenated_gzip_JSONL_base64_tensors_v1')
        if plan.get('contact_digest_scheme')=='sha256_blocks_4096_v1':
            slot_ledger(plan)
            result.update(schema='lite3_source_shared_scale_G3_delivery_v3',
                attempt_id=STAGE+'-g3-fast-v3',contact_digest_scheme='sha256_blocks_4096_v1',
                g3_supplement=plan['g3_supplement'])
            prior_plan=read_reference(read_reference(plan['predecessor_qualification'],'g2')['manifest'],'run_manifest')
            result['g2_source_compatibility']=g2_compatible_inputs(prior_plan,plan['input_sha256'])
    if kind=='g4':result.update(schema='lite3_source_collection_G4_delivery_v1',warmup_control_steps=48,measured_control_steps=240,
        minimum_transitions_per_second=600.,reset_p95_limit_s=1.,heavy_contact_journal=False,
        policy_loaded=True,actions='frozen_actor_plus_seeded_unit_Gaussian',noise_seed=132001,noise_std=1.)
    if revision:
        result['stage_revision']=revision
        if kind=='g4':result.update(schema='lite3_source_collection_G4_delivery_v2',scheduled_resets=approval['scheduled_resets'])
    if kind=='compare':
        receipt=read_reference(plan['predecessor_qualification'],'numerics')
        numeric_plan=read_reference(receipt['manifest'],'run_manifest')
        expected_checkpoint=reference(Path(numeric_plan['output_dir'])/'training/actor_update_90.pt','actor_checkpoint')
        if plan['new_checkpoint']!=expected_checkpoint:raise ValueError('comparison actor differs from the accepted numeric checkpoint')
        result.update(schema='lite3_source_paired_diagnostics_delivery_v1',policy_loaded=True,control_steps_per_actor=2000,
            actions='deterministic_actor_mean',old_checkpoint_sha256=plan['checkpoint_sha256'],
            new_checkpoint=plan['new_checkpoint'])
    return result


def preflight(args,plan,*,kind,child=False):
    from source_training_entrypoint import _git_state,_input_hashes
    from source_scale_startup_supervisor import dependencies
    from .training_budget_ledger import BudgetLedger,canonical_budget_path
    from .frozen_actor import CHECKPOINT_SHA256
    import os
    import shutil
    root=Path(__file__).resolve().parents[4]
    if getattr(args,'num_envs',120)!=120:raise ValueError('delivery stage is fixed to 120 environments')
    if not child and shutil.disk_usage(Path(plan['output_dir']).parent).free<32*1024**3:
        raise RuntimeError('delivery disk free space is below 32 GiB')
    if plan.get('status')!='approved' or plan.get('user_approved') is not True:
        raise PermissionError('delivery job is not approved')
    validate_slot(plan)
    revision,remote,dirty=_git_state();out=Path(plan['output_dir']).resolve()
    if dirty or revision!=remote or plan['base_revision']!=revision or plan['remote_revision_verified']!=remote:
        raise ValueError('delivery source/remote is not clean and frozen')
    if (plan['input_sha256']!=_input_hashes() or plan['dependency_sha256']!=dependencies()
            or plan['budget_ledger']!=BudgetLedger(canonical_budget_path(root)).reference()
            or file_digest(args.checkpoint)!=CHECKPOINT_SHA256
            or os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'0')):
        raise ValueError('delivery source/dependency/checkpoint/budget/device changed')
    expected=job_contract(kind,plan)
    for k,v in expected.items():
        if type(plan.get(k)) is not type(v) or plan[k]!=v:raise ValueError('delivery contract mismatch: '+k)
    if child:
        if (args.manifest.resolve()!=out/'run_manifest.json' or (out/'result.json').exists()
                or json.loads((out/'supervision.json').read_text())!=dict(revision=revision,manifest_sha256=file_digest(args.manifest))):
            raise ValueError('delivery child ticket or restart differs')
    elif out.exists() or args.output_dir.resolve()!=out:raise FileExistsError('delivery output already exists or differs')
    return plan


def seal_receipt(folder,kind):
    from .smoke_artifacts import atomic_json
    folder=Path(folder)
    acceptance=json.loads((folder/'acceptance.json').read_text())
    receipt=dict(schema='lite3_delivery_job_receipt_v1',stage_id=STAGE,kind=kind,
        accepted=acceptance.get('accepted') is True,manifest=reference(folder/'run_manifest.json','run_manifest'),
        acceptance=reference(folder/'acceptance.json','acceptance'),execution=reference(folder/'execution.json','execution'),
        report=reference(folder/'result.json','report'))
    atomic_json(folder/'receipt.json',receipt)
    if receipt['accepted']:validate_receipt(reference(folder/'receipt.json',kind),kind)
    return receipt


def slot_ledger(plan):
    grant=validate_grant(plan['delivery_stage'])
    ref=plan.get('g3_supplement')
    if ref is None:return Path(grant['slot_ledger'])
    supplement=read_reference(ref,'g3_supplement')
    expected=dict(schema='lite3_G3_fast_supplement_v1',status='approved',user_approved=True,
        stage_id=STAGE,job='g3',num_envs=120,control_steps=2001,physics_substeps=8004,
        diagnostic_transitions=240120,training_transitions=0,ppo_updates=0,max_launches=1,
        wall_timeout_s=1800,validation_timeout_s=1800,noise_seed=132001,noise_std=1.,command_mps=[.3,0.,0.],
        contact_digest_scheme='sha256_blocks_4096_v1',source_revision=plan['base_revision'],
        input_sha256=plan['input_sha256'],output_dir=plan['output_dir'])
    if (plan.get('delivery_job')!='g3' or plan.get('contact_digest_scheme')!='sha256_blocks_4096_v1'
            or any(type(supplement.get(k)) is not type(v) or supplement[k]!=v for k,v in expected.items())):
        raise PermissionError('a new separately approved bounded G3 supplement is required')
    path=Path(supplement.get('slot_ledger',''))
    if not path.is_absolute() or path==Path(grant['slot_ledger']):raise ValueError('supplement must retain a separate single-use ledger')
    return path


def g2_compatible_inputs(old_plan,inputs):
    """Retain the real G2 scope only for G3 changes and equivalent v1 CPU decoding."""
    import ast
    import copy
    import subprocess
    if old_plan.get('base_revision')!='9d56d7ef9916a7bbaae439dc4b9557fd95859ee1':
        raise ValueError('G2 source compatibility requires the reviewed real combined run')
    prefix='source/rl_training/rl_training/source_scan/'
    permitted={prefix+'sparse_contact.py',prefix+'delivery_stage.py',prefix+'scale_motion.py',prefix+'scale_motion_evidence.py',
        'scripts/tools/source_training_delivery.py',*['scripts/tools/source_scale_motion_'+s+'.py' for s in ('probe','collection','supervisor','validation')]}
    old=old_plan['input_sha256'];changed={name for name in old if inputs.get(name)!=old[name]};added=set(inputs)-set(old)
    if not changed<=permitted or not added<={prefix+'fast_sparse_contact.py'} or not set(old)<=set(inputs):
        raise ValueError('changes affect the qualified G2 startup/reset implementation')
    root=Path(__file__).resolve().parents[4];name=prefix+'sparse_contact.py'
    before=ast.parse(subprocess.check_output(['git','show',old_plan['base_revision']+':'+name],cwd=root,text=True))
    after=ast.parse((root/name).read_text());find=lambda tree,key:next(n for n in tree.body if getattr(n,'name',None)==key)
    dump=lambda node:ast.dump(node,include_attributes=False)
    ignored={'ContactDecoder','_reconstruct_frame'}
    if dump(ast.Module(body=[n for n in before.body if getattr(n,'name',None) not in ignored],type_ignores=[]))!=dump(ast.Module(body=[n for n in after.body if getattr(n,'name',None) not in ignored],type_ignores=[])):
        raise ValueError('legacy G2 encoder/topology/decoder entry changed')
    previous=copy.deepcopy(find(before,'_reconstruct_frame'));current=copy.deepcopy(find(after,'_reconstruct_frame'))
    class Defaults(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='array_factory':return ast.copy_location(ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),attr='zeros',ctx=node.ctx),node)
            return ast.copy_location(ast.Name(id={'dense_digest_fn':'dense_digest','frame_schema':'SCHEMA'}.get(node.id,node.id),ctx=node.ctx),node)
    current.args.kwonlyargs=[];current.args.kw_defaults=[];current=Defaults().visit(current)
    if dump(previous)!=dump(current):raise ValueError('legacy G2 reconstruction behavior changed')
    return dict(schema='lite3_G2_unchanged_scope_proof_v1',qualified_revision=old_plan['base_revision'],
        unchanged_inputs=len(old)-len(changed),G3_or_authorization_changes=sorted(changed),new_G3_files=sorted(added),
        legacy_encoder_and_default_decoder_unchanged=True,new_G2_GPU_launches=0,
        scope='retain actual startup/selective-reset evidence; no G3 or G4 credit')
