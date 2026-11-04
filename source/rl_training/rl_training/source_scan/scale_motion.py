"""Separately approved, read-only 120-environment G3 motion contract."""
import hashlib
import json
from pathlib import Path

from .scale_startup import CONTRACT
from .scale_reset import same_value,validate_predecessor as validate_g1_chain

G2=dict(archive='docs/lite3-source-shared-bank-g2-120env-v2-runtime-20261104',
    executed_revision='717f21fbfe7d08b957208d487b8e97f0c3af0bea',
    sha256sums_sha256='afe5ee37a6cc7956813f752e24b7037971fd33801bbc26d415c7fecb14f66ab4',
    review_sha256='4d825cda028c43265c2012775f3388b7cc9ccd056244e5d16a20e9d2cd4c9959')
FLAT=dict(archive='docs/lite3-source-policy-prequalification-flat-20261029',
    executed_revision='3b79cdda6574d9c7936437243f980166b65e2614',
    sha256sums_sha256='d8a61af98bb2e5d68513dfd231d0fdb8d944f5e60e790b784b41c9be8a1f65ee',
    review_sha256='1811809c415bac284fa76db1250be85bff5006fa8e8017801d115eac7a643708')


def contract_for(num_envs=120):
    if type(num_envs) is not int or num_envs!=120:raise ValueError('G3 requires exactly 120 environments')
    return CONTRACT|dict(schema='lite3_source_shared_scale_G3_v1',purpose='shared_bank_G3_120_envs',
        attempt_id='lite3-source-shared-g3-120env-20261104-v1',num_envs=120,main_control_steps=2000,
        recovery_control_steps=1,control_steps=2001,physics_substeps=8004,diagnostic_transitions=240120,
        policy_loaded=True,actions='frozen_actor_plus_seeded_unit_Gaussian',noise_seed=132001,noise_std=1.,
        actor_compute='float32_no_TF32_no_autocast',
        command_mps=[.3,0.,0.],require_ground_positive=True,contact_capacity=1048576,
        ordinary_task_endings='record_reset_and_continue',integrity_errors='stop_and_retain_evidence',
        flat_completion_required=True,all_cells_observed=True,continuous_time_safety_claim=False,
        final_recovery_segment_resets='report_as_unobserved_not_qualified',
        events_encoding='concatenated_gzip_JSONL_lossless',events_byte_limit=16*1024**3,
        events_uncompressed_byte_limit=64*1024**3,event_record_byte_limit=512*1024**2,
        steps_byte_limit=4*1024**3,runtime_log_byte_limit=512*1024**2,minimum_free_disk_bytes=32*1024**3,
        wall_timeout_s=7200,validation_timeout_s=7200,predecessor_qualification=dict(G2),policy_prequalification=dict(FLAT))


def validate_policy_prequalification(reference,*,source_root=None):
    if reference!=FLAT:raise ValueError('G3 requires the fixed independent Flat policy prerequisite')
    root=Path(source_root) if source_root is not None else Path(__file__).resolve().parents[4]
    folder=root/FLAT['archive']
    for name,key in [('SHA256SUMS','sha256sums_sha256'),('post_run_review.json','review_sha256')]:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=FLAT[key]:raise ValueError('Flat prerequisite anchor changed')
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Flat prerequisite evidence changed')
    review=json.loads((folder/'post_run_review.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    if (review['review_result']!='single_condition_safety_passed_with_scope_limits'
            or review['revision']!=FLAT['executed_revision'] or review['policy_updated'] is not False
            or review['ppo_initialized'] is not False or review['g3_qualified'] is not False
            or review['reconstructed_final_hold_steps']<50 or review['metrics_reconstructed_from_samples']['forward_progress_m']<9.
            or manifest['checkpoint_sha256']!=CONTRACT['checkpoint_sha256']
            or manifest['command']['lin_vel_x_mps']!=[.3,.3] or manifest['safety_height']['update_period_s']!=.005):
        raise ValueError('Flat policy prerequisite scope differs')


def validate_predecessor(reference,*,source_root=None,verify_current_inputs=True):
    if isinstance(reference,dict) and reference.get('schema')=='lite3_delivery_artifact_reference_v1':
        from .delivery_stage import validate_receipt
        validate_receipt(reference,'g2');validate_policy_prequalification(FLAT,source_root=source_root);return
    if reference!=G2:raise ValueError('G3 requires the accepted real G2 v2 archive')
    root=Path(source_root) if source_root is not None else Path(__file__).resolve().parents[4]
    folder=root/G2['archive']
    for name,key in [('SHA256SUMS','sha256sums_sha256'),('review.json','review_sha256')]:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=G2[key]:raise ValueError('G2 anchor changed: '+name)
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('G2 evidence changed: '+name)
    value=json.loads((folder/'review.json').read_text())
    if (value.get('accepted') is not True or value.get('executed_revision')!=G2['executed_revision']
            or value.get('runtime_GPU_launches')!=1 or value.get('real_read_only_diagnostic_transitions')!=480
            or value.get('reset_boundary_qualified') is not True or value.get('reset_first_recovery_step_qualified') is not True
            or value.get('training_ready') is not False or value.get('g3_g4_qualified') is not False):
        raise ValueError('G2 scope does not qualify for G3 preparation')
    plan=json.loads((folder/'raw/run_manifest.json').read_text())
    # New G3 files may be added, but none of the already qualified inputs may change.
    if verify_current_inputs:
        for name,digest in plan['input_sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('qualified G2 input changed: '+name)
    validate_g1_chain(plan['predecessor_qualification'],source_root=root)
    validate_policy_prequalification(FLAT,source_root=root)


def proposal(*,revision,remote,inputs,dependencies,budget_reference,output_dir,num_envs=120):
    return contract_for(num_envs)|dict(status='proposed',user_approved=False,base_revision=revision,
        remote_revision_verified=remote,input_sha256=inputs,dependency_sha256=dependencies,
        budget_ledger=budget_reference,output_dir=str(Path(output_dir).resolve()))


def validate_plan(plan,*,revision,inputs,dependencies,budget_reference,output_dir,approved=True,num_envs=120):
    if not isinstance(plan,dict):raise ValueError('G3 plan must be a mapping')
    if approved and (plan.get('status')!='approved' or plan.get('user_approved') is not True):
        raise PermissionError('separate approval of the one G3 GPU run is required')
    if not approved and (plan.get('status')!='proposed' or plan.get('user_approved') is not False):
        raise ValueError('G3 preparation must remain unapproved')
    expected=proposal(revision=revision,remote=revision,inputs=inputs,dependencies=dependencies,
        budget_reference=budget_reference,output_dir=output_dir,num_envs=num_envs)
    for key,value in expected.items():
        if key not in ('status','user_approved') and not same_value(plan.get(key),value):raise ValueError('G3 contract mismatch: '+key)
    if len(inputs)<249 or len(dependencies)<51:raise ValueError('G3 input/dependency inventory incomplete')
    validate_predecessor(plan['predecessor_qualification'])
    if Path(output_dir).exists():raise FileExistsError('G3 attempt exists; retry/resume disabled')
    return plan
