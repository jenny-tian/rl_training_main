"""Separately approved 120-env G2 reset boundaries and four recovery controls."""
import hashlib
import json
from pathlib import Path

from .scale_startup import CONTRACT as G1_CONTRACT,validate_predecessor as validate_g1_chain

PREDECESSOR_120 = dict(archive='docs/lite3-source-shared-bank-g1-120env-runtime-20261103',
    executed_revision='a455b59aa9e429b3c210e731432bf4cc96d23be0',
    sha256sums_sha256='3a01b0eb749f2dbaaf9fbfc114e6baec64395f7a34b2a352517e13becd685106',
    review_sha256='63ce03e1d652827d7fca65bc44d371e01645894f63613a99f6940330a5c4a8a8')


def contract_for(num_envs=120):
    if type(num_envs) is not int or num_envs!=120:raise ValueError('G2 is registered only for exactly 120 environments')
    return G1_CONTRACT|dict(schema='lite3_source_shared_scale_G2_v1',purpose='shared_bank_G2_120_envs',
        attempt_id='lite3-source-shared-g2-120env-20261103-v1',num_envs=120,control_steps=4,
        physics_substeps=16,diagnostic_transitions=480,contact_capacity=1048576,
        reset_batches=[[0],[1,17,119],list(range(0,120,2)),list(range(120))],
        selective_reset_calls=4,selected_environment_resets=184,assignment_count=304,
        complete_assignment_blocks=2,tail_assignment_count=64,recovery_control_steps_per_batch=1,
        events_byte_limit=268435456,wall_timeout_s=480,minimum_free_disk_bytes=4294967296,
        predecessor_qualification=dict(PREDECESSOR_120))


def same_value(actual,expected):
    return type(actual) is type(expected) and json.dumps(actual,sort_keys=True,allow_nan=False)==json.dumps(expected,sort_keys=True,allow_nan=False)


def validate_predecessor(reference,*,source_root=None):
    if reference!=PREDECESSOR_120:raise ValueError('G2 requires the fixed accepted 120-env G1 predecessor')
    root=Path(source_root) if source_root is not None else Path(__file__).resolve().parents[4]
    archive=(root/PREDECESSOR_120['archive']).resolve()
    for name,key in (('SHA256SUMS','sha256sums_sha256'),('review.json','review_sha256')):
        if hashlib.sha256((archive/name).read_bytes()).hexdigest()!=reference[key]:raise ValueError('120-env G1 anchor changed: '+name)
    for line in (archive/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(archive/name).resolve()
        if not path.is_relative_to(archive) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('120-env G1 evidence changed: '+name)
    review=json.loads((archive/'review.json').read_text())
    if (review.get('status')!='passed_bounded_120_env_G1_archive_review' or review.get('accepted') is not True
            or review.get('executed_revision')!=reference['executed_revision'] or review.get('runtime_GPU_launches')!=1
            or review.get('real_read_only_diagnostic_transitions')!=120 or review.get('initial_120_ticket_block_verified') is not True
            or review.get('ground_positive_qualified') is not False or review.get('training_ready') is not False
            or review.get('g3_g4_qualified') is not False):raise ValueError('120-env G1 scope does not qualify for G2 preparation')
    prior=json.loads((archive/'raw/run_manifest.json').read_text())
    validate_g1_chain(120,prior.get('predecessor_qualification'),source_root=root)


def proposal(*,revision,remote,inputs,dependencies,budget_reference,output_dir,num_envs=120):
    return contract_for(num_envs)|dict(status='proposed',user_approved=False,base_revision=revision,
        remote_revision_verified=remote,input_sha256=inputs,dependency_sha256=dependencies,
        budget_ledger=budget_reference,output_dir=str(Path(output_dir).resolve()))


def validate_plan(plan,*,revision,inputs,dependencies,budget_reference,output_dir,approved=True,num_envs=120):
    if not isinstance(plan,dict):raise ValueError('G2 plan must be a mapping')
    if approved and (plan.get('status')!='approved' or plan.get('user_approved') is not True):
        raise PermissionError('separate approval of the single G2 launch is required')
    if not approved and (plan.get('status')!='proposed' or plan.get('user_approved') is not False):
        raise ValueError('G2 preparation must remain unapproved')
    expected=proposal(revision=revision,remote=revision,inputs=inputs,dependencies=dependencies,
        budget_reference=budget_reference,output_dir=output_dir,num_envs=num_envs)
    for key,value in expected.items():
        if key not in ('status','user_approved') and not same_value(plan.get(key),value):raise ValueError('G2 contract mismatch: '+key)
    if len(inputs)<243 or len(dependencies)<51:raise ValueError('G2 source/dependency inventory incomplete')
    validate_predecessor(plan.get('predecessor_qualification'))
    if Path(output_dir).exists():raise FileExistsError('G2 attempt exists; retry/resume disabled')
    return plan
