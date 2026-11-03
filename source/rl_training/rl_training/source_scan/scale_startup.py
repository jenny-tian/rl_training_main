"""One separately approved 16- or 32-environment startup check, without PPO."""
import hashlib
import json
from pathlib import Path

from .frozen_actor import CHECKPOINT_SHA256

CONTRACT = dict(schema='lite3_source_shared_scale_G1_v1',purpose='shared_bank_G1_16_envs',
    attempt_id='lite3-source-shared-g1-16env-20261103-v1',num_envs=16,shared_bank_rows=1,level='L1',
    control_steps=1,physics_substeps=4,diagnostic_transitions=16,training_transitions=0,ppo_updates=0,
    policy_loaded=False,actions='all_zero',device='cuda:0',environment_seed=130001,geometry_seed=30000,
    sampler_seed=131001,reset_state_id=40000,reset_randomized=False,registered_cells=15,retain_six_families=True,
    contact_capacity=131072,events_byte_limit=134217728,require_ground_positive=False,
    startup_timeout_s=120,wall_timeout_s=240,cleanup_grace_s=15,minimum_free_vram_mib=12288,
    minimum_free_disk_bytes=2147483648,retry=False,resume=False,automatic_scale_increase=False,
    checkpoint_sha256=CHECKPOINT_SHA256,development_or_sealed_feedback_allowed=False,
    training_ready=False,g3_g4_qualified=False)


PREDECESSOR_16 = dict(archive='docs/lite3-source-shared-bank-g1-16env-runtime-20261103',
    executed_revision='8b2aec0230c4dd2e85dc085a0cdf8cb418f0f5e9',
    sha256sums_sha256='c77ecde74f0e8d9eb5da2234d5eb36bbb0cc1bc577db513e81ed3d6b7ec894f4',
    review_sha256='53ffd0df090934d9f1be7dd6fb7ba3f137282e0f7fb1ed44fa9608752d143758')


def contract_for(num_envs):
    if type(num_envs) is not int or num_envs not in (16,32):
        raise ValueError('G1 launch contract is registered only for 16 or 32 environments')
    if num_envs==16:return dict(CONTRACT)
    return CONTRACT|dict(purpose='shared_bank_G1_32_envs',
        attempt_id='lite3-source-shared-g1-32env-20261103-v1',num_envs=32,
        diagnostic_transitions=32,contact_capacity=262144,predecessor_qualification=dict(PREDECESSOR_16))


def validate_predecessor(num_envs,reference=None,*,source_root=None):
    """Bind the already accepted G1 archive, without rerunning its simulator/reviewer."""
    contract_for(num_envs)
    if num_envs==16:return
    if reference!=PREDECESSOR_16:raise ValueError('32-env G1 requires the fixed accepted 16-env predecessor')
    root=Path(source_root) if source_root is not None else Path(__file__).resolve().parents[4]
    archive=(root/PREDECESSOR_16['archive']).resolve()
    for name,key in (('SHA256SUMS','sha256sums_sha256'),('review.json','review_sha256')):
        if hashlib.sha256((archive/name).read_bytes()).hexdigest()!=reference[key]:
            raise ValueError('16-env G1 predecessor anchor changed: '+name)
    for line in (archive/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=(archive/name).resolve()
        if not path.is_relative_to(archive) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('16-env G1 predecessor evidence changed: '+name)
    review=json.loads((archive/'review.json').read_text())
    if (review.get('status')!='passed_bounded_16_env_G1_archive_review' or review.get('accepted') is not True
            or review.get('executed_revision')!=reference['executed_revision']
            or review.get('runtime_gpu_launches')!=1 or review.get('real_read_only_diagnostic_transitions')!=16
            or review.get('training_ready') is not False or review.get('g3_g4_qualified') is not False):
        raise ValueError('16-env predecessor does not qualify for the next G1 scale')


def proposal(*,revision,remote,inputs,dependencies,budget_reference,output_dir,num_envs=16):
    return contract_for(num_envs)|dict(status='proposed',user_approved=False,base_revision=revision,
        remote_revision_verified=remote,input_sha256=inputs,dependency_sha256=dependencies,
        budget_ledger=budget_reference,output_dir=str(Path(output_dir).resolve()))


def validate_plan(plan,*,revision,inputs,dependencies,budget_reference,output_dir,approved=True,num_envs=16):
    if not isinstance(plan,dict):raise ValueError('G1 plan must be a mapping')
    if approved and (plan.get('status')!='approved' or plan.get('user_approved') is not True):
        raise PermissionError('separate approval of the selected single G1 launch is required')
    if not approved and (plan.get('status')!='proposed' or plan.get('user_approved') is not False):
        raise ValueError('G1 preparation must remain unapproved')
    expected=proposal(revision=revision,remote=revision,inputs=inputs,dependencies=dependencies,
        budget_reference=budget_reference,output_dir=output_dir,num_envs=num_envs)
    for key,value in expected.items():
        if key in ('status','user_approved'):continue
        if type(plan.get(key)) is not type(value) or plan[key]!=value:raise ValueError('G1 contract mismatch: '+key)
    if len(inputs)<230 or len(dependencies)<44:raise ValueError('G1 source/dependency inventory incomplete')
    validate_predecessor(num_envs,plan.get('predecessor_qualification'))
    if Path(output_dir).exists():raise FileExistsError('G1 attempt directory exists; retry/resume disabled')
    return plan
