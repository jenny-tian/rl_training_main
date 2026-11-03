"""One separately approved 16-environment startup check, with no policy or PPO."""
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


def proposal(*,revision,remote,inputs,dependencies,budget_reference,output_dir):
    return CONTRACT|dict(status='proposed',user_approved=False,base_revision=revision,
        remote_revision_verified=remote,input_sha256=inputs,dependency_sha256=dependencies,
        budget_ledger=budget_reference,output_dir=str(Path(output_dir).resolve()))


def validate_plan(plan,*,revision,inputs,dependencies,budget_reference,output_dir,approved=True):
    if not isinstance(plan,dict):raise ValueError('G1 plan must be a mapping')
    if approved and (plan.get('status')!='approved' or plan.get('user_approved') is not True):
        raise PermissionError('separate approval of the single 16-environment G1 launch is required')
    if not approved and (plan.get('status')!='proposed' or plan.get('user_approved') is not False):
        raise ValueError('G1 preparation must remain unapproved')
    expected=proposal(revision=revision,remote=revision,inputs=inputs,dependencies=dependencies,
        budget_reference=budget_reference,output_dir=output_dir)
    for key,value in expected.items():
        if key in ('status','user_approved'):continue
        if type(plan.get(key)) is not type(value) or plan[key]!=value:raise ValueError('G1 contract mismatch: '+key)
    if len(inputs)<230 or len(dependencies)<44:raise ValueError('G1 source/dependency inventory incomplete')
    if Path(output_dir).exists():raise FileExistsError('G1 attempt directory exists; retry/resume disabled')
    return plan
