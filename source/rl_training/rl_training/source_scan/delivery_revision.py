"""Explicit approval and limited evidence for the revised first numeric trial."""
import ast
import datetime
import json
from pathlib import Path
import subprocess

from .delivery_stage import (reference, read_reference, file_digest, validate_grant,
                             validate_receipt)

REVISION = '5107448972f87a37d85f276a9b0d1078c55f3376'
PROPOSAL = 'docs/lite3-source-training-delivery-revision-proposal-20261104.md'
CONTROLS = 1884
ROOT = Path(__file__).resolve().parents[4]
SCHEMA = 'lite3_training_delivery_revision_approval_v1'
LIMITED_SCHEMA = 'lite3_limited_real_motion_evidence_v1'
SCHEDULE = [dict(step=48+40*(i+1), ids=list(range(20*i,20*(i+1)))) for i in range(6)]


def validate_revision(ref, stage_ref=None):
    import hashlib
    approval = read_reference(ref, 'stage_revision')
    original = subprocess.check_output(['git', 'show', REVISION+':'+PROPOSAL], cwd=ROOT)
    expected = dict(schema=SCHEMA, status='approved', user_approved=True, user_message='同意',
        proposal_revision=REVISION, proposal_sha256=hashlib.sha256(original).hexdigest(),
        complete_controls=CONTROLS, validation_timeout_s=1800,
        remaining_jobs=['g4','numerics','compare'], scheduled_resets=SCHEDULE,
        original_G3_accepted=False, extra_GPU_G3_launches=0, automatic_retry=False)
    if any(type(approval.get(k)) is not type(v) or approval[k]!=v for k,v in expected.items()):
        raise PermissionError('revised delivery scope differs from user approval')
    grant = validate_grant(approval['original_stage'])
    if stage_ref is not None and approval['original_stage']!=stage_ref:
        raise PermissionError('revision belongs to another original stage')
    plan = read_reference(approval['source_manifest'], 'run_manifest')
    if (Path(plan['output_dir'])!=Path(grant['output_root'])/'g3_fast'
            or plan['base_revision']!='dee8efe377fd46e06263bc69a5e1e46648afc3d6'
            or plan.get('delivery_stage')!=approval['original_stage']
            or plan.get('contact_digest_scheme')!='sha256_blocks_4096_v1'):
        raise ValueError('revision must use the actual frozen repaired G3')
    return approval, plan


def compatible_inputs(plan, inputs):
    prefix='source/rl_training/rl_training/source_scan/'
    allowed={prefix+n+'.py' for n in ('delivery_stage','cumulative_training','runtime_readiness')}
    allowed.add(prefix+'delivery_metrics.py')
    allowed.update('scripts/tools/'+n+'.py' for n in
        ('source_scale_motion_validation','source_training_delivery','source_delivery_light','source_training_supervisor'))
    additions={prefix+'delivery_revision.py','scripts/tools/source_limited_motion_validation.py'}
    old=plan['input_sha256'];changed={k for k,v in old.items() if inputs.get(k)!=v};added=set(inputs)-set(old)
    if not set(old)<=set(inputs) or not changed<=allowed or not added<=additions:
        raise ValueError('revision changes production physics/environment/reset/actor inputs')
    # The light probe may add measurement, but its scene construction and scan
    # check remain identical to the already frozen implementation.
    name='scripts/tools/source_delivery_light.py'
    before=ast.parse(subprocess.check_output(['git','show',plan['base_revision']+':'+name],cwd=ROOT,text=True))
    after=ast.parse((ROOT/name).read_text())
    def signature(tree):
        scan=next(n for n in tree.body if getattr(n,'name',None)=='check_scans')
        probe=next(n for n in tree.body if getattr(n,'name',None)=='probe')
        configuration=[]
        for n in ast.walk(probe):
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                if any(isinstance(t,ast.Name) and t.id in ('cfg','raw_env') or
                       isinstance(t,ast.Attribute) and ast.unparse(t).startswith(('cfg.','ranges.')) for t in targets):
                    configuration.append(ast.dump(n,include_attributes=False))
        return ast.dump(scan,include_attributes=False),configuration
    if signature(before)!=signature(after):
        raise ValueError('revision changes scene configuration or production scan checks')
    name=prefix+'delivery_metrics.py'
    old_tree=ast.parse(subprocess.check_output(['git','show',plan['base_revision']+':'+name],cwd=ROOT,text=True))
    new_tree=ast.parse((ROOT/name).read_text())
    preserved={'append_json','TrainingCurves','FirstEpisodes','paired_conditions'}
    if any(ast.dump(next(n for n in old_tree.body if getattr(n,'name',None)==key),include_attributes=False)
           !=ast.dump(next(n for n in new_tree.body if getattr(n,'name',None)==key),include_attributes=False)
           for key in preserved):
        raise ValueError('revision changes existing training or episode metric behavior')
    name='scripts/tools/source_training_supervisor.py'
    old_tree=ast.parse(subprocess.check_output(['git','show',plan['base_revision']+':'+name],cwd=ROOT,text=True))
    new_tree=ast.parse((ROOT/name).read_text())
    def execution_signature(tree):
        return ast.dump(ast.Module(body=[n for n in tree.body if getattr(n,'name',None)!='validate_result'],
                                   type_ignores=[]),include_attributes=False)
    if execution_signature(old_tree)!=execution_signature(new_tree):
        raise ValueError('revision changes numeric execution rather than acceptance metadata')
    return dict(unchanged_inputs=len(old)-len(changed),qualification_or_measurement_changes=sorted(changed),
        added_review_modules=sorted(added),production_configuration_unchanged=True)


def source_artifacts(approval, *, verify=True):
    index=read_reference(approval['source_archive_index'],'evidence_index')
    folder=Path(read_reference(approval['source_manifest'],'run_manifest')['output_dir'])
    records={Path(r['original_path']).name:r for r in index['files']}
    required={'events.jsonl.gz','result.steps.jsonl','execution.json','result.json',
              'run_manifest.json','supervision.json','acceptance.json'}
    if not required<=set(records):raise ValueError('source artifact index is incomplete')
    for name in required:
        row=records[name];path=folder/name
        if Path(row['original_path'])!=path or path.stat().st_size!=row['size_bytes']:
            raise ValueError('limited source artifact path/size differs: '+name)
        if verify and file_digest(path)!=row['sha256']:
            raise ValueError('limited source artifact changed: '+name)
    execution=json.loads((folder/'execution.json').read_text())
    acceptance=json.loads((folder/'acceptance.json').read_text())
    if (execution.get('returncode')!=124 or execution.get('timed_out') is not True
            or execution.get('process_group_empty') is not True
            or execution.get('validation_process_group_empty') is not True
            or acceptance.get('accepted') is not False):
        raise ValueError('the failed G3 execution must remain failed and cleaned up')
    return records


def validate_limited_receipt(ref, *, revision_ref, inputs):
    approval, source=validate_revision(revision_ref)
    receipt=read_reference(ref,'limited_motion')
    report=read_reference(receipt['report'],'limited_motion_report')
    execution=read_reference(receipt['execution'],'limited_review_execution')
    if (receipt.get('schema')!=LIMITED_SCHEMA or receipt.get('scope_qualified') is not True
            or receipt.get('stage_revision')!=revision_ref or report.get('schema')!=LIMITED_SCHEMA
            or report.get('scope_qualified') is not True or report.get('accepted') is not False
            or report.get('G3_qualified') is not False or report.get('training_ready') is not False
            or report.get('synthetic_cpu_fixture') or report.get('input_sha256')!=inputs
            or report.get('complete_controls')!=CONTROLS or report.get('physics_substeps')!=4*CONTROLS
            or report.get('diagnostic_transitions')!=120*CONTROLS
            or report.get('deferred_final_recovery_ids')!=[]
            or report.get('observed_cells')!=list(range(15))
            or report.get('ground_positive_environment_ids')!=list(range(120))
            or report.get('original_G3_accepted') is not False
            or report.get('independent_CPU_review') is not True
            or report.get('source_manifest')!=approval['source_manifest']
            or execution.get('returncode')!=0 or execution.get('timed_out') is not False
            or execution.get('process_group_empty') is not True):
        raise ValueError('limited real motion evidence did not independently qualify')
    compatible_inputs(source,inputs)
    source_artifacts(approval)
    validate_receipt(source['predecessor_qualification'],'g2',inputs=source['input_sha256'])
    return receipt
