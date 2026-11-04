"""Revised scope must retain real evidence and reject incomplete reset metrics."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from rl_training.source_scan.delivery_stage import reference
from rl_training.source_scan.delivery_revision import validate_revision,compatible_inputs,SCHEDULE
from rl_training.source_scan.runtime_readiness import numerics_readiness_template,validate_numerics_runtime_receipt
from source_training_entrypoint import _input_hashes
import source_delivery_light as light

APPROVAL=Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/.git/source-delivery-evidence/lite3-training-delivery-20261104-v1/revision_approval.json')


@pytest.fixture
def approved_revision():
    if not APPROVAL.exists():pytest.skip('requires the actual user-approved revision')
    return reference(APPROVAL,'stage_revision')


@pytest.mark.parametrize('fault',[None,'approval','controls','jobs','resets','retry'])
def test_revision_cannot_expand_or_replace_the_approved_scope(tmp_path,approved_revision,fault):
    approval=json.loads(APPROVAL.read_text())
    if fault=='approval':approval['user_approved']=False
    if fault=='controls':approval['complete_controls']=64
    if fault=='jobs':approval['remaining_jobs'].append('g3')
    if fault=='resets':approval['scheduled_resets'][0]['ids']=[0]*20
    if fault=='retry':approval['automatic_retry']=True
    p=tmp_path/'approval.json';p.write_text(json.dumps(approval));ref=reference(p,'stage_revision')
    if fault is None:
        _,source=validate_revision(ref)
        proof=compatible_inputs(source,_input_hashes())
        assert proof['production_configuration_unchanged']
        assert not torch.cuda.is_initialized()
    else:
        with pytest.raises(PermissionError):validate_revision(ref)


@pytest.mark.parametrize('name',['training_env.py','scan_footprint.py','training_runner.py','fast_sparse_contact.py'])
def test_limited_evidence_rejects_changes_to_execution_or_native_decoding(approved_revision,name):
    _,source=validate_revision(approved_revision);inputs=_input_hashes()
    inputs['source/rl_training/rl_training/source_scan/'+name]='0'*64
    with pytest.raises(ValueError,match='production'):compatible_inputs(source,inputs)


def test_allowed_measurement_file_cannot_change_the_scene_configuration(tmp_path,monkeypatch,approved_revision):
    import rl_training.source_scan.delivery_revision as revision
    _,source=validate_revision(approved_revision)
    original=revision.subprocess.check_output(['git','show',source['base_revision']+':scripts/tools/source_delivery_light.py'],cwd=revision.ROOT,text=True)
    target=tmp_path/'scripts/tools/source_delivery_light.py';target.parent.mkdir(parents=True)
    target.write_text(original.replace("cfg.sim.device='cuda:0'","cfg.sim.device='cuda:1'"))
    monkeypatch.setattr(revision,'ROOT',tmp_path)
    monkeypatch.setattr(revision.subprocess,'check_output',lambda *a,**k:original)
    with pytest.raises(ValueError,match='scene configuration'):compatible_inputs(source,_input_hashes())


@pytest.mark.parametrize('fault',[None,'full_g3_claim','missing_revision','wrong_schema'])
def test_limited_readiness_requires_explicit_revision_and_preserves_failed_full_G3(approved_revision,fault):
    runtime=numerics_readiness_template({'fixture':'a'*64})
    runtime.update(schema='lite3_bounded_numerics_limited_readiness_v2',status='qualified_for_bounded_numerics',
        g0_passed=True,g1_passed=True,g2_passed=True,g3_passed=False,g4_collection_passed=True,
        limited_motion_evidence_passed=True,stage_revision=approved_revision,
        measured_collection_transitions_per_second=1000.)
    ref=approved_revision
    if fault=='full_g3_claim':runtime['g3_passed']=True
    if fault=='missing_revision':ref=None
    if fault=='wrong_schema':runtime['schema']='lite3_bounded_numerics_runtime_readiness_v1'
    if fault is None:assert validate_numerics_runtime_receipt(runtime,{'fixture':'a'*64},stage_revision=ref)==runtime
    else:
        with pytest.raises(ValueError):validate_numerics_runtime_receipt(runtime,{'fixture':'a'*64},stage_revision=ref)


@pytest.mark.parametrize('fault',[None,'missing','duplicate_ids','wrong_step','missing_ready','negative','slow','no_samples','natural'])
def test_G4_validates_actual_scheduled_and_natural_reset_measurements(tmp_path,monkeypatch,fault):
    plan=dict(delivery_job='g4',stage_revision={'explicit_CPU_contract_fixture':True},
        input_sha256={'fixture':'a'*64},dependency_sha256={'fixture':'b'*64},base_revision='c'*40,
        attempt_id='CPU_light_contract',scheduled_resets=copy.deepcopy(SCHEDULE))
    monkeypatch.setattr('rl_training.source_scan.delivery_stage.job_contract',lambda *a,**k:plan)
    rows=[dict(step=i,finite=True,wall_mono=i*.05) for i in range(1,289)]
    steps=tmp_path/'result.steps.jsonl';steps.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    samples=[r|dict(kind='scheduled',elapsed_s=.2) for r in copy.deepcopy(SCHEDULE)]
    ready=[r|dict(elapsed_s=.25) for r in copy.deepcopy(SCHEDULE)]
    if fault=='missing':samples.pop()
    if fault=='duplicate_ids':samples[0]['ids']=[0]*20
    if fault=='wrong_step':samples[0]['step']=48
    if fault=='missing_ready':ready.pop()
    if fault=='negative':samples[0]['elapsed_s']=-1.
    if fault=='slow':samples[0]['elapsed_s']=2.
    if fault=='no_samples':samples=[]
    if fault=='natural':samples.insert(0,dict(kind='natural',step=80,ids=[23],elapsed_s=.3))
    reset=[r['elapsed_s'] for r in samples]
    import numpy as np
    start=rows[47]['wall_mono'];end=rows[-1]['wall_mono']
    report=plan|dict(status='passed',simulation_control_steps=288,finite_controls=288,
        completed_diagnostic_transitions=34560,steps_written=288,
        steps_sha256=hashlib.sha256(steps.read_bytes()).hexdigest(),measured_start_mono=start,measured_end_mono=end,
        reset_latency_s=reset,reset_samples=samples,scheduled_ready_samples=ready,measured_reset_start=0,
        measured_reset_count=len(reset),measured_transitions=28800,
        measured_transitions_per_second=28800/(end-start),reset_p95_s=float(np.percentile(reset,95)) if reset else None)
    (tmp_path/'result.json').write_text(json.dumps(report))
    (tmp_path/'result.cleanup.json').write_text(json.dumps(dict(environment='returned',application='returned',persistence_errors=[])))
    result=light.validate(tmp_path,plan,dict(returncode=0,process_group_empty=True,timed_out=False,monitor_errors=[]))
    assert result['accepted']==(fault in (None,'natural')),result
    if result['accepted']:
        assert result['scheduled_reset_batches']==6 and result['scheduled_environment_resets']==120
        assert result['measured_natural_reset_batches']==(fault=='natural')
