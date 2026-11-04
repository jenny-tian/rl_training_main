import base64
import copy
import math

import numpy as np
import pytest
import torch

from rl_training.source_scan.scan_footprint import scanner_boundary
from rl_training.source_scan.tensor_codec import pack,unpack,DTYPES
from rl_training.source_scan.scale_motion_evidence import CompressedJournal,event_records
from rl_training.source_scan.delivery_stage import LIMITS,STAGE,BASE,PROPOSAL,reference,validate_grant,claim_slot,validate_slot


def state(x=0.,y=0.,yaw=0.,vx=0.,vy=0.,wz=0.):
    return (torch.tensor([[x,y,.3]]),torch.tensor([[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]]),
            torch.tensor([[vx,vy,0.]]),torch.tensor([wz]),torch.zeros(1,3))


@pytest.mark.parametrize('x,y',[(0.,0.),(5.,0.),(9.,.5)])
def test_scanner_safe_footprints(x,y):assert not scanner_boundary(*state(x,y)).any()


@pytest.mark.parametrize('x,y,yaw',[(-.62400246,-.1870369,.19088265),(-.6,0.,0.),
    (9.6,0.,0.),(1.,1.4,0.),(1.,-1.4,0.),(1.,1.1,.8)])
def test_scanner_guard_covers_rear_front_sides_and_rotated_corners(x,y,yaw):
    assert scanner_boundary(*state(x,y,yaw)).all()


def test_lookahead_includes_translation_and_yaw():
    a=state(1.,1.,yaw=.4,wz=10.)
    d={};assert scanner_boundary(*a,diagnostics=d).all()
    assert not d['current'].any() and d['predicted'].all()
    assert scanner_boundary(*state(-.4,0.,vx=-10.)).all()


@pytest.mark.parametrize('field',range(5))
def test_guard_nonfinite_is_a_hard_error(field):
    args=list(state());args[field].flatten()[0]=float('nan')
    with pytest.raises(FloatingPointError):scanner_boundary(*args)


@pytest.mark.parametrize('name',DTYPES)
def test_binary_tensor_codec_preserves_each_supported_dtype(name):
    dtype=getattr(torch,name.split('.')[1]);t=torch.tensor([[0,1],[1,0]],dtype=dtype)
    record=dict(dtype=name,shape=[2,2],values=t.tolist())
    assert unpack(pack(record))==record


def test_codec_does_not_round_unrepresentable_evidence_or_accept_nonfinite():
    with pytest.raises(ValueError):pack(dict(dtype='torch.float32',shape=[1],values=[1.00000000001]))
    for value in (float('inf'),float('nan')):
        with pytest.raises(ValueError):pack(dict(dtype='torch.float32',shape=[1],values=[value]))


@pytest.mark.parametrize('fault',['shape','length','nonfinite','extra','dtype','base64'])
def test_decoder_rejects_corrupted_binary_payload(fault):
    r=pack(dict(dtype='torch.float32',shape=[1],values=[1.]))
    if fault=='shape':r['shape']=[2**60]
    elif fault=='length':r['data']=''
    elif fault=='nonfinite':r['data']=base64.b64encode(np.asarray([np.inf],dtype='<f4').tobytes()).decode()
    elif fault=='extra':r['extra']=0
    elif fault=='dtype':r['dtype']='torch.complex64'
    else:r['data']='!!!!'
    with pytest.raises(ValueError):unpack(r)


def test_compressed_binary_journal_is_lossless_with_multiple_members(tmp_path):
    rows=[dict(phase='tensor',record=dict(dtype='torch.float32',shape=[2],values=[-0.,1.25])),dict(phase='tail',n=2)]
    path=tmp_path/'events.gz'
    with CompressedJournal(path,byte_limit=10000,uncompressed_limit=10000,record_limit=10000,binary_tensors=True) as journal:
        for r in rows:journal.record(r);journal.flush()
    actual=list(event_records(path));assert actual==rows
    assert math.copysign(1,actual[0]['record']['values'][0])==-1


def grant(tmp_path):
    import hashlib,json,subprocess
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    payload=dict(schema='lite3_training_delivery_approval_v1',stage_id=STAGE,user_approved=True,
        user_message='同意',proposal_revision=BASE,
        proposal_sha256=hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+PROPOSAL],cwd=root)).hexdigest(),
        job_limits=copy.deepcopy(LIMITS),minimum_free_vram_mib=12288,run=1,max_updates=90,max_training_transitions=259200,
        slot_ledger=str(tmp_path/'slots.json'),output_root=str(tmp_path/'runs'))
    path=tmp_path/'approval.json';path.write_text(json.dumps(payload))
    return path,payload


@pytest.mark.parametrize('fault',['approval','updates','job_wall','extra_job','training_budget'])
def test_stage_grant_cannot_expand_user_approved_scope(tmp_path,fault):
    import json
    path,payload=grant(tmp_path)
    if fault=='approval':payload['user_approved']=False
    elif fault=='updates':payload['max_updates']=91
    elif fault=='job_wall':payload['job_limits']['g3']['wall_s']=7200
    elif fault=='extra_job':payload['job_limits']['retry']=payload['job_limits']['g3']
    else:payload['max_training_transitions']=262144
    path.write_text(json.dumps(payload))
    with pytest.raises(PermissionError):validate_grant(reference(path,'stage_approval'))


def test_stage_slot_is_single_use_and_binds_the_frozen_manifest(tmp_path):
    import json
    path,_=grant(tmp_path);ref=reference(path,'stage_approval')
    plan=dict(delivery_stage=ref,delivery_job='g1');manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(plan))
    claim_slot(ref,'g1',manifest);validate_slot(plan)
    with pytest.raises(PermissionError):claim_slot(ref,'g1',manifest)
    manifest.write_text('{}')
    with pytest.raises(ValueError):validate_slot(plan)


def test_cached_contact_decoder_owns_its_topology_and_still_checks_every_frame():
    from test_sparse_scale_contact import buffers
    from rl_training.source_scan.sparse_contact import ContactEncoder,ContactDecoder
    topology,(net,matrix,data)=buffers()
    frame=ContactEncoder(topology).encode(net,matrix,data,step=1);decoder=ContactDecoder(topology)
    topology['sensor_paths'][0]='changed externally'
    assert np.array_equal(decoder.decode(frame)['matrix'],matrix)
    frame['net_forces_N'][0][2]+=1
    with pytest.raises(ValueError):decoder.decode(frame)


def test_motion_topology_is_rechecked_after_each_reset_without_rebuilding_during_motion():
    from types import SimpleNamespace as NS
    from source_scale_motion_collection import MotionContactCapture
    capture=object.__new__(MotionContactCapture);capture._checked_assignments=None
    capture.start_step=0;capture.decimation=4;capture.control_poll_counts=[];capture.topology={'fixed':True}
    calls=[];capture._topology=lambda:(calls.append(1) or capture.topology)
    env=NS(_source_parallel_terrain_allocator=NS(assignments=(1,)))
    for step in (1,2,3):
        if step==3:env._source_parallel_terrain_allocator.assignments=(2,)
        capture.poll_steps=list(range(1,4*step+1));env._sim_step_counter=4*step
        capture.check_control(env,step)
    assert len(calls)==2
    env._source_parallel_terrain_allocator.assignments=(3,);env._sim_step_counter=16
    capture.poll_steps=list(range(1,17));capture._topology=lambda:{'wrong':True}
    with pytest.raises(ValueError):capture.check_control(env,4)


@pytest.mark.parametrize('fault',[None,'missing_gate','formal_claim','formal_protocol','extra_update','wrong_job','unapproved'])
def test_numeric_only_schema_keeps_actual_gate_budget_and_scope_requirements(tmp_path,monkeypatch,fault):
    from test_cumulative_training import fixture_authorization
    from rl_training.source_scan.cumulative_training import NUMERICS_SCHEMA
    from rl_training.source_scan.training_runner import validate_training_authorization
    import rl_training.source_scan.delivery_stage as delivery
    ledger,plan,kwargs=fixture_authorization(tmp_path)
    path,_=grant(tmp_path)
    plan.pop('source_selection_lock');plan.pop('metrics_qualification')
    plan.update(schema=NUMERICS_SCHEMA,delivery_stage=reference(path,'stage_approval'),delivery_job='numerics',
        predecessor_qualification={'explicit_cpu_contract_fixture':True},formal_selection_authorized=False,metrics_scope='diagnostic_only')
    calls=[]
    def fake_gate(ref,kind,*,inputs):
        calls.append(kind)
        assert kind=='g4' and inputs==plan['input_sha256']
        if fault=='missing_gate':raise ValueError('CPU fixture: real gate missing')
    monkeypatch.setattr(delivery,'validate_receipt',fake_gate)
    if fault=='formal_claim':plan['formal_selection_authorized']=True
    if fault=='formal_protocol':plan['source_selection_lock']={'not_allowed':True}
    if fault=='extra_update':plan['max_updates']=91
    if fault=='wrong_job':plan['delivery_job']='pilot'
    if fault=='unapproved':plan['user_approved']=False
    if fault is None:
        lock=validate_training_authorization(plan,**kwargs)
        assert lock['max_updates']==90 and lock['max_transitions']==259200 and calls==['g4']
        assert lock['formal_selection_authorized'] is False
    else:
        with pytest.raises((ValueError,PermissionError)):validate_training_authorization(plan,**kwargs)
