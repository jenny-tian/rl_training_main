from copy import deepcopy
from pathlib import Path
import gzip
import json
import sys

import numpy as np
import pytest
import torch

from scale_cpu_fixture import payload,make_environment,install_tensor_backend,BODY_NAMES,GROUND,ROOT
from rl_training.source_scan.sparse_contact import compact_topology,ContactEncoder,reconstruct_frame


def buffers(num_envs=16,capacity=None):
    capacity=capacity or num_envs*2
    sensors=sorted(f'/World/envs/env_{i}/Robot/HR_FOOT' for i in range(num_envs))
    filters=[sensors+[GROUND] for _ in sensors]
    topology=compact_topology(sensors,filters,[GROUND],num_envs=num_envs,capacity=capacity)
    return topology,payload(sensors,filters,capacity)


@pytest.mark.parametrize('num_envs',[16,32,60,120])
def test_complete_nonzero_union_round_trips_at_all_scales(num_envs):
    topology,(net,matrix,data)=buffers(num_envs)
    # A same-environment, count-zero force below reconstruction tolerance still must survive.
    matrix[0,0,0]=1e-8;net[0,0]+=1e-8
    frame=ContactEncoder(topology).encode(net,matrix,data,step=13)
    result=reconstruct_frame(frame,topology)
    assert np.array_equal(result['matrix'],matrix) and np.array_equal(result['counts'],data[-2])
    assert [0,0] in frame['pair_indices']
    assert len(json.dumps(frame)) < matrix.nbytes+data[-2].nbytes+10000


def test_true_120_environment_17_body_dense_shapes_and_compact_topology():
    sensors=sorted(f'/World/envs/env_{i}/Robot/{name}' for i in range(120) for name in BODY_NAMES)
    targets=sensors+[f'/World/ground/collider_{i}' for i in range(24)]
    topology=compact_topology(sensors,[targets]*len(sensors),targets[-24:],num_envs=120,capacity=1048576)
    n,f=2040,2064;matrix=np.zeros((n,f,3),np.float32);counts=np.zeros((n,f),np.uint32)
    data=(np.full((1048576,1),np.nan,np.float32),np.full((1048576,3),np.nan,np.float32),
          np.full((1048576,3),np.nan,np.float32),np.full((1048576,1),np.nan,np.float32),counts,np.zeros_like(counts))
    frame=ContactEncoder(topology).encode(np.zeros((n,3),np.float32),matrix,data,step=1)
    reconstructed=reconstruct_frame(frame,topology)
    assert reconstructed['matrix'].shape==(2040,2064,3) and not reconstructed['matrix'].any()
    assert len(topology['filter_orders'])==1 and len(json.dumps(topology))<250000
    assert len(json.dumps(frame))<60000


@pytest.mark.parametrize('fault',['cross_force','cross_count','nonfinite_matrix','saturated','negative_count',
    'outside','overlap','detail_nan','bad_normal','pair_mismatch','net_mismatch','float64','missing_filter'])
def test_encoder_rejects_missing_or_corrupt_dense_evidence(fault):
    topology,(net,matrix,data)=buffers();forces,points,normals,distances,counts,starts=data
    last=topology['filter_count']-1
    if fault=='cross_force':matrix[0,1,0]=1e-8
    if fault=='cross_count':counts[0,1]=1
    if fault=='nonfinite_matrix':matrix[0,0,0]=np.nan
    if fault=='saturated':counts[0,last]=17
    if fault=='negative_count':counts=counts.astype(np.int64);counts[0,last]=-1;data=(*data[:4],counts,starts)
    if fault=='outside':starts[0,last]=topology['capacity']
    if fault=='overlap':starts[1,last]=starts[0,last]
    if fault=='detail_nan':points[0,0]=np.nan
    if fault=='bad_normal':normals[0]=0
    if fault=='pair_mismatch':matrix[0,last,2]+=1
    if fault=='net_mismatch':net[0,2]+=1
    if fault=='float64':matrix=matrix.astype(np.float64)
    if fault=='missing_filter':topology['filter_orders'][0].pop()
    with pytest.raises((ValueError,IndexError)):
        ContactEncoder(topology).encode(net,matrix,data,step=1)


@pytest.mark.parametrize('fault',['drop_force_only','duplicate_pair','drop_point','alter_point_force','alter_net','wrong_topology','all_zero_entry'])
def test_independent_decoder_rejects_omitted_or_changed_values(fault):
    topology,(net,matrix,data)=buffers();matrix[0,0,0]=1e-8;net[0,0]=1e-8
    frame=ContactEncoder(topology).encode(net,matrix,data,step=1)
    if fault=='drop_force_only':
        for key in ('pair_indices','pair_counts','pair_starts','pair_forces_N'):frame[key].pop(0)
    if fault=='duplicate_pair':frame['pair_indices'][1]=frame['pair_indices'][0]
    if fault=='drop_point':frame['detail_indices'].pop()
    if fault=='alter_point_force':frame['normal_forces_N'][0]+=1
    if fault=='alter_net':frame['net_forces_N'][0][2]+=1
    if fault=='wrong_topology':frame['topology_sha256']='a'*64
    if fault=='all_zero_entry':frame['pair_forces_N'][0]=[0,0,0]
    with pytest.raises(ValueError):reconstruct_frame(frame,topology)


@pytest.mark.parametrize('fault',[None,'empty','cross','nonfinite','backend_failure'])
def test_pinned_tensor_api_capture_restores_hook_and_keeps_fault_buffers(tmp_path,fault):
    from source_sparse_tensor_contact import SparseTensorContactCapture
    from source_shared_contact_report import capture_substeps
    env,interface,events,wrapped=make_environment();runtime=install_tensor_backend(env,interface,fault=fault)
    records=[];sensors=sorted(f'/World/envs/env_{i}/Robot/{name}' for i in range(16) for name in BODY_NAMES)
    capture=SparseTensorContactCapture(physics_view=env.sim.physics_sim_view,ground_colliders=[GROUND],num_envs=16,
        physics_dt=.005,fault_path=tmp_path/'fault.npz',require_ground_positive=False,
        stage_id=73,reporting_bodies=sensors,record=records.append,start_step=12,decimation=4,control_steps=1)
    def execute():
        with capture_substeps(env,interface,capture,runtime):
            wrapped.step(torch.zeros(16,12));capture.check_control(env,1)
    if fault in (None,'empty'):
        execute();assert capture.samples==4 and not capture.positive_controls_validated
        assert not (tmp_path/'fault.npz').exists()
        for frame in [row for row in records if row.get('phase')=='sparse_contact_sample']:reconstruct_frame(frame,capture.topology)
    else:
        with pytest.raises(Exception,match='backend' if fault=='backend_failure' else 'cross-environment|nonfinite'):execute()
        assert capture.samples==0
        with np.load(tmp_path/'fault.npz',allow_pickle=False) as saved:
            if fault=='backend_failure':assert set(saved.files)=={'net'}
            else:assert saved['matrix'].shape==(272,273,3) and len(saved.files)==8
    assert 'record_post_physics_decimation_step' not in vars(env.recorder_manager)


def test_first_archived_real_contact_frame_reencodes_without_new_simulation():
    # This is a CPU transformation of previously archived real eight-env data.
    path=ROOT/'docs/lite3-source-live-ppo-integration-runtime-20261102/raw/result.events.jsonl.gz'
    if not path.exists():path=next((ROOT/'docs/lite3-source-live-ppo-integration-runtime-20261102/raw').glob('*.gz'))
    records=[];topology=None;sample=None
    with gzip.open(path,'rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('phase')=='tensor_contact_topology':topology=row
            if row.get('phase')=='tensor_contact':records.append(row)
            if row.get('phase')=='tensor_contact_sample':sample=row;break
    assert topology is not None and sample is not None and records
    compact=compact_topology(topology['sensor_paths'],topology['filter_paths'],topology['ground_colliders'],
        num_envs=8,capacity=65536)
    n,f=compact['sensor_count'],compact['filter_count'];counts=np.asarray(sample['counts'],np.uint32);starts=np.zeros_like(counts)
    forces=np.full((65536,1),np.nan,np.float32);points=np.full((65536,3),np.nan,np.float32)
    normals=np.full_like(points,np.nan);distances=np.full_like(forces,np.nan);offset=0
    for row in records:
        i=topology['sensor_paths'].index(row['actors'][0]);j=topology['filter_paths'][i].index(row['actors'][1]);count=row['count']
        starts[i,j]=offset
        for target,key in ((forces,'forces_N'),(points,'points_m'),(normals,'normals'),(distances,'separations_m')):target[offset:offset+count]=row[key]
        offset+=count
    matrix=np.asarray(sample['pair_forces_N'],np.float32);net=np.asarray(sample['net_forces_N'],np.float32)
    frame=ContactEncoder(compact).encode(net,matrix,(forces,points,normals,distances,counts,starts),step=sample['sim_step'])
    result=reconstruct_frame(frame,compact)
    assert np.array_equal(result['matrix'],matrix) and result['used_contact_count']==sample['used']
