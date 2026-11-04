from copy import deepcopy
import numpy as np
import pytest
import torch

from test_sparse_scale_contact import buffers
from rl_training.source_scan.sparse_contact import ContactEncoder,ContactDecoder
from rl_training.source_scan.fast_sparse_contact import FastContactEncoder,numpy_digest,tensor_digest


def tensors(values):return tuple(torch.from_numpy(v.copy()) for v in values)


@pytest.mark.parametrize('n',[16,32,60,120])
def test_fast_path_reconstructs_identical_dense_arrays_and_all_legacy_sparse_values(n):
    top,(net,matrix,data)=buffers(n)
    matrix[0,0,0]=1e-8;net[0,0]+=1e-8
    old=ContactEncoder(top).encode(net,matrix,data,step=1)
    new=FastContactEncoder(top,allow_cpu=True).encode_tensors(*tensors((net,matrix)),tensors(data),step=1)
    for k in old:
        if k not in ('schema','matrix_sha256','counts_sha256','net_sha256','max_net_residual_N'):assert new[k]==old[k],k
    decoded=ContactDecoder(top).decode(new)
    assert np.array_equal(decoded['matrix'],matrix) and np.array_equal(decoded['counts'],data[-2])
    assert new['native_device']=='cpu' and not torch.cuda.is_initialized()


@pytest.mark.parametrize('shape',[(0,3),(1,3),(341,3),(1024,),(1025,),(16,17,3)])
@pytest.mark.parametrize('dtype',[np.float32,np.uint32])
def test_independent_cpu_and_tensor_block_roots_cover_padding_and_dtype(shape,dtype):
    a=np.zeros(shape,dtype)
    if a.size:a.flat[0]=1;a.flat[-1]=2
    assert numpy_digest(a)==tensor_digest(torch.from_numpy(a),counts=dtype==np.uint32)
    if a.size:
        previous=numpy_digest(a);a.flat[a.size//2]=3
        assert numpy_digest(a)!=previous


def test_signed_zero_is_canonical_but_subnormal_bits_are_retained():
    a=np.asarray([-0.,0.,np.nextafter(np.float32(0),np.float32(1))],np.float32)
    assert numpy_digest(a)==tensor_digest(torch.from_numpy(a))
    b=a.copy();b[0]=0.;assert numpy_digest(a)==numpy_digest(b)
    b[2]=0.;assert numpy_digest(a)!=numpy_digest(b)


@pytest.mark.parametrize('fault',['cross_force','cross_count','nan_matrix','saturated','negative_count',
    'outside','overlap','detail_nan','normal','pair_mismatch','net_mismatch'])
def test_fast_dense_and_live_detail_checks_keep_all_rejection_conditions(fault):
    top,(net,matrix,data)=buffers();forces,points,normals,distances,counts,starts=data
    last=top['filter_count']-1
    if fault=='cross_force':matrix[0,1,0]=1e-8
    if fault=='cross_count':counts[0,1]=1
    if fault=='nan_matrix':matrix[0,0,0]=np.nan
    if fault=='saturated':counts[0,last]=top['capacity']
    if fault=='negative_count':counts=counts.astype(np.int32);counts[0,last]=-1;data=(*data[:4],counts,starts)
    if fault=='outside':starts[0,last]=top['capacity']
    if fault=='overlap':starts[1,last]=starts[0,last]
    if fault=='detail_nan':points[0,0]=np.nan
    if fault=='normal':normals[0]=0.
    if fault=='pair_mismatch':matrix[0,last,2]+=1
    if fault=='net_mismatch':net[0,2]+=1
    with pytest.raises(ValueError):FastContactEncoder(top,allow_cpu=True).encode_tensors(*tensors((net,matrix)),tensors(data),step=1)


@pytest.mark.parametrize('fault',['omitted_force_only','omitted_point','altered_net','altered_root','wrong_scheme'])
def test_independent_decoder_rejects_compaction_omissions_and_changed_roots(fault):
    top,(net,matrix,data)=buffers();matrix[0,0,0]=1e-8;net[0,0]+=1e-8
    frame=FastContactEncoder(top,allow_cpu=True).encode_tensors(*tensors((net,matrix)),tensors(data),step=1)
    if fault=='omitted_force_only':
        for key in ('pair_indices','pair_counts','pair_starts','pair_forces_N'):frame[key].pop(0)
    if fault=='omitted_point':frame['detail_indices'].pop()
    if fault=='altered_net':frame['net_forces_N'][0][2]+=1
    if fault=='altered_root':frame['matrix_sha256']='0'*64
    if fault=='wrong_scheme':frame['dense_digest_scheme']='not_sha256'
    with pytest.raises(ValueError):ContactDecoder(top).decode(frame)


def test_cpu_tensors_never_enter_the_real_cuda_only_encoder():
    top,(net,matrix,data)=buffers()
    with pytest.raises(ValueError,match='requires CUDA'):
        FastContactEncoder(top).encode_tensors(*tensors((net,matrix)),tensors(data),step=1)


def test_fast_capture_and_independent_G3_review_preserve_reset_and_action_evidence(tmp_path,monkeypatch):
    from scale_motion_cpu_fixture import build_fixture,CHECKPOINT
    from source_scale_motion_validation import validate
    original=FastContactEncoder.__init__
    def cpu_fixture(self,topology,**kwargs):original(self,topology,allow_cpu=True)
    monkeypatch.setattr(FastContactEncoder,'__init__',cpu_fixture)
    folder=tmp_path/'synthetic'
    plan,execution=build_fixture(folder,fast_contacts=True)
    result=validate(folder,plan,execution,CHECKPOINT,cpu_profile=(5,4))
    assert result.get('cpu_evidence_validated') is True,result
    assert result['accepted'] is False and result['motion_integrity_qualified'] is False
    assert not torch.cuda.is_initialized()
