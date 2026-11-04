"""GPU dense checks with lossless sparse transfer and full-array SHA256 block roots."""
import hashlib
import json
import math

import numpy as np
import torch

from .sparse_contact import ContactEncoder,_check,ATOL,RTOL,_reconstruct_frame

SCHEMA='lite3_sparse_contact_frame_v2'
SCHEME='sha256_blocks_4096_v1'
WORDS=1024
ZERO=hashlib.sha256(bytes(4096)).digest()


def _root(shape,kind,indices,blocks):
    count=(math.prod(shape)*4+4095)//4096
    hashes=[ZERO]*count
    for index,block in zip(indices,blocks):
        if not 0<=int(index)<count or len(block)!=4096:raise ValueError('invalid dense digest block')
        hashes[int(index)]=hashlib.sha256(block).digest()
    header=json.dumps(dict(scheme=SCHEME,shape=list(shape),dtype=kind,block_bytes=4096),
                      sort_keys=True,separators=(',',':')).encode()+b'\n'
    return hashlib.sha256(header+b''.join(hashes)).hexdigest()


def numpy_digest(value):
    if value.dtype not in (np.dtype('float32'),np.dtype('uint32')) or not value.flags.c_contiguous:
        raise ValueError('dense digest requires contiguous native float32/uint32')
    if value.dtype==np.float32 and not np.isfinite(value).all():raise ValueError('nonfinite dense digest value')
    raw=value.view(np.uint32).reshape(-1)
    if value.dtype==np.float32 and np.any(raw==0x80000000):
        raw=raw.copy();raw[raw==0x80000000]=0
    padded=np.zeros(((len(raw)+WORDS-1)//WORDS)*WORDS,np.uint32);padded[:len(raw)]=raw
    blocks=padded.reshape(-1,WORDS);indices=np.flatnonzero(np.any(blocks!=0,axis=1))
    return _root(value.shape,'float32' if value.dtype==np.float32 else 'uint32',indices,
                 [blocks[i].astype('<u4',copy=False).tobytes() for i in indices])


def tensor_digest(value,*,counts=False):
    """Select nonzero blocks independently of the sparse pair/point extraction."""
    if value.dtype not in (torch.float32,torch.int32,torch.uint32):raise ValueError('unsupported native digest dtype')
    if not counts and (value.dtype!=torch.float32 or not bool(torch.isfinite(value).all())):
        raise ValueError('nonfinite native dense tensor')
    words=value.contiguous().view(torch.int32).reshape(-1)
    if not counts:words=torch.where(words==-(2**31),0,words)
    pad=(-len(words))%WORDS
    if pad:words=torch.cat((words,words.new_zeros(pad)))
    blocks=words.reshape(-1,WORDS);indices=(blocks!=0).any(dim=1).nonzero().flatten()
    selected=blocks.index_select(0,indices).cpu().numpy()
    return _root(value.shape,'uint32' if counts else 'float32',indices.cpu().tolist(),
                 [row.astype('<i4',copy=False).tobytes() for row in selected])


def native_tensor(value):
    if isinstance(value,torch.Tensor):return value
    if isinstance(value,np.ndarray):return torch.from_numpy(value)
    import warp as wp
    return wp.to_torch(value)


def snapshot(value):
    # PhysX reuses borrowed buffers across reads. Finish a device copy before
    # another native query; no full buffer crosses PCIe on the success path.
    result=native_tensor(value).clone()
    if result.is_cuda:torch.cuda.current_stream(result.device).synchronize()
    return result


class FastContactEncoder(ContactEncoder):
    def __init__(self,topology,*,allow_cpu=False):
        super().__init__(topology);self.allow_cpu=allow_cpu

    def encode_tensors(self,net,matrix,data,*,step):
        t=self.topology;n,f,capacity=(t[k] for k in ('sensor_count','filter_count','capacity'))
        _check(type(step) is int and step>=0,'invalid contact step')
        _check(len(data)==6,'six contact buffers required')
        forces,points,normals,distances,raw_counts,raw_starts=data
        values=(net,matrix,*data)
        shapes=((n,3),(n,f,3),(capacity,1),(capacity,3),(capacity,3),(capacity,1),(n,f),(n,f))
        _check(all(isinstance(v,torch.Tensor) and tuple(v.shape)==s for v,s in zip(values,shapes)),
               'native contact dimensions differ')
        _check(self.allow_cpu or all(v.is_cuda for v in values),'fast native contact requires CUDA')
        _check(len({str(v.device) for v in values})==1,'native contact devices differ')
        _check(all(v.dtype==torch.float32 for v in values[:6]),'native forces/details must be float32')
        _check(raw_counts.dtype in (torch.int32,torch.uint32) and raw_starts.dtype in (torch.int32,torch.uint32),
               'native count/offset dtype differs')
        counts=raw_counts.view(torch.int32);starts=raw_starts.view(torch.int32)
        _check(bool(torch.isfinite(net).all()) and bool(torch.isfinite(matrix).all()),'nonfinite dense contact force')
        _check(not bool(((counts<0)|(counts>capacity)).any()),'invalid contact counts')
        used=int(counts.sum(dtype=torch.int64));_check(used<capacity,'saturated contact buffer: completeness unknown')
        # Integer bit tests preserve subnormal force-only pairs and ignore only
        # +/- zero, independent of the GPU's floating-point denormal mode.
        nonzero=((matrix.view(torch.int32)&0x7fffffff)!=0).any(dim=2)
        ij=((counts!=0)|nonzero).nonzero();pairs=ij.cpu().numpy();i,j=pairs.T if len(pairs) else (np.array([],int),np.array([],int))
        c=counts[ij[:,0],ij[:,1]].to(torch.int64).cpu().numpy()
        s=starts[ij[:,0],ij[:,1]].to(torch.int64).cpu().numpy();live=c>0
        target=self.orders[self.row_orders[i],j];owners=self.target_owners[target]
        _check(not ((owners>=0)&(owners!=self.sensor_owners[i])).any(),'cross-environment contact, including force-only pairs')
        _check(not (s[live]<0).any() and not (s[live]+c[live]>capacity).any(),'live range outside contact capacity')
        ranges=sorted(zip(s[live].tolist(),c[live].tolist()))
        _check(all(a+length<=b for (a,length),(b,_) in zip(ranges,ranges[1:])),'overlapping contact ranges')
        indices=np.concatenate([np.arange(start,start+count) for start,count in zip(s[live],c[live])]) if live.any() else np.array([],np.int64)
        _check(len(indices)==used,'contact detail count mismatch')
        gather=torch.as_tensor(indices,dtype=torch.int64,device=matrix.device)
        details=[v.index_select(0,gather).cpu().numpy() for v in (forces,points,normals,distances)]
        _check(all(np.isfinite(v).all() for v in details),'nonfinite live contact detail')
        _check(np.allclose(np.linalg.norm(details[2],axis=1),1,atol=1e-3,rtol=0),'contact normals not unit')
        reconstructed=np.zeros((len(i),3),np.float64)
        if used:
            offsets=np.concatenate(([0],np.cumsum(c[live])[:-1]))
            reconstructed[live]=np.add.reduceat((details[0]*details[2]).astype(np.float64),offsets,axis=0)
        pair_forces=matrix[ij[:,0],ij[:,1]].cpu().numpy()
        _check(np.allclose(reconstructed,pair_forces,atol=ATOL,rtol=RTOL),'detail forces disagree with dense pair matrix')
        summed=matrix.sum(dim=1,dtype=torch.float64)
        _check(bool(torch.allclose(summed,net.to(torch.float64),atol=ATOL,rtol=RTOL)),
               'dense pair matrix disagrees with net force')
        return dict(schema=SCHEMA,dense_digest_scheme=SCHEME,native_device=str(matrix.device),
            sim_step=step,topology_sha256=self.digest,matrix_sha256=tensor_digest(matrix),
            counts_sha256=tensor_digest(raw_counts,counts=True),net_sha256=tensor_digest(net),
            pair_indices=pairs.tolist(),pair_counts=c.tolist(),pair_starts=np.where(live,s,0).tolist(),
            pair_forces_N=pair_forces.tolist(),detail_indices=indices.tolist(),normal_forces_N=details[0][:,0].tolist(),
            points_m=details[1].tolist(),normals=details[2].tolist(),separations_m=details[3][:,0].tolist(),
            net_forces_N=net.cpu().tolist(),used_contact_count=used,
            max_pair_residual_N=float(np.abs(reconstructed-pair_forces).max(initial=0)),
            max_net_residual_N=float((summed-net).abs().max()))


def reconstruct(frame,topology,digest):
    _check(frame.get('dense_digest_scheme')==SCHEME,'unknown dense contact digest scheme')
    return _reconstruct_frame(frame,topology,digest,dense_digest_fn=virtual_digest,frame_schema=SCHEMA,array_factory=VirtualDense)


class VirtualDense:
    """Exact zero-filled array reconstruction; materialize only if explicitly requested."""
    def __init__(self,shape,dtype):
        self.shape=shape;self.dtype=np.dtype(dtype);self.values={};self.blocks={}
    def __setitem__(self,index,value):
        array=np.asarray(value,dtype=self.dtype).reshape(self.shape[2:])
        self.values[index]=array.copy()
        words=array.reshape(-1).view(np.uint32).copy()
        if self.dtype==np.float32:words[words==0x80000000]=0
        start=(index[0]*self.shape[1]+index[1])*(3 if len(self.shape)==3 else 1)
        for offset,word in enumerate(words):
            if word:
                position=start+offset;block=position//WORDS
                if block not in self.blocks:self.blocks[block]=np.zeros(WORDS,np.uint32)
                self.blocks[block][position%WORDS]=word
    def sum(self,axis,dtype):
        if axis!=1 or len(self.shape)!=3:raise ValueError('unsupported virtual reconstruction reduction')
        result=np.zeros((self.shape[0],3),dtype)
        for (i,j),value in sorted(self.values.items()):result[i]+=value
        return result
    def __array__(self,dtype=None,copy=None):
        result=np.zeros(self.shape,self.dtype)
        for index,value in self.values.items():result[index]=value
        return result.astype(dtype,copy=False) if dtype is not None else result


def virtual_digest(value):
    if not isinstance(value,VirtualDense):return numpy_digest(value)
    indices=sorted(value.blocks)
    return _root(value.shape,'float32' if value.dtype==np.float32 else 'uint32',indices,
                 [value.blocks[i].astype('<u4',copy=False).tobytes() for i in indices])
