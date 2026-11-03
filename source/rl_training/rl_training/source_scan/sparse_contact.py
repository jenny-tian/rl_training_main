"""Compact pair contact evidence. NumPy only; unused native buffers are undefined."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

SCHEMA = 'lite3_sparse_contact_frame_v1'
ATOL = RTOL = 1e-4


def owner(path):
    match = re.fullmatch(r'/World/envs/env_(\d+)/Robot/.+', path)
    return int(match[1]) if match else -1


def compact_topology(sensors, filter_rows, ground, *, num_envs, capacity):
    sensors, ground = list(sensors), sorted(ground)
    if (type(num_envs) is not int or num_envs < 1 or type(capacity) is not int or capacity < 2
            or not sensors or len(set(sensors)) != len(sensors) or not ground or len(set(ground)) != len(ground)
            or any(not path.startswith('/World/ground/') for path in ground)
            or {owner(path) for path in sensors} != set(range(num_envs)) or set(sensors)&set(ground)
            or len(filter_rows) != len(sensors)):
        raise ValueError('invalid complete contact inventory')
    targets = sorted(sensors)+ground
    lookup = {path:i for i,path in enumerate(targets)}
    orders, row_orders, unique = [], [], {}
    for row in filter_rows:
        if len(row) != len(targets) or set(row) != set(targets):
            raise ValueError('contact filters omit or duplicate targets')
        order = tuple(lookup[path] for path in row)
        if order not in unique:
            unique[order] = len(orders); orders.append(list(order))
        row_orders.append(unique[order])
    return dict(schema='lite3_compact_contact_topology_v1', sensor_paths=sensors, target_paths=targets,
        filter_orders=orders, sensor_filter_order=row_orders, ground_colliders=ground,
        num_envs=num_envs, sensor_count=len(sensors), filter_count=len(targets), capacity=capacity)


def validate_topology(topology):
    rows = [[topology['target_paths'][index] for index in topology['filter_orders'][order]]
            for order in topology['sensor_filter_order']]
    actual = compact_topology(topology['sensor_paths'],rows,topology['ground_colliders'],
        num_envs=topology['num_envs'],capacity=topology['capacity'])
    if actual != topology: raise ValueError('noncanonical compact topology')
    return topology


def topology_digest(topology):
    return hashlib.sha256(json.dumps(topology,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _check(condition, message):
    if not condition: raise ValueError(message)


def dense_digest(value):
    """Canonical positive zeros; chunking bounds temporary CPU memory."""
    digest=hashlib.sha256()
    for start in range(0,len(value),64):
        chunk=np.ascontiguousarray(value[start:start+64]).copy()
        chunk[chunk==0]=0
        digest.update(chunk.tobytes())
    return digest.hexdigest()


class ContactEncoder:
    """Scan every dense pair, then persist the exact nonzero union and live points."""
    def __init__(self, topology):
        self.topology = validate_topology(topology)
        self.digest = topology_digest(topology)
        self.target_owners = np.array([owner(path) for path in topology['target_paths']])
        self.sensor_owners = np.array([owner(path) for path in topology['sensor_paths']])
        self.orders = np.asarray(topology['filter_orders'],dtype=np.int64)
        self.row_orders = np.asarray(topology['sensor_filter_order'],dtype=np.int64)

    def encode(self, net, matrix, data, *, step):
        t = self.topology; n,f,capacity = (t[key] for key in ('sensor_count','filter_count','capacity'))
        _check(type(step) is int and step >= 0,'invalid contact step')
        _check(len(data)==6,'six contact buffers required')
        forces,points,normals,distances,counts,starts = data
        shapes = ((n,3),(n,f,3),(capacity,1),(capacity,3),(capacity,3),(capacity,1),(n,f),(n,f))
        values = (net,matrix,*data)
        _check(all(isinstance(v,np.ndarray) and v.shape==s for v,s in zip(values,shapes)),'contact buffer shape mismatch')
        _check(all(v.dtype==np.float32 for v in values[:6]),'contact floats must be native float32')
        _check(counts.dtype.kind in 'iu' and starts.dtype.kind in 'iu','contact offsets/counts must be integers')
        _check(np.isfinite(net).all() and np.isfinite(matrix).all(),'nonfinite dense contact force')
        _check(not (counts<0).any() and not (counts>capacity).any(),'invalid contact counts')
        used = int(counts.sum(dtype=np.int64))
        _check(used<capacity,'saturated contact buffer: completeness unknown')
        pair_indices = np.argwhere((counts!=0)|np.any(matrix!=0,axis=2))
        i,j = pair_indices.T if len(pair_indices) else (np.array([],int),np.array([],int))
        target_indices = self.orders[self.row_orders[i],j]
        other = self.target_owners[target_indices]
        _check(not ((other>=0)&(other!=self.sensor_owners[i])).any(),'cross-environment contact, including force-only pairs')
        c = counts[i,j].astype(np.int64); s = starts[i,j].astype(np.int64)
        live = c>0
        _check(not (s[live]<0).any() and not (s[live]+c[live]>capacity).any(),'live range outside contact capacity')
        sorted_ranges = sorted(zip(s[live].tolist(),c[live].tolist()))
        _check(all(a+length<=b for (a,length),(b,_) in zip(sorted_ranges,sorted_ranges[1:])),'overlapping contact ranges')
        indices = np.concatenate([np.arange(start,start+count) for start,count in zip(s[live],c[live])]) if live.any() else np.array([],int)
        _check(len(indices)==used,'contact detail count mismatch')
        details = [v[indices] for v in (forces,points,normals,distances)]
        _check(all(np.isfinite(v).all() for v in details),'nonfinite live contact detail')
        _check(np.allclose(np.linalg.norm(details[2],axis=1),1,atol=1e-3,rtol=0),'contact normals not unit')
        reconstructed = np.zeros((len(i),3),np.float64)
        if used:
            offsets = np.concatenate(([0],np.cumsum(c[live])[:-1]))
            reconstructed[live] = np.add.reduceat((details[0]*details[2]).astype(np.float64),offsets,axis=0)
        pair_forces = matrix[i,j]
        _check(np.allclose(reconstructed,pair_forces,atol=ATOL,rtol=RTOL),'detail forces disagree with dense pair matrix')
        summed = matrix.sum(axis=1,dtype=np.float64)
        _check(np.allclose(summed,net,atol=ATOL,rtol=RTOL),'dense pair matrix disagrees with net force')
        return dict(schema=SCHEMA,sim_step=step,topology_sha256=self.digest,
            matrix_sha256=dense_digest(matrix),counts_sha256=dense_digest(counts.astype(np.uint32)),
            net_sha256=dense_digest(net),
            pair_indices=pair_indices.tolist(),pair_counts=c.tolist(),pair_starts=np.where(live,s,0).tolist(),
            pair_forces_N=pair_forces.tolist(),detail_indices=indices.tolist(),
            normal_forces_N=details[0][:,0].tolist(),points_m=details[1].tolist(),normals=details[2].tolist(),
            separations_m=details[3][:,0].tolist(),net_forces_N=net.tolist(),used_contact_count=used,
            max_pair_residual_N=float(np.abs(reconstructed-pair_forces).max(initial=0)),
            max_net_residual_N=float(np.abs(summed-net).max(initial=0)))


def reconstruct_frame(frame, topology):
    """Independent CPU decoder: explicit pair/point loops, no encoder invocation."""
    validate_topology(topology)
    _check(frame['schema']==SCHEMA and frame['topology_sha256']==topology_digest(topology),'frame topology binding differs')
    n,f,capacity = (topology[key] for key in ('sensor_count','filter_count','capacity'))
    matrix = np.zeros((n,f,3),np.float32); counts = np.zeros((n,f),np.uint32)
    detail_indices = frame['detail_indices']
    _check(all(type(v) is int and 0<=v<capacity for v in detail_indices),'invalid detail index')
    _check(len(set(detail_indices))==len(detail_indices)<capacity,'duplicate or saturated detail indices')
    k = len(detail_indices)
    _check(all(len(frame[key])==k for key in ('normal_forces_N','points_m','normals','separations_m')),'detail lengths differ')
    details = {}
    for offset,index in enumerate(detail_indices):
        force = frame['normal_forces_N'][offset]; point = frame['points_m'][offset]
        normal = frame['normals'][offset]; distance = frame['separations_m'][offset]
        _check(len(point)==len(normal)==3 and all(type(v) in (int,float) and np.isfinite(v)
            for v in [force,*point,*normal,distance]),'invalid detailed point')
        _check(abs(sum(v*v for v in normal)**.5-1)<=1e-3,'invalid decoded normal')
        details[index] = (force,normal)
    pairs = frame['pair_indices']; length = len(pairs)
    _check(all(len(frame[key])==length for key in ('pair_counts','pair_starts','pair_forces_N')),'pair lengths differ')
    seen,used,ground = set(),set(),{str(i):0 for i in range(topology['num_envs'])}
    max_residual = 0.
    for pair,count,start,force in zip(pairs,frame['pair_counts'],frame['pair_starts'],frame['pair_forces_N']):
        _check(len(pair)==2 and all(type(v) is int for v in pair),'invalid pair index')
        i,j = pair
        _check(0<=i<n and 0<=j<f and (i,j) not in seen,'duplicate/outside pair')
        seen.add((i,j))
        _check(type(count) is int and 0<=count<capacity and type(start) is int and 0<=start<=capacity-count,'invalid pair range')
        _check(len(force)==3 and all(type(v) in (int,float) and np.isfinite(v) for v in force),'invalid decoded pair force')
        _check(count>0 or any(v!=0 for v in force),'noncanonical all-zero sparse entry')
        _check(count>0 or start==0,'unused start must be canonical zero')
        sensor = topology['sensor_paths'][i]
        target = topology['target_paths'][topology['filter_orders'][topology['sensor_filter_order'][i]][j]]
        _check(owner(target)<0 or owner(target)==owner(sensor),'decoded cross-environment contact')
        expected = np.zeros(3,np.float64)
        for index in range(start,start+count):
            _check(index in details and index not in used,'missing/overlapping detailed contact')
            used.add(index); magnitude,normal = details[index]
            expected += float(magnitude)*np.asarray(normal,dtype=np.float64)
            if target in topology['ground_colliders'] and abs(magnitude)>ATOL: ground[str(owner(sensor))] += 1
        _check(np.allclose(expected,force,atol=ATOL,rtol=RTOL),'decoded pair force does not reconstruct')
        max_residual = max(max_residual,float(np.abs(expected-force).max()))
        matrix[i,j],counts[i,j] = force,count
    _check(used==set(detail_indices) and frame['used_contact_count']==len(used),'unreferenced/missing detail evidence')
    net = np.asarray(frame['net_forces_N'],np.float32)
    _check(net.shape==(n,3) and np.isfinite(net).all(),'invalid decoded net force')
    _check(np.allclose(matrix.sum(axis=1,dtype=np.float64),net,atol=ATOL,rtol=RTOL),'decoded pair/net force mismatch')
    for key,value in (('matrix_sha256',matrix),('counts_sha256',counts),('net_sha256',net)):
        _check(frame.get(key)==dense_digest(value),'decoded dense digest differs: '+key)
    return dict(matrix=matrix,counts=counts,net=net,ground_contact_points=ground,
                max_pair_residual_N=max_residual,used_contact_count=len(used))
