"""Nominal FK of deployed USD joint frames, without AppLauncher or physics."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from pxr import Gf, Usd, UsdPhysics

from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES, physical_targets_from_action
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.structured_code import fixed_joint_basis


def pose(position, rotation):
    result = np.eye(4)
    result[:3,:3] = np.asarray(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(rotation))),dtype=float).T
    result[:3,3] = position
    return result


def forward(joints, targets):
    transforms = {'/Lite3/TORSO': np.eye(4)}
    pending = list(joints)
    while pending:
        ready = [j for j in pending if j['parent'] in transforms]
        if not ready:
            raise ValueError('disconnected/cyclic joint graph')
        for j in ready:
            rotation = np.eye(4)
            if j['revolute']:
                vector = {'X':Gf.Vec3d(1,0,0),'Y':Gf.Vec3d(0,1,0),'Z':Gf.Vec3d(0,0,1)}[j['axis']]
                rotation[:3,:3] = np.asarray(Gf.Matrix3d(Gf.Rotation(vector, np.degrees(targets[j['name']])))).T
            if j['child'] in transforms:
                raise ValueError('multiple parents')
            transforms[j['child']] = transforms[j['parent']] @ np.array(j['parent_anchor']) @ rotation @ np.linalg.inv(j['child_anchor'])
            pending.remove(j)
    return {leg:transforms[f'/Lite3/{leg}_FOOT'][:3,3] for leg in ('FL','FR','HL','HR')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    p=json.loads(Path(args.protocol).read_text())
    assert p['schema']=='lite3_fixed_code_kinematic_audit_v1'
    for path,digest in p['input_hashes'].items():
        assert sha256_file(path)==digest
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    stage=Usd.Stage.Open(p['asset'])
    joints=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):continue
        j=UsdPhysics.Joint(prim)
        parents,children=j.GetBody0Rel().GetTargets(),j.GetBody1Rel().GetTargets()
        if not parents:continue
        assert len(parents)==len(children)==1
        revolute=prim.IsA(UsdPhysics.RevoluteJoint)
        assert revolute or prim.IsA(UsdPhysics.FixedJoint)
        joints.append({'name':prim.GetName(),'parent':str(parents[0]),'child':str(children[0]),
            'revolute':revolute,'axis':UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() if revolute else None,
            'parent_anchor':pose(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get()).tolist(),
            'child_anchor':pose(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get()).tolist()})
    assert {j['name'] for j in joints if j['revolute']}==set(LITE3_POLICY_JOINT_NAMES)
    base=torch.tensor(p['default_action'],dtype=torch.float64)
    def foot_positions(action):
        q=physical_targets_from_action(action).tolist()
        return forward(joints,dict(zip(LITE3_POLICY_JOINT_NAMES,q)))
    baseline=foot_positions(base)
    cases=[]
    for code in p['codes']:
        residual=torch.tensor(p['normal_limits'],dtype=torch.float64)*torch.tanh(fixed_joint_basis(dtype=torch.float64)@torch.tensor(code,dtype=torch.float64))*p['gain']
        feet=foot_positions(base+residual)
        cases.append({'code':code,'requested_normalized_residual_after_gain':residual.tolist(),
                      'foot_delta_body_m':{leg:(feet[leg]-baseline[leg]).tolist() for leg in feet}})
    for path,digest in p['input_hashes'].items():assert sha256_file(path)==digest
    report={'status':'complete','protocol_sha256':sha256_file(args.protocol),'script_sha256':sha256_file(__file__),
            'input_hashes':p['input_hashes'],'joint_frames':joints,'baseline_foot_body_m':{k:v.tolist() for k,v in baseline.items()},
            'cases':cases,'new_physics_steps':0,'new_optimizer_updates':0,'seconds':time.perf_counter()-start,'limits':p['claim']}
    write_json_atomic(out/'report.json',report)
    print(json.dumps(cases,indent=2))


if __name__=='__main__':main()
