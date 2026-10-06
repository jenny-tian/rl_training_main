"""Extend only the pinned training episodes with earlier, lower-tilt states."""
import argparse
import json
from pathlib import Path
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=Path('docs/lite3_recovery_bank_20261005.json')
    old=json.loads(manifest.read_text())
    original=Path('artifacts/recovery_teacher_20261005/bank-v1-attempt02/bank.npz')
    assert sha256_file(original)==old['bank_sha256']
    with np.load(original,allow_pickle=False) as data:
        arrays={k:list(data[k]) for k in data.files}
    bins=[2]*len(old['episodes']);rows=[];provenance={str(manifest):sha256_file(manifest),str(original):sha256_file(original)}
    for case,episode in enumerate(old['episodes']):
        if episode['split']!='train':continue
        folder=Path(episode['source']).parent.parent;env_id=int(episode['id'].split('/')[-1])
        initial_path=folder/'initial_state.npz';initial=np.load(initial_path,allow_pickle=False)
        provenance[str(initial_path)]=sha256_file(initial_path)
        found={};offset=0
        for file in sorted((folder/'trajectory').glob('chunk-*.npz')):
            with np.load(file,allow_pickle=False) as d:
                obs=d['policy_observation'][:,env_id];tilt=np.rad2deg(np.arccos(np.clip(-obs[:,5],-1,1)))
                for b,(lo,hi) in enumerate(((10,15),(15,20))):
                    if b in found:continue
                    mask=d['active'][:,env_id]&(tilt>=lo)&(tilt<hi)&(np.arange(len(obs))+offset<episode['step'])
                    idx=np.flatnonzero(mask)
                    if not len(idx):continue
                    j=int(idx[0]);root=d['policy_root_state'][j,env_id].copy();root[:3]-=initial['origins'][env_id]
                    state=dict(root=root,joint_pos=obs[j,12:24]+np.array([0,-.65,1.3]*4,np.float32),joint_vel=obs[j,24:36]/np.float32(.05),terrain=episode['terrain'],validation=False)
                    assert all(np.isfinite(v).all() for v in state.values())
                    np.testing.assert_allclose(np.linalg.norm(root[3:7]),1,atol=2e-5)
                    found[b]=(state,dict(id=episode['id'],original_case=case,bin=b,step=offset+j,tilt_deg=float(tilt[j]),gravity=obs[j,3:6].tolist(),angular_velocity_world=root[10:13].tolist(),source=str(file),sha256=sha256_file(file)))
            offset+=len(obs)
            if len(found)==2:break
        assert len(found)==2, f'missing tier for training source {episode["id"]}; do not silently remove source'
        for b in (0,1):
            state,row=found[b];row['case']=len(bins);rows.append(row);bins.append(b)
            for k,v in state.items():arrays[k].append(v)
            provenance[row['source']]=row['sha256']
    bank={k:np.asarray(v) for k,v in arrays.items()};bank['tilt_bin']=np.asarray(bins)
    for k in arrays:
        with np.load(original,allow_pickle=False) as data:np.testing.assert_array_equal(bank[k][:27],data[k])
    assert len(rows)==40 and not bank['validation'][27:].any()
    np.savez_compressed(out/'bank.npz',**bank)
    write_json_atomic(out/'report.json',dict(status='complete',new_physics=0,bank_sha256=sha256_file(out/'bank.npz'),original_rows_unchanged=27,new_training_states=rows,provenance=provenance,scope='Two earlier states per original TRAIN source episode; original27 rows and validation membership unchanged. Kinematic states, no action labels or solver/contact memory.'))


if __name__=='__main__':main()
