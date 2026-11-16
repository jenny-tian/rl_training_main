"""只读历史起点 57，验证恢复候选的范围、动作偏移和输出梯度。"""
from pathlib import Path

import numpy as np
import torch

from reference_stairs_eval import ROOT,MODELS,load_actor,sha,save
from reference_low_stairs_reward_state_probe import DOCS,CHECKPOINT,DIGEST,read
from reference_low_stairs_motion_probe import OUTPUT as MOTION
from reference_low_stairs_trial import CHECKPOINT as PARENT
from reference_low_stairs_evaluate import EXPECTED_SHA
from reference_low_stairs_recovery_gate import recovery_mask,recovery_loss
from reference_low_stairs_preservation import ACTION_SCALES
from reference_core500_analyze import write_csv


def main():
    assert sha(CHECKPOINT)==DIGEST and sha(PARENT)==EXPECTED_SHA
    historic={r['path']:r['sha256'] for r in read(ROOT/'docs/lite3-low-stairs-motion-diagnosis-20261115/raw_manifest.json')}
    MODELS['recovery_teacher']=PARENT;MODELS['recovery_candidate']=CHECKPOINT
    teacher,actor=load_actor('recovery_teacher'),load_actor('recovery_candidate')
    frames,groups,files=[],[],[]
    inputs={}
    for policy in ('parent','zero'):
        path=MOTION/f'{policy}_training_randomized'
        for name in ('motion.npz','initial.npz','trace.npz','report.json'):
            f=path/name;assert sha(f)==historic[str(f)];files.append(dict(path=str(f),sha256=sha(f)))
        with np.load(path/'motion.npz') as m,np.load(path/'initial.npz') as initial,np.load(path/'trace.npz') as tr:
            obs=m['observation'][:100];gravity=m['gravity'][:100];active=tr['active'][:100]
            q=initial['root'][:,3:7]
            tilt0=np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1.,1.))
            tilt=np.concatenate((tilt0[None],np.arccos(np.clip(-gravity[:-1,:,2],-1.,1.))))
            command=torch.tensor(obs[...,6:9]);tilt_t=torch.tensor(tilt)
            mask=recovery_mask(command,tilt_t).numpy() & active
            record=next(r for r in read(path/'report.json')['records'] if r['env_id']==57)
            inputs[policy]=obs[0,57].copy()
            count=int(active[:,57].sum())
            with torch.no_grad():
                selected_obs=torch.tensor(obs[:count,57]);pred=actor(selected_obs);target=teacher(selected_obs)
            prediction=pred.clone().requires_grad_(True)
            loss=recovery_loss(prediction,target,command[:count,57],tilt_t[:count,57]);loss.backward()
            grad=prediction.grad;eligible=mask[:count,57]
            assert not grad[~eligible].any()
            assert (grad[eligible].abs().sum(-1)>0.).all() and all(p.grad is None for p in actor.parameters())
            delta=(pred-target)*pred.new_tensor(ACTION_SCALES)
            for k in range(count):
                frames.append(dict(policy=policy,env_id=57,action_time_s=k*.02,
                    input_tilt_deg=float(np.rad2deg(tilt[k,57])),selected=bool(eligible[k]),
                    command_vx=float(command[k,57,0]),command_vy=float(command[k,57,1]),command_wz=float(command[k,57,2]),
                    joint_target_rmse_rad=float(delta[k].square().mean().sqrt()),
                    gradient_norm=float(grad[k].norm()),post_step_terminated=bool(tr['terminated'][k,57])))
            groups.append(dict(policy=policy,source_outcome=record['outcome'],source_duration_s=record['duration_s'],
                zero_command_frames=int((np.abs(obs[...,6:9]).max(-1)<=1e-5)[active].sum()),
                all_initial_frames=int(active.sum()),all_selected_frames=int(mask.sum()),
                selected_env_ids=np.flatnonzero(mask.any(0)).tolist(),
                env57_frames=count,env57_selected_frames=int(eligible.sum()),
                first_input_tilt_deg=float(np.rad2deg(tilt0[57])),first_target_rmse_rad=float(delta[0].square().mean().sqrt()),
                first_selected=bool(eligible[0]),output_gradient_only_selected=True,loss=float(loss)))
    assert np.array_equal(inputs['parent'],inputs['zero'])
    assert sha(CHECKPOINT)==DIGEST and sha(PARENT)==EXPECTED_SHA
    write_csv(DOCS/'recovery_gate_frames.csv',frames)
    save(DOCS/'recovery_gate_audit.json',dict(status='passed',optimizer_updates=0,new_simulation_episodes=0,
        teacher_sha256=EXPECTED_SHA,candidate_sha256=DIGEST,training_data_created=False,
        lower_tilt_deg=10.,upper_tilt_deg=30.,zero_command_epsilon=1e-5,groups=groups,sources=files,
        implementation_sources=[dict(path=str(p),sha256=sha(p)) for p in
            (Path(__file__),Path(__file__).with_name('reference_low_stairs_recovery_gate.py'))],
        limitation='只验证门控和动作输出梯度；共享参数更新仍可能影响其他状态，尚未证明恢复有效。'))
    print(groups,flush=True)


if __name__=='__main__':main()
