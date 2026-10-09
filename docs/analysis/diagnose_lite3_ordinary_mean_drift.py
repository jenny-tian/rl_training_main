"""Post-evaluation action-reference error on actual ordinary visited states.

Pilot observations remain diagnostics only: no fitting or dataset export.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    checkpoint=Path('artifacts/retention_coverage_20261008/learner2247/train/final.pt')
    assert sha256_file(checkpoint)=='6a470cec8c1f1b1ad6f237c1ecc6401367528715ccca0f44b4f02b4860fa2ead'
    reference=FrozenPPOExpert(torch.load(checkpoint,map_location='cpu',weights_only=False)['actor_state_dict'])
    sources=dict(self_mean=('mean_anchor_long_20261009/pilot-evaluation','mean_anchor_long'),
        demonstration=('demo_target_20261009/pilot-evaluation-retry01','demo_target'),
        tail=('tail_anchor_20261009/pilot-evaluation','tail_anchor'))
    evidence={str(checkpoint):sha256_file(checkpoint)}
    results={}
    initial_hashes=[]
    for name,(folder,stem) in sources.items():
        root=Path('artifacts')/folder
        audit_path=Path(f'docs/lite3_{stem}_pilot_trajectory_audit_20261009.json')
        audit=json.loads(audit_path.read_text())
        report=json.loads((root/'report.json').read_text())
        assert audit['status']=='verified' and report['status']=='complete'
        assert audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
        initial_hashes.append(report['phases']['course-trial']['initial_sha256'])
        for path in (audit_path,root/'report.json'):evidence[str(path)]=sha256_file(path)
        worst=max((e for e in report['phases']['course-trial']['episodes'] if e['terrain']!='flat'),key=lambda e:e['max_body_lateral_m'])
        env_id=worst['env_id']
        with np.load(root/'course-trial/initial.npz',allow_pickle=False) as initial:
            start_y=float(initial['root'][env_id,1]-initial['origins'][env_id,1])
            origin_y=float(initial['origins'][env_id,1])
        all_errors,all_y,roots=[],[],[]
        for path in sorted((root/'course-trial/trajectory').glob('chunk-*.npz')):
            assert sha256_file(path)==audit['phases']['course-trial']['trace_sha256'][path.name]
            with np.load(path,allow_pickle=False) as d:
                active=d['active']
                x=d['policy_observation'][active]
                action=d['requested_action'][active]
                roots.append(d['post_root_state'][:,env_id][active[:,env_id]])
            errors=[]
            with torch.no_grad():
                for start in range(0,len(x),2048):
                    target=reference(torch.from_numpy(x[start:start+2048])).numpy()
                    errors.append(((action[start:start+2048]-target)**2).mean(-1))
            all_errors.append(np.concatenate(errors));all_y.append(x[:,9]*1.65)
        errors=np.concatenate(all_errors);y=np.concatenate(all_y)
        results[name]=dict(active_rows=len(y),mean_raw_action_mse=float(errors.mean()),
            p95_per_row_raw_action_mse=float(np.quantile(errors,.95)),
            rows_outside_old_ordinary_30cm_domain=int((np.abs(y)>.30).sum()),
            maximum_pre_action_abs_lateral_m=float(np.abs(y).max()))
        states=np.concatenate(roots)
        lateral=states[:,1]-origin_y
        peak=int(np.argmax(np.abs(lateral)))
        np.testing.assert_allclose(abs(lateral[peak]),worst['max_body_lateral_m'],atol=1e-6)
        states=states[:peak+1]
        w,x,qy,z=states[:,3:7].T
        yaw=np.arctan2(2*(w*z+x*qy),1-2*(qy*qy+z*z))
        vx,vy=states[:,7],states[:,8]
        results[name]['worst_rough_episode_before_peak']=dict(env_id=env_id,terrain=worst['terrain'],
            peak_time_s=(peak+1)*.02,initial_lateral_m=start_y,signed_peak_lateral_m=float(lateral[peak]),
            mean_yaw_deg=float(np.rad2deg(yaw).mean()),mean_world_vy_m_s=float(vy.mean()),
            mean_yaw_frame_vy_m_s=float((-np.sin(yaw)*vx+np.cos(yaw)*vy).mean()),
            velocity_integral_y_m=float(vy.sum()*.02),net_position_change_y_m=float(lateral[peak]-start_y))
    assert len(set(initial_hashes))==1
    write_json_atomic(args.output,dict(status='verified',results=results,evidence=evidence,new_controls=0,optimizer_steps=0,
        scope='Action MSE uses full active ordinary first episodes, including flat. Kinematics uses each model worst rough episode. Different state distributions and diagnostic slot selection, descriptive only. Pilot observations never become training data.'))


if __name__=='__main__':main()
