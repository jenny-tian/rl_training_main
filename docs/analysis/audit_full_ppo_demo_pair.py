"""Read-only audit of six preselected Oct14 demonstration clips and actions."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from rl_training.policy_flat_anchor import FrozenFlatMean
from rl_training.lite3_joint_conventions import clip_lite3_policy_action


def main():
    root = Path.cwd()
    extension = json.loads((root/'docs/lite3_full_ppo_video_extension_protocol_20261014.json').read_text())
    runs = extension['existing_runs'] + extension['additional_runs']
    result = dict(schema='lite3_full_ppo_video_audit_v1', runs=[], charged=0,
        confirmed=0, uncertain=0, training_updates=0,
        scope='Preselected new demonstration episodes; not original59/60 footage or new qualification trials.')
    initial_hashes=[]
    torch.set_num_threads(1)
    for run in runs:
        protocol=json.loads((root/run['protocol']).read_text())
        report=json.loads((root/run['output']/'report.json').read_text())
        assert report['status']=='complete'
        ledger=report['ledger'];assert ledger['confirmed']==ledger['charged']<=run['max_controls']
        assert ledger['uncertain_transitions']==0 and ledger['pending'] is None
        phase=report['phases']['course-'+run['name']]
        folder=root/run['output']/('course-'+run['name'])
        index=protocol['video']['env_id'];episode=phase['episodes'][index]
        expected_terrain=str(run.get('terrain',107))
        assert episode['terrain']==expected_terrain
        model_spec=protocol['models'][run['name']]
        assert hashlib.sha256(Path(model_spec['path']).read_bytes()).hexdigest()==model_spec['sha256']
        model=FrozenFlatMean(torch.load(model_spec['path'],map_location='cpu',weights_only=False))
        obs=[];actions=[];roots=[];active=[];done=[];success=[];failure=[]
        for file in sorted((folder/'trajectory').glob('chunk-*.npz')):
            with np.load(file,allow_pickle=False) as a:
                for target,key in [(obs,'policy_observation'),(actions,'emitted_action'),
                    (roots,'post_root_state'),(active,'active'),(done,'done'),(success,'success'),(failure,'failure')]:
                    target.append(a[key][:,index].copy())
        obs,actions,roots,active,done,success,failure=map(np.concatenate,[obs,actions,roots,active,done,success,failure])
        assert len(obs)*protocol['num_envs']==ledger['confirmed']
        assert int(active.sum())==episode['steps']
        states=roots[:episode['steps']]
        with np.load(folder/'initial.npz',allow_pickle=False) as a:
            origin=a['origins'][index,1]
        y=states[:,1]-origin
        np.testing.assert_allclose(np.sqrt(np.mean(y*y)),episode['body_lateral_rms_m'],atol=2e-6)
        np.testing.assert_allclose(np.max(np.abs(y)),episode['max_body_lateral_m'],atol=2e-6)
        assert bool(success[episode['steps']-1])==episode['success']
        assert bool(done[episode['steps']-1])
        sampled=np.unique(np.linspace(0,episode['steps']-1,80).astype(int))
        predicted=model(torch.from_numpy(obs[sampled])).numpy()
        np.testing.assert_allclose(predicted,actions[sampled],atol=2e-5,rtol=2e-5)
        commands=obs[:,6:9]
        np.testing.assert_allclose(commands,np.broadcast_to([.3,0,0],commands.shape),atol=1e-7)
        # Action history: post-reset is zero; otherwise the prior emitted command.
        expected=actions[:-1].copy();expected[done[:-1].astype(bool)]=0
        live=active[1:].astype(bool)
        np.testing.assert_allclose(obs[1:,-12:][live],expected[live],atol=1e-6)
        initial_hashes.append(phase['initial_sha256'])
        result['charged']+=ledger['charged'];result['confirmed']+=ledger['confirmed']
        result['runs'].append(dict(name=run['name'],terrain=expected_terrain,output=run['output'],
            episode=episode,initial_sha256=phase['initial_sha256'],model_sha256=model_spec['sha256'],
            video=phase['video'],cpu_action_samples=len(sampled),cpu_actions_verified=True,
            metrics_recomputed=True,action_history_verified=True))
    assert len(set(initial_hashes))==1, 'All preselected runs must use exactly paired initial arrays and delays'
    assert result['charged']<=extension['max_controls_all_runs']
    result.update(status='verified',initial_states_and_actuator_delays_matched=True,
        run_count=len(runs),maximum_controls=extension['max_controls_all_runs'])
    path=root/'docs/lite3_full_ppo_video_final_audit_20261014.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
