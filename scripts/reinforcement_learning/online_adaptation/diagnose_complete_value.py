"""Fit one preregistered reward-only critic diagnostic with episode-level holdout."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.value_diagnostics import (
    complete_episode_returns, fit_value_network, make_value_network, time_features, value_metrics,
)


def load_episode(root, trial, session_manifest, gamma):
    path = Path(trial['output'])
    summary = json.loads((path / 'summary.json').read_text())
    manifest = json.loads((path / 'trajectory/manifest.json').read_text())
    if (trial['status'] != 'complete' or trial['phase'] != 'evaluation'
            or manifest['status'] != 'complete' or manifest['vector_dtype'] != 'float32'
            or summary['mode'] != 'base_frozen' or summary['updates'] or summary['actor_updates']):
        raise ValueError('diagnosis requires completed frozen-base trials')
    if summary['critic_observation_schema'] != 'policy_plus_body_velocity_xy_elapsed_v1':
        raise ValueError('unregistered critic input schema')
    if (summary['session_budget']['session_id'] != session_manifest['session_id']
            or summary['session_budget']['trial_id'] != trial['id']
            or summary['task_objective_sha256'] != session_manifest['task_objective_sha256']):
        raise ValueError('session/task identity mismatch')
    chunks = []
    hashes = {str(path / 'summary.json'): sha256_file(path / 'summary.json'),
              str(path / 'trajectory/manifest.json'): sha256_file(path / 'trajectory/manifest.json')}
    for chunk in manifest['chunks']:
        f = path / 'trajectory' / chunk['file']
        hashes[str(f)] = sha256_file(f)
        with np.load(f, allow_pickle=False) as d:
            chunks.append({k: np.array(d[k]) for k in d.files})
    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    n = len(arrays['reward'])
    if n != summary['steps'] or n != trial['confirmed_steps'] or n != manifest['steps']:
        raise ValueError('trace/ledger step mismatch')
    np.testing.assert_array_equal(arrays['step'].reshape(-1), np.arange(1, n + 1))
    np.testing.assert_array_equal(arrays['bootstrap_mask'].reshape(-1), ~arrays['learning_terminated'].reshape(-1).astype(bool))
    np.testing.assert_allclose(arrays['reward'].reshape(-1), (arrays['ordinary_reward'] + arrays['terminal_reward_override']).reshape(-1), atol=2e-5, rtol=0)
    np.testing.assert_allclose(np.sum(arrays['reward']), summary['total_reward'], atol=1e-4, rtol=1e-7)
    for key in ('requested_action', 'mean_action', 'executed_action'):
        np.testing.assert_allclose(arrays[key], arrays['base_action'], atol=1e-6, rtol=0)
    x = arrays['critic_observation'].reshape(n, -1)
    if x.shape[1] != 48 or not np.isfinite(x).all():
        raise ValueError('expected finite 48-dimensional deployed critic features')
    np.testing.assert_allclose(x[:, -1], np.arange(n) / 2000., atol=1e-7, rtol=0)
    y = complete_episode_returns(arrays['reward'], arrays['learning_terminated'], arrays['learning_truncated'], arrays['collector_cutoff'], gamma=gamma)
    checkpoint = torch.load(path / 'latest.pt', map_location='cpu', weights_only=False)
    hashes[str(path / 'latest.pt')] = sha256_file(path / 'latest.pt')
    if sha256_file(checkpoint['base_checkpoint']) != session_manifest['base']['sha256']:
        raise ValueError('base checkpoint differs from the locked identity')
    if checkpoint['session']['updater']['actor_optimizer']['state'] or checkpoint['session']['updater']['policy_version'] != 0:
        raise ValueError('frozen collection altered actor state')
    if checkpoint['session']['updater']['critic_optimizer']['state']:
        raise ValueError('frozen collection altered critic optimizer state')
    if checkpoint['discount_gamma'] != gamma:
        raise ValueError('return discount differs from collection contract')
    rows = checkpoint['rows']
    if len(rows) != n:
        raise ValueError('row diagnostics do not match trace')
    contact = np.asarray([r['nonfoot_contact'] for r in rows])
    if contact.shape != (n,) or not np.isin(contact, [0, 1]).all():
        raise ValueError('nonfoot contact diagnostics must be binary scalars')
    progress = (arrays['root_state'].reshape(n, -1)[:, 0] - summary['initial_x']) / (summary['completion_target_x'] - summary['initial_x'])
    return {'x':x, 'y':y, 'contact':contact.astype(bool), 'progress':progress,
            'initial_critic':checkpoint['session']['updater']['critic'],
            'metadata':{'trial_id':trial['id'], 'reset_seed':trial['reset_seed'], 'steps':n,
                        'stop_reason':summary['stop_reason'], 'discounted_start_return':float(y[0]),
                        'undiscounted_return':summary['total_reward'], 'input_sha256':hashes}}


def evaluate_episodes(episodes, predictor):
    ys, ps, contact, progress = [], [], [], []
    per_episode = []
    for e in episodes:
        p = np.asarray(predictor(e['x'])).reshape(-1)
        per_episode.append({'trial_id':e['metadata']['trial_id'], **value_metrics(e['y'],p)})
        ys.append(e['y']);ps.append(p);contact.append(e['contact']);progress.append(e['progress'])
    y,p,c,t = [np.concatenate(v) for v in (ys,ps,contact,progress)]
    masks={'nonfoot_contact':c, 'no_nonfoot_contact':~c,
           'progress_first_third':t<1/3, 'progress_middle_third':(t>=1/3)&(t<2/3), 'progress_last_third':t>=2/3}
    return {'pooled':value_metrics(y,p), 'per_episode':per_episode,
            'episode_equal_mean_rmse':float(np.mean([e['rmse'] for e in per_episode])),
            'stage_metrics':{name:value_metrics(y[mask],p[mask]) for name,mask in masks.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    protocol=json.loads(Path(args.protocol).read_text())
    if protocol['schema'] != 'lite3_complete_value_diagnostic_v1':
        raise ValueError('unsupported diagnostic protocol')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    write_json_atomic(output/'protocol.json',protocol)
    torch.set_num_threads(1)
    report={'schema':protocol['schema'],'protocol_sha256':sha256_file(args.protocol),'terrains':{},
            'target_semantics':'complete task reward-only discounted MC, not SAC soft-Q',
            'selection':'fixed final epoch; heldout evaluated only after fitting', 'status':'running'}
    write_json_atomic(output/'report.json',report)
    for terrain in protocol['terrains']:
        root=Path(terrain['session_dir'])
        state=json.loads((root/'session_summary.json').read_text())
        manifest=json.loads((root/'ledger/manifest.json').read_text())
        if state['active'] is not None or state['pending'] is not None or state['uncertain_steps']:
            raise ValueError('session has incomplete or uncertain trials')
        if manifest['algorithm']!='lora' or manifest['terrain_seed']!=terrain['seed'] or manifest['base']['sha256']!=protocol['base_sha256']:
            raise ValueError('session identity differs from preregistration')
        if manifest['learner_seed']!=protocol['learner_seed'] or manifest['worker_configuration']['arguments']['task_deadline_steps']!=2000:
            raise ValueError('session seed/deadline differs from preregistration')
        expected=terrain['train_trials']+terrain['validation_trials']
        if (len(terrain['train_trials'])!=4 or len(terrain['validation_trials'])!=2
                or len(set(expected))!=6 or set(state['trials'])!=set(expected)):
            raise ValueError('episode split must be six distinct preregistered trials')
        loaded={trial:load_episode(root,state['trials'][trial],manifest,protocol['gamma']) for trial in expected}
        if len({e['metadata']['reset_seed'] for e in loaded.values()})!=6:
            raise ValueError('train/validation reset seed overlap')
        train=[loaded[t] for t in terrain['train_trials']]
        heldout=[loaded[t] for t in terrain['validation_trials']]
        x=np.concatenate([e['x'] for e in train]);y=np.concatenate([e['y'] for e in train])
        model=make_value_network(48)
        model.load_state_dict(train[0]['initial_critic'],strict=True)
        initial=copy.deepcopy(model)
        constant=float(y.mean())
        coefficients=np.linalg.lstsq(time_features(x,protocol['time_degree']),y,rcond=None)[0]
        started=time.perf_counter()
        fit=fit_value_network(model,x,y,**protocol['fit'])
        fit_seconds=time.perf_counter()-started
        def predict(network):
            def run(values):
                with torch.no_grad():return network(torch.as_tensor(values,dtype=torch.float32)).numpy().reshape(-1)
            return run
        predictors={'constant':lambda values:np.full(len(values),constant),
                    'time_only':lambda values:time_features(values,protocol['time_degree'])@coefficients,
                    'initial_critic':predict(initial),'learned_critic':predict(model)}
        result={'episodes':[dict(e['metadata'],role='train' if t in terrain['train_trials'] else 'validation') for t,e in loaded.items()],
                'source_sha256':manifest['worker_configuration']['source_sha256'],
                'fit_seconds':fit_seconds,'optimizer_steps':fit['optimizer_steps'],'training_curve':fit['training_curve'],
                'constant':constant,'time_coefficients':coefficients.tolist(),
                'metrics':{name:{'train':evaluate_episodes(train,f),'validation':evaluate_episodes(heldout,f)} for name,f in predictors.items()}}
        torch.save({'model':model.state_dict(),'optimizer':fit['optimizer'], 'protocol':protocol,
                    'diagnostic_only':True,'no_actor_or_online_session_checkpoint':True},output/f"critic-{terrain['seed']}.pt")
        report['terrains'][str(terrain['seed'])]=result
        write_json_atomic(output/'report.json',report)
        print(terrain['seed'],{name:m['validation']['pooled'] for name,m in result['metrics'].items()},flush=True)
    report['status']='complete'
    report['actual_control_transitions']=sum(e['steps'] for t in report['terrains'].values() for e in t['episodes'])
    write_json_atomic(output/'report.json',report)


if __name__=='__main__':
    main()
