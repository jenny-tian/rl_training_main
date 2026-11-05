"""Summarize the completed two-update trial using saved data only; no simulation."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/rl_training'), str(ROOT/'scripts/tools')]
import numpy as np
import torch
from source_adaptation_runtime import OLD, HASHES, digest, write_json
from rl_training.source_scan.frozen_actor import FrozenSourceActor
from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES

BASELINE = ROOT/'docs/lite3-source-reward-live-verification-20261105'
TRIAL = ROOT/'docs/lite3-source-two-update-trial-20261105'


def csv_rows(path, rows):
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    raw = TRIAL/'raw/runtime'
    out = TRIAL/'deliverables'
    baseline = json.loads((BASELINE/'raw/runtime/result.json').read_text())['cases'][0]
    result = json.loads((raw/'result.json').read_text())
    candidate = result['cases'][0]
    if baseline['control_dt'] != candidate['control_dt']:
        raise ValueError('control periods differ')
    execution = json.loads((raw/'execution.json').read_text())
    manifest = json.loads((raw/'manifest.json').read_text())
    if baseline['initial'] != candidate['initial']:
        raise ValueError('paired geometry and states differ')
    old = np.load(BASELINE/'raw/runtime/old_mean.npz')
    new = np.load(raw/'actor_update_2.npz')
    if not np.array_equal(old['columns'],new['columns']):
        raise ValueError('condition assignments changed during the comparison')
    columns = {row['column']:row['subterrain'] for row in baseline['initial']['assignments']}
    conditions = []
    for col, terrain in sorted(columns.items()):
        row = dict(column=col,terrain=terrain)
        for label, data in [('original',old),('update2',new)]:
            mask = data['columns'][100:] == col
            state = data['state'][100:][mask].astype(np.float64)
            row[label+'_vx_mps'] = float(state[:,0].mean())
            row[label+'_velocity_rms_error_mps'] = float(np.sqrt(state[:,3].mean()))
            row[label+'_reward_per_control'] = float(data['rewards'][100:][mask].astype(np.float64).sum(-1).mean()*baseline['control_dt'])
        row['vx_change_mps'] = row['update2_vx_mps']-row['original_vx_mps']
        row['error_change_mps'] = row['update2_velocity_rms_error_mps']-row['original_velocity_rms_error_mps']
        conditions.append(row)
    csv_rows(out/'conditions.csv',conditions)
    terms = []
    for name, value in baseline['after_100']['weighted_reward_per_second'].items():
        other = candidate['after_100']['weighted_reward_per_second'][name]
        terms.append(dict(term=name,original_per_second=value,update2_per_second=other,change_per_second=other-value))
    csv_rows(out/'reward_components.csv',terms)
    if digest(OLD) != HASHES[0]:
        raise ValueError('original checkpoint changed')
    states = {'original':torch.load(OLD,map_location='cpu',weights_only=True)['actor_state_dict']}
    for n in (1,2):
        path = out/f'actor_update_{n}.pt'
        if digest(path) != result['checkpoint_sha256'][path.name]:
            raise ValueError('candidate checkpoint changed')
        states[f'update{n}'] = torch.load(path,map_location='cpu',weights_only=True)
    actors = {k:FrozenSourceActor(v) for k,v in states.items()}
    stds = {k:v['distribution.log_std_param'].exp().double() for k,v in states.items()}
    probes = []
    for label,path in [('original_trajectory',BASELINE/'raw/runtime/old_mean_observations.pt'),
                       ('update2_trajectory',raw/'actor_update_2_observations.pt')]:
        snapshots = torch.load(path,map_location='cpu',weights_only=True)
        for step,snapshot in sorted(snapshots.items()):
            outputs = {k:actor(snapshot['observation']).double() for k,actor in actors.items()}
            for start,end in [('original','update1'),('original','update2'),('update1','update2')]:
                delta = outputs[end]-outputs[start]
                s1,s2 = stds[start],stds[end]
                kl = (torch.log(s2/s1)+(s1.square()+delta.square())/(2*s2.square())-.5).sum(-1)
                probes.append(dict(trajectory=label,control_step=step,observations=len(delta),
                    comparison=start+'_to_'+end,action_shift_rms=float(delta.square().mean().sqrt()),
                    action_shift_max=float(delta.abs().max()),mean_gaussian_kl=float(kl.mean()),
                    joint_rms=dict(zip(LITE3_POLICY_JOINT_NAMES,delta.square().mean(0).sqrt().tolist()))))
    offline = dict(mode='cpu_read_only',simulation_transitions=0,training_transitions=0,ppo_updates=0,
        probe_note='Same saved observations for all actors. Gaussian KL is offline, not the PPO rollout KL.',
        cuda_initialized=torch.cuda.is_initialized(),actor_action_std={k:v.tolist() for k,v in stds.items()},
        probes=probes)
    write_json(out/'offline_action_comparison.json',offline)
    before = json.loads((raw/'budget_before.json').read_text())
    after = json.loads((raw/'budget_after.json').read_text())
    budget_checks = dict(history_preserved=after['entries'][:-1]==before['entries'],
        old_ceilings_preserved=before['ceilings']==after['ceilings'],
        cumulative_charge_exact=sum(r['reserved_transitions'] for r in after['entries'] if r['run']==1)==265344)
    if not all(budget_checks.values()):
        raise ValueError('budget history changed')
    m0,m2 = baseline['after_100'],candidate['after_100']
    summary = dict(status=result['status'],execution=execution,baseline=m0,candidate=m2,
        baseline_done_count=baseline['done_count'],candidate_done_count=candidate['done_count'],
        evaluation_thresholds=manifest['evaluation_thresholds'],evaluation_checks=result['evaluation_checks'],
        speed_change_percent=100*(m2['vx_mps']/m0['vx_mps']-1),
        error_change_percent=100*(m2['velocity_rms_error_mps']/m0['velocity_rms_error_mps']-1),
        reward_change_percent=100*(m2['reward_per_control']/m0['reward_per_control']-1),
        condition_count=15,conditions_with_lower_speed=sum(r['vx_change_mps']<0 for r in conditions),
        conditions_with_larger_error=sum(r['error_change_mps']>0 for r in conditions),
        updates=result['updates'],checkpoint_sha256=result['checkpoint_sha256'],budget_checks=budget_checks,
        recommendation='retain original checkpoint; do not extend or promote this candidate',
        limitations=['400 controls represent 8 seconds, not full terrain qualification',
            'update1 has only offline probes, no deterministic runtime evaluation',
            'no ablation isolates critic initialization from learning-rate effects'])
    write_json(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','speed_change_percent','error_change_percent',
        'reward_change_percent','conditions_with_lower_speed','conditions_with_larger_error']},indent=2))
    print('Offline analysis completed without CUDA, simulator steps, or training updates.')


if __name__ == '__main__':
    main()
