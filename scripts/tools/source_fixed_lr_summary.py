"""Summarize the completed fixed-rate comparison from archived data only."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/rl_training'), str(ROOT/'scripts/tools')]
import numpy as np
from source_adaptation_runtime import write_json

BASE = ROOT/'docs/lite3-source-reward-live-verification-20261105'
ADAPTIVE = ROOT/'docs/lite3-source-two-update-trial-20261105'
TRIAL = ROOT/'docs/lite3-source-fixed-low-lr-trial-20261105'


def load(path):
    return json.loads(path.read_text())


def csv_rows(path, rows):
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw = TRIAL/'raw/runtime'
    out = TRIAL/'deliverables'
    result = load(raw/'result.json')
    original = load(BASE/'raw/runtime/result.json')['cases'][0]
    previous = load(ADAPTIVE/'raw/runtime/result.json')
    cases = [('original',original,BASE/'raw/runtime/old_mean.npz'),
        ('adaptive_update2',previous['cases'][0],ADAPTIVE/'raw/runtime/actor_update_2.npz')]
    cases += [(c['label'],c,raw/(c['label']+'.npz')) for c in result['cases']]
    if len(cases) != 4:
        raise ValueError('both fixed-rate checkpoints must be evaluated')
    baseline = np.load(cases[0][2])
    labels = {x['column']:x['subterrain'] for x in original['initial']['assignments']}
    conditions = []
    rewards = []
    metrics = []
    ref = original['after_100']
    for name,case,path in cases:
        if case['initial'] != original['initial'] or case['control_dt'] != original['control_dt']:
            raise ValueError('paired initial state, terrain, or control period changed')
        data = np.load(path)
        if not np.array_equal(data['columns'],baseline['columns']):
            raise ValueError('per-control terrain columns changed')
        for column,terrain in sorted(labels.items()):
            mask = data['columns'][100:] == column
            state = data['state'][100:][mask].astype(np.float64)
            reward = data['rewards'][100:][mask].astype(np.float64)
            conditions.append(dict(actor=name,column=column,terrain=terrain,transitions=int(mask.sum()),
                vx_mps=float(state[:,0].mean()),velocity_rms_error_mps=float(np.sqrt(state[:,3].mean())),
                reward_per_control=float(reward.sum(-1).mean()*case['control_dt'])))
        m = case['after_100']
        metrics.append(dict(actor=name,vx_mps=m['vx_mps'],velocity_rms_error_mps=m['velocity_rms_error_mps'],
            reward_per_control=m['reward_per_control'],clipped_joint_fraction=m['clipped_joint_fraction'],
            done_count=case['done_count'],speed_change_percent=100*(m['vx_mps']/ref['vx_mps']-1),
            error_change_percent=100*(m['velocity_rms_error_mps']/ref['velocity_rms_error_mps']-1),
            reward_change_percent=100*(m['reward_per_control']/ref['reward_per_control']-1)))
        rewards += [dict(actor=name,term=term,weighted_reward_per_second=value)
            for term,value in m['weighted_reward_per_second'].items()]
    csv_rows(out/'conditions.csv',conditions)
    csv_rows(out/'reward_components.csv',rewards)
    csv_rows(out/'actor_comparison.csv',metrics)
    minibatches = result['minibatches']
    csv_rows(out/'minibatches.csv',[{k:v for k,v in row.items() if k!='learning_rates'} |
        dict(learning_rate=row['learning_rates'][0]) for row in minibatches])
    csv_rows(out/'walking_probes.csv',result['walking_probes'])
    monitor_summary = []
    for update in (1,2):
        rows = [r for r in minibatches if r['update']==update]
        if len(rows) != 20 or any(r['learning_rates'] != [1e-5] for r in rows):
            raise ValueError('minibatch count or actual learning rate differs')
        monitor_summary.append(dict(update=update,minibatches=len(rows),
            max_minibatch_mean_kl_before=max(r['rollout_kl_before_mean'] for r in rows),
            max_minibatch_mean_kl_after=max(r['rollout_kl_after_mean'] for r in rows),
            mean_minibatch_kl_before=float(np.mean([r['rollout_kl_before_mean'] for r in rows])),
            actual_learning_rate=1e-5))
    before = load(raw/'budget_before.json')
    after = load(raw/'budget_after.json')
    preserved = before['entries'] == after['entries'][:-1] and before['ceilings'] == after['ceilings']
    new_cfg,old_cfg = result['runner_config'],previous['runner_config']
    cfg_changes = {k:[old_cfg['algorithm'][k],v] for k,v in new_cfg['algorithm'].items()
                   if old_cfg['algorithm'][k] != v}
    if cfg_changes != dict(learning_rate=[0.001,1e-5],schedule=['adaptive','fixed']):
        raise ValueError('unexpected PPO configuration difference')
    if {k:v for k,v in new_cfg.items() if k!='algorithm'} != {k:v for k,v in old_cfg.items() if k!='algorithm'}:
        raise ValueError('unexpected runner configuration difference')
    keys = ['mean_step_reward','post_step_vx_mps','sampled_action_clipped_fraction','episode_endings','rollout']
    paired_collection = {key:result['updates'][0][key] == previous['updates'][0][key] for key in keys}
    if not preserved or not all(paired_collection.values()):
        raise ValueError('history or first-rollout control comparison failed')
    condition_checks = []
    for actor in ('actor_update_1','actor_update_2'):
        matched = [(a,b) for a in conditions for b in conditions
                   if a['actor']=='original' and b['actor']==actor and a['column']==b['column']]
        condition_checks.append(dict(actor=actor,
            conditions_with_larger_error=sum(b['velocity_rms_error_mps']>a['velocity_rms_error_mps'] for a,b in matched),
            largest_error_increase_mps=max(b['velocity_rms_error_mps']-a['velocity_rms_error_mps'] for a,b in matched)))
    summary = dict(status=result['status'],execution=load(raw/'execution.json'),actors=metrics,
        evaluation_thresholds=load(raw/'manifest.json')['evaluation_thresholds'],
        evaluation_checks=result['evaluation_checks'],monitor_summary=monitor_summary,
        first_rollout_checks=paired_collection,configuration_changes=cfg_changes,
        history_preserved=preserved,condition_summary=condition_checks,
        updates=result['updates'],walking_probes=result['walking_probes'],checkpoint_sha256=result['checkpoint_sha256'],
        recommendation='both candidates pass the 8-second retention check; request approval for full first-episode comparison',
        limits=['same registered geometry and one training seed only',
            '40-second episodes and the 9m completion criterion have not been evaluated',
            'shared optimizer changes both actor and critic learning rates',
            'near-baseline error and reward are not evidence of improved terrain performance'])
    write_json(out/'summary.json',summary)
    print(json.dumps(dict(actors=metrics,monitor_summary=monitor_summary,
                         first_rollout_checks=paired_collection,condition_summary=condition_checks),indent=2))


if __name__ == '__main__':
    main()
