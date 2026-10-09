"""Audit the predeclared PPO reward pair, fixed pilots, and complete physical cost."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    evidence = {}
    def read(path):
        evidence[str(path)] = sha256_file(path)
        return json.loads(Path(path).read_text())
    plan = read('docs/lite3_recovery_return_plan_20261008.json')
    original = read('docs/lite3_prefix_imitation_plan_20261008.json')
    assert plan['pilot_acceptance'] == original['pilot_acceptance']
    before = read(plan['base_result'])
    assert sha256_file(plan['base_result']) == plan['base_result_sha256']
    smoke = read('docs/lite3_recovery_return_smoke_audit_20261008.json')
    assert smoke['status'] == 'verified' and smoke['controls'] <= plan['max_smoke_controls']
    protocols, reports, outcomes, models = {}, {}, {}, {}
    costs = dict(smoke=smoke['controls'])
    initial_run = read('artifacts/retention_coverage_20261008/learner2247/pilot-evaluation/report.json')
    failures = {}
    for arm in ('control', 'trial'):
        root = Path('artifacts/recovery_return_20261008')/arm
        protocols[arm] = read(f'docs/lite3_recovery_return_{arm}_train_protocol_20261008.json')
        reports[arm] = read(root/'train/report.json')
        anchor = read(f'docs/lite3_recovery_return_{arm}_anchor_audit_20261008.json')
        outcome = read(f'docs/lite3_recovery_return_{arm}_pilot_result_20261008.json')
        evaluation = read(root/'pilot-evaluation/report.json')
        for result in (anchor, outcome):
            assert result['status'] == 'verified'
            for path, digest in result['evidence'].items():
                assert sha256_file(path) == digest
        t = protocols[arm]
        assert t['base_sha256'] == plan['base_sha256']
        assert t['recovery_resets'] == plan['recovery_resets'] and t['ppo'] == plan['ppo']
        assert t['recovery_course_lateral_weight'] == plan['arms'][arm]['recovery_course_lateral_weight']
        assert t['seed'] == plan['training_seed'] and t['iterations'] == plan['iterations']
        assert t['initial_std'] == plan['initial_std'] and t['base_actor_mean_only']
        assert reports[arm]['ppo_updates'] == plan['iterations'] and anchor['controls'] == plan['controls_per_arm']
        for condition in ('course', 'recovery'):
            assert evaluation['phases'][condition+'-trial']['initial_sha256'] == initial_run['phases'][condition+'-trial']['initial_sha256']
        outcomes[arm] = dict(metrics=outcome['results']['trial'], gates=outcome['gates'], pilot_passed=outcome['pilot_passed'])
        models[arm] = anchor['final_checkpoint']
        costs[arm+'_training'] = anchor['controls']
        costs[arm+'_pilot'] = outcome['controls']
        failures[arm] = {phase: [e for e in value['episodes'] if not e['success']] for phase, value in evaluation['phases'].items()}
    for key in protocols['control']:
        if key not in ('arm', 'recovery_course_lateral_weight'):
            assert protocols['control'][key] == protocols['trial'][key], key
    assert reports['control']['source_sha256'] == reports['trial']['source_sha256']
    prefix = {}
    paths = [Path('artifacts/recovery_return_20261008')/arm/'train/trajectory' for arm in ('control','trial')]
    chunks = [next(iter(sorted(path.glob('chunk-*.npz')))) for path in paths]
    with np.load(chunks[0], allow_pickle=False) as a, np.load(chunks[1], allow_pickle=False) as b:
        for key in ('policy_observation', 'critic_observation', 'pre_root_state', 'requested_action', 'emitted_action'):
            np.testing.assert_array_equal(a[key][0], b[key][0])
            prefix[key] = float(np.max(np.abs(a[key]-b[key])))
    for chunk in chunks:
        evidence[str(chunk)] = sha256_file(chunk)
    dataset_path=Path('artifacts/retention_coverage_20261008/dataset/dataset.npz')
    dataset_report=read(dataset_path.parent/'report.json')
    assert sha256_file(dataset_path)==dataset_report['dataset_sha256']
    evidence[str(dataset_path)]=sha256_file(dataset_path)
    torch.set_num_threads(4)
    payload=torch.load(plan['base_checkpoint'],map_location='cpu',weights_only=False)
    teacher=FrozenPPOExpert(payload['actor_state_dict'])
    offline={}
    with np.load(dataset_path,allow_pickle=False) as data, torch.no_grad():
        predictions={group:torch.cat([teacher(torch.from_numpy(data[group+'_validation_obs'][i:i+4096])) for i in range(0,len(data[group+'_validation_obs']),4096)]) for group in ('flat','rough101','rough107','correction','post_recovery')}
        for arm in ('control','trial'):
            checkpoint=torch.load(models[arm]['path'],map_location='cpu',weights_only=False)
            model=FrozenPPOExpert(checkpoint['actor_state_dict'])
            offline[arm]=dict(groups={},final_std=checkpoint['actor_state_dict']['distribution.log_std_param'].exp().tolist())
            for group,reference in predictions.items():
                x=data[group+'_validation_obs']
                predicted=torch.cat([model(torch.from_numpy(x[i:i+4096])) for i in range(0,len(x),4096)])
                delta=predicted-reference
                offline[arm]['groups'][group]=dict(rows=len(x),mean_squared_raw_action_change=float(delta.square().mean()),
                    p95_absolute_raw_action_change=float(torch.quantile(delta.abs().flatten(),.95)),max_absolute_raw_action_change=float(delta.abs().max()))
    total = sum(costs.values())
    assert total <= plan['max_smoke_controls']+plan['max_training_controls']+plan['max_pilot_controls']
    costs.update(stage_through_pilots=total, new_uncertain=0,
        cumulative_charged=plan['cumulative_before']['charged']+total,
        cumulative_confirmed=plan['cumulative_before']['confirmed']+total,
        historical_uncertain=plan['cumulative_before']['uncertain'])
    conditional = outcomes[plan['primary_arm']]['pilot_passed']
    result = dict(status='verified', primary_arm=plan['primary_arm'], initial_actor=before['results']['trial'],
        arms=outcomes, checkpoints=models, failures=failures, costs=costs,offline_action_drift_from_initial_actor=offline,
        first_rollout_max_absolute_difference=prefix, conditional_validation_triggered=conditional,
        evidence=evidence, scope='One PPO learning seed, one TRAIN recovery source, developer pilot2251. Original7-source qualification and independent learning replication remain separate.')
    write_json_atomic('docs/lite3_recovery_return_result_20261008.json', result)
    values = [before['results']['trial'], outcomes['control']['metrics'], outcomes['trial']['metrics']]
    oldbase = before['results']['base']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    labels = ['Supervised init', 'PPO lateral -4', 'PPO lateral -8']
    specs = [('course_total',1,'Ordinary course completion (of 64)',None),
             ('case26_recovery',1,'TRAIN case26 recovery + completion (of 24)',20),
             ('course_peak_m',100,'Ordinary rough peak lateral error (cm)',oldbase['course_peak_m']*100),
             ('case26_peak_m',100,'Recovery peak lateral error (cm)',70)]
    for ax, (key, scale, title, gate) in zip(axes.flat, specs):
        heights = [v[key]*scale for v in values]
        ax.bar(labels, heights, color=['#65758B','#2A8C82','#B27C48'], width=.6)
        for i, height in enumerate(heights):
            ax.text(i,height,f'{height:.2f}' if scale==100 else str(int(height)),ha='center',va='bottom')
        if gate is not None:
            ax.axhline(gate,color='#B44747',ls='--',label=f'Gate {gate:.2f}')
            ax.legend(frameon=False)
        ax.set_ylim(0,max(heights+[gate or 0])*1.25)
        ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"PPO return-to-route pair: primary pilot {'PASSED' if conditional else 'FAILED'}\nFrozen final checkpoints; same evaluation2251; no teacher or prefix at inference",fontsize=14)
    target=Path('docs/figures/lite3-recovery-return-20261008')
    for suffix in ('.png','.svg'):
        fig.savefig(target.with_suffix(suffix),dpi=160)
    svg=target.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':
    main()
