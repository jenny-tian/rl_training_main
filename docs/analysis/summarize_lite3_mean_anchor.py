"""Summarize the direct mean replay ablation against its fixed paid PPO control."""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=dict(plan='docs/lite3_mean_anchor_plan_20261009.json',update_audit='docs/lite3_mean_anchor_update_audit_20261009.json',
        pilot_result='docs/lite3_mean_anchor_pilot_result_20261009.json',run='artifacts/mean_anchor_20261009/pilot-evaluation',
        control_run='artifacts/recovery_return_20261008/trial/pilot-evaluation',output='docs/lite3_mean_anchor_result_20261009.json',
        figure='docs/figures/lite3-mean-anchor-20261009')
    for key,value in defaults.items():parser.add_argument('--'+key.replace('_','-'),default=value)
    parser.add_argument('--control-update-audit')
    parser.add_argument('--retry-addendum')
    parser.add_argument('--engineering-audit')
    args=parser.parse_args()
    evidence={}
    def read(path):
        evidence[str(path)]=sha256_file(path)
        return json.loads(Path(path).read_text())
    plan=read(args.plan)
    control=read(plan['control_result'])
    assert sha256_file(plan['control_result'])==plan['control_result_sha256']
    initial=read('docs/lite3_retention_coverage_pilot_result_20261008.json')
    trained=read(args.update_audit)
    result=read(args.pilot_result)
    for r in (control,initial,trained,result):
        assert r['status']=='verified'
        for path,digest in r['evidence'].items():assert sha256_file(path)==digest
    oldrun=read(Path(args.control_run)/'report.json')
    newrun=read(Path(args.run)/'report.json')
    for condition in ('course','recovery'):
        assert oldrun['phases'][condition+'-trial']['initial_sha256']==newrun['phases'][condition+'-trial']['initial_sha256']
    olddiagnostic=(read(args.control_update_audit)['offline_action_drift_from_initial_actor'] if args.control_update_audit else
        read('docs/lite3_recovery_return_result_20261008.json')['offline_action_drift_from_initial_actor']['trial']['groups'])
    ratios={key:value['mean_squared_raw_action_change']/olddiagnostic[key]['mean_squared_raw_action_change']
            for key,value in trained['offline_action_drift_from_initial_actor'].items()}
    assert trained['controls']==plan['training_controls'] and result['controls']<=plan['max_pilot_controls']
    aborted_controls=0
    if args.retry_addendum:
        retry=read(args.retry_addendum)
        assert retry['original_plan']==args.plan and retry['original_plan_sha256']==sha256_file(args.plan)
        interrupted=read(retry['interruption_audit'])
        assert sha256_file(retry['interruption_audit'])==retry['interruption_audit_sha256']
        assert interrupted['status']=='verified_interrupted_run' and interrupted['uncertain']==0
        for path,digest in interrupted['evidence'].items():assert sha256_file(path)==digest
        aborted_controls=interrupted['controls']
        assert aborted_controls==retry['aborted_controls']
        assert retry['original_stage_max_controls']==plan['max_new_controls']
        assert retry['revised_stage_max_controls']==plan['max_new_controls']+aborted_controls
        assert Path(trained['final_checkpoint']['path']).parent==Path(retry['retry_training_run']).resolve()
        assert retry['cumulative_before_retry']['charged']==plan['cumulative_before']['charged']+aborted_controls
        assert retry['cumulative_before_retry']['confirmed']==plan['cumulative_before']['confirmed']+aborted_controls
        tests=retry['tests']
        assert tests['failed']==0 and sha256_file(tests['log'])==tests['sha256']
    demo_target=plan['schema']=='lite3_mean_anchor_demo_target_ablation_v1'
    tail_anchor=plan['schema']=='lite3_tail_anchor_ablation_v1'
    heading_weight=plan['schema']=='lite3_route_heading_weight_ablation_v1'
    heading=plan['schema']=='lite3_route_heading_ablation_v1' or heading_weight
    self_target=plan['schema']=='lite3_self_target_heading_ablation_v1'
    no_tail=plan['schema']=='lite3_no_tail_heading_ablation_v1'
    velocity=plan['schema']=='lite3_lateral_velocity_ablation_v1'
    correction_weight=plan['schema']=='lite3_correction_group_weight_ablation_v1'
    if demo_target:assert args.retry_addendum
    engineering=0
    if heading or velocity:
        assert args.engineering_audit
        engineering_audit=read(args.engineering_audit)
        assert engineering_audit['status']=='verified'
        for path,digest in engineering_audit['evidence'].items():assert sha256_file(path)==digest
        engineering=engineering_audit['controls']
        assert engineering==plan['max_engineering_controls']
        engineering_run=plan.get('engineering_run','artifacts/lateral_velocity_20261010/smoke')
        assert engineering_audit['evidence'][str(Path(engineering_run)/'report.json')]==sha256_file(Path(engineering_run)/'report.json')
    total=trained['controls']+result['controls']+aborted_controls+engineering
    costs=dict(training=trained['controls'],pilot=result['controls'],replay_new_physics=0,new_uncertain=0,
        stage_through_pilot=total,cumulative_charged_through_pilot=plan['cumulative_before']['charged']+total,
        cumulative_confirmed_through_pilot=plan['cumulative_before']['confirmed']+total,historical_uncertain=64)
    if args.retry_addendum:costs['interrupted_training']=aborted_controls
    if heading or velocity:costs['engineering']=engineering
    if self_target or no_tail or correction_weight:
        assert plan['max_engineering_controls']==0
        costs['engineering']=0
    out=dict(status='verified',initial=initial['results']['trial'],control=control['results']['trial'],trial=result['results']['trial'],
        gates=result['gates'],pilot_passed=result['pilot_passed'],costs=costs,final_checkpoint=trained['final_checkpoint'],
        failures={key:[e for e in value['episodes'] if not e['success']] for key,value in newrun['phases'].items()},
        offline_mean_action_mse_ratio_vs_control=ratios,mean_anchor_loss=trained['mean_anchor_loss'],evidence=evidence,
        scope=('Fixed TRAIN replay experiment, learner2281 and developer eval2251. Repeated TRAIN case26 slots, not original7-source qualification or independent learning replication.'
            if plan['schema']=='lite3_mean_anchor_length_ablation_v1' else
            'One fixed mean-loss ablation, same learner2281 and developer eval2251. Repeated TRAIN case26 slots, not original7-source qualification or independent learning replication.'))
    if demo_target:
        out['offline_correction_mse_to_demonstrated_emitted_action']=trained['offline_correction_mse_to_demonstrated_emitted_action']
        out['scope']='Correction TRAIN demonstration-target ablation. Same learner2281/eval2251, fixed final256 updates. Repeated TRAIN case26 slots, not original7-source qualification or independent learning replication. Interrupted launch charged separately.'
    if tail_anchor or heading or self_target:
        out['offline_correction_mse_to_demonstrated_emitted_action']=trained['offline_correction_mse_to_demonstrated_emitted_action']
        out['tail_anchor']=trained['tail_anchor']
        out['total_replay_rows']=trained['total_replay_rows']
        out['scope']='Additive post-recovery TRAIN mean preservation, same learner2281/eval2251 and final256 updates. Repeated TRAIN source only; original qualification and independent learning replication remain separate.'
    if heading:
        out['first_rollout_reward_difference']=trained['first_rollout_reward_difference']
        out['scope']='Centerline-directed training heading reward. Same learner2281/eval2251 and final256. Physical first rollout paired across identical-model GPUs; same replay losses. TRAIN-source development only, no inference controller.'
        if heading_weight:
            out['scope']='Only heading reward weight-1 to-4 changed. Same GPU1, learner2281/eval2251 and final256, same heading geometry and replay losses. TRAIN-source development only, no inference controller.'
    if self_target:
        assert trained['stored_action_target_rows_for_updates']==0 and trained['first_rollout_all_recorded_fields_bitwise_equal']
        out['stored_action_target_rows_for_updates']=0
        out['scope']='Correction targets only: frozen initialization means replace stored demo labels. Same heading weight-1, tail25, learner2281/eval2251/final256; first rollout including rewards identical. TRAIN-source developer pilot only.'
    if no_tail:
        assert trained['stored_action_target_rows_for_updates']==trained['tail_replay_rows']==0
        assert trained['first_rollout_all_recorded_fields_bitwise_equal']
        out.update(stored_action_target_rows_for_updates=0,tail_replay_rows=0,
            total_replay_rows=trained['total_replay_rows'],removed_tail_coefficient=25.,
            offline_correction_mse_to_demonstrated_emitted_action=trained['offline_correction_mse_to_demonstrated_emitted_action'],
            scope='Only separate tail preservation removed; same four-group self-mean targets, heading weight-1, learner2281/eval2251/final256. Main draws and first full rollout identical. Repeated TRAIN-source pilot only.')
    if velocity:
        assert trained['stored_action_target_rows_for_updates']==trained['tail_replay_rows']==0
        out.update(stored_action_target_rows_for_updates=0,tail_replay_rows=0,
            total_replay_rows=trained['total_replay_rows'],lateral_velocity=plan['lateral_velocity'],
            first_rollout_velocity_difference=trained['first_rollout_velocity_difference'],
            scope='Only world lateral return velocity reward added, same no-tail/self-mean/heading configuration, learner2281/eval2251/final256. Repeated TRAIN-source pilot only; no inference controller.')
    if correction_weight:
        assert trained['stored_action_target_rows_for_updates']==trained['tail_replay_rows']==0
        assert trained['first_rollout_all_recorded_fields_bitwise_equal']
        out.update(stored_action_target_rows_for_updates=0,tail_replay_rows=0,
            total_replay_rows=trained['total_replay_rows'],group_weights=trained['group_weights'],gradient_records=trained['gradient_records'],
            evaluation_execution=newrun.get('execution'),
            scope='Only correction TRAIN mean preservation halved; ordinary weights, denominator, draws, rewards and final256/learner2281 unchanged. Read-only first-minibatch gradients. Per-condition process evaluation; repeated TRAIN case26 only.')
    write_json_atomic(args.output,out)
    metrics=[out['initial'],out['control'],out['trial']]
    length=plan['schema']=='lite3_mean_anchor_length_ablation_v1'
    labels=['Supervised init','PPO 128 updates','PPO 256 updates'] if length else ['Supervised init','PPO reward anchor','+ Direct mean loss']
    if demo_target:labels=['Supervised init','PPO self-mean targets','PPO demo correction targets']
    if tail_anchor:labels=['Supervised init','PPO demo targets','+ Tail preservation']
    if heading:labels=['Supervised init','Zero heading target','Centerline heading target']
    if heading_weight:labels=['Supervised init','Heading weight -1','Heading weight -4']
    if self_target:labels=['Supervised init','Demo correction\ntargets','Self-mean correction\ntargets']
    if no_tail:labels=['Supervised init','Self-mean targets\nwith tail25','Self-mean targets\nwithout tail']
    if velocity:labels=['Supervised init','No-tail heading\ncontrol','+ World lateral\nvelocity reward']
    if correction_weight:labels=['Supervised init','Correction group\nweight 1.0','Correction group\nweight 0.5']
    base=result['results']['base']
    plots=[('course_total',1,'Ordinary completion (of 64)',None),
        ('course_peak_m',100,'Ordinary rough peak lateral error (cm)',base['course_peak_m']*100),
        ('case26_recovery',1,'TRAIN case26 recovery + completion (of 24)',20),
        ('case26_peak_m',100,'Recovery peak lateral error (cm)',70)]
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,(key,scale,title,gate) in zip(axes.flat,plots):
        heights=[m[key]*scale for m in metrics]
        ax.bar(labels,heights,color=['#65758B','#B27C48','#2A8C82'],width=.6)
        for i,value in enumerate(heights):ax.text(i,value,f'{value:.2f}' if scale==100 else str(int(value)),ha='center',va='bottom')
        if gate is not None:
            ax.axhline(gate,color='#B44747',ls='--',label=f'Gate {gate:.2f}')
            ax.legend(frameon=False)
        ax.set_ylim(0,max(heights+[gate or 0])*1.25)
        ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
    title='PPO training length' if length else 'Direct mean-action loss'
    if demo_target:title='Correction replay demonstration targets'
    if tail_anchor:title='Additive post-recovery mean preservation'
    if heading:title='Centerline-directed heading reward'
    if heading_weight:title='Centerline heading reward strength'
    if self_target:title='Correction targets with fixed centerline heading reward'
    if no_tail:title='Tail preservation with fixed self-mean targets and heading reward'
    if velocity:title='World lateral return velocity reward'
    if correction_weight:title='Correction-only mean preservation strength'
    fig.suptitle(f"{title}: pilot {'PASSED' if out['pilot_passed'] else 'FAILED'}\nFixed final{plan['iterations']} PPO updates, same evaluation2251, no teacher/prefix at inference",fontsize=14)
    target=Path(args.figure)
    for suffix in ('.png','.svg'):fig.savefig(target.with_suffix(suffix),dpi=160)
    svg=target.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
