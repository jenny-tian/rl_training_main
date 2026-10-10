"""Audit direct replay updates, source isolation, draw stream, and paired PPO prefix."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.algorithms import PPO

from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','protocol','run','training-audit','anchor-audit','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--heading-audit')
    parser.add_argument('--velocity-audit')
    args=parser.parse_args()
    torch.set_num_threads(4)
    plan=json.loads(Path(args.plan).read_text())
    p=json.loads(Path(args.protocol).read_text())
    root=Path(args.run)
    r=json.loads((root/'report.json').read_text())
    assert r['status']=='complete' and r['protocol_sha256']==sha256_file(args.protocol)
    evidence={str(f):sha256_file(f) for f in (Path(args.plan),Path(args.protocol),root/'report.json',Path(args.training_audit),Path(args.anchor_audit))}
    for path in (args.training_audit,args.anchor_audit):
        audit=json.loads(Path(path).read_text())
        assert audit['status']=='verified' and audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
    control=json.loads(Path(plan['control_protocol']).read_text())
    assert sha256_file(plan['control_protocol'])==plan['control_protocol_sha256']
    assert sha256_file(inspect.getfile(PPO))==plan['vendor_ppo_source_sha256']
    allowed={'registered_at_utc','arm','scope','selection'}
    demo_target=plan['schema']=='lite3_mean_anchor_demo_target_ablation_v1'
    tail_anchor=plan['schema']=='lite3_tail_anchor_ablation_v1'
    heading_weight=plan['schema']=='lite3_route_heading_weight_ablation_v1'
    heading=plan['schema']=='lite3_route_heading_ablation_v1' or heading_weight
    self_target=plan['schema']=='lite3_self_target_heading_ablation_v1'
    no_tail=plan['schema']=='lite3_no_tail_heading_ablation_v1'
    velocity=plan['schema']=='lite3_lateral_velocity_ablation_v1'
    self_mean=self_target or no_tail or velocity
    has_heading=heading or self_mean
    has_tail=(tail_anchor or has_heading) and not (no_tail or velocity)
    fixed_demo=(demo_target or has_tail) and not self_target
    if velocity:
        allowed|={'lateral_velocity','lateral_velocity_smoke'}
        assert set(plan['allowed_control_protocol_changes'])==allowed
        assert set(p)==set(control)|{'lateral_velocity','lateral_velocity_smoke'}
        assert p['lateral_velocity']==plan['lateral_velocity']
        assert 'tail_anchor' not in p and 'tail_anchor' not in r
        assert args.velocity_audit
        v=json.loads(Path(args.velocity_audit).read_text())
        assert v['status']=='verified' and v['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
        evidence[args.velocity_audit]=sha256_file(args.velocity_audit)
        proof=p['lateral_velocity_smoke']
        for key in ('report','audit'):
            digest_key='sha256' if key=='report' else 'audit_sha256'
            assert sha256_file(proof[key])==proof[digest_key]
            evidence[proof[key]]=proof[digest_key]
        assert sha256_file(plan['calibration'])==plan['calibration_sha256']
        calibration=json.loads(Path(plan['calibration']).read_text())
        assert {k:calibration[k] for k in p['lateral_velocity']}==p['lateral_velocity']
        evidence[plan['calibration']]=plan['calibration_sha256']
        tests=plan['tests']
        assert tests['failed']==0 and tests['passed']>=40 and sha256_file(tests['log'])==tests['sha256']
        evidence[tests['log']]=tests['sha256']
    if no_tail:
        allowed.add('tail_anchor')
        assert set(plan['allowed_control_protocol_changes'])==allowed
        assert set(p)==set(control)-{'tail_anchor'}
        assert control['tail_anchor']==plan['removed_tail_anchor']
        assert control['tail_anchor']['coefficient']==25.
        assert 'tail_anchor' not in r and 'tail_anchor' not in plan
        assert plan['tail_queries']==0 and plan['total_replay_queries']==plan['original_queries']
    if self_target:
        allowed.add('mean_anchor')
        assert set(plan['allowed_control_protocol_changes'])==allowed and set(p)==set(control)
        removed={'fixed_target_groups','fixed_target_array_sha256'}
        assert control['mean_anchor']['fixed_target_groups']==['correction']
        assert set(p['mean_anchor'])==set(control['mean_anchor'])-removed
        assert all(p['mean_anchor'][k]==v for k,v in control['mean_anchor'].items() if k not in removed)
    if heading:
        allowed|={'route_training','route_heading_smoke'}
        assert set(plan['allowed_control_protocol_changes'])==allowed
        assert set(p)==set(control)|{'route_heading_smoke'}
        assert p['route_training']==plan['route_training']
        if heading_weight:
            assert set(p)==set(control) and set(p['route_training'])==set(control['route_training'])
            assert p['route_training']['heading_weight']==4*control['route_training']['heading_weight']==-4.
            assert all(p['route_training'][k]==v for k,v in control['route_training'].items() if k!='heading_weight')
        else:
            assert set(p['route_training'])==set(control['route_training'])|{'heading_lookahead_m','heading_maximum_deg'}
            assert all(p['route_training'][k]==v for k,v in control['route_training'].items())
    if has_heading:
        assert args.heading_audit
        heading_audit=json.loads(Path(args.heading_audit).read_text())
        assert heading_audit['status']=='verified' and heading_audit['evidence'][str(root/'report.json')]==sha256_file(root/'report.json')
        evidence[args.heading_audit]=sha256_file(args.heading_audit)
        proof=p['route_heading_smoke']
        assert sha256_file(proof['report'])==proof['sha256']
        evidence[proof['report']]=proof['sha256']
    if tail_anchor:
        allowed.add('tail_anchor')
        assert set(plan['allowed_control_protocol_changes'])==allowed
        assert set(p)==set(control)|{'tail_anchor'}
        assert p['tail_anchor']==plan['tail_anchor']
    if demo_target:
        allowed.add('mean_anchor')
        assert set(plan['allowed_control_protocol_changes'])==allowed
        additions={'fixed_target_groups','fixed_target_array_sha256'}
        assert set(p['mean_anchor'])==set(control['mean_anchor'])|additions
        assert all(p['mean_anchor'][key]==value for key,value in control['mean_anchor'].items())
        assert p['mean_anchor']['fixed_target_groups']==['correction']
        assert set(p['mean_anchor']['fixed_target_array_sha256'])=={'correction'}
        assert set(p)==set(control)
    if plan['schema']=='lite3_mean_anchor_length_ablation_v1':
        allowed|={'iterations','training_control_transitions','save_interval'}
        assert set(plan['allowed_control_protocol_changes'])==allowed
        assert p['iterations']==2*control['iterations'] and p['training_control_transitions']==2*control['training_control_transitions']
        assert p['save_interval']==p['iterations']
    assert p['iterations']==plan['iterations'] and r['ledger']['confirmed']==plan['training_controls']
    for key,value in control.items():
        if key not in allowed:
            assert p[key]==value,key
    spec=p['mean_anchor'];observed=r['mean_anchor']
    if self_mean:
        assert not any(k.startswith('fixed_target') for k in spec)
        assert not any(k.startswith('fixed_target') for k in observed)
        assert plan['stored_action_target_rows_for_updates']==0
    assert spec==plan['mean_anchor'] and observed['validation_rows_used']==0
    assert observed['coefficient']==spec['coefficient'] and observed['sampling_device']=='cpu'
    assert observed['teacher_sha256']==p['base_sha256']==plan['base_sha256']
    assert observed['teacher_unchanged']
    assert sha256_file(spec['dataset'])==observed['dataset_sha256']==spec['dataset_sha256']
    assert sha256_file(spec['dataset_audit'])==spec['dataset_audit_sha256']
    evidence[spec['dataset']]=sha256_file(spec['dataset'])
    with np.load(spec['dataset'],allow_pickle=False) as dataset:
        counts={group:len(dataset[group+'_train_obs']) for group in spec['batch_groups']}
        assert counts==observed['group_rows']
        assert set(counts)=={'correction','flat','rough101','rough107'}
        calls=p['iterations']*p['ppo']['num_learning_epochs']*p['ppo']['num_mini_batches']
        assert observed['calls']==calls and observed['rows']==calls*sum(spec['batch_groups'].values())
        assert r['optimizer_steps']==[calls]
        generator=torch.Generator(device='cpu').manual_seed(spec['seed'])
        digest=hashlib.sha256()
        target_digest=hashlib.sha256()
        fixed_rows=0
        fixed_targets={group:dataset[group+'_train_target'] for group in spec.get('fixed_target_groups',[])}
        for group,targets in fixed_targets.items():
            assert targets.shape==(counts[group],12) and np.isfinite(targets).all()
            assert hashlib.sha256(targets.tobytes()).hexdigest()==spec['fixed_target_array_sha256'][group]
        if fixed_demo:
            assert counts['correction']==216
            assert observed['fixed_target_groups']==['correction']
            assert observed['fixed_target_array_sha256']==spec['fixed_target_array_sha256']
        for _ in range(calls):
            for group,n in spec['batch_groups'].items():
                indices=torch.randint(counts[group],(n,),generator=generator).numpy()
                digest.update(indices.tobytes())
                if group in fixed_targets:
                    target_digest.update(fixed_targets[group][indices].tobytes())
                    fixed_rows+=n
        assert digest.hexdigest()==observed['draw_sha256']
        if fixed_demo:
            assert fixed_rows==observed['fixed_target_rows']==calls*spec['batch_groups']['correction']
            assert target_digest.hexdigest()==observed['fixed_target_sha256']
        teacher=FrozenPPOExpert(torch.load(p['base_checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict'])
        assert sha256_file(root/'final.pt')==r['final_checkpoint']['sha256']
        candidate=FrozenPPOExpert(torch.load(root/'final.pt',map_location='cpu',weights_only=False)['actor_state_dict'])
        offline={}
        with torch.no_grad():
            for group in ('correction','flat','rough101','rough107','post_recovery'):
                values=dataset[group+'_validation_obs']
                deltas=torch.cat([candidate(torch.from_numpy(values[i:i+4096]))-teacher(torch.from_numpy(values[i:i+4096])) for i in range(0,len(values),4096)])
                offline[group]=dict(rows=len(values),mean_squared_raw_action_change=float(deltas.square().mean()),
                    p95_absolute_raw_action_change=float(torch.quantile(deltas.abs().flatten(),.95)),max_absolute_raw_action_change=float(deltas.abs().max()))
            if fixed_demo or self_mean:
                control_report=json.loads((Path(plan['control_training_run'])/'report.json').read_text())
                control_checkpoint=control_report['final_checkpoint']
                assert sha256_file(control_checkpoint['path'])==control_checkpoint['sha256']
                control_actor=FrozenPPOExpert(torch.load(control_checkpoint['path'],map_location='cpu',weights_only=False)['actor_state_dict'])
                evidence[control_checkpoint['path']]=control_checkpoint['sha256']
                demo_mse={}
                for split in ('train','validation'):
                    states=torch.from_numpy(dataset['correction_'+split+'_obs'])
                    targets=torch.from_numpy(dataset['correction_'+split+'_target'])
                    demo_mse[split]=dict(rows=len(states),**{name:float((actor(states)-targets).square().mean())
                        for name,actor in [('initial',teacher),('control',control_actor),('trial',candidate)]})
        if has_tail:
            tail=p['tail_anchor'];seen=r['tail_anchor']
            for key,value in tail.items():assert seen[key]==value
            assert tail['coefficient']==spec['coefficient']/4==25.
            assert tail['batch_groups']=={'post_recovery':128} and tail['seed']!=spec['seed']
            assert seen['validation_rows_used']==seen['stored_target_rows_used']==0
            assert seen['sampling_device']=='cpu' and seen['teacher_unchanged']
            assert seen['teacher_sha256']==p['base_sha256'] and seen['dataset_sha256']==spec['dataset_sha256']
            rows=dataset['post_recovery_train_obs']
            assert len(rows)==5681 and seen['group_rows']=={'post_recovery':len(rows)}
            assert hashlib.sha256(rows.tobytes()).hexdigest()==tail['observation_array_sha256']
            tail_generator=torch.Generator(device='cpu').manual_seed(tail['seed'])
            tail_digest=hashlib.sha256()
            for _ in range(calls):tail_digest.update(torch.randint(len(rows),(128,),generator=tail_generator).numpy().tobytes())
            assert tail_digest.hexdigest()==seen['draw_sha256']
            assert seen['calls']==calls and seen['rows']==calls*128==plan['tail_queries']
            assert observed['rows']==plan['original_queries']
            assert observed['rows']+seen['rows']==plan['total_replay_queries']
    updates=[json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()]
    assert [u['update'] for u in updates]==list(range(1,p['iterations']+1))
    losses=[u['losses']['mean_anchor'] for u in updates]
    assert np.isfinite(losses).all() and min(losses)>=0 and max(losses)>0
    control_root=Path(plan.get('control_training_run','artifacts/recovery_return_20261008/trial/train'))
    previous=control_root/'trajectory/chunk-00000.npz'
    first=root/'trajectory/chunk-00000.npz'
    prefix={}
    with np.load(previous,allow_pickle=False) as before,np.load(first,allow_pickle=False) as after:
        for key in ('policy_observation','critic_observation','pre_root_state','requested_action','emitted_action'):
            np.testing.assert_array_equal(before[key],after[key])
            prefix[key]=0.
        if self_mean and not velocity:
            assert set(before.files)==set(after.files)
            for key in before.files:np.testing.assert_array_equal(before[key],after[key])
        if velocity:
            assert set(after.files)==set(before.files)|{'velocity_root_state','velocity_origins','velocity_target','velocity_cost','velocity_reward'}
            for key in set(before.files)-{'reward'}:
                np.testing.assert_array_equal(before[key],after[key])
            delta=after['velocity_reward']
            residual=after['reward']-before['reward']-delta
            np.testing.assert_allclose(after['reward']-before['reward'],delta,atol=3e-6,rtol=1e-4)
            assert abs(delta).max()>0
            velocity_prefix=dict(common_fields_except_total_reward_bitwise_equal=True,
                maximum_total_reward_difference=float(abs(after['reward']-before['reward']).max()),
                maximum_velocity_only_residual=float(abs(residual).max()))
        if heading:
            if heading_weight:
                for key in ('heading_root_state','heading_origins','heading_target','heading_cost','heading_zero_target_cost'):
                    np.testing.assert_array_equal(before[key],after[key])
                delta=after['heading_cost']*(p['route_training']['heading_weight']-control['route_training']['heading_weight'])*.02
                np.testing.assert_allclose(after['heading_reward'],4*before['heading_reward'],atol=1e-7,rtol=1e-6)
            else:
                delta=(after['heading_cost']-after['heading_zero_target_cost'])*p['route_training']['heading_weight']*.02
            residual=after['reward']-before['reward']-delta
            np.testing.assert_allclose(after['reward']-before['reward'],delta,atol=3e-6,rtol=1e-4)
            assert abs(delta).max()>0
            reward_prefix=dict(maximum_absolute_difference=float(abs(after['reward']-before['reward']).max()),
                maximum_heading_only_residual=float(abs(residual).max()))
    for path in (first,previous,root/'updates.jsonl'):
        evidence[str(path)]=sha256_file(path)
    result=dict(status='verified',controls=r['ledger']['confirmed'],mean_anchor_calls=calls,
        replay_rows=observed['rows'],draw_sha256=digest.hexdigest(),validation_rows_used_for_updates=0,
        final_checkpoint=r['final_checkpoint'],first_rollout_max_absolute_difference=prefix,
        mean_anchor_loss=dict(first=losses[0],last=losses[-1],mean=float(np.mean(losses)),maximum=max(losses)),
        offline_action_drift_from_initial_actor=offline,evidence=evidence,
        scope=('Fixed TRAIN replay ablation; offline validation only used here for diagnostics, not optimizer updates.'
            if plan['schema']=='lite3_mean_anchor_length_ablation_v1' else
            'One fixed TRAIN replay auxiliary loss ablation; offline validation only used here for diagnostics, not optimizer updates.'))
    if fixed_demo:
        result.update(fixed_target_groups=['correction'],fixed_target_rows=fixed_rows,
            fixed_target_sha256=target_digest.hexdigest(),fixed_target_array_sha256=spec['fixed_target_array_sha256'],
            offline_correction_mse_to_demonstrated_emitted_action=demo_mse,
            scope='Only correction TRAIN replay labels changed; ordinary frozen means retained. Validation label MSE is post-training diagnostic only. First rollout paired, later updates intentionally differ.')
    if fixed_demo or self_mean:
        control_report=json.loads((control_root/'report.json').read_text())
        assert digest.hexdigest()==control_report['mean_anchor']['draw_sha256']
    if has_tail:
        tail_losses=[u['losses']['tail_anchor'] for u in updates]
        assert np.isfinite(tail_losses).all() and min(tail_losses)>=0 and max(tail_losses)>0
        result.update(tail_anchor=dict(calls=calls,replay_rows=seen['rows'],draw_sha256=tail_digest.hexdigest(),
            coefficient=tail['coefficient'],train_rows=len(rows),observation_array_sha256=tail['observation_array_sha256'],
            loss=dict(first=tail_losses[0],last=tail_losses[-1],mean=float(np.mean(tail_losses)),maximum=max(tail_losses))),
            total_replay_rows=observed['rows']+seen['rows'],
            scope='Only additive25*tail TRAIN preservation changed. Original four-group loss coefficients, data, normalization, draw stream and fixed demo targets unchanged. Offline validation is diagnostic only.')
        if not self_target:assert target_digest.hexdigest()==control_report['mean_anchor']['fixed_target_sha256']
    if has_tail or no_tail:
        tests=plan['tests']
        assert tests['failed']==0 and tests['passed']==(53 if has_heading else 42) and sha256_file(tests['log'])==tests['sha256']
        assert sha256_file(plan['diagnostic'])==plan['diagnostic_sha256']
        for path in (tests['log'],plan['diagnostic']):evidence[path]=sha256_file(path)
    if heading:
        assert tail_digest.hexdigest()==control_report['tail_anchor']['draw_sha256']
        result.update(first_rollout_reward_difference=reward_prefix,
            scope='Only centerline-directed heading reward changed; same physical first rollout across GPU0/GPU1. First reward difference explained by heading alone. Both replay streams and targets unchanged; offline validation is diagnostic only.')
        if heading_weight:
            result['scope']='Only heading weight-1 to-4 changed on the same GPU1. First physical rollout and heading geometry identical; reward difference explained by weight scaling. Both replay streams and targets unchanged; offline validation diagnostic only.'
    if self_target:
        assert fixed_rows==0 and not fixed_targets
        assert tail_digest.hexdigest()==control_report['tail_anchor']['draw_sha256']
        result.update(stored_action_target_rows_for_updates=0,
            offline_correction_mse_to_demonstrated_emitted_action=demo_mse,
            first_rollout_all_recorded_fields_bitwise_equal=True,first_rollout_reward_maximum_difference=0.,
            scope='Only correction replay target changes from stored demonstration actions to frozen initialization means. Both draw streams and first rollout including rewards identical. Stored action labels read only here for offline diagnostics, never for updates.')
    if no_tail:
        assert fixed_rows==0 and not fixed_targets
        assert observed['rows']==plan['original_queries']==plan['total_replay_queries']
        assert all('tail_anchor' not in u['losses'] for u in updates)
        assert r['source_files']==control_report['source_files']
        for path,digest_value in r['source_files'].items():assert sha256_file(path)==digest_value
        result.update(stored_action_target_rows_for_updates=0,tail_replay_rows=0,
            total_replay_rows=observed['rows'],removed_tail_coefficient=25.,
            offline_correction_mse_to_demonstrated_emitted_action=demo_mse,
            first_rollout_all_recorded_fields_bitwise_equal=True,first_rollout_reward_maximum_difference=0.,
            runtime_sources_equal_to_control=True,
            scope='Only the separate25*tail preservation term removed. Main replay configuration and draw stream, self-mean targets, heading reward and first full rollout unchanged. Stored labels used only in post-training diagnostics; no validation updates.')
    if velocity:
        assert fixed_rows==0 and not fixed_targets
        assert observed['rows']==plan['original_queries']==plan['total_replay_queries']
        assert all('tail_anchor' not in u['losses'] for u in updates)
        allowed_sources={'source/rl_training/rl_training/policy_route.py',
            'scripts/reinforcement_learning/train_lite3_policy.py','scripts/reinforcement_learning/smoke_lite3_recovery_anchor.py'}
        assert set(r['source_files'])==set(control_report['source_files'])
        changed={k for k,v in r['source_files'].items() if v!=control_report['source_files'][k]}
        assert changed==allowed_sources
        for path,digest_value in r['source_files'].items():assert sha256_file(path)==digest_value
        result.update(stored_action_target_rows_for_updates=0,tail_replay_rows=0,
            total_replay_rows=observed['rows'],first_rollout_velocity_difference=velocity_prefix,
            runtime_changed_files=sorted(changed),
            offline_correction_mse_to_demonstrated_emitted_action=demo_mse,
            scope='Only world lateral return velocity reward added. First common rollout fields except total reward bitwise equal; reward delta explained by new component. Same TRAIN self-mean replay/draw stream, no tail or stored action labels in updates. Offline validation diagnostics only.')
    if plan['schema']=='lite3_mean_anchor_length_ablation_v1':
        maxima={key:0. for key in prefix}
        files=sorted((control_root/'trajectory').glob('chunk-*.npz'))
        assert len(files)==control['iterations']
        for file in files:
            with np.load(file,allow_pickle=False) as before,np.load(root/'trajectory'/file.name,allow_pickle=False) as after:
                for key in maxima:maxima[key]=max(maxima[key],float(np.max(np.abs(before[key]-after[key]))))
        result['full_control_prefix']=dict(rollouts=len(files),maximum_absolute_difference=maxima,bitwise_equal=all(v==0 for v in maxima.values()))
    write_json_atomic(args.output,result)


if __name__=='__main__':
    main()
