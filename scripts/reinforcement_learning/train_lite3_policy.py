"""Fresh actor-initialized PPO training with immutable control-transition budget."""
import argparse
import copy
import importlib.metadata
import json
import math
import hashlib
import signal
from pathlib import Path
import time

from isaaclab.app import AppLauncher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic
    from rl_training.online_adaptation.interaction_budget import digest
    from rl_training.vector_step_ledger import VectorStepLedger,ChargedVecEnv
    p=json.loads(Path(args.protocol).read_text());assert p['schema']=='lite3_full_policy_pilot_v1'
    expected=p['num_envs']*p['rollout_steps']*p['iterations'];assert expected==p['training_control_transitions']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    paths=sorted((root/'source/rl_training/rl_training').rglob('*.py'))+sorted((root/'scripts/reinforcement_learning').rglob('*.py'))
    sources={str(x.relative_to(root)):sha256_file(x) for x in paths}
    protocol_hash=sha256_file(args.protocol)
    ledger=VectorStepLedger(out/'ledger.json',num_envs=p['num_envs'],max_transitions=expected,
                            metadata={'protocol_sha256':protocol_hash,'source_sha256':digest(sources)})
    report={'status':'running','protocol_sha256':protocol_hash,'source_sha256':digest(sources),'source_files':sources,
            'base_sha256':p['base_sha256'],'ppo_updates':0,'episode_counts':{'success':0,'failure':0,'deadline':0}}
    write_json_atomic(out/'report.json',report)
    app=None;env=None;trace=None;started=time.perf_counter()
    try:
        assert sha256_file(p['base_checkpoint'])==p['base_sha256']
        assert sha256_file(p['smoke_report'])==p['smoke_sha256']
        assert json.loads(Path(p['smoke_report']).read_text())['status']=='complete'
        if p.get('route_observations'):
            assert sha256_file(p['frozen_smoke_report'])==p['frozen_smoke_sha256']
            assert json.loads(Path(p['frozen_smoke_report']).read_text())['status']=='complete'
        heading_options=p.get('route_training',{})
        heading_trace='heading_lookahead_m' in heading_options
        velocity_options=p.get('lateral_velocity')
        history_options=p.get('route_history')
        if history_options is not None:
            from rl_training.policy_route_history import validate_history_options, RouteHistoryVecEnv, expand_actor_history_inputs
            validate_history_options(history_options)
            proof=p['route_history_smoke']
            assert sha256_file(proof['report'])==proof['sha256']
            checked=json.loads(Path(proof['report']).read_text())
            assert checked['status']=='complete' and checked['route_history_options']==history_options
            assert sha256_file(proof['audit'])==proof['audit_sha256']
            checked_audit=json.loads(Path(proof['audit']).read_text())
            assert checked_audit['status']=='verified' and checked_audit['evidence'][proof['report']]==proof['sha256']
        if velocity_options:
            proof=p['lateral_velocity_smoke']
            assert sha256_file(proof['report'])==proof['sha256']
            checked=json.loads(Path(proof['report']).read_text())
            assert checked['status']=='complete' and checked['lateral_velocity_options']==velocity_options
            assert checked['base_sha256']==p['base_sha256'] and checked['lateral_velocity_timing_passed']
            assert sha256_file(proof['audit'])==proof['audit_sha256']
            checked_audit=json.loads(Path(proof['audit']).read_text())
            assert checked_audit['status']=='verified'
            assert checked_audit['evidence'][proof['report']]==proof['sha256']
        if heading_trace:
            proof=p['route_heading_smoke']
            assert sha256_file(proof['report'])==proof['sha256']
            checked=json.loads(Path(proof['report']).read_text())
            assert checked['status']=='complete' and checked['route_heading_options']==heading_options
            assert checked['base_sha256']==p['base_sha256'] and checked['route_heading_timing_passed']
        if p.get('recovery_resets'):
            assert p['recovery_resets']['split']=='train'
            assert sha256_file(p['recovery_smoke_report'])==p['recovery_smoke_sha256']
            assert json.loads(Path(p['recovery_smoke_report']).read_text())['status']=='complete'
        if p.get('flat_anchor_weight') is not None:
            smoke=json.loads(Path(p['smoke_report']).read_text())
            assert smoke['flat_anchor_timing_and_mask_passed'] and smoke['flat_anchor_model_unchanged']
        recovery_anchor = p.get('recovery_anchor')
        if recovery_anchor is not None:
            assert p.get('flat_anchor_weight') is None and p.get('recovery_resets')
            smoke = json.loads(Path(recovery_anchor['smoke_report']).read_text())
            assert sha256_file(recovery_anchor['smoke_report']) == recovery_anchor['smoke_sha256']
            assert smoke['status'] == 'complete' and smoke['recovery_anchor_timing_and_mask_passed']
            assert smoke['anchor_model_unchanged'] and smoke['base_sha256'] == p['base_sha256']
            assert smoke['anchor_options'] == {k: recovery_anchor[k] for k in ('weight', 'early_steps')}
        app=AppLauncher(args)
        stop_requested=[]
        def request_stop(signum, frame):stop_requested.append(signum)
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        from rsl_rl.runners import OnPolicyRunner
        import rl_training.tasks  # noqa: F401
        from rl_training.policy_learning import warmstart_actor_only
        from rl_training.lite3_joint_conventions import clip_lite3_policy_action
        from rl_training.online_adaptation.rsl_rl_compat import prepare_rsl_rl_agent_cfg
        from rl_training.online_adaptation.step_trajectory import StepTrajectory
        from rl_training.online_adaptation.checkpointing import capture_rng_state
        torch.manual_seed(p['seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        cfg=parse_env_cfg(p['task'],device=args.device,num_envs=p['num_envs']);cfg.seed=p['seed']
        if p.get('fixed_forward_command') is not None:
            from rl_training.online_adaptation.runner_contract import configure_forward_command
            configure_forward_command(cfg.commands.base_velocity, p['fixed_forward_command'])
        route=p.get('route_observations')
        if route:
            from rl_training.policy_route import ROUTE_SCHEMA, configure_route_training, expand_actor_route_inputs, route_heading_trace
            assert route==ROUTE_SCHEMA
            configure_route_training(cfg,p['route_training'])
        if velocity_options:
            assert route
            from rl_training.policy_route import configure_lateral_velocity_training, route_lateral_velocity_trace
            configure_lateral_velocity_training(cfg,velocity_options)
        if p.get('recovery_resets'):
            from rl_training.recovery_teacher import configure_recovery_resets, check_recovery_event_order
            configure_recovery_resets(cfg,p['recovery_resets'])
            cfg.rewards.course_lateral.weight=p['recovery_course_lateral_weight']
        agent_cfg=load_cfg_from_registry(p['task'],'rsl_rl_cfg_entry_point')
        agent_cfg.device=args.device;agent_cfg.seed=p['seed'];agent_cfg.num_steps_per_env=p['rollout_steps']
        agent_cfg.save_interval=p['save_interval'];agent_cfg.policy.init_noise_std=p['initial_std']
        for key,value in p['ppo'].items():setattr(agent_cfg.algorithm,key,value)
        if p.get('symmetry'):
            from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
            agent_cfg.algorithm.symmetry_cfg=RslRlSymmetryCfg(**p['symmetry'])
        agent_cfg=prepare_rsl_rl_agent_cfg(agent_cfg)
        anchor_weight=p.get('flat_anchor_weight')
        teacher=None
        if anchor_weight is not None:
            from rl_training.policy_flat_anchor import FrozenFlatMean, configure_flat_anchor, prepare_flat_anchor, flat_anchor_cost
            configure_flat_anchor(cfg,anchor_weight)
        if recovery_anchor is not None:
            from rl_training.policy_recovery_anchor import configure_recovery_anchor, prepare_recovery_anchor
            from rl_training.policy_flat_anchor import FrozenFlatMean, flat_anchor_cost
            configure_recovery_anchor(cfg, recovery_anchor)
        env=gym.make(p['task'],cfg=cfg);raw=env.unwrapped
        anchor_reward_index=raw.reward_manager.active_terms.index('recovery_policy_anchor') if recovery_anchor is not None else None
        if p.get('recovery_resets'):
            check_recovery_event_order(raw)
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent_cfg.clip_actions)
        if history_options is not None:
            wrapped=RouteHistoryVecEnv(wrapped,history_options)
            report['route_history_options']=history_options
        trace=StepTrajectory(out/'trajectory',metadata={'kind':'vector_training','num_envs':p['num_envs'],
            'observation':'pre-action','next_observation_and_root':'step return; reset rows already reset',
            'action':'requested for PPO likelihood; emitted clipped command drives physical action history'},chunk_steps=p['rollout_steps'],vector_dtype='float32')
        def before(obs,action):
            if stop_requested:raise InterruptedError('stop requested before next physical step')
            if not torch.isfinite(action).all():raise ValueError('nonfinite requested action')
            extra={}
            if history_options is not None:extra.update(wrapped.history_snapshot())
            if heading_trace:raw._route_heading_snapshot=None
            if velocity_options:raw._route_velocity_snapshot=None
            if p.get('fixed_forward_command') is not None:
                expected=obs['policy'].new_tensor([p['fixed_forward_command'],0.,0.]).expand(p['num_envs'],3)
                torch.testing.assert_close(obs['policy'][:,6:9],expected,atol=1e-7,rtol=0)
            if p.get('recovery_resets'):
                extra.update(recovery_case=raw._recovery_case,pre_root_state=raw.scene['robot'].data.root_state_w)
            if recovery_anchor is not None:
                target,mask=prepare_recovery_anchor(raw,teacher,obs['policy'],recovery_anchor['early_steps'])
                report['recovery_anchor_label_rows']+=int(mask.sum())
                extra.update(recovery_anchor_target=target,recovery_anchor_mask=mask,
                    recovery_anchor_cost=flat_anchor_cost(clip_lite3_policy_action(action),target,mask),
                    episode_steps=raw.episode_length_buf.clone())
            elif teacher is not None:
                target,mask=prepare_flat_anchor(raw,teacher,obs['policy'])
                report['flat_anchor_label_rows']+=int(mask.sum())
                extra.update(flat_anchor_target=target,flat_anchor_mask=mask,flat_anchor_cost=flat_anchor_cost(clip_lite3_policy_action(action),target,mask))
            return trace.snapshot(policy_observation=obs['policy'],critic_observation=obs['critic'],requested_action=action,emitted_action=clip_lite3_policy_action(action),**extra)
        def after(pre,result):
            obs,reward,done,extras=result
            if recovery_anchor is not None:
                torch.testing.assert_close(raw._recovery_anchor_applied_cost,torch.as_tensor(pre['recovery_anchor_cost'],device=done.device),atol=1e-6,rtol=0)
                anchor_component=raw.reward_manager._step_reward[:,anchor_reward_index]*raw.step_dt
                torch.testing.assert_close(anchor_component,-recovery_anchor['weight']*raw._recovery_anchor_applied_cost*raw.step_dt,atol=1e-6,rtol=1e-5)
            elif teacher is not None:
                torch.testing.assert_close(raw._flat_anchor_applied_cost,torch.as_tensor(pre['flat_anchor_cost'],device=done.device),atol=1e-6,rtol=0)
            expected_history=torch.as_tensor(pre["emitted_action"],device=done.device).clone()
            expected_history[done.bool()]=0
            for group in ("policy","critic"):
                torch.testing.assert_close(obs[group][...,-12:],expected_history,atol=1e-6,rtol=0)
            assert 'time_outs' not in extras or not extras['time_outs'].bool().any()
            manager=raw.termination_manager
            failure=manager.get_term('illegal_contact')|manager.get_term('bad_orientation_2')|manager.get_term('course_lateral')
            success=manager.get_term('course_success')&~failure
            deadline=manager.get_term('time_out')&~failure&~success
            assert torch.equal(done.bool(),failure|success|deadline)
            for name,mask in (('success',success),('failure',failure),('deadline',deadline)):
                report['episode_counts'][name]+=int(mask.sum())
            if p.get('recovery_resets'):
                recovery=torch.as_tensor(pre['recovery_case'],device=done.device)>=0
                for name,mask in (('success',success),('failure',failure),('deadline',deadline)):
                    report.setdefault('recovery_episode_counts',{}).setdefault(name,0)
                    report['recovery_episode_counts'][name]+=int((mask&recovery).sum())
            trace.append(**pre,next_policy_observation=obs['policy'],next_critic_observation=obs['critic'],
                reward=reward,done=done,returned_root_state=raw.scene['robot'].data.root_state_w,
                task_success=success,task_failure=failure,task_deadline=deadline,
                **(dict(recovery_anchor_applied_cost=raw._recovery_anchor_applied_cost,recovery_anchor_reward=anchor_component) if recovery_anchor is not None else {}),
                **(route_heading_trace(raw) if heading_trace else {}),
                **(route_lateral_velocity_trace(raw) if velocity_options else {}))
        counted=ChargedVecEnv(wrapped,ledger,before_step=before,after_step=after,action_transform=clip_lite3_policy_action)
        runner=OnPolicyRunner(counted,agent_cfg.to_dict(),log_dir=str(out/'rsl'),device=args.device)
        obs=counted.get_observations()
        assert obs['policy'].shape==(p['num_envs'],49 if history_options is not None else (48 if route else 45)) and obs['critic'].shape==(p['num_envs'],51 if route else 48)
        payload=torch.load(p['base_checkpoint'],map_location='cpu',weights_only=False)
        if anchor_weight is not None or recovery_anchor is not None:
            teacher=FrozenFlatMean(payload).to(args.device).eval()
            teacher_initial={k:v.clone() for k,v in teacher.state_dict().items()}
            if recovery_anchor is not None:
                report['recovery_anchor_label_rows']=0
                report['recovery_anchor_options']={k: recovery_anchor[k] for k in ('weight','early_steps')}
            else:
                report['flat_anchor_label_rows']=0
                report['flat_anchor_weight']=anchor_weight
        if route:
            if p.get('base_observation_schema') == route:
                assert payload['actor_state_dict']['mlp.0.weight'].shape[1]==48
            else:
                payload=expand_actor_route_inputs(payload)
        report['route_observations']=route
        warm_payload=expand_actor_history_inputs(payload) if history_options is not None else payload
        report['warmstart']=warmstart_actor_only(runner,warm_payload,initial_std=p['initial_std'],allow_mean_only=p.get('base_actor_mean_only',False))
        mean_replay=None
        tail_replay=None
        assert not p.get('tail_anchor') or p.get('mean_anchor'), 'tail anchor requires the registered mean replay'
        if p.get('mean_anchor'):
            import numpy as np
            from rl_training.online_adaptation.flowdagger import FrozenPPOExpert
            from rl_training.policy_mean_anchor import MeanActionReplay, update_with_mean_anchor
            spec=p['mean_anchor']
            assert sha256_file(spec['dataset'])==spec['dataset_sha256']
            assert sha256_file(spec['dataset_audit'])==spec['dataset_audit_sha256']
            assert json.loads(Path(spec['dataset_audit']).read_text())['status']=='verified'
            assert set(spec['batch_groups'])=={'correction','flat','rough101','rough107'}
            assert spec.get('fixed_target_groups',[]) in ([],['correction'])
            with np.load(spec['dataset'],allow_pickle=False) as data:
                states={group:torch.as_tensor(data[group+'_train_obs'].copy(),device=args.device) for group in spec['batch_groups']}
                fixed_targets={group:torch.as_tensor(data[group+'_train_target'].copy(),device=args.device) for group in spec.get('fixed_target_groups',[])}
            if history_options is not None:
                assert all(value.shape[1]==49 for value in states.values())
                if not history_options['enabled']:
                    for value in states.values():value[:,12]=0
            # Constructing Linear layers consumes CPU RNG even when loading frozen weights.
            # Restore it so adding replay does not change the paired PPO sampling stream.
            with torch.random.fork_rng(devices=[]):
                reference=FrozenPPOExpert(payload['actor_state_dict']).to(args.device)
            mean_replay=MeanActionReplay(reference,states,spec['batch_groups'],coefficient=spec['coefficient'],seed=spec['seed'],device=args.device,fixed_targets=fixed_targets,group_weights=spec.get('group_weights'))
            report['mean_anchor']={'dataset_sha256':spec['dataset_sha256'],'group_rows':{k:len(v) for k,v in states.items()},
                'coefficient':spec['coefficient'],'seed':spec['seed'],'batch_groups':spec['batch_groups'],'teacher_sha256':p['base_sha256'],
                'validation_rows_used':0,'sampling_device':'cpu','calls':0,'rows':0}
            if 'group_weights' in spec:
                report['mean_anchor']['group_weights']=mean_replay.group_weights
                report['mean_anchor']['weight_denominator']='all sampled rows times12; not sum of weights'
            if fixed_targets:
                report['mean_anchor']['fixed_target_groups']=list(fixed_targets)
                report['mean_anchor']['fixed_target_array_sha256']={group:hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for group,value in fixed_targets.items()}
                assert report['mean_anchor']['fixed_target_array_sha256']==spec['fixed_target_array_sha256']
            if p.get('tail_anchor'):
                tail=p['tail_anchor']
                assert set(tail)=={'batch_groups','coefficient','seed','observation_array_sha256'}
                assert tail['batch_groups']=={'post_recovery':128}
                assert tail['seed']!=spec['seed']
                with np.load(spec['dataset'],allow_pickle=False) as data:
                    tail_states=data['post_recovery_train_obs'].copy()
                assert hashlib.sha256(tail_states.tobytes()).hexdigest()==tail['observation_array_sha256']
                tail_replay=MeanActionReplay(reference,{'post_recovery':torch.as_tensor(tail_states,device=args.device)},
                    tail['batch_groups'],coefficient=tail['coefficient'],seed=tail['seed'],device=args.device)
                report['tail_anchor']=dict(tail,group_rows={'post_recovery':len(tail_states)},dataset_sha256=spec['dataset_sha256'],
                    teacher_sha256=p['base_sha256'],validation_rows_used=0,stored_target_rows_used=0,sampling_device='cpu',calls=0,rows=0)
        initial={k:v.detach().cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
        dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent_cfg)
        report['versions']={name:importlib.metadata.version(name) for name in ('torch','rsl-rl-lib')}
        gradient_observer=None
        if p.get('gradient_diagnostics'):
            assert p['gradient_diagnostics']=={'minibatch':0,'frequency_updates':1,'parameters':'actor_mean'}
            assert mean_replay is not None and tail_replay is None
            report['gradient_diagnostics']=dict(options=p['gradient_diagnostics'],records=0)
            def gradient_observer(values):
                event=dict(update=report['ppo_updates']+1,minibatch=0,**values)
                with (out/'gradient_diagnostics.jsonl').open('a') as stream:stream.write(json.dumps(event,sort_keys=True)+'\n')
                report['gradient_diagnostics']['records']+=1
        update=(lambda:update_with_mean_anchor(runner.alg,mean_replay,tail_replay=tail_replay,gradient_observer=gradient_observer)) if mean_replay is not None else runner.alg.update
        def checked_update():
            losses=update()
            if mean_replay is not None:
                report['mean_anchor'].update(calls=mean_replay.calls,rows=mean_replay.rows,draw_sha256=mean_replay.draw_digest.hexdigest())
                if mean_replay.fixed_targets:
                    report['mean_anchor'].update(fixed_target_rows=mean_replay.fixed_target_rows,fixed_target_sha256=mean_replay.fixed_target_digest.hexdigest())
            if tail_replay is not None:
                report['tail_anchor'].update(calls=tail_replay.calls,rows=tail_replay.rows,draw_sha256=tail_replay.draw_digest.hexdigest())
            for model in (runner.alg.actor,runner.alg.critic):
                if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite learned parameters')
            if not all(math.isfinite(float(v)) for v in losses.values()):raise ValueError('nonfinite PPO losses')
            report['ppo_updates']+=1
            event={'update':report['ppo_updates'],'confirmed_transitions':ledger.state['confirmed'],
                   'losses':{k:float(v) for k,v in losses.items()},'episode_counts':dict(report['episode_counts'])}
            with (out/'updates.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            report['ledger']=dict(ledger.state);write_json_atomic(out/'report.json',report)
            return losses
        runner.alg.update=checked_update
        runner.learn(num_learning_iterations=p['iterations'],init_at_random_ep_len=False)
        assert report['ppo_updates']==p['iterations'] and ledger.state['confirmed']==expected
        if gradient_observer is not None:
            assert report['gradient_diagnostics']['records']==p['iterations']
            report['gradient_diagnostics']['sha256']=sha256_file(out/'gradient_diagnostics.jsonl')
        assert any(not torch.equal(v,runner.alg.actor.state_dict()[k].cpu()) for k,v in initial.items() if k.startswith('mlp.'))
        assert sha256_file(p['base_checkpoint'])==p['base_sha256']
        assert sources=={str(x.relative_to(root)):sha256_file(x) for x in paths}
        if teacher is not None:
            assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in teacher_initial.items())
            report['recovery_anchor_model_unchanged' if recovery_anchor is not None else 'flat_anchor_model_unchanged']=True
        if mean_replay is not None:
            assert mean_replay.calls==p['iterations']*p['ppo']['num_learning_epochs']*p['ppo']['num_mini_batches']
            assert all(torch.equal(value.cpu(),payload['actor_state_dict'][key]) for key,value in mean_replay.teacher.state_dict().items())
            assert sha256_file(p['mean_anchor']['dataset'])==p['mean_anchor']['dataset_sha256']
            report['mean_anchor']['teacher_unchanged']=True
        if tail_replay is not None:
            assert tail_replay.calls==mean_replay.calls
            assert tail_replay.teacher is mean_replay.teacher
            report['tail_anchor']['teacher_unchanged']=True
        checkpoint=runner.alg.save();checkpoint.update(iter=runner.current_learning_iteration,
            infos={'protocol_sha256':protocol_hash,'base_sha256':p['base_sha256'],'rng_state':capture_rng_state()})
        torch.save(checkpoint,out/'final.pt')
        trace.close(complete=True);ledger.finish()
        report.update(status='complete',ledger=ledger.state,seconds=time.perf_counter()-started,
            final_checkpoint={'path':str((out/'final.pt').resolve()),'sha256':sha256_file(out/'final.pt')},
            mean_actor_changed=True,optimizer_steps=sorted({int(v['step']) for v in runner.alg.optimizer.state.values()}))
        write_json_atomic(out/'report.json',report)
    except BaseException as error:
        if ledger.state['status']=='running':ledger.finish(aborted=True)
        if trace is not None:trace.close(complete=False)
        report.update(status='failed',error=repr(error),ledger=ledger.state,seconds=time.perf_counter()-started)
        write_json_atomic(out/'report.json',report)
        raise
    finally:
        if env is not None:env.close()
        if app is not None:app.app.close()


if __name__=='__main__':main()
