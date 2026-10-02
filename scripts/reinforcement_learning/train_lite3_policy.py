"""Fresh actor-initialized PPO training with immutable control-transition budget."""
import argparse
import copy
import importlib.metadata
import json
import math
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
        app=AppLauncher(args)
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
        agent_cfg=load_cfg_from_registry(p['task'],'rsl_rl_cfg_entry_point')
        agent_cfg.device=args.device;agent_cfg.seed=p['seed'];agent_cfg.num_steps_per_env=p['rollout_steps']
        agent_cfg.save_interval=p['save_interval'];agent_cfg.policy.init_noise_std=p['initial_std']
        for key,value in p['ppo'].items():setattr(agent_cfg.algorithm,key,value)
        if p.get('symmetry'):
            from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
            agent_cfg.algorithm.symmetry_cfg=RslRlSymmetryCfg(**p['symmetry'])
        agent_cfg=prepare_rsl_rl_agent_cfg(agent_cfg)
        env=gym.make(p['task'],cfg=cfg);raw=env.unwrapped
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent_cfg.clip_actions)
        trace=StepTrajectory(out/'trajectory',metadata={'kind':'vector_training','num_envs':p['num_envs'],
            'observation':'pre-action','next_observation_and_root':'step return; reset rows already reset',
            'action':'requested for PPO likelihood; emitted clipped command drives physical action history'},chunk_steps=p['rollout_steps'],vector_dtype='float32')
        def before(obs,action):
            if not torch.isfinite(action).all():raise ValueError('nonfinite requested action')
            return trace.snapshot(policy_observation=obs['policy'],critic_observation=obs['critic'],requested_action=action,emitted_action=clip_lite3_policy_action(action))
        def after(pre,result):
            obs,reward,done,extras=result
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
            trace.append(**pre,next_policy_observation=obs['policy'],next_critic_observation=obs['critic'],
                reward=reward,done=done,returned_root_state=raw.scene['robot'].data.root_state_w,
                task_success=success,task_failure=failure,task_deadline=deadline)
        counted=ChargedVecEnv(wrapped,ledger,before_step=before,after_step=after,action_transform=clip_lite3_policy_action)
        runner=OnPolicyRunner(counted,agent_cfg.to_dict(),log_dir=str(out/'rsl'),device=args.device)
        obs=counted.get_observations()
        assert obs['policy'].shape==(p['num_envs'],45) and obs['critic'].shape==(p['num_envs'],48)
        payload=torch.load(p['base_checkpoint'],map_location='cpu',weights_only=False)
        report['warmstart']=warmstart_actor_only(runner,payload,initial_std=p['initial_std'])
        initial={k:v.detach().cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
        dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent_cfg)
        report['versions']={name:importlib.metadata.version(name) for name in ('torch','rsl-rl-lib')}
        update=runner.alg.update
        def checked_update():
            losses=update()
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
        assert any(not torch.equal(v,runner.alg.actor.state_dict()[k].cpu()) for k,v in initial.items() if k.startswith('mlp.'))
        assert sha256_file(p['base_checkpoint'])==p['base_sha256']
        assert sources=={str(x.relative_to(root)):sha256_file(x) for x in paths}
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
