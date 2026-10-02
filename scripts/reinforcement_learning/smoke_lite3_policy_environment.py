"""Registered no-update vector-environment smoke and explicit deadline boundary."""
import argparse
import copy
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    from rl_training.online_adaptation.provenance import sha256_file,write_json_atomic
    from rl_training.vector_step_ledger import VectorStepLedger
    import json
    p=json.loads(Path(args.protocol).read_text());assert p['schema']=='lite3_full_policy_environment_smoke_v1'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    ledger=VectorStepLedger(out/'ledger.json',num_envs=p['num_envs'],max_transitions=p['max_control_transitions'],
                            metadata={'protocol_sha256':sha256_file(args.protocol)})
    report={'status':'running','new_optimizer_updates':0,'protocol_sha256':sha256_file(args.protocol)}
    write_json_atomic(out/'report.json',report)
    launcher=None;env=None
    try:
        assert sha256_file(p['base_checkpoint'])==p['base_sha256']
        launcher=AppLauncher(args)
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        import rl_training.tasks  # noqa: F401
        from rl_training.policy_learning import warmstart_actor_only
        from rl_training.online_adaptation.rsl_rl_compat import prepare_rsl_rl_agent_cfg
        torch.manual_seed(p['seed'])
        cfg=parse_env_cfg(p['task'],device=args.device,num_envs=p['num_envs'])
        cfg.seed=p['seed']
        route=p.get('route_observations')
        if route:
            from rl_training.policy_route import ROUTE_SCHEMA, configure_route_training, expand_actor_route_inputs
            assert route==ROUTE_SCHEMA
            configure_route_training(cfg,p['route_training'])
        from rl_training.lite3_joint_conventions import clip_lite3_policy_action
        agent_cfg=load_cfg_from_registry(p['task'],'rsl_rl_cfg_entry_point')
        agent_cfg.device=args.device;agent_cfg.seed=p['seed']
        agent_cfg=prepare_rsl_rl_agent_cfg(agent_cfg)
        anchor_weight=p.get('flat_anchor_weight')
        teacher=None
        if anchor_weight is not None:
            from rl_training.policy_flat_anchor import FrozenFlatMean,configure_flat_anchor,prepare_flat_anchor,flat_anchor_cost
            configure_flat_anchor(cfg,anchor_weight)
        env=gym.make(p['task'],cfg=cfg)
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent_cfg.clip_actions)
        runner=OnPolicyRunner(wrapped,agent_cfg.to_dict(),log_dir=None,device=args.device)
        payload=torch.load(p['base_checkpoint'],map_location='cpu',weights_only=False)
        if anchor_weight is not None:
            teacher=FrozenFlatMean(payload).to(args.device).eval()
            teacher_before=copy.deepcopy(teacher.state_dict())
            report['flat_anchor_label_rows']=0
        original_payload=payload
        if route:payload=expand_actor_route_inputs(payload)
        critic_before=copy.deepcopy(runner.alg.critic.state_dict())
        report['warmstart']=warmstart_actor_only(runner,payload)
        actor_before=copy.deepcopy(runner.alg.actor.state_dict())
        observations=wrapped.get_observations()
        assert observations['policy'].shape==(p['num_envs'],p['actor_obs_dim'])
        assert observations['critic'].shape==(p['num_envs'],p['critic_obs_dim'])
        raw=env.unwrapped
        report['observation_terms']=raw.observation_manager.active_terms
        report['policy_term_scales']={name:getattr(cfg.observations.policy,name).scale
                                      for name in raw.observation_manager.active_terms['policy']}
        report['action_joint_names']=cfg.actions.joint_pos.joint_names
        report['action_scales']=cfg.actions.joint_pos.scale
        report['action_clip']=cfg.actions.joint_pos.clip
        report['terrain_origins']=raw.scene.env_origins.cpu().tolist()
        report['terrain_types']=raw.scene.terrain.terrain_types.cpu().tolist()
        write_json_atomic(out/'report.json',report)
        policy=runner.get_inference_policy(device=args.device)
        rows=[]
        for step in range(p['normal_vector_steps']+p['forced_deadline_vector_steps']):
            forced=step==p['normal_vector_steps']
            if forced:raw.episode_length_buf.fill_(raw.max_episode_length)
            pre=observations['policy'].clone()
            if teacher is not None:
                target,mask=prepare_flat_anchor(raw,teacher,pre)
                report['flat_anchor_label_rows']+=int(mask.sum())
            with torch.inference_mode():
                action=policy(observations)
                if route:
                    reference=torch.cat((pre[:,:9],pre[:,12:]),dim=-1)
                    state=original_payload['actor_state_dict']
                    for index in (0,2,4,6):
                        reference=torch.nn.functional.linear(reference,state[f'mlp.{index}.weight'].to(pre.device),state[f'mlp.{index}.bias'].to(pre.device))
                        if index!=6:reference=torch.nn.functional.elu(reference)
                    torch.testing.assert_close(action,reference,atol=2e-5,rtol=1e-5)
                if teacher is not None and step==p['anchor_perturb_step']:
                    action=action.clone();action[:,0]+=p['anchor_perturb_action']
                action=clip_lite3_policy_action(action)
            if teacher is not None:
                anchor_cost=flat_anchor_cost(action,target,mask)
            ledger.reserve()
            observations,reward,done,extras=wrapped.step(action)
            ledger.confirm()
            if teacher is not None:
                torch.testing.assert_close(raw._flat_anchor_applied_cost,anchor_cost,atol=1e-6,rtol=0)
                assert (anchor_cost[~mask]==0).all()
                if step==p['anchor_perturb_step']:assert (anchor_cost[mask]>0).all()
            assert torch.isfinite(observations['policy']).all() and torch.isfinite(reward).all()
            expected=action.clone();expected[done.bool()]=0
            for group in ('policy','critic'):
                torch.testing.assert_close(observations[group][...,-12:],expected,atol=1e-6,rtol=0)
            if forced:
                assert done.bool().all()
                assert 'time_outs' not in extras or not extras['time_outs'].bool().any()
            rows.append({'observation':pre.cpu().numpy(),'action':action.cpu().numpy(),
                         'reward':reward.cpu().numpy(),'done':done.cpu().numpy(),
                         'returned_root_pos':raw.scene['robot'].data.root_pos_w.cpu().numpy().copy()})
            if teacher is not None:
                rows[-1].update(flat_anchor_target=target.cpu().numpy(),flat_anchor_mask=mask.cpu().numpy(),flat_anchor_cost=anchor_cost.cpu().numpy())
        np.savez_compressed(out/'trajectory.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        for before,model in ((actor_before,runner.alg.actor),(critic_before,runner.alg.critic)):
            for k,v in before.items():assert torch.equal(v,model.state_dict()[k])
        assert not runner.alg.optimizer.state and sha256_file(p['base_checkpoint'])==p['base_sha256']
        if teacher is not None:
            assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in teacher_before.items())
            report['flat_anchor_model_unchanged']=True
            report['flat_anchor_timing_and_mask_passed']=True
        if route:
            checkpoint=runner.alg.save();checkpoint.update(iter=0,infos={'engineering_only':True,'route_observations':route})
            torch.save(checkpoint,out/'expanded_base.pt')
            report['expanded_checkpoint']={'path':str((out/'expanded_base.pt').resolve()),'sha256':sha256_file(out/'expanded_base.pt')}
            report['route_initial_mean_parity']=True
        ledger.finish()
        report.update(status='complete',ledger=ledger.state,actor_critic_unchanged=True,deadline_terminal_no_bootstrap=True,
                      trajectory_sha256=sha256_file(out/'trajectory.npz'))
        write_json_atomic(out/'report.json',report)
    except BaseException as error:
        if ledger.state['status']=='running':ledger.finish(aborted=True)
        report.update(status='failed',error=repr(error),ledger=ledger.state)
        write_json_atomic(out/'report.json',report)
        raise
    finally:
        if env is not None:env.close()
        if launcher is not None:launcher.app.close()


if __name__=='__main__':main()
