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
        agent_cfg=load_cfg_from_registry(p['task'],'rsl_rl_cfg_entry_point')
        agent_cfg.device=args.device;agent_cfg.seed=p['seed']
        agent_cfg=prepare_rsl_rl_agent_cfg(agent_cfg)
        env=gym.make(p['task'],cfg=cfg)
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent_cfg.clip_actions)
        runner=OnPolicyRunner(wrapped,agent_cfg.to_dict(),log_dir=None,device=args.device)
        payload=torch.load(p['base_checkpoint'],map_location='cpu',weights_only=False)
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
            with torch.inference_mode():action=policy(observations)
            ledger.reserve()
            observations,reward,done,extras=wrapped.step(action)
            ledger.confirm()
            assert torch.isfinite(observations['policy']).all() and torch.isfinite(reward).all()
            if forced:
                assert done.bool().all()
                assert 'time_outs' not in extras or not extras['time_outs'].bool().any()
            rows.append({'observation':pre.cpu().numpy(),'action':action.cpu().numpy(),
                         'reward':reward.cpu().numpy(),'done':done.cpu().numpy(),
                         'returned_root_pos':raw.scene['robot'].data.root_pos_w.cpu().numpy().copy()})
        np.savez_compressed(out/'trajectory.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        for before,model in ((actor_before,runner.alg.actor),(critic_before,runner.alg.critic)):
            for k,v in before.items():assert torch.equal(v,model.state_dict()[k])
        assert not runner.alg.optimizer.state and sha256_file(p['base_checkpoint'])==p['base_sha256']
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
