"""Validate low-stair geometry and reward accounting with a frozen old actor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from reference_lite3_loader import ROOT, load_reference, resolve_lite3_asset, source_manifest
from reference_stairs_eval import MODELS, load_actor, save, sha


def geometry_audit(base, cfg):
    import torch
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh
    from reference_six_terrain_cfg import column_names
    names=column_names(cfg);rows=[]
    offsets=torch.zeros(306,3,device=base.device)
    offsets[:,0]=torch.linspace(.005,3.055,306,device=base.device)
    offsets[:,1]=.013;offsets[:,2]=2.
    rays=torch.zeros_like(offsets);rays[:,2]=-1
    for row in range(10):
        for col,name in enumerate(names):
            if name not in ('stairs_up','stairs_down'):continue
            origin=base.scene.terrain.terrain_origins[row,col]
            heights=raycast_mesh(origin+offsets,rays,RayCaster.meshes[cfg.scene.terrain.prim_path])[0][:,2]
            assert torch.isfinite(heights).all()
            delta=heights[1:]-heights[:-1]
            steps=delta[delta.abs()>.005]
            assert len(steps)==7,(row,col,steps)
            assert (steps>0).all() if name=='stairs_up' else (steps<0).all()
            assert steps.abs().min()>=.0199 and steps.abs().max()<=.0501
            rows.append(dict(row=row,column=col,direction=name,step_count=len(steps),
                             min_height_m=float(steps.abs().min()),max_height_m=float(steps.abs().max())))
    assert len(rows)==40
    return dict(status='passed',tiles=rows,all_40_stair_tiles_in_2_to_5_cm=True,
                maximum_step_height_m=max(r['max_height_m'] for r in rows),
                six_to_ten_cm_absent_from_training_stairs=True)


def collect(args):
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    reference,agent=load_reference()
    from reference_low_stairs_cfg import apply_terrain, apply_rewards
    from reference_low_stairs_rewards import sample
    from reference_reward_variants import feet_air_time_lin_xy_nonnegative
    from reference_six_terrain_cfg import replace_terrain, column_names, FAMILIES
    from reference_six_train import configuration_audit, audit_simulation
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as original_rewards

    six=replace_terrain(reference)
    save(args.output/'reference_audit.json',configuration_audit(reference,six,agent))
    cfg,terrain_audit=apply_terrain(six);cfg,reward_audit=apply_rewards(cfg)
    save(args.output/'terrain_audit.json',terrain_audit);save(args.output/'reward_audit.json',reward_audit)
    resolve_lite3_asset(cfg)
    cfg.seed=20261112;cfg.scene.num_envs=120;cfg.sim.device='cuda:0'
    cfg.commands.base_velocity.debug_vis=False;cfg.scene.terrain.terrain_generator.use_cache=False
    from reference_reward_variants import plain
    save(args.output/'config.json',plain(cfg))
    base=ManagerBasedRLEnv(cfg=cfg);obs,_=base.reset()
    audit_simulation(base,cfg,args.output,make_plot=True)
    geometry=geometry_audit(base,cfg);save(args.output/'geometry.json',geometry)
    assert int(base.scene.terrain.terrain_levels.max())<=1
    names=column_names(cfg)
    initial_stair_heights=[r for r in geometry['tiles'] if r['row']<=1]
    assert max(r['max_height_m'] for r in initial_stair_heights)<.03
    model_sha=sha(MODELS['update10000']);actor=load_actor('update10000').to(base.device)
    digest=lambda:hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in actor.state_dict().values())).hexdigest()
    before=digest();command=base.command_manager.get_term('base_velocity');tick=0
    schedule=((0.,0.,0.),(0.,0.,.5),(0.,0.,-.5),(.4,0.,0.))
    def set_command(next_tick):
        command.is_heading_env[:]=command.is_standing_env[:]=False
        command.vel_command_b[:]=command.vel_command_b.new_tensor(schedule[min(next_tick//60,3)])
    original_compute=base.command_manager.compute
    def command_compute(dt):original_compute(dt);set_command(tick+1)
    base.command_manager.compute=command_compute;set_command(0);obs=base.observation_manager.compute()
    reward_names=list(base.reward_manager.active_terms)
    landing=base.reward_manager.get_term_cfg('stair_landing').func
    expected_classes={'flat','rough','slope','stairs','friction','slope_rough'}
    assert {FAMILIES[names[int(i)]] for i in base.scene.terrain.terrain_types.cpu()}==expected_classes
    arrays={key:[] for key in ('command','new_rewards','root','feet','ground','obstacle','support','enabled','reset')}
    errors=[];reset_check=False;gated_samples=0;opportunity_samples=0
    original_reward=base.reward_manager.compute
    def capture(dt):
        nonlocal gated_samples,opportunity_samples
        total=original_reward(dt);rates=base.reward_manager._step_reward;x=sample(base)
        assert torch.isfinite(rates).all() and torch.allclose(total,rates.sum(-1)*dt,atol=1e-5,rtol=1e-4)
        gated=x['enabled'];gated_samples+=int(gated.sum());opportunity_samples+=int(x['upcoming'].sum())
        term=base.reward_manager.get_term_cfg('phase_foot_trajectory_exp')
        expected=original_rewards.phase_foot_trajectory_exp(base,**term.params)*term.weight*~gated
        error=(rates[:,reward_names.index('phase_foot_trajectory_exp')]-expected).abs().max()
        assert error<1e-6;errors.append(float(error))
        term=base.reward_manager.get_term_cfg('feet_air_time_lin_xy')
        old=original_rewards.feet_air_time_lin_xy_cmd(base,**term.params)
        new=feet_air_time_lin_xy_nonnegative(base,**term.params)
        expected=torch.where(gated,new,old)*term.weight
        assert torch.allclose(rates[:,reward_names.index('feet_air_time_lin_xy')],expected,atol=1e-6)
        new_rates=rates[:,[reward_names.index(k) for k in ('stair_clearance','stair_progress','stair_landing')]]
        assert (new_rates[~gated]==0.).all()
        assert (new_rates[:,0]<=0.).all() and (new_rates[:,2]>=0.).all()
        assert (new_rates[:,1].abs()<=1.).all()
        values=dict(command=command.command,new_rewards=new_rates,root=base.scene['robot'].data.root_pos_w,
                    feet=x['feet'],ground=x['ground'],obstacle=x['obstacle'],support=x['support'],enabled=gated,
                    reset=base.reset_terminated|base.reset_time_outs)
        for k,v in values.items():arrays[k].append(v.cpu().numpy().copy())
        return total
    base.reward_manager.compute=capture
    with torch.inference_mode():
        for tick in range(600):
            if tick==300:
                ids=torch.tensor([0,72,84],device=base.device)
                base._reset_idx(ids)
                assert not landing.state.initialized[ids].any() and not landing.state.air_time[ids].any()
                reset_check=True
                set_command(tick);obs=base.observation_manager.compute()
            assert torch.equal(obs['policy'][:,6:9],command.command)
            obs,reward,done,truncated,_=base.step(actor(obs['policy']).clamp(-100,100))
            assert obs['policy'].shape==(120,45) and obs['critic'].shape==(120,393)
            assert all(torch.isfinite(v).all() for v in obs.values())
            if (tick+1)%100==0:print(f'LOW STAIRS VALIDATION {tick+1}/600',flush=True)
    assert digest()==before and sha(MODELS['update10000'])==model_sha
    assert reset_check and gated_samples>0 and opportunity_samples>0
    data={k:np.asarray(v) for k,v in arrays.items()};np.savez_compressed(args.output/'trace.npz',**data)
    save(args.output/'report.json',dict(status='passed',environments=120,steps=600,optimizer_updates=0,
        frozen_diagnostic_model='update10000',checkpoint_sha256=model_sha,actor_unchanged=True,
        training_model_created=False,observation_dimensions=[45,393],families=sorted(expected_classes),
        geometry_verified=True,gated_samples=gated_samples,upcoming_foot_samples=opportunity_samples,
        phase_accounting_max_error=max(errors),selected_reset_clears_landing_history=reset_check,
        new_reward_integrals=data['new_rewards'].sum(axis=(0,1)).tolist(),rate_integral_factor=base.step_dt,
        sources=source_manifest(),
        note='Frozen integration check, not training or a claim of improved stair behavior. Old actor is not an initializer.'))
    print('LOW STAIRS GEOMETRY AND REWARD VALIDATION PASSED; no optimization',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,choices=(0,1),default=1)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--child',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],gpu=args.gpu,output=args.output,timeout=1800)
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,device='cuda:0',kit_args=f'--/renderer/multiGpu/enabled=False --/renderer/activeGpu={args.gpu}').app
        collect(args);sys.stdout.flush();os._exit(0)
    except Exception:
        traceback.print_exc();save(args.output/'error.json',dict(error=traceback.format_exc()))
        sys.stdout.flush();os._exit(1)


if __name__=='__main__':sys.exit(main())
