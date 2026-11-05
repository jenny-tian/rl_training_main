"""One bounded, read-only reward/noise diagnostic; never constructs PPO or an optimizer."""
import argparse
from dataclasses import asdict
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'source/rl_training'), str(ROOT / 'scripts/tools')]
OLD = Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/logs/rsl_rl/deeprobotics_lite3_flat_adapted/2026-08-05_14-00-57_lite3_lowspeed_finetune1499_seed42_20260805/model_1998.pt')
NEW = ROOT / 'docs/lite3-source-training-delivery-revision-20261104/deliverables/actor_update_90.pt'
HASHES = ['87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c',
          '14d11c8b57392c3ac0ac1e08b26de1cc1254ee059d1f653172433df9a05bdf42']
BUDGET = Path('/home/tjx/rl_training_main/.worktrees/lite3-online-adaptation/.git/source-ppo-ledgers/lite3-l1-v1.json')
CASES = [('old_mean', 0, 'zero'), ('old_original_std', 0, 'saved'),
         ('old_std1', 0, 'one'), ('new_mean', 1, 'zero'), ('new_saved_std', 1, 'saved')]
STEPS, NUM_ENVS, NOISE_SEED = 400, 120, 132001
STATE_NAMES = ['vx_mps', 'vy_mps', 'vz_mps', 'velocity_error_squared',
               'base_height_above_scan_m', 'progress_m', 'joint_deviation_l2',
               'joint_velocity_rms', 'tilt_xy_norm']
ACTION_NAMES = ['mean_action_rms', 'noise_rms', 'raw_action_rms',
                'clipped_joint_fraction', 'projected', 'paused', 'executed_delta_max']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, slice):
        return dict(slice=[value.start, value.stop, value.step])
    if callable(value):
        return getattr(value, '__module__', type(value).__module__) + ':' + getattr(value, '__qualname__', type(value).__qualname__)
    if hasattr(value, '__dict__'):
        return plain(vars(value))
    raise TypeError(f'unsupported config value: {type(value)}')


def child(folder):
    from source_delivery_light import record_phase, close_comparison_environment, check_scans
    report = dict(status='starting',training_transitions=0,ppo_updates=0,
                  diagnostic_transitions=0,cases=[],checkpoint_sha256=HASHES,
                  budget_before=digest(BUDGET),steps_per_case=STEPS,num_envs=NUM_ENVS,
                  noise_seed=NOISE_SEED,state_names=STATE_NAMES,action_names=ACTION_NAMES)
    app = wrapped = None
    stack_file = (folder / 'python_stacks.log').open('w')
    faulthandler.enable(file=stack_file)

    def phase(name, label=None, limit=120):
        faulthandler.cancel_dump_traceback_later()
        record_phase(folder, name, label, limit)
        if limit:
            faulthandler.dump_traceback_later(max(1, limit - 5), file=stack_file)

    try:
        if [digest(p) for p in (OLD, NEW)] != HASHES:
            raise ValueError('checkpoint hash mismatch')
        phase('application_start')
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device='cuda:0').app
        import gymnasium as gym
        import numpy as np
        import torch
        import rl_training.tasks
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.models import MLPModel
        from rl_training.source_scan.training_env import SourceTrainingVecEnv
        from rl_training.source_scan.frozen_actor import FrozenSourceActor
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as reward_mdp
        from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_training_env_cfg import DeeproboticsLite3SourceTrainingEnvCfg
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        states = [torch.load(OLD, map_location='cpu', weights_only=True)['actor_state_dict'],
                  torch.load(NEW, map_location='cpu', weights_only=True)]
        identity0 = None
        for label, index, noise_kind in CASES:
            phase('environment_start', label)
            cfg = DeeproboticsLite3SourceTrainingEnvCfg(source_training_run=1,
                source_parallel_sampler_enabled=True, source_parallel_sampler_num_envs=NUM_ENVS,
                source_shared_bank_rows=1)
            cfg.sim.device = 'cuda:0'
            raw = gym.make('SourceTerrain-Deeprobotics-Lite3-v0', cfg=cfg)
            wrapped = SourceTrainingVecEnv(RslRlVecEnvWrapper(raw, clip_actions=None))
            base = wrapped.unwrapped
            obs = wrapped.get_observations()
            actor = FrozenSourceActor(states[index]).to('cuda:0')
            # Reconstruct the installed training actor for deterministic-output parity only.
            with torch.random.fork_rng(devices=[]):
                reference = MLPModel(obs, {'actor':['policy']}, 'actor', 12,
                    hidden_dims=[512,256,128], activation='elu', obs_normalization=False,
                    distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution',
                                      'std_type':'log','init_std':1.0})
            reference.load_state_dict(states[index], strict=True)
            reference = reference.to('cuda:0').eval().requires_grad_(False)
            saved_std = states[index]['distribution.log_std_param'].exp().to('cuda:0')
            std = torch.zeros_like(saved_std) if noise_kind == 'zero' else torch.ones_like(saved_std) if noise_kind == 'one' else saved_std
            generator = torch.Generator(device='cuda:0').manual_seed(NOISE_SEED)
            data = base.scene['robot'].data
            identity = dict(assignments=[asdict(t) for t in base._source_parallel_terrain_allocator.assignments],
                terrain_origins=base.scene.terrain.terrain_origins.cpu().tolist(),
                initial_state_sha256=hashlib.sha256(data.root_link_state_w.cpu().numpy().tobytes()
                    +data.joint_pos.cpu().numpy().tobytes()+data.joint_vel.cpu().numpy().tobytes()).hexdigest())
            if identity0 is not None and identity != identity0:
                raise ValueError('initial geometry/state differs across cases')
            identity0 = identity
            manager = base.reward_manager
            names = list(manager.active_terms)
            reward_cfg = {name:plain(manager.get_term_cfg(name)) for name in names}
            original_compute = manager.compute
            captured = {}

            def compute(dt):
                value = original_compute(dt)
                d = base.scene['robot'].data
                vel = d.root_lin_vel_b
                err2 = (vel[:,:2]-base.command_manager.get_command('base_velocity')[:,:2]).square().sum(-1)
                ground = base.scene.sensors['height_scanner_base'].data.ray_hits_w[:,:,2].mean(-1)
                state = torch.stack((vel[:,0],vel[:,1],vel[:,2],err2,d.root_link_pos_w[:,2]-ground,
                    d.root_link_pos_w[:,0]-base.scene.terrain.env_origins[:,0],
                    (d.joint_pos-d.default_joint_pos).norm(dim=-1),d.joint_vel.square().mean(-1).sqrt(),
                    d.projected_gravity_b[:,:2].norm(dim=-1)),dim=-1)
                captured.update(rewards=manager._step_reward.clone(),state=state,
                    columns=base.scene.terrain.terrain_types.clone(),gate=float(reward_mdp.gait_level))
                return value

            manager.compute = compute
            chunks = {k:[] for k in ('rewards','state','columns','actions','done','gate')}
            snapshots = {}
            terminal_counts = {}
            parity_max = reconstruction_max = 0.0
            check_scans(base)
            phase('actor_controls',label,180)
            with torch.inference_mode():
                for step in range(1,STEPS+1):
                    if not app.is_running():
                        raise RuntimeError('simulator stopped')
                    mean = actor(obs)
                    noise = torch.randn(mean.shape,device='cuda:0',generator=generator)*std
                    action = mean+noise
                    saved_action = action.clone()
                    if step in (1,100,200,400):
                        ref = reference(obs,stochastic_output=False)
                        parity_max = max(parity_max,float((ref-mean).abs().max()))
                        snapshots[step] = dict(observation=obs['policy'].cpu().clone(),
                            mean=mean.cpu().clone(),reference_mean=ref.cpu().clone(),std=std.cpu().clone())
                    obs,reward,done,extras = wrapped.step(action)
                    trace = wrapped.last_trace
                    if not torch.equal(action,saved_action):
                        raise RuntimeError('adapter modified PPO input in place')
                    reconstruction_max = max(reconstruction_max,
                        float((captured['rewards'].sum(-1)*base.step_dt-reward).abs().max()))
                    actions = torch.stack((mean.square().mean(-1).sqrt(),noise.square().mean(-1).sqrt(),
                        action.square().mean(-1).sqrt(),(trace.raw_policy_action!=trace.base_action).float().mean(-1),
                        trace.decision.projected[:,0].float(),trace.decision.paused[:,0].float(),
                        (trace.executed_action-action).abs().amax(-1)),dim=-1)
                    for key in ('rewards','state','columns'):
                        chunks[key].append(captured[key].cpu().numpy().copy())
                    chunks['actions'].append(actions.cpu().numpy())
                    chunks['done'].append(done.cpu().numpy())
                    chunks['gate'].append(captured['gate'])
                    terminal = extras['source_terminal']
                    if terminal is not None:
                        for name,mask in terminal['terms'].items():
                            terminal_counts[name] = terminal_counts.get(name,0)+int(mask.sum())
                    check_scans(base)
                    report['diagnostic_transitions'] += NUM_ENVS
                    if step % 100 == 0:
                        print(f'[diagnosis] {label} {step}/{STEPS} gate={captured["gate"]}',flush=True)
            manager.compute = original_compute
            phase('case_finalize',label,15)
            arrays = {k:np.asarray(v) for k,v in chunks.items()}
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise FloatingPointError('nonfinite diagnostic output')
            if parity_max > 1e-6 or reconstruction_max > 1e-5:
                raise RuntimeError('actor parity or reward reconstruction mismatch')
            if any(not torch.equal(v.cpu(),states[index]['mlp.'+k]) for k,v in actor.mlp.state_dict().items()):
                raise RuntimeError('actor parameters changed')
            np.savez_compressed(folder/(label+'.npz'),**arrays)
            torch.save(snapshots,folder/(label+'_observations.pt'))
            case = dict(label=label,std=std.cpu().tolist(),identity=identity,reward_names=names,
                reward_config=reward_cfg,control_dt=base.step_dt,terminal_counts=terminal_counts,
                gait_level_min=float(arrays['gate'].min()),gait_level_max=float(arrays['gate'].max()),
                actor_parity_max_abs=parity_max,reward_reconstruction_max_abs=reconstruction_max,
                actor_parameters_unchanged=True,input_actions_unchanged=True,
                means=dict(reward_per_control=float(arrays['rewards'].sum(-1).mean()*base.step_dt),
                    vx_mps=float(arrays['state'][:,:,0].mean()),
                    velocity_rms_error_mps=float(np.sqrt(arrays['state'][:,:,3].mean()))))
            report['cases'].append(case)
            write_json(folder/'result.json',report)
            print('[diagnosis] completed '+label+' '+json.dumps(case['means']),flush=True)
            phase('environment_close',label,15)
            close_comparison_environment(wrapped)
            wrapped = None
            if label != CASES[-1][0]:
                phase('stage_replace',label)
                from isaaclab.sim import create_new_stage
                create_new_stage()
        if [digest(p) for p in (OLD,NEW)] != HASHES or digest(BUDGET) != report['budget_before']:
            raise RuntimeError('checkpoint or training budget changed')
        report.update(status='passed',budget_after=digest(BUDGET))
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(folder/'result.json',report)
        if wrapped is not None:
            phase('environment_close',limit=15)
            close_comparison_environment(wrapped)
        if app is not None:
            phase('application_close',limit=15)
            app.close()
        phase('finished',limit=None)
        faulthandler.disable()
        stack_file.close()
    return 0 if report['status'] == 'passed' else 1


def supervise(folder):
    from source_delivery_light import lifecycle_deadline
    folder.mkdir(parents=True,exist_ok=False)
    before = digest(BUDGET)
    started = time.monotonic()
    write_json(folder/'manifest.json',dict(source_revision=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),probe_sha256=digest(__file__),
        python=sys.executable,cases=CASES,control_steps=STEPS,num_envs=NUM_ENVS,
        diagnostic_transition_cap=STEPS*NUM_ENVS*len(CASES),wall_limit_s=600,
        budget_sha256=before,training_transitions=0,ppo_updates=0))
    with (folder/'runtime.log').open('w') as log:
        proc = subprocess.Popen([sys.executable,'-u',__file__,'--child','--output',str(folder)],
                                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        reason = None
        while proc.poll() is None:
            now = time.monotonic()
            reason = 'wall limit exceeded' if now-started > 600 else lifecycle_deadline(folder,now)
            if reason:
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                break
            time.sleep(1)
        code = proc.wait()
    try:
        os.killpg(proc.pid,0)
        remaining = True
        os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:
        remaining = False
    receipt = dict(exit_code=code,deadline_failure=reason,elapsed_s=time.monotonic()-started,
        process_group_had_remaining_children=remaining,budget_before=before,budget_after=digest(BUDGET))
    write_json(folder/'execution.json',receipt)
    print(json.dumps(receipt),flush=True)
    return code if code else int(reason is not None or remaining or before != digest(BUDGET))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    args = parser.parse_args()
    raise SystemExit(child(args.output) if args.child else supervise(args.output))
