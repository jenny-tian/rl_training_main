"""Bounded production collection and paired actor diagnostics for the approved stage."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
from rl_training.source_scan.delivery_stage import preflight,file_digest
from rl_training.source_scan.smoke_artifacts import atomic_json


def record_phase(folder,phase,actor=None,limit_s=None):
    """Publish lifecycle deadlines independently of buffered simulator output."""
    row=dict(phase=phase,actor=actor,started_mono=time.monotonic(),limit_s=limit_s)
    from rl_training.source_scan.delivery_metrics import append_json
    append_json(Path(folder)/'phases.jsonl',row)
    atomic_json(Path(folder)/'phase.json',row)
    print('[source-delivery] phase='+phase+' actor='+str(actor),flush=True)


def close_comparison_environment(wrapped):
    """Stop the old timeline before a new context registers its STOP callback."""
    sim=wrapped.unwrapped.sim
    # IsaacLab's STOP callback waits for play. Suppress it only after all
    # actor controls and metrics are finished, when disposing of this context.
    sim._disable_app_control_on_stop_handle=True
    sim.stop()
    if sim.is_playing():raise RuntimeError('comparison timeline did not stop before teardown')
    wrapped.close()


def persist_actor_and_close(writer,report,wrapped,phase,kind,label):
    writer.persist(report)
    phase('environment_close',label,15)
    if kind=='compare':close_comparison_environment(wrapped)
    else:wrapped.close()


def lifecycle_deadline(folder,now):
    path=Path(folder)/'phase.json'
    if not path.exists():return None
    row=json.loads(path.read_text());limit=row['limit_s']
    if limit is not None and now-row['started_mono']>limit:
        return f"{row['phase']} actor={row['actor']} exceeded {limit}s"
    return None


def check_scans(base):
    import torch
    from rl_training.source_scan.moving_diagnostic import check_scan
    robot=base.scene['robot'].data
    for name in ('height_scanner','height_scanner_base'):
        sensor=base.scene.sensors[name];value=sensor.data
        mesh=torch.as_tensor(sensor._mesh_ids_wp.numpy(),dtype=torch.int64,device=base.device)
        check_scan(robot.root_link_pos_w,value.ray_hits_w,value.ray_mesh_ids,mesh,
            sensor.meshes['/World/ground'].id,base.scene.terrain.env_origins,
            age=sensor._timestamp-sensor._timestamp_last_update,max_age=1e-6 if name.endswith('_base') else .02,
            base=name.endswith('_base'))


def probe(args):
    plan=json.loads(args.manifest.read_text());kind=plan['delivery_job']
    preflight(args,plan,kind=kind,child=True)
    from rl_training.source_scan.smoke_artifacts import SmokeArtifactWriter,finalize_and_close
    folder=Path(plan['output_dir']);report=plan|dict(status='starting',simulation_control_steps=0,
        completed_diagnostic_transitions=0,reset_latency_s=[],reset_samples=[],scheduled_ready_samples=[],actors=[],finite_controls=0)
    writer=SmokeArtifactWriter(folder/'result.json',report,schema=plan['schema'])
    import faulthandler
    stack_log=(folder/'python_stacks.log').open('w')
    faulthandler.enable(file=stack_log)
    def phase(name,label=None,limit=None):
        faulthandler.cancel_dump_traceback_later()
        record_phase(folder,name,label,limit)
        if limit is not None:faulthandler.dump_traceback_later(max(1,limit-5),file=stack_log)
    app=wrapped=base=None
    try:
        phase('application_start',limit=120)
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,device='cuda:0').app
        import gymnasium as gym
        import numpy as np
        import torch
        import rl_training.tasks
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rl_training.source_scan.training_env import SourceTrainingVecEnv
        from rl_training.source_scan.frozen_actor import load_frozen_source_actor,FrozenSourceActor
        from rl_training.source_scan.episode_metrics import PreResetMetricCapture
        from rl_training.source_scan.delivery_metrics import FirstEpisodes,paired_conditions
        from source_eight_env_moving_probe import serial_terminal
        from source_scale_motion_collection import actor_digest
        from rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3.source_training_env_cfg import DeeproboticsLite3SourceTrainingEnvCfg
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        names=['old'] if kind=='g4' else ['old','new']
        initial=None
        for label in names:
            phase('environment_start',label,120)
            cfg=DeeproboticsLite3SourceTrainingEnvCfg(source_training_run=1,source_parallel_sampler_enabled=True,
                source_parallel_sampler_num_envs=120,source_shared_bank_rows=1)
            cfg.sim.device='cuda:0';ranges=cfg.commands.base_velocity.ranges
            ranges.lin_vel_x=(.3,.3);ranges.lin_vel_y=(0.,0.);ranges.ang_vel_z=(0.,0.)
            if kind=='compare':cfg.observations.policy.enable_corruption=False
            raw_env=gym.make('SourceTerrain-Deeprobotics-Lite3-v0',cfg=cfg)
            wrapped=SourceTrainingVecEnv(RslRlVecEnvWrapper(raw_env,clip_actions=None));base=wrapped.unwrapped
            if label=='old':actor=load_frozen_source_actor(args.checkpoint,device='cuda:0')
            else:
                ref=plan['new_checkpoint'];path=Path(ref['path'])
                if file_digest(path)!=ref['sha256']:raise ValueError('new actor checkpoint changed')
                actor=FrozenSourceActor(torch.load(path,map_location='cpu',weights_only=True)).to('cuda:0')
            fingerprint=actor_digest(actor)
            assignments=[asdict(t) for t in base._source_parallel_terrain_allocator.assignments]
            import hashlib
            initial_state=hashlib.sha256(base.scene['robot'].data.root_link_state_w.cpu().numpy().tobytes()
                +base.scene['robot'].data.joint_pos.cpu().numpy().tobytes()
                +base.scene['robot'].data.joint_vel.cpu().numpy().tobytes()).hexdigest()
            identity=dict(assignments=assignments,terrain_origins=base.scene.terrain.terrain_origins.cpu().tolist(),
                          initial_state_sha256=initial_state)
            if initial is not None and initial!=identity:raise ValueError('paired environment initial geometry/state differs')
            initial=identity;episodes=FirstEpisodes([t['column'] for t in assignments])
            original_reset=base._reset_idx
            reset_context=dict(kind='natural',step=0)
            def reset(ids):
                torch.cuda.synchronize()
                tick=time.monotonic();value=original_reset(ids)
                torch.cuda.synchronize();elapsed=time.monotonic()-tick
                report['reset_latency_s'].append(elapsed)
                report['reset_samples'].append(dict(kind=reset_context['kind'],step=reset_context['step'],
                    ids=torch.as_tensor(ids).cpu().tolist(),elapsed_s=elapsed))
                return value
            base._reset_idx=reset
            observations=wrapped.get_observations();check_scans(base)
            generator=torch.Generator(device='cuda:0').manual_seed(132001)
            print('[source-delivery] initial-ready',flush=True);writer.persist(report)
            phase('actor_controls',label)
            count=288 if kind=='g4' else 2000
            metrics=PreResetMetricCapture(base)
            with torch.inference_mode(),metrics.installed() if kind=='compare' else _null_context():
                for step in range(1,count+1):
                    if not app.is_running():raise RuntimeError('simulator stopped during bounded diagnostics')
                    reset_context['step']=step
                    scheduled=next((r for r in plan.get('scheduled_resets',[]) if r['step']==step),None) if kind=='g4' else None
                    if scheduled is not None:
                        ready_tick=time.monotonic();reset_context['kind']='scheduled'
                        base._reset_idx(torch.tensor(scheduled['ids'],device=base.device,dtype=torch.long))
                        reset_context['kind']='natural'
                        base.scene.write_data_to_sim();base.sim.forward();base.scene.update(0.)
                        observations=wrapped.get_observations();check_scans(base)
                        report['scheduled_ready_samples'].append(dict(step=step,ids=scheduled['ids'],
                            elapsed_s=time.monotonic()-ready_tick))
                    mean=actor(observations)
                    action=mean+torch.randn(mean.shape,device='cuda:0',generator=generator) if kind=='g4' else mean
                    if kind=='compare':metrics.begin_control()
                    observations,reward,done,extras=wrapped.step(action)
                    report['simulation_control_steps']+=1;report['completed_diagnostic_transitions']+=120
                    if (any(not bool(torch.isfinite(x).all()) for x in observations.values())
                            or not torch.isfinite(reward).all() or not torch.isfinite(action).all()):
                        raise FloatingPointError('nonfinite collection observation/reward/action')
                    check_scans(base);report['finite_controls']+=1
                    if kind=='compare':episodes.add(metrics.collect_control(done),done.cpu().tolist(),serial_terminal(extras['source_terminal']))
                    writer.append_step(dict(step=report['simulation_control_steps'],actor=label,
                        actor_step=step,done=done.cpu().tolist(),finite=True,wall_mono=time.monotonic()))
                    if kind=='g4' and step==48:
                        report['measured_start_mono']=time.monotonic()
                        report['measured_reset_start']=len(report['reset_latency_s'])
                    if step%48==0:print(f'[source-delivery] {kind} actor={label} controls={step}/{count}',flush=True)
            if kind=='g4':
                report['measured_end_mono']=time.monotonic()
                reset_values=report['reset_latency_s'][report['measured_reset_start']:]
                elapsed=report['measured_end_mono']-report['measured_start_mono']
                report.update(measured_elapsed_s=elapsed,measured_transitions=28800,
                    measured_transitions_per_second=28800/elapsed,
                    measured_reset_count=len(reset_values),reset_p95_s=float(np.percentile(reset_values,95)) if reset_values else None)
            phase('actor_finalize',label,15)
            if actor_digest(actor)!=fingerprint:raise RuntimeError('read-only actor changed')
            report['actors'].append(dict(label=label,actor_sha256=fingerprint,initial=identity,
                first_episodes=episodes.result() if kind=='compare' else None))
            persist_actor_and_close(writer,report,wrapped,phase,kind,label)
            wrapped=base=None
            phase('environment_closed',label)
            if kind=='compare' and label=='old':
                phase('stage_replace',label,120)
                from isaaclab.sim import create_new_stage
                create_new_stage()
        if kind=='compare':
            report['paired_conditions']=paired_conditions(report['actors'][0]['first_episodes'],report['actors'][1]['first_episodes'])
            from rl_training.source_scan.delivery_metrics import write_paired_artifacts
            report['comparison_artifacts']=write_paired_artifacts(folder,
                report['actors'][0]['first_episodes'],report['actors'][1]['first_episodes'],report['paired_conditions'])
        report['status']='passed'
    except Exception:
        report.update(status='failed',error=traceback.format_exc());traceback.print_exc()
    finally:
        def close_remaining():
            phase('environment_close',limit=15)
            if wrapped is not None:
                if kind=='compare':close_comparison_environment(wrapped)
                else:wrapped.close()
        def close_application():
            phase('application_close',limit=15)
            app.close()
        finalize_and_close(writer,report,environment_close=close_remaining,
                           application_close=close_application if app is not None else None)
        phase('finished')
        faulthandler.disable();stack_log.close()
    return 0 if report['status']=='passed' else 1


def _null_context():
    from contextlib import nullcontext
    return nullcontext()


def validate(folder,plan,execution):
    result=dict(schema='lite3_delivery_light_acceptance_v1',accepted=False,reasons=[],training_ready=False)
    try:
        import math
        from rl_training.source_scan.delivery_stage import job_contract,LIMITS
        from rl_training.source_scan.delivery_metrics import paired_conditions
        kind=plan['delivery_job'];expected=job_contract(kind,plan)
        for key,value in expected.items():
            if type(plan.get(key)) is not type(value) or plan[key]!=value:raise ValueError('light contract mismatch: '+key)
        if (execution.get('returncode')!=0 or execution.get('process_group_empty') is not True
                or execution.get('timed_out') or execution.get('monitor_errors')):raise ValueError('light process/monitor failed')
        report=json.loads((folder/'result.json').read_text());cleanup=json.loads((folder/'result.cleanup.json').read_text())
        if report['status']!='passed' or report.get('synthetic_cpu_fixture'):
            raise ValueError('light probe failed or is synthetic: '+report.get('error','missing runtime evidence'))
        for key in ('input_sha256','dependency_sha256','base_revision','attempt_id'):
            if report[key]!=plan[key]:raise ValueError('light identity differs: '+key)
        count=LIMITS[kind]['controls']
        if (report['simulation_control_steps']!=count or report['finite_controls']!=count
                or report['completed_diagnostic_transitions']!=120*count
                or report['steps_written']!=count or file_digest(folder/'result.steps.jsonl')!=report['steps_sha256']
                or cleanup.get('environment')!='returned' or cleanup.get('application') not in ('closing','returned')
                or cleanup.get('persistence_errors')):raise ValueError('light counters/hash/cleanup differ')
        rows=[json.loads(x) for x in (folder/'result.steps.jsonl').read_text().splitlines()]
        if [r['step'] for r in rows]!=list(range(1,count+1)) or not all(r['finite'] is True for r in rows):
            raise ValueError('light step sequence/finite checks differ')
        if kind=='g4':
            elapsed=report['measured_end_mono']-report['measured_start_mono'];rate=28800/elapsed
            import numpy as np
            resets=report['reset_latency_s'][report['measured_reset_start']:]
            if not resets:raise ValueError('G4 has no measured reset latency samples')
            if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in resets)
                    or report['measured_reset_count']!=len(resets) or report['measured_transitions']!=28800
                    or not rows[47]['wall_mono']<=report['measured_start_mono']<=rows[48]['wall_mono']
                    or report['measured_end_mono']<rows[-1]['wall_mono']):
                raise ValueError('G4 reset samples or measurement interval differs')
            p95=float(np.percentile(resets,95))
            if plan.get('stage_revision'):
                samples=report['reset_samples']
                scheduled=[dict(step=r['step'],ids=r['ids']) for r in samples if r['kind']=='scheduled']
                ready=[dict(step=r['step'],ids=r['ids']) for r in report['scheduled_ready_samples']]
                if (scheduled!=plan['scheduled_resets'] or ready!=plan['scheduled_resets']
                        or [r['elapsed_s'] for r in samples]!=report['reset_latency_s']
                        or any(r['kind'] not in ('natural','scheduled') for r in samples)
                        or any(not math.isfinite(r['elapsed_s']) or r['elapsed_s']<0 for r in report['scheduled_ready_samples'])):
                    raise ValueError('G4 planned/natural reset evidence differs')
                result.update(scheduled_reset_batches=len(scheduled),scheduled_environment_resets=120,
                    measured_natural_reset_batches=sum(r['kind']=='natural' for r in samples[report['measured_reset_start']:]))
            if (not math.isfinite(rate) or rate<600 or p95>1. or p95!=report['reset_p95_s']
                    or rate!=report['measured_transitions_per_second']):raise ValueError('G4 collection throughput/reset gate failed')
            result.update(measured_collection_transitions_per_second=rate,reset_p95_s=p95,measured_reset_count=len(resets))
        else:
            old,new=report['actors']
            if old['initial']!=new['initial'] or report['paired_conditions']!=paired_conditions(old['first_episodes'],new['first_episodes']):
                raise ValueError('paired diagnostics do not reconstruct')
            artifacts=report.get('comparison_artifacts',{})
            csv_row=artifacts.get('csv',{});png_row=artifacts.get('png',{})
            csv_path=folder/csv_row.get('path','');png_path=folder/png_row.get('path','')
            import csv
            if (csv_row.get('rows')!=120 or not csv_path.is_file() or not png_path.is_file()
                    or file_digest(csv_path)!=csv_row.get('sha256') or file_digest(png_path)!=png_row.get('sha256')
                    or not png_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')):
                raise ValueError('paired comparison CSV/PNG artifacts are missing or changed')
            with csv_path.open(newline='') as stream:
                if sum(1 for _ in csv.DictReader(stream))!=120:raise ValueError('paired comparison CSV row count differs')
            result.update(paired_conditions=15,episode_pairs=120,formal_selection_qualified=False)
        result.update(accepted=True,diagnostic_transitions=120*count,ppo_updates=0)
    except Exception as error:result['reasons'].append(type(error).__name__+': '+str(error))
    return result


def supervise(args):
    import shutil
    from source_live_ppo_supervisor import group_exists,terminate_group
    from source_training_entrypoint import _input_hashes
    from source_scale_startup_supervisor import dependencies
    from rl_training.source_scan.training_budget_ledger import BudgetLedger,canonical_budget_path
    plan=json.loads(args.manifest.read_text());preflight(args,plan,kind=plan['delivery_job'])
    out=args.output_dir.resolve()
    if shutil.disk_usage(out.parent).free<32*1024**3:raise RuntimeError('delivery disk gate failed')
    query=['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader']
    gpu=subprocess.check_output(query,text=True,timeout=10)
    if int(next(r for r in gpu.splitlines() if r.startswith('0,')).split(',')[3].split()[0])<12288:
        raise RuntimeError('delivery GPU0 memory gate failed')
    out.mkdir(exist_ok=False);atomic_json(out/'run_manifest.json',plan)
    atomic_json(out/'supervision.json',dict(revision=plan['base_revision'],manifest_sha256=file_digest(out/'run_manifest.json')))
    (out/'gpu_before.csv').write_text(gpu)
    cmd=['timeout','--signal=TERM','--kill-after=15s',str(plan['wall_timeout_s'])+'s',sys.executable,str(Path(__file__).resolve()),
         'probe','--manifest',str(out/'run_manifest.json'),'--checkpoint',str(args.checkpoint.resolve()),'--output-dir',str(out)]
    execution=dict(command=cmd,timed_out=False,monitor_errors=[]);process=None;start=time.monotonic()
    try:
        with (out/'runtime.log').open('x') as log,(out/'gpu_samples.jsonl').open('x') as samples:
            process=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            execution['process_group']=process.pid;atomic_json(out/'execution.json',execution);next_sample=0
            while process.poll() is None:
                elapsed=time.monotonic()-start;text=(out/'runtime.log').read_text(errors='replace')
                deadline=lifecycle_deadline(out,time.monotonic())
                if deadline:
                    execution['timed_out']=True;raise TimeoutError(deadline)
                if elapsed>plan['wall_timeout_s'] or elapsed>120 and '[source-delivery] initial-ready' not in text:
                    execution['timed_out']=True;raise TimeoutError('delivery startup/wall deadline reached')
                if (out/'runtime.log').stat().st_size>512*1024**2:raise RuntimeError('delivery log ceiling reached')
                if elapsed>=next_sample:
                    samples.write(json.dumps(dict(elapsed_s=elapsed,gpu=subprocess.check_output(query,text=True,timeout=10)))+'\n')
                    samples.flush();next_sample=elapsed+5
                time.sleep(.2)
    except BaseException as error:execution['monitor_errors'].append(type(error).__name__+': '+str(error))
    finally:
        if process is not None:
            if group_exists(process.pid):terminate_group(process,15)
            execution.update(returncode=process.poll(),process_group_empty=not group_exists(process.pid))
        execution['elapsed_s']=time.monotonic()-start
        try:
            (out/'gpu_after.csv').write_text(subprocess.check_output(query,text=True,timeout=10))
            if (_input_hashes()!=plan['input_sha256'] or dependencies()!=plan['dependency_sha256']
                    or BudgetLedger(canonical_budget_path(ROOT)).reference()!=plan['budget_ledger']
                    or file_digest(args.checkpoint)!=plan['checkpoint_sha256']):raise ValueError('light immutable inputs changed')
        except Exception as error:execution['monitor_errors'].append(str(error))
        atomic_json(out/'execution.json',execution)
    acceptance=validate(out,plan,execution);atomic_json(out/'acceptance.json',acceptance)
    print(json.dumps(acceptance,indent=2));return 0 if acceptance['accepted'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('supervise','probe'))
    for name in ('manifest','checkpoint','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=120);args=parser.parse_args()
    raise SystemExit(probe(args) if args.action=='probe' else supervise(args))
