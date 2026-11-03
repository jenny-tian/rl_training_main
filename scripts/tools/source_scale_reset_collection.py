"""Bounded G2 collection shared by the simulator probe and synthetic CPU tests."""
from dataclasses import asdict
import time

import torch

from rl_training.source_scan.scale_reset import contract_for
from rl_training.source_scan.moving_diagnostic import tensor_record
from rl_training.source_scan.episode_metrics import native_metric_rows
from source_parallel_reset_lifecycle import summarize_tickets
from source_shared_contact_report import capture_substeps


def collect(env,wrapped,observer,capture,journal,writer,report,interface,contact_runtime):
    contract=contract_for(env.num_envs)
    env.observation_manager.compute(update_history=False);initial_scans=observer.scans()
    memory=observer.memory()
    report.update(initial_assignments=[asdict(t) for t in observer.tickets],start_sim_step=env._sim_step_counter,
        terrain_origins=tensor_record(env.scene.terrain.terrain_origins),body_names=list(env.scene['robot'].body_names),
        default_root_state=tensor_record(env.scene['robot'].data.default_root_state),
        initial_root_positions_w=tensor_record(env.scene['robot'].data.root_link_pos_w),
        reset_snapshot_inventory={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in memory.items()},
        observation_history_terms=sorted(key for key in memory if key.startswith('history/')))
    journal.record(dict(phase='scale_initial',scanners=initial_scans))
    latencies=[]
    with torch.inference_mode(),capture_substeps(env,interface,capture,contact_runtime),observer.installed():
        for step,ids in enumerate(contract['reset_batches'],1):
            observer.direct_resets([ids])
            journal.record(dict(phase='g2_reset_scans',batch=step,ids=ids,sim_step=env._sim_step_counter,
                roots=tensor_record(env.scene['robot'].data.root_link_pos_w),scanners=observer.scans()))
            raw=torch.zeros((env.num_envs,12),device=env.device);started=time.monotonic()
            _,_,done,extras=wrapped.step(raw)
            latencies.append(time.monotonic()-started);capture.check_control(env,step)
            if bool(done.any()) or wrapped.runtime.terminal is not None:raise RuntimeError('unexpected G2 episode ending')
            journal.record(dict(phase='scale_control_scans',step=step,sim_step=env._sim_step_counter,scanners=observer.scans()))
            journal.flush()
            writer.append_step(dict(step=step,done=done.cpu().tolist(),time_outs=extras['time_outs'].cpu().tolist(),
                raw_action=raw.cpu().tolist(),executed_action=wrapped.last_trace.executed_action.cpu().tolist(),
                projected=wrapped.last_trace.decision.projected.cpu().tolist(),paused=wrapped.last_trace.decision.paused.cpu().tolist(),
                metric_rows=native_metric_rows(env)))
    report.update(status='passed',simulation_control_steps=4,contact_capture=capture.summary(),
        motion_samples=observer.samples,reset_batches=observer.resets,pending_recovery_ids=sorted(observer.pending_recovery),
        all_assignment_tickets=[asdict(t) for t in observer.tickets],ticket_summary=summarize_tickets(observer.tickets),
        reset_latency_s=list(observer.reset_latency_s),recovery_control_latency_s=latencies,first_step_s=latencies[0])
