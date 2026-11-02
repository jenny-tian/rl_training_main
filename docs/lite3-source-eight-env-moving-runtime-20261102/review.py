"""CPU-only independent replay and descriptive metrics from this archived run."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/rl_training'))
sys.path.insert(0,str(ROOT/'scripts/tools'))
from source_eight_env_moving_validation import validate,open_events


def flattened(value):
    if isinstance(value,list):
        for item in value:yield from flattened(item)
    else:yield value


def review(folder):
    raw=folder/'raw'
    report=json.loads((raw/'result.json').read_text())
    manifest=json.loads((raw/'run_manifest.json').read_text())
    execution=json.loads((raw/'execution.json').read_text())
    parent=json.loads((raw/'acceptance.json').read_text())
    cleanup=json.loads((folder/'cleanup_check.json').read_text())
    result=validate(raw/'result.json',returncode=execution['returncode'],attempt_id=manifest['attempt_id'],
        revision=manifest['base_revision'],checkpoint_sha256=manifest['checkpoint_sha256'],
        process_group_empty=execution['process_group_empty'])
    # JSON object keys are strings on disk; the replay's Counter uses integer
    # environment IDs. Compare the same serialized representation.
    assert result['accepted'] and json.loads(json.dumps(result))==parent
    assert manifest['user_approved'] and manifest['retry'] is False and not execution['monitor_errors']
    assert not cleanup['remaining_group_processes'] and not cleanup['group_leader_exists']
    controls=[json.loads(line) for line in (raw/'result.steps.jsonl').read_text().splitlines()]
    clipped=projected=paused=0;endings=[]
    for row in controls:
        for i in range(8):
            clipped+=any(abs(a-b)>1e-6 for a,b in zip(row['raw_action'][i],row['base_action'][i]))
            projected+=row['projected'][i][0];paused+=row['paused'][i][0]
        if row['terminal'] is not None:
            term=row['terminal'];assert term['ids']==row['forced_ids']
            assert term['terminated']==[False]*8
            assert all(not any(values) for name,values in term['terms'].items() if name!='source_training_timeout')
            endings.append({'control_step':row['step'],'ids':term['ids'],'terminal_steps':term['steps'],
                            'post_reset_counters':row['episode_counters']})
    counts=Counter();max_displacement=max_acceleration=max_link_speed=0.;minimum_margin=float('inf')
    minimum_base_clearance=float('inf');max_root_progress=0.;recoveries=[]
    for line in open_events(raw/'result.events.jsonl.gz'):
        event=json.loads(line);phase=event.get('phase');counts[phase]+=1
        if phase!='motion_substep':continue
        positions=event['body_positions_w']['values'];velocities=event['body_link_velocities_w']['values']
        delta=event['displacements_m']['values'];accel=event['finite_difference_acceleration_mps2']['values']
        root=event['root_positions_w']['values'];origins=event['origins_w']['values']
        for i in range(8):
            max_root_progress=max(max_root_progress,root[i][0]-origins[i][0])
            hits=event['scanners']['height_scanner_base']['hits']['values'][i]
            minimum_base_clearance=min(minimum_base_clearance,root[i][2]-max(point[2] for point in hits))
            for b in range(17):
                max_displacement=max(max_displacement,math.sqrt(sum(x*x for x in delta[i][b])))
                max_acceleration=max(max_acceleration,math.sqrt(sum(x*x for x in accel[i][b])))
                max_link_speed=max(max_link_speed,math.sqrt(sum(x*x for x in velocities[i][b])))
                minimum_margin=min(minimum_margin,*[min(positions[i][b][axis]-event['safe_lower_xy_w'][i][axis],
                    event['safe_upper_xy_w'][i][axis]-positions[i][b][axis]) for axis in (0,1)])
        if event['recovery_ids']:recoveries.append({'sim_step':event['sim_step'],'ids':event['recovery_ids']})
    assert counts['motion_substep']==counts['tensor_contact_sample']==384
    assert counts['reset_boundary']==counts['reset_checked']==3 and len(recoveries)==3
    assert minimum_margin>=0 and result['ordinary_failure_episodes']==0
    samples=[json.loads(line) for line in (raw/'gpu_samples.jsonl').read_text().splitlines()]
    gpu0=[[v.strip() for v in next(line for line in s['gpu'].splitlines() if line.startswith('0,')).split(',')]
          for s in samples]
    return {'schema':'lite3_eight_env_runtime_offline_review_v1','status':'passed_within_registered_short_probe',
        'acceptance_reproduced':result,'run_revision':manifest['base_revision'],
        'control_steps':96,'diagnostic_transitions':768,'physics_substeps':384,'ppo_updates':0,'retry_count':0,
        'raw_clip_environment_steps':clipped,'raw_clip_denominator':768,'raw_clip_fraction':clipped/768,
        'additional_filter_projection_environment_steps':projected,'safety_pause_environment_steps':paused,
        'timeout_batches':endings,'recovery_first_substeps':recoveries,'final_episode_counters':controls[-1]['episode_counters'],
        'exposure_nonzero':{name:value for name,value in report['exposure'].items() if any(value.values())},
        'event_phase_counts':dict(counts),'measured_sample_extrema':{
            'maximum_link_displacement_per_5ms_m':max_displacement,
            'maximum_finite_difference_link_acceleration_mps2':max_acceleration,
            'maximum_link_speed_mps':max_link_speed,'minimum_instantaneous_envelope_margin_m':minimum_margin,
            'minimum_base_clearance_m':minimum_base_clearance,'maximum_reset_relative_root_progress_m':max_root_progress},
        'extrema_limit':'Discrete sampled descriptive maxima/minima, not continuous-time bounds or terrain skill qualification.',
        'gpu0_sampled_total_peak_used_mib':max(int(row[3].split()[0]) for row in gpu0),
        'gpu0_sampled_min_free_mib':min(int(row[4].split()[0]) for row in gpu0),
        'resource_limit':'5-second whole-card samples with instrumentation and I/O, not training throughput.',
        'supervisor_elapsed_s':execution['elapsed_s'],'g3_g4_qualified':False,'training_ready':False}


if __name__=='__main__':
    folder=Path(__file__).resolve().parent
    result=review(folder)
    with (folder/'review.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
