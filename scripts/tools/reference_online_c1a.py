"""Prepare or run the restored C1-a stage. Runtime budget is proposed, not approved."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from reference_online_targets import HASHES, MODEL_PATHS, read
from reference_stairs_eval import ROOT, save, sha

OUTPUT = ROOT / 'logs/reference_six/online_c1a_20261117'
DOCS = ROOT / 'docs/lite3-online-c1a-20261117'
MARKS = (0, 5000, 10000, 20000, 40000, 60000)
TERRAINS = dict(flat=0, rough=4, slope_up=8, slope_down=10, stairs_up=12, stairs_down=14,
                friction_lateral=16, friction_longitudinal=17, slope_rough_up=18, slope_rough_down=19)


def protocol():
    return dict(schema=1, algorithm='C1-a', scope='one-target mechanism study; not v2 multi-target qualification',
        model='B10200', checkpoint_sha256=HASHES['B'], pairs=60, sessions_per_arm=60,
        train_environments=120, train_control_steps=60000, primary_mark=20000, marks=list(MARKS),
        max_episodes=400, batch_size=4, parameters=2, sigma=.6, learning_rate=.3,
        adam_betas=[.9,.999], adam_eps=1.e-8, mean_bound=2.5, gradient_clipping=False,
        reward='clip(final_forward_displacement_m,0,9)/9 + success - posture_failure',
        return_discount=None, baseline='leave-one-out in four episodes from same session and parameter version',
        reset_seed=20261117, noise_seed=2026111702,
        namespaces=dict(smoke=30,train=31,target=32,retention=33),
        dt=.02, physics_dt=.005, preparation_steps=50, policy_steps=1000,
        basis='sag', residual_limits=[.24,.20,.20]*4, command=[.4,0.,0.], ea=False,
        smoke=dict(runs=2,environments=120,control_steps=600,updates=0),
        target_starts=4, target_marks=list(MARKS), target_parts=4,
        retention_starts_per_subtype=2, retention_subtypes=list(TERRAINS), retention_marks=[20000], retention_parts=20,
        retention_rows=[1,8], target_terrain='stairs_06', target_terrain_seed=20261116,
        retention_terrain_seed=42, gpu=1, minimum_free_mib=10240,
        budgets=dict(smoke=144000,train=7200000,target=3024000,retention=2520000,total=12888000),
        snapshot_rule='exact budget tick; no completion or batch forced across boundary',
        success_gain_min=.10, ci_lower_min=0., safety_increase_max=.02, family_loss_max=.05,
        flat_xy_velocity_rmse_ratio_max=1.10, bootstrap_samples=10000, bootstrap_seed=0,
        paired_reset='wait for both arms; count preparation, policy and waiting; no automatic retries',
        simulator_privilege='existing terrain-relative safety; geometric success enters original C1-a reward; not deployment-qualified')


def rows(kind, *, part=0, smoke_index=1):
    if kind not in ('smoke','train','target','retention'):
        raise ValueError('unknown run kind')
    parts = 4 if kind == 'target' else 20 if kind == 'retention' else 1
    if not 0 <= part < parts or smoke_index not in (1,2):
        raise ValueError('invalid partition or smoke index')
    width = 60 // parts
    result = []
    cases = list(TERRAINS) if kind == 'retention' else ['stairs_06']
    for pair in range(part*width,(part+1)*width):
        for case_index, case in enumerate(cases):
            for repeat in range(4 if kind == 'target' else 2 if kind == 'retention' else 1):
                slot = (pair if kind == 'train' else (smoke_index-1)*60+pair if kind == 'smoke'
                        else pair*1000+case_index*100+repeat)
                for arm in ('Frozen','Updated'):
                    result.append(dict(env_id=len(result),session=pair,arm=arm,condition=case,slot=slot,
                        repeat=repeat,column=TERRAINS.get(case,0),row=(1,8)[repeat] if kind=='retention' else 0))
    return result


def jobs():
    result = [dict(name=f'smoke_{i}',kind='smoke',smoke_index=i,part=0,mark=0,steps=600) for i in (1,2)]
    result.append(dict(name='train',kind='train',smoke_index=1,part=0,mark=0,steps=60000))
    for kind, marks, parts in (('target',MARKS,4),('retention',(20000,),20)):
        for mark in marks:
            for part in range(parts):
                result.append(dict(name=f'{kind}_{mark}_{part:02d}',kind=kind,mark=mark,part=part,smoke_index=1,steps=1050))
    for job in result:
        job['environments'] = len(rows(job['kind'],part=job['part'],smoke_index=job['smoke_index']))
        job['reserved_steps'] = job['steps'] * job['environments']
    return result


def source_manifest():
    # Verify old experiment locks read-only; never rewrite their evidence.
    from reference_online_c1b_reset_api import registration as verify_previous
    verify_previous()
    paths = [ROOT/'source/rl_training/rl_training/ea_lpl/c1a.py', DOCS/'PLAN.md',
        *[Path(__file__).with_name(name) for name in ('reference_online_c1a.py','reference_online_c1a_collect.py',
          'reference_online_c1a_env.py','reference_online_c1a_audit.py','reference_online_c1a_results.py',
          'source_block_runtime.py','reference_online_c1b_env_repaired.py','reference_online_c1b_env_reset_api.py')],
        ROOT/'source/rl_training/rl_training/ea_lpl/current_safety.py',
        ROOT/'source/rl_training/rl_training/ea_lpl/residual_bridge.py',
        ROOT/'source/rl_training/rl_training/ea_lpl/basis.py']
    if sha(MODEL_PATHS['B']) != HASHES['B']:
        raise ValueError('base checkpoint changed')
    return dict(protocol=protocol(),jobs=jobs(),base_path=str(MODEL_PATHS['B']),
        previous_lock_sha256=sha(ROOT/'docs/lite3-online-c1b-reset-api-20261117/LOCK.json'),
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths])


def completed(name):
    directory = OUTPUT/name
    execution, report = read(directory/'execution.json'), read(directory/'report.json')
    if execution['status'] != 'completed' or execution['exit_code'] != 0 or report['status'] != 'passed':
        raise RuntimeError(f'incomplete or failed predecessor: {name}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true',help='CPU only; no reservation or simulator')
    parser.add_argument('--job',choices=[j['name'] for j in jobs()])
    parser.add_argument('--child',action='store_true')
    args = parser.parse_args()
    if args.prepare:
        value = source_manifest()
        value.update(status='prepared_pending_runtime_budget_approval',simulation_steps_added=0,online_updates_added=0)
        save(DOCS/'PREPARATION.json',value)
        print('Prepared CPU manifest; no simulator started; proposed total steps:',protocol()['budgets']['total'])
        return 0
    if args.job is None:
        parser.error('--prepare or --job required')
    job = next(j for j in jobs() if j['name']==args.job)
    for key,value in job.items(): setattr(args,key,value)
    args.output = OUTPUT/args.job
    if not args.child:
        manifest = source_manifest()
        if (OUTPUT/'lock.json').exists():
            if read(OUTPUT/'lock.json') != manifest: raise RuntimeError('registered sources changed')
        else:
            save(OUTPUT/'lock.json',manifest); save(DOCS/'LOCK.json',manifest)
            save(OUTPUT/'attempts.json',[])
        attempts = read(OUTPUT/'attempts.json')
        order = [j['name'] for j in jobs()]
        if [a['name'] for a in attempts] != order[:len(attempts)] or args.job != order[len(attempts)]:
            raise RuntimeError('run the registered order once; no retry')
        for attempt in attempts:
            completed(attempt['name'])
            audit = read(DOCS/f'AUDIT_{attempt["name"]}.json')
            if audit['status'] != 'passed': raise RuntimeError('predecessor audit failed')
        attempts.append(job)
        if sum(j['reserved_steps'] for j in attempts)>protocol()['budgets']['total']:
            raise RuntimeError('budget exceeded')
        save(OUTPUT/'attempts.json',attempts)
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),'--job',args.job,'--child'],gpu=1,
                   output=args.output,timeout=14400 if args.kind=='train' else 3600,minimum_free_mib=10240)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True,device='cuda:0',kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        from reference_online_c1a_collect import collect
        collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception:
        error = traceback.format_exc(); traceback.print_exc()
        save(args.output/'error.json',dict(error=error))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__': sys.exit(main())
