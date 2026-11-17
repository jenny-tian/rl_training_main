"""Approved bounded C1-b pilot: paired frozen/updated sessions, then evaluation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from reference_stairs_eval import ROOT, save, sha
from reference_online_targets import HASHES, MODEL_PATHS, read
from reference_online_current import registration as prior_registration

OUTPUT = ROOT / 'logs/reference_six/online_c1b_20261117'
DOCS = ROOT / 'docs/lite3-online-c1b-20261117'
PROPOSAL = ROOT / 'docs/lite3-online-targets-current-20261117/NEXT_STAGE_PROPOSAL.md'
TERRAINS = dict(flat=0, rough=4, slope_up=8, slope_down=10, stairs_up=12, stairs_down=14,
                friction_lateral=16, friction_longitudinal=17, slope_rough_up=18, slope_rough_down=19)


def protocol():
    return dict(schema=1, approval='User 同意 after state-conditioned pilot proposal, 2026-11-17',
        model='B', checkpoint_sha256=HASHES['B'], reset_seed=20261117, noise_seed=2026111701,
        namespaces=dict(smoke=20, train=21, target=22, retention=23), sessions=8,
        train_control_steps=20000, max_episodes=64, policy_steps=1000, preparation_steps=50,
        macro_steps=10, history_steps=25, sigma=.6, gamma=.99, learning_rate=.01,
        gradient_norm_max=1., ema=.9, parameters=18, basis='sag', residual_limits=[.24, .20, .20]*4,
        reward='clip(dx,-.02,.02)-.5*abs(clip(dy,-.02,.02))-.001*mean((learned_effect/limits)^2)-posture-.5*lateral',
        mean_scale=[.5,.5,.2,.8,10.,1.,1.,1.], feature_clip=[-3.,3.], loss_divisor=100,
        budgets=dict(smoke=19200, train=320000, target=268800, retention=336000, total=944000),
        gpu=1, minimum_free_mib=10240, command=[.4,0.,0.], dt=.02, physics_dt=.005,
        terrain_seed_target=20261116, terrain_seed_retention=42, retention_rows=[1,8],
        paired_reset='wait for both arms to finish; count every wait/preparation step; reset both together',
        native_auto_reset='deferred without altering termination flags; explicit paired manager reset',
        smoke=dict(runs=2, environments=16, control_steps=600, optimizer_updates=0,
            run1_policy_cap=[150,150], run2_policy_cap=[100,150], reason='exercise multiple and asynchronous endings'),
        target_evaluation_starts=16, retention_starts=2, evaluation_exploration=False,
        base_optimizer_updates=0, ea=False, no_automatic_retry=True,
        simulator_privilege='terrain-relative safety rays and contact forces; dx/dy state-estimate proxy only in score')


def rows(kind, smoke_index=1):
    result = []
    for session in range(8):
        cases = list(TERRAINS) if kind == 'retention' else ['stairs_06']
        for case_index, condition in enumerate(cases):
            repeats = 16 if kind == 'target' else 2 if kind == 'retention' else 1
            for repeat in range(repeats):
                slot = (session if kind == 'train' else (smoke_index-1)*8+session if kind == 'smoke'
                        else session*1000 + case_index*100 + repeat)
                for arm in ('Frozen', 'Updated'):
                    result.append(dict(env_id=len(result), session=session, arm=arm, condition=condition,
                        slot=slot, repeat=repeat, column=TERRAINS.get(condition,0),
                        row=(1,8)[repeat] if kind == 'retention' else 0))
    return result


def registration():
    prior_registration()
    paths = [Path(__file__), Path(__file__).with_name('reference_online_c1b_env.py'),
             Path(__file__).with_name('reference_online_c1b_collect.py'),
             ROOT / 'source/rl_training/rl_training/ea_lpl/c1b.py',
             *[Path(__file__).with_name(name) for name in ('reference_low_stairs_cfg.py',
                 'reference_six_terrain_cfg.py','reference_lite3_loader.py','reference_stair_allocation_retention.py')]]
    value = dict(protocol=protocol(), proposal_sha256=sha(PROPOSAL),
        prior_lock_sha256=sha(ROOT / 'docs/lite3-online-targets-current-20261117/LOCK.json'),
        sources=[dict(path=str(p), sha256=sha(p)) for p in paths])
    if (OUTPUT / 'lock.json').exists():
        assert value == read(OUTPUT / 'lock.json') == read(DOCS / 'LOCK.json')
    else:
        save(OUTPUT / 'lock.json', value); save(DOCS / 'LOCK.json', value)
        save(OUTPUT / 'attempts.json', [])
    assert sha(MODEL_PATHS['B']) == HASHES['B']
    return value


def completed(directory):
    execution, report = read(directory / 'execution.json'), read(directory / 'report.json')
    assert execution['status'] == 'completed' and execution['exit_code'] == 0
    assert report['status'] == 'passed' and report['base_unchanged']
    assert report['base_optimizer_updates'] == 0 and report['frozen_unchanged']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--kind', choices=('smoke','train','target','retention'))
    parser.add_argument('--smoke-index', type=int, choices=(1,2), default=1)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.register_only:
        registration(); return 0
    if args.kind is None:
        parser.error('kind required')
    name = f'smoke_{args.smoke_index}' if args.kind == 'smoke' else args.kind
    args.output = OUTPUT / name
    if not args.child:
        registration()
        attempts = read(OUTPUT / 'attempts.json')
        if any(a['name'] == name for a in attempts):
            raise RuntimeError('attempt already reserved; no retry')
        for attempt in attempts:
            completed(OUTPUT / attempt['name'])
        if name == 'smoke_2': completed(OUTPUT / 'smoke_1')
        if args.kind != 'smoke':
            completed(OUTPUT / 'smoke_1'); completed(OUTPUT / 'smoke_2')
        if args.kind in ('target','retention'): completed(OUTPUT / 'train')
        n = len(rows(args.kind, args.smoke_index))
        steps = 600 if args.kind == 'smoke' else 20000 if args.kind == 'train' else 1050
        reserve = n*steps
        assert sum(a['reserved_steps'] for a in attempts if a['kind'] == args.kind)+reserve <= protocol()['budgets'][args.kind]
        attempts.append(dict(name=name, kind=args.kind, reserved_steps=reserve, environments=n))
        save(OUTPUT / 'attempts.json', attempts)
        from source_block_runtime import run
        return run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],
                   gpu=1, output=args.output, timeout=3600, minimum_free_mib=10240)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device='cuda:0',
            kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        from reference_online_c1b_collect import collect
        collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception:
        original = traceback.format_exc(); traceback.print_exc()
        try:
            from reference_online_failure_capture import capture_failure
            capture_failure(sys.exception(), args.output)
        except Exception:
            save(args.output / 'failure_capture_error.json',dict(error=traceback.format_exc()))
        save(args.output / 'error.json', dict(error=original))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
