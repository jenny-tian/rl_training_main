"""C1-a，航向闭环打开，目标由环境变量 C1A2_TARGET 指定。方案见 docs/lite3-online-c1a-heading-20261117/PLAN.md。

复用 reference_online_c1a 的会话学习器、配对重置和采集循环；不修改那些文件。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from reference_stairs_eval import ROOT, save, sha
from reference_online_targets import HASHES

TARGETS = ('stairs_up_06', 'boxes_06')
TARGET = os.environ.get('C1A2_TARGET', TARGETS[0])
assert TARGET in TARGETS
OUTPUT = ROOT / 'logs/reference_six/online_c1a_heading_20261117' / TARGET
MARKS = (0, 20000, 40000, 60000)
PRIMARY = 40000
NOISE_SEED = 2026111703
TERRAINS = dict(flat=0, rough=4, slope_up=8, slope_down=10, stairs_up=12, stairs_down=14,
                friction_lateral=16, friction_longitudinal=17, slope_rough_up=18, slope_rough_down=19)


def protocol():
    return dict(schema=1, algorithm='C1-a', heading_feedback=dict(gain=.5, limit=.8, target='world +x'),
        target=TARGET, model='B10200', checkpoint_sha256=HASHES['B'], pairs=60, train_control_steps=60000,
        primary_mark=PRIMARY, marks=list(MARKS), max_episodes=400, batch_size=4, sigma=.6, learning_rate=.3,
        mean_bound=2.5, basis='sag', residual_limits=[.24, .20, .20] * 4, command=[.4, 0., 'feedback'],
        reward='clip(final_forward_displacement/route_length,0,1) + success - posture_failure',
        reset_seed=20261117, noise_seed=NOISE_SEED, namespaces=dict(smoke=50, train=51, target=52, retention=53),
        target_starts=4, retention_starts_per_subtype=2, retention_rows=[1, 8],
        success_gain_min=.10, ci_lower_min=0., safety_increase_max=.02, family_loss_max=.05,
        bootstrap_samples=10000, bootstrap_seed=0)


def episode_return(displacement, length, completed, safety_ending):
    assert np.isfinite(displacement) and length > 0 and not (completed and safety_ending)
    return float(np.clip(displacement / length, 0., 1.) + bool(completed) - bool(safety_ending))


def rows(kind, *, part=0, smoke_index=1):
    parts = 4 if kind == 'target' else 20 if kind == 'retention' else 1
    assert kind in ('smoke', 'train', 'target', 'retention') and 0 <= part < parts
    width = 60 // parts
    result = []
    cases = list(TERRAINS) if kind == 'retention' else [TARGET]
    for pair in range(part * width, (part + 1) * width):
        for case_index, case in enumerate(cases):
            for repeat in range(4 if kind == 'target' else 2 if kind == 'retention' else 1):
                slot = pair if kind in ('train', 'smoke') else pair * 1000 + case_index * 100 + repeat
                for arm in ('Frozen', 'Updated'):
                    result.append(dict(env_id=len(result), session=pair, arm=arm, condition=case, slot=slot,
                        repeat=repeat, column=TERRAINS.get(case, 0), row=(1, 8)[repeat] if kind == 'retention' else 0))
    return result


def jobs():
    result = [dict(name='smoke', kind='smoke', part=0, mark=0, steps=600),
              dict(name='train', kind='train', part=0, mark=0, steps=60000)]
    result += [dict(name=f'target_{m}_{p:02d}', kind='target', mark=m, part=p, steps=1050) for m in MARKS[1:] for p in range(4)]
    result += [dict(name=f'retention_{PRIMARY}_{p:02d}', kind='retention', mark=PRIMARY, part=p, steps=1050) for p in range(20)]
    for job in result: job['smoke_index'] = 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=[j['name'] for j in jobs()], required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    for key, value in next(j for j in jobs() if j['name'] == args.job).items(): setattr(args, key, value)
    args.output = OUTPUT / args.job
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'], gpu=args.gpu,
                   output=args.output, timeout=18000 if args.kind == 'train' else 3600, minimum_free_mib=8192)
    try:
        from isaaclab.app import AppLauncher
        AppLauncher(headless=True, device='cuda:0',
                    kit_args=f'--/renderer/multiGpu/enabled=False --/renderer/activeGpu={args.gpu}').app
        from reference_online_c1a2_collect import collect
        collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc(); save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__': sys.exit(main())
