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
# C1A2_VARIANT=ea 时改用执行感知的探索协方差（C1-a-EA），其余设置完全相同。方案见 docs/lite3-online-c1a-ea-20261117/PLAN.md。
# C1A2_VARIANT=m4 时每个会话用 4 个并行环境、普通梯度步、探索标准差 1.0。方案见 docs/lite3-online-c1a-m4-20261118/PLAN.md。
# C1A2_VARIANT=bh 时码由高度特征决定（C1-b-H），每 10 步重新计算，探索加在 8 个参数上。方案见 docs/lite3-online-c1b-height-20261118/PLAN.md。
# C1A2_VARIANT=bs 时学固定偏置和摆动缩腿共 4 个幅度（C1-b-S）；bsf 是放开每步变化上限的对照组，只做上阶。方案见 docs/lite3-online-c1b-swing-20261118/PLAN.md。
VARIANT = os.environ.get('C1A2_VARIANT', 'fixed')
assert VARIANT in ('fixed', 'ea', 'm4', 'bh', 'bs', 'bsf')
assert VARIANT != 'bsf' or TARGET == 'stairs_up_06'
SWING_BASIS = VARIANT in ('bs', 'bsf')
REPLICAS = 4 if VARIANT in ('m4', 'bh', 'bs', 'bsf') else 1
OUTPUT = ROOT / 'logs/reference_six' / dict(fixed='online_c1a_heading_20261117', ea='online_c1a_ea_20261117', m4='online_c1a_m4_20261118',
                                            bh='online_c1b_height_20261118', bs='online_c1b_swing_20261118',
                                            bsf='online_c1b_swing_fast_20261118')[VARIANT] / TARGET
MARKS = (0, 20000, 40000, 60000)
PRIMARY = 40000
NOISE_SEED = 2026111802 if SWING_BASIS else 2026111801 if VARIANT == 'bh' else 2026111703
POSTURE_PENALTY = 3. if VARIANT == 'bh' or SWING_BASIS else 1.
SCRIPT_STARTS, SCRIPT_SLOT = 120, 900000  # 脚本化检查的起点槽位不与训练和评估共用
# 摆动抬脚检查（docs/lite3-online-swing-probe-20261118/PLAN.md）：任务名 -> 残差上限倍数、每步变化上限、起点槽位。
SWING = dict(swing_a=dict(scale=1., rate=.02, slot=910000), swing_b=dict(scale=1., rate=None, slot=910000),
             swing_c=dict(scale=2., rate=None, slot=910000), swing_d=dict(scale=4., rate=None, slot=910000),
             swing_confirm=dict(scale=None, rate=None, slot=930000),
             swing_confirm_a=dict(scale=1., rate=.02, slot=930000))  # 方案之外加的：现有安全过滤在确认起点上的结果
TERRAINS = dict(flat=0, rough=4, slope_up=8, slope_down=10, stairs_up=12, stairs_down=14,
                friction_lateral=16, friction_longitudinal=17, slope_rough_up=18, slope_rough_down=19)


def protocol():
    extra = dict(algorithm='C1-a-M4', replicas=4, optimizer='plain gradient step', sigma=1., learning_rate=.6, max_rounds=100) if VARIANT == 'm4' else {}
    if VARIANT == 'bh':
        extra = dict(algorithm='C1-b-H', replicas=4, optimizer='plain gradient step', sigma=.7, learning_rate=.6, max_rounds=100,
            parameters=8, features=['rough', 'entry', 'middle', 'top'], decision_interval=10, exploration='per-episode parameter noise',
            reward='clip(final_forward_displacement/route_length,0,1) + success - 3*posture_failure',
            script_starts=SCRIPT_STARTS, script_slot=SCRIPT_SLOT)
    if SWING_BASIS:
        extra = dict(algorithm='C1-b-S', replicas=4, optimizer='plain gradient step', sigma=.7, learning_rate=.6, max_rounds=100,
            parameters=['front_static', 'rear_static', 'front_swing', 'rear_swing'], basis='sag static + sag in swing, times roughness',
            exploration='per-episode parameter noise', residual_rate_limit=.24 if VARIANT == 'bsf' else .02,
            reward='clip(final_forward_displacement/route_length,0,1) + success - 3*posture_failure')
    return dict(_protocol(), **extra)


def _protocol():
    return dict(schema=1, algorithm='C1-a-EA' if VARIANT == 'ea' else 'C1-a', covariance='fixed' if VARIANT == 'm4' else VARIANT, heading_feedback=dict(gain=.5, limit=.8, target='world +x'),
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
    return float(np.clip(displacement / length, 0., 1.) + bool(completed) - POSTURE_PENALTY * bool(safety_ending))


def rows(kind, *, part=0, smoke_index=1, script=None):
    parts = 4 if kind == 'target' else 20 if kind == 'retention' else 1
    assert kind in ('smoke', 'train', 'target', 'retention') and 0 <= part < parts
    width = 60 // parts
    result = []
    if script == 'target':
        # 每个起点三对环境：Frozen 都是零码（A），Updated 依次是 B、C、D 三组固定参数。
        assert VARIANT == 'bh' and kind == 'target'
        for start in range(SCRIPT_STARTS):
            for repeat, name in enumerate('BCD'):
                for arm in ('Frozen', 'Updated'):
                    result.append(dict(env_id=len(result), session=start, arm=arm, condition=TARGET, slot=SCRIPT_SLOT + start,
                        repeat=repeat, column=0, row=0, script=name if arm == 'Updated' else 'A'))
        return result
    if script in SWING:
        # 每个起点三对环境：Frozen 是零残差，Updated 依次是 R、Q、S 三组摆动抬脚残差；W 都是 0。
        assert VARIANT == 'bh' and kind == 'target'
        for start in range(SCRIPT_STARTS):
            for repeat, name in enumerate('RQS'):
                for arm in ('Frozen', 'Updated'):
                    result.append(dict(env_id=len(result), session=start, arm=arm, condition=TARGET, slot=SWING[script]['slot'] + start,
                        repeat=repeat, column=0, row=0, script='A', swing=name if arm == 'Updated' else None))
        return result
    cases = list(TERRAINS) if kind == 'retention' else [TARGET]
    for pair in range(part * width, (part + 1) * width):
        for case_index, case in enumerate(cases):
            training = kind in ('train', 'smoke')
            for repeat in range(4 if kind == 'target' else 2 if kind == 'retention' else REPLICAS):
                # 训练时 repeat 是会话内并行环境的编号，各用自己的起点槽位。
                slot = pair + 1000 * repeat if training else pair * 1000 + case_index * 100 + repeat
                for arm in ('Frozen', 'Updated'):
                    result.append(dict(env_id=len(result), session=pair, arm=arm, condition=case, slot=slot,
                        repeat=repeat, column=TERRAINS.get(case, 0), row=(1, 8)[repeat] if kind == 'retention' else 0))
    if script:
        assert VARIANT == 'bh' and script == 'retention' and kind == 'retention'
        for row in result: row['script'] = 'A'  # 保持地形上两组都是零码，只记录特征
    return result


def jobs():
    result = [dict(name='smoke', kind='smoke', part=0, mark=0, steps=600),
              dict(name='train', kind='train', part=0, mark=0, steps=60000)]
    result += [dict(name=f'target_{m}_{p:02d}', kind='target', mark=m, part=p, steps=1050) for m in MARKS[1:] for p in range(4)]
    result += [dict(name=f'retention_{PRIMARY}_{p:02d}', kind='retention', mark=PRIMARY, part=p, steps=1050) for p in range(20)]
    if VARIANT == 'bh':
        result += [dict(name='script', kind='target', part=0, mark=0, steps=1050, script='target'),
                   dict(name='script_retention', kind='retention', part=0, mark=0, steps=1050, script='retention')]
        result += [dict(name=name, kind='target', part=0, mark=0, steps=1050, script=name) for name in SWING]
    for job in result: job['smoke_index'] = 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=[j['name'] for j in jobs()], required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--swing-scale', type=float)
    parser.add_argument('--swing-rate', type=float)
    args = parser.parse_args()
    for key, value in next(j for j in jobs() if j['name'] == args.job).items(): setattr(args, key, value)
    args.script = getattr(args, 'script', None)
    args.output = OUTPUT / args.job
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'], gpu=args.gpu,
                   output=args.output, timeout=18000 * REPLICAS if args.kind == 'train' else 3600, minimum_free_mib=8192)
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
