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
# 只用于脚本化检查的候选目标（2026-11-17 筛选时搁下的两个），不能用来训练。
PROBE_TARGETS = ('stairs_up_065', 'slope_rough_up_x11')
# 没有训练过的新地形，用于加快学习 C（docs/lite3-online-library-20261122/PLAN.md）；所有变体在它们上面共用槽位 61/62/63。
# 后两个是 2026-11-23 加的，用于检验扩充后的参数库（docs/lite3-online-library2-20261123/PLAN.md）。
NEW_TARGETS = ('stairs_up_07', 'boxes_07', 'stairs_up_075', 'boxes_08')
TARGET = os.environ.get('C1A2_TARGET', TARGETS[0])
assert TARGET in TARGETS + PROBE_TARGETS + NEW_TARGETS
# C1A2_VARIANT=ea 时改用执行感知的探索协方差（C1-a-EA），其余设置完全相同。方案见 docs/lite3-online-c1a-ea-20261117/PLAN.md。
# C1A2_VARIANT=m4 时每个会话用 4 个并行环境、普通梯度步、探索标准差 1.0。方案见 docs/lite3-online-c1a-m4-20261118/PLAN.md。
# C1A2_VARIANT=bh 时码由高度特征决定（C1-b-H），每 10 步重新计算，探索加在 8 个参数上。方案见 docs/lite3-online-c1b-height-20261118/PLAN.md。
# C1A2_VARIANT=bs 时学固定偏置和摆动缩腿共 4 个幅度（C1-b-S）；bsf 是放开每步变化上限的对照组，只做上阶。方案见 docs/lite3-online-c1b-swing-20261118/PLAN.md。
VARIANT = os.environ.get('C1A2_VARIANT', 'fixed')
# s2f / s2n 是 C1-b-S 第二轮的两种做法，目标可以是 6 cm 或 6.5 cm 上阶。方案见 docs/lite3-online-c1b-swing2-20261119/PLAN.md。
# s2ea 是 C1-b-EA：做法二（s2n）加执行感知的探索标准差，其余相同。方案见 docs/lite3-online-c1b-ea-20261120/PLAN.md。
# w2 是加快学习 B（C1-b-W）：扰动和更新放到回合内部，每 100 个策略步一个窗口。上阶从 0 学两个摆动幅度，起点和安全过滤与 s2n 相同；
# 箱体从上阶学到的平均参数出发学 4 个幅度，起点和安全过滤与 bs 相同。方案见 docs/lite3-online-window-20261121/PLAN.md。
# c3 是加快学习 C（C1-b-L）：先从参数库里选一组，再按 C1-b-W 修正 4 个幅度，只用于新地形。
# d1 是加快学习 D（C1-b-D）：整个会话在参数库里持续选择，不加扰动、不做梯度更新；d2 另外在第 4 个回合后加入由领先两组生成的候选。
# 只用于新地形。方案见 docs/lite3-online-select-20261123/PLAN.md。
# s4n 是在新地形上做本地学习用的 C1-b-S：4 个幅度都学，现有安全过滤，起点槽位与其他变体在新地形上的相同，40000 步。
# d3 与 d1 相同，参数库换成扩充后的 6 组。方案见 docs/lite3-online-library2-20261123/PLAN.md。
assert VARIANT in ('fixed', 'ea', 'm4', 'bh', 'bs', 'bsf', 's2f', 's2n', 's2ea', 'w2', 'c3', 'd1', 'd2', 's4n', 'd3', 'r1')
# r1 是 C1-b-R：从 0 起，用摆动受阻比例直接修正两个摆动幅度，不加扰动。方案见 docs/lite3-online-stall-20261123/PLAN.md。
WINDOWED, LIBRARY, SELECT, STALL = VARIANT in ('w2', 'c3', 'd1', 'd2', 'd3', 'r1'), VARIANT == 'c3', VARIANT in ('d1', 'd2', 'd3'), VARIANT == 'r1'
assert not WINDOWED or TARGET in ('stairs_up_06', 'stairs_up_065', 'boxes_06') + NEW_TARGETS
assert TARGET not in NEW_TARGETS or VARIANT in ('s2n', 'w2', 'c3', 'd1', 'd2', 's4n', 'd3', 'r1')
assert not (LIBRARY or SELECT or STALL or VARIANT == 's4n') or TARGET in NEW_TARGETS
assert VARIANT != 'bsf' or TARGET == 'stairs_up_06'
ROUND2 = VARIANT in ('s2f', 's2n', 's2ea', 's4n') or (WINDOWED and TARGET != 'boxes_06')
assert not ROUND2 or TARGET in ('stairs_up_06', 'stairs_up_065') + NEW_TARGETS
TRAINABLE = ('stairs_up_06', 'stairs_up_065') + NEW_TARGETS if ROUND2 else TARGETS
SWING_BASIS = VARIANT in ('bs', 'bsf', 's2f', 's2n', 's2ea', 'w2', 'c3', 'd1', 'd2', 's4n', 'd3', 'r1')
REPLICAS = 4 if VARIANT in ('m4', 'bh') or SWING_BASIS else 1
RUNS = dict(fixed='online_c1a_heading_20261117', ea='online_c1a_ea_20261117', m4='online_c1a_m4_20261118',
            bh='online_c1b_height_20261118', bs='online_c1b_swing_20261118', bsf='online_c1b_swing_fast_20261118',
            s2f='online_c1b_s2f_20261119', s2n='online_c1b_s2n_20261119', s2ea='online_c1b_ea_20261120', w2='online_c1b_window_20261121', c3='online_c1b_library_20261122',
            d1='online_c1b_select_20261123', d2='online_c1b_select_expand_20261123', s4n='online_c1b_s4n_20261123', d3='online_c1b_select6_20261123', r1='online_c1b_stall_20261123')
OUTPUT = ROOT / 'logs/reference_six' / RUNS[VARIANT] / TARGET
# 参数迁移（docs/lite3-online-transfer-20261120/PLAN.md）：C1A2_SOURCE=<变体>:<地形>:<快照>[:pooled] 时只做目标评估，
# Updated 组载入在另一个地形上学到的参数，不再学习；加 pooled 表示所有会话都用 60 个会话的平均参数。
# 变体仍决定评估起点和安全过滤设置：箱体用 bs，上阶用 s2n，所以 Frozen 组与该目标原来的评估逐回合相同。
SOURCE = os.environ.get('C1A2_SOURCE')
if SOURCE:
    _variant, _target, _mark, *_pooled = SOURCE.split(':')
    assert VARIANT in ('bs', 's2n') and _variant in ('bs', 's2n', 's4n') and _target != TARGET and _pooled in ([], ['pooled'])
    SOURCE_FILE, SOURCE_MARK, SOURCE_POOLED = ROOT / 'logs/reference_six' / RUNS[_variant] / _target / 'train' / f'at_{int(_mark)}.pt', int(_mark), bool(_pooled)
    OUTPUT = ROOT / 'logs/reference_six' / ('online_library2_transfer_20261123' if TARGET in NEW_TARGETS[2:] else 'online_library_transfer_20261122' if TARGET in NEW_TARGETS
                                            else 'online_transfer_20261120') / TARGET / ('from_' + SOURCE.replace(':', '_'))
# 候选参数的不学习评估（加快学习 D 的 D0）：C1A2_CANDIDATE=<名字> 时只做目标评估，Updated 组全部用这一组固定参数。
# 名字是 <领先的库编号><第二的库编号><m|u|d>：m 是 tanh 空间里两组的中点，u、d 是领先一组的摆动幅度乘 1.2、0.8（与 c1b_d.candidates 相同）。
CANDIDATE = os.environ.get('C1A2_CANDIDATE')
if CANDIDATE:
    assert not SOURCE and VARIANT == 's2n' and TARGET in NEW_TARGETS and len(CANDIDATE) == 3 and CANDIDATE[0] != CANDIDATE[1]
    assert CANDIDATE[0] in '012' and CANDIDATE[1] in '0123' and CANDIDATE[2] in 'mud'
    SOURCE_MARK = 0
    OUTPUT = ROOT / 'logs/reference_six/online_select_candidates_20261123' / TARGET / CANDIDATE
# 提前起抬的脚本化检查（docs/lite3-online-lead-20261123/PLAN.md）：C1A2_LEAD=<规则>:<库编号> 时只做目标评估，
# Updated 组全部用库里这一组参数，并按规则在腿离地前就开始请求摆动残差。安全过滤不变。
LEAD = os.environ.get('C1A2_LEAD')
if LEAD:
    LEAD_RULE, _entry = LEAD.split(':')
    assert not SOURCE and not CANDIDATE and VARIANT == 's2n' and TARGET in ('stairs_up_06',) + NEW_TARGETS
    assert LEAD_RULE in ('none', 'knee', 'partner', 'age') and _entry in '012'
    FIXED_ENTRY, SOURCE_MARK = int(_entry), 0
    OUTPUT = ROOT / 'logs/reference_six/online_lead_20261123' / TARGET / LEAD.replace(':', '_')
# 每步变化上限的仿真扫描（docs/lite3-online-rate-20261123/PLAN.md）：C1A2_RATE=<上限>:<库编号> 时只做目标评估，
# Updated 组全部用库里这一组参数，安全过滤的每步变化上限换成给定的值，其余不变。只在仿真里改。
RATE = os.environ.get('C1A2_RATE')
if RATE:
    _rate, _entry = RATE.split(':')
    assert not SOURCE and not CANDIDATE and not LEAD and VARIANT == 's2n' and TARGET in ('stairs_up_06', 'stairs_up_07', 'boxes_07')
    assert _rate in ('0.04', '0.06', '0.10', '0.24') and _entry in '012'
    RATE_LIMIT, FIXED_ENTRY, SOURCE_MARK = float(_rate), int(_entry), 0
    OUTPUT = ROOT / 'logs/reference_six/online_rate_20261123' / TARGET / ('rate_' + RATE.replace(':', '_'))
# 新地形上的本地学习直接保存 5000、10000 步的快照，用来报告学习速度。
MARKS = ((0, 1250, 2500, 5000, 10000) if WINDOWED else (0, 5000, 10000, 20000, 40000) if VARIANT == 's4n'
         else (0, 5000, 10000, 20000, 40000, 60000) if VARIANT == 's2ea' or (VARIANT == 's2n' and TARGET in NEW_TARGETS) else (0, 20000, 40000, 60000))
PRIMARY = 2500 if LIBRARY or STALL else 10000 if SELECT else 40000 if VARIANT == 's4n' else (10000 if TARGET == 'boxes_06' else 5000) if WINDOWED else 60000 if VARIANT in ('s2n', 's2ea') else 40000
TRAIN_STEPS = 10000 if WINDOWED else 40000 if VARIANT == 's4n' else 60000
# w2 在箱体上的起点：s2n 在 6 cm 上阶 60000 步的 60 个会话平均参数（与迁移评估 T3 的平均参数相同）。
WINDOW_INITIAL = ROOT / 'logs/reference_six' / RUNS['s2n'] / 'stairs_up_06/train/at_60000.pt' if WINDOWED and TARGET == 'boxes_06' else None
# 只作描述的早期快照，评估排在登记的任务之后。s2ea 训练时直接保存；s2n 当时没有存，由训练记录重建
# （reference_online_c1bea_early.py），s2n 已登记的 MARKS 不变。
EARLY_MARKS = (5000, 10000) if VARIANT in ('s2n', 's2ea', 's4n') else ()
# 参数库：（变体, 地形, 快照）的 60 个会话平均参数，最后一组是全 0。
LIBRARY_SOURCES = (('s2n', 'stairs_up_06', 60000), ('s2n', 'stairs_up_065', 60000), ('bs', 'boxes_06', 40000))
# 扩充后的参数库（d3 用）：加上在 7 cm 上阶和 7 cm 箱体上本地学到的两组。
LIBRARY_SOURCES_6 = LIBRARY_SOURCES + (('s2n', 'stairs_up_07', 60000), ('s4n', 'boxes_07', 40000))
# d1、d2 的 Frozen 训练组用和 c3 相同的扰动表；选择用的随机数另有种子。
SELECT_SEED = 2026112301
NOISE_SEED = 2026112201 if LIBRARY or SELECT else 2026112101 if WINDOWED else 2026111901 if ROUND2 else 2026111802 if SWING_BASIS else 2026111801 if VARIANT == 'bh' else 2026111703
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
            exploration='per-episode parameter noise', residual_rate_limit=.24 if VARIANT in ('bsf', 's2f') else .02,
            reward='clip(final_forward_displacement/route_length,0,1) + success - 3*posture_failure')
        if ROUND2:
            extra.update(algorithm='C1-b-S round 2 ' + VARIANT, namespaces=dict(smoke=60, train=61, target=62, retention=63),
                         active_parameters=['front_swing', 'rear_swing'] if VARIANT in ('s2n', 's2ea', 'w2', 'r1') else None)
        if VARIANT == 's4n':
            extra.update(train_control_steps=TRAIN_STEPS)
        if LIBRARY:
            extra.update(library=[list(x) for x in LIBRARY_SOURCES] + ['zero'], select_rounds=2)
        if SELECT:
            extra.update(library=[list(x) for x in (LIBRARY_SOURCES_6 if VARIANT == 'd3' else LIBRARY_SOURCES)] + ['zero'], rotate_rounds=3 if VARIANT == 'd3' else 2, allocation='Thompson sampling on success, Beta(1, 1) prior',
                         score='success + 0.01 * mean window feedback', select_seed=SELECT_SEED, expand_after_round=4 if VARIANT == 'd2' else None)
        if VARIANT == 's2ea':
            extra.update(algorithm='C1-b-S-EA', covariance='execution-aware')
        if WINDOWED:
            extra.update(algorithm='C1-b-L' if LIBRARY else 'C1-b-D' if SELECT else 'C1-b-R' if STALL else 'C1-b-W',
                         exploration='none; one candidate per attempt' if SELECT else 'none; stall-ratio feedback' if STALL else 'per-window parameter noise', window_steps=100, settle_steps=10,
                         train_control_steps=TRAIN_STEPS, primary_mark=PRIMARY, marks=list(MARKS),
                         reward='own-leg unimpeded swing ratio + window progress - 3*posture_failure',
                         initial='pooled s2n stairs_up_06 at 60000' if WINDOW_INITIAL else 'selected from library' if LIBRARY or SELECT else 'zero')
        if SOURCE:
            extra.update(transfer_source=SOURCE)
        if CANDIDATE:
            extra.update(candidate=CANDIDATE)
        if LEAD:
            extra.update(lead=LEAD)
        if RATE:
            extra.update(rate_check=RATE, residual_rate_limit=RATE_LIMIT)
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
    if SOURCE or CANDIDATE or LEAD or RATE:
        return [dict(name=f'target_{SOURCE_MARK}_{p:02d}', kind='target', mark=SOURCE_MARK, part=p, steps=1050, smoke_index=1) for p in range(4)]
    result = [dict(name='smoke', kind='smoke', part=0, mark=0, steps=600),
              dict(name='train', kind='train', part=0, mark=0, steps=TRAIN_STEPS)]
    result += [dict(name=f'target_{m}_{p:02d}', kind='target', mark=m, part=p, steps=1050) for m in MARKS[1:] if m not in EARLY_MARKS for p in range(4)]
    result += [dict(name=f'retention_{PRIMARY}_{p:02d}', kind='retention', mark=PRIMARY, part=p, steps=1050) for p in range(20)]
    result += [dict(name=f'target_{m}_{p:02d}', kind='target', mark=m, part=p, steps=1050, extra=True) for m in EARLY_MARKS for p in range(4)]
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
    assert TARGET in TRAINABLE or args.script in SWING, 'probe targets only run the swing checks'
    args.output = OUTPUT / args.job
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'], gpu=args.gpu,
                   output=args.output, timeout=18000 * REPLICAS if args.kind == 'train' else 3600,
                   minimum_free_mib=int(os.environ.get('C1A2_MIN_FREE_MIB', 8192)))  # 并行调度时由 reference_online_c1a2_pool.py 给出
    try:
        from isaaclab.app import AppLauncher
        AppLauncher(headless=True, device='cuda:0',
                    kit_args=f'--/renderer/multiGpu/enabled=False --/renderer/activeGpu={args.gpu}').app
        from reference_online_c1a2_collect import collect
        if os.environ.get('C1A2_PROFILE'):  # 只用于查用时分布，不改变采集内容
            import cProfile
            profiler = cProfile.Profile(); profiler.runcall(collect, args); profiler.dump_stats(str(args.output / 'profile.pstats'))
        else:
            collect(args)
        sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc(); save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__': sys.exit(main())
