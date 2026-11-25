"""S0 目标筛选里不依赖仿真的部分：每个回合的诊断量、入选判定。方案见 docs/lite3-online-targets-20261125/PLAN.md。

诊断量都按控制步累计，只统计回合还在进行的环境：
  支撑脚滑动      脚在这一步开始和结束时都着地，水平位移折合速度超过 SLIP_SPEED
  摆动脚受阻      脚在这一步开始时离地，前向位移折合速度低于 STALL_SPEED（与受阻修正 c1b_w 的定义相同），前腿、后腿分开
  摆动脚碰撞      脚在这一步结束时不承重，水平接触力超过 HIT_FORCE（方案登记的四个量之外加的，用来区分脚被挡住和原地踏步）
  力矩到限        任一关节的力矩被执行器限幅
  最大机身倾角    重力方向与机身 z 轴的夹角
"""
from __future__ import annotations

import collections

import torch

DT, SLIP_SPEED, STALL_SPEED, HIT_FORCE, TORQUE_MARGIN = .02, .10, .10, 5., 1e-3
OUTCOMES = ('success', 'posture_failure', 'lateral_exit', 'stuck', 'timeout', 'start_failure')
EPISODES, SUCCESS_MIN, SUCCESS_MAX, PROGRESS_MIN, START_FAILURE_MAX, PREFERRED_SUCCESS = 64, 6, 45, .30, 6, 20
_BANDS = ('1020', '0512', '0206')
# 每一族里的候选按由易到难排成链；摩擦变化有左右、前后两种布局，各自成链。
CHAINS = dict(
    rough=[['rough_09', 'rough_12', 'rough_15']],
    slope_up=[['slope_up_50', 'slope_up_60', 'slope_up_70']],
    slope_down=[['slope_down_50', 'slope_down_60', 'slope_down_70']],
    slope_rough_up=[['slope_rough_up_x11', 'slope_rough_up_x12']],
    slope_rough_down=[['slope_rough_down_x085', 'slope_rough_down_x10', 'slope_rough_down_x11']],
    stairs_down=[['stairs_down_09', 'stairs_down_10']],
    friction=[[f'friction_{layout}_{band}' for band in _BANDS] for layout in ('lateral', 'longitudinal')])
REFERENCE = ('stairs_up_07', 'boxes_07')  # 已有的在线目标，只用来对比失败形式


class EpisodeDiagnostics:
    def __init__(self, environments, device='cpu'):
        zeros = lambda *shape: torch.zeros(environments, *shape, device=device)
        self.n = environments
        self.steps, self.limited, self.tilt = zeros(), zeros(), zeros()
        self.stance_steps, self.slip_steps, self.stance_distance = zeros(), zeros(), zeros()
        self.swing_steps, self.stall_steps = zeros(2), zeros(2)
        self.free_steps, self.hit_steps = zeros(), zeros()
        self.stance = self.feet = None

    def start(self, stance, feet_xy):
        """策略开始前调用一次：记下这时四只脚是否着地和水平位置。"""
        if stance.dtype != torch.bool or stance.shape != (self.n, 4) or feet_xy.shape != (self.n, 4, 2):
            raise ValueError('stance (n, 4) bool and feet (n, 4, 2) required')
        self.stance, self.feet = stance.clone(), feet_xy.clone().float()

    def step(self, active, stance, feet_xy, force_xy, computed_torque, applied_torque, gravity):
        """每个控制步之后调用一次，传入走完这一步后的量。active (n,) 是这个环境的回合是否还在进行。"""
        if self.stance is None:
            raise ValueError('start must be called first')
        if (active.dtype != torch.bool or active.shape != (self.n,) or stance.dtype != torch.bool or stance.shape != (self.n, 4)
                or feet_xy.shape != (self.n, 4, 2) or force_xy.shape != (self.n, 4, 2)
                or computed_torque.shape != applied_torque.shape or computed_torque.shape[0] != self.n or gravity.shape != (self.n, 3)):
            raise ValueError('unexpected diagnostic shapes')
        on = active.float()
        moved = feet_xy.float() - self.feet
        speed = torch.linalg.vector_norm(moved, dim=-1) / DT
        grounded = self.stance & stance
        self.stance_steps += grounded.sum(1) * on
        self.slip_steps += (grounded & (speed > SLIP_SPEED)).sum(1) * on
        self.stance_distance += (speed * grounded).sum(1) * on
        swing = ~self.stance
        stalled = swing & (moved[..., 0] / DT < STALL_SPEED)
        self.swing_steps += swing.view(self.n, 2, 2).sum(2) * on[:, None]
        self.stall_steps += stalled.view(self.n, 2, 2).sum(2) * on[:, None]
        free = ~stance
        self.free_steps += free.sum(1) * on
        self.hit_steps += (free & (torch.linalg.vector_norm(force_xy.float(), dim=-1) > HIT_FORCE)).sum(1) * on
        self.limited += ((computed_torque.abs() - applied_torque.abs()) > TORQUE_MARGIN).any(1) * on
        tilt = torch.rad2deg(torch.arccos((-gravity[:, 2]).clamp(-1., 1.)))
        self.tilt = torch.where(active, torch.maximum(self.tilt, tilt.float()), self.tilt)
        self.steps += on
        self.stance, self.feet = stance.clone(), feet_xy.clone().float()

    def summary(self, i):
        ratio = lambda a, b: float(a) / float(b) if float(b) > 0 else None
        steps = int(self.steps[i])
        return dict(steps=steps, stance_slip_ratio=ratio(self.slip_steps[i], self.stance_steps[i]),
            stance_foot_speed=ratio(self.stance_distance[i], self.stance_steps[i]),
            swing_stall_ratio_front=ratio(self.stall_steps[i, 0], self.swing_steps[i, 0]),
            swing_stall_ratio_rear=ratio(self.stall_steps[i, 1], self.swing_steps[i, 1]),
            swing_hit_ratio=ratio(self.hit_steps[i], self.free_steps[i]),
            torque_limit_ratio=ratio(self.limited[i], steps), max_tilt_deg=float(self.tilt[i]) if steps else None)


def start_failure_record(spec, final_x, final_y=None):
    """站立准备阶段就终止的回合：记为起步失败，不进入策略阶段。"""
    return dict(**spec, outcome='start_failure', duration_s=0., final_x=float(final_x), final_y=None if final_y is None else float(final_y),
                progress=0., max_abs_heading_rad=0., speed_mae=None, diagnostics=None)


def candidate_stats(records):
    counts = collections.Counter(r['outcome'] for r in records)
    if set(counts) - set(OUTCOMES):
        raise ValueError(f'unknown outcomes: {sorted(set(counts) - set(OUTCOMES))}')
    return dict(n=len(records), **{name: counts.get(name, 0) for name in OUTCOMES},
                mean_progress=sum(r['progress'] for r in records) / len(records))


def classify(stats):
    """方案第 4 节登记的入选规则，按 64 个回合。"""
    if stats['n'] != EPISODES:
        raise ValueError('the registered rule is defined on 64 episodes')
    if stats['success'] > SUCCESS_MAX: return 'too_easy'
    failures = stats['n'] - stats['success']
    if stats['start_failure'] > START_FAILURE_MAX or stats['lateral_exit'] > failures / 2: return 'excluded'
    if stats['success'] < SUCCESS_MIN or stats['mean_progress'] < PROGRESS_MIN: return 'too_hard'
    return 'eligible'


def select(stats, chains=None):
    """每一族取成功数最接近 20 的入选候选；没有入选时，报告链上相邻的一个太容易、一个太难的两个候选。"""
    result = {}
    for family, groups in (chains or CHAINS).items():
        classes = {name: classify(stats[name]) for chain in groups for name in chain}
        eligible = [name for name, kind in classes.items() if kind == 'eligible']
        chosen = min(eligible, key=lambda name: abs(stats[name]['success'] - PREFERRED_SUCCESS)) if eligible else None
        bracket = None
        if chosen is None:
            pairs = [(a, b) for chain in groups for a, b in zip(chain[:-1], chain[1:]) if classes[a] == 'too_easy' and classes[b] == 'too_hard']
            bracket = pairs[0] if pairs else None
        result[family] = dict(chosen=chosen, bracket=bracket, classes=classes)
    return result
