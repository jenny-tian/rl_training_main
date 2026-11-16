"""固定 zero10100 的三组航向诊断；名义上阶只能复用历史结果。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from reference_eval_checkpoint import register_checkpoint
from reference_stairs_eval import ROOT, save, sha

OUTPUT = ROOT/'logs/reference_six/low_stairs_heading_stability_20261116'
DOCS = ROOT/'docs/lite3-low-stairs-stability-priority-20261116'
CHECKPOINT = ROOT/'logs/reference_six/low_stairs_zero_preservation_20261115/train_001_100/model_updates_10100.pt'
DIGEST = '290d94746c467b5cd21ef74c47a908011521c1f145da35ba7b94f1b92d9b7497'
MODEL = 'lowstairs_zero_preservation10100'
CASES = {
    'heading_down_nominal': ('nominal', 'down'),
    'heading_up_training_randomized': ('training_randomized', 'up'),
    'heading_down_training_randomized': ('training_randomized', 'down'),
}
PROTOCOL = Path(__file__).with_name('reference_isaac_stairs_diagnose.py')
ORIGINAL = Path(__file__).with_name('reference_low_stairs_guarded_heading.py')
PROTOCOL_SHA = 'c155ab8d2050e53eaabda24cb00bee91092a97b8b60a4303638c1a363157bab9'
ORIGINAL_SHA = '36c746c628df7bcdbf8110ae78988b0acc6025c4baa61a7c3e94c70e7bfbafc5'


def feedback(heading, forward):
    import torch
    error = torch.remainder(-heading + torch.pi, 2*torch.pi) - torch.pi
    return torch.where(forward > 0, (.5*error).clamp(-.8, .8), 0.)


def attach_feedback(base):
    """与旧探针相同：50 次准备 step 后包裹协议命令更新。"""
    original_step = base.step
    ticks = 0

    def step(action):
        nonlocal ticks
        if ticks == 50:
            original_compute = base.command_manager.compute
            command = base.command_manager.get_term('base_velocity')

            def compute(dt):
                original_compute(dt)
                command.vel_command_b[:, 2] = feedback(
                    base.scene['robot'].data.heading_w, command.vel_command_b[:, 0])

            base.command_manager.compute = compute
        result = original_step(action)
        ticks += 1
        return result

    base.step = step


def collect(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0',
        kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
    import isaaclab.envs as envs
    import reference_isaac_stairs_diagnose as protocol
    native = envs.ManagerBasedRLEnv

    class HeadingEnvironment(native):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            attach_feedback(self)

    envs.ManagerBasedRLEnv = HeadingEnvironment
    args.profile, args.direction = CASES[args.case]
    args.model, args.action_mode = MODEL, 'mean'
    args.height_set, args.repeats, args.steps, args.video_env = 'low_stairs', 20, None, None
    try:
        protocol.collect(args, app)
    finally:
        envs.ManagerBasedRLEnv = native
    save(args.output/'heading_intervention.json', dict(diagnostic_only=True, optimizer_updates=0,
        gain=.5, limit_rad_s=.8, target='world +x', lateral_position_feedback=False,
        checkpoint_sha256=DIGEST, source_sha256=sha(__file__), case=args.case,
        original_probe_sha256=sha(ORIGINAL), original_protocol_sha256=sha(PROTOCOL),
        initial_preparation_steps=50, rollout_pause_s=.5, rollout_ramp_s=.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(CASES), required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    args.output = OUTPUT/args.case
    assert sha(CHECKPOINT) == DIGEST and sha(PROTOCOL) == PROTOCOL_SHA and sha(ORIGINAL) == ORIGINAL_SHA
    register_checkpoint(MODEL, CHECKPOINT, DIGEST)
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
            gpu=1, output=args.output, minimum_free_mib=10240, timeout=2400)
    try:
        collect(args)
        sys.stdout.flush()
        os._exit(0)
    except Exception:
        traceback.print_exc()
        save(args.output/'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush()
        os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
