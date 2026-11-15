"""独立记录世界 +x 航向反馈的上阶诊断，不能替代原协议验收。"""
import argparse
import os
from pathlib import Path
import sys
import traceback

from reference_eval_checkpoint import register_checkpoint
from reference_stairs_eval import save, sha


def collect(args):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0',
        kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
    import torch
    import isaaclab.envs as envs
    import reference_isaac_stairs_diagnose as protocol
    native = envs.ManagerBasedRLEnv
    class HeadingEnvironment(native):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            original_step = self.step
            ticks = 0
            def step(action):
                nonlocal ticks
                if ticks == 50:
                    original_compute = self.command_manager.compute
                    command = self.command_manager.get_term('base_velocity')
                    def compute(dt):
                        original_compute(dt)
                        heading = self.scene['robot'].data.heading_w
                        error = torch.remainder(-heading + torch.pi, 2*torch.pi) - torch.pi
                        command.vel_command_b[:,2] = torch.where(command.vel_command_b[:,0] > 0,
                            (.5*error).clamp(-.8,.8), 0.)
                    self.command_manager.compute = compute
                result = original_step(action); ticks += 1
                return result
            self.step = step
    envs.ManagerBasedRLEnv = HeadingEnvironment
    args.profile, args.direction, args.action_mode = 'nominal', 'up', 'mean'
    args.height_set, args.repeats, args.steps, args.video_env = 'low_stairs', 20, None, None
    try:
        protocol.collect(args, app)
    finally:
        envs.ManagerBasedRLEnv = native
    save(args.output/'heading_intervention.json', dict(diagnostic_only=True, optimizer_updates=0,
        gain=.5, limit_rad_s=.8, target='world +x', lateral_position_feedback=False,
        checkpoint_sha256=args.checkpoint_sha256, source_sha256=sha(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args(); args.output = args.output.resolve()
    register_checkpoint(args.model, args.checkpoint, args.checkpoint_sha256)
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=1, output=args.output, minimum_free_mib=10240, timeout=2400)
    try:
        collect(args); sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc(); save(args.output/'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__':
    sys.exit(main())
