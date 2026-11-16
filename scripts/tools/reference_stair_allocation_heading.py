"""Reuse the frozen stair protocol with the unchanged diagnostic yaw feedback."""
import argparse
import os
from pathlib import Path
import sys
import traceback

from reference_eval_checkpoint import register_checkpoint
from reference_stairs_eval import save, sha
from reference_low_stairs_heading_probe import attach_feedback, PROTOCOL, PROTOCOL_SHA, ORIGINAL, ORIGINAL_SHA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--profile', choices=('nominal', 'training_randomized'), required=True)
    p.add_argument('--direction', choices=('up', 'down'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args(); args.output = args.output.resolve()
    assert sha(PROTOCOL) == PROTOCOL_SHA and sha(ORIGINAL) == ORIGINAL_SHA
    register_checkpoint(args.model, args.checkpoint, args.checkpoint_sha256)
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'],
                   gpu=1, output=args.output, timeout=2400, minimum_free_mib=10240)
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device='cuda:0',
            kit_args='--/renderer/multiGpu/enabled=False --/renderer/activeGpu=1').app
        import isaaclab.envs as envs
        import reference_isaac_stairs_diagnose as protocol
        native = envs.ManagerBasedRLEnv
        class HeadingEnvironment(native):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                attach_feedback(self)
        envs.ManagerBasedRLEnv = HeadingEnvironment
        args.action_mode, args.height_set = 'mean', 'low_stairs'
        args.repeats, args.steps, args.video_env = 20, None, None
        try:
            protocol.collect(args, app)
        finally:
            envs.ManagerBasedRLEnv = native
        save(args.output / 'heading_intervention.json', dict(diagnostic_only=True, optimizer_updates=0,
            gain=.5, limit_rad_s=.8, target='world +x', lateral_position_feedback=False,
            checkpoint_sha256=args.checkpoint_sha256, source_sha256=sha(__file__),
            feedback_source_sha256=sha(Path(__file__).with_name('reference_low_stairs_heading_probe.py')),
            original_probe_sha256=ORIGINAL_SHA, original_protocol_sha256=PROTOCOL_SHA,
            initial_preparation_steps=50, rollout_pause_s=.5, rollout_ramp_s=.5))
        sys.stdout.flush(); os._exit(0)
    except Exception:
        traceback.print_exc(); save(args.output / 'error.json', dict(error=traceback.format_exc()))
        sys.stdout.flush(); os._exit(1)


if __name__ == '__main__': sys.exit(main())
