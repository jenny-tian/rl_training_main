"""Portable entry point for the original six-family, low-stair scratch training."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "validate", "train"), required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.updates is None:
        if args.mode == "train":
            parser.error("train requires an explicit --updates budget (1..10000)")
        args.updates = 0 if args.mode == "smoke" else 3
    if ((args.mode == "smoke" and args.updates != 0)
            or (args.mode == "validate" and args.updates != 3)
            or (args.mode == "train" and not 1 <= args.updates <= 10000)):
        parser.error("smoke requires 0 updates; validate requires 3; train accepts 1..10000")
    args.reference_root = args.reference_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    import reference_lite3_loader as loader

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.reference_root, text=True).strip()
    if revision != loader.REFERENCE_COMMIT:
        parser.error(f"reference checkout must be at {loader.REFERENCE_COMMIT}")
    if not args.child:
        from source_block_runtime import run

        command = [sys.executable, str(Path(__file__).resolve()),
                   "--reference-root", str(args.reference_root), "--mode", args.mode,
                   "--updates", str(args.updates), "--seed", str(args.seed),
                   "--gpu", str(args.gpu), "--output", str(args.output), "--child"]
        return run(command, gpu=args.gpu, output=args.output,
                   timeout=172800 if args.mode == "train" else 3600)
    loader.REFERENCE = args.reference_root
    from reference_six_train import save, worker

    args.terrain_profile = args.reward_variant = "low_stairs"
    args.resume_final = False
    try:
        worker(args)
    except Exception:
        traceback.print_exc()
        save(args.output / "error.json", dict(error=traceback.format_exc()))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
