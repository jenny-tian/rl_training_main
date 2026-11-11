"""Verify the actual one-function config difference and candidate GPU physics."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from source_block_b2_protocol import save_json
from source_block_b4_protocol import RAW, TASKS, config_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.output = RAW / "smoke"
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), "--child"], gpu=args.gpu, output=args.output)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    from source_curriculum_check import Checks
    checks = Checks()
    report = dict(task=TASKS["turngate"], num_envs=120, checks=checks.results)
    try:
        import rl_training.tasks  # noqa: F401
        _, _, audit = config_audit(42, 4096)
        save_json(args.output / "config_audit.json", audit)
        checks.add("single_config_change", True, str(audit["differences"]))
        import source_block_check
        source_block_check.B2_TASK = TASKS["turngate"]
        args.b2, args.reference_rough, args.num_envs = True, True, 120
        source_block_check.check(args, checks, report)
    except Exception:
        traceback.print_exc()
        checks.add("execution", False, traceback.format_exc())
    report["all_passed"] = checks.all_passed
    save_json(args.output / "checks.json", report)
    print(f"B4 SMOKE {sum(c['passed'] for c in checks.results)}/{len(checks.results)}", flush=True)
    os._exit(0 if checks.all_passed else 1)


if __name__ == "__main__":
    sys.exit(main())
