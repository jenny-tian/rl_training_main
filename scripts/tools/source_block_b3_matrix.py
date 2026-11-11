"""Run one physical GPU lane of the registered B3 frozen evaluation matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from source_block_b3_protocol import ROOT, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("flat", "block"), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    for spec in models():
        for mode in ("controlled", "native"):
            output = ROOT / "logs/stage1/block_b3/matrix" / args.domain / mode / spec["name"]
            command = [sys.executable, str(ROOT / "scripts/tools/source_block_b3_diagnose.py"),
                "--model", spec["name"], "--domain", args.domain, "--mode", mode,
                "--gpu", str(args.gpu), "--output", str(output)]
            subprocess.run(command, cwd=ROOT, check=True)
            report = json.loads((output / "report.json").read_text())
            if report["status"] != "passed" or report["cleanup"] != {"environment": "returned", "application": "returned"}:
                raise RuntimeError(f"incomplete B3 job: {output}")
            print(f"MATRIX COMPLETE {args.domain}/{mode}/{spec['name']}", flush=True)


if __name__ == "__main__":
    main()
