"""Offline audit or execution-contract preflight for a Source scan manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/rl_training"))

from rl_training.source_scan.manifest import (  # noqa: E402
    ManifestRejected,
    audit_manifest,
    load_manifest,
    validate_manifest_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="require a locked, approved v0.2 manifest and verify exact artifact hashes; does not start Isaac Sim",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.preflight:
            report = validate_manifest_file(args.manifest)
        else:
            report = audit_manifest(load_manifest(args.manifest))
    except (OSError, ValueError, ManifestRejected) as error:
        if isinstance(error, ManifestRejected):
            print(json.dumps(error.report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.schema_valid and (not args.preflight or report.execution_ready) else 1


if __name__ == "__main__":
    raise SystemExit(main())
