"""Write paired stage-1 run specifications for explicitly selected actor files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training")]
from rl_training.ea_lpl.specs import stage1_specs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--kind", required=True, choices=("qualify", "preview", "extend"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    policies = {}
    for item in args.policy:
        name, separator, value = item.partition("=")
        if not separator or not value or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None:
            parser.error("--policy must be NAME=PATH with an alphanumeric, underscore or hyphen name")
        if name in policies:
            parser.error(f"duplicate policy name: {name}")
        path = Path(value).expanduser().resolve(strict=True)
        policies[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    specs = stage1_specs(policies, kind=args.kind)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in specs:
        path = args.out_dir / f"{spec['label']}.json"
        path.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
        paths.append(str(path))
    (args.out_dir / "specs.txt").write_text("\n".join(paths) + "\n")


if __name__ == "__main__":
    main()
