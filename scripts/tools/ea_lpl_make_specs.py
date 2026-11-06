"""Write EA-LPL run specifications as JSON files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training")]
from rl_training.ea_lpl.specs import p0_specs, validate_spec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("p0",))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in p0_specs():
        validate_spec(spec)
        (args.out / f"{spec['label']}.json").write_text(json.dumps(spec, indent=2) + "\n")
        print(args.out / f"{spec['label']}.json")


if __name__ == "__main__":
    main()
