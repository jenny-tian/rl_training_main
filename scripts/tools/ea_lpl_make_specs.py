"""Write EA-LPL run specifications as JSON files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/rl_training")]
from rl_training.ea_lpl.headroom import GRID_NAMESPACE, grid_rows, pack_specs  # noqa: E402
from rl_training.ea_lpl.specs import p0_specs, validate_spec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("p0", "p1-grid", "p1-zero-check"))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "p0":
        specs = p0_specs()
    elif args.phase == "p1-grid":
        targets = json.loads(args.inputs.read_text())["targets"]
        specs = pack_specs(grid_rows(targets), label_prefix="p1_grid", namespace=GRID_NAMESPACE)
    elif args.phase == "p1-zero-check":
        base = next(s for s in p0_specs() if s["label"] == "p0_e1p_l1")
        specs = [dict(base, label="p1_zero_check_l1", codes=dict(
            basis=["sag"] * 120, u=[[0.0, 0.0]] * 120, code_name=["zero"] * 120, record_replay=False))]
    for spec in specs:
        validate_spec(spec)
        (args.out / f"{spec['label']}.json").write_text(json.dumps(spec, indent=2) + "\n")
        print(args.out / f"{spec['label']}.json")


if __name__ == "__main__":
    main()
