"""Count pre-action tilt exposure, separately from recovery-episode membership."""
import argparse
import json
from pathlib import Path
import numpy as np
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def summarize(root):
    report_path = root / 'report.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete'
    counts = {group: {'controls': 0, **{f'above_{angle}_deg': 0 for angle in (10, 20, 30)}}
              for group in ('all', 'recovery', 'ordinary')}
    evidence = {}
    for path in sorted((root / 'trajectory').glob('chunk-*.npz')):
        evidence[path.name] = sha256_file(path)
        with np.load(path, allow_pickle=False) as data:
            # Observations 3:6 are body-frame projected unit gravity. At rest z=-1.
            gravity = data['policy_observation'][..., 3:6].astype(np.float64)
            norm = np.linalg.norm(gravity, axis=-1)
            np.testing.assert_allclose(norm, 1., atol=2e-5)
            cos_tilt = np.clip(-gravity[..., 2] / norm, -1., 1.)
            recovery = data['recovery_case'] >= 0
            for group, mask in (('all', np.ones_like(recovery)), ('recovery', recovery), ('ordinary', ~recovery)):
                counts[group]['controls'] += int(mask.sum())
                for angle in (10, 20, 30):
                    counts[group][f'above_{angle}_deg'] += int((mask & (cos_tilt < np.cos(np.deg2rad(angle)))).sum())
    assert counts['all']['controls'] == report['ledger']['confirmed']
    return {'counts': counts, 'report_sha256': sha256_file(report_path), 'trace_sha256': evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write_json_atomic(args.output, {'status': 'verified_exposure_counts',
        'measurement': 'Pre-action projected-gravity tilt; threshold counts overlap. Recovery episode membership includes upright controls after recovery.',
        'runs': {run: summarize(Path(run)) for run in args.runs}})


if __name__ == '__main__':
    main()
