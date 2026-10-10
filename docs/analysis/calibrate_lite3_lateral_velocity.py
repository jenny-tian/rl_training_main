"""Fix one reward scale from audited TRAIN trajectories; never read pilot outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'protocol', 'training-audit', 'heading-audit', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    root = Path(args.run)
    paths = [root/'report.json', Path(args.protocol), Path(args.training_audit), Path(args.heading_audit)]
    report, protocol, audit, heading = [json.loads(p.read_text()) for p in paths]
    assert report['status'] == 'complete' and audit['status'] == heading['status'] == 'verified'
    assert report['protocol_sha256'] == sha256_file(args.protocol)
    assert audit['exposed_training_cases'] == [26] and audit['validation_cases_used'] == 0
    assert protocol['recovery_resets']['split'] == 'train'
    for checked in (audit, heading):
        assert checked['evidence'][str(paths[0])] == sha256_file(paths[0])
    # A two-second local return timescale; cap at half the fixed forward speed.
    gain = .5
    maximum = protocol['fixed_forward_command'] / 2
    groups = {k: [] for k in ('ordinary', 'early', 'tail')}
    for path in sorted((root/'trajectory').glob('chunk-*.npz')):
        assert sha256_file(path) == audit['trace_sha256'][path.name] == heading['trace_sha256'][path.name]
        with np.load(path, allow_pickle=False) as data:
            state = data['heading_root_state']
            y = state[..., 1] - data['heading_origins'][..., 1]
            vy = state[..., 8]
            target = np.clip(-gain*y, -maximum, maximum)
            ordinary = data['recovery_case'] < 0
            early = ~ordinary & (data['episode_steps'] < 12)
            values = np.stack((y, vy, target, (vy-target)**2, 8*y*y), -1)
            for name, mask in [('ordinary', ordinary), ('early', early), ('tail', ~ordinary & ~early)]:
                groups[name].append(values[mask])
    groups = {name: np.concatenate(parts) for name, parts in groups.items()}
    assert sum(len(v) for v in groups.values()) == audit['controls']
    assert len(groups['early']) + len(groups['tail']) == audit['recovery_controls']
    assert all(np.isfinite(v).all() for v in groups.values())
    # Predeclared scale rule: about 10% of the root-only lateral penalty proxy
    # on ALL recovery controls after the existing 12-step early window.
    tail = groups['tail']
    ratio = .1 * tail[:, 4].mean() / tail[:, 3].mean()
    magnitude = float(2. ** np.clip(np.rint(np.log2(ratio)), -1, 3))
    summary = {}
    for name, values in groups.items():
        y, vy, target, cost, proxy = values.T
        summary[name] = dict(rows=len(values), mean_abs_y_m=float(abs(y).mean()),
            mean_outward_world_velocity_m_s=float((np.sign(y)*vy).mean()),
            mean_cost=float(cost.mean()), p95_cost=float(np.quantile(cost, .95)),
            mean_root_lateral_penalty_proxy=float(proxy.mean()),
            mean_new_penalty_per_second=float(magnitude*cost.mean()),
            fraction_target_saturated=float((abs(target) >= maximum).mean()))
    write_json_atomic(args.output, dict(status='verified', gain_per_second=gain,
        maximum_m_s=maximum, weight=-magnitude, unrounded_weight_magnitude=float(ratio), groups=summary,
        rule='gain0.5/s; cap=forward_command/2; abs(weight)=nearest power of2 to 0.1*mean(8*y^2)/mean((vy-target)^2) on all TRAIN recovery age>=12, clipped[0.5,8]. Root-only lateral proxy, not reconstruction of body/foot max penalty.',
        scope='All changing-policy TRAIN controls including failures. No pilot/held-out input, no optimization labels, no hyperparameter sweep.',
        new_controls=0, evidence={str(p): sha256_file(p) for p in paths},
        calibration_source_sha256=sha256_file(__file__)))


if __name__ == '__main__':
    main()
