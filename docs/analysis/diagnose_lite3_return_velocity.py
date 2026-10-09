"""Describe TRAIN recovery velocity by age without producing optimization labels."""
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
    paths = [root / 'report.json', Path(args.protocol), Path(args.training_audit), Path(args.heading_audit)]
    report, protocol, audit, heading = [json.loads(p.read_text()) for p in paths]
    assert report['status'] == 'complete' and audit['status'] == heading['status'] == 'verified'
    assert report['protocol_sha256'] == sha256_file(args.protocol)
    assert audit['evidence'][str(paths[0])] == heading['evidence'][str(paths[0])] == sha256_file(paths[0])
    assert protocol['recovery_resets']['split'] == 'train'
    assert audit['exposed_training_cases'] == [26] and audit['validation_cases_used'] == 0
    bins = [(0, 12), (12, 50), (50, 100), (100, 200), (200, None)]
    pieces = {str(lo): [] for lo, _ in bins}
    for path in sorted((root / 'trajectory').glob('chunk-*.npz')):
        assert sha256_file(path) == audit['trace_sha256'][path.name] == heading['trace_sha256'][path.name]
        with np.load(path, allow_pickle=False) as d:
            state = d['heading_root_state']
            y = state[..., 1] - d['heading_origins'][..., 1]
            w, x, qy, z = np.moveaxis(state[..., 3:7], -1, 0)
            yaw = np.arctan2(2 * (w*z + x*qy), 1 - 2 * (qy*qy + z*z))
            vy = state[..., 8]
            yaw_frame_vy = -np.sin(yaw)*state[..., 7] + np.cos(yaw)*vy
            for lo, hi in bins:
                selected = (d['recovery_case'] == 26) & (d['episode_steps'] >= lo)
                if hi is not None:
                    selected &= d['episode_steps'] < hi
                pieces[str(lo)].append(np.stack([y[selected], yaw[selected], vy[selected], yaw_frame_vy[selected]], -1))
    results = {}
    for lo, hi in bins:
        values = np.concatenate(pieces[str(lo)])
        assert len(values) and np.isfinite(values).all()
        y, yaw, vy, yaw_frame_vy = values.T
        outward = np.sign(y)*vy > 0
        inward_heading = y*yaw < 0
        results[str(lo)] = dict(pre_control_age_steps=[lo, hi], rows=len(values),
            mean_abs_lateral_m=float(abs(y).mean()),
            mean_outward_world_velocity_m_s=float((np.sign(y)*vy).mean()),
            mean_outward_yaw_frame_velocity_m_s=float((np.sign(y)*yaw_frame_vy).mean()),
            fraction_world_velocity_outward=float(outward.mean()),
            fraction_heading_inward=float(inward_heading.mean()),
            fraction_world_outward_given_inward_heading=(float(outward[inward_heading].mean()) if inward_heading.any() else None))
    assert sum(v['rows'] for v in results.values()) == audit['recovery_controls']
    write_json_atomic(args.output, dict(status='verified', groups=results,
        evidence={str(p): sha256_file(p) for p in paths}, new_controls=0, optimization_labels_created=0,
        scope='All recorded TRAIN case26 controls across changing policies, including failures and terminal steps. Age groups have survival bias; yaw frame is horizontal, not full body frame. Descriptive kinematics only, not a reward reconstruction or causal test. No pilot or held-out data used.'))


if __name__ == '__main__':
    main()
