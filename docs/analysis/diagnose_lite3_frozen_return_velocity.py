"""Compare complete frozen recovery trajectories after gates, without fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control-run', 'trial-run', 'control-audit', 'trial-audit', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    evidence, results, initial_hashes = {}, {}, []
    episodes = {}
    bins = [(0, 12), (12, 50), (50, 100), (100, 200), (200, None)]
    for label in ('control', 'trial'):
        root = Path(getattr(args, label+'_run'))
        audit_path = Path(getattr(args, label+'_audit'))
        report = json.loads((root/'report.json').read_text())
        audit = json.loads(audit_path.read_text())
        assert report['status'] == 'complete' and audit['status'] == 'verified'
        assert audit['evidence'][str(root/'report.json')] == sha256_file(root/'report.json')
        episodes[label] = {
            'ordinary_rough': {e['env_id']: e for e in report['phases']['course-trial']['episodes'] if e['terrain'] != 'flat'},
            'recovery_case26': {e['env_id']: e for e in report['phases']['recovery-trial']['episodes'] if e['case'] == 26}}
        for p in (root/'report.json', audit_path):
            evidence[str(p)] = sha256_file(p)
        folder = root/'recovery-trial'
        initial_hashes.append(report['phases']['recovery-trial']['initial_sha256'])
        with np.load(folder/'initial.npz', allow_pickle=False) as data:
            slots = np.flatnonzero(data['case'] == 26)
            origins = data['origins'][slots, 1]
            assert len(slots) == 24
        evidence[str(folder/'initial.npz')] = sha256_file(folder/'initial.npz')
        states, active = [], []
        for path in sorted((folder/'trajectory').glob('chunk-*.npz')):
            assert sha256_file(path) == audit['phases']['recovery-trial']['trace_sha256'][path.name]
            with np.load(path, allow_pickle=False) as data:
                states.append(data['post_root_state'][:, slots])
                active.append(data['active'][:, slots])
        state, active = np.concatenate(states), np.concatenate(active)
        y = state[..., 1] - origins
        vy = state[..., 8]
        w, x, qy, z = np.moveaxis(state[..., 3:7], -1, 0)
        yaw = np.arctan2(2*(w*z+x*qy), 1-2*(qy*qy+z*z))
        yaw_frame_vy = -np.sin(yaw)*state[..., 7] + np.cos(yaw)*vy
        time = np.arange(len(y))[:, None]
        groups = {}
        for lo, hi in bins:
            mask = active & (time >= lo)
            if hi is not None:
                mask &= time < hi
            assert mask.any()
            outward = np.sign(y[mask])*vy[mask]
            groups[str(lo)] = dict(pre_control_age_steps=[lo, hi], rows=int(mask.sum()),
                mean_outward_world_velocity_m_s=float(outward.mean()),
                mean_outward_yaw_frame_velocity_m_s=float((np.sign(y[mask])*yaw_frame_vy[mask]).mean()),
                fraction_world_velocity_outward=float((outward > 0).mean()),
                mean_absolute_lateral_m=float(abs(y[mask]).mean()))
        results[label] = dict(slots=24, unique_train_sources=1, groups=groups)
    assert len(set(initial_hashes)) == 1
    paired = {}
    for condition, expected_count in [('ordinary_rough', 48), ('recovery_case26', 24)]:
        old, new = episodes['control'][condition], episodes['trial'][condition]
        assert old.keys() == new.keys() and len(old) == expected_count
        rows = []
        for slot in sorted(old):
            a, b = old[slot], new[slot]
            assert a['terrain'] == b['terrain'] and a['case'] == b['case']
            rows.append(dict(env_id=slot, control_rms_m=a['body_lateral_rms_m'], trial_rms_m=b['body_lateral_rms_m'],
                control_peak_m=a['max_body_lateral_m'], trial_peak_m=b['max_body_lateral_m'],
                control_success=a['success'], trial_success=b['success']))
        paired[condition] = dict(rows=rows, slots=len(rows),
            rms_improved=sum(r['trial_rms_m'] < r['control_rms_m'] for r in rows),
            peak_improved=sum(r['trial_peak_m'] < r['control_peak_m'] for r in rows))
    write_json_atomic(args.output, dict(status='verified', results=results, evidence=evidence,
        paired_episodes=paired,
        new_controls=0, optimizer_steps=0,
        scope='Post-gate frozen-model diagnostic of all active TRAIN case26 trajectories, including failed terminal steps. Age groups retain survival bias. No labels, reward calibration or checkpoint selection.'))


if __name__ == '__main__':
    main()
