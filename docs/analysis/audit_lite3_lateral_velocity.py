"""Independently reconstruct world lateral velocity reward at pre-reset time."""
import argparse
import json
from pathlib import Path

import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'run', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    root = Path(args.run)
    p = json.loads(Path(args.protocol).read_text())
    r = json.loads((root/'report.json').read_text())
    assert r['status'] == 'complete' and r['protocol_sha256'] == sha256_file(args.protocol)
    options = p['lateral_velocity']
    assert p['recovery_resets']['split'] == 'train'
    total = terminal = different = saturated = 0
    cost_sum = 0.
    hashes = {}
    groups = {k: dict(rows=0, cost_sum=0., outward_velocity_sum=0.) for k in ('ordinary', 'early', 'tail')}
    for path in sorted((root/'trajectory').glob('chunk-*.npz')):
        hashes[path.name] = sha256_file(path)
        with np.load(path, allow_pickle=False) as data:
            state, origins, done = data['velocity_root_state'], data['velocity_origins'], data['done'].astype(bool)
            assert state.shape == (*done.shape, 13) and np.isfinite(state).all()
            np.testing.assert_array_equal(state, data['heading_root_state'])
            np.testing.assert_array_equal(origins, data['heading_origins'])
            y = state[..., 1] - origins[..., 1]
            target = np.clip(-options['gain_per_second']*y, -options['maximum_m_s'], options['maximum_m_s'])
            cost = (state[..., 8]-target)**2
            np.testing.assert_allclose(data['velocity_target'], target, atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(data['velocity_cost'], cost, atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(data['velocity_reward'], options['weight']*cost*.02, atol=1e-6, rtol=1e-5)
            expected_route = np.clip(y/1.65, -2, 2)
            np.testing.assert_allclose(data['next_policy_observation'][..., 9][~done], expected_route[~done], atol=1e-6, rtol=1e-5)
            different += int((abs(data['next_policy_observation'][..., 9][done]-expected_route[done]) > 1e-3).sum())
            ordinary = data['recovery_case'] < 0
            early = ~ordinary & (data['episode_steps'] < 12)
            for name, mask in [('ordinary', ordinary), ('early', early), ('tail', ~ordinary & ~early)]:
                group = groups[name]
                group['rows'] += int(mask.sum())
                group['cost_sum'] += float(cost[mask].sum())
                group['outward_velocity_sum'] += float((np.sign(y[mask])*state[..., 8][mask]).sum())
            total += done.size
            terminal += int(done.sum())
            saturated += int((abs(target) >= options['maximum_m_s']).sum())
            cost_sum += float(cost.sum())
    assert total == r['ledger']['charged'] == r['ledger']['confirmed']
    assert r['ledger']['uncertain_transitions'] == 0
    assert terminal > 0 and different > 0 and saturated > 0 and all(v['rows'] > 0 for v in groups.values())
    write_json_atomic(args.output, dict(status='verified', controls=total, terminal_rows=terminal,
        pre_reset_vs_returned_differences=different, target_saturated_rows=saturated,
        mean_cost=cost_sum/total, groups=groups, trace_sha256=hashes,
        evidence={str(path): sha256_file(path) for path in (Path(args.protocol), root/'report.json')},
        scope='World-Y, reward-time/pre-reset geometry, actual weight and 0.02s charge reconstructed independently; all ordinary/early/tail controls included.'))


if __name__ == '__main__':
    main()
