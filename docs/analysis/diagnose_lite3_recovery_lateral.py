"""Describe all TRAIN case26 lateral trajectories after the frozen gate audit.

This is a post-evaluation diagnostic. It does not change acceptance or select a
checkpoint, slot, seed, or training label.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('control-run', 'trial-run', 'control-audit', 'trial-audit', 'output', 'figure'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    evidence, results, series = {}, {}, {}
    initial_hashes = []
    for label in ('control', 'trial'):
        root = Path(getattr(args, label+'_run'))
        audit_path = Path(getattr(args, label+'_audit'))
        audit = json.loads(audit_path.read_text())
        report = json.loads((root/'report.json').read_text())
        assert report['status'] == 'complete' and audit['status'] == 'verified'
        assert audit['evidence'][str(root/'report.json')] == sha256_file(root/'report.json')
        for path in (root/'report.json', audit_path):
            evidence[str(path)] = sha256_file(path)
        phase = report['phases']['recovery-trial']
        initial_hashes.append(phase['initial_sha256'])
        folder = root/'recovery-trial'
        with np.load(folder/'initial.npz', allow_pickle=False) as initial:
            ids = np.flatnonzero(initial['case'] == 26)
            assert len(ids) == 24
            origins = initial['origins'][ids, 1].copy()
            start = initial['root'][ids, 1]-origins
        evidence[str(folder/'initial.npz')] = sha256_file(folder/'initial.npz')
        pieces = []
        for path in sorted((folder/'trajectory').glob('chunk-*.npz')):
            assert sha256_file(path) == audit['phases']['recovery-trial']['trace_sha256'][path.name]
            with np.load(path, allow_pickle=False) as d:
                y = d['post_root_state'][:, ids, 1]-origins
                pieces.append(np.where(d['active'][:, ids], y, np.nan))
        lateral = np.concatenate(pieces)
        episodes = {e['env_id']: e for e in phase['episodes']}
        rows = []
        for j, env_id in enumerate(ids):
            y = lateral[np.isfinite(lateral[:, j]), j]
            peak_step = int(np.argmax(np.abs(y)))
            episode = episodes[int(env_id)]
            assert len(y) == episode['steps']
            np.testing.assert_allclose(np.abs(y).max(), episode['max_body_lateral_m'], atol=1e-6)
            rows.append(dict(env_id=int(env_id), success=episode['recovery_success'],
                initial_lateral_m=float(start[j]), peak_absolute_lateral_m=float(np.abs(y).max()),
                peak_time_s=(peak_step+1)*.02, final_absolute_lateral_m=float(abs(y[-1])),
                max_displacement_from_initial_m=float(np.abs(y-start[j]).max())))
        worst = max(rows, key=lambda row: row['peak_absolute_lateral_m'])
        results[label] = dict(slots=24, unique_train_sources=1, episodes=rows, worst=worst,
            mean_final_absolute_lateral_m=float(np.mean([r['final_absolute_lateral_m'] for r in rows])),
            maximum_displacement_from_initial_m=max(r['max_displacement_from_initial_m'] for r in rows))
        series[label] = lateral
    assert len(set(initial_hashes)) == 1
    write_json_atomic(args.output, dict(status='verified', results=results, evidence=evidence,
        scope='Read-only diagnosis of every TRAIN case26 slot. All repetitions share one source. No new physics or training labels; original gates unchanged.'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)
    titles = {'control': 'PPO self-mean targets', 'trial': 'PPO demo correction targets'}
    for ax, label in zip(axes, ('control', 'trial')):
        y = series[label]
        time = np.arange(1, len(y)+1)*.02
        ax.plot(time, np.abs(y), color='#2A8C82', alpha=.25, linewidth=.8)
        ax.axhline(.70, color='#B44747', linestyle='--', label='Original peak gate: 0.70 m')
        ax.axvline(.24, color='#65758B', linestyle=':', label='Training correction window: 0.24 s')
        ax.set(title=titles[label], xlabel='Time since recovery reset (s)', ylabel='Absolute lateral error (m)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('All 24 TRAIN case26 slots: recovery completion does not imply return to route')
    target = Path(args.figure)
    for suffix in ('.png', '.svg'):
        fig.savefig(target.with_suffix(suffix), dpi=160)
    svg = target.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
