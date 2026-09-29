"""Plot sealed history/original critics on the completed fresh confirmation only."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_training.online_adaptation.value_diagnostics import make_value_network, time_features
from scripts.reinforcement_learning.online_adaptation.confirm_value_history import (
    checked_path, load_episode, load_protocol, predict_model, read_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--fit-lock', required=True)
    parser.add_argument('--confirmation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol, _ = load_protocol(args.protocol)
    lock = read_json(args.fit_lock)
    fitted = read_json(checked_path(lock['fit_manifest']))
    confirmation = read_json(args.confirmation)
    if (confirmation['status'] != 'complete'
            or confirmation['fit_manifest_sha256'] != lock['fit_manifest']['sha256']):
        raise ValueError('completed confirmation with the sealed fit required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for row, terrain in enumerate(protocol['terrains']):
        seed = terrain['seed']
        root = Path(terrain['confirmation_session_dir'])
        state = read_json(root / 'session_summary.json')
        manifest = read_json(root / 'ledger/manifest.json')
        bundle = torch.load(checked_path(fitted['terrains'][str(seed)]['bundle']),
                            map_location='cpu', weights_only=False)
        original, history = make_value_network(48), make_value_network(50)
        original.load_state_dict(bundle['original_model'])
        history.load_state_dict(bundle['history_model'])
        for col, trial_id in enumerate(terrain['confirmation_trials']):
            episode = load_episode(root, state['trials'][trial_id], manifest, protocol['gamma'])
            x, y = episode['x'], episode['y']
            ax = axes[row, col]
            t = np.arange(len(y)) * protocol['candidate']['step_dt']
            ax.plot(t, y, color='black', lw=1.5, label='Complete-task MC')
            ax.plot(t, predict_model(original)(x), color='#c96b35', lw=.9, alpha=.8, label='Original 48D critic')
            ax.plot(t, predict_model(history, history=True)(x), color='#238b63', lw=.9, alpha=.8, label='History 50D critic')
            ax.plot(t, time_features(x) @ np.asarray(bundle['time_coefficients']),
                    color='#3478bb', lw=1.1, ls='--', label='Time-only baseline')
            ax.set_title(f"Terrain {seed}, new {trial_id}: {episode['metadata']['stop_reason']}", fontsize=10)
            ax.set_xlabel('Elapsed time (s)')
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel('Discounted reward return')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Fresh confirmation: sealed models, no refitting or checkpoint selection')
    fig.tight_layout()
    fig.savefig(output / 'confirmation_value.png', dpi=160)
    svg = output / 'confirmation_value.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
