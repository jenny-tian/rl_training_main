"""Plot measured post-action frozen paths for the locked schedule comparison."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from docs.analysis.audit_sac_huber_coverage import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    root, output = Path(protocol['output']), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {'reference':'#3478bb','actor025':'#c35b37'}
    for ax, seed in zip(axes, protocol['terrains']):
        for arm in protocol['arms']:
            state = read(root/f'{arm}-{seed}/session_summary.json')
            trials = [t for t in state['trials'].values() if t['phase']=='evaluation' and t['status']=='complete']
            if len(trials)!=1:
                raise ValueError('one completed evaluation required')
            path = Path(trials[0]['output'])
            summary, manifest = read(path/'episode_summary.json'), read(path/'trajectory/manifest.json')
            xs, ys = [], []
            for chunk in manifest['chunks']:
                with np.load(path/'trajectory'/chunk['file']) as data:
                    xs.append(data['base_x'].reshape(-1)); ys.append(data['base_y'].reshape(-1))
            x = np.concatenate(xs)-summary['initial_x']
            y = np.concatenate(ys)-summary['initial_y']
            ax.plot(x, y, color=colors[arm], label=f"{arm}: {summary['task_stop_reason']}")
            ax.scatter(x[-1], y[-1], color=colors[arm], marker='x', s=55)
        limit = summary['allowed_lateral_deviation_m']
        ax.axhline(limit, color='gray', ls='--', lw=.8)
        ax.axhline(-limit, color='gray', ls='--', lw=.8)
        ax.axvline(summary['completion_target_x']-summary['initial_x'], color='black', ls=':', lw=1)
        ax.scatter([0], [0], color='black', s=20)
        ax.set(title=f'rough_4cm, terrain {seed}', xlabel='Forward displacement (m)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='lower left')
    axes[0].set_ylabel('Lateral displacement (m)')
    fig.suptitle('Frozen body paths: one matched start per terrain; crosses mark endpoints')
    fig.text(.5, .01, 'Lateral qualification checks body and feet; the plotted paths show the body center.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(output/'path_comparison.png', dpi=160)
    svg = output/'path_comparison.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':
    main()
