"""CPU-only allocation visualization and preparation evidence; no simulator."""
import json
from pathlib import Path

import numpy as np
import torch

from reference_stair_allocation import parent_state, allocate, OUTPUT, DOCS, DIGEST, PARENT, GROUPS
from reference_low_stairs_resume import STATE_KEY
from reference_stairs_eval import save, sha


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    DOCS.mkdir(parents=True, exist_ok=True)
    state = parent_state()['infos'][STATE_KEY]
    levels, original = state['terrain_levels'], state['terrain_types']
    audits = {}
    matrices = {}
    for variant in ('A','B'):
        types, audit = allocate(original, levels, variant)
        save(DOCS/f'allocation_{variant}.json', audit)
        audits[variant] = audit
        matrices[variant] = torch.bincount(levels*20+types, minlength=200).reshape(10,20).numpy()
    fig, axes = plt.subplots(3,1,figsize=(14,10),gridspec_kw=dict(height_ratios=[1,1,.9]))
    upper = max(int(m.max()) for m in matrices.values())
    for ax,variant in zip(axes[:2],('A','B')):
        im = ax.imshow(matrices[variant], origin='lower', aspect='auto', vmin=0, vmax=upper, cmap='YlGnBu')
        for (row,col),value in np.ndenumerate(matrices[variant]):
            ax.text(col,row,str(value),ha='center',va='center',fontsize=7,color='white' if value > upper*.58 else 'black')
        ax.set_title(f'{variant}: environments assigned to each unchanged map tile (checkpoint levels)')
        ax.set_xticks(range(20)); ax.set_xlabel('Terrain column'); ax.set_ylabel('Difficulty row')
        for boundary in (3.5,7.5,11.5,13.5,15.5,17.5): ax.axvline(boundary,color='red',lw=.7)
    names = list(GROUPS); x = np.arange(len(names))
    for variant,offset in [('A',-.18),('B',.18)]:
        bars = axes[2].bar(x+offset,[audits[variant]['counts_after'][n] for n in names],.36,label=variant)
        axes[2].bar_label(bars,padding=2,fontsize=9)
    axes[2].set_xticks(x,names); axes[2].set_ylim(0,930); axes[2].set_ylabel('Environment count'); axes[2].legend()
    fig.suptitle('Fixed-map allocation plan: 4096 environments; no new rollout data',fontsize=14)
    fig.tight_layout(); fig.savefig(DOCS/'allocation.png',dpi=160); fig.savefig(DOCS/'allocation.pdf'); plt.close(fig)
    failed = OUTPUT/'A/integration'
    execution = json.loads((failed/'execution.json').read_text())
    error = json.loads((failed/'error.json').read_text())
    save(DOCS/'preparation.json', dict(status='awaiting_approved_restart_after_startup_fix', parent_sha256=DIGEST,
        parent_unchanged=sha(PARENT)==DIGEST, formal_environment_steps=0, optimizer_updates=0,
        frozen_evaluation_episodes=0, failed_startup=dict(path=str(failed), execution=execution, error=error),
        fix='move sensor/collider audit after make_resumed_runner performs the single restored initial reset',
        simulator_restarted=False, simulation_fix_verified=False,
        files=[dict(path=str(p),sha256=sha(p)) for p in sorted(Path(__file__).parent.glob('reference_stair_allocation*.py'))]))


if __name__ == '__main__': main()
