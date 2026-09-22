"""Plot only verified, descriptive development statistics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'evidence/paired_five_seeds.json').read_text())['aggregate']
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'path', 'pdf.fonttype': 42})
names = ['冻结 PPO', 'LoRA-PPO', 'Residual SAC']
keys = ['base_frozen', 'lora_ppo', 'residual_sac_control_step']
colors = ['#919ba5', '#315a7d', '#7399a9']
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
for ax, field, ylabel, ymax in [(axs[0], 'mean_max_forward_displacement_m', '平均最大前进距离 / m', 10),
                                (axs[1], 'terminated_runs', '发生终止的运行数 / 5', 6)]:
    values = [data[k][field] for k in keys]
    bars = ax.bar(range(3), values, color=colors, width=.58)
    ax.set_xticks(range(3), names, fontproperties=font)
    ax.set_ylabel(ylabel, fontproperties=font)
    ax.set_ylim(0, ymax)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
    for bar, value in zip(bars, values):
        label = f'{value:.3f}' if field.startswith('mean_') else f'{value}/5'
        ax.text(bar.get_x()+bar.get_width()/2, value+.12, label, ha='center', va='bottom')
axs[1].set_yticks(range(6))
for ext in ['png', 'svg', 'pdf']:
    fig.savefig(Path(__file__).parent / f'figure3_development_results.{ext}', dpi=220)

svg = Path(__file__).parent / "figure3_development_results.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
