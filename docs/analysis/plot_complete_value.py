"""Render fixed heldout episodes; does not fit or select a critic."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from rl_training.online_adaptation.value_diagnostics import complete_episode_returns,make_value_network,time_features

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--analysis',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.analysis);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
report=json.loads((root/'report.json').read_text())
if report['status']!='complete':raise ValueError('completed fixed-budget analysis required')
protocol=json.loads((root/'protocol.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,7),sharey=True)
for row,terrain in enumerate(protocol['terrains']):
 seed=terrain['seed'];model=make_value_network(48)
 model.load_state_dict(torch.load(root/f'critic-{seed}.pt',map_location='cpu',weights_only=False)['model']);model.eval()
 result=report['terrains'][str(seed)]
 for col,trial in enumerate(terrain['validation_trials']):
  path=Path(terrain['session_dir'])/trial/'trajectory'
  manifest=json.loads((path/'manifest.json').read_text())
  chunks=[]
  for chunk in manifest['chunks']:
   with np.load(path/chunk['file']) as d:chunks.append({k:np.array(d[k]) for k in ('reward','learning_terminated','learning_truncated','collector_cutoff','critic_observation')})
  arrays={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
  y=complete_episode_returns(arrays['reward'],arrays['learning_terminated'],arrays['learning_truncated'],arrays['collector_cutoff'],gamma=protocol['gamma'])
  x=arrays['critic_observation'].reshape(len(y),48)
  with torch.no_grad():pred=model(torch.as_tensor(x,dtype=torch.float32)).numpy().reshape(-1)
  time_pred=time_features(x,protocol['time_degree'])@np.array(result['time_coefficients'])
  ax=axes[row,col];t=np.arange(len(y))*.02
  ax.plot(t,y,color='black',lw=1.4,label='Complete-task MC')
  ax.plot(t,time_pred,color='#3478bb',lw=1.2,label='Time-only baseline')
  ax.plot(t,pred,color='#d0603b',lw=1.,alpha=.85,label='Learned critic')
  ax.set_title(f'Terrain {seed}, heldout {trial}');ax.set_xlabel('Elapsed time (s)');ax.grid(alpha=.2)
  if col==0:ax.set_ylabel('Discounted reward return')
axes[0,0].legend(fontsize=8)
fig.suptitle('Frozen-base value diagnostic: two heldout episodes per terrain')
fig.tight_layout()
fig.savefig(output/'heldout_value.svg');fig.savefig(output/'heldout_value.png',dpi=160)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for seed,r in report['terrains'].items():
 axes[0].plot([p['epoch'] for p in r['training_curve']],[p['training_rmse'] for p in r['training_curve']],label=f'Terrain {seed}')
 labels=['constant','time_only','initial_critic','learned_critic']
 offset=-.18 if seed=='101' else .18
 axes[1].bar(np.arange(4)+offset,[r['metrics'][k]['validation']['pooled']['normalized_rmse'] if r['metrics'][k]['validation']['pooled']['normalized_rmse'] is not None else np.nan for k in labels],width=.36,label=f'Terrain {seed}')
axes[0].set(xlabel='Fixed training epoch',ylabel='Training RMSE');axes[0].grid(alpha=.2);axes[0].legend()
axes[1].set_xticks(range(4),['Constant','Time only','Initial','Learned']);axes[1].set_ylabel('Heldout normalized RMSE');axes[1].axhline(1,color='gray',ls='--',lw=.8);axes[1].legend()
fig.tight_layout();fig.savefig(output/'fit_and_error.svg');fig.savefig(output/'fit_and_error.png',dpi=160)

# Matplotlib emits trailing spaces in SVG paths; keep generated text diff-clean.
for name in ('heldout_value.svg', 'fit_and_error.svg'):
 path=output/name
 path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
