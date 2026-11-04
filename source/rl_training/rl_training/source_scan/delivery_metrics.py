"""Small durable training curves and first-episode diagnostic aggregates."""
import json
import math
import os
import csv
import hashlib
from pathlib import Path


def append_json(path,row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


class TrainingCurves:
    def __init__(self,path):self.path=Path(path);self.reward=0.;self.endings=0;self.steps=0
    def initial(self,training,observations):pass
    def before_step(self,training,step):pass
    def before_update(self,training,update):pass
    def after_step(self,training,step,old_observations,observations,actions,rewards,dones,extras):
        self.reward+=float(rewards.mean());self.endings+=int(dones.sum());self.steps+=1
    def after_update(self,training,update,receipt):
        optimizer=training.algorithm.optimizer
        std=training.algorithm.actor.state_dict()['distribution.log_std_param'].exp().mean().item()
        row=dict(update=update,training_transitions=update*2880,mean_step_reward=self.reward/self.steps,
            episode_endings=self.endings,learning_rate=float(optimizer.param_groups[0]['lr']),
            mean_action_std=std,losses=receipt['losses'],update_seconds=receipt['elapsed_seconds'])
        append_json(self.path,row)
        print(f'[source-numerics] update={update}/90 transitions={update*2880} reward={row["mean_step_reward"]:.6f}',flush=True)
        self.reward=0.;self.endings=0;self.steps=0


class FirstEpisodes:
    def __init__(self,columns):
        self.columns=list(columns);n=len(columns);self.count=[0]*n;self.error_sq=[0.]*n
        self.lateral=[0.]*n;self.roll=[0.]*n;self.pitch=[0.]*n;self.last=[None]*n;self.finished={}

    def add(self,rows,done,terminal):
        ended={} if terminal is None else {i:j for j,i in enumerate(terminal['ids'])}
        for i,row in enumerate(rows):
            if i in self.finished:continue
            self.count[i]+=1;self.error_sq[i]+=row['velocity_error_mps']**2
            self.lateral[i]=max(self.lateral[i],abs(row['lateral_m']))
            self.roll[i]=max(self.roll[i],abs(row['roll_rad']));self.pitch[i]=max(self.pitch[i],abs(row['pitch_rad']))
            self.last[i]=row
            if done[i]:
                j=ended[i]
                self._finish(i,terminal['completed'][j],[k for k,v in terminal['terms'].items() if v[j]])

    def _finish(self,i,completed,reasons):
        self.finished[i]=dict(env_id=i,column=self.columns[i],steps=self.count[i],completed=bool(completed),
            forward_progress_m=self.last[i]['forward_progress_m'],lateral_peak_m=self.lateral[i],
            roll_peak_rad=self.roll[i],pitch_peak_rad=self.pitch[i],
            velocity_error_rms_mps=math.sqrt(self.error_sq[i]/self.count[i]),termination_reasons=reasons)

    def result(self):
        for i in range(len(self.columns)):
            if i not in self.finished:self._finish(i,False,['run_horizon'])
        return [self.finished[i] for i in range(len(self.columns))]


def paired_conditions(old,new):
    if len(old)!=120 or len(new)!=120:raise ValueError('paired diagnostic requires 120 first episodes per actor')
    by=[]
    for column in range(15):
        pairs=[(a,b) for a,b in zip(old,new) if a['column']==column]
        if not pairs or any(a['env_id']!=b['env_id'] or a['column']!=b['column'] for a,b in pairs):
            raise ValueError('paired actors differ in environment/condition assignment')
        row=dict(column=column,pairs=len(pairs),old_completed=sum(a['completed'] for a,b in pairs),
                 new_completed=sum(b['completed'] for a,b in pairs))
        for name in ('forward_progress_m','lateral_peak_m','roll_peak_rad','pitch_peak_rad','velocity_error_rms_mps'):
            row['old_'+name]=sum(a[name] for a,b in pairs)/len(pairs)
            row['new_'+name]=sum(b[name] for a,b in pairs)/len(pairs)
            row['delta_'+name]=row['new_'+name]-row['old_'+name]
        by.append(row)
    return by


def write_paired_artifacts(folder,old,new,conditions):
    """Persist episode-level pairs and a compact per-condition comparison plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder=Path(folder);csv_path=folder/'paired_episodes.csv';plot_path=folder/'paired_conditions.png'
    metrics=('forward_progress_m','lateral_peak_m','roll_peak_rad','pitch_peak_rad','velocity_error_rms_mps')
    fields=['env_id','column','old_steps','new_steps','old_completed','new_completed',
        'old_termination_reasons','new_termination_reasons']
    fields += [f'{side}_{name}' for name in metrics for side in ('old','new','delta')]
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for before,after in zip(old,new):
            row=dict(env_id=before['env_id'],column=before['column'],old_steps=before['steps'],
                new_steps=after['steps'],old_completed=int(before['completed']),new_completed=int(after['completed']),
                old_termination_reasons=json.dumps(before['termination_reasons'],separators=(',',':')),
                new_termination_reasons=json.dumps(after['termination_reasons'],separators=(',',':')))
            for name in metrics:
                row['old_'+name]=before[name];row['new_'+name]=after[name]
                row['delta_'+name]=after[name]-before[name]
            writer.writerow(row)
        stream.flush();os.fsync(stream.fileno())
    columns=[r['column'] for r in conditions]
    fig,axes=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
    axes=axes.ravel()
    axes[0].bar([c-.2 for c in columns],[r['old_completed'] for r in conditions],width=.4,label='Original',color='#4776a8')
    axes[0].bar([c+.2 for c in columns],[r['new_completed'] for r in conditions],width=.4,label='90 updates',color='#d27b43')
    axes[0].set_title('Completed first episodes');axes[0].set_ylabel('Count');axes[0].legend(frameon=False)
    titles=('Forward progress change (m)','Lateral peak change (m)','Roll peak change (rad)',
        'Pitch peak change (rad)','Velocity RMS error change (m/s)')
    for ax,name,title in zip(axes[1:],metrics,titles):
        ax.bar(columns,[r['delta_'+name] for r in conditions],color='#4d8b74')
        ax.axhline(0,color='#333333',linewidth=.8);ax.set_title(title)
    for ax in axes:
        ax.set_xlabel('Terrain condition / number of pairs')
        ax.set_xticks(columns,[str(r['column'])+'\nn='+str(r['pairs']) for r in conditions],fontsize=8)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed-condition paired diagnostics: 90 updates minus original',fontsize=15)
    fig.savefig(plot_path,dpi=150);plt.close(fig)
    return dict(csv=dict(path=csv_path.name,sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),rows=len(old)),
        png=dict(path=plot_path.name,sha256=hashlib.sha256(plot_path.read_bytes()).hexdigest()))
