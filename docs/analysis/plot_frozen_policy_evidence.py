"""Plot audited frozen body paths and descriptive foot contacts without new simulation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docs.analysis.audit_sac_huber_coverage import read, trace_arrays
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    p=read(args.protocol);root=Path(p['output']);audit=read(root/'evaluation_audit.json')
    assert audit['status']=='complete' and audit['protocol_sha256']==sha256_file(args.protocol)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    terrains=p['evaluation_terrains'];fig,axes=plt.subplots(1,len(terrains),figsize=(11,4),squeeze=False,layout='constrained')
    inputs={};first={}
    for ax,terrain in zip(axes[0],terrains):
        for role,color in (('base','#777777'),('candidate','#0868ac')):
            session=read(root/f'{role}-{terrain}'/'session_summary.json')
            trials=[t for t in session['trials'].values() if t['phase']=='evaluation' and t['status']=='complete']
            for index,trial in enumerate(trials):
                path=Path(trial['output']);s=read(path/'episode_summary.json');trace,a,hashes=trace_arrays(path);inputs.update(hashes)
                x=a['base_x'].reshape(-1)-s['initial_x'];y=a['base_y'].reshape(-1)-s['initial_y']
                ax.plot(x,y,color=color,alpha=.7,lw=1.3,label=role if index==0 else None)
                ax.scatter(x[-1],y[-1],color=color,marker='o' if s['task_stop_reason']=='completion' else 'x',s=25)
                if index==0:first[(terrain,role)]=(path,s,trace)
        ax.axhline(1.65,color='#b2182b',ls=':',lw=1);ax.axhline(-1.65,color='#b2182b',ls=':',lw=1)
        ax.axvline(8,color='#238b45',ls='--',lw=1)
        ax.set(title=f'Terrain {terrain}',xlabel='Forward progress (m)',ylabel='Body lateral offset (m)',xlim=(-.2,8.3),ylim=(-1.8,1.8))
        ax.grid(alpha=.2);ax.legend(loc='upper left')
    fig.suptitle('Frozen paired trials: body paths\nCircle = complete; cross = unsuccessful; foot limits also apply',fontsize=12)
    fig.savefig(out/'body_paths.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(len(terrains),2,figsize=(11,5.5),squeeze=False,layout='constrained')
    for row,terrain in enumerate(terrains):
        for col,role in enumerate(('base','candidate')):
            path,s,trace=first[(terrain,role)]
            records=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
            inputs[str(path/'metrics.jsonl')]=sha256_file(path/'metrics.jsonl')
            entered=[r for r in records if r['base_x']>=s['challenge_start_x']]
            ax=axes[row,col]
            if entered:
                start=entered[0]['step'];rows=[r for r in records if 0<=(r['step']-start)*trace['metadata']['step_dt']<=6]
                contacts=np.asarray([r['foot_contact'] for r in rows],dtype=float).T
                duration=(rows[-1]['step']-start+1)*trace['metadata']['step_dt']
                ax.imshow(contacts,origin='lower',aspect='auto',interpolation='nearest',cmap='Blues',vmin=0,vmax=1,extent=(0,duration,-.5,3.5))
            ax.set(title=f'{role}, terrain {terrain}, first registered trial',xlabel='Time after rough-section entry (s)',xlim=(0,6),yticks=range(4),yticklabels=['FL','FR','HL','HR'])
    fig.suptitle('Foot contact timing; dark = contact. Descriptive, not a causal comparison.')
    fig.savefig(out/'foot_contacts.png',dpi=160);plt.close(fig)
    write_json_atomic(out/'manifest.json',{'protocol_sha256':sha256_file(args.protocol),'evaluation_audit_sha256':sha256_file(root/'evaluation_audit.json'),
        'inputs':inputs,'plotter_sha256':sha256_file(__file__),'additional_environment_controls':0,'figures':{name:sha256_file(out/name) for name in ('body_paths.png','foot_contacts.png')},
        'limit':'All registered task paths shown; foot timing uses the first registered trial only and each policy reaches rough terrain at its own time.'})
    print(str(out))


if __name__=='__main__':main()
