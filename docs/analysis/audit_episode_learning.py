"""Reproduce learned search updates from charged episodes and audit frozen tests."""
import argparse
from pathlib import Path

from docs.analysis.audit_c0_headroom import audit_child, check_pairs
from docs.analysis.audit_learning_replication import retention_pass, gait
from docs.analysis.audit_sac_huber_coverage import read
from docs.analysis.run_episode_learning import costs, outcome
from rl_training.online_adaptation.episode_search import EpisodeSearch
from rl_training.online_adaptation.interaction_budget import digest
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def replication_gate(terrains):
    if len(terrains)!=2 or not all(t['complete'] for t in terrains.values()):return False
    if not all(t['retention_pass'] and t['evaluation_safe'] and t['uncertain_steps']==0 for t in terrains.values()):return False
    if not all(t['candidate_completions']>=4 for t in terrains.values()):return False
    gains=[t['candidate_completions']-t['base_completions'] for t in terrains.values()]
    return max(gains)>=1 and min(gains)>=-1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    p=read(args.protocol);root,out=Path(p['output']),Path(args.output)
    assert p['schema']=='lite3_c0_episode_learning_v1' and args.seed in p['learner_seeds']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','protocol_sha256':sha256_file(args.protocol),'seed':args.seed,'terrains':{},
            'actual_control_transitions':0,'goal_complete':False,'limits':p['limits']}
    for terrain in p['terrains']:
        group=root/f'seed-{args.seed}-terrain-{terrain}';state=read(group/'execution.json')
        assert state['status'] in ('complete','failed') and state['protocol_sha256']==report['protocol_sha256']
        charged=costs(group);assert charged==state['costs']
        total=sum(charged[k] for k in ('training','evaluation','retention'))
        assert charged['training']<=p['training_max_steps_per_session'] and total<=p['max_steps_per_terrain_seed']
        report['actual_control_transitions']+=total
        item={'complete':False,'costs':charged,'uncertain_steps':charged['uncertain_steps'],
              'execution_sha256':sha256_file(group/'execution.json'),'training':[]}
        report['terrains'][str(terrain)]=item
        if state['status']=='failed':
            item['reason']='failed session retained, no qualification'
            write_json_atomic(out/'report.json',report);continue
        replay=EpisodeSearch(args.seed,p['code_dim'])
        sources={state['source_sha256']}
        for generation in range(p['generations']):
            pending=replay.ask();results=[];trials=[]
            for i,attempt in enumerate(pending):
                stage=f'g{generation:02d}-c{i:02d}';directory=group/stage
                child=audit_child(directory,p,terrain,args.seed+100000*(generation+1),attempt['code'],1)
                sources.add(child['source_sha256']);item['training'].append(child)
                results.append(outcome(directory,attempt['id']));trials.append(child['trials'])
            for other in trials[1:]:check_pairs(trials[0],other)
            replay.tell(results)
        assert replay.state_dict()==state['search_state']
        final=read(group/'final_code.json')
        assert sha256_file(group/'final_code.json')==state['final_code_sha256']
        assert final['search_state_sha256']==digest(replay.state_dict()) and final['code']==replay.mean.tolist()
        assert final['protocol_sha256']==report['protocol_sha256'] and final['source_sha256']==state['source_sha256']
        item['final_code']=final['code'];item['search_generations']=replay.generation
        item['distribution_updates']=sum(bool(h['elite_ids']) for h in replay.history)
        children={role:audit_child(group/role,p,terrain,args.seed,final['code'] if role=='candidate' else None,7,
                    expected_phases={'evaluation':5,'retention':2}) for role in ('candidate','base')}
        item['children']=children
        phases={phase:{role:[t for t in child['trials'] if t['phase']==phase] for role,child in children.items()}
                for phase in ('evaluation','retention')}
        for phase in phases:check_pairs(phases[phase]['candidate'],phases[phase]['base'])
        for role,child in children.items():
            sources.add(child['source_sha256'])
            item[f'{role}_completions']=sum(t.get('completion',False) for t in phases['evaluation'][role])
            item[f'{role}_gait']=[gait(group/role,{'trial_id':t['id'],'request_rms':t['requested_residual_rms']})
                                  for t in phases['evaluation'][role] if t['status']=='complete']
        assert len(sources)==1
        flat=phases['retention']
        item['retention_pass']=all(len(rows)==2 and all(t['status']=='complete' for t in rows) for rows in flat.values()) and retention_pass(flat['candidate'],flat['base'])
        item['evaluation_safe']=all(t.get('safety_violation',True)==False for rows in phases['evaluation'].values() for t in rows)
        item['complete']=all(c['all_expected_trials_complete'] for c in [*item['training'],*children.values()])
        write_json_atomic(out/'report.json',report)
    assert report['actual_control_transitions']<=p['max_steps_first_seed']
    assert sha256_file(p['base_checkpoint'])==p['base_sha256']
    report.update(status='complete',replication_gate=replication_gate(report['terrains']))
    write_json_atomic(out/'report.json',report)
    print({k:v for k,v in report.items() if k!='terrains'})


if __name__=='__main__':main()
