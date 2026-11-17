"""Recompute trained adapter updates from saved inputs on CPU only."""
import sys

import numpy as np
import torch

from reference_online_c1b_reset_api import OUTPUT,DOCS,registration
from reference_stairs_eval import ROOT,save,sha
from reference_online_targets import read
from reference_online_c1b_recover import recovered_completed

sys.path.insert(0,str(ROOT/'source/rl_training'))
from rl_training.ea_lpl.c1b import Session,digest,exploration_table


def main():
    registration()
    directory=OUTPUT/'train'
    report=recovered_completed(directory)
    episodes=read(directory/'episodes.json')
    specs=read(directory/'config.json')['specs']
    final=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
    with np.load(directory/'trace.npz') as archive:
        data={key:archive[key] for key in ('phase','episode','features','code','decision','age','score')}
    noise=exploration_table()
    errors=dict(code=0.,discounted_return=0.,baseline=0.,loss=0.,gradient_norm=0.)
    session_results=[]
    for i,spec in enumerate(specs):
        head=Session(spec['arm']=='Updated')
        records={r['episode']:r for r in episodes['records'] if r['env_id']==i}
        for episode in sorted(set(data['episode'][:,i].tolist())):
            head.begin()
            ticks=np.flatnonzero((data['phase'][:,i]==1)&(data['episode'][:,i]==episode))
            for tick in ticks:
                if data['decision'][tick,i]:
                    code=head.choose(data['features'][tick,i],noise[spec['session'],episode,data['age'][tick,i]//10])
                    errors['code']=max(errors['code'],float(np.abs(code-data['code'][tick,i]).max()))
                head.rewards.append(float(data['score'][tick,i]))
            if episode in records:
                result=head.finish(allow_update=True)
                record=records[episode]
                for key in ('discounted_return','loss','gradient_norm'):
                    if key in result: errors[key]=max(errors[key],abs(result[key]-record[key]))
                errors['baseline']=max(errors['baseline'],abs(result['new_baseline']-record['new_baseline']))
                assert result['parameter_sha256']==record['parameter_sha256']
        snapshot=final['heads'][i]
        assert digest(head.model)==snapshot['parameter_sha256']
        assert head.baseline==snapshot['baseline'] and head.updates==snapshot['updates']
        if head.optimizer:
            saved=snapshot['optimizer']['state']; replayed=head.optimizer.state_dict()['state']
            for key,values in saved.items():
                for field,value in values.items():
                    assert torch.equal(value,replayed[key][field])
        session_results.append(dict(session=spec['session'],arm=spec['arm'],updates=head.updates,
                                    final_parameter_sha256=digest(head.model)))
    assert max(errors.values())<1.e-7
    result=dict(status='passed',recomputed_from_saved_features_noise_and_scores=True,max_abs_errors=errors,
        final_parameters_and_adam_state_exact=True,sessions=session_results,
        original_online_updates=report['adapter_updates'],simulation_steps_added=0,
        online_updates_added=0,cpu_recomputed_updates=sum(s['updates'] for s in session_results),
        caveat='CPU recomputation of existing updates is not additional online learning or independent physical evidence',
        trace_sha256=sha(directory/'trace.npz'),final_sha256=sha(directory/'final.pt'),source_sha256=sha(ROOT/'scripts/tools/reference_online_c1b_learning_replay.py'))
    save(DOCS/'LEARNING_REPLAY.json',result)
    print(result['status'],errors,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
