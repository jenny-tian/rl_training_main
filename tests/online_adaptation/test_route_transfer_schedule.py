from docs.analysis.run_route_transfer import commands


def test_transfer_slots_cover_all_pairs_once_and_preserve_budget():
    p={'output':'transfer','terrain_split':'registered-development','terrain_seeds':[1,2,3,4],
       'policies':[{'learning_seed':257,'evaluation_seed':11,'checkpoint':{'path':'first.pt'}},
                   {'learning_seed':269,'evaluation_seed':12,'checkpoint':{'path':'second.pt'}}],
       'base_checkpoint':'original.pt','task_steps':2000,'task_trials':5,'retention_trials':2,'retention_steps':1000}
    keys=set();cost=0
    for slot in (0,1):
        jobs=list(commands(p,slot,f'cuda:{slot}'));assert len(jobs)==16
        for name,phase,cmd in jobs:
            assert (name,phase) not in keys;keys.add((name,phase))
            assert cmd[cmd.index('--target-warmup-steps')+1]=='0'
            assert int(cmd[cmd.index('--terrain-seed')+1]) in p['terrain_seeds'][slot::2]
            role,seed,_=name.split('-')
            expected='original.pt' if role=='base' else ('first.pt' if seed=='257' else 'second.pt')
            assert cmd[cmd.index('--base-checkpoint')+1]==expected
            cost+=10000 if phase=='evaluation' else 2000
    assert len(keys)==32 and cost==192000
