"""核验已完成的单项训练，执行 1600 回合原协议和 240 回合航向诊断。"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time

from reference_stairs_eval import ROOT, save, sha
from reference_low_stairs_zero_preservation_train import OUTPUT, directory, sources, preflight, VARIANT, GUARDED_OUTPUT
from reference_low_stairs_guarded_evaluate import read
from reference_low_stairs_guarded_ppo import KEY
from reference_low_stairs_clearance_train import check_algorithm
from reference_low_stairs_resume import STATE_KEY
from reference_low_stairs_evaluate import OUTPUT as PARENT_RUNS, EXPECTED_SHA, verify_run as verify_parent
from reference_low_stairs_trial import CHECKPOINT
from reference_reward_resume import equal_state
from reference_core500_evaluate import same_initial
from reference_low_stairs_a_evaluate import same_configuration

EVALUATION = OUTPUT/'evaluation_10100'
MODEL = 'lowstairs_zero_preservation10100'
GUARDED_RUNS = GUARDED_OUTPUT/'evaluation_10100'


def checked_training():
    import torch
    preflight(1)
    run = directory(1)
    execution, report = read(run/'execution.json'),read(run/'report.json')
    assert execution['status']=='completed' and execution['exit_code']==0 and execution['physical_gpu']==1
    assert report['status']=='passed' and report['max_updates']==report['optimizer_updates']==report['phase_updates']==100
    assert report['cumulative_updates']==10100 and report['environment_steps']==2400
    assert report['interactions']==9830400 and not report['automatic_extension']
    assert report['variant']==VARIANT and report['auxiliary_coefficient']==0.
    assert report['original_parent_sha256']==EXPECTED_SHA and report['parent_sha256']==EXPECTED_SHA
    assert report['implementation_sources']==sources() and report['initial_algorithm_and_curriculum_match_guarded']
    assert [r['update'] for r in report['updates']]==list(range(1,101))
    assert all(r['losses']['capability_preservation']==0. for r in report['updates'])
    assert [r['phase_updates'] for r in report['checkpoints']]==[0,50,100]
    old_cfg = read(GUARDED_OUTPUT/'train_001_100/configuration.json')
    expected_cfg = copy.deepcopy(old_cfg); expected_cfg['agent']['max_iterations']=100
    assert read(run/'configuration.json')==expected_cfg
    parent = torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    old_initial = torch.load(GUARDED_OUTPUT/'train_001_100/model_updates_10000.pt',map_location='cpu',weights_only=True)
    initial = None
    for row in report['checkpoints']:
        checkpoint = Path(row['path']); n=row['phase_updates']
        assert sha(checkpoint)==row['sha256']
        state = torch.load(checkpoint,map_location='cpu',weights_only=True)
        check_algorithm(state,n)
        info=state['infos']
        assert state['iter']==9999+n and info['phase_updates']==n and info['completed_updates']==10000+n
        assert info['trial_budget']==100 and info['variant']==VARIANT and info['parent_sha256']==EXPECTED_SHA
        assert info['validation_only']==(n==0)
        assert info[KEY]['coefficient']==0. and info[KEY]['calls']==n*20
        assert info[KEY]['bank_sha256']==report['bank_sha256'] and info[KEY]['per_group']==128
        curriculum=info[STATE_KEY];levels=curriculum['terrain_levels'];types=curriculum['terrain_types']
        assert levels.shape==types.shape==(4096,) and levels.min()>=0 and levels.max()<10
        assert equal_state(curriculum['env_origins'],curriculum['terrain_origins'][levels,types])
        if n==0:
            initial=state
            assert all(equal_state(state[k],parent[k]) and equal_state(state[k],old_initial[k])
                       for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict'))
            assert equal_state(curriculum,old_initial['infos'][STATE_KEY])
        else:
            assert all(not equal_state(state[k],initial[k]) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict'))
    assert checkpoint==Path(report['output_checkpoint']) and sha(checkpoint)==report['output_sha256']
    assert sha(CHECKPOINT)==EXPECTED_SHA
    return checkpoint,report['output_sha256']


def verify_run(path,kind,profile,digest,model,direction=None,heading=False):
    report,execution,cfg=(read(path/f'{n}.json') for n in ('report','execution','config'))
    assert execution['status']=='completed' and execution['exit_code']==0 and execution['physical_gpu']==1
    assert report['status']=='passed' and report['actor_unchanged'] and not report['probe']
    assert report['model']==model and report['profile']==profile and report['video_env'] is None
    assert report.get('model_sha256',report.get('checkpoint_sha256'))==digest
    n=320 if kind=='turn' else 240
    assert report['episodes']==len(report['records'])==n and {r['env_id'] for r in report['records']}==set(range(n))
    assert report['step_dt']==.02 and cfg['reward_scoring']=='reference_frozen_common'
    if kind=='stairs':
        assert report['direction']==direction and report['height_set']=='low_stairs' and report['action_mode']=='mean'
        assert report['first_episodes_only'] and report['pre_reset_capture']
        assert {r['height'] for r in report['records']}=={0.,.02,.03,.05}
        assert {r['speed'] for r in report['records']}=={.32,.4,.6}
    if heading:
        intervention=read(path/'heading_intervention.json')
        assert intervention['checkpoint_sha256']==digest and intervention['optimizer_updates']==0
        assert intervention['gain']==.5 and intervention['limit_rad_s']==.8 and intervention['diagnostic_only']
        assert intervention['target']=='world +x' and not intervention['lateral_position_feedback']
        assert intervention['source_sha256']==sha(ROOT/'scripts/tools/reference_low_stairs_guarded_heading.py')
    return report


def wait_training():
    deadline=time.monotonic()+7200
    path=directory(1)/'execution.json'
    if not path.exists(): raise RuntimeError('只等待本阶段已启动的训练，不启动或重启训练')
    while True:
        state=read(path)
        if state['status']=='completed':return
        if state['status']!='running' or time.monotonic()>deadline:
            raise RuntimeError('训练未完成或已失败，不重启')
        time.sleep(30)


def main(wait=False):
    if wait:wait_training()
    checkpoint,digest=checked_training()
    jobs=[(f'turn_{p}','turn',p,None,False) for p in ('nominal','training_randomized')]
    jobs += [(f'{d}_{p}','stairs',p,d,False) for p in ('nominal','training_randomized') for d in ('up','down')]
    jobs += [('heading_up_nominal','stairs','nominal','up',True)]
    EVALUATION.mkdir(parents=True,exist_ok=True)
    save(EVALUATION/'plan.json',dict(checkpoint=str(checkpoint),sha256=digest,new_episodes=1840,
        formal_episodes=1600,heading_episodes=240,training=False,physical_gpu=1,model=MODEL,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=sha(__file__)))
    matrix=[]
    for name,kind,profile,direction,heading in jobs:
        dest=EVALUATION/name
        flags=['--model',MODEL,'--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,'--output',str(dest)]
        if heading: script='reference_low_stairs_guarded_heading.py'
        else:
            script='reference_turn_diagnose.py' if kind=='turn' else 'reference_isaac_stairs_diagnose.py'
            flags += ['--profile',profile,'--repeats','20','--gpu','1','--minimum-free-mib','10240']
            if kind=='stairs':flags += ['--direction',direction,'--height-set','low_stairs','--action-mode','mean']
        if not dest.exists():
            print('EVALUATE',name,flush=True)
            subprocess.run([sys.executable,str(ROOT/'scripts/tools'/script),*flags],cwd=ROOT,check=True)
        result=verify_run(dest,kind,profile,digest,MODEL,direction,heading)
        parent=PARENT_RUNS/('up_nominal' if heading else name)
        verify_parent(parent,kind,profile,direction)
        for old in (parent,GUARDED_RUNS/name):
            assert same_initial(dest,old) and same_configuration(read(dest/'config.json'),read(old/'config.json'))
        assert sha(checkpoint)==digest
        matrix.append(dict(name=name,kind=kind,profile=profile,direction=direction,heading=heading,
            candidate=str(dest),parent=str(parent),guarded=str(GUARDED_RUNS/name),episodes=result['episodes'],
            initial_equal=True,configuration_equal=True))
        save(EVALUATION/'matrix.json',matrix)
    assert sum(r['episodes'] for r in matrix)==1840
    save(EVALUATION/'complete.json',dict(status='passed',status_scope='运行完成，行为验收另行分析',
        checkpoint=str(checkpoint),checkpoint_sha256=digest,formal_episodes=1600,heading_episodes=240,
        matrix=matrix,automatic_training_extension=False))
    print('ZERO PRESERVATION EVALUATION COMPLETE: 1840 frozen episodes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-training',action='store_true')
    main(parser.parse_args().wait_for_training)
