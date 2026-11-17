"""Exercise native manager methods on CPU, with a synthetic physics clock.

No Isaac application/PhysX/GPU starts here; these are interface tests and cannot
establish physical performance or substitute for the registered smoke runs.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_c1b_native_reset_contract import factory  # Actual installed native method fixture.
import reference_online_c1a_collect as collector
import reference_online_c1a_env as environment
import reference_online_c1a_audit as auditor
from reference_online_c1a import rows
from reference_stairs_eval import save,sha
from reference_online_targets import read
from rl_training.ea_lpl.c1a import Session


@pytest.mark.parametrize('kind,steps',[('smoke',600),('train',1300),('target',1050)])
def test_complete_collection_and_replay_with_native_managers(factory,monkeypatch,tmp_path,kind,steps):
    specs = rows(kind,smoke_index=2)[:2]
    name = 'target_20000_00' if kind=='target' else kind
    args = NS(kind=kind,name=name,part=0,smoke_index=2,mark=20000 if kind=='target' else 0,steps=steps,output=tmp_path/name)
    args.output.mkdir()
    monkeypatch.setattr(collector,'OUTPUT',tmp_path)
    if kind=='target':
        train = tmp_path/'train'; train.mkdir()
        frozen,updated = Session(False),Session(True)
        updated.b[:] = torch.tensor([.3,-.2])
        torch.save(dict(heads=[frozen.snapshot(),updated.snapshot()],control_steps=20000),train/'at_20000.pt')
    env = factory(args=args,specs=specs)
    env.__class__ = environment.Environment
    env.robot.data.root_lin_vel_b = torch.zeros(2,3)
    # Synthetic motion only exposes book-keeping/learning errors. It is not a
    # model of the robot's dynamics or evidence for improvement on stairs.
    def advance(**kwargs):
        if kind=='train':
            movement = (.001+.001*env.base.action_manager.action[:,1])*torch.as_tensor(env.policy_active)
            env.robot.data.root_state_w[:,0] += movement
            env.base.termination_manager.forced[:] = torch.as_tensor(env.policy_active & (env.policy_age>=np.array([100,150])))
    env.base.sim.step = advance
    monkeypatch.setattr(environment,'Environment',lambda *args:env)
    monkeypatch.setattr(collector,'rows',lambda *args,**kwargs:specs)
    monkeypatch.setattr(collector,'MARKS',(400,800,1200))
    monkeypatch.setattr(auditor,'MARKS',(400,800,1200))
    model = tmp_path/'synthetic_model'
    model.write_bytes(b'CPU contract fixture, not a robot model')
    monkeypatch.setattr(collector,'MODEL_PATHS',{'B':model})
    monkeypatch.setattr(collector,'HASHES',{'B':sha(model)})
    save(args.output/'config.json',dict(specs=specs))
    collector.collect(args)
    report = read(args.output/'report.json')
    assert report['control_steps']==steps and report['interactions']==2*steps
    if kind=='train':
        assert report['session_updates']==[0,1]
        final = torch.load(args.output/'final.pt',weights_only=True)
        assert final['heads'][1]['active'] is not None
        assert len(final['heads'][1]['batch'])==2
        assert final['heads'][1]['b'].abs().max()>0
    elif kind=='smoke':
        assert report['adapter_updates']==0
        assert report['counts']['waiting'][0]>=50
    else:
        assert report['adapter_updates']==0 and report['complete_episodes']==2
        final = torch.load(args.output/'final.pt',weights_only=True)
        assert torch.equal(final['heads'][1]['b'],updated.b)
    monkeypatch.setattr(auditor,'OUTPUT',tmp_path)
    monkeypatch.setattr(auditor,'DOCS',tmp_path/'audit')
    monkeypatch.setattr(auditor,'source_manifest',lambda:{})
    monkeypatch.setattr(auditor,'completed',lambda name:report)
    save(tmp_path/'lock.json',{})
    result = auditor.audit(name)
    assert result['parameters_adam_batches_and_snapshots_exact']
    assert max(result['replay_errors'].values())<=1.e-6
    if kind=='train':
        final['heads'][1]['m'][0] += 1.
        torch.save(final,args.output/'final.pt')
        with pytest.raises(AssertionError): auditor.audit(name)
    if kind=='target':
        updated.b[:] = 0.
        torch.save(dict(heads=[frozen.snapshot(),updated.snapshot()],control_steps=20000),train/'at_20000.pt')
        with pytest.raises(AssertionError): auditor.audit(name)
