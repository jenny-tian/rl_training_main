import torch
import pytest

from rl_training.recovery_teacher import bank_indices
from rl_training.policy_flat_anchor import FrozenFlatMean


def test_recovery_split_excludes_flat_and_never_leaks_validation_states():
    types=torch.tensor([0,1,2,3,5,6]);ids=torch.arange(6);counts=torch.zeros(6,dtype=torch.long)
    terrain=torch.tensor([101,101,101,107,107,107]);validation=torch.tensor([False,False,True,False,False,True])
    train=bank_indices(types,ids,counts,terrain,validation,split='train',mixed=False)
    held=bank_indices(types,ids,counts,terrain,validation,split='validation',mixed=False)
    assert train[:2].tolist()==[-1,-1] and held[:2].tolist()==[-1,-1]
    assert not validation[train[2:]].any() and validation[held[2:]].all()
    assert torch.equal(terrain[train[2:]],torch.tensor([101,101,107,107]))
    first=bank_indices(types,ids,counts,terrain,validation,split='train',mixed=True)
    second=bank_indices(types,ids,counts+1,terrain,validation,split='train',mixed=True)
    assert torch.equal((first[2:]>=0),~(second[2:]>=0))
    exposed=set()
    for count in range(8):
        row=bank_indices(torch.tensor([2]),torch.tensor([2]),torch.tensor([count]),
            terrain,validation,split='train',mixed=True)
        if row[0]>=0:exposed.add(int(row[0]))
    assert exposed=={0,1}
    with pytest.raises(ValueError):
        bank_indices(types,ids,counts,terrain,validation,split='test',mixed=False)


def test_route_anchor_retains_route_inputs_instead_of_stripping_them():
    state={}
    for i,width in zip((0,2,4,6),(48,12,12,12)):
        state[f'mlp.{i}.weight']=torch.zeros(12,width)
        state[f'mlp.{i}.bias']=torch.zeros(12)
    state['mlp.0.weight'][0,9]=1
    for i in (2,4,6):state[f'mlp.{i}.weight'].copy_(torch.eye(12))
    anchor=FrozenFlatMean({'actor_state_dict':state})
    obs=torch.zeros(2,48);obs[1,9]=.2
    assert anchor(obs)[1,0]>.19 and anchor(obs)[0,0]==0
    assert all(not p.requires_grad for p in anchor.parameters())
