import torch
import pytest

from rl_training.recovery_teacher import bank_indices, dedicated_recovery_mask
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


def test_dedicated_recovery_uses_fixed_25percent_slots_and_preserves_flat():
    terrain=torch.arange(8).repeat_interleave(16)
    mask=dedicated_recovery_mask(terrain,16)
    assert int(mask.sum())==32 and not mask[:32].any()
    assert torch.where(mask)[0].tolist()==list(range(32,48))+list(range(80,96))
    # Repeated fragmented reset calls must retain the scene-level allocation.
    ids=torch.tensor([33,70,82,120])
    assert mask[ids].tolist()==[True,False,True,False]
    with pytest.raises(ValueError):dedicated_recovery_mask(terrain,49)


def test_curriculum_boundaries_and_split_filtering():
    from rl_training.recovery_teacher import curriculum_bin
    schedule={"stage_starts":[0,4096,8192],"bins":[0,1,2]}
    assert [curriculum_bin(s,schedule) for s in (0,4095,4096,8191,8192,16384)]==[0,0,1,1,2,2]
    terrain=torch.tensor([101,101,101,107,107,107])
    validation=torch.tensor([False,False,True,False,False,True])
    tier=torch.tensor([0,1,0,0,1,0])
    for b,expected in ((0,[0,3]),(1,[1,4])):
        cases=bank_indices(torch.tensor([2,5]),torch.tensor([2,5]),torch.zeros(2,dtype=torch.long),
            terrain,validation,split='train',mixed=False,eligible=tier==b)
        assert cases.tolist()==expected
    with pytest.raises(ValueError):
        curriculum_bin(0,{"stage_starts":[0,0],"bins":[0,1]})
    with pytest.raises(ValueError):
        bank_indices(torch.tensor([2,5]),torch.tensor([2,5]),torch.zeros(2,dtype=torch.long),
            terrain,validation,split='train',mixed=False,eligible=tier==2)


def test_source_stratification_covers_all_sources_including_short_failures():
    from rl_training.recovery_teacher import source_stratified_cases, reset_recovery_states
    terrain=torch.arange(8).repeat_interleave(16)
    bank_terrain=torch.tensor([101]*9+[107]*13)
    held=torch.zeros(22,dtype=torch.bool);held[8]=True;held[21]=True
    cases=source_stratified_cases(terrain,16,bank_terrain,held)
    assert torch.where(cases>=0)[0].tolist()==list(range(32,48))+list(range(80,96))
    assert set(cases[cases>=0].tolist())==set(range(8))|set(range(9,21))
    assert not held[cases[cases>=0]].any()
    assert cases[32:48].tolist()==list(range(8))*2
    assert cases[80:96].tolist()==list(range(9,21))+list(range(9,13))
    # Fragmented reset groups reference the same permanent mapping.
    assert cases[torch.tensor([35,85,3])].tolist()==[3,14,-1]
    with pytest.raises(ValueError):source_stratified_cases(terrain,8,bank_terrain,held)
    with pytest.raises(ValueError):
        reset_recovery_states(None,[],bank_path='',bank_sha256='',split='validation',mixed=False,
            dedicated_per_terrain=16,source_stratified=True)
