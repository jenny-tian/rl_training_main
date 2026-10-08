import numpy as np
import pytest
import torch
from rl_training.recovery_sequence_search import sample_population, update_distribution, prefix_at_step, short_recovery_metrics
from rl_training.recovery_teacher import single_training_case


def test_search_keeps_zero_and_incumbent_and_hands_back_control():
    mean=np.zeros((3,12),np.float32);std=np.ones_like(mean);best=np.ones_like(mean)
    candidates=sample_population(np.random.default_rng(1),mean,std,best,population=24,bound=2.)
    assert not candidates[0].any() and np.array_equal(candidates[1],best)
    assert np.abs(candidates).max()<=2
    assert np.array_equal(prefix_at_step(candidates,11,4),candidates[:,2])
    assert not prefix_at_step(candidates,12,4).any()
    mu,sigma,elite=update_distribution(candidates,np.zeros(24),elite_count=6,std_floor=.25)
    assert elite.tolist()==list(range(6)) and (sigma>=.25).all()
    np.testing.assert_allclose(mu,candidates[:6].mean(0))


def test_short_positive_requires_survival_and_post_handoff_grounded_recovery():
    root=np.zeros((64,13));root[:,3]=1
    force=np.zeros((64,4,3));force[:,:2,2]=10
    active=np.ones(64,bool);done=np.zeros(64,bool)
    good=short_recovery_metrics(root,force,active,done,handoff_step=12,horizon=64)
    assert good['short_recovery'] and good['longest_post_handoff_upright']==52
    assert not short_recovery_metrics(root,force*0,active,done,handoff_step=12,horizon=64)['short_recovery']
    done[-1]=True
    assert not short_recovery_metrics(root,force,active,done,handoff_step=12,horizon=64)['short_recovery']
    success=np.zeros(64,bool);success[-1]=True
    assert short_recovery_metrics(root,force,active,done,handoff_step=12,horizon=64,success=success)['short_recovery']
    done[:]=False;active[10:]=False;done[9]=True
    failed=short_recovery_metrics(root,force,active,done,handoff_step=12,horizon=64)
    assert failed['active_controls']==10 and not failed['short_recovery'] and failed['longest_post_handoff_upright']==0


def test_fixed_train_case_excludes_other_terrain_and_validation():
    terrain=torch.tensor([0,1,2,4,5,7]);bank=torch.tensor([101,107,107]);held=torch.tensor([False,False,True])
    assert single_training_case(terrain,bank,held,1).tolist()==[-1,-1,-1,-1,1,1]
    with pytest.raises(ValueError):single_training_case(terrain,bank,held,2)
    with pytest.raises(ValueError):single_training_case(terrain,bank,held,-1)
