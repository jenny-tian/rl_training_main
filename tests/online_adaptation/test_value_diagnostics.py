import numpy as np
import pytest
import torch
from rl_training.online_adaptation.value_diagnostics import complete_episode_returns, value_metrics, time_features, make_value_network, fit_value_network


def test_complete_mc_includes_terminal_reward_without_value_bootstrap():
    actual=complete_episode_returns([1.,2.,8.],[0,0,1],[0,0,0],[0,0,0],gamma=.5)
    np.testing.assert_array_equal(actual,[4.,6.,8.])


@pytest.mark.parametrize('term,trunc,cutoff', [([0,0],[0,1],[0,1]),([1,1],[0,0],[0,0]),([0,0],[0,0],[0,0]),([0,1],[0,0],[0,1])])
def test_mc_rejects_cutoffs_missing_ends_and_joined_episodes(term,trunc,cutoff):
    with pytest.raises(ValueError,match='full task terminal'):
        complete_episode_returns([1,2],term,trunc,cutoff,gamma=.99)


def test_ev_does_not_hide_bias_and_constant_target_is_not_interpretable():
    m=value_metrics([1,2,3],[11,12,13])
    assert m['explained_variance']==1. and m['bias']==10. and m['r_squared']<0
    flat=value_metrics([100.,100.],[99.,99.])
    assert flat['normalized_rmse'] is None and flat['explained_variance'] is None
    assert flat['rmse']==1
    assert value_metrics([],[])=={'samples':0}


def test_time_baseline_uses_absolute_elapsed_feature_not_episode_length():
    obs=np.zeros((3,48));obs[:,-1]=[0,.1,.2]
    np.testing.assert_allclose(time_features(obs),[[1,0,0,0],[1,.1,.01,.001],[1,.2,.04,.008]])


def test_fixed_budget_fit_only_updates_value_model_and_reproduces():
    torch.manual_seed(8)
    initial=make_value_network(48).state_dict()
    x=np.zeros((32,48),dtype=np.float32);x[:,-1]=np.linspace(0,1,32)
    y=1+2*x[:,-1]
    models=[]
    for _ in range(2):
        model=make_value_network(48);model.load_state_dict(initial)
        result=fit_value_network(model,x,y,epochs=5,batch_size=8,learning_rate=.001,seed=11)
        assert result['optimizer_steps']==20
        assert result['training_curve'][-1]['training_rmse']<result['training_curve'][0]['training_rmse']
        models.append(model)
    for key,value in models[0].state_dict().items():assert torch.equal(value,models[1].state_dict()[key])


def test_loader_checks_real_trace_contract_and_rejects_collector_terminal(tmp_path):
    import json
    from rl_training.online_adaptation.provenance import sha256_file
    from scripts.reinforcement_learning.online_adaptation.diagnose_complete_value import load_episode
    base=tmp_path/'base.pt';base.write_bytes(b'fixture base')
    path=tmp_path/'trial-00000';(path/'trajectory').mkdir(parents=True)
    x=np.zeros((3,1,48),dtype=np.float32);x[:,0,-1]=np.arange(3)/2000.
    a=np.zeros((3,1,12),dtype=np.float32)
    arrays={'step':np.arange(1,4),'reward':np.array([1.,2.,3.]),'ordinary_reward':np.array([1.,2.,3.]),
            'terminal_reward_override':np.zeros(3),'learning_terminated':np.array([0,0,1]),
            'learning_truncated':np.zeros(3),'collector_cutoff':np.zeros(3),'bootstrap_mask':np.array([1,1,0]),
            'requested_action':a,'executed_action':a,'mean_action':a,'base_action':a,
            'critic_observation':x,'root_state':np.zeros((3,1,13))}
    np.savez(path/'trajectory/chunk.npz',**arrays)
    (path/'trajectory/manifest.json').write_text(json.dumps({'status':'complete','vector_dtype':'float32',
        'steps':3,'chunks':[{'file':'chunk.npz'}]}))
    (path/'summary.json').write_text(json.dumps({'mode':'base_frozen','updates':0,'actor_updates':0,
        'critic_observation_schema':'policy_plus_body_velocity_xy_elapsed_v1','session_budget':{'session_id':'s','trial_id':'trial-00000'},
        'task_objective_sha256':'task','steps':3,'total_reward':6.,'initial_x':0.,'completion_target_x':1.,'stop_reason':'task_timeout'}))
    torch.save({'base_checkpoint':str(base),'discount_gamma':.99,'rows':[{'nonfoot_contact':False}]*3,
        'session':{'updater':{'actor_optimizer':{'state':{}},'critic_optimizer':{'state':{}},'policy_version':0,'critic':make_value_network(48).state_dict()}}},path/'latest.pt')
    trial={'id':'trial-00000','output':str(path),'status':'complete','phase':'evaluation','confirmed_steps':3,'reset_seed':17}
    manifest={'session_id':'s','task_objective_sha256':'task','base':{'sha256':sha256_file(base)}}
    result=load_episode(tmp_path,trial,manifest,.99)
    np.testing.assert_allclose(result['y'],[1+.99*2+.99**2*3,2+.99*3,3])
    arrays['collector_cutoff'][-1]=1
    np.savez(path/'trajectory/chunk.npz',**arrays)
    with pytest.raises(ValueError,match='full task terminal'):
        load_episode(tmp_path,trial,manifest,.99)
    arrays['collector_cutoff'][-1]=0
    arrays['critic_observation'][0,0,0]=np.nan
    np.savez(path/'trajectory/chunk.npz',**arrays)
    with pytest.raises(ValueError,match='finite 48-dimensional'):
        load_episode(tmp_path,trial,manifest,.99)
    arrays['critic_observation'][0,0,0]=0
    np.savez(path/'trajectory/chunk.npz',**arrays)
    checkpoint=torch.load(path/'latest.pt',weights_only=False)
    checkpoint['session']['updater']['critic_optimizer']['state']={0:{'step':1}}
    torch.save(checkpoint,path/'latest.pt')
    with pytest.raises(ValueError,match='critic optimizer'):
        load_episode(tmp_path,trial,manifest,.99)
    checkpoint['session']['updater']['critic_optimizer']['state']={}
    checkpoint['rows']=[{'nonfoot_contact':0},{'nonfoot_contact':1},{'nonfoot_contact':0}]
    torch.save(checkpoint,path/'latest.pt')
    result=load_episode(tmp_path,trial,manifest,.99)
    assert result['contact'].dtype==np.bool_
    from scripts.reinforcement_learning.online_adaptation.diagnose_complete_value import evaluate_episodes
    metrics=evaluate_episodes([result],lambda x:np.zeros(len(x)))
    assert metrics['stage_metrics']['nonfoot_contact']['samples']==1
    assert metrics['stage_metrics']['no_nonfoot_contact']['samples']==2
    checkpoint['rows'][0]['nonfoot_contact']=2
    torch.save(checkpoint,path/'latest.pt')
    with pytest.raises(ValueError,match='binary scalars'):
        load_episode(tmp_path,trial,manifest,.99)
