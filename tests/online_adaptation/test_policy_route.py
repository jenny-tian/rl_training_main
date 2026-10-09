import math
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict
from rl_training.policy_route import route_features, expand_actor_route_inputs, lookahead_heading_values, route_heading_cost, route_heading_trace
from rl_training.policy_symmetry import mirror_observation, augment_lite3_mirror


def test_route_geometry_and_reflection():
    yaw=torch.tensor([0.,math.pi/2,-math.pi/2,math.pi])
    q=torch.zeros(4,4);q[:,0]=torch.cos(yaw/2);q[:,3]=torch.sin(yaw/2)
    pos=torch.zeros(4,3);pos[:,1]=torch.tensor([0.,1.65,-1.65,10.])
    origins=torch.zeros_like(pos)
    features=route_features(pos,q,origins)
    torch.testing.assert_close(features[:,0],torch.tensor([0.,1.,-1.,2.]))
    torch.testing.assert_close(features[:,1],yaw.sin(),atol=2e-7,rtol=0)
    torch.testing.assert_close(features[:,2],yaw.cos(),atol=2e-7,rtol=0)
    mirrored_pos=pos.clone();mirrored_pos[:,1]*=-1
    mirrored_q=q.clone();mirrored_q[:,3]*=-1
    torch.testing.assert_close(route_features(mirrored_pos,mirrored_q,origins),features*torch.tensor([-1,-1,1]))


def test_expanded_actor_preserves_original_mapping_for_arbitrary_route_input():
    torch.manual_seed(29)
    weight=torch.randn(32,45);bias=torch.randn(32)
    original={'actor_state_dict':{'mlp.0.weight':weight,'mlp.0.bias':bias}}
    expanded=expand_actor_route_inputs(original)
    obs=torch.randn(128,45);route=torch.randn(128,3)*100
    extra=torch.cat((obs[:,:9],route,obs[:,9:]),dim=-1)
    before=torch.nn.functional.linear(obs,weight,bias)
    after=torch.nn.functional.linear(extra,expanded['actor_state_dict']['mlp.0.weight'],bias)
    torch.testing.assert_close(before,after,atol=1e-5,rtol=1e-6)
    assert original['actor_state_dict']['mlp.0.weight'].shape==(32,45)
    assert torch.count_nonzero(expanded['actor_state_dict']['mlp.0.weight'][:,9:12])==0


def test_route_observation_mirror_keeps_action_history_at_end():
    obs=TensorDict({'policy':torch.randn(8,48),'critic':torch.randn(8,51)},batch_size=[8])
    augmented,_=augment_lite3_mirror(obs)
    for name in ('policy','critic'):
        mirrored=mirror_observation(obs[name],is_critic=name=='critic')
        torch.testing.assert_close(augmented[name][8:],mirrored)
        torch.testing.assert_close(mirror_observation(mirrored,is_critic=name=='critic'),obs[name],atol=0,rtol=0)
        start=12 if name=='critic' else 9
        torch.testing.assert_close(mirrored[:,start:start+3],obs[name][:,start:start+3]*torch.tensor([-1,-1,1]))


def test_lookahead_points_inward_reflects_and_respects_turn_bound():
    origins=torch.tensor([[10.,20.,0.]]).repeat(5,1)
    pos=origins.clone();pos[:,1]+=torch.tensor([0.,.2,-.2,2.,-2.])
    q=torch.tensor([[1.,0.,0.,0.]]).repeat(5,1)
    target,cost,old=lookahead_heading_values(pos,q,origins,lookahead_m=1.,maximum_heading_deg=20.)
    expected=torch.tensor([0.,-math.atan(.2),math.atan(.2),-math.radians(20),math.radians(20)])
    torch.testing.assert_close(target,expected,atol=1e-6,rtol=0)
    assert cost[0]==old[0]==0 and torch.equal(old,torch.zeros_like(old))
    q[:,0]=torch.cos(target/2);q[:,3]=torch.sin(target/2)
    _,aimed,_=lookahead_heading_values(pos,q,origins,lookahead_m=1.,maximum_heading_deg=20.)
    torch.testing.assert_close(aimed,torch.zeros_like(aimed),atol=1e-7,rtol=0)
    q[:,3]*=-1
    _,outward,_=lookahead_heading_values(pos,q,origins,lookahead_m=1.,maximum_heading_deg=20.)
    assert (outward[1:]>cost[1:]).all()


@pytest.mark.parametrize('length,limit',[(0.,20.),(-1.,20.),(float('nan'),20.),(1.,0.),(1.,90.),(1.,float('inf'))])
def test_invalid_heading_geometry_is_rejected(length,limit):
    with pytest.raises(ValueError):
        lookahead_heading_values(torch.zeros(1,3),torch.tensor([[1.,0.,0.,0.]]),torch.zeros(1,3),lookahead_m=length,maximum_heading_deg=limit)


def test_legacy_reward_is_exact_and_new_snapshot_survives_reset():
    root=torch.tensor([[0.,.3,.4,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]])
    data=SimpleNamespace(root_state_w=root,root_pos_w=root[:,:3],root_quat_w=root[:,3:7])
    class Scene(dict):pass
    scene=Scene(robot=SimpleNamespace(data=data));scene.env_origins=torch.zeros(1,3)
    env=SimpleNamespace(scene=scene,step_dt=.02)
    torch.testing.assert_close(route_heading_cost(env),1-route_features(root[:,:3],root[:,3:7],scene.env_origins)[:,2],atol=0,rtol=0)
    cost=route_heading_cost(env,lookahead_m=1.,maximum_heading_deg=20.,record=True)
    env.reward_manager=SimpleNamespace(active_terms=['course_heading'],_step_reward=-cost[:,None],get_term_cfg=lambda _:SimpleNamespace(weight=-1.))
    root[:,1]=0.  # Simulate automatic reset after the reward was calculated.
    recorded=route_heading_trace(env)
    assert recorded['heading_root_state'][0,1]==.3
    torch.testing.assert_close(recorded['heading_reward'],-cost*.02)
