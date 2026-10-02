import torch
from rl_training.policy_flat_anchor import FrozenFlatMean, flat_anchor_cost


def test_frozen_target_ignores_route_inputs_and_has_no_trainable_parameters():
    torch.manual_seed(41)
    model=torch.nn.Sequential(torch.nn.Linear(45,16),torch.nn.ELU(),torch.nn.Linear(16,8),torch.nn.ELU(),
                              torch.nn.Linear(8,4),torch.nn.ELU(),torch.nn.Linear(4,12))
    payload={'actor_state_dict':{'mlp.'+k:v for k,v in model.state_dict().items()}}
    target=FrozenFlatMean(payload)
    assert not list(target.parameters())
    x=torch.randn(20,45,requires_grad=True)
    expanded=torch.cat((x[:,:9],torch.randn(20,3)*100,x[:,9:]),-1)
    from rl_training.lite3_joint_conventions import clip_lite3_policy_action
    torch.testing.assert_close(target(expanded),clip_lite3_policy_action(model(x)))
    assert not target(expanded).requires_grad
    before={k:v.clone() for k,v in target.state_dict().items()}
    target(expanded)
    assert all(torch.equal(v,target.state_dict()[k]) for k,v in before.items())


def test_anchor_is_exactly_zero_on_rough_and_uses_emitted_action():
    action=torch.ones(3,12);desired=torch.zeros_like(action)
    mask=torch.tensor([True,False,True])
    torch.testing.assert_close(flat_anchor_cost(action,desired,mask),torch.tensor([1.,0.,1.]))
    torch.testing.assert_close(flat_anchor_cost(desired,desired,mask),torch.zeros(3))
