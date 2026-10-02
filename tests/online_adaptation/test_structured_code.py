import copy
import io

import pytest
import torch

from rl_training.online_adaptation.networks import ResidualActor
from rl_training.online_adaptation.structured_code import compile_constant_code, fixed_joint_basis


def test_zero_code_is_exact_zero_for_arbitrary_observations_and_preserves_source_rng():
    actor = ResidualActor(49, 12, action_limits=torch.tensor([.24,.2,.2]*4))
    with torch.no_grad(): actor.mean_layer.bias.fill_(.7)
    state = copy.deepcopy(actor.state_dict())
    rng = torch.get_rng_state().clone()
    compiled = compile_constant_code(actor, torch.zeros(2))
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in state.items(): assert torch.equal(value, actor.state_dict()[key])
    with torch.no_grad():
        action, _, mean = compiled.sample(torch.randn(3,25,49),torch.randn(3,49),torch.randn(3,12),deterministic=True)
    assert torch.equal(action, torch.zeros_like(action)) and torch.equal(action, mean)
    assert all(not p.requires_grad for p in compiled.parameters())


def test_basis_order_front_hind_independence_and_normalized_columns():
    b = fixed_joint_basis()
    torch.testing.assert_close(b.T@b, torch.eye(2))
    assert not b[::3].any()
    assert not b[:6,1].any() and not b[6:,0].any()
    assert (b[[1,4],0] < 0).all() and (b[[2,5],0] > 0).all()


def test_code_corners_obey_existing_limits_and_roundtrip_matches():
    actor = ResidualActor(3,12,action_limits=torch.tensor([.24,.2,.2]*4))
    for front in (-1.,0.,1.):
        for hind in (-1.,0.,1.):
            code = torch.tensor([front,hind])
            compiled = compile_constant_code(actor,code)
            buf = io.BytesIO()
            torch.save(compiled.state_dict(),buf);buf.seek(0)
            restored = copy.deepcopy(actor)
            restored.load_state_dict(torch.load(buf,weights_only=True))
            inputs = (torch.randn(2,5,3),torch.randn(2,3),torch.randn(2,12))
            with torch.no_grad():
                action = compiled.sample(*inputs,deterministic=True)[0]
                other = restored.sample(*inputs,deterministic=True)[0]
            torch.testing.assert_close(action, (actor.action_limits*torch.tanh(fixed_joint_basis()@code)).expand_as(action))
            assert torch.equal(action,other)
            assert (action.abs() <= actor.action_limits).all()


@pytest.mark.parametrize('code', [torch.tensor([float('nan'),0.]), torch.tensor([1.01,0.]),torch.zeros(3),torch.zeros(2,dtype=torch.int64)])
def test_invalid_codes_are_rejected(code):
    with pytest.raises(ValueError):compile_constant_code(ResidualActor(3,12),code)
