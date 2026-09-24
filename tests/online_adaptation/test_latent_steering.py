import pytest
import torch

from rl_training.online_adaptation.latent_steering import invert_steering_code


def test_inversion_fits_teacher_action_without_touching_policy_parameter_grad():
    policy = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        policy.weight.copy_(torch.eye(2))
    policy.weight.grad = torch.full_like(policy.weight, 7.0)
    code_map = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        code_map.weight.copy_(torch.eye(2))
    target = torch.tensor([[0.4, -0.2], [0.1, 0.3]])
    initial = torch.zeros_like(target)

    result = invert_steering_code(
        lambda code: policy(code_map(code)), target, initial,
        code_l2=0.0, learning_rate=0.08, iterations=100,
    )

    assert result.action_mse.item() < 1.0e-5
    assert torch.equal(policy.weight.grad, torch.full_like(policy.weight, 7.0))
    assert code_map.weight.grad is None
    assert not result.code.requires_grad


def test_inversion_rejects_mismatched_or_nonfinite_inputs():
    target = torch.zeros(2, 3)
    initial = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="matching target_action"):
        invert_steering_code(lambda code: code, target, initial)
    with pytest.raises(ValueError, match="finite floating-point"):
        invert_steering_code(lambda code: code, torch.full_like(target, float("nan")), torch.zeros_like(target))
    with pytest.raises(ValueError, match="return finite actions matching"):
        invert_steering_code(lambda code: torch.zeros(2, 2), target, torch.zeros_like(target))
    with pytest.raises(ValueError, match="remain differentiable"):
        invert_steering_code(lambda code: code.detach(), target, torch.zeros_like(target))
