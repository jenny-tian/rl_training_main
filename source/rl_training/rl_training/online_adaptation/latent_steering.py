"""Utilities for fitting compact steering codes to safe teacher actions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SteeringInversionResult:
    code: Tensor
    action: Tensor
    action_mse: Tensor
    regularized_loss: Tensor
    iterations: int


def invert_steering_code(
    action_from_code: Callable[[Tensor], Tensor],
    target_action: Tensor,
    initial_code: Tensor,
    *,
    code_l2: float = 1.0e-3,
    learning_rate: float = 5.0e-2,
    iterations: int = 100,
) -> SteeringInversionResult:
    """Fit a low-dimensional code to a teacher action without updating policy weights.

    ``action_from_code`` must map the full batch of codes to normalized actions.
    Only the code tensor is optimized; gradients are requested specifically for
    it, so model parameter ``.grad`` buffers remain untouched.
    """
    if not callable(action_from_code):
        raise TypeError("action_from_code must be callable")
    if (
        not isinstance(target_action, Tensor)
        or target_action.ndim < 2
        or not target_action.is_floating_point()
        or not bool(torch.isfinite(target_action).all().item())
    ):
        raise ValueError("target_action must be a finite floating-point batch tensor")
    if (
        not isinstance(initial_code, Tensor)
        or initial_code.ndim < 2
        or initial_code.shape[0] != target_action.shape[0]
        or not initial_code.is_floating_point()
        or initial_code.device != target_action.device
        or initial_code.dtype != target_action.dtype
        or not bool(torch.isfinite(initial_code).all().item())
    ):
        raise ValueError("initial_code must be a finite batch tensor matching target_action")
    for name, value in (("code_l2", code_l2), ("learning_rate", learning_rate)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (name == "learning_rate" and value == 0):
            raise ValueError(f"{name} must be {'positive' if name == 'learning_rate' else 'nonnegative'}")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be a positive integer")

    code = torch.nn.Parameter(initial_code.detach().clone())
    optimizer = torch.optim.Adam((code,), lr=float(learning_rate))
    for _ in range(iterations):
        predicted = action_from_code(code)
        if (
            not isinstance(predicted, Tensor)
            or predicted.shape != target_action.shape
            or predicted.device != target_action.device
            or predicted.dtype != target_action.dtype
            or not bool(torch.isfinite(predicted).all().item())
        ):
            raise ValueError("action_from_code must return finite actions matching target_action")
        if not predicted.requires_grad:
            raise ValueError("action_from_code must remain differentiable with respect to code")
        mse = (predicted - target_action).square().mean()
        regularized = mse + float(code_l2) * code.square().mean()
        gradient = torch.autograd.grad(regularized, code, only_inputs=True)[0]
        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError("steering inversion produced a non-finite code gradient")
        optimizer.zero_grad(set_to_none=True)
        code.grad = gradient
        optimizer.step()

    with torch.no_grad():
        fitted_code = code.detach().clone()
        fitted_action = action_from_code(fitted_code).detach().clone()
        action_mse = (fitted_action - target_action).square().mean()
        regularized_loss = action_mse + float(code_l2) * fitted_code.square().mean()
    return SteeringInversionResult(
        code=fitted_code,
        action=fitted_action,
        action_mse=action_mse,
        regularized_loss=regularized_loss,
        iterations=iterations,
    )
