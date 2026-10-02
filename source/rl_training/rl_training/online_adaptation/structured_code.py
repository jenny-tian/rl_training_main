"""Compile fixed joint synergies into the existing deterministic residual actor.

No phase or contact inference is made. Positive codes request coupled hip/knee
changes, not certified swing lift. The caller retains the registered residual
gain, physical safety projection, and rate limiter.
"""
from __future__ import annotations

import copy
import torch

from .networks import ResidualActor

BASIS_VERSION = 'front_hind_hip_knee_unit_columns_v1'


def fixed_joint_basis(*, device=None, dtype=torch.float32):
    """Canonical leg-major Lite3 order: front/hind columns, HipX unchanged."""
    basis = torch.zeros(12, 2, device=device, dtype=dtype)
    for leg in range(4):
        column = int(leg >= 2)
        basis[3*leg+1, column] = -1.
        basis[3*leg+2, column] = 2.
    return basis / basis.norm(dim=0, keepdim=True)


def compile_constant_code(actor: ResidualActor, code: torch.Tensor):
    """Return a frozen copy whose deterministic output is L*tanh(B*code).

    All distribution/encoder state is preserved except the mean head; only
    deterministic evaluation is supported by this compiler's contract. The
    source actor and global RNG are unchanged. No learned weights are imported.
    """
    if type(actor) is not ResidualActor or actor.action_limits.shape != (12,):
        raise ValueError('requires direct 12-joint ResidualActor')
    if (not isinstance(code, torch.Tensor) or code.shape != (2,) or not code.is_floating_point()
            or not torch.isfinite(code).all() or (code.abs() > 1).any()):
        raise ValueError('finite two-dimensional code in [-1,1] required')
    if not torch.isfinite(actor.action_limits).all() or (actor.action_limits < 0).any():
        raise ValueError('finite nonnegative action limits required')
    compiled = copy.deepcopy(actor).eval().requires_grad_(False)
    basis = fixed_joint_basis(device=compiled.mean_layer.bias.device, dtype=compiled.mean_layer.bias.dtype)
    with torch.no_grad():
        compiled.mean_layer.weight.zero_()
        compiled.mean_layer.bias.copy_(basis @ code.to(basis))
    return compiled
