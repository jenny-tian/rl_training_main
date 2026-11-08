"""Two-dimensional residual bases in canonical leg-major Lite3 order (HipX, HipY, Knee per leg).

Columns are scaled to the Source SafetyFilter residual limits, so V @ z with
z in [-1, 1]^2 never exceeds them. The HipY:Knee ratio -1:2 follows C0. The
hipx pair spans both lateral shift and stance width regardless of the left
and right HipX axis conventions.
"""
from __future__ import annotations

import torch

LEGS = ("FL", "FR", "HL", "HR")
RESIDUAL_LIMITS = (0.24, 0.20, 0.20)
BASIS_NAMES = ("sag", "roll", "hipx")


def _synergy(legs):
    column = torch.zeros(12, dtype=torch.float64)
    for leg in legs:
        i = LEGS.index(leg)
        column[3 * i + 1] = -0.10
        column[3 * i + 2] = 0.20
    return column


def _hipx(signs):
    column = torch.zeros(12, dtype=torch.float64)
    for i, sign in enumerate(signs):
        column[3 * i] = 0.12 * sign
    return column


_BASES = {
    "sag": torch.stack((_synergy(("FL", "FR")), _synergy(("HL", "HR"))), dim=1),
    "roll": torch.stack((_synergy(("FL", "HL")), _synergy(("FR", "HR"))), dim=1),
    "hipx": torch.stack((_hipx((1, 1, 1, 1)), _hipx((1, -1, 1, -1))), dim=1),
}


def basis(name, *, device=None, dtype=torch.float32):
    if name not in _BASES:
        raise ValueError(f"unknown basis {name!r}")
    return _BASES[name].to(device=device, dtype=dtype)


def stacked_bases(*, device=None, dtype=torch.float32):
    return torch.stack([basis(name, device=device, dtype=dtype) for name in BASIS_NAMES])


def residual_request(basis_index, u, bases):
    """r_i = V[basis_index[i]] @ tanh(u_i)."""
    return torch.einsum("njk,nk->nj", bases[basis_index], torch.tanh(u))
