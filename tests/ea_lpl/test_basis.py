import itertools
import math

import torch

from rl_training.ea_lpl.basis import BASIS_NAMES, RESIDUAL_LIMITS, basis, residual_request, stacked_bases


def test_bases_are_orthogonal_and_inside_the_filter_envelope():
    limits = torch.tensor(RESIDUAL_LIMITS * 4, dtype=torch.float64)
    for name in BASIS_NAMES:
        v = basis(name, dtype=torch.float64)
        assert v.shape == (12, 2)
        gram = v.T @ v
        assert abs(float(gram[0, 1])) < 1e-12 and float(gram[0, 0]) > 0
        assert torch.isclose(gram[0, 0], gram[1, 1])
        for z in itertools.product((-1.0, 1.0), repeat=2):
            assert torch.all((v @ torch.tensor(z, dtype=torch.float64)).abs() <= limits + 1e-12)
    assert torch.count_nonzero(basis("hipx")[[1, 2, 4, 5, 7, 8, 10, 11]]) == 0
    assert torch.count_nonzero(basis("sag")[[0, 3, 6, 9]]) == 0
    assert torch.count_nonzero(basis("roll")[[0, 3, 6, 9]]) == 0


def test_residual_request_uses_each_environment_basis():
    bases = stacked_bases(dtype=torch.float64)
    index = torch.tensor([0, 1, 2])
    u = torch.tensor([[2.0, 0.0], [0.0, -1.0], [1.0, 1.0]], dtype=torch.float64)
    r = residual_request(index, u, bases)
    assert torch.allclose(r[0], bases[0][:, 0] * math.tanh(2.0))
    assert torch.allclose(r[1], -bases[1][:, 1] * math.tanh(1.0))
    assert torch.allclose(r[2], bases[2] @ torch.tanh(u[2]))
