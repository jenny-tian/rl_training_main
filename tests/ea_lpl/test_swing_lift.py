import pytest
import torch

from rl_training.ea_lpl.basis import basis
from rl_training.ea_lpl.swing_lift import swing_request


def test_only_swinging_selected_legs_get_the_sag_direction():
    stance = torch.tensor([[True, True, False, True], [False, True, False, False], [True, True, True, True]])
    r = swing_request(stance, torch.ones(3), ['R', 'Q', 'R'], 1.)
    assert torch.allclose(r[0, 6:9], torch.tensor([0., -.10, .20])) and not r[0, :6].any() and not r[0, 9:].any()
    assert torch.allclose(r[1].reshape(4, 3), torch.tensor([[0., -.10, .20], [0., 0., 0.], [0., -.10, .20], [0., -.10, .20]]))
    assert not r[2].any()
    v = basis('sag')
    assert torch.allclose(swing_request(torch.zeros(1, 4, dtype=torch.bool), torch.ones(1), ['Q'], 1.)[0], v.sum(1))


def test_scale_roughness_and_untouched_rows():
    stance = torch.zeros(3, 4, dtype=torch.bool)
    r = swing_request(stance, torch.tensor([0., .5, 1.]), ['Q', 'R', None], 2.)
    assert not r[0].any() and not r[2].any()
    assert torch.allclose(r[1, 6:9], torch.tensor([0., -.10, .20])) and not r[1, :6].any()
    assert swing_request(stance, torch.ones(3), ['R'] * 3, 4.).abs().max() == pytest.approx(.8)


def test_front_static_part_does_not_scale_and_ignores_contact():
    stance = torch.tensor([[True, False, True, False]])
    r = swing_request(stance, torch.ones(1), ['S'], 2.)[0].reshape(4, 3)
    assert torch.allclose(r[0], torch.tensor([0., -.0905, .181])) and torch.allclose(r[1], r[0])
    assert not r[2].any() and torch.allclose(r[3], torch.tensor([0., -.20, .40]))


def test_rejects_bad_inputs():
    with pytest.raises(ValueError): swing_request(torch.zeros(1, 4), torch.ones(1), ['R'], 1.)
    with pytest.raises(ValueError): swing_request(torch.zeros(1, 4, dtype=torch.bool), torch.ones(1), ['R'], 5.)
    with pytest.raises(ValueError): swing_request(torch.zeros(1, 4, dtype=torch.bool), torch.tensor([1.5]), ['R'], 1.)
