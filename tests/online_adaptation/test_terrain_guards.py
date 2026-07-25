from types import SimpleNamespace

import pytest

from rl_training.online_adaptation.terrain_guards import assert_heldout_terrains_disabled


def terrain(boxes=0.0, stairs=0.0, stairs_inv=0.0):
    sub = {
        "boxes": SimpleNamespace(proportion=boxes),
        "pyramid_stairs": SimpleNamespace(proportion=stairs),
        "pyramid_stairs_inv": SimpleNamespace(proportion=stairs_inv),
    }
    return SimpleNamespace(sub_terrains=sub)


def test_guard_accepts_zero_heldout_proportions():
    assert_heldout_terrains_disabled(terrain())


def test_guard_rejects_any_heldout_probability():
    with pytest.raises(ValueError, match="held-out terrain leakage"):
        assert_heldout_terrains_disabled(terrain(boxes=0.01))
