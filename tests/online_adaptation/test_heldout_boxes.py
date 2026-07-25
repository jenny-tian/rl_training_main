import numpy as np

import pytest

from rl_training.tasks.manager_based.locomotion.velocity.terrains.heldout_boxes import (
    HeldoutBoxesTrackCfg,
    heldout_boxes_track,
)

pytestmark = pytest.mark.isaaclab


def test_track_is_deterministic_and_has_flat_then_variable_region():
    cfg = HeldoutBoxesTrackCfg(
        proportion=1.0,
        size=(20.0, 4.0),
        flat_length=3.0,
        lane_width=2.0,
        grid_width=0.4,
        grid_height_range=(0.04, 0.12),
    )
    cfg.seed = 7
    meshes, origin = heldout_boxes_track(0.5, cfg)
    vertices = np.concatenate([mesh.vertices for mesh in meshes])
    flat_z = vertices[vertices[:, 0] < cfg.flat_length - 0.1, 2]
    heldout_z = vertices[vertices[:, 0] > cfg.flat_length + 0.5, 2]
    assert origin[0] < cfg.flat_length
    assert flat_z.max() <= 1e-6
    assert heldout_z.max() > 0.04
