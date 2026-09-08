import numpy as np

from rl_training.motion_prior.sources import (
    AI4ANIMATION_TOE_INDICES,
    load_bvh_foot_kinematics,
    load_bvh_foot_positions,
    load_ai4animation_foot_positions,
)


def test_ai4animation_loader_uses_explicit_pelvis_and_toe_indices(tmp_path):
    points = np.zeros((4, 27, 3), dtype=np.float32)
    for frame in range(4):
        points[frame, 0] = (0.1 * frame, 0.0, 0.0)
        for toe_index, point_index in enumerate(AI4ANIMATION_TOE_INDICES):
            points[frame, point_index] = points[frame, 0] + np.array(
                (0.2 * (1 if toe_index < 2 else -1), 0.1 * (1 if toe_index % 2 == 0 else -1), -0.3)
            )
    source_path = tmp_path / "dog_walk00_joint_pos.txt"
    np.savetxt(source_path, points.reshape(4, -1), delimiter=",")

    time, feet = load_ai4animation_foot_positions(
        source_path,
        position_scale=1.0,
        coordinate_rotation_xyz=(0.0, 0.0, 0.0),
        root_rotation_xyz=(0.0, 0.0, 0.0),
        frame_start=1,
        frame_end=3,
    )

    assert time.shape == (2,)
    assert feet.shape == (2, 4, 3)
    assert np.allclose(feet[0, 0], (0.2, 0.1, -0.3), atol=1e-6)
    assert np.allclose(feet[1, 0], (0.2, 0.1, -0.3), atol=1e-6)


def _synthetic_bvh() -> str:
    return """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftShoulder
  {
    OFFSET 1 0 1
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftHand
    {
      OFFSET 0 -2 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0 -1 0
      }
    }
  }
  JOINT RightShoulder
  {
    OFFSET 1 0 -1
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightHand
    {
      OFFSET 0 -2 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0 -1 0
      }
    }
  }
  JOINT LeftUpLeg
  {
    OFFSET -1 0 1
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftFoot
    {
      OFFSET 0 -2 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0 -1 0
      }
    }
  }
  JOINT RightUpLeg
  {
    OFFSET -1 0 -1
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightFoot
    {
      OFFSET 0 -2 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0 -1 0
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.02
0 7 11 90 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
2 9 13 90 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
"""


def test_bvh_loader_returns_root_relative_lite3_frame(tmp_path):
    path = tmp_path / "clip.bvh"
    path.write_text(_synthetic_bvh(), encoding="utf-8")

    time, feet = load_bvh_foot_positions(path, position_scale=1.0, forward_sign=1.0)

    assert np.allclose(time, (0.0, 0.02), atol=1e-7)
    assert feet.shape == (2, 4, 3)
    # BVH [x, y-vertical, z-left] is mapped to Lite3 [x, y-left, z-up].
    expected = np.asarray(
        ((1.0, 1.0, -3.0), (1.0, -1.0, -3.0), (-1.0, 1.0, -3.0), (-1.0, -1.0, -3.0)),
        dtype=np.float32,
    )
    assert np.allclose(feet[0], expected, atol=1e-5)
    # Root translation is removed, and root rotation is removed by body-frame conversion.
    assert np.allclose(feet[1], expected, atol=1e-5)


def test_bvh_loader_reads_zip_member_and_honors_frame_range(tmp_path):
    import zipfile

    archive = tmp_path / "MotionCapture.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("dog/walk.bvh", _synthetic_bvh())

    time, feet = load_bvh_foot_positions(
        archive,
        member="dog/walk.bvh",
        frame_start=1,
        frame_end=2,
        position_scale=0.5,
        forward_sign=1.0,
    )

    assert time.shape == (1,)
    assert feet.shape == (1, 4, 3)
    assert np.allclose(feet[0, 0], (0.5, 0.5, -1.5), atol=1e-5)


def test_bvh_kinematics_preserves_world_and_root_translation(tmp_path):
    path = tmp_path / "clip.bvh"
    path.write_text(_synthetic_bvh(), encoding="utf-8")

    time, body_feet, world_feet, root = load_bvh_foot_kinematics(
        path, position_scale=1.0, forward_sign=1.0
    )

    assert time.shape == (2,)
    assert body_feet.shape == world_feet.shape == (2, 4, 3)
    assert root.shape == (2, 3)
    assert np.allclose(body_feet[1], body_feet[0], atol=1e-5)
    assert np.allclose(root[1] - root[0], (2.0, 2.0, 2.0), atol=1e-5)
    assert np.allclose(world_feet[1] - world_feet[0], (2.0, 2.0, 2.0), atol=1e-5)
