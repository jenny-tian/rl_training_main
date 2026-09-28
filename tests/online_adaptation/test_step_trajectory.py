import json

import numpy as np
import pytest
import torch

from rl_training.online_adaptation.step_trajectory import StepTrajectory


def test_chunks_reconstruct_terminal_pose_without_reset_contamination(tmp_path):
    writer = StepTrajectory(tmp_path / "trajectory", metadata={"initial_x": -3.5}, chunk_steps=2)
    observation = torch.ones(1, 45, requires_grad=True)
    before = writer.snapshot(observation=observation, requested_action=torch.ones(1, 12))
    with torch.no_grad():
        observation.zero_()
    positions = [0.65212345, 1.23456789, -3.4]
    for index, x in enumerate(positions):
        writer.append(**before, step=index + 1, base_x=x, reward=-8. if index == 1 else 2.,
                      terminated=index == 1, executed_action=torch.zeros(1, 12))
    manifest = writer.close(complete=True)
    assert manifest["status"] == "complete"
    assert manifest["steps"] == 3
    assert [chunk["steps"] for chunk in manifest["chunks"]] == [2, 1]
    chunks = [np.load(writer.directory / chunk["file"], allow_pickle=False) for chunk in manifest["chunks"]]
    assert np.concatenate([chunk["base_x"] for chunk in chunks]).max() == max(positions)
    assert chunks[0]["observation"].dtype == np.float16
    assert np.all(chunks[0]["observation"] == 1)
    assert chunks[0]["reward"][1] == -8.
    assert chunks[0]["terminated"].tolist() == [False, True]
    assert np.all(chunks[0]["requested_action"] != chunks[0]["executed_action"])
    assert manifest["bytes"] == sum((writer.directory / chunk["file"]).stat().st_size for chunk in manifest["chunks"])
    writer.close()  # finally must not relabel a completed run as interrupted
    assert json.loads((writer.directory / "manifest.json").read_text())["status"] == "complete"


def test_interruption_retains_last_partial_chunk_and_rejects_overwrite(tmp_path):
    path = tmp_path / "trajectory"
    writer = StepTrajectory(path, metadata={})
    writer.append(observation=torch.zeros(1, 45), base_x=0.)
    manifest = writer.close()
    assert manifest["status"] == "interrupted"
    assert manifest["steps"] == 1
    with pytest.raises(FileExistsError):
        StepTrajectory(path, metadata={})
    with pytest.raises(RuntimeError, match="closed"):
        writer.append(observation=torch.zeros(1, 45), base_x=1.)


def test_schema_drift_and_float16_overflow_fail_explicitly(tmp_path):
    writer = StepTrajectory(tmp_path / "trajectory", metadata={})
    writer.append(observation=torch.zeros(1, 45), base_x=0.)
    with pytest.raises(ValueError, match="constant"):
        writer.append(observation=torch.zeros(1, 48), base_x=0.)
    with pytest.raises(ValueError, match="overflows"):
        writer.append(observation=torch.full((1, 45), 100000.), base_x=0.)
    with pytest.raises(ValueError, match="non-finite"):
        writer.append(observation=torch.zeros(1, 45), base_x=float("nan"))
    assert writer.close()["steps"] == 1
