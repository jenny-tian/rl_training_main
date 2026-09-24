import pytest
import torch

from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol
from rl_training.online_adaptation.teacher_dataset import (
    TEACHER_DATASET_SCHEMA,
    TeacherTrajectoryDataset,
    load_teacher_dataset,
    save_teacher_dataset,
    select_safe_rows,
    validate_teacher_dataset,
)


def make_dataset() -> TeacherTrajectoryDataset:
    protocol = default_flat_online_protocol()
    return TeacherTrajectoryDataset(
        history=torch.zeros(4, 3, 5),
        current_frame=torch.zeros(4, 5),
        base_action=torch.zeros(4, 2),
        teacher_action=torch.full((4, 2), 0.1),
        terrain_seed=torch.tensor([101, 101, 103, 103]),
        episode_index=torch.tensor([0, 0, 1, 1]),
        step_index=torch.tensor([0, 1, 0, 1]),
        safe_mask=torch.tensor([True, False, True, True]),
        metadata={
            "terrain_split": "train",
            "protocol_sha256": protocol.sha256(),
            "base_checkpoint_sha256": "base",
            "teacher_checkpoint_sha256": "teacher",
            "sealed_seeds": list(protocol.sealed_seeds),
            "frame_dim": 5,
            "action_dim": 2,
            "history_length": 3,
        },
    )


def test_teacher_dataset_round_trip_and_safe_selection(tmp_path):
    dataset = make_dataset()
    path = tmp_path / "teacher.pt"
    save_teacher_dataset(path, dataset, sealed_seeds=(1009,))
    restored = load_teacher_dataset(path, sealed_seeds=(1009,))
    assert restored.metadata["terrain_split"] == "train"
    assert restored.size == 4
    safe = select_safe_rows(restored)
    assert safe.size == 3
    assert bool(safe.safe_mask.all())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == TEACHER_DATASET_SCHEMA
    with pytest.raises(FileExistsError):
        save_teacher_dataset(path, dataset)


def test_teacher_dataset_rejects_sealed_seed_and_invalid_metadata():
    dataset = make_dataset()
    with pytest.raises(ValueError, match="sealed"):
        validate_teacher_dataset(dataset, sealed_seeds=(101,))
    invalid = make_dataset()
    invalid.metadata["terrain_split"] = "sealed"
    with pytest.raises(ValueError, match="train or development"):
        validate_teacher_dataset(invalid)


def test_teacher_dataset_enforces_formal_split_seed_assignment_and_protocol():
    dataset = make_dataset()
    dataset.terrain_seed[0] = 109
    with pytest.raises(ValueError, match="outside the formal train split"):
        validate_teacher_dataset(dataset)

    dataset = make_dataset()
    dataset.metadata["protocol_sha256"] = "other-protocol"
    with pytest.raises(ValueError, match="protocol hash"):
        validate_teacher_dataset(dataset)

    dataset = make_dataset()
    dataset.metadata["sealed_seeds"] = [1009]
    with pytest.raises(ValueError, match="every protocol sealed seed"):
        validate_teacher_dataset(dataset)
