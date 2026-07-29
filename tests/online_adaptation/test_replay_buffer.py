from dataclasses import fields

import pytest
import torch

from rl_training.online_adaptation.replay_buffer import ReplayBuffer


def make_transition(value: float, batch: int = 2):
    return dict(
        history=torch.full((batch, 3, 2), value),
        current_frame=torch.full((batch, 2), value),
        base_action=torch.full((batch, 1), value),
        residual_action=torch.full((batch, 1), value),
        reward=torch.full((batch, 1), value),
        next_history=torch.full((batch, 3, 2), value + 1),
        next_frame=torch.full((batch, 2), value + 1),
        next_base_action=torch.full((batch, 1), value + 1),
        terminated=torch.zeros(batch, 1),
        truncated=torch.zeros(batch, 1),
        bootstrap_mask=torch.ones(batch, 1),
        discount=torch.full((batch, 1), 0.99),
        safety_mask=torch.ones(batch, 1),
        curriculum_stage_id=torch.zeros(batch, 1, dtype=torch.int64),
        episode_id=torch.zeros(batch, 1, dtype=torch.int64),
    )


def make_replay():
    return ReplayBuffer(capacity=1, history_length=3, frame_dim=2, action_dim=1)


def assert_single_stored_transition(replay, expected):
    assert len(replay) == 1
    sampled = replay.sample(batch_size=1, device="cpu", generator=torch.Generator().manual_seed(0))
    for field in fields(sampled):
        expected_value = expected[field.name][:1]
        if field.name in {"curriculum_stage_id", "episode_id"}:
            assert getattr(sampled, field.name).dtype == torch.int64
        else:
            expected_value = expected_value.float()
            assert getattr(sampled, field.name).dtype == torch.float32
        torch.testing.assert_close(getattr(sampled, field.name), expected_value, rtol=0, atol=0)


def test_replay_overwrites_oldest_and_samples_float32():
    replay = ReplayBuffer(capacity=3, history_length=3, frame_dim=2, action_dim=1, storage_dtype=torch.float16)
    replay.add_batch(**make_transition(1.0))
    replay.add_batch(**make_transition(2.0))
    assert len(replay) == 3
    batch = replay.sample(batch_size=3, device="cpu", generator=torch.Generator().manual_seed(0))
    assert batch.history.dtype == torch.float32
    assert batch.history.shape == (3, 3, 2)


def test_replay_add_batch_filters_paused_environments():
    replay = ReplayBuffer(capacity=3, history_length=3, frame_dim=2, action_dim=1)

    replay.add_batch(**make_transition(4.0), valid_mask=torch.tensor([True, False]))

    assert len(replay) == 1
    batch = replay.sample(batch_size=1, device="cpu")
    assert batch.reward.item() == 4.0


def test_replay_round_trips_every_transition_field_and_preserves_id_dtype():
    replay = make_replay()
    transition = make_transition(0.25, batch=1)
    transition.update(
        terminated=torch.tensor([[1.0]]),
        truncated=torch.tensor([[0.0]]),
        bootstrap_mask=torch.tensor([[0.0]]),
        discount=torch.tensor([[0.75]]),
        safety_mask=torch.tensor([[1.0]]),
        curriculum_stage_id=torch.tensor([[7]], dtype=torch.int64),
        episode_id=torch.tensor([[11]], dtype=torch.int64),
    )

    replay.add_batch(**transition)

    assert_single_stored_transition(replay, transition)


def test_replay_rejects_terminated_bootstrap_without_mutation():
    replay = make_replay()
    original = make_transition(0.25, batch=1)
    replay.add_batch(**original)
    invalid = make_transition(0.5, batch=1)
    invalid.update(
        terminated=torch.ones(1, 1),
        truncated=torch.zeros(1, 1),
        bootstrap_mask=torch.ones(1, 1),
    )

    with pytest.raises(ValueError, match="terminated.*bootstrap"):
        replay.add_batch(**invalid)

    assert_single_stored_transition(replay, original)


def test_replay_accepts_bootstrapping_truncation():
    replay = make_replay()
    transition = make_transition(0.5, batch=1)
    transition.update(
        terminated=torch.zeros(1, 1),
        truncated=torch.ones(1, 1),
        bootstrap_mask=torch.ones(1, 1),
    )

    replay.add_batch(**transition)

    assert_single_stored_transition(replay, transition)


@pytest.mark.parametrize("field", ["capacity", "history_length", "frame_dim", "action_dim"])
@pytest.mark.parametrize("invalid_value", [0, -1, True, 1.5])
def test_replay_constructor_rejects_non_positive_exact_int_dimensions(field, invalid_value):
    arguments = dict(capacity=3, history_length=3, frame_dim=2, action_dim=1)
    arguments[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        ReplayBuffer(**arguments)


@pytest.mark.parametrize(
    "storage_dtype",
    [torch.int64, torch.bool, torch.complex64, "float32", None],
)
def test_replay_constructor_rejects_non_real_floating_storage_dtype(storage_dtype):
    with pytest.raises(ValueError, match="storage_dtype"):
        ReplayBuffer(
            capacity=3,
            history_length=3,
            frame_dim=2,
            action_dim=1,
            storage_dtype=storage_dtype,
        )


@pytest.mark.parametrize("storage_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_replay_allowed_storage_dtypes_preserve_fractional_data(storage_dtype):
    replay = ReplayBuffer(
        capacity=1,
        history_length=3,
        frame_dim=2,
        action_dim=1,
        storage_dtype=storage_dtype,
    )

    replay.add_batch(**make_transition(0.125, batch=1))

    assert replay.sample(1, "cpu").history[0, 0, 0].item() == 0.125


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("terminated", 2.0),
        ("truncated", float("nan")),
        ("bootstrap_mask", -1.0),
        ("safety_mask", 0.5),
        ("terminated", 0.0 + 1.0j),
    ],
)
def test_replay_rejects_invalid_explicit_masks_without_mutation(field, invalid_value):
    replay = make_replay()
    original = make_transition(0.25, batch=1)
    replay.add_batch(**original)
    invalid = make_transition(0.5, batch=1)
    invalid[field] = torch.tensor([[invalid_value]])

    with pytest.raises(ValueError, match=field):
        replay.add_batch(**invalid)

    assert_single_stored_transition(replay, original)


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01, float("nan"), 0.5 + 1.0j])
def test_replay_rejects_invalid_discount_without_mutation(invalid_value):
    replay = make_replay()
    original = make_transition(0.25, batch=1)
    replay.add_batch(**original)
    invalid = make_transition(0.5, batch=1)
    invalid["discount"] = torch.tensor([[invalid_value]])

    with pytest.raises(ValueError, match="discount"):
        replay.add_batch(**invalid)

    assert_single_stored_transition(replay, original)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("curriculum_stage_id", torch.tensor([[-1]], dtype=torch.int64)),
        ("episode_id", torch.tensor([[1.0]])),
        ("curriculum_stage_id", torch.tensor([[True]])),
    ],
)
def test_replay_rejects_invalid_ids_without_mutation(field, invalid_value):
    replay = make_replay()
    original = make_transition(0.25, batch=1)
    replay.add_batch(**original)
    invalid = make_transition(0.5, batch=1)
    invalid[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        replay.add_batch(**invalid)

    assert_single_stored_transition(replay, original)


@pytest.mark.parametrize(
    ("field", "invalid_shape"),
    [
        ("history", (1, 2, 2)),
        ("current_frame", (1, 1)),
        ("base_action", (1, 2)),
        ("residual_action", (1, 2)),
        ("reward", (1,)),
        ("next_history", (1, 2, 2)),
        ("next_frame", (1, 1)),
        ("next_base_action", (1, 2)),
        ("terminated", (1,)),
        ("truncated", (1,)),
        ("bootstrap_mask", (1,)),
        ("discount", (1,)),
        ("safety_mask", (1,)),
        ("curriculum_stage_id", (1,)),
        ("episode_id", (1,)),
    ],
)
def test_replay_rejects_wrong_field_shapes_without_mutation(field, invalid_shape):
    replay = make_replay()
    original = make_transition(0.25, batch=1)
    replay.add_batch(**original)
    invalid = make_transition(0.5, batch=1)
    invalid[field] = torch.zeros(invalid_shape, dtype=invalid[field].dtype)

    with pytest.raises(ValueError, match=field):
        replay.add_batch(**invalid)

    assert_single_stored_transition(replay, original)


@pytest.mark.parametrize(
    "valid_mask",
    [
        torch.tensor([1.0, 0.5]),
        torch.tensor([1.0, float("nan")]),
        torch.ones(2, 2),
        torch.tensor([1.0 + 0.0j, 0.0 + 1.0j]),
    ],
)
def test_replay_rejects_invalid_valid_mask_without_mutation(valid_mask):
    replay = make_replay()
    original = make_transition(0.25, batch=1)
    replay.add_batch(**original)

    with pytest.raises(ValueError, match="valid_mask"):
        replay.add_batch(**make_transition(0.5), valid_mask=valid_mask)

    assert_single_stored_transition(replay, original)


def test_replay_rejects_mismatched_batch_dimensions_without_mutation():
    replay = make_replay()
    original = make_transition(0.25, batch=1)
    replay.add_batch(**original)
    invalid = make_transition(0.5)
    invalid["next_frame"] = torch.zeros(1, 2)

    with pytest.raises(ValueError, match="batch dimension"):
        replay.add_batch(**invalid)

    assert_single_stored_transition(replay, original)


def test_replay_clear_discards_all_rows_and_resets_for_reuse():
    replay = ReplayBuffer(capacity=3, history_length=3, frame_dim=2, action_dim=1)
    replay.add_batch(**make_transition(1.0))

    replay.clear()

    assert len(replay) == 0
    replay.add_batch(**make_transition(2.0, batch=1))
    assert len(replay) == 1
    assert replay.sample(1, "cpu").reward.item() == 2.0
