import threading
from dataclasses import fields

import pytest
import torch

from rl_training.online_adaptation.replay_buffer import (
    ReplayBuffer,
    sample_stratified_batch,
)


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


def filled_replay(value: float, count: int, *, frozen: bool = False):
    replay = ReplayBuffer(
        capacity=count, history_length=3, frame_dim=2, action_dim=1
    )
    replay.add_batch(**make_transition(value, batch=count))
    if frozen:
        replay.freeze()
    return replay


def test_stratified_batch_uses_requested_flat_fraction():
    flat = filled_replay(value=1.0, count=20, frozen=True)
    online = filled_replay(value=2.0, count=20)

    sampled = sample_stratified_batch(
        flat_replay=flat,
        online_replay=online,
        batch_size=8,
        flat_fraction=0.75,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )

    assert sampled.source_is_flat.sum().item() == 6
    assert sampled.flat_fraction == pytest.approx(0.75)
    assert torch.equal(sampled.transitions.reward[:6], torch.ones(6, 1))
    assert torch.equal(sampled.transitions.reward[6:], torch.full((2, 1), 2.0))


def test_stratified_batch_requires_frozen_flat_replay():
    flat = filled_replay(value=1.0, count=4)
    online = filled_replay(value=2.0, count=4)

    with pytest.raises(ValueError, match="frozen"):
        sample_stratified_batch(
            flat_replay=flat,
            online_replay=online,
            batch_size=4,
            flat_fraction=0.5,
            device="cpu",
        )


STORAGE_KEYS = {
    "history",
    "current_frame",
    "base_action",
    "residual_action",
    "reward",
    "next_history",
    "next_frame",
    "next_base_action",
    "terminated",
    "truncated",
    "bootstrap_mask",
    "discount",
    "safety_mask",
    "curriculum_stage_id",
    "episode_id",
}


def assert_replay_states_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            assert actual[key].device.type == "cpu"
            assert actual[key].dtype == expected[key].dtype
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual[key] == expected[key]


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


def test_replay_rejects_contradictory_terminated_and_truncated_without_mutation():
    replay = make_replay()
    original = make_transition(0.25, batch=1)
    replay.add_batch(**original)
    invalid = make_transition(0.5, batch=1)
    invalid.update(
        terminated=torch.ones(1, 1),
        truncated=torch.ones(1, 1),
        bootstrap_mask=torch.zeros(1, 1),
    )

    with pytest.raises(ValueError, match="terminated.*truncated"):
        replay.add_batch(**invalid)

    assert_single_stored_transition(replay, original)


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


def test_replay_state_round_trip_preserves_complete_storage_and_metadata():
    replay = ReplayBuffer(
        capacity=4,
        history_length=3,
        frame_dim=2,
        action_dim=1,
        storage_dtype=torch.float64,
    )
    first = make_transition(1.25, batch=2)
    first["curriculum_stage_id"] = torch.tensor([[7], [8]], dtype=torch.int64)
    first["episode_id"] = torch.tensor([[2**53 + 1], [2**53 + 3]], dtype=torch.int64)
    replay.add_batch(**first)
    replay.add_batch(**make_transition(3.5, batch=1))
    replay.freeze()

    state = replay.state_dict()
    restored = ReplayBuffer(
        capacity=4,
        history_length=3,
        frame_dim=2,
        action_dim=1,
        storage_dtype=torch.float64,
    )
    restored.load_state_dict(state)

    assert set(state) == {
        "schema_version",
        "capacity",
        "history_length",
        "frame_dim",
        "action_dim",
        "storage_dtype",
        "size",
        "cursor",
        "frozen",
        *STORAGE_KEYS,
    }
    assert state["schema_version"] == 1
    assert state["capacity"] == 4
    assert state["history_length"] == 3
    assert state["frame_dim"] == 2
    assert state["action_dim"] == 1
    assert state["storage_dtype"] == "float64"
    assert state["size"] == 3
    assert state["cursor"] == 3
    assert state["frozen"] is True
    assert state["episode_id"][0, 0].item() == 2**53 + 1
    assert state["episode_id"].dtype == torch.int64
    assert_replay_states_equal(restored.state_dict(), state)
    assert restored.frozen is True


def test_replay_state_dict_returns_detached_cpu_clones_without_aliasing():
    replay = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)
    replay.add_batch(**make_transition(2.0, batch=1))

    first = replay.state_dict()
    second = replay.state_dict()
    original_reward = replay.sample(1, "cpu").reward.clone()
    for key in STORAGE_KEYS:
        assert first[key].device.type == "cpu"
        assert first[key].grad_fn is None
        assert first[key].data_ptr() != second[key].data_ptr()
        first[key].zero_()

    torch.testing.assert_close(replay.sample(1, "cpu").reward, original_reward, rtol=0, atol=0)


def test_frozen_replay_rejects_add_clear_and_atomic_participation_without_mutation():
    frozen = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)
    mutable = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)
    frozen.add_batch(**make_transition(1.0, batch=1))
    frozen.freeze()
    frozen.freeze()
    frozen_before = frozen.state_dict()
    mutable_before = mutable.state_dict()

    with pytest.raises(RuntimeError, match="replay buffer is frozen"):
        frozen.add_batch(**make_transition(2.0, batch=1))
    with pytest.raises(RuntimeError, match="replay buffer is frozen"):
        frozen.clear()
    with pytest.raises(RuntimeError, match="replay buffer is frozen"):
        ReplayBuffer.add_batch_atomically(
            (mutable, frozen),
            **make_transition(2.0, batch=1),
        )

    assert frozen.frozen is True
    assert_replay_states_equal(frozen.state_dict(), frozen_before)
    assert_replay_states_equal(mutable.state_dict(), mutable_before)


def test_loading_frozen_state_into_mutable_empty_replay_is_allowed_and_freezes_it():
    source = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)
    source.add_batch(**make_transition(1.0, batch=1))
    source.freeze()
    restored = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)

    restored.load_state_dict(source.state_dict())

    assert restored.frozen is True
    with pytest.raises(RuntimeError, match="replay buffer is frozen"):
        restored.load_state_dict(source.state_dict())


def _invalid_replay_states(valid_state):
    missing = dict(valid_state)
    missing.pop("reward")
    yield "keys", missing

    extra = dict(valid_state, unexpected=1)
    yield "keys", extra

    for key, value in (
        ("schema_version", 2),
        ("capacity", 3),
        ("history_length", 4),
        ("frame_dim", 3),
        ("action_dim", 2),
        ("storage_dtype", "float32"),
        ("size", 3),
        ("cursor", 2),
        ("frozen", 1),
    ):
        invalid = dict(valid_state)
        invalid[key] = value
        yield key, invalid

    inconsistent = dict(valid_state, size=1, cursor=0)
    yield "cursor", inconsistent

    not_tensor = dict(valid_state)
    not_tensor["reward"] = [[1.0]]
    yield "reward", not_tensor

    wrong_shape = dict(valid_state)
    wrong_shape["history"] = valid_state["history"][:1]
    yield "history", wrong_shape

    wrong_dtype = dict(valid_state)
    wrong_dtype["episode_id"] = valid_state["episode_id"].float()
    yield "episode_id", wrong_dtype

    wrong_device = dict(valid_state)
    wrong_device["history"] = torch.empty(
        valid_state["history"].shape,
        dtype=valid_state["history"].dtype,
        device="meta",
    )
    yield "history", wrong_device

    nonfinite = dict(valid_state)
    nonfinite["reward"] = valid_state["reward"].clone()
    nonfinite["reward"][0] = float("nan")
    yield "reward", nonfinite

    nonbinary = dict(valid_state)
    nonbinary["safety_mask"] = valid_state["safety_mask"].clone()
    nonbinary["safety_mask"][0] = 0.5
    yield "safety_mask", nonbinary

    invalid_discount = dict(valid_state)
    invalid_discount["discount"] = valid_state["discount"].clone()
    invalid_discount["discount"][0] = 1.5
    yield "discount", invalid_discount

    invalid_id = dict(valid_state)
    invalid_id["episode_id"] = valid_state["episode_id"].clone()
    invalid_id["episode_id"][0] = -1
    yield "episode_id", invalid_id

    invalid_ending = dict(valid_state)
    invalid_ending["terminated"] = valid_state["terminated"].clone()
    invalid_ending["bootstrap_mask"] = valid_state["bootstrap_mask"].clone()
    invalid_ending["terminated"][0] = 1
    invalid_ending["bootstrap_mask"][0] = 1
    yield "bootstrap_mask", invalid_ending


def test_load_state_dict_validates_complete_state_before_mutating():
    source = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)
    source.add_batch(**make_transition(4.0, batch=1))
    valid_state = source.state_dict()
    target = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)
    target.add_batch(**make_transition(9.0, batch=1))

    for expected_error, invalid_state in _invalid_replay_states(valid_state):
        before = target.state_dict()
        with pytest.raises((TypeError, ValueError), match=expected_error):
            target.load_state_dict(invalid_state)
        assert_replay_states_equal(target.state_dict(), before)


def test_load_state_dict_rejects_contradictory_endings_without_mutation():
    source = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)
    invalid_state = source.state_dict()
    invalid_state["terminated"][0] = 1
    invalid_state["truncated"][0] = 1
    invalid_state["bootstrap_mask"][0] = 0
    target = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)
    target.add_batch(**make_transition(9.0, batch=1))
    before = target.state_dict()

    with pytest.raises(ValueError, match="terminated.*truncated"):
        target.load_state_dict(invalid_state)

    assert_replay_states_equal(target.state_dict(), before)


def test_load_state_dict_blocks_add_until_validation_and_install_complete(monkeypatch):
    replay = ReplayBuffer(capacity=2, history_length=3, frame_dim=2, action_dim=1)
    state = replay.state_dict()
    validation_entered = threading.Event()
    release_validation = threading.Event()
    add_done = threading.Event()
    validation_lock_states = []
    original_validate = replay._validate_and_clone_state

    def controlled_validation(incoming_state):
        validation_lock_states.append(replay._lock.locked())
        validation_entered.set()
        assert release_validation.wait(timeout=2)
        return original_validate(incoming_state)

    monkeypatch.setattr(replay, "_validate_and_clone_state", controlled_validation)
    load_thread = threading.Thread(target=replay.load_state_dict, args=(state,))
    load_thread.start()
    assert validation_entered.wait(timeout=2)

    def add_row():
        replay.add_batch(**make_transition(7.0, batch=1))
        add_done.set()

    add_thread = threading.Thread(target=add_row)
    add_thread.start()
    committed_during_validation = add_done.wait(timeout=0.1)
    release_validation.set()
    load_thread.join(timeout=2)
    add_thread.join(timeout=2)

    assert not load_thread.is_alive()
    assert not add_thread.is_alive()
    assert validation_lock_states == [True]
    assert committed_during_validation is False
    assert len(replay) == 1
    assert replay.sample(1, "cpu").reward.item() == 7.0
