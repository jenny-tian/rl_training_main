import pytest
import torch

from rl_training.online_adaptation.nstep import NStepAccumulator, RawTransition


def make_raw_transition(
    *,
    reward: float = 1.0,
    marker: int = 0,
    terminated: bool = False,
    truncated: bool = False,
    bootstrap_mask: float = 1.0,
    safety_mask: float = 1.0,
    env_count: int = 1,
):
    return RawTransition(
        history=torch.full((env_count, 1, 1), float(marker)),
        current_frame=torch.full((env_count, 1), float(marker)),
        base_action=torch.full((env_count, 1), float(marker)),
        residual_action=torch.full((env_count, 1), float(marker)),
        requested_residual_action=torch.full((env_count, 1), float(marker + 10)),
        reward=torch.full((env_count, 1), reward),
        next_history=torch.full((env_count, 1, 1), float(marker)),
        next_frame=torch.full((env_count, 1), float(marker)),
        next_base_action=torch.full((env_count, 1), float(marker)),
        terminated=torch.full((env_count, 1), float(terminated)),
        truncated=torch.full((env_count, 1), float(truncated)),
        bootstrap_mask=torch.full((env_count, 1), bootstrap_mask),
        safety_mask=torch.full((env_count, 1), safety_mask),
        curriculum_stage_id=torch.full((env_count, 1), marker, dtype=torch.int64),
        episode_id=torch.zeros(env_count, 1, dtype=torch.int64),
    )


@pytest.mark.parametrize("n_step", [1, 3, 5])
def test_nstep_accumulates_reward_discount_and_next_state(n_step):
    accumulator = NStepAccumulator(num_envs=1, n_step=n_step, gamma=0.9)
    emitted = []
    for index in range(n_step):
        emitted.extend(accumulator.append(make_raw_transition(reward=1.0, marker=index)))
    assert len(emitted) == 1
    transition = emitted[0]
    assert transition.reward.item() == pytest.approx(sum(0.9**i for i in range(n_step)))
    assert transition.discount.item() == pytest.approx(0.9**n_step)
    assert transition.next_frame.item() == n_step - 1
    assert transition.requested_residual_action.item() == 10.0


def run_terminal_case(*, terminated: bool, truncated: bool):
    accumulator = NStepAccumulator(num_envs=1, n_step=5, gamma=0.9)
    emitted = accumulator.append(
        make_raw_transition(
            reward=1.0,
            terminated=terminated,
            truncated=truncated,
            bootstrap_mask=1.0,
        )
    )
    assert len(emitted) == 1
    return emitted[0]


def test_true_termination_disables_bootstrap_but_truncation_keeps_it():
    terminated = run_terminal_case(terminated=True, truncated=False)
    truncated = run_terminal_case(terminated=False, truncated=True)
    assert terminated.bootstrap_mask.item() == 0.0
    assert truncated.bootstrap_mask.item() == 1.0


def test_terminal_or_truncated_step_flushes_every_pending_prefix():
    accumulator = NStepAccumulator(num_envs=1, n_step=5, gamma=0.9)
    assert accumulator.append(make_raw_transition(reward=1.0, marker=0)) == []
    emitted = accumulator.append(make_raw_transition(reward=2.0, marker=1, truncated=True))

    assert len(emitted) == 2
    assert emitted[0].reward.item() == pytest.approx(1.0 + 0.9 * 2.0)
    assert emitted[0].discount.item() == pytest.approx(0.9**2)
    assert emitted[0].next_frame.item() == 1
    assert emitted[1].reward.item() == pytest.approx(2.0)
    assert emitted[1].discount.item() == pytest.approx(0.9)
    assert emitted[1].next_frame.item() == 1


def test_nstep_propagates_minimum_bootstrap_and_safety_masks():
    accumulator = NStepAccumulator(num_envs=1, n_step=3, gamma=0.9)
    assert accumulator.append(make_raw_transition(marker=0, bootstrap_mask=1.0, safety_mask=1.0)) == []
    assert accumulator.append(make_raw_transition(marker=1, bootstrap_mask=0.0, safety_mask=1.0)) == []
    emitted = accumulator.append(make_raw_transition(marker=2, bootstrap_mask=1.0, safety_mask=0.0))

    assert len(emitted) == 1
    assert emitted[0].bootstrap_mask.item() == 0.0
    assert emitted[0].safety_mask.item() == 0.0


def test_batched_env_queues_emit_independently():
    accumulator = NStepAccumulator(num_envs=2, n_step=2, gamma=0.9)
    emitted = accumulator.append(make_raw_transition(reward=1.0, marker=0, env_count=2))
    assert emitted == []

    second = make_raw_transition(reward=1.0, marker=1, env_count=2)
    second = RawTransition(
        **{
            **second.__dict__,
            "terminated": torch.tensor([[0.0], [1.0]]),
            "bootstrap_mask": torch.tensor([[1.0], [0.0]]),
        }
    )
    emitted = accumulator.append(second)

    assert len(emitted) == 2
    torch.testing.assert_close(emitted[0].reward, torch.full((2, 1), 1.9))
    torch.testing.assert_close(emitted[0].bootstrap_mask, torch.tensor([[1.0], [0.0]]))
    torch.testing.assert_close(emitted[1].reward, torch.tensor([[1.0]]))
    torch.testing.assert_close(emitted[1].bootstrap_mask, torch.tensor([[0.0]]))


def test_state_dict_preserves_pending_queues():
    source = NStepAccumulator(num_envs=1, n_step=3, gamma=0.9)
    target = NStepAccumulator(num_envs=1, n_step=3, gamma=0.9)
    source.append(make_raw_transition(reward=1.0, marker=0))
    source.append(make_raw_transition(reward=2.0, marker=1))

    target.load_state_dict(source.state_dict())
    emitted = target.append(make_raw_transition(reward=3.0, marker=2))

    assert len(emitted) == 1
    assert emitted[0].reward.item() == pytest.approx(1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)
    assert emitted[0].next_frame.item() == 2
