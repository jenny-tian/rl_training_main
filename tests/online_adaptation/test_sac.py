import copy
import json

import pytest
import torch
from dataclasses import asdict, replace

from rl_training.online_adaptation.metrics import AdaptationPerformanceDecision
from rl_training.online_adaptation.networks import LatentResidualActor, ResidualActor, TwinCritic
from rl_training.online_adaptation.sac import (
    CandidateActorRejected,
    SACConfig,
    SACUpdater,
    StableStateManager,
    compute_sac_target,
)
from rl_training.online_adaptation.types import TransitionBatch


def make_batch(reward_value=1.0):
    return TransitionBatch(
        history=torch.randn(32, 25, 49),
        current_frame=torch.randn(32, 49),
        base_action=torch.randn(32, 12),
        residual_action=torch.zeros(32, 12),
        reward=torch.full((32, 1), reward_value),
        next_history=torch.randn(32, 25, 49),
        next_frame=torch.randn(32, 49),
        next_base_action=torch.randn(32, 12),
        terminated=torch.zeros(32, 1),
        truncated=torch.zeros(32, 1),
        bootstrap_mask=torch.ones(32, 1),
        discount=torch.full((32, 1), 0.99),
        safety_mask=torch.ones(32, 1),
        curriculum_stage_id=torch.zeros(32, 1, dtype=torch.int64),
        episode_id=torch.zeros(32, 1, dtype=torch.int64),
    )


class ConstantActor(torch.nn.Module):
    def __init__(self, log_prob: float):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        self.log_prob = log_prob

    def sample(self, history, current_frame, base_action, deterministic=False):
        batch_size, action_dim = base_action.shape
        residual = self.value.expand(batch_size, action_dim)
        log_prob = self.value.expand(batch_size, 1) * 0.0 + self.log_prob
        return residual, log_prob, residual


class ConstantTwinCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q1_value = torch.nn.Parameter(torch.tensor(0.0))
        self.q2_value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, history, current_frame, base_action, residual_action):
        batch_size = history.shape[0]
        return self.q1_value.expand(batch_size, 1), self.q2_value.expand(batch_size, 1)


class ResidualEchoCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.last_residual = None

    def forward(self, history, current_frame, base_action, residual_action):
        self.last_residual = residual_action.detach().clone()
        value = residual_action.sum(dim=-1, keepdim=True) + self.bias
        return value, value


def assert_state_dict_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_dict_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_state_dict_equal(actual_item, expected_item)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    else:
        assert actual == expected


def test_updater_checkpoint_round_trip_preserves_action_contract(tmp_path):
    updater = SACUpdater(ResidualActor(49, 12), TwinCritic(49, 12), SACConfig())
    assert updater.update(make_batch()).accepted
    checkpoint = tmp_path / "updater.pt"
    updater.save(checkpoint)
    restored = SACUpdater(ResidualActor(49, 12), TwinCritic(49, 12), SACConfig())
    restored.load(checkpoint, map_location="cpu")
    assert restored.state_dict()["schema_version"] == 1
    assert restored.state_dict()["critic_action_semantics"] == "requested_residual_before_safety_v1"
    assert_state_dict_equal(restored.state_dict(), updater.state_dict())


def test_sac_updater_trains_latent_code_actor_against_12d_critic():
    limits = torch.tensor([0.24, 0.20, 0.20] * 4)
    actor = LatentResidualActor(49, 12, code_dim=4, history_dim=16, action_limits=limits)
    critic = TwinCritic(49, 12, history_dim=16)
    updater = SACUpdater(actor, critic, SACConfig(initial_alpha=0.02, alpha_max=0.05))

    stats = updater.update(make_batch())

    assert stats.accepted
    assert stats.actor_updated
    assert actor.decoder_matrix.grad is None


@pytest.mark.parametrize("contract", [
    {},
    {"schema_version": 1},
    {"schema_version": True, "critic_action_semantics": "requested_residual_before_safety_v1"},
    {"schema_version": 2, "critic_action_semantics": "requested_residual_before_safety_v1"},
    {"schema_version": 1, "critic_action_semantics": "executed_residual_after_safety_v1"},
])
def test_updater_load_rejects_incompatible_contract_before_mutation(tmp_path, contract):
    updater = SACUpdater(ResidualActor(49, 12), TwinCritic(49, 12), SACConfig())
    assert updater.update(make_batch()).accepted
    before = copy.deepcopy(updater.state_dict())
    incompatible = copy.deepcopy(before)
    incompatible.pop("schema_version")
    incompatible.pop("critic_action_semantics")
    incompatible.update(contract)
    for tensor in incompatible["actor"].values():
        tensor.add_(1.0)
    checkpoint = tmp_path / "incompatible.pt"
    torch.save(incompatible, checkpoint)
    with pytest.raises(ValueError, match="action semantics/schema"):
        updater.load(checkpoint, map_location="cpu")
    assert_state_dict_equal(updater.state_dict(), before)


@pytest.mark.parametrize("broken_component", ["critic", "alpha_optimizer"])
def test_updater_failed_load_is_atomic(broken_component):
    updater = SACUpdater(ResidualActor(49, 12), TwinCritic(49, 12), SACConfig())
    assert updater.update(make_batch()).accepted
    before = copy.deepcopy(updater.state_dict())
    candidate = copy.deepcopy(before)
    for name in ("actor", "critic", "target_critic"):
        for tensor in candidate[name].values():
            tensor.add_(1.0)
    candidate[broken_component] = {}
    with pytest.raises((RuntimeError, KeyError, ValueError)):
        updater.load_state_dict(candidate)
    assert_state_dict_equal(updater.state_dict(), before)


def test_stable_manager_rejects_legacy_updater_before_mutation():
    updater = SACUpdater(ResidualActor(49, 12), TwinCritic(49, 12), SACConfig())
    manager = StableStateManager()
    manager.checkpoint(updater)
    before = manager.state_dict()
    incompatible = copy.deepcopy(before)
    incompatible["stable_state"].pop("critic_action_semantics")
    with pytest.raises(ValueError, match="action semantics/schema"):
        manager.load_state_dict(incompatible)
    assert_state_dict_equal(manager.state_dict(), before)


def test_target_uses_saved_discount_and_bootstrap_mask():
    batch = make_batch()
    batch = replace(
        batch,
        reward=torch.tensor([[2.0], [2.0]]),
        discount=torch.tensor([[0.9], [0.9**3]]),
        bootstrap_mask=torch.tensor([[0.0], [1.0]]),
    )

    target = compute_sac_target(
        reward=batch.reward,
        discount=batch.discount,
        bootstrap_mask=batch.bootstrap_mask,
        next_q=torch.tensor([[10.0], [10.0]]),
        alpha_log_prob=torch.zeros(2, 1),
    )

    torch.testing.assert_close(target, torch.tensor([[2.0], [2.0 + 0.9**3 * 10.0]]))


def test_configured_q_limits_clamp_bootstrapped_critic_target():
    updater = SACUpdater(
        ConstantActor(log_prob=0.0),
        ConstantTwinCritic(),
        SACConfig(critic_lr=0.0, target_q_min=-10.0, target_q_max=5.0),
    )
    with torch.no_grad():
        updater.target_critic.q1_value.fill_(1_000.0)
        updater.target_critic.q2_value.fill_(1_000.0)

    stats = updater.update(make_batch(), update_actor=False)

    assert stats.accepted
    assert stats.mean_target_q == pytest.approx(5.0)
    assert stats.mean_td_error == pytest.approx(5.0)
    assert stats.target_q_clip_fraction == 1.0
    assert stats.mean_unclipped_target_q > 5.0


def test_low_temperature_preserves_reward_differences_that_old_cap_erased():
    batch = replace(make_batch(), reward=torch.linspace(0.0, 2.0, 32).reshape(-1, 1))
    updater = SACUpdater(ConstantActor(-8.0), ConstantTwinCritic(),
                         SACConfig(critic_lr=0.0, initial_alpha=0.02, alpha_max=0.05))
    with torch.no_grad():
        updater.target_critic.q1_value.fill_(300.0)
        updater.target_critic.q2_value.fill_(300.0)
    stats = updater.update(batch, update_actor=False)
    assert stats.accepted
    assert stats.mean_entropy_bonus == pytest.approx(0.16)
    assert stats.target_q_clip_fraction == 0.0
    assert stats.mean_unclipped_target_q == stats.mean_target_q


def test_terminal_failure_is_not_discarded_by_squared_loss_guard():
    # One terminal failure amid 127 ordinary transitions: Q is initially 300.
    # Squared loss exceeds 800 and rejects the very evidence needed to lower Q.
    batch = make_batch()
    from dataclasses import fields
    batch = replace(batch, **{f.name: getattr(batch, f.name).repeat_interleave(4, dim=0)
                              for f in fields(batch)})
    reward = torch.full((128, 1), 3.0)
    reward[0] = -2.0
    bootstrap = torch.ones(128, 1)
    bootstrap[0] = 0
    batch = replace(batch, reward=reward, bootstrap_mask=bootstrap)
    for delta, accepted in [(None, False), (1.0, True)]:
        updater = SACUpdater(ConstantActor(0), ConstantTwinCritic(),
                             SACConfig(critic_huber_delta=delta, max_critic_loss=800))
        for critic in (updater.critic, updater.target_critic):
            with torch.no_grad():
                critic.q1_value.fill_(300)
                critic.q2_value.fill_(300)
        stats = updater.update(batch, update_actor=False)
        assert stats.accepted is accepted
        if accepted:
            assert updater.critic.q1_value.item() < 300
        else:
            assert stats.rejection_reason == "critic_loss_limit"


def test_temperature_ceiling_applies_after_optimizer_and_survives_restore():
    config = SACConfig(initial_alpha=0.02, alpha_max=0.02, target_entropy=100.0, alpha_lr=0.5)
    updater = SACUpdater(ResidualActor(49, 12), TwinCritic(49, 12), config)
    assert updater.update(make_batch()).accepted
    assert updater.alpha.item() == pytest.approx(0.02)
    saved = copy.deepcopy(updater.state_dict())
    restored = SACUpdater(ResidualActor(49, 12), TwinCritic(49, 12), config)
    restored.load_state_dict(saved)
    assert restored.alpha.item() == pytest.approx(0.02)
    incompatible = copy.deepcopy(saved)
    incompatible["log_alpha"].fill_(0.0)
    with pytest.raises(ValueError, match="temperature exceeds"):
        restored.load_state_dict(incompatible)
    assert_state_dict_equal(restored.state_dict(), saved)


@pytest.mark.parametrize("options", [{"initial_alpha": 0}, {"initial_alpha": float("nan")},
                                    {"alpha_max": float("inf")}, {"initial_alpha": .1, "alpha_max": .05}])
def test_invalid_temperature_configuration(options):
    with pytest.raises(ValueError, match="alpha"):
        SACConfig(**options)


def test_residual_gain_scales_actor_action_before_target_critic():
    actor = ConstantActor(log_prob=0.0)
    with torch.no_grad():
        actor.value.fill_(1.0)
    critic = ResidualEchoCritic()
    updater = SACUpdater(actor, critic, SACConfig(critic_lr=0.0))

    stats = updater.update(make_batch(), update_actor=False, residual_gain=0.25)

    assert stats.accepted
    assert updater.target_critic.last_residual is not None
    torch.testing.assert_close(
        updater.target_critic.last_residual,
        torch.full_like(updater.target_critic.last_residual, 0.25),
    )


def test_critic_regresses_request_when_safety_changes_executed_action():
    batch = make_batch()
    batch = replace(batch, residual_action=torch.full_like(batch.residual_action, -0.01),
                    requested_residual_action=torch.full_like(batch.residual_action, -0.04))
    critic = ResidualEchoCritic()
    updater = SACUpdater(ConstantActor(0.0), critic, SACConfig(critic_lr=0.0))
    stats = updater.update(batch, update_actor=False, residual_gain=0.25)
    assert stats.accepted
    torch.testing.assert_close(critic.last_residual, batch.requested_residual_action)


def test_actor_and_deterministic_q_use_scaled_requests_and_anchor_keeps_gradient():
    actor = ConstantActor(0.0)
    with torch.no_grad():
        actor.value.fill_(0.2)
    critic = ResidualEchoCritic()
    inputs = []
    hook = critic.register_forward_pre_hook(lambda module, args: inputs.append(args[-1].detach().clone()))
    updater = SACUpdater(actor, critic, SACConfig(critic_lr=0.0, flat_anchor_weight=0.5,
                                               deterministic_actor_weight=0.25))
    batch = make_batch()
    stats = updater.update(batch, flat_anchor_batch=batch, residual_gain=0.25)
    hook.remove()
    assert stats.accepted and stats.actor_updated
    # The live critic sees replay, stochastic request, deterministic request.
    torch.testing.assert_close(inputs[-2], torch.full_like(batch.residual_action, 0.05))
    torch.testing.assert_close(inputs[-1], torch.full_like(batch.residual_action, 0.05))
    assert stats.flat_anchor_loss == pytest.approx(0.5 * 0.05 ** 2)
    assert actor.value.item() > 0.2


@pytest.mark.parametrize("gain", [-0.1, float("nan"), float("inf"), True])
def test_residual_gain_must_be_finite_unit_interval(gain):
    updater = SACUpdater(ConstantActor(log_prob=0.0), ConstantTwinCritic(), SACConfig())
    with pytest.raises(ValueError, match="residual_gain"):
        updater.update(make_batch(), update_actor=False, residual_gain=gain)


def test_configured_huber_loss_bounds_large_td_penalty():
    updater = SACUpdater(
        ConstantActor(log_prob=0.0),
        ConstantTwinCritic(),
        SACConfig(critic_lr=0.0, critic_huber_delta=1.0),
    )

    stats = updater.update(make_batch(reward_value=3.0), update_actor=False)

    assert stats.accepted
    assert stats.critic_loss == pytest.approx(5.0)
    assert stats.mean_td_error == pytest.approx(3.0)


def test_sac_update_changes_version_and_target_parameters():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    critic = TwinCritic(49, 12, 64)
    updater = SACUpdater(actor, critic, SACConfig())
    before = [p.detach().clone() for p in updater.target_critic.parameters()]
    stats = updater.update(make_batch())
    assert stats.accepted
    assert stats.critic_updated
    assert stats.actor_updated
    assert stats.version == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, updater.target_critic.parameters()))


def test_flat_anchor_penalizes_deterministic_residual_mean():
    actor = ConstantActor(log_prob=0.0)
    with torch.no_grad():
        actor.value.fill_(0.1)
    updater = SACUpdater(
        actor, ConstantTwinCritic(), SACConfig(flat_anchor_weight=2.0)
    )

    stats = updater.update(
        critic_batch=make_batch(),
        actor_batch=make_batch(),
        flat_anchor_batch=make_batch(),
    )

    assert stats.accepted
    assert stats.flat_anchor_loss == pytest.approx(0.02)


def test_zero_anchor_weight_matches_legacy_actor_loss():
    first = SACUpdater(
        ConstantActor(log_prob=0.5), ConstantTwinCritic(),
        SACConfig(flat_anchor_weight=0.0),
    )
    second = copy.deepcopy(first)
    batch = make_batch()
    torch.manual_seed(123)
    legacy = first.update(batch, batch)
    torch.manual_seed(123)
    anchored = second.update(batch, batch, flat_anchor_batch=batch)

    assert anchored.actor_loss == pytest.approx(legacy.actor_loss)
    assert anchored.flat_anchor_loss == 0.0


def test_deterministic_actor_term_optimizes_mean_action_q():
    baseline = SACUpdater(
        ConstantActor(log_prob=0.5),
        ConstantTwinCritic(),
        SACConfig(critic_lr=0.0),
    )
    with torch.no_grad():
        baseline.critic.q1_value.fill_(2.0)
        baseline.critic.q2_value.fill_(2.0)
    weighted = copy.deepcopy(baseline)
    weighted.config = replace(weighted.config, deterministic_actor_weight=0.25)
    batch = make_batch()

    baseline_stats = baseline.update(batch)
    weighted_stats = weighted.update(batch)

    assert baseline_stats.accepted
    assert weighted_stats.accepted
    assert weighted_stats.actor_loss == pytest.approx(baseline_stats.actor_loss - 0.5)


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_flat_anchor_weight_must_be_finite_and_nonnegative(weight):
    with pytest.raises(ValueError, match="flat_anchor_weight"):
        SACConfig(flat_anchor_weight=weight)


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf"), True])
def test_deterministic_actor_weight_must_be_finite_and_nonnegative(weight):
    with pytest.raises(ValueError, match="deterministic_actor_weight"):
        SACConfig(deterministic_actor_weight=weight)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_q_min": float("nan")}, "target_q_min"),
        ({"target_q_max": float("inf")}, "target_q_max"),
        ({"target_q_min": 5.0, "target_q_max": 5.0}, "less than"),
        ({"target_q_min": 6.0, "target_q_max": 5.0}, "less than"),
    ],
)
def test_q_target_limits_must_be_finite_and_ordered(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SACConfig(**kwargs)


def test_nonfinite_flat_anchor_batch_rolls_back_complete_update():
    updater = SACUpdater(
        ConstantActor(log_prob=0.0), ConstantTwinCritic(),
        SACConfig(flat_anchor_weight=2.0),
    )
    before = copy.deepcopy(updater.state_dict())
    flat_batch = make_batch()
    flat_batch = replace(
        flat_batch,
        current_frame=torch.full_like(flat_batch.current_frame, float("nan")),
    )

    stats = updater.update(make_batch(), make_batch(), flat_batch)

    assert not stats.accepted
    assert stats.rejection_reason == "nonfinite_update"
    assert_state_dict_equal(updater.state_dict(), before)


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (SACConfig(max_critic_loss=1.0), "critic_loss_limit"),
        (SACConfig(max_mean_td_error=0.5), "mean_td_error_limit"),
        (SACConfig(max_critic_grad_norm=0.5), "critic_grad_norm_limit"),
    ],
)
def test_finite_but_unstable_critic_update_restores_complete_state(config, reason):
    updater = SACUpdater(ConstantActor(log_prob=0.0), ConstantTwinCritic(), config)
    before = copy.deepcopy(updater.state_dict())

    stats = updater.update(make_batch(), update_actor=True)

    assert not stats.accepted
    assert not stats.critic_updated
    assert not stats.actor_updated
    assert stats.version == 0
    assert stats.rejection_reason == reason
    assert stats.critic_loss > 0.0
    assert stats.mean_target_q == pytest.approx(1.0)
    assert stats.mean_td_error == pytest.approx(1.0)
    if reason == "critic_grad_norm_limit":
        assert stats.grad_norm > config.max_critic_grad_norm
    assert_state_dict_equal(updater.state_dict(), before)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_critic_loss", 0.0),
        ("max_mean_td_error", float("nan")),
        ("max_critic_grad_norm", float("inf")),
        ("critic_huber_delta", 0.0),
    ],
)
def test_critic_stability_limits_must_be_finite_and_positive(field, value):
    with pytest.raises(ValueError, match=field):
        SACConfig(**{field: value})


def test_critic_target_uses_bootstrap_mask_and_per_transition_discount():
    updater = SACUpdater(
        ConstantActor(log_prob=2.0),
        ConstantTwinCritic(),
        SACConfig(gamma=0.01, critic_lr=0.0),
    )
    with torch.no_grad():
        updater.target_critic.q1_value.fill_(10.0)
        updater.target_critic.q2_value.fill_(12.0)
    batch = TransitionBatch(
        history=torch.zeros(3, 1, 1),
        current_frame=torch.zeros(3, 1),
        base_action=torch.zeros(3, 1),
        residual_action=torch.zeros(3, 1),
        reward=torch.ones(3, 1),
        next_history=torch.zeros(3, 1, 1),
        next_frame=torch.zeros(3, 1),
        next_base_action=torch.zeros(3, 1),
        terminated=torch.tensor([[1.0], [0.0], [0.0]]),
        truncated=torch.tensor([[0.0], [1.0], [0.0]]),
        bootstrap_mask=torch.tensor([[0.0], [1.0], [1.0]]),
        discount=torch.tensor([[0.2], [0.5], [0.8]]),
        safety_mask=torch.tensor([[1.0], [0.0], [1.0]]),
        curriculum_stage_id=torch.tensor([[3], [4], [5]], dtype=torch.int64),
        episode_id=torch.tensor([[30], [40], [50]], dtype=torch.int64),
    )

    stats = updater.update(batch, update_actor=False)

    expected_target = torch.tensor([[1.0], [5.0], [7.4]])
    expected_loss = 2.0 * expected_target.square().mean()
    assert stats.accepted
    assert stats.critic_loss == pytest.approx(expected_loss.item())
    assert stats.mean_target_q == pytest.approx(expected_target.mean().item())
    assert stats.mean_td_error == pytest.approx(expected_target.mean().item())


def test_nonfinite_batch_is_rejected_without_version_change():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    stats = updater.update(make_batch(float("nan")))
    assert not stats.accepted
    assert stats.version == 0
    assert stats.rejection_reason == "nonfinite_batch"
    json.dumps(asdict(stats), allow_nan=False)
    assert stats.mean_target_q is None
    assert stats.mean_td_error is None


def test_candidate_validator_rejection_restores_the_full_pre_update_state():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    warmup_stats = updater.update(make_batch())
    assert warmup_stats.accepted
    assert warmup_stats.version == 1
    before = copy.deepcopy(updater.state_dict())
    seen = []

    def validator(candidate, version):
        seen.append((candidate, version))
        return "nonfinite_policy_drift"

    stats = updater.update(make_batch(), candidate_validator=validator)

    assert seen == [(actor, 2)]
    assert not stats.accepted
    assert not stats.critic_updated
    assert not stats.actor_updated
    assert stats.version == 1
    assert stats.rejection_reason == "nonfinite_policy_drift"
    assert_state_dict_equal(updater.state_dict(), before)


def test_critic_only_update_does_not_call_candidate_validator():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    called = False

    def validator(candidate, version):
        nonlocal called
        called = True
        raise AssertionError("critic-only updates must not validate actor candidates")

    stats = updater.update(make_batch(), update_actor=False, candidate_validator=validator)

    assert stats.accepted
    assert stats.rejection_reason is None
    assert stats.version == 0
    assert not called


def test_candidate_validator_exception_restores_the_full_pre_update_state():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    before = copy.deepcopy(updater.state_dict())

    def validator(candidate, version):
        raise RuntimeError("validator failed")

    with pytest.raises(RuntimeError, match="validator failed"):
        updater.update(make_batch(), candidate_validator=validator)

    assert_state_dict_equal(updater.state_dict(), before)


def test_candidate_actor_rejected_exception_propagates_and_restores_state():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    before = copy.deepcopy(updater.state_dict())

    def validator(candidate, version):
        raise CandidateActorRejected("validator rejected via exception")

    with pytest.raises(CandidateActorRejected, match="validator rejected via exception"):
        updater.update(make_batch(), candidate_validator=validator)

    assert_state_dict_equal(updater.state_dict(), before)


def test_candidate_validator_floating_point_error_propagates_and_restores_state():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    before = copy.deepcopy(updater.state_dict())

    def validator(candidate, version):
        raise FloatingPointError("validator numerical failure")

    with pytest.raises(FloatingPointError, match="validator numerical failure"):
        updater.update(make_batch(), candidate_validator=validator)

    assert_state_dict_equal(updater.state_dict(), before)


def test_critic_only_update_keeps_actor_and_version_unchanged():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    target_before = [parameter.detach().clone() for parameter in updater.target_critic.parameters()]

    stats = updater.update(critic_batch=make_batch(), actor_batch=None, update_actor=False)

    assert stats.accepted
    assert stats.critic_updated
    assert not stats.actor_updated
    assert stats.version == 0
    assert stats.rejection_reason is None
    assert all(torch.equal(before, after) for before, after in zip(actor_before, actor.parameters()))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, updater.target_critic.parameters())
    )


def test_full_update_accepts_a_separate_actor_batch():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())

    stats = updater.update(critic_batch=make_batch(), actor_batch=make_batch(0.5), update_actor=True)

    assert stats.accepted
    assert stats.critic_updated
    assert stats.actor_updated
    assert stats.version == 1


def test_nonfinite_actor_batch_rolls_back_the_preceding_critic_step():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    before = copy.deepcopy(updater.state_dict())
    actor_batch = make_batch()
    actor_batch = replace(actor_batch, current_frame=torch.full_like(actor_batch.current_frame, float("nan")))

    stats = updater.update(critic_batch=make_batch(), actor_batch=actor_batch, update_actor=True)

    assert not stats.accepted
    assert not stats.critic_updated
    assert not stats.actor_updated
    assert stats.version == 0
    assert stats.rejection_reason == "nonfinite_update"
    assert_state_dict_equal(updater.state_dict(), before)


def test_stable_state_rolls_back_after_sustained_degradation():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    manager = StableStateManager(patience=3, min_delta=0.01)
    manager.observe(score=1.0, updater=updater)
    updater.version = 7
    assert not manager.observe(score=0.9, updater=updater)
    assert not manager.observe(score=0.8, updater=updater)
    assert manager.observe(score=0.7, updater=updater)
    assert updater.version == 0


def test_stable_state_can_roll_back_to_explicit_checkpoint():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    manager = StableStateManager()
    manager.checkpoint(updater)
    updater.version = 11

    assert manager.rollback(updater)
    assert updater.version == 0


def test_stable_state_manager_round_trip_restores_checkpoint_and_counters():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    manager = StableStateManager(patience=3, min_delta=0.02)
    manager.observe(score=1.0, updater=updater)
    manager.observe(score=0.5, updater=updater)
    state = manager.state_dict()
    restored = StableStateManager(patience=3, min_delta=0.02)

    restored.load_state_dict(state)
    updater.version = 9

    assert restored.rollback(updater)
    assert updater.version == 0


def test_stable_state_manager_observes_adaptation_decisions():
    actor = ResidualActor(49, 12, 64, torch.tensor([0.2] * 12))
    updater = SACUpdater(actor, TwinCritic(49, 12, 64), SACConfig())
    manager = StableStateManager()
    manager.checkpoint(updater)
    updater.version = 2
    improved = AdaptationPerformanceDecision(
        cost=0.8,
        improved=True,
        degraded=False,
        should_rollback=False,
        rollback_reason=None,
    )

    assert not manager.observe_adaptation(improved, updater)
    updater.version = 3
    rollback = AdaptationPerformanceDecision(
        cost=1.0,
        improved=False,
        degraded=True,
        should_rollback=True,
        rollback_reason="performance_degradation",
    )

    assert manager.observe_adaptation(rollback, updater)
    assert updater.version == 2
