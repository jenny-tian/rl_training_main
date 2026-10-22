from __future__ import annotations

import pytest
import torch

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState
from rl_training.source_scan.bridge import FrozenPolicySafetyBridge


class _Adapter:
    def __init__(self, *, tilt: float = 0.0):
        self.tilt = tilt

    def safety_state(self, control_timed_out):
        batch = control_timed_out.shape[0]
        device = control_timed_out.device
        return SafetyState(
            projected_gravity_xy=torch.tensor([[self.tilt, 0.0]], device=device).expand(batch, -1),
            base_height=torch.full((batch, 1), 0.4, device=device),
            joint_pos=torch.zeros((batch, 12), device=device),
            joint_vel=torch.zeros((batch, 12), device=device),
            joint_lower=torch.full((batch, 12), -10.0, device=device),
            joint_upper=torch.full((batch, 12), 10.0, device=device),
            default_joint_pos=torch.zeros((batch, 12), device=device),
            action_scale=torch.ones((batch, 12), device=device),
            nonfoot_contact=torch.zeros((batch, 1), dtype=torch.bool, device=device),
            control_timed_out=control_timed_out,
        )


def _bridge(policy, *, tilt: float = 0.0):
    safety = SafetyFilter(SafetyLimits(residual_limits=torch.ones(12)))
    return FrozenPolicySafetyBridge(policy, _Adapter(tilt=tilt), safety)


def test_bridge_clips_and_traces_frozen_base_action_before_filtering():
    raw = torch.tensor([[99.0, -99.0, 0.2] * 4])
    bridge = _bridge(lambda _observation: raw)

    trace = bridge.step(torch.zeros(1, 45), torch.zeros((1, 1), dtype=torch.bool))

    assert torch.equal(trace.raw_policy_action, raw)
    assert torch.equal(trace.base_action, clip_lite3_policy_action(raw))
    assert torch.equal(trace.requested_action, trace.base_action)
    assert torch.equal(trace.executed_action, trace.base_action)
    assert not trace.decision.paused.any()
    assert torch.equal(trace.filter_state_before["filter_previous_residual_action"], torch.zeros(1, 12))
    assert trace.action_latency_ms >= 0.0


def test_bridge_rejects_nonfinite_or_non_lite3_policy_actions():
    timeout = torch.zeros((1, 1), dtype=torch.bool)
    with pytest.raises(FloatingPointError, match="non-finite"):
        _bridge(lambda _observation: torch.full((1, 12), float("nan"))).step(
            torch.zeros(1, 45), timeout
        )
    with pytest.raises(ValueError, match="12 columns"):
        _bridge(lambda _observation: torch.zeros(1, 11)).step(
            torch.zeros(1, 45), timeout
        )


def test_bridge_reports_safety_pause_for_caller_to_stop_before_execution():
    bridge = _bridge(lambda _observation: torch.zeros(1, 12), tilt=0.8)

    trace = bridge.step(torch.zeros(1, 45), torch.zeros((1, 1), dtype=torch.bool))

    assert trace.decision.paused.tolist() == [[True]]
    assert "tilt" in trace.decision.reasons
    # The caller can inspect paused before passing the candidate action to env.step().
    assert trace.executed_action.shape == (1, 12)
