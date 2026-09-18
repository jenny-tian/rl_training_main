"""Reset-relative completion and progress cost for formal terrain challenges."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


FULL_TRACK_COMPLETION_MODE = "full_track_v1"
PROGRESS_COST_VERSION = "remaining_plus_safety_penalty_v1"
COMPLETION_QUALIFICATION_VERSION = "in_lane_obstacle_interaction_edge_slip_v3"
STEP_COMPLETION_STAGES = frozenset({"step_2cm", "step_4cm", "step_6cm"})
ROUGH_COMPLETION_STAGES = frozenset({"rough_2cm", "rough_4cm"})


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    return normalized


def _positive(value: object, name: str) -> float:
    normalized = _finite_real(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative(value: object, name: str) -> float:
    normalized = _finite_real(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


@dataclass
class TrackCompletion:
    """Track a single reset-relative formal challenge.

    `max_x` is deliberately monotonic. A robot that backs up after reaching a
    target remains a completed traversal, and the worker can stop before it
    walks off the end of the generated terrain.
    """

    stage_name: str
    initial_x: float
    target_x: float
    target_distance: float
    step_edge_distance: float | None = None
    completion_mode: str = FULL_TRACK_COMPLETION_MODE
    progress_cost_version: str = PROGRESS_COST_VERSION
    max_x: float | None = None
    # Formal challenge qualification is enabled when the reset-relative
    # lateral geometry is supplied.  Keeping these fields optional preserves
    # the legacy helper behaviour for callers that only need an X target
    # (for example, unit-test fixtures and generic demonstrations).
    initial_y: float | None = None
    lane_width_m: float | None = None
    track_width_m: float | None = None
    lateral_margin_m: float = 0.35
    challenge_start_distance: float | None = None
    challenge_end_distance: float | None = None
    required_swing_lift_m: float = 0.02
    min_challenge_observation_steps: int = 20
    min_challenge_swing_steps: int = 4
    min_stable_completion_steps: int = 50
    max_completion_velocity_error: float = 0.25
    max_completion_tilt_rms: float = 0.20
    max_completion_foot_slip: float = 0.35
    edge_slip_margin_m: float = 0.35
    max_abs_lateral_deviation_m: float = 0.0
    lateral_violation_steps: int = 0
    edge_slip_events: int = 0
    challenge_observation_steps: int = 0
    challenge_swing_steps: int = 0
    stable_completion_steps: int = 0

    @classmethod
    def start(
        cls,
        *,
        stage_name: str,
        initial_x: float,
        flat_length: float,
        step_length: float,
        track_length: float,
        step_clearance_margin_m: float,
        track_end_buffer_m: float,
        completion_mode: str = FULL_TRACK_COMPLETION_MODE,
        progress_cost_version: str = PROGRESS_COST_VERSION,
        initial_y: float | None = None,
        lane_width_m: float | None = None,
        track_width_m: float | None = None,
        lateral_margin_m: float = 0.35,
        required_swing_lift_m: float = 0.02,
        min_challenge_observation_steps: int = 20,
        min_challenge_swing_steps: int = 4,
        min_stable_completion_steps: int = 50,
        max_completion_velocity_error: float = 0.25,
        max_completion_tilt_rms: float = 0.20,
        max_completion_foot_slip: float = 0.35,
        edge_slip_margin_m: float = 0.35,
    ) -> "TrackCompletion":
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("stage_name must be a nonempty string")
        initial = _finite_real(initial_x, "initial_x")
        flat = _positive(flat_length, "flat_length")
        step = _positive(step_length, "step_length")
        track = _positive(track_length, "track_length")
        clearance = _nonnegative(step_clearance_margin_m, "step_clearance_margin_m")
        buffer = _nonnegative(track_end_buffer_m, "track_end_buffer_m")
        if not isinstance(completion_mode, str) or not completion_mode.strip():
            raise ValueError("completion_mode must be a nonempty string")
        if not isinstance(progress_cost_version, str) or not progress_cost_version.strip():
            raise ValueError("progress_cost_version must be a nonempty string")

        if initial_y is not None:
            initial_y = _finite_real(initial_y, "initial_y")
        if (lane_width_m is None) != (track_width_m is None):
            raise ValueError("lane_width_m and track_width_m must be supplied together")
        strict = lane_width_m is not None
        if strict:
            lane_width_m = _positive(lane_width_m, "lane_width_m")
            track_width_m = _positive(track_width_m, "track_width_m")
            if lane_width_m > track_width_m:
                raise ValueError("lane_width_m must not exceed track_width_m")
            lateral_margin_m = _nonnegative(lateral_margin_m, "lateral_margin_m")
            if lateral_margin_m >= lane_width_m / 2.0:
                raise ValueError("lateral_margin_m leaves no valid center corridor")
        elif initial_y is not None:
            raise ValueError("initial_y requires lane_width_m and track_width_m")
        required_swing_lift_m = _positive(required_swing_lift_m, "required_swing_lift_m")
        for name, value in (
            ("min_challenge_observation_steps", min_challenge_observation_steps),
            ("min_challenge_swing_steps", min_challenge_swing_steps),
            ("min_stable_completion_steps", min_stable_completion_steps),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        max_completion_velocity_error = _nonnegative(
            max_completion_velocity_error, "max_completion_velocity_error"
        )
        max_completion_tilt_rms = _nonnegative(
            max_completion_tilt_rms, "max_completion_tilt_rms"
        )
        max_completion_foot_slip = _nonnegative(
            max_completion_foot_slip, "max_completion_foot_slip"
        )
        edge_slip_margin_m = _nonnegative(edge_slip_margin_m, "edge_slip_margin_m")

        if stage_name in STEP_COMPLETION_STAGES:
            target_distance = flat / 2.0 + step + clearance
            step_edge_distance = flat / 2.0
            challenge_start_distance = step_edge_distance
            challenge_end_distance = flat / 2.0 + step
        elif stage_name in ROUGH_COMPLETION_STAGES:
            target_distance = track - flat / 2.0 - buffer
            step_edge_distance = None
            challenge_start_distance = flat / 2.0 + step
            challenge_end_distance = target_distance
        else:
            raise ValueError(f"unsupported completion stage: {stage_name!r}")
        if not math.isfinite(target_distance) or target_distance <= 0.0:
            raise ValueError("completion target distance must be positive and finite")
        if flat + step >= track:
            raise ValueError("flat and step segments must leave a non-empty track")
        target = initial + target_distance
        if not math.isfinite(target):
            raise ValueError("completion target must be finite")

        # The generated track extends forward from the reset-relative origin.
        # Reject a target that lies at or beyond its physical end.
        if target_distance >= track:
            raise ValueError("completion target lies outside the track interior")
        if stage_name in STEP_COMPLETION_STAGES and flat + step + clearance >= track:
            raise ValueError("step completion target lies outside the track interior")

        return cls(
            stage_name=stage_name,
            initial_x=initial,
            target_x=target,
            target_distance=target_distance,
            step_edge_distance=step_edge_distance,
            completion_mode=completion_mode,
            progress_cost_version=progress_cost_version,
            max_x=initial,
            initial_y=initial_y,
            lane_width_m=lane_width_m,
            track_width_m=track_width_m,
            lateral_margin_m=lateral_margin_m,
            challenge_start_distance=challenge_start_distance,
            challenge_end_distance=challenge_end_distance,
            required_swing_lift_m=required_swing_lift_m,
            min_challenge_observation_steps=min_challenge_observation_steps,
            min_challenge_swing_steps=min_challenge_swing_steps,
            min_stable_completion_steps=min_stable_completion_steps,
            max_completion_velocity_error=max_completion_velocity_error,
            max_completion_tilt_rms=max_completion_tilt_rms,
            max_completion_foot_slip=max_completion_foot_slip,
            edge_slip_margin_m=edge_slip_margin_m,
        )

    @property
    def step_edge_x(self) -> float | None:
        if self.step_edge_distance is None:
            return None
        return self.initial_x + self.step_edge_distance

    @property
    def past_step_edge(self) -> bool:
        edge_x = self.step_edge_x
        return edge_x is not None and self.max_x is not None and self.max_x >= edge_x

    def observe(self, base_x: float) -> None:
        observed = _finite_real(base_x, "base_x")
        if self.max_x is None or observed > self.max_x:
            self.max_x = observed

    @property
    def strict_qualification_enabled(self) -> bool:
        return (
            self.initial_y is not None
            and self.lane_width_m is not None
            and self.track_width_m is not None
        )

    @property
    def allowed_lateral_deviation_m(self) -> float | None:
        if not self.strict_qualification_enabled:
            return None
        assert self.lane_width_m is not None
        return self.lane_width_m / 2.0 - self.lateral_margin_m

    @property
    def challenge_start_x(self) -> float | None:
        if self.challenge_start_distance is None:
            return None
        return self.initial_x + self.challenge_start_distance

    @property
    def challenge_end_x(self) -> float | None:
        if self.challenge_end_distance is None:
            return None
        return self.initial_x + self.challenge_end_distance

    @staticmethod
    def _max_abs_sequence_deviation(
        values: object, *, reference: float, name: str
    ) -> float:
        if values is None:
            return 0.0
        if isinstance(values, Real):
            values = (values,)
        try:
            values = tuple(values)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite scalar or iterable") from error
        deviations = []
        for value in values:
            deviations.append(abs(_finite_real(value, name) - reference))
        return max(deviations, default=0.0)

    @staticmethod
    def _sequence_length(values: object) -> int:
        if values is None or isinstance(values, Real):
            return 0 if values is None else 1
        try:
            return len(values)  # type: ignore[arg-type]
        except TypeError:
            # Unsized iterables are not accepted as complete four-foot pose
            # evidence.  Do not consume them here; the deviation validator
            # below may still report malformed/non-finite values accurately.
            return -1

    @staticmethod
    def _any_at_least(values: object, threshold: float, *, name: str) -> bool:
        if values is None:
            return False
        if isinstance(values, Real):
            values = (values,)
        try:
            values = tuple(values)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite scalar or iterable") from error
        return any(_finite_real(value, name) >= threshold for value in values)

    def observe_pose(
        self,
        base_x: float,
        base_y: float,
        *,
        foot_y: object = None,
        swing_lift: object = None,
        velocity_error: float | None = None,
        tilt_rms: float | None = None,
        foot_slip: float | None = None,
    ) -> None:
        """Record path, obstacle-interaction, and stable-gait evidence.

        The strict gate is deliberately based on reset-relative geometry and
        observable foot motion.  It does not use a terrain oracle, so a
        lateral bypass or a stationary slide cannot satisfy the formal pass.
        """
        self.observe(base_x)
        if not self.strict_qualification_enabled:
            return
        assert self.initial_y is not None
        start_x = self.challenge_start_x
        end_x = self.challenge_end_x
        if start_x is None or end_x is None:
            raise ValueError("strict completion is missing challenge geometry")
        current_x = _finite_real(base_x, "base_x")
        current_y = _finite_real(base_y, "base_y")
        deviation = max(
            abs(current_y - self.initial_y),
            self._max_abs_sequence_deviation(
                foot_y, reference=self.initial_y, name="foot_y"
            ),
        )
        self.max_abs_lateral_deviation_m = max(
            self.max_abs_lateral_deviation_m, deviation
        )
        # A high slip signal while crossing either face of a raised step is a
        # failed edge interaction.  Keep this evidence sticky: reaching the
        # target after sliding off the edge must not turn into a success.
        edge_start = self.step_edge_x
        edge_end = self.challenge_end_x
        near_edge = (
            edge_start is not None
            and edge_end is not None
            and (
                abs(current_x - edge_start) <= self.edge_slip_margin_m
                or abs(current_x - edge_end) <= self.edge_slip_margin_m
            )
        )
        if (
            self.stage_name in STEP_COMPLETION_STAGES
            and near_edge
            and foot_slip is not None
            and _finite_real(foot_slip, "foot_slip") > self.max_completion_foot_slip
        ):
            self.edge_slip_events += 1
        # Edge windows extend to both sides of the obstacle, including the
        # approach and the landing after its far edge.  Check them before
        # restricting gait evidence to the challenge interior.
        if not start_x <= current_x <= end_x:
            return
        self.challenge_observation_steps += 1
        allowed = self.allowed_lateral_deviation_m
        assert allowed is not None
        # A strict pass must be auditable from both the body and all feet.  If
        # the foot pose sensor is unavailable for a challenge sample, fail
        # closed rather than silently treating the missing values as centred;
        # otherwise a lateral bypass could evade the lane check.
        # Lite3 has four feet.  A missing or partial foot pose is evidence
        # failure, rather than an implicit centred pose that could hide a
        # lateral bypass.
        if self._sequence_length(foot_y) != 4 or deviation > allowed:
            self.lateral_violation_steps += 1
        if self._any_at_least(
            swing_lift, self.required_swing_lift_m, name="swing_lift"
        ):
            self.challenge_swing_steps += 1
        if velocity_error is not None and tilt_rms is not None and foot_slip is not None:
            stable_values = (
                _finite_real(velocity_error, "velocity_error"),
                _finite_real(tilt_rms, "tilt_rms"),
                _finite_real(foot_slip, "foot_slip"),
            )
            if (
                stable_values[0] <= self.max_completion_velocity_error
                and stable_values[1] <= self.max_completion_tilt_rms
                and stable_values[2] <= self.max_completion_foot_slip
                and deviation <= allowed
            ):
                self.stable_completion_steps += 1

    @property
    def reached(self) -> bool:
        return self.max_x is not None and self.max_x >= self.target_x

    @property
    def remaining_m(self) -> float:
        current = self.initial_x if self.max_x is None else self.max_x
        return max(0.0, self.target_x - current)

    def cost(self, *, fell: bool = False, safety_violation: bool = False) -> float:
        if type(fell) is not bool or type(safety_violation) is not bool:
            raise TypeError("fell and safety_violation must be bools")
        qualification_failed = self.strict_qualification_enabled and not self.qualification_satisfied
        failure_penalty = (
            self.target_distance if fell or safety_violation or qualification_failed else 0.0
        )
        result = self.remaining_m + failure_penalty
        if not math.isfinite(result):
            raise ValueError("completion cost must be finite")
        return result

    def evidence(self, *, fell: bool = False, safety_violation: bool = False) -> dict[str, object]:
        return {
            "initial_x": self.initial_x,
            "max_x": self.max_x if self.max_x is not None else self.initial_x,
            "completion_target_x": self.target_x,
            "completion_remaining_m": self.remaining_m,
            "completion_reached": self.reached,
            "completion_mode": self.completion_mode,
            "progress_cost_version": self.progress_cost_version,
            "cost": self.cost(fell=fell, safety_violation=safety_violation),
            "completion_qualification_version": COMPLETION_QUALIFICATION_VERSION,
            "strict_qualification_enabled": self.strict_qualification_enabled,
            "completion_qualification_passed": self.qualification_satisfied,
            "initial_y": self.initial_y,
            "max_abs_lateral_deviation_m": self.max_abs_lateral_deviation_m,
            "allowed_lateral_deviation_m": self.allowed_lateral_deviation_m,
            "lateral_violation_steps": self.lateral_violation_steps,
            "edge_slip_events": self.edge_slip_events,
            "edge_slip_detected": self.edge_slip_events > 0,
            "challenge_start_x": self.challenge_start_x,
            "challenge_end_x": self.challenge_end_x,
            "challenge_observation_steps": self.challenge_observation_steps,
            "challenge_swing_steps": self.challenge_swing_steps,
            "stable_completion_steps": self.stable_completion_steps,
        }

    @property
    def qualification_satisfied(self) -> bool:
        if not self.strict_qualification_enabled:
            return True
        return (
            self.lateral_violation_steps == 0
            and self.edge_slip_events == 0
            and self.challenge_observation_steps >= self.min_challenge_observation_steps
            and self.challenge_swing_steps >= self.min_challenge_swing_steps
            and self.stable_completion_steps >= self.min_stable_completion_steps
        )
