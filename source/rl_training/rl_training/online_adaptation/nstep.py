from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any

import torch

from .types import TransitionBatch


@dataclass(frozen=True)
class RawTransition:
    history: torch.Tensor
    current_frame: torch.Tensor
    base_action: torch.Tensor
    residual_action: torch.Tensor
    reward: torch.Tensor
    next_history: torch.Tensor
    next_frame: torch.Tensor
    next_base_action: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    bootstrap_mask: torch.Tensor
    safety_mask: torch.Tensor
    curriculum_stage_id: torch.Tensor
    episode_id: torch.Tensor
    requested_residual_action: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.requested_residual_action is None:
            object.__setattr__(self, "requested_residual_action", self.residual_action)


class NStepAccumulator:
    def __init__(self, *, num_envs: int, n_step: int, gamma: float):
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError(f"num_envs must be a positive int, got {num_envs!r}")
        if type(n_step) is not int or n_step <= 0:
            raise ValueError(f"n_step must be a positive int, got {n_step!r}")
        if isinstance(gamma, bool) or not isinstance(gamma, Real) or gamma < 0 or gamma > 1:
            raise ValueError(f"gamma must be a real finite scalar in [0, 1], got {gamma!r}")
        if float(gamma) != float(gamma):
            raise ValueError(f"gamma must be a real finite scalar in [0, 1], got {gamma!r}")
        self.num_envs = num_envs
        self.n_step = n_step
        self.gamma = float(gamma)
        self._queues: list[deque[RawTransition]] = [deque() for _ in range(num_envs)]

    def append(self, transition: RawTransition) -> list[TransitionBatch]:
        self._validate_transition(transition)
        per_env_emissions: list[list[TransitionBatch]] = []
        for env_id in range(self.num_envs):
            queue = self._queues[env_id]
            row = self._select_env(transition, env_id)
            queue.append(row)
            emitted: list[TransitionBatch] = []
            if self._is_episode_end(row):
                while queue:
                    emitted.append(self._aggregate(tuple(queue)))
                    queue.popleft()
            elif len(queue) >= self.n_step:
                emitted.append(self._aggregate(tuple(queue)[: self.n_step]))
                queue.popleft()
            per_env_emissions.append(emitted)
        return self._merge_emissions(per_env_emissions)

    def clear(self) -> None:
        """Discard incomplete returns at an external environment boundary."""
        for queue in self._queues:
            queue.clear()

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "n_step": self.n_step,
            "gamma": self.gamma,
            "queues": tuple(
                tuple(self._clone_raw_transition(transition) for transition in queue)
                for queue in self._queues
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("n-step state must be a mapping")
        expected_keys = {"num_envs", "n_step", "gamma", "queues"}
        if set(state) != expected_keys:
            missing = sorted(expected_keys.difference(state))
            extra = sorted(set(state).difference(expected_keys))
            raise ValueError(f"n-step state keys mismatch: missing={missing}, extra={extra}")
        if state["num_envs"] != self.num_envs:
            raise ValueError(f"n-step state num_envs must equal {self.num_envs}")
        if state["n_step"] != self.n_step:
            raise ValueError(f"n-step state n_step must equal {self.n_step}")
        if state["gamma"] != self.gamma:
            raise ValueError(f"n-step state gamma must equal {self.gamma!r}")
        queues = state["queues"]
        if not isinstance(queues, tuple) or len(queues) != self.num_envs:
            raise ValueError("n-step state queues must contain one queue per environment")
        loaded: list[deque[RawTransition]] = []
        for queue in queues:
            if not isinstance(queue, tuple) or len(queue) >= self.n_step:
                raise ValueError("n-step state queues must contain pending transition tuples")
            loaded_queue: deque[RawTransition] = deque()
            for transition in queue:
                if not isinstance(transition, RawTransition):
                    raise TypeError("n-step state queues must contain RawTransition values")
                self._validate_transition(transition, batch_size=1)
                loaded_queue.append(self._clone_raw_transition(transition))
            loaded.append(loaded_queue)
        self._queues = loaded

    def _validate_transition(self, transition: RawTransition, batch_size: int | None = None) -> None:
        if not isinstance(transition, RawTransition):
            raise TypeError("transition must be a RawTransition")
        expected_batch = self.num_envs if batch_size is None else batch_size
        for field in fields(RawTransition):
            value = getattr(transition, field.name)
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                raise ValueError(f"{field.name} must be a batched tensor")
            if value.shape[0] != expected_batch:
                raise ValueError(f"{field.name} must contain {expected_batch} rows")
            if value.requires_grad:
                raise ValueError(f"{field.name} must be detached")

    @staticmethod
    def _select_env(transition: RawTransition, env_id: int) -> RawTransition:
        return RawTransition(
            **{
                field.name: getattr(transition, field.name)
                .detach()
                .narrow(0, env_id, 1)
                .clone()
                for field in fields(RawTransition)
            }
        )

    @staticmethod
    def _clone_raw_transition(transition: RawTransition) -> RawTransition:
        return RawTransition(
            **{
                field.name: getattr(transition, field.name).detach().clone()
                for field in fields(RawTransition)
            }
        )

    @staticmethod
    def _is_episode_end(transition: RawTransition) -> bool:
        return bool((transition.terminated.bool() | transition.truncated.bool()).any().item())

    def _aggregate(self, transitions: tuple[RawTransition, ...]) -> TransitionBatch:
        first = transitions[0]
        last = transitions[-1]
        reward = torch.zeros_like(first.reward, dtype=torch.float32)
        bootstrap_mask = torch.ones_like(first.bootstrap_mask, dtype=torch.float32)
        safety_mask = torch.ones_like(first.safety_mask, dtype=torch.float32)
        for index, transition in enumerate(transitions):
            reward = reward + transition.reward.to(dtype=torch.float32) * (self.gamma**index)
            bootstrap_mask = torch.minimum(bootstrap_mask, transition.bootstrap_mask.to(dtype=torch.float32))
            safety_mask = torch.minimum(safety_mask, transition.safety_mask.to(dtype=torch.float32))
        bootstrap_mask = torch.where(last.terminated.bool(), torch.zeros_like(bootstrap_mask), bootstrap_mask)
        discount = torch.full_like(reward, self.gamma ** len(transitions), dtype=torch.float32)
        return TransitionBatch(
            history=first.history,
            current_frame=first.current_frame,
            base_action=first.base_action,
            residual_action=first.residual_action,
            requested_residual_action=first.requested_residual_action,
            reward=reward,
            next_history=last.next_history,
            next_frame=last.next_frame,
            next_base_action=last.next_base_action,
            terminated=last.terminated,
            truncated=last.truncated,
            bootstrap_mask=bootstrap_mask,
            discount=discount,
            safety_mask=safety_mask,
            curriculum_stage_id=last.curriculum_stage_id,
            episode_id=last.episode_id,
        )

    @staticmethod
    def _merge_emissions(per_env_emissions: list[list[TransitionBatch]]) -> list[TransitionBatch]:
        if not per_env_emissions:
            return []
        merged: list[TransitionBatch] = []
        max_count = max(len(emissions) for emissions in per_env_emissions)
        for index in range(max_count):
            rows = [emissions[index] for emissions in per_env_emissions if index < len(emissions)]
            merged.append(
                TransitionBatch(
                    **{
                        field.name: torch.cat(
                            [getattr(row, field.name) for row in rows],
                            dim=0,
                        )
                        for field in fields(TransitionBatch)
                    }
                )
            )
        return merged
