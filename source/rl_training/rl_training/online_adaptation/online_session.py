"""Rollout buffering and publication boundary for online PPO.

The Isaac Lab worker can keep control-time code small by handing complete
``PPORollout`` blocks to this class.  A block is collected with one immutable
policy version; learning runs on the training copy and a successful update is
published only after the updater accepts its KL and finite-parameter checks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .ppo_online import OnlinePPOUpdater, PPOUpdateStats
from .ppo_rollout import PPORollout
from .publisher import DoubleBufferedActorPublisher


@dataclass(frozen=True)
class SessionCounters:
    transitions_collected: int = 0
    update_attempts: int = 0
    accepted_updates: int = 0
    rejected_updates: int = 0
    discarded_transitions: int = 0


def _cat(values: list[Tensor], name: str) -> Tensor:
    if not values:
        raise ValueError(f"cannot concatenate empty {name} list")
    return torch.cat(values, dim=0)


class OnlinePPOSession:
    """Own the on-policy buffer and the policy publication/version boundary."""

    def __init__(
        self,
        updater: OnlinePPOUpdater,
        *,
        rollout_capacity: int = 256,
        publisher: DoubleBufferedActorPublisher | None = None,
    ) -> None:
        if not isinstance(updater, OnlinePPOUpdater):
            raise TypeError("updater must be an OnlinePPOUpdater")
        if isinstance(rollout_capacity, bool) or not isinstance(rollout_capacity, int) or rollout_capacity <= 0:
            raise ValueError("rollout_capacity must be a positive integer")
        self.updater = updater
        self.rollout_capacity = rollout_capacity
        self.publisher = publisher
        self._blocks: list[PPORollout] = []
        self._num_pending = 0
        self._pending_version: int | None = None
        self.counters = SessionCounters()

    @property
    def pending_transitions(self) -> int:
        return self._num_pending

    @property
    def ready(self) -> bool:
        return self._num_pending >= self.rollout_capacity

    def clear(self, *, discarded: bool = True) -> None:
        if discarded and self._num_pending:
            self.counters = SessionCounters(
                **{
                    **self.counters.__dict__,
                    "discarded_transitions": self.counters.discarded_transitions + self._num_pending,
                }
            )
        self._blocks.clear()
        self._num_pending = 0
        self._pending_version = None

    def append(self, rollout: PPORollout) -> None:
        if not isinstance(rollout, PPORollout):
            raise TypeError("append expects a PPORollout")
        if rollout.length == 0:
            return
        versions = rollout.version_tensor()
        if not bool((versions == self.updater.policy_version).all().item()):
            raise ValueError("rollout policy version does not match the current updater")
        if self._pending_version is None:
            self._pending_version = self.updater.policy_version
        if self._pending_version != self.updater.policy_version:
            raise ValueError("pending rollout crosses a policy publication boundary")
        if self._num_pending + rollout.length > self.rollout_capacity:
            raise ValueError(
                f"rollout capacity exceeded: pending={self._num_pending}, incoming={rollout.length}, "
                f"capacity={self.rollout_capacity}"
            )
        self._blocks.append(rollout)
        self._num_pending += rollout.length
        self.counters = SessionCounters(
            **{
                **self.counters.__dict__,
                "transitions_collected": self.counters.transitions_collected + rollout.length,
            }
        )

    def _combined(self) -> PPORollout:
        if not self._blocks:
            raise ValueError("no pending rollout")
        first = self._blocks[0]
        valid_masks = [block.valid_mask for block in self._blocks]
        valid = None if any(mask is None for mask in valid_masks) else _cat([mask for mask in valid_masks if mask is not None], "valid masks")
        return PPORollout(
            observations=_cat([block.observations for block in self._blocks], "observations"),
            actions=_cat([block.actions for block in self._blocks], "actions"),
            old_log_probs=_cat([block.old_log_probs for block in self._blocks], "old log-probs"),
            old_values=_cat([block.old_values for block in self._blocks], "old values"),
            rewards=_cat([block.rewards for block in self._blocks], "rewards"),
            terminated=_cat([block.terminated for block in self._blocks], "terminated"),
            truncated=_cat([block.truncated for block in self._blocks], "truncated"),
            next_values=_cat([block.next_values for block in self._blocks], "next values"),
            policy_version=self._pending_version if self._pending_version is not None else first.policy_version,
            valid_mask=valid,
        )

    def update(
        self,
        *,
        force: bool = False,
        flat_anchor_observations: Any | None = None,
        flat_anchor_means: Tensor | None = None,
    ) -> PPOUpdateStats | None:
        if not self.ready and not force:
            return None
        rollout = self._combined()
        stats = self.updater.update(
            rollout,
            flat_anchor_observations=flat_anchor_observations,
            flat_anchor_means=flat_anchor_means,
        )
        self.counters = SessionCounters(
            **{
                **self.counters.__dict__,
                "update_attempts": self.counters.update_attempts + 1,
                "accepted_updates": self.counters.accepted_updates + int(stats.accepted),
                "rejected_updates": self.counters.rejected_updates + int(not stats.accepted),
            }
        )
        # Whether accepted or rejected, these rows are tied to a completed
        # update attempt.  Keeping them would silently reuse on-policy data;
        # the next block starts at the published (or unchanged) version.
        self.clear(discarded=False)
        if stats.accepted and self.publisher is not None:
            self.publisher.publish(self.updater.policy, stats.policy_version_after)
        return stats

    def state_dict(self) -> dict[str, Any]:
        pending = None if not self._blocks else self._combined()
        return {
            "schema_version": 1,
            "rollout_capacity": self.rollout_capacity,
            "pending_rollout": None if pending is None else {
                "observations": pending.observations.detach().cpu(),
                "actions": pending.actions.detach().cpu(),
                "old_log_probs": pending.old_log_probs.detach().cpu(),
                "old_values": pending.old_values.detach().cpu(),
                "rewards": pending.rewards.detach().cpu(),
                "terminated": pending.terminated.detach().cpu(),
                "truncated": pending.truncated.detach().cpu(),
                "next_values": pending.next_values.detach().cpu(),
                "policy_version": pending.version_tensor().detach().cpu(),
                "valid_mask": None if pending.valid_mask is None else pending.valid_mask.detach().cpu(),
            },
            "updater": self.updater.state_dict(),
            "counters": self.counters,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("unsupported online PPO session schema")
        if state.get("rollout_capacity") != self.rollout_capacity:
            raise ValueError("online PPO rollout capacity differs from checkpoint")
        counters = state.get("counters")
        if isinstance(counters, dict):
            counters = SessionCounters(**counters)
        if not isinstance(counters, SessionCounters):
            raise ValueError("online PPO session counters are invalid")
        previous = self.state_dict()
        try:
            self.updater.load_state_dict(state["updater"])
            self.clear(discarded=False)
            pending = state.get("pending_rollout")
            if pending is not None:
                if not isinstance(pending, dict):
                    raise ValueError("pending rollout is invalid")
                self.append(PPORollout(
                    observations=pending["observations"],
                    actions=pending["actions"],
                    old_log_probs=pending["old_log_probs"],
                    old_values=pending["old_values"],
                    rewards=pending["rewards"],
                    terminated=pending["terminated"],
                    truncated=pending["truncated"],
                    next_values=pending["next_values"],
                    policy_version=pending["policy_version"],
                    valid_mask=pending.get("valid_mask"),
                ))
            self.counters = counters
        except BaseException:
            self.updater.load_state_dict(previous["updater"])
            self.clear(discarded=False)
            raise


__all__ = ["OnlinePPOSession", "SessionCounters"]
