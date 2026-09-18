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
    rollbacks: int = 0
    terminal_rollbacks: int = 0
    nonfoot_contact_rollbacks: int = 0
    last_rollback_reason: str | None = None


@dataclass(frozen=True)
class PPORollbackStats:
    """Audit record for a safety-triggered restoration."""

    reason: str
    policy_version_before: int
    restored_stable_version: int
    policy_version_after: int
    discarded_transitions: int


@dataclass(frozen=True)
class PPOFinalizationStats:
    """Audit record for the policy selected for the final checkpoint."""

    policy_version_before: int
    stable_version_before: int
    policy_version_after: int
    discarded_transitions: int
    restored_unverified_candidate: bool


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
        # The current policy is a candidate until it survives a complete
        # rollout block.  Keep the previous candidate as the rollback target.
        self._stable_state = copy.deepcopy(self.updater.state_dict())
        self._has_unverified_candidate = False
        if self.publisher is not None:
            self.publisher.publish(self.updater.policy, self.updater.policy_version)

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
        if self._blocks and (
            (self._blocks[0].critic_observations is None) != (rollout.critic_observations is None)
        ):
            raise ValueError("pending rollout cannot mix blocks with and without critic observations")
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
        critic_observations = (
            None if first.critic_observations is None else _cat(
                [block.critic_observations for block in self._blocks if block.critic_observations is not None],
                "critic observations",
            )
        )
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
            critic_observations=critic_observations,
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
        # A candidate is promoted only after it has completed a clean block.
        # This makes a subsequent terminal event roll back the candidate that
        # was active during the failing block, instead of treating it as stable.
        clean_block = self.ready and not bool(
            (rollout.terminated.bool() | rollout.truncated.bool()).any().item()
        )
        if clean_block:
            self._stable_state = copy.deepcopy(self.updater.state_dict())
            self._has_unverified_candidate = False
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
        self._has_unverified_candidate |= bool(stats.accepted)
        return stats

    def rollback(self, reason: str) -> PPORollbackStats:
        """Restore the last stable candidate and invalidate pending samples.

        The restored parameters come from the checkpoint captured before the
        most recent candidate update.  The policy version always advances so
        a control worker cannot accidentally reuse a rollout collected under
        the failed candidate.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("rollback reason must be a non-empty string")
        before = self.updater.policy_version
        stable_version = int(self._stable_state["policy_version"])
        discarded = self._num_pending
        self.updater.load_state_dict(copy.deepcopy(self._stable_state))
        after = before + 1
        self.updater.policy_version = after
        self._has_unverified_candidate = False
        self.clear(discarded=True)
        self.counters = SessionCounters(
            **{
                **self.counters.__dict__,
                "rollbacks": self.counters.rollbacks + 1,
                "terminal_rollbacks": self.counters.terminal_rollbacks + int(reason == "terminated"),
                "nonfoot_contact_rollbacks": self.counters.nonfoot_contact_rollbacks
                + int(reason == "severe_nonfoot_contact"),
                "last_rollback_reason": reason,
            }
        )
        if self.publisher is not None:
            self.publisher.publish(self.updater.policy, after)
        return PPORollbackStats(
            reason=reason,
            policy_version_before=before,
            restored_stable_version=stable_version,
            policy_version_after=after,
            discarded_transitions=discarded,
        )

    def finalize_for_checkpoint(
        self, *, restore_unverified_candidate: bool = True
    ) -> PPOFinalizationStats:
        """Discard an incomplete rollout and select the policy for checkpointing.

        An accepted update remains a candidate until the following complete
        rollout is clean.  At the end of a finite run there may be no such
        rollout, so the default restores the last promoted state before
        serializing. Training supervisors may keep a terminal-informed
        candidate and subject it to an external deterministic probe instead.
        """

        before = self.updater.policy_version
        stable_version = int(self._stable_state["policy_version"])
        discarded = self._num_pending
        restored = self._has_unverified_candidate and restore_unverified_candidate
        if restored:
            self.updater.load_state_dict(copy.deepcopy(self._stable_state))
            self.updater.policy_version = before + 1
        self.clear(discarded=True)
        if restored:
            self._has_unverified_candidate = False
        # Keeping a candidate for an external probe does not verify it.  Its
        # previous stable rollback target must survive serialization/resume.
        if not self._has_unverified_candidate:
            self._stable_state = copy.deepcopy(self.updater.state_dict())
        if self.publisher is not None:
            self.publisher.publish(self.updater.policy, self.updater.policy_version)
        return PPOFinalizationStats(
            policy_version_before=before,
            stable_version_before=stable_version,
            policy_version_after=self.updater.policy_version,
            discarded_transitions=discarded,
            restored_unverified_candidate=restored,
        )

    def state_dict(self) -> dict[str, Any]:
        pending = None if not self._blocks else self._combined()
        return {
            "schema_version": 1,
            "rollout_capacity": self.rollout_capacity,
            "pending_rollout": None if pending is None else {
                "observations": pending.observations.detach().cpu(),
                "critic_observations": (
                    None if pending.critic_observations is None else pending.critic_observations.detach().cpu()
                ),
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
            "stable_state": copy.deepcopy(self._stable_state),
            "has_unverified_candidate": self._has_unverified_candidate,
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
            stable_state = state.get("stable_state")
            if stable_state is None:
                stable_state = copy.deepcopy(state["updater"])
            stable_version = stable_state.get("policy_version")
            active_version = state["updater"].get("policy_version")
            if (
                isinstance(stable_version, bool)
                or not isinstance(stable_version, int)
                or isinstance(active_version, bool)
                or not isinstance(active_version, int)
                or stable_version > active_version
            ):
                raise ValueError("stable online PPO checkpoint version must not exceed active version")
            # Validate the stable checkpoint without changing the active
            # updater, then restore the active state below.
            self.updater.load_state_dict(stable_state)
            self.updater.load_state_dict(state["updater"])
            self._stable_state = copy.deepcopy(stable_state)
            has_unverified_candidate = state.get(
                "has_unverified_candidate", stable_version < active_version
            )
            if not isinstance(has_unverified_candidate, bool):
                raise ValueError("online PPO candidate status is invalid")
            self._has_unverified_candidate = has_unverified_candidate
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
                    critic_observations=pending.get("critic_observations"),
                ))
            self.counters = counters
            if self.publisher is not None:
                self.publisher.publish(self.updater.policy, self.updater.policy_version)
        except BaseException:
            self.updater.load_state_dict(previous["updater"])
            self._stable_state = copy.deepcopy(previous["stable_state"])
            self._has_unverified_candidate = previous["has_unverified_candidate"]
            self.clear(discarded=False)
            raise


__all__ = [
    "OnlinePPOSession",
    "PPOFinalizationStats",
    "PPORollbackStats",
    "SessionCounters",
]
