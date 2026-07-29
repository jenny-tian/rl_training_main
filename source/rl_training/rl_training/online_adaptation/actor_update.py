from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .policy_drift import ActorDriftMonitor, PolicyDrift
from .publisher import DoubleBufferedActorPublisher
from .sac import SACUpdater
from .types import TransitionBatch, UpdateStats


@dataclass(frozen=True)
class ActorUpdateOutcome:
    stats: UpdateStats
    policy_drift: PolicyDrift | None


def update_and_publish_actor(
    *,
    updater: SACUpdater,
    publisher: DoubleBufferedActorPublisher,
    drift_monitor: ActorDriftMonitor,
    critic_batch: TransitionBatch,
    actor_batch: TransitionBatch,
    adaptation_episode: int,
    update_step: int,
    episode_is_active: Callable[[], bool],
) -> ActorUpdateOutcome:
    policy_drift: PolicyDrift | None = None

    def validate_candidate(candidate_actor, candidate_version: int) -> str | None:
        nonlocal policy_drift
        if not episode_is_active():
            return "stale_adaptation_episode"
        try:
            policy_drift = drift_monitor.measure(
                candidate_actor,
                actor_batch,
                actor_version=candidate_version,
                update_step=update_step,
                adaptation_episode=adaptation_episode,
            )
        except FloatingPointError:
            return "nonfinite_policy_drift"
        return None

    stats = updater.update(
        critic_batch,
        actor_batch,
        update_actor=True,
        candidate_validator=validate_candidate,
    )
    if stats.accepted and stats.actor_updated:
        if policy_drift is None:
            raise RuntimeError("accepted actor update has no policy drift")
        publisher.publish(updater.actor, stats.version)
        return ActorUpdateOutcome(stats=stats, policy_drift=policy_drift)

    return ActorUpdateOutcome(stats=stats, policy_drift=None)
