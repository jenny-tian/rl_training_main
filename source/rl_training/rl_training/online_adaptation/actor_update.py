from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, replace

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
    final_publish_attempt: Callable[[object, int], bool] | None = None,
) -> ActorUpdateOutcome:
    pre_update_state = copy.deepcopy(updater.state_dict())
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
    if stats.accepted and policy_drift is None:
        updater.load_state_dict(pre_update_state)
        raise RuntimeError("accepted actor update has no policy drift")

    if stats.accepted and stats.actor_updated:
        def direct_publish(candidate_actor: object, candidate_version: int) -> bool:
            publisher.publish(candidate_actor, candidate_version)
            return True

        publish_candidate = final_publish_attempt or direct_publish
        try:
            published = publish_candidate(updater.actor, stats.version)
        except BaseException as publication_error:
            updater.load_state_dict(pre_update_state)
            try:
                publisher.publish(updater.actor, int(pre_update_state["version"]))
            except BaseException as resync_error:
                raise RuntimeError(
                    "actor publication failed and publisher resynchronization failed: "
                    f"publication_error={publication_error!r}; resync_error={resync_error!r}"
                ) from resync_error
            raise
        if not published:
            updater.load_state_dict(pre_update_state)
            return ActorUpdateOutcome(
                stats=replace(
                    stats,
                    accepted=False,
                    version=int(pre_update_state["version"]),
                    critic_updated=False,
                    actor_updated=False,
                    rejection_reason="stale_adaptation_episode",
                ),
                policy_drift=None,
            )
        return ActorUpdateOutcome(stats=stats, policy_drift=policy_drift)

    return ActorUpdateOutcome(stats=stats, policy_drift=None)
