"""Checkpointed target-domain collection and critic-only pretraining gates."""

from dataclasses import asdict, dataclass


@dataclass
class TargetWarmup:
    algorithm: str
    required_steps: int = 0
    required_critic_updates: int = 0
    collected_steps: int = 0
    critic_updates: int = 0

    def __post_init__(self):
        if self.algorithm not in {"lora", "sac"}:
            raise ValueError("unknown warmup algorithm")
        for name in ("required_steps", "required_critic_updates", "collected_steps", "critic_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"invalid warmup {name}")
        if self.collected_steps > self.required_steps:
            raise ValueError("warmup collection exceeds its target")
        if self.algorithm == "sac" and self.required_steps:
            if not 1 <= self.required_critic_updates <= 1000:
                raise ValueError("SAC warmup needs 1..1000 critic updates")
            if self.critic_updates > self.required_critic_updates or (self.collecting and self.critic_updates):
                raise ValueError("invalid SAC warmup update progress")

    @property
    def collecting(self):
        return self.collected_steps < self.required_steps

    @property
    def ready(self):
        if not self.required_steps:
            return True
        target = self.required_critic_updates if self.algorithm == "sac" else 1
        return not self.collecting and self.critic_updates >= target

    def record_transition(self):
        if self.collecting:
            self.collected_steps += 1

    def state_dict(self):
        return asdict(self)

    def restore(self, state):
        restored = TargetWarmup(**state)
        for key in ("algorithm", "required_steps", "required_critic_updates"):
            if getattr(restored, key) != getattr(self, key):
                raise ValueError("target warmup configuration changed across resume")
        self.collected_steps = restored.collected_steps
        self.critic_updates = restored.critic_updates


def restore_target_warmup(args, algorithm, checkpoint):
    warmup = TargetWarmup(algorithm, args.target_warmup_steps,
                          args.target_critic_warmup_updates if algorithm == "sac" and args.target_warmup_steps else 0)
    if checkpoint is not None:
        state = checkpoint.get("target_warmup")
        if state is None and warmup.required_steps:
            raise ValueError("checkpoint has no target warmup progress; start an independent session")
        if state is not None:
            warmup.restore(state)
    return warmup


def pretrain_target_critic(warmup, agent, *, on_update):
    """Run once after collection; the caller serializes access to the learner."""
    if warmup.algorithm != "sac" or warmup.collecting:
        raise ValueError("target critic pretraining requires completed SAC collection")
    if warmup.ready:
        return
    if len(agent.replay) < max(agent.config.batch_size, agent.config.min_replay_size):
        raise RuntimeError("target warmup replay has insufficient distinct transitions")
    while not warmup.ready:
        stats = agent.update(update_actor=False)
        on_update(stats)
        if not stats.accepted:
            raise RuntimeError(f"target critic warmup rejected: {stats.rejection_reason}")
        if stats.actor_updated:
            raise RuntimeError("target critic warmup unexpectedly updated actor")
        warmup.critic_updates += 1
