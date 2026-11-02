"""RSL-RL 5.0.1 adapter using actual pre-reset values for timeouts."""
from __future__ import annotations

import importlib.metadata

import torch
from rsl_rl.algorithms import PPO


class SourcePPO(PPO):
    def __init__(self, *args, **kwargs):
        if importlib.metadata.version("rsl-rl-lib") != "5.0.1":
            raise RuntimeError("Source PPO adapter is audited only for RSL-RL 5.0.1")
        super().__init__(*args, **kwargs)

    def process_env_step(self, obs, rewards, dones, extras):
        if "source_terminal" not in extras or "time_outs" not in extras:
            raise RuntimeError("SourcePPO requires the Source training environment adapter")
        timeouts = extras["time_outs"].to(self.device)
        if timeouts.shape != dones.shape or timeouts.dtype != torch.bool:
            raise ValueError("Source timeout mask has invalid shape/dtype")
        if bool((timeouts & ~dones.bool()).any()):
            raise ValueError("Source timeout without a done flag")
        adjusted = rewards.clone()
        if bool(timeouts.any()):
            terminal = extras["source_terminal"]
            if terminal is None or bool((timeouts & terminal["terminated"].to(self.device)).any()):
                raise RuntimeError("missing or contradictory Source timeout snapshot")
            with torch.no_grad():
                terminal_values = self.critic(terminal["observations"].to(self.device)).squeeze(-1)
            if not bool(torch.isfinite(terminal_values).all()):
                raise FloatingPointError("Source terminal critic values are non-finite")
            adjusted += self.gamma * torch.where(timeouts, terminal_values, 0.0)
        # Disable RSL-RL's current-state timeout shortcut. The captured terminal
        # value was added above; done still cuts GAE across the reset boundary.
        forwarded = {key: value for key, value in extras.items() if key != "time_outs"}
        super().process_env_step(obs, adjusted, dones, forwarded)
