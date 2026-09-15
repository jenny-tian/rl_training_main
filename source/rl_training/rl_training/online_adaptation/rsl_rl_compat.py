from __future__ import annotations

import importlib.metadata as metadata
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn


class RslMLPPolicyAdapter(nn.Module):
    """Tensor-facing adapter for an RSL-RL ``MLPModel`` actor.

    Online PPO stores observations as ordinary ``[T, obs_dim]`` tensors, while
    current RSL-RL models expect a TensorDict keyed by their observation group.
    This wrapper preserves the model's Gaussian distribution and exposes the
    small ``evaluate_actions``/``mean_action`` interface consumed by
    :class:`OnlinePPOUpdater`.
    """

    def __init__(self, model: nn.Module, observation_group: str | None = None) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        groups = getattr(model, "obs_groups", None)
        if observation_group is None:
            if not isinstance(groups, (list, tuple)) or len(groups) != 1 or not isinstance(groups[0], str):
                raise ValueError("RSL actor must expose exactly one observation group or an explicit group name")
            observation_group = groups[0]
        if not isinstance(observation_group, str) or not observation_group:
            raise ValueError("observation_group must be a non-empty string")
        self.model = model
        self.observation_group = observation_group

    def _as_model_observation(self, observations: Any) -> Any:
        if isinstance(observations, Mapping):
            return observations
        if not isinstance(observations, torch.Tensor) or observations.ndim < 2:
            raise ValueError("RSL actor observations must be a TensorDict or [batch, obs_dim] tensor")
        from tensordict import TensorDict

        return TensorDict(
            {self.observation_group: observations},
            batch_size=list(observations.shape[:-1]),
            device=observations.device,
        )

    def forward(self, observations: Any) -> torch.Tensor:
        return self.model(self._as_model_observation(observations), stochastic_output=False)

    def get_distribution(self, observations: Any):
        output = self.forward(observations)
        distribution = getattr(self.model, "distribution", None)
        if distribution is None or not hasattr(distribution, "update"):
            raise RuntimeError("RSL actor does not expose a stochastic distribution")
        distribution.update(output)
        current = getattr(distribution, "_distribution", None)
        if current is None:
            raise RuntimeError("RSL actor distribution was not initialized")
        return current

    def evaluate_actions(self, observations: Any, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.get_distribution(observations)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(dim=-1)
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)
        return log_prob, entropy

    def mean_action(self, observations: Any) -> torch.Tensor:
        return self.forward(observations)


def load_frozen_rsl_rl_checkpoint(runner: Any, checkpoint: str) -> dict[str, object]:
    """Load a frozen PPO checkpoint, tolerating deployment-only critic dimensions.

    The deployed actor only consumes proprioception.  Online-adaptation environments
    may expose additional privileged critic observations, so a checkpoint trained with
    a smaller critic cannot be restored through ``runner.load`` even though its actor
    is valid.  In that case load and validate the actor explicitly and leave the
    runtime critic freshly initialized for the residual learner.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("PPO checkpoint must contain a mapping")
    actor_state = payload.get("actor_state_dict")
    if not isinstance(actor_state, Mapping):
        # Preserve RSL-RL's native handling for older or non-PPO checkpoint layouts.
        runner.load(checkpoint)
        return {"actor_loaded": True, "critic_loaded": True, "critic_skipped": False}

    algorithm = getattr(runner, "alg", None)
    actor = getattr(algorithm, "actor", None)
    if actor is None or not hasattr(actor, "load_state_dict"):
        actor = getattr(algorithm, "policy", None)
    if actor is None or not hasattr(actor, "load_state_dict"):
        raise RuntimeError("unable to locate the RSL-RL actor module")
    try:
        actor.load_state_dict(actor_state, strict=True)
    except RuntimeError as error:
        raise RuntimeError("frozen PPO actor does not match the runtime observation/action schema") from error

    critic_state = payload.get("critic_state_dict")
    critic = getattr(algorithm, "critic", None)
    critic_loaded = False
    critic_skipped = False
    if isinstance(critic_state, Mapping) and critic is not None:
        runtime_state = critic.state_dict()
        shapes_match = all(
            key in runtime_state
            and isinstance(value, torch.Tensor)
            and runtime_state[key].shape == value.shape
            for key, value in critic_state.items()
        ) and set(runtime_state) == set(critic_state)
        if shapes_match:
            critic.load_state_dict(critic_state, strict=True)
            critic_loaded = True
        else:
            critic_skipped = True
    elif critic_state is not None:
        critic_skipped = True

    return {
        "actor_loaded": True,
        "critic_loaded": critic_loaded,
        "critic_skipped": critic_skipped,
    }


def split_rsl_rl_endings(
    done: torch.Tensor,
    extras: Mapping[str, Any],
    *,
    bootstrap_timeouts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split RSL-RL endings without bootstrapping from post-reset observations.

    Isaac Lab's RSL-RL wrapper returns observations after resetting ended
    environments. Timeout bootstrapping is therefore only valid when the caller
    separately replaces those rows with terminal observations.
    """
    if type(bootstrap_timeouts) is not bool:
        raise TypeError("bootstrap_timeouts must be a bool")
    if not isinstance(done, torch.Tensor) or done.ndim not in (1, 2):
        raise RuntimeError("RSL-RL dones must be a tensor with one value per environment")
    if done.ndim == 2 and done.shape[1] != 1:
        raise RuntimeError("RSL-RL dones must be a tensor with one value per environment")
    if done.is_complex() or not bool(torch.isfinite(done).all().item()):
        raise RuntimeError("RSL-RL dones must contain only binary values")
    if not bool(((done == 0) | (done == 1)).all().item()):
        raise RuntimeError("RSL-RL dones must contain only binary values")

    time_outs = extras.get("time_outs") if isinstance(extras, Mapping) else None
    if not isinstance(time_outs, torch.Tensor) or time_outs.shape != done.shape:
        raise RuntimeError("extras['time_outs'] must be a tensor matching dones shape")
    if time_outs.is_complex() or not bool(torch.isfinite(time_outs).all().item()):
        raise RuntimeError("extras['time_outs'] must contain only binary values")
    if not bool(((time_outs == 0) | (time_outs == 1)).all().item()):
        raise RuntimeError("extras['time_outs'] must contain only binary values")

    done_column = done.bool().reshape(-1, 1)
    truncated = time_outs.bool().reshape(-1, 1)
    if bool((truncated & ~done_column).any().item()):
        raise RuntimeError("extras['time_outs'] cannot mark a row that is not done")
    terminated = done_column & ~truncated
    bootstrap_mask = ~terminated if bootstrap_timeouts else ~done_column
    return terminated, truncated, bootstrap_mask


def prepare_rsl_rl_agent_cfg(
    agent_cfg: Any,
    *,
    installed_version: str | None = None,
    converter: Callable[[Any, str], Any] | None = None,
) -> Any:
    """Convert legacy Isaac Lab RSL-RL configs for the installed RSL-RL version."""
    if installed_version is None:
        installed_version = metadata.version("rsl-rl-lib")
    if converter is None:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        converter = handle_deprecated_rsl_rl_cfg
    return converter(agent_cfg, installed_version)


def freeze_rsl_rl_models(runner: Any) -> tuple[Any, ...]:
    """Freeze the PPO models across legacy and current RSL-RL layouts."""
    algorithm = runner.alg
    modules = []
    seen_ids = set()
    for name in ("actor", "critic", "policy", "actor_critic"):
        module = getattr(algorithm, name, None)
        if module is None or not hasattr(module, "parameters") or id(module) in seen_ids:
            continue
        seen_ids.add(id(module))
        module.requires_grad_(False)
        module.eval()
        modules.append(module)
    if not modules:
        raise RuntimeError("unable to locate RSL-RL policy modules to freeze")
    if any(parameter.requires_grad for module in modules for parameter in module.parameters()):
        raise RuntimeError("base PPO policy must remain frozen")
    return tuple(modules)
