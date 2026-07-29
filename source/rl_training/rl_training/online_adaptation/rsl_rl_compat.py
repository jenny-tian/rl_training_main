from __future__ import annotations

import importlib.metadata as metadata
from collections.abc import Callable, Mapping
from typing import Any

import torch


def split_rsl_rl_endings(
    done: torch.Tensor,
    extras: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split RSL-RL done rows using the wrapper's required time_outs metadata."""
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
    bootstrap_mask = ~terminated
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
