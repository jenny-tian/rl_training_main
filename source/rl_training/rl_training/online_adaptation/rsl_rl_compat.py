from __future__ import annotations

import importlib.metadata as metadata
from collections.abc import Callable
from typing import Any


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
