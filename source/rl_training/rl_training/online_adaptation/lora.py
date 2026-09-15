"""Small, dependency-free LoRA modules used by online policy adaptation.

The module deliberately keeps the original :class:`torch.nn.Linear` as a
child module.  This makes loading an RSL-RL actor, calling its normal
``forward``/distribution methods, and exporting the merged policy
straightforward.  Only ``lora_A`` and ``lora_B`` require gradients; the
original weight and bias are immutable for the whole session.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low-rank update.

    ``B`` is initialized to zero and ``A`` to a small random matrix.  Thus a
    newly injected adapter is exactly equivalent to the source linear layer,
    including on non-zero observations.  The ``merge`` operation is useful for
    deployment and is numerically reversible up to floating point round-off.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        *,
        alpha: float | None = None,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be torch.nn.Linear")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not torch.isfinite(torch.tensor(float(init_scale))) or init_scale < 0:
            raise ValueError("init_scale must be finite and non-negative")
        self.base = base
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        if not torch.isfinite(torch.tensor(self.alpha)) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.scaling = self.alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        if init_scale:
            nn.init.normal_(self.lora_A, mean=0.0, std=float(init_scale))
        else:
            nn.init.zeros_(self.lora_A)
        # The source policy is immutable.  Do this after constructing the
        # adapter so ``parameters()`` exposes only the intended trainables.
        self.base.requires_grad_(False)
        self.register_buffer("_merged", torch.tensor(False, dtype=torch.bool), persistent=True)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, **kwargs: Any) -> "LoRALinear":
        """Construct an adapter while retaining the source linear parameters."""

        return cls(linear, rank, **kwargs)

    @property
    def merged(self) -> bool:
        return bool(self._merged.item())

    def delta_weight(self) -> Tensor:
        """Return the dense low-rank update in ``(out_features, in_features)`` order."""

        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, input: Tensor) -> Tensor:
        output = self.base(input)
        if not self.merged:
            # F.linear handles arbitrary leading dimensions and matches the
            # dtype/device behavior of the wrapped source layer.
            output = output + F.linear(F.linear(input, self.lora_A), self.lora_B) * self.scaling
        return output

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into the frozen base weight for deployment."""

        if not self.merged:
            self.base.weight.add_(self.delta_weight().to(self.base.weight))
            self._merged.fill_(True)

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo :meth:`merge` and restore the explicit adapter path."""

        if self.merged:
            self.base.weight.sub_(self.delta_weight().to(self.base.weight))
            self._merged.fill_(False)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield self.lora_A
        yield self.lora_B

    @property
    def parameter_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Accept both injected and original ``nn.Linear`` state-dict keys.

        RSL-RL checkpoints are commonly loaded after constructing the adapted
        actor.  Their keys end in ``weight``/``bias`` whereas this wrapper's
        native keys are ``base.weight``/``base.bias``.  Normalize the former
        in-place before delegating to PyTorch's strict loader; adapter matrices
        remain optional so an old checkpoint naturally gets the zero update.
        """

        for name in ("weight", "bias"):
            old_key = prefix + name
            new_key = prefix + "base." + name
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        # Adapter fields did not exist in the source PPO checkpoint.  Keep the
        # freshly initialized values (B=0) when they are absent so callers can
        # continue to use ``strict=True`` and still obtain exact base behavior.
        for name, value in (
            ("lora_A", self.lora_A),
            ("lora_B", self.lora_B),
            ("_merged", self._merged),
        ):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


def iter_lora_modules(module: nn.Module) -> Iterator[LoRALinear]:
    """Yield all injected adapters below ``module``."""

    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def _matches_target(name: str, child: nn.Module, target_modules: set[str] | Callable[[str, nn.Module], bool] | None) -> bool:
    if target_modules is None:
        return isinstance(child, nn.Linear)
    if callable(target_modules):
        return bool(target_modules(name, child))
    return name in target_modules or name.rsplit(".", 1)[-1] in target_modules


def inject_lora(
    module: nn.Module,
    rank: int = 2,
    *,
    alpha: float | None = None,
    init_scale: float = 1e-3,
    target_modules: set[str] | Callable[[str, nn.Module], bool] | None = None,
) -> tuple[LoRALinear, ...]:
    """Recursively replace linear children with :class:`LoRALinear`.

    The returned tuple provides an auditable list of changed layers.  By
    default every ``nn.Linear`` is adapted; ``target_modules`` can contain
    module names (or a predicate receiving the qualified name) for selective
    injection.
    """

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    injected: list[LoRALinear] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            if _matches_target(qualified, child, target_modules):
                adapter = LoRALinear.from_linear(child, rank, alpha=alpha, init_scale=init_scale)
                setattr(parent, child_name, adapter)
                injected.append(adapter)
            else:
                visit(child, qualified)

    visit(module)
    return tuple(injected)


# Descriptive alias used by callers that prefer the terminology from PEFT.
replace_linear_with_lora = inject_lora


def lora_parameters(module: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return only adapter parameters, in stable module traversal order."""

    parameters: list[nn.Parameter] = []
    for adapter in iter_lora_modules(module):
        parameters.extend((adapter.lora_A, adapter.lora_B))
    return tuple(parameters)


def mark_only_lora_trainable(module: nn.Module) -> tuple[nn.Parameter, ...]:
    """Freeze all parameters except injected adapter matrices."""

    adapters = set(id(parameter) for parameter in lora_parameters(module))
    for parameter in module.parameters():
        parameter.requires_grad_(id(parameter) in adapters)
    return tuple(parameter for parameter in module.parameters() if parameter.requires_grad)


@torch.no_grad()
def merge_lora(module: nn.Module) -> None:
    for adapter in iter_lora_modules(module):
        adapter.merge()


@torch.no_grad()
def unmerge_lora(module: nn.Module) -> None:
    for adapter in iter_lora_modules(module):
        adapter.unmerge()


def lora_parameter_count(module: nn.Module) -> int:
    return sum(adapter.parameter_count for adapter in iter_lora_modules(module))


__all__ = [
    "LoRALinear",
    "inject_lora",
    "replace_linear_with_lora",
    "iter_lora_modules",
    "lora_parameters",
    "mark_only_lora_trainable",
    "merge_lora",
    "unmerge_lora",
    "lora_parameter_count",
]
