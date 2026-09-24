"""Offline teacher residual fitting for latent steering initialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


TEACHER_DECODER_SCHEMA = "rgls_teacher_decoder_v1"


@dataclass(frozen=True)
class TeacherDecoderFit:
    action_limits: Tensor
    decoder_matrix: Tensor
    code_targets: Tensor
    reconstruction: Tensor
    normalized_mse: float
    explained_variance: float

    def artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "schema": TEACHER_DECODER_SCHEMA,
            "decoder_matrix": self.decoder_matrix.detach().cpu(),
            "action_limits": self.action_limits.detach().cpu(),
            "code_targets": self.code_targets.detach().cpu(),
            "metrics": {
                "normalized_mse": self.normalized_mse,
                "explained_variance": self.explained_variance,
            },
        }


def supervise_latent_actor(
    actor: Any,
    history: Tensor,
    current_frame: Tensor,
    base_action: Tensor,
    code_targets: Tensor,
    *,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
) -> dict[str, float | int]:
    """Train a latent actor's observation path to predict offline teacher codes.

    The decoder remains frozen.  This is deliberately a small, deterministic
    supervised stage that can run before SAC and does not consume sealed data.
    ``actor`` is duck-typed to keep this utility independent of the Isaac
    runtime; it must expose the latent actor's ``_code_distribution`` method.
    """
    if not callable(getattr(actor, "_code_distribution", None)):
        raise TypeError("actor must expose _code_distribution")
    if (
        not isinstance(history, Tensor)
        or history.ndim != 3
        or not history.is_floating_point()
        or not isinstance(current_frame, Tensor)
        or current_frame.ndim != 2
        or not current_frame.is_floating_point()
        or not isinstance(base_action, Tensor)
        or base_action.ndim != 2
        or not base_action.is_floating_point()
        or not isinstance(code_targets, Tensor)
        or code_targets.ndim != 2
        or not code_targets.is_floating_point()
    ):
        raise ValueError("supervised tensors must be finite floating-point tensors with batch dimensions")
    count = history.shape[0]
    if (
        count < 1
        or current_frame.shape[0] != count
        or base_action.shape[0] != count
        or code_targets.shape[0] != count
        or history.device != current_frame.device
        or history.device != base_action.device
        or history.device != code_targets.device
        or history.dtype != current_frame.dtype
        or history.dtype != base_action.dtype
        or history.dtype != code_targets.dtype
        or not bool(torch.isfinite(history).all().item())
        or not bool(torch.isfinite(current_frame).all().item())
        or not bool(torch.isfinite(base_action).all().item())
        or not bool(torch.isfinite(code_targets).all().item())
        or bool((code_targets.abs() > 1.0 + 1.0e-5).any().item())
    ):
        raise ValueError("supervised tensors must share a finite batch and bounded code targets")
    actor_code_dim = getattr(actor, "code_dim", None)
    if actor_code_dim is not None and code_targets.shape[1] != int(actor_code_dim):
        raise ValueError("code_targets width must match the actor code dimension")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for name, value in (("learning_rate", learning_rate), ("weight_decay", weight_decay)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
            raise ValueError(f"{name} must be finite")
        invalid = value <= 0.0 if name == "learning_rate" else value < 0.0
        if invalid:
            raise ValueError(f"{name} must be {'positive' if name == 'learning_rate' else 'nonnegative'}")

    trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("actor has no trainable parameters")
    decoder = getattr(actor, "decoder_matrix", None)
    if isinstance(decoder, Tensor) and decoder.requires_grad:
        raise ValueError("latent decoder must remain frozen during supervised fitting")
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=float(weight_decay))
    was_training = bool(actor.training)
    actor.train()
    final_loss = float("nan")
    for _ in range(epochs):
        permutation = torch.arange(count, device=history.device)
        for start in range(0, count, batch_size):
            indices = permutation[start : start + batch_size]
            mean, _ = actor._code_distribution(
                history.index_select(0, indices),
                current_frame.index_select(0, indices),
                base_action.index_select(0, indices),
            )
            prediction = torch.tanh(mean)
            loss = (prediction - code_targets.index_select(0, indices)).square().mean()
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("latent actor supervision produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FloatingPointError("latent actor supervision produced non-finite gradients")
            optimizer.step()
            final_loss = float(loss.detach().item())
    actor.train(was_training)
    return {"epochs": epochs, "samples": count, "final_code_mse": final_loss}


def _validate_inputs(
    teacher_residual: Tensor,
    action_limits: Tensor,
    code_dim: int,
) -> None:
    if (
        not isinstance(teacher_residual, Tensor)
        or teacher_residual.ndim != 2
        or teacher_residual.shape[1] < 1
        or not teacher_residual.is_floating_point()
        or not bool(torch.isfinite(teacher_residual).all().item())
    ):
        raise ValueError("teacher_residual must be a finite floating-point [N, action_dim] tensor")
    if (
        not isinstance(action_limits, Tensor)
        or action_limits.ndim != 1
        or action_limits.shape[0] != teacher_residual.shape[1]
        or not action_limits.is_floating_point()
        or action_limits.device != teacher_residual.device
        or action_limits.dtype != teacher_residual.dtype
        or not bool(torch.isfinite(action_limits).all().item())
        or bool((action_limits <= 0).any().item())
    ):
        raise ValueError("action_limits must be finite, positive, and match teacher_residual")
    if type(code_dim) is not int or not 1 <= code_dim <= min(teacher_residual.shape):
        raise ValueError("code_dim must be an integer no larger than the data rank")
    normalized = teacher_residual / action_limits
    if bool((normalized.abs() > 1.0 + 1.0e-5).any().item()):
        raise ValueError("teacher residual exceeds the configured action limits")


def fit_low_rank_teacher_decoder(
    teacher_residual: Tensor,
    action_limits: Tensor,
    code_dim: int,
) -> TeacherDecoderFit:
    """Fit a deterministic bounded decoder and bounded teacher code targets.

    The fit uses a truncated SVD in normalized action space. The decoder is
    fixed after fitting; only the history/scan encoder or latent SAC actor is
    trained online. The returned matrix is compatible with
    :class:`LatentResidualActor`'s ``tanh(code @ decoder_matrix.T)`` path.
    """
    _validate_inputs(teacher_residual, action_limits, code_dim)
    normalized = teacher_residual / action_limits
    _, singular_values, right_basis = torch.linalg.svd(normalized, full_matrices=False)
    basis = right_basis[:code_dim].T.contiguous()
    scores = normalized @ basis
    scale = scores.abs().amax().clamp_min(torch.tensor(1.0e-6, device=scores.device, dtype=scores.dtype))
    decoder = basis * scale
    codes = (scores / scale).clamp(-1.0, 1.0)
    reconstruction = torch.tanh(codes @ decoder.T) * action_limits
    residual_error = (reconstruction - teacher_residual) / action_limits
    normalized_mse = float(residual_error.square().mean().item())
    total = float(normalized.square().sum().item())
    explained = 1.0 if total == 0.0 else max(0.0, 1.0 - normalized_mse * normalized.numel() / total)
    del singular_values
    return TeacherDecoderFit(
        action_limits=action_limits.detach().clone(),
        decoder_matrix=decoder.detach().clone(),
        code_targets=codes.detach().clone(),
        reconstruction=reconstruction.detach().clone(),
        normalized_mse=normalized_mse,
        explained_variance=explained,
    )


def save_teacher_decoder(
    path: str | Path,
    fit: TeacherDecoderFit,
    action_limits: Tensor,
    *,
    source_metadata: dict[str, Any] | None = None,
) -> None:
    _validate_inputs(fit.reconstruction, action_limits, fit.decoder_matrix.shape[1])
    if not torch.equal(fit.action_limits, action_limits):
        raise ValueError("action limits differ from the limits used to fit the teacher decoder")
    target = Path(path)
    if target.exists():
        raise FileExistsError("teacher decoder artifacts are immutable; choose a new path")
    target.parent.mkdir(parents=True, exist_ok=True)
    artifact = fit.artifact()
    if source_metadata is not None:
        if not isinstance(source_metadata, dict):
            raise TypeError("source_metadata must be a dictionary")
        artifact["source_metadata"] = dict(source_metadata)
    torch.save(artifact, target)


def load_teacher_decoder(
    path: str | Path,
    *,
    action_limits: Tensor,
    code_dim: int,
) -> Tensor:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 1
        or artifact.get("schema") != TEACHER_DECODER_SCHEMA
    ):
        raise ValueError("teacher decoder artifact schema is incompatible")
    decoder = artifact.get("decoder_matrix")
    saved_limits = artifact.get("action_limits")
    code_targets = artifact.get("code_targets")
    source_metadata = artifact.get("source_metadata")
    if (
        not isinstance(action_limits, Tensor)
        or action_limits.ndim != 1
        or not action_limits.is_floating_point()
        or not bool(torch.isfinite(action_limits).all().item())
        or bool((action_limits <= 0).any().item())
    ):
        raise ValueError("action limits must be finite and positive")
    if (
        not isinstance(decoder, Tensor)
        or decoder.shape != (action_limits.shape[0], code_dim)
        or not decoder.is_floating_point()
        or not bool(torch.isfinite(decoder).all().item())
        or not isinstance(saved_limits, Tensor)
        or saved_limits.shape != action_limits.shape
        or not torch.equal(saved_limits.to(dtype=action_limits.dtype), action_limits.detach().cpu())
        or not isinstance(code_targets, Tensor)
        or code_targets.ndim != 2
        or code_targets.shape[1] != code_dim
        or not code_targets.is_floating_point()
        or not bool(torch.isfinite(code_targets).all().item())
        or bool((code_targets.abs() > 1.0 + 1.0e-5).any().item())
        or (source_metadata is not None and not isinstance(source_metadata, dict))
    ):
        raise ValueError("teacher decoder artifact dimensions, code targets, or action limits do not match")
    return decoder.to(device=action_limits.device, dtype=action_limits.dtype).detach().clone()
