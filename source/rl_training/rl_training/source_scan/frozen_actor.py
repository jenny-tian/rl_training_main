"""Load the registered deterministic Lite3 actor without constructing PPO."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re

import torch
from torch import nn


CHECKPOINT_SHA256 = "87ec1cd3ab44079045d8da627c70b61e4918c22a715a21d5f7204acb73fbbc3c"
LAYER_DIMS = (45, 512, 256, 128, 12)


class FrozenSourceActor(nn.Module):
    """The pinned checkpoint's ELU MLP; no critic, optimizer or normalizer."""

    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        expected = {"distribution.log_std_param": (12,)}
        for index, (n_in, n_out) in enumerate(zip(LAYER_DIMS, LAYER_DIMS[1:])):
            expected[f"mlp.{2 * index}.weight"] = (n_out, n_in)
            expected[f"mlp.{2 * index}.bias"] = (n_out,)
        if set(state) != set(expected):
            raise ValueError("actor schema differs from the registered unnormalized ELU MLP")
        for name, shape in expected.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32:
                raise ValueError(f"invalid registered actor tensor: {name}")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"non-finite actor tensor: {name}")
        # Do not perturb the environment's seed stream during module construction.
        with torch.random.fork_rng(devices=[]):
            layers = []
            for index, (n_in, n_out) in enumerate(zip(LAYER_DIMS, LAYER_DIMS[1:])):
                layers.append(nn.Linear(n_in, n_out))
                if index < len(LAYER_DIMS) - 2:
                    layers.append(nn.ELU())
            self.mlp = nn.Sequential(*layers)
        self.mlp.load_state_dict({key[4:]: value for key, value in state.items() if key.startswith("mlp.")})
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, observation):
        if not isinstance(observation, torch.Tensor):
            observation = observation["policy"]
        if observation.ndim != 2 or observation.shape[1] != 45:
            raise ValueError("frozen actor requires [batch, 45] policy observations")
        if observation.dtype != torch.float32 or not bool(torch.isfinite(observation).all()):
            raise FloatingPointError("frozen actor observations must be finite float32")
        return self.mlp(observation)


def load_frozen_source_actor(checkpoint: Path, *, device="cpu", expected_sha256=CHECKPOINT_SHA256) -> FrozenSourceActor:
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected_sha256 must be 64 lowercase hex characters")
    payload_bytes = Path(checkpoint).read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != expected_sha256:
        if expected_sha256 == CHECKPOINT_SHA256:
            raise ValueError("checkpoint hash differs from the registered model_1998.pt")
        raise ValueError("checkpoint hash differs from the expected SHA256")
    payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
    # Only actor tensors enter modules; optimizer/critic payloads are never restored.
    return FrozenSourceActor(payload["actor_state_dict"]).to(device)
