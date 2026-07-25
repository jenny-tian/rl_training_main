from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class HistoryEncoder(nn.Module):
    def __init__(self, frame_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(frame_dim)
        self.gru = nn.GRU(frame_dim, hidden_dim, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(self.norm(history))
        return hidden[-1]


def _elu_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    def __init__(
        self,
        frame_dim: int,
        action_dim: int,
        history_dim: int = 64,
        action_limits: torch.Tensor | None = None,
    ):
        super().__init__()
        self.encoder = HistoryEncoder(frame_dim, history_dim)
        input_dim = history_dim + frame_dim + action_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
        )
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)
        nn.init.uniform_(self.mean_layer.weight, -1e-4, 1e-4)
        nn.init.uniform_(self.mean_layer.bias, -1e-4, 1e-4)
        if action_limits is None:
            action_limits = torch.ones(action_dim)
        if action_limits.shape != (action_dim,):
            raise ValueError(f"expected action_limits shape {(action_dim,)}, got {tuple(action_limits.shape)}")
        self.register_buffer("action_limits", action_limits.detach().clone())

    def _distribution_parameters(
        self, history: torch.Tensor, current_frame: torch.Tensor, base_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(history)
        features = self.backbone(torch.cat((latent, current_frame, base_action), dim=-1))
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features).clamp(-5.0, 1.0)
        return mean, log_std

    def sample(
        self,
        history: torch.Tensor,
        current_frame: torch.Tensor,
        base_action: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self._distribution_parameters(history, current_frame, base_action)
        distribution = Normal(mean, log_std.exp())
        raw_action = mean if deterministic else distribution.rsample()
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_limits
        if deterministic:
            log_prob = torch.zeros(mean.shape[0], 1, dtype=mean.dtype, device=mean.device)
        else:
            log_prob = distribution.log_prob(raw_action)
            log_prob -= torch.log(1.0 - squashed_action.square() + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        squashed_mean = torch.tanh(mean) * self.action_limits
        return action, log_prob, squashed_mean


class TwinCritic(nn.Module):
    def __init__(self, frame_dim: int, action_dim: int, history_dim: int = 64):
        super().__init__()
        self.encoder = HistoryEncoder(frame_dim, history_dim)
        input_dim = history_dim + frame_dim + 2 * action_dim
        self.q1 = _elu_mlp(input_dim, (128, 128), 1)
        self.q2 = _elu_mlp(input_dim, (128, 128), 1)

    def forward(
        self,
        history: torch.Tensor,
        current_frame: torch.Tensor,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(history)
        features = torch.cat((latent, current_frame, base_action, residual_action), dim=-1)
        return self.q1(features), self.q2(features)
