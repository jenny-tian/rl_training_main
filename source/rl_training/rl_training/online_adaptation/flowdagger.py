"""Small conditional rectified flow and expert-labelled FlowDAgger adaptation.

Time runs from noise (0) to requested joint action (1). The decoder is an
explicit Euler map, so inversion solves each *discrete* step, in reverse order.
Clipping belongs to the environment interface and is never inverted here.
This Lite3 implementation uses single actions, not language/image action chunks.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class FrozenPPOExpert(nn.Module):
    """Canonical route48 PPO mean, copied without its critic or optimizer."""

    def __init__(self, state):
        super().__init__()
        self.mlp = _mlp((48, 512, 256, 128, 12))
        self.mlp.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith("mlp.")}, strict=True)
        if not all(torch.isfinite(v).all() for v in self.parameters()):
            raise ValueError("nonfinite expert")
        self.requires_grad_(False).eval()

    def forward(self, obs):
        return self.mlp(obs)


class ConditionalFlow(nn.Module):
    def __init__(self, observation_dim=48, action_dim=12, hidden=(512, 256, 128), steps=8):
        super().__init__()
        if min(observation_dim, action_dim) < 1 or type(steps) is not int or steps < 1:
            raise ValueError("positive dimensions and Euler step count required")
        self.config = dict(observation_dim=observation_dim, action_dim=action_dim, hidden=list(hidden), steps=steps)
        self.steps = steps
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.field = _mlp((observation_dim + action_dim + 1, *hidden, action_dim))

    def velocity(self, obs, state, time):
        if obs.shape[:-1] != state.shape[:-1] or obs.shape[-1] != self.observation_dim or state.shape[-1] != self.action_dim:
            raise ValueError("flow observation/state shapes differ")
        time = torch.as_tensor(time, device=state.device, dtype=state.dtype)
        time = torch.broadcast_to(time, (*state.shape[:-1], 1))
        return self.field(torch.cat((obs, state, time), -1))

    def forward(self, obs, noise):
        state = noise
        for index in range(self.steps):
            state = state + self.velocity(obs, state, index / self.steps) / self.steps
        return state

    def initialize_from_expert(self, expert):
        """Initialize the field to the mean; subsequent FM learns state/time dependence.

        Initially G(o,z)=z+mu(o), not the Gaussian PPO sampler. This is only
        initialization: the field is trained and frozen before DAgger begins.
        """
        if self.config != dict(observation_dim=48, action_dim=12, hidden=[512, 256, 128], steps=self.steps):
            raise ValueError("expert initialization requires canonical Lite3 layout")
        with torch.no_grad():
            for i in (0, 2, 4, 6):
                self.field[i].bias.copy_(expert.mlp[i].bias)
                if i == 0:
                    self.field[i].weight.zero_()
                    self.field[i].weight[:, :48].copy_(expert.mlp[i].weight)
                else:
                    self.field[i].weight.copy_(expert.mlp[i].weight)


def gaussian_distillation_loss(flow, observation, mean_action, std, noise, time, center_weight=5.):
    """Conditional flow matching with a shared-noise Gaussian transport coupling.

    Endpoint a1=mu(o)+std*z, a0=z. This deliberately distils a unimodal PPO
    prior; it does NOT create or claim a multimodal gait prior. A center-path
    loss retains the teacher's deterministic mode while fitting the flow.
    """
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("finite positive endpoint standard deviation required")
    endpoint = mean_action + std * noise
    state = (1 - time) * noise + time * endpoint
    fm = F.mse_loss(flow.velocity(observation, state, time), endpoint - noise)
    center = F.mse_loss(flow.velocity(observation, time * mean_action, time), mean_action)
    return fm + center_weight * center, fm.detach(), center.detach()


@dataclass(frozen=True)
class Inversion:
    noise: torch.Tensor
    reconstructed_action: torch.Tensor
    max_error: torch.Tensor
    accepted: torch.Tensor
    iterations: int


@torch.no_grad()
def invert_euler(flow, observation, requested_action, *, iterations=64, tolerance=1.e-4, noise_bound=6.):
    """Fixed-point inverse with mandatory forward reconstruction acceptance.

    Convergence is not guaranteed. Rejected rows remain in the audit but must
    not enter the training buffer. Never silently clamp an inverse label.
    """
    if type(iterations) is not int or iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid inverse iteration/tolerance")
    if not math.isfinite(noise_bound) or noise_bound <= 0:
        raise ValueError("invalid noise bound")
    if requested_action.ndim != 2 or not torch.isfinite(requested_action).all() or not torch.isfinite(observation).all():
        raise ValueError("finite batched observations and requested targets required")
    state = requested_action.clone()
    total_iterations = 0
    for index in reversed(range(flow.steps)):
        endpoint = state.clone()
        guess = endpoint.clone()
        for _ in range(iterations):
            previous = guess
            guess = endpoint - flow.velocity(observation, guess, index / flow.steps) / flow.steps
            total_iterations += 1
            delta = (guess - previous).abs().amax()
            if bool(torch.isfinite(delta)) and float(delta) <= tolerance * .05:
                break
        state = guess
    reconstructed = flow(observation, state)
    error = (reconstructed - requested_action).abs().amax(-1)
    accepted = torch.isfinite(state).all(-1) & torch.isfinite(error) & (error <= tolerance)
    accepted &= state.abs().amax(-1) <= noise_bound
    return Inversion(state, reconstructed, error, accepted, total_iterations)


class NoisePolicy(nn.Module):
    def __init__(self, observation_dim=48, action_dim=12, bound=6.):
        super().__init__()
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("positive finite noise bound required")
        self.bound = bound
        self.mlp = _mlp((observation_dim, 256, 128, action_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, obs):
        return self.bound * torch.tanh(self.mlp(obs) / self.bound)


class DualBuffer:
    """Aggregate successful-episode correction and autonomous-segment labels.

    The caller must finish the episode before admitting labels. Corrections
    come from the expert at visited states; autonomous labels are the actual
    noise emitted by the learner, never a recovered/guessed seed.
    """

    def __init__(self):
        self.data = {"correction": [], "autonomous": []}
        self.episode_ids = set()

    def add_episode(self, episode_id, *, successful, observations, noises, interventions, valid):
        if episode_id in self.episode_ids:
            raise ValueError("episode already aggregated")
        if observations.ndim != 2 or noises.ndim != 2 or len(observations) != len(noises):
            raise ValueError("invalid episode matrix")
        for flag in (interventions, valid):
            if flag.dtype != torch.bool or flag.shape != (len(observations),):
                raise ValueError("explicit per-step boolean flags required")
        if not torch.isfinite(observations).all() or not torch.isfinite(noises[valid]).all():
            raise ValueError("nonfinite eligible replay")
        self.episode_ids.add(episode_id)
        if not successful:
            return
        for name, mask in (("correction", interventions & valid), ("autonomous", ~interventions & valid)):
            if mask.any():
                self.data[name].append((observations[mask].detach().cpu().clone(), noises[mask].detach().cpu().clone()))

    def counts(self):
        return {name: sum(len(o) for o, _ in rows) for name, rows in self.data.items()}

    def tensors(self, device):
        if min(self.counts().values()) == 0:
            raise ValueError("50/50 sampling requires both correction and autonomous data")
        return {name: tuple(torch.cat([row[i] for row in rows]).to(device) for i in (0, 1)) for name, rows in self.data.items()}


def balanced_batch(data, batch_size, generator):
    if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
        raise ValueError("50/50 sampling requires a positive even batch size")
    observations, noises = [], []
    for name in ("correction", "autonomous"):
        obs, noise = data[name]
        index = torch.randint(len(obs), (batch_size // 2,), generator=generator, device=obs.device)
        observations.append(obs[index]); noises.append(noise[index])
    return torch.cat(observations), torch.cat(noises)
