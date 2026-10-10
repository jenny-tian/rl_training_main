# PPO update adapted from RSL-RL.
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""PPO with an explicit mean-action replay loss for a narrow, audited configuration."""
import math
import hashlib

import torch
from tensordict import TensorDict


class MeanActionReplay:
    """Sample TRAIN states; optional paired targets override selected reference means."""

    def __init__(self, teacher, observations, batch_groups, *, coefficient, seed, device, fixed_targets=None, group_weights=None):
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError('finite nonnegative mean anchor coefficient required')
        if set(observations) != set(batch_groups) or not observations:
            raise ValueError('replay groups differ from the registered batch')
        for group, values in observations.items():
            if values.ndim != 2 or values.shape[1] not in (48, 49) or len(values) == 0 or not torch.isfinite(values).all():
                raise ValueError('nonempty finite route48 TRAIN observations required')
            if type(batch_groups[group]) is not int or batch_groups[group] < 1:
                raise ValueError('positive group batch sizes required')
        self.teacher = teacher.eval().requires_grad_(False)
        self.observations = observations
        self.batch_groups = batch_groups
        self.coefficient = coefficient
        if group_weights is not None and set(group_weights) != set(observations):
            raise ValueError('group weights must name every replay group exactly')
        self.group_weights = {k: float(group_weights[k]) if group_weights is not None else 1. for k in observations}
        if not all(math.isfinite(w) and w >= 0 for w in self.group_weights.values()):
            raise ValueError('finite nonnegative replay group weights required')
        self.last_group_losses = None
        fixed_targets = fixed_targets or {}
        if not set(fixed_targets) <= set(observations):
            raise ValueError('fixed target group has no observation group')
        self.fixed_targets = {}
        for group, target in fixed_targets.items():
            if target.shape != (len(observations[group]), 12) or not torch.isfinite(target).all():
                raise ValueError('fixed targets must be finite paired12D actions')
            if target.device != observations[group].device or target.dtype != observations[group].dtype:
                raise ValueError('fixed target dtype/device differs from paired observations')
            self.fixed_targets[group] = target.detach().clone()
        # Isolated CPU draws can be replayed by the auditor without simulator/GPU RNG.
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.draw_digest = hashlib.sha256()
        self.calls = 0
        self.rows = 0
        self.fixed_target_digest = hashlib.sha256()
        self.fixed_target_rows = 0

    def loss(self, actor, *, record_groups=False):
        if actor.obs_normalization or actor.obs_groups != ['policy'] or actor.obs_dim not in (48, 49):
            raise ValueError('mean replay requires the unnormalized route48 actor')
        if any(v.shape[1] != actor.obs_dim for v in self.observations.values()):
            raise ValueError('replay width differs from actor observation width')
        pieces = []
        overrides = []
        offset = 0
        for group, count in self.batch_groups.items():
            states = self.observations[group]
            indices = torch.randint(len(states), (count,), generator=self.generator)
            self.draw_digest.update(indices.numpy().tobytes())
            device_indices = indices.to(states.device)
            pieces.append(states[device_indices])
            if group in self.fixed_targets:
                paired = self.fixed_targets[group][device_indices]
                overrides.append((offset, offset + count, paired))
                self.fixed_target_digest.update(paired.cpu().numpy().tobytes())
                self.fixed_target_rows += count
            offset += count
        states = torch.cat(pieces)
        with torch.no_grad():
            target = self.teacher(states)
            if overrides:
                target = target.clone()  # Reference output may be a view of the input.
            for start, end, paired in overrides:
                target[start:end] = paired
        predicted = actor(TensorDict({'policy': states}, batch_size=[len(states)]))
        self.calls += 1
        self.rows += len(states)
        errors = (predicted-target).square()
        self.last_group_losses = None
        if record_groups:
            self.last_group_losses = {}
            start = 0
            for group, count in self.batch_groups.items():
                self.last_group_losses[group] = self.group_weights[group]*errors[start:start+count].sum()/errors.numel()
                start += count
        if all(w == 1. for w in self.group_weights.values()):
            return errors.mean()  # Preserve the exact original reduction and optimizer path.
        weights = errors.new_tensor([self.group_weights[k] for k,n in self.batch_groups.items() for _ in range(n)])
        # Keep the original total-row/action denominator; never renormalize weights.
        return (errors*weights[:,None]).mean()


def mean_gradient_geometry(actor, ppo_loss, replay, anchor_loss):
    """Read mean-network gradients without changing .grad, RNG or the optimizer."""
    losses = dict(ppo=ppo_loss, anchor=replay.coefficient*anchor_loss)
    if 'correction' in replay.last_group_losses:
        losses['correction'] = replay.coefficient*replay.last_group_losses['correction']
        ordinary = [v for k,v in replay.last_group_losses.items() if k != 'correction']
        if ordinary:losses['ordinary'] = replay.coefficient*sum(ordinary)
    parameters = list(actor.mlp.parameters())
    gradients = {k: torch.cat([g.detach().flatten() for g in torch.autograd.grad(v, parameters, retain_graph=True)])
                 for k,v in losses.items()}
    norms = {k:float(v.double().norm()) for k,v in gradients.items()}
    cosine = {}
    for k in gradients:
        if k != 'ppo':
            denominator = norms['ppo']*norms[k]
            cosine[k+'_vs_ppo'] = float(torch.dot(gradients[k].double(),gradients['ppo'].double())/denominator) if denominator else None
    return dict(mean_parameter_count=sum(v.numel() for v in parameters), norms=norms, cosine=cosine,
                weighted_group_losses={k:float((replay.coefficient*v).detach()) for k,v in replay.last_group_losses.items()},
                scope='Actual first minibatch, actor mean MLP only, before clipping and Adam; cosine does not describe the preconditioned optimizer step.')


def update_with_mean_anchor(algorithm, replay, *, tail_replay=None, gradient_observer=None):
    """The feedforward, fixed-LR, single-GPU PPO update plus one explicit auxiliary loss.

    Unsupported algorithm options fail rather than silently dropping their behavior.
    The zero-coefficient path is regression-tested against the installed PPO.update.
    """
    a = algorithm
    if a.actor.is_recurrent or a.critic.is_recurrent or a.rnd or a.symmetry or a.is_multi_gpu or a.schedule != 'fixed':
        raise ValueError('mean anchor supports feedforward fixed-LR PPO without RND, symmetry or multi-GPU')
    means = dict(value=0., surrogate=0., entropy=0., mean_anchor=0.)
    if tail_replay is not None:
        means['tail_anchor'] = 0.
    updates = 0
    for batch in a.storage.mini_batch_generator(a.num_mini_batches, a.num_learning_epochs):
        if a.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                batch.advantages = (batch.advantages-batch.advantages.mean())/(batch.advantages.std()+1e-8)
        a.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
        log_prob = a.actor.get_output_log_prob(batch.actions)
        values = a.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
        entropy = a.actor.output_entropy[:batch.observations.batch_size[0]]
        ratio = torch.exp(log_prob-torch.squeeze(batch.old_actions_log_prob))
        surrogate = -torch.squeeze(batch.advantages)*ratio
        clipped = -torch.squeeze(batch.advantages)*torch.clamp(ratio, 1.-a.clip_param, 1.+a.clip_param)
        surrogate_loss = torch.max(surrogate, clipped).mean()
        if a.use_clipped_value_loss:
            value_clipped = batch.values+(values-batch.values).clamp(-a.clip_param, a.clip_param)
            value_loss = torch.max((values-batch.returns).pow(2), (value_clipped-batch.returns).pow(2)).mean()
        else:
            value_loss = (batch.returns-values).pow(2).mean()
        loss = surrogate_loss+a.value_loss_coef*value_loss-a.entropy_coef*entropy.mean()
        ppo_loss = loss
        record_gradients = gradient_observer is not None and updates == 0
        if record_gradients and (not replay.coefficient or tail_replay is not None):
            raise ValueError('gradient diagnostics require nonzero main replay and no tail replay')
        anchor_loss = replay.loss(a.actor, record_groups=record_gradients) if replay.coefficient else loss.new_zeros(())
        loss = loss+replay.coefficient*anchor_loss
        if tail_replay is not None:
            tail_loss = tail_replay.loss(a.actor) if tail_replay.coefficient else loss.new_zeros(())
            loss = loss+tail_replay.coefficient*tail_loss
        if not torch.isfinite(loss):
            raise ValueError('nonfinite PPO/mean anchor objective')
        if record_gradients:
            gradient_observer(mean_gradient_geometry(a.actor, ppo_loss, replay, anchor_loss))
        replay.last_group_losses = None
        a.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a.actor.parameters(), a.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(a.critic.parameters(), a.max_grad_norm)
        a.optimizer.step()
        for key, value in (('value', value_loss), ('surrogate', surrogate_loss), ('entropy', entropy.mean()), ('mean_anchor', anchor_loss)):
            means[key] += float(value.detach())
        if tail_replay is not None:
            means['tail_anchor'] += float(tail_loss.detach())
        updates += 1
    if updates != a.num_learning_epochs*a.num_mini_batches:
        raise RuntimeError('unexpected PPO mini-batch count')
    a.storage.clear()
    return {key: value/updates for key, value in means.items()}
