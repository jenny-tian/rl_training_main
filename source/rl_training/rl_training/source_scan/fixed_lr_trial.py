"""Pinned low-learning-rate trial contract and observational PPO diagnostics."""
from __future__ import annotations

import torch


def fixed_low_lr_plan():
    return dict(profile='lite3_source_lowspeed_v1', num_envs=120, rollout_length=24,
        max_updates=2, reserved_transitions=5760, baseline_charged_transitions=265344,
        cumulative_ceiling=271104, learning_rate=1e-5, schedule='fixed',
        initialization='registered_actor_checkpoint_std_fresh_critic_optimizer')


class MinibatchMonitor:
    """Record the real minibatch distributions without sampling or changing PPO.

    Hooks observe the distribution already computed by PPO before optimizer.step,
    then use a deterministic forward to measure the same batch after that step.
    They do not implement KL early stopping or adapt the fixed learning rate.
    """
    def __init__(self, algorithm, sink, *, learning_rate=1e-5):
        self.algorithm = algorithm
        self.sink = sink
        self.learning_rate = learning_rate
        self.update = self.index = 0
        self.batch = None
        self.rows = []

    def __enter__(self):
        alg = self.algorithm
        if alg.actor.is_recurrent or alg.critic.is_recurrent or alg.rnd or alg.symmetry:
            raise ValueError('minibatch monitor requires feedforward PPO without extensions')
        if alg.schedule != 'fixed' or alg.learning_rate != self.learning_rate:
            raise ValueError('minibatch monitor requires the pinned fixed learning rate')
        self.original_generator = alg.storage.mini_batch_generator
        alg.storage.mini_batch_generator = self._batches
        self.handles = [alg.optimizer.register_step_pre_hook(self._before_step),
                        alg.optimizer.register_step_post_hook(self._after_step)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.algorithm.storage.mini_batch_generator = self.original_generator
        self.batch = None

    def begin_update(self, update):
        if update != self.update + 1 or self.batch is not None:
            raise ValueError('minibatch monitor update sequence differs from training')
        self.update = update
        self.index = 0

    def finish_update(self):
        expected = self.algorithm.num_learning_epochs * self.algorithm.num_mini_batches
        if self.index != expected or self.batch is not None:
            raise RuntimeError('missing minibatch optimizer observations')

    def _batches(self, *args, **kwargs):
        for batch in self.original_generator(*args, **kwargs):
            self.batch = batch
            self.pending = None
            self.observed_step = False
            try:
                yield batch
                if not self.observed_step:
                    raise RuntimeError('minibatch did not complete one observed optimizer step')
            finally:
                self.batch = None

    @torch.no_grad()
    def _before_step(self, optimizer, args, kwargs):
        if self.batch is None or self.pending is not None or self.observed_step:
            raise RuntimeError('optimizer step occurred outside its minibatch')
        rates = [float(group['lr']) for group in optimizer.param_groups]
        if any(rate != self.learning_rate for rate in rates):
            raise ValueError('optimizer learning rate changed')
        alg = self.algorithm
        params = tuple(value.detach() for value in alg.actor.output_distribution_params)
        kl = alg.actor.get_kl_divergence(self.batch.old_distribution_params, params)
        if not bool(torch.isfinite(kl).all()):
            raise FloatingPointError('nonfinite minibatch KL before optimizer step')
        self.pending = dict(update=self.update, minibatch=self.index+1,
            optimizer_step=alg.source_optimizer_steps_started, batch_size=len(kl), learning_rates=rates,
            rollout_kl_before_mean=float(kl.mean()), rollout_kl_before_max=float(kl.max()))

    @torch.no_grad()
    def _after_step(self, optimizer, args, kwargs):
        if self.pending is None:
            raise RuntimeError('optimizer post-hook lacks its pre-hook')
        actor = self.algorithm.actor
        # stochastic_output=False keeps the cached PPO distribution and RNG intact.
        mean = actor(self.batch.observations)
        std = actor.distribution.log_std_param.exp().expand_as(mean)
        kl = actor.get_kl_divergence(self.batch.old_distribution_params, (mean, std))
        if not bool(torch.isfinite(kl).all()):
            raise FloatingPointError('nonfinite minibatch KL after optimizer step')
        row = dict(self.pending, rollout_kl_after_mean=float(kl.mean()), rollout_kl_after_max=float(kl.max()))
        self.sink(row)
        self.rows.append(row)
        self.index += 1
        self.pending = None
        self.observed_step = True


class WalkingProbeMonitor:
    """Compare each checkpoint on the original actor's saved walking observations."""
    def __init__(self, actor, snapshots, device):
        self.probes = {step:{'policy':row['observation'].to(device)} for step,row in snapshots.items()}
        with torch.no_grad():
            self.reference = {step:actor(obs).detach().clone() for step,obs in self.probes.items()}
            self.std = actor.distribution.log_std_param.exp().detach().clone()

    @torch.no_grad()
    def measure(self, actor, update):
        rows = []
        std = actor.distribution.log_std_param.exp()
        for step,obs in sorted(self.probes.items()):
            mean = actor(obs)
            reference = self.reference[step]
            delta = mean-reference
            kl = actor.get_kl_divergence((reference,self.std.expand_as(mean)), (mean,std.expand_as(mean)))
            rows.append(dict(update=update, original_control_step=step, observations=len(mean),
                action_shift_rms=float(delta.square().mean().sqrt()), action_shift_max=float(delta.abs().max()),
                original_policy_kl_mean=float(kl.mean()), original_policy_kl_max=float(kl.max())))
        return rows
