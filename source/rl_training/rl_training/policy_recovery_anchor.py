"""Training-only action preservation for ordinary walking and early recovery."""
import math
import torch

from rl_training.policy_flat_anchor import flat_anchor_cost


def recovery_anchor_mask(cases, episode_steps, early_steps):
    if type(early_steps) is not int or early_steps < 1:
        raise ValueError('positive integer early recovery window required')
    if cases.ndim != 1 or episode_steps.shape != cases.shape:
        raise ValueError('one case and age per environment required')
    if cases.dtype not in (torch.int32, torch.int64) or episode_steps.dtype not in (torch.int32, torch.int64):
        raise ValueError('integer cases and episode ages required')
    if bool((cases < -1).any()) or bool((episode_steps < 0).any()):
        raise ValueError('invalid case or negative age')
    return (cases < 0) | (episode_steps < early_steps)


def prepare_recovery_anchor(env, teacher, observation, early_steps):
    mask = recovery_anchor_mask(env._recovery_case, env.episode_length_buf, early_steps)
    target = observation.new_zeros((len(observation), 12))
    target[mask] = teacher(observation[mask])
    env._recovery_anchor_target = target
    env._recovery_anchor_mask = mask.clone()
    env._recovery_anchor_prepared_step = int(env.common_step_counter)
    return target, mask


def recovery_anchor_reward(env):
    if env._recovery_anchor_prepared_step != int(env.common_step_counter) - 1:
        raise RuntimeError('recovery anchor was not prepared immediately before this control step')
    cost = flat_anchor_cost(env.action_manager.action, env._recovery_anchor_target, env._recovery_anchor_mask)
    env._recovery_anchor_applied_cost = cost.detach().clone()
    return cost


def configure_recovery_anchor(cfg, options):
    from isaaclab.managers import RewardTermCfg
    if not math.isfinite(options['weight']) or options['weight'] <= 0:
        raise ValueError('positive finite recovery anchor weight required')
    early = options['early_steps']
    if type(early) is not int or early < 1:
        raise ValueError('positive integer early recovery window required')
    cfg.rewards.recovery_policy_anchor = RewardTermCfg(func=recovery_anchor_reward, weight=-options['weight'])
