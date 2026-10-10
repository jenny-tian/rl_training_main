"""One causal route-difference feature, shared by training and frozen inference."""
import math

import torch

HISTORY_INDEX = 12


def validate_history_options(options):
    if set(options) != {'enabled', 'dt', 'lateral_scale_m', 'speed_scale_m_s', 'clip_m_s'}:
        raise ValueError('incomplete route history options')
    if type(options['enabled']) is not bool:
        raise ValueError('history enabled must be boolean')
    if not all(math.isfinite(options[k]) and options[k] > 0 for k in options if k != 'enabled'):
        raise ValueError('positive finite route history scales required')


def strip_route_history(observation):
    if observation.shape[-1] == 48:
        return observation
    if observation.shape[-1] != 49:
        raise ValueError('expected route48 or route-history49 observation')
    return torch.cat((observation[..., :12], observation[..., 13:]), dim=-1)


def insert_route_history(observation, feature):
    if observation.shape[-1] != 48 or feature.shape != observation.shape[:-1]:
        raise ValueError('route48 and one scalar history per observation required')
    return torch.cat((observation[..., :12], feature[..., None], observation[..., 12:]), dim=-1)


def expand_actor_history_inputs(payload):
    state = payload['actor_state_dict']
    weight = state['mlp.0.weight']
    if weight.ndim != 2 or weight.shape[1] != 48 or not torch.isfinite(weight).all():
        raise ValueError('finite route48 actor required for history expansion')
    expanded = torch.cat((weight[:, :12], weight.new_zeros(weight.shape[0], 1), weight[:, 12:]), dim=1)
    return dict(payload, actor_state_dict=dict(state, **{'mlp.0.weight': expanded}))


class RouteHistoryTracker:
    def __init__(self, options):
        validate_history_options(options)
        self.options = dict(options)
        self.current_y = None
        self.snapshot = None

    def advance(self, observation, reset):
        if observation.ndim != 2 or observation.shape[-1] != 48:
            raise ValueError('vector route48 observations required')
        if reset.shape != observation.shape[:-1] or reset.dtype != torch.bool:
            raise ValueError('one boolean reset flag per environment required')
        y = observation[:, 9].detach().clone()
        if self.current_y is None:
            reset = torch.ones_like(reset)
            previous = y.clone()
        else:
            previous = torch.where(reset, y, self.current_y)
        o = self.options
        velocity = ((y-previous)*o['lateral_scale_m']/o['dt']).clamp(-o['clip_m_s'], o['clip_m_s'])
        feature = velocity/o['speed_scale_m_s'] if o['enabled'] else torch.zeros_like(velocity)
        self.current_y = y
        self.snapshot = dict(history_previous_y=previous.clone(), history_current_y=y.clone(),
            history_velocity=velocity.clone(), history_input=feature.clone(), history_reset=reset.clone())
        return insert_route_history(observation, feature)


class RouteHistoryVecEnv:
    """Wrap RSL observations; repeated reads never advance the history clock."""
    def __init__(self, env, options):
        self.env = env
        self.tracker = RouteHistoryTracker(options)
        self._observations = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _convert(self, observations, reset):
        converted = observations.clone(recurse=False)
        converted['policy'] = self.tracker.advance(observations['policy'], reset)
        self._observations = converted
        return converted

    def get_observations(self):
        if self._observations is None:
            observations = self.env.get_observations()
            reset = torch.ones(self.num_envs, dtype=torch.bool, device=observations['policy'].device)
            self._convert(observations, reset)
        return self._observations

    def step(self, actions):
        if self._observations is None:
            raise RuntimeError('read initial observations before advancing route history')
        observations, reward, done, extras = self.env.step(actions)
        return self._convert(observations, done.bool()), reward, done, extras

    def history_snapshot(self):
        return {k:v.clone() for k,v in self.tracker.snapshot.items()}
