"""Strict actor-only initialization for separately budgeted full-policy learning."""
from collections.abc import Mapping
import copy
import math

import torch


def warmstart_actor_only(runner, payload, *, initial_std=None):
    """Copy compatible actor tensors into a fresh learner, preserving critic/optimizers.

    This is initialization, never a training resume. Mean-network tensors are
    copied exactly; optional exploration reset changes only log standard
    deviation. The original checkpoint object is not mutated. The caller must
    separately verify observation meaning, order and scales; matching shapes
    alone do not establish semantic compatibility.
    """
    if not isinstance(payload, Mapping) or not isinstance(payload.get('actor_state_dict'), Mapping):
        raise ValueError('separate actor_state_dict checkpoint required')
    if getattr(runner, 'current_learning_iteration', 0) != 0:
        raise ValueError('actor warmstart requires a fresh runner')
    algorithm=runner.alg
    actor=getattr(algorithm,'actor',None)
    if not isinstance(actor,torch.nn.Module):
        raise ValueError('runner must expose an independent actor module')
    optimizers=[getattr(algorithm,k) for k in ('optimizer','actor_optimizer','critic_optimizer') if hasattr(algorithm,k)]
    if not optimizers or any(not isinstance(o,torch.optim.Optimizer) or o.state for o in optimizers):
        raise ValueError('fresh empty optimizer state required')
    current=actor.state_dict();source=payload['actor_state_dict']
    if current.keys()!=source.keys():raise ValueError('actor observation/action schema differs')
    for name,value in source.items():
        if (not isinstance(value,torch.Tensor) or value.shape!=current[name].shape
                or value.dtype!=current[name].dtype or not torch.isfinite(value).all()):
            raise ValueError(f'incompatible or nonfinite actor tensor: {name}')
    replacement={k:v.detach().clone() for k,v in source.items()}
    if initial_std is not None:
        if (isinstance(initial_std,bool) or not isinstance(initial_std,(int,float))
                or not math.isfinite(initial_std) or initial_std<=0):
            raise ValueError('initial_std must be finite and positive')
        key='distribution.log_std_param'
        if key not in replacement:raise ValueError('log-standard-deviation actor required for exploration reset')
        replacement[key].fill_(math.log(initial_std))
    key="distribution.log_std_param"
    if key in replacement:
        scale=replacement[key].exp()
        if not torch.isfinite(scale).all() or (scale<=0).any():
            raise ValueError("actor exploration scale is not representable")
    backup=copy.deepcopy(current)
    try:actor.load_state_dict(replacement,strict=True)
    except Exception:
        actor.load_state_dict(backup,strict=True)
        raise
    actor.requires_grad_(True)
    return {'actor_loaded':True,'critic_loaded':False,'optimizer_loaded':False,
            'initial_std_override':initial_std,'mean_state_preserved':True,
            'actor_tensor_shapes':{k:list(v.shape) for k,v in current.items()}}
