"""Match native delay storage dtype before indexed writes, without changing draws."""
import torch


def set_indexed_time_lag(buffer, lag, env_ids):
    if not isinstance(lag, torch.Tensor) or lag.dtype not in (torch.int32,torch.int64):
        raise TypeError('paired lag must be an integer tensor')
    if env_ids.ndim != 1 or env_ids.dtype != torch.long or lag.shape != env_ids.shape:
        raise ValueError('lag and int64 environment IDs must have equal rank-one shape')
    if len(env_ids) != len(env_ids.unique()) or (env_ids < 0).any() or (env_ids >= buffer.batch_size).any():
        raise ValueError('invalid or duplicate environment IDs')
    if (lag < 0).any() or (lag > buffer.history_length).any():
        raise ValueError('lag outside native history range')
    # Basic slice assignment can cast int64 into int32. Advanced indexed
    # assignment cannot. Preserve the buffer's declared dtype on both paths.
    value = lag.to(device=buffer.device, dtype=buffer.time_lags.dtype)
    buffer.set_time_lag(value, batch_ids=env_ids.to(device=buffer.device))
