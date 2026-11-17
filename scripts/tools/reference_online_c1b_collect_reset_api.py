"""Preserve the locked collector while converting NumPy log scalars for JSON."""
import numpy as np

import reference_online_c1b_collect as original


def json_scalars(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key:json_scalars(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_scalars(item) for item in value]
    return value


def collect(args):
    native_save = original.save
    def save(path, value):
        # Keep the existing writer's strict allow_nan=False check. Do not
        # stringify arbitrary objects, alter values, or touch the learning loop.
        return native_save(path, json_scalars(value))
    original.save = save
    try:
        return original.collect(args)
    finally:
        original.save = native_save
