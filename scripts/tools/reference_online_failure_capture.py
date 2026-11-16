"""Persist existing exception-frame evidence without rerunning any simulation."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from reference_stairs_eval import save, sha


def capture_failure(error, output):
    output = Path(output)
    frame = None
    tb = error.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_name == 'collect':
            frame = tb.tb_frame
        tb = tb.tb_next
    if frame is None:
        save(output / 'failure_capture.json', dict(status='no_collection_frame', error=str(error)))
        return
    values = frame.f_locals
    arrays, scalar = {}, {}

    def capture(name, value):
        if hasattr(value, 'detach') and hasattr(value, 'cpu'):
            arrays[name] = value.detach().cpu().numpy().copy()
        elif isinstance(value, np.ndarray):
            arrays[name] = value.copy()
        elif is_dataclass(value):
            for field in fields(value): capture(f'{name}.{field.name}', getattr(value, field.name))
        elif isinstance(value, dict):
            for key, item in value.items(): capture(f'{name}.{key}', item)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            scalar[name] = value
        elif isinstance(value, (list, tuple)) and value and all(isinstance(x, np.ndarray) for x in value):
            arrays[name] = np.asarray(value)

    for key in ('tick', 'steps', 'n', 'trace', 'previous', 'active', 'counts', 'buffers', 'base_projection'):
        if key in values: capture(key, values[key])
    trace = values.get('trace')
    if trace is not None:
        # SafetyState can contain live simulator views. The failing decision
        # snapshot is captured immediately while the process is still alive.
        capture('failure_decision', trace)
    np.savez_compressed(output / 'failure_trace.npz', **arrays)
    save(output / 'failure_capture.json', dict(status='captured_existing_state', error=str(error),
        scalars=scalar, array_shapes={k: list(v.shape) for k, v in arrays.items()},
        recorded_control_steps=len(values.get('buffers', {}).get('root', [])),
        preparation_completed=(output / 'initial.npz').exists(),
        partial_records=values.get('records'),
        trace_sha256=sha(output / 'failure_trace.npz'), simulation_rerun=False))
