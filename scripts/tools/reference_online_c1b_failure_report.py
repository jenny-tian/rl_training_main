"""Seal the initial-reset failure and CPU-only native-buffer repair evidence."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch

from reference_online_c1b import OUTPUT, DOCS, registration
from reference_online_targets import HASHES, MODEL_PATHS, read
from reference_stairs_eval import ROOT, save, sha


def main():
    registration()
    directory = OUTPUT / 'smoke_1'
    execution, capture = read(directory/'execution.json'),read(directory/'failure_capture.json')
    assert execution['status']=='failed' and execution['exit_code']==1
    assert capture['recorded_control_steps']==0 and capture['partial_records']==[] and capture['scalars']['trace'] is None
    with np.load(directory/'failure_trace.npz') as trace:
        assert all(np.count_nonzero(trace[k])==0 for k in trace.files)
    try: os.kill(execution['pid'],0)
    except ProcessLookupError: pass
    else: raise RuntimeError('simulation worker still running')
    assert sha(MODEL_PATHS['B']) == HASHES['B']
    sys.path.insert(0,str(ROOT/'source/rl_training'))
    from rl_training.ea_lpl.c1b_delay import set_indexed_time_lag
    folder = Path('/home/tjx/IsaacLab/source/isaaclab/isaaclab/utils/buffers')
    package = types.ModuleType('_c1b_failure_native_buffers'); package.__path__=[str(folder)]
    sys.modules[package.__name__]=package
    cls = importlib.import_module(package.__name__+'.delay_buffer').DelayBuffer
    buffer = cls(1,4,'cpu')
    buffer.set_time_lag(torch.ones(4,dtype=torch.long))
    ids,lag = torch.tensor([1,3]),torch.tensor([0,1])
    try: buffer.set_time_lag(lag,batch_ids=ids)
    except RuntimeError as error: reproduction_error=str(error)
    else: raise AssertionError('native indexed dtype failure not reproduced')
    set_indexed_time_lag(buffer,lag,ids)
    assert buffer.time_lags.tolist()==[1,0,1,1]
    sources = [ROOT/'source/rl_training/rl_training/ea_lpl/c1b_delay.py',
        ROOT/'scripts/tools/reference_online_c1b_env_repaired.py',ROOT/'scripts/tools/reference_online_c1b_continue.py',
        ROOT/'tests/ea_lpl/test_c1b_delay.py',Path(__file__)]
    result = dict(status='stopped_on_initial_reset_dtype',runtime=execution,
        error=read(directory/'error.json'),failure_capture=capture,
        actual_preparation_control_steps=0,actual_policy_control_steps=0,actor_forward_calls=0,
        base_optimizer_updates=0,adapter_updates=0,formal_train_or_evaluation_started=False,
        smoke_attempts_used=1,smoke_attempts_remaining=1,reserved_smoke_interactions_consumed=9600,
        original_total_budget=944000,failed_slot_not_restored=True,checkpoint_sha256=HASHES['B'],
        repair_simulation_verified=False,simulations_after_failure=0,
        reproduction=dict(kind='CPU installed native buffers; synthetic values, not original lag tensor',
            basic_slice_accepts_int64=True,indexed_error=reproduction_error,
            source_dtype=str(lag.dtype),destination_dtype=str(buffer.time_lags.dtype),
            repaired_lags=buffer.time_lags.tolist(),untouched_rows_preserved=True),
        repair_sources=[dict(path=str(p),sha256=sha(p)) for p in sources],
        native_buffer_sources=[dict(path=str(folder/name),sha256=sha(folder/name)) for name in ('delay_buffer.py','circular_buffer.py')],
        raw_sources=[dict(path=str(p),sha256=sha(p)) for p in sorted(directory.iterdir()) if p.is_file()],
        attempts=read(OUTPUT/'attempts.json'),lock_sha256=sha(OUTPUT/'lock.json'))
    save(DOCS/'EVIDENCE.json',result)
    print(result['status'],result['reproduction'])


if __name__ == '__main__': main()
