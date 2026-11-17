"""Exercise installed Isaac's actual CPU buffers, without starting Isaac Sim."""
import importlib
from pathlib import Path
import sys
import types

import pytest
import torch

from rl_training.ea_lpl.c1b_delay import set_indexed_time_lag


@pytest.fixture
def buffer():
    root = Path('/home/tjx/IsaacLab/source/isaaclab/isaaclab/utils/buffers')
    package = types.ModuleType('_c1b_native_buffers_test')
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    native = importlib.import_module(package.__name__+'.delay_buffer')
    return native.DelayBuffer(history_length=1,batch_size=4,device='cpu')


def test_native_full_assignment_casts_but_indexed_assignment_reproduces_failure(buffer):
    long = torch.ones(4,dtype=torch.long)
    buffer.set_time_lag(long)
    assert buffer.time_lags.dtype == torch.int32
    with pytest.raises(RuntimeError,match='dtypes match'):
        buffer.set_time_lag(long[:2],batch_ids=torch.tensor([1,3]))


def test_fix_preserves_other_rows_and_native_delay_history(buffer):
    buffer.set_time_lag(1)
    buffer.compute(torch.arange(4.).reshape(4,1))
    buffer.compute(torch.arange(4.).reshape(4,1)+10)
    ids = torch.tensor([1,3])
    set_indexed_time_lag(buffer,torch.tensor([0,1]),ids)
    buffer.reset(ids)
    assert buffer.time_lags.tolist() == [1,0,1,1]
    output = buffer.compute(torch.arange(4.).reshape(4,1)+20)
    assert output[:,0].tolist() == [10.,21.,12.,23.]


def test_fix_uses_storage_dtype_instead_of_hard_coding_int32(buffer):
    buffer._time_lags = buffer._time_lags.to(torch.int64)
    set_indexed_time_lag(buffer,torch.tensor([1,0],dtype=torch.int32),torch.tensor([0,2]))
    assert buffer.time_lags.tolist() == [1,0,0,0]


@pytest.mark.parametrize('lag,ids,error',[
    (torch.tensor([.5]),torch.tensor([0]),TypeError),
    (torch.tensor([2]),torch.tensor([0]),ValueError),
    (torch.tensor([-1]),torch.tensor([0]),ValueError),
    (torch.tensor([0,1]),torch.tensor([0,0]),ValueError),
])
def test_invalid_lags_do_not_mutate_native_buffer(buffer,lag,ids,error):
    before = buffer.time_lags.clone()
    with pytest.raises(error): set_indexed_time_lag(buffer,lag,ids)
    assert torch.equal(before,buffer.time_lags)
