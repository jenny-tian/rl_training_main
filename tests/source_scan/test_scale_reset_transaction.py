"""Real shared-bank assignment adapter rollback on an injected CPU buffer write."""
from copy import deepcopy

import pytest
import torch

from test_parallel_terrain_allocation import _fake_parallel_env
from rl_training.source_scan.parallel_terrain_allocation import assign_source_terrain_for_parallel_reset


def test_partial_origin_write_rolls_back_all_buffers_without_consuming_tickets(monkeypatch):
    env=_fake_parallel_env(num_envs=120,num_rows=1)
    env.scene.terrain.cfg.shared_bank_rows=1
    env.scene.terrain.env_origins=env.scene.terrain.terrain_origins[0,0].repeat(120,1)
    # The physical USD graph has separate tests; inject only its audit here.
    monkeypatch.setattr('rl_training.source_scan.shared_terrain_bank.audit_shared_collision_groups',lambda scene:{'inverted':True})
    assign_source_terrain_for_parallel_reset(env,torch.arange(120),sampler_seed=131001)
    terrain=env.scene.terrain;allocator=env._source_parallel_terrain_allocator
    before={name:getattr(terrain,name).clone() for name in ('terrain_levels','terrain_types','env_origins')}
    expected=deepcopy(allocator).reset([1,17,119])
    class FailOnce(torch.Tensor):
        def __setitem__(self,index,value):
            super().__setitem__(index,value)
            if getattr(self,'fail_next',False):
                self.fail_next=False
                raise RuntimeError('injected after partial origin write')
    terrain.env_origins=terrain.env_origins.as_subclass(FailOnce);terrain.env_origins.fail_next=True
    with pytest.raises(RuntimeError,match='partial origin write'):
        assign_source_terrain_for_parallel_reset(env,[1,17,119],sampler_seed=131001)
    assert env._source_parallel_terrain_allocator is allocator
    for name,value in before.items():assert torch.equal(getattr(terrain,name),value),name
    actual=assign_source_terrain_for_parallel_reset(env,[1,17,119],sampler_seed=131001)
    assert actual==expected
    assert [ticket.assignment_index for ticket in actual]==[120,121,122]
    assert not torch.cuda.is_initialized()
