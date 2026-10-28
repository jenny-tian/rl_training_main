"""Offline reproduction using the actual transactional reset adapter; no simulator."""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'source/rl_training'))
spec = importlib.util.spec_from_file_location('allocation_test_helpers', root / 'tests/source_scan/test_parallel_terrain_allocation.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
from rl_training.source_scan.parallel_terrain_allocation import (
    ReservedRowTerrainAllocator, SOURCE_RESET_BLOCK_COUNTS, assign_source_terrain_for_parallel_reset,
)
env = helpers._fake_parallel_env(num_envs=60)
assign_source_terrain_for_parallel_reset(env, list(range(60)), 130001)
captured = env._source_parallel_terrain_allocator
oracle = ReservedRowTerrainAllocator(60, SOURCE_RESET_BLOCK_COUNTS, 130001)
assert captured.assignments == oracle.reset(list(range(60)))
assign_source_terrain_for_parallel_reset(env, [0], 130001)
expected = oracle.reset([0])[0]
current = env._source_parallel_terrain_allocator
assert captured is not current
assert captured.assignments[0] != expected
assert current.assignments[0] == expected
assert captured.assignments[1:] == current.assignments[1:]
print(json.dumps({
    'status': 'stale_allocator_reference_reproduced_offline',
    'captured_allocator_replaced': captured is not current,
    'stale_assignment': asdict(captured.assignments[0]),
    'current_assignment': asdict(current.assignments[0]),
    'expected_assignment': asdict(expected),
    'current_assignment_matches_expected': current.assignments[0] == expected,
    'other_assignments_unchanged': captured.assignments[1:] == current.assignments[1:],
    'simulator_started': False,
    'runtime_pass_claim': False,
}, indent=2))
