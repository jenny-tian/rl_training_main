"""Seal second initial-reset failure; no AppLauncher or simulation is imported."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from reference_online_c1b_continue import OUTPUT, DOCS, registration
from reference_online_targets import HASHES, MODEL_PATHS, read
from reference_stairs_eval import ROOT, save, sha


def main():
    registration()
    directory = OUTPUT/'smoke_2'
    execution,capture = read(directory/'execution.json'),read(directory/'failure_capture.json')
    assert execution['status']=='failed' and execution['exit_code']==1
    assert "has no attribute 'reset_buf'" in capture['error']
    assert capture['recorded_control_steps']==0 and not capture['partial_records']
    assert capture['scalars']['trace'] is None
    with np.load(directory/'failure_trace.npz') as trace:
        assert all(np.count_nonzero(trace[k])==0 for k in trace.files)
    try: os.kill(execution['pid'],0)
    except ProcessLookupError: pass
    else: raise RuntimeError('simulation worker still running')
    assert sha(MODEL_PATHS['B']) == HASHES['B']
    attempts = read(OUTPUT/'attempts.json')
    assert [a['name'] for a in attempts] == ['smoke_1','smoke_2']
    assert sum(a['reserved_steps'] for a in attempts)==19200
    native = Path('/home/tjx/IsaacLab/source/isaaclab/isaaclab')
    locations = []
    for filename,owner in [('envs/manager_based_rl_env.py','ManagerBasedRLEnv'),('envs/manager_based_env.py','ManagerBasedEnv')]:
        tree = ast.parse((native/filename).read_text())
        cls = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==owner)
        for method in (node for node in cls.body if isinstance(node,ast.FunctionDef)):
            for node in ast.walk(method):
                if isinstance(node,ast.Attribute) and isinstance(node.ctx,ast.Store) and node.attr in ('reset_buf','reset_terminated','reset_time_outs'):
                    locations.append(dict(file=filename,method=method.name,attribute=node.attr,line=node.lineno))
    assert len(locations)==3 and all(item['method']=='step' for item in locations)
    sources = [ROOT/'scripts/tools'/name for name in ('reference_online_c1b_env_reset_api.py',
        'reference_online_c1b_collect_reset_api.py','reference_online_c1b_reset_api.py',Path(__file__).name)]
    sources.append(ROOT/'tests/source_curriculum/test_c1b_native_reset_contract.py')
    native_files = ['envs/manager_based_env.py','envs/manager_based_rl_env.py','managers/action_manager.py',
        'managers/termination_manager.py','utils/math.py','utils/buffers/delay_buffer.py','utils/buffers/circular_buffer.py']
    test_path = OUTPUT/'offline_tests.xml'
    suites = list(ET.parse(test_path).getroot().iter('testsuite'))
    counts = {key:sum(int(suite.attrib[key]) for suite in suites) for key in ('tests','errors','failures','skipped')}
    assert counts == dict(tests=60,errors=0,failures=0,skipped=0)
    result = dict(status='stopped_on_missing_step_created_reset_buffer',runtime=execution,
        error=read(directory/'error.json'),failure_capture=capture,
        actual_preparation_control_steps=0,actual_policy_control_steps=0,actual_waiting_control_steps=0,
        actor_forward_calls=0,base_optimizer_updates=0,adapter_updates=0,
        formal_train_or_evaluation_started=False,smoke_attempts_used=2,smoke_attempts_remaining=0,
        reserved_smoke_interactions_consumed=19200,original_total_budget=944000,
        original_non_smoke_remaining_budget=924800,failed_slots_not_restored=True,
        checkpoint_sha256=HASHES['B'],repair_simulation_verified=False,simulations_after_failure=0,
        delay_assignment_reached_past=True,native_reset_output_assignment_locations=locations,
        additional_offline_finding='NumPy int64 env_id in update_logs is not JSON serializable',
        cpu_test_scope='installed native method bodies plus locked collector; synthetic scene, physics, actor and checkpoint; not simulation or capability evidence',
        offline_tests=dict(**counts,path=str(test_path),sha256=sha(test_path),seconds=sum(float(suite.attrib['time']) for suite in suites)),
        repair_sources=[dict(path=str(p),sha256=sha(p)) for p in sources],
        native_sources=[dict(path=str(native/p),sha256=sha(native/p)) for p in native_files],
        raw_sources=[dict(path=str(p),sha256=sha(p)) for p in sorted(directory.iterdir()) if p.is_file()],
        attempts=attempts,lock_sha256=sha(OUTPUT/'lock.json'))
    save(DOCS/'EVIDENCE.json',result)
    print(result['status'],result['native_reset_output_assignment_locations'])


if __name__ == '__main__': main()
