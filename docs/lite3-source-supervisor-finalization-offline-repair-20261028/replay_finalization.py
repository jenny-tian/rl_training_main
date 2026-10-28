
import json,hashlib,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts/tools'))
from source_parallel_terrain_startup_supervisor import finalize_run
root=Path.cwd();archive=root/'docs/lite3-source-shared-bank-2env-tensor-repair-20261028'
paths=['scripts/tools/source_parallel_terrain_startup_supervisor.py','scripts/tools/source_shared_report_validation.py','scripts/tools/source_shared_tensor_contact.py','scripts/tools/source_shared_contact_report.py']
hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
execution=json.loads((archive/'execution.json').read_text())
with tempfile.TemporaryDirectory(prefix='lite3_finalization_offline_') as d:
    replay=Path(d)
    for name in ('result.json','gpu_samples.jsonl'):
        (replay/name).write_bytes((archive/name).read_bytes())
    outcome=finalize_run(run_dir=replay,root=root,execution=execution,source_hashes=hashes,dependency_hashes={},num_envs=2,shared_bank=True)
    assert outcome['status']=='diagnostic_smoke_passed'
    assert not any(k.split('.')[0] in {'rl_training','pxr','omni','isaacsim','torch'} for k in sys.modules)
    print(json.dumps({'mode':'offline_finalization_replay','simulation_started':False,'original_runtime_revision':'93f1cd46278e5a0c0bf7882c0df93a20d6b8bb71','checked_hashes_are_current_replay_sources':True,'source_sha256':hashes,'outcome':outcome},indent=2))
