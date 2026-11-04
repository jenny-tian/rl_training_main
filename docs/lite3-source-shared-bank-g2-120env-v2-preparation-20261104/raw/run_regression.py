import json
import os
from pathlib import Path
import subprocess
import sys
import time

scratch=Path(__file__).resolve().parent
prior=Path('docs/lite3-source-shared-bank-g2-120env-preparation-20261103/raw/regression_execution.json')
files=[p for p in json.loads(prior.read_text())['command'] if p.startswith('tests/')]
files.append('tests/source_scan/test_scale_reset_contact_cache.py')
overrides=dict(CUDA_VISIBLE_DEVICES='',PYTHONPATH='source/rl_training:.',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
command=[sys.executable,'-m','pytest','-q',*files,'--junitxml='+str(scratch/'regression.junit.xml')]
started=time.monotonic()
with (scratch/'regression.log').open('w') as stream:
    result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,**overrides})
record=dict(command=command,registered_test_files=len(files),environment_overrides=overrides,returncode=result.returncode,
    elapsed_seconds=time.monotonic()-started,gpu_jobs=0,isaac_sim_launches=0)
(scratch/'regression_execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2));print((scratch/'regression.log').read_text()[-12000:])
