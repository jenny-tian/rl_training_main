"""Review the archived synthetic example only. No simulation, CUDA or PPO update."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'source/rl_training'),str(ROOT/'scripts/tools')]
import torch
from source_live_ppo_validation import validate


def verify():
    torch.set_num_threads(1)
    require=lambda condition: condition or (_ for _ in ()).throw(AssertionError('archive check failed'))
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,entry in provenance['copies'].items():
        require(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==entry['sha256'])
    for key in ('input_sha256','dependency_sha256'):
        for name,digest in provenance[key].items():
            path=ROOT/name if key=='input_sha256' else Path(name)
            require(hashlib.sha256(path.read_bytes()).hexdigest()==digest)
    counts=[]
    for name,expected in (('regression',461),('budget_followup',42)):
        suites=ET.parse(HERE/f'raw/{name}.junit.xml').getroot().findall('testsuite')
        require(sum(int(s.attrib['tests']) for s in suites)==expected)
        require(all(int(s.attrib[k])==0 for s in suites for k in ('errors','failures','skipped')))
        counts.append(expected)
    report=json.loads((HERE/'cpu_example/result.json').read_text())
    plan={key:report[key] for key in ('attempt_id','base_revision','checkpoint_sha256')}
    result=validate(HERE/'cpu_example',plan,{'returncode':0,'process_group_empty':True},allow_synthetic=True)
    require(result['accepted'] and result['synthetic_cpu_fixture'])
    require(not validate(HERE/'cpu_example',plan,{'returncode':0,'process_group_empty':True})['accepted'])
    require(provenance['production_reserved_transitions']==0 and not provenance['cuda_initialized'])
    require(json.loads((HERE/'next_stage.json').read_text())['user_approved'] is False)
    require(not torch.cuda.is_initialized())
    return {'status':'cpu_preparation_archive_verified','regression_tests':counts[0],
            'followup_tests':counts[1],'distinct_tests':462,'isaac_sim_launches':0,'gpu_jobs':0,
            'real_source_updates':0,'synthetic_example_acceptance':result,
            'production_inputs':len(provenance['input_sha256']),
            'core_dependencies':len(provenance['dependency_sha256'])}


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
