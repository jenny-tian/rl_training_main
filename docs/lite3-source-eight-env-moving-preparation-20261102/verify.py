"""Check this CPU package and uncompressed evidence identity; no simulator imports."""
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def verify(folder):
    audit=json.loads((folder/'cpu_audit.json').read_text())
    suite=ET.parse(folder/'regression.junit.xml').getroot().find('testsuite')
    assert suite.attrib['tests']=='410'
    assert all(suite.attrib.get(key,'0')=='0' for key in ('failures','errors','skipped'))
    assert audit['isaac_sim_launches']==audit['gpu_jobs']==audit['source_ppo_updates']==0
    assert json.loads((folder/'next_stage.json').read_text())['user_approved'] is False
    example=folder/'cpu_example'
    report=json.loads((example/'result.json').read_text())
    assert report['synthetic_cpu_fixture'] is True and report['schema']=='lite3_source_eight_env_motion_v1'
    assert report['simulation_control_steps']==96 and report['total_transitions']==768
    assert hashlib.sha256((example/'result.steps.jsonl').read_bytes()).hexdigest()==report['steps_sha256']
    digest=hashlib.sha256()
    with gzip.open(example/'result.events.jsonl.gz','rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    assert digest.hexdigest()==report['events_sha256']
    acceptance=json.loads((example/'cpu_acceptance.json').read_text())
    assert acceptance['accepted'] and acceptance['ordinary_failure_episodes']==1
    assert acceptance['event_review']['motion_samples']==acceptance['event_review']['contact_samples']==384
    return {'status':'cpu_package_verified','tests':410,'example_is_synthetic_cpu':True,
            'event_bytes_sha256':digest.hexdigest(),'runtime_approved':False}


if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent),indent=2))
