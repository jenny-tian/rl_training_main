"""Verify saved runtime outcomes and CPU repair evidence without launching a simulator."""
import hashlib
import json
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def main(folder):
    checked=0
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);path=(folder/name).resolve()
        assert path.is_relative_to(folder.resolve()) and digest(path)==expected,name
        checked+=1
    load=lambda name:json.loads((folder/name).read_text())
    index=load('raw/runtime_artifacts.json');bulk=0
    for row in index['files']:
        p=Path(row['original_path']) if row['repository_path'] is None else folder/row['repository_path']
        assert p.stat().st_size==row['size_bytes'] and digest(p)==row['sha256'],str(p)
        bulk+=row['repository_path'] is None
    summary=load('raw/runtime_summary.json')
    assert summary['gpu_jobs']==2 and summary['G2_accepted'] is True and summary['G3_accepted'] is False
    assert summary['readonly_transitions']==8160 and summary['new_ppo_updates']==summary['new_training_transitions']==0
    g2=load('raw/runtime/g2/acceptance.json');g3=load('raw/runtime/g3/result.json')
    assert g2['accepted'] and g2['startup_and_first_zero_control_qualified']
    assert g2['reset_batches']==4 and g2['selected_environment_resets']==184
    assert g3['simulation_control_steps']==g3['steps_written']==64 and g3['motion_samples']==256
    assert g3['status']=='failed' and 'prefix speed' in g3['error']
    assert g3['actor_sha256_before']==g3['actor_sha256_after']
    for kind in ('g2','g3'):
        execution=load('raw/runtime/'+kind+'/execution.json')
        assert execution['process_group_empty'] and not execution['monitor_errors']
    assert (folder/'raw/production_ledger_before.json').read_bytes()==(folder/'raw/production_ledger_after_runtime.json').read_bytes()
    assert digest(folder/'raw/production_ledger_after_runtime.json')=='b1d224ba20a93c93c99189d700aeb145d506259f56462eb506def64f8f888189'
    tests=list(ET.parse(folder/'raw/cpu/fast_scope_release.junit.xml').iter('testcase'))
    assert len(tests)==89 and all(t.find('failure') is None and t.find('error') is None and t.find('skipped') is None for t in tests)
    assert load('raw/cpu/g2_reuse_proof.json')['legacy_encoder_and_default_decoder_unchanged']
    assert load('raw/cpu/fast_virtual_benchmark.json')['dense_arrays_reconstructed_exactly']
    supplement=runpy.run_path(str(folder/'g3-fast-runtime/review.py'))['main'](folder/'g3-fast-runtime')
    assert supplement['archive_integrity_verified'] and supplement['readonly_transitions']==226080
    return dict(archive_integrity_verified=True,hashed_files=checked,bulk_files_verified=bulk,
        initial_GPU_jobs=2,actual_GPU_jobs=3,total_readonly_transitions=234240,
        supplemental_run=supplement,G2_accepted=True,G3_accepted=False,latest_targeted_CPU_passed=89,
        new_ppo_updates=0,production_ledger_unchanged=True,model_delivery_complete=False,
        scope='archive integrity and CPU repair evidence; no new G3/G4 qualification')


if __name__=='__main__':print(json.dumps(main(Path(__file__).resolve().parent),indent=2))
