import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from rl_training.online_adaptation.registered_development import REGISTERED_DEVELOPMENT_SEEDS,REGISTRATION_SHA256,registered_development_geometry
from rl_training.online_adaptation.terrain_protocol import default_flat_online_protocol,development_geometry,sealed_geometry
from rl_training.online_adaptation.provenance import sha256_file
from rl_training.online_adaptation.managed_trial import attach_session
from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser
from scripts.reinforcement_learning.online_adaptation.run_adaptation_session import worker_arguments


def arguments():
    return ['--base-checkpoint','base.pt','--terrain-split','registered-development','--terrain-seed',str(REGISTERED_DEVELOPMENT_SEEDS[0]),
            '--terrain-stage','rough_4cm','--mode','base','--evaluation-only','--target-warmup-steps','0']


def test_membership_binds_registration_and_preserves_legacy_and_sealed_guards():
    path=Path('docs/lite3_route_new_development_seeds_20261002.json')
    assert sha256_file(path)==REGISTRATION_SHA256
    assert list(REGISTERED_DEVELOPMENT_SEEDS)==json.loads(path.read_text())['terrain_seeds']
    old=default_flat_online_protocol()
    assert old.development_seeds==(101,103,107,109,113)
    for seed in REGISTERED_DEVELOPMENT_SEEDS:
        geometry=registered_development_geometry('rough_4cm',seed)
        assert geometry.step_height==0 and geometry.rough_amplitude==.04
        with pytest.raises(ValueError):development_geometry('rough_4cm',seed)
        with pytest.raises(ValueError):sealed_geometry(seed)
        with pytest.raises(ValueError):registered_development_geometry('rough_2cm',seed)
    for seed in old.development_seeds+old.sealed_seeds:
        with pytest.raises(ValueError):registered_development_geometry('rough_4cm',seed)


@pytest.mark.parametrize('extra', [['--mode','online','--experiment-checkpoint-out','x.pt'],['--stochastic-control'],
                                  ['--target-warmup-steps','1'],['--terrain-stage','rough_2cm'],['--terrain-seed','1009']])
def test_new_split_rejects_learning_exploration_other_geometry_and_sealed_seed(extra):
    with pytest.raises(SystemExit):build_play_parser().parse_args(arguments()+extra)


def test_new_split_requires_managed_runtime_and_propagates_to_frozen_worker(tmp_path):
    parsed=build_play_parser().parse_args(arguments())
    with pytest.raises(ValueError,match='locked managed'):attach_session(parsed,'sac')
    args=SimpleNamespace(algorithm='sac',phase='evaluation',base_checkpoint='base.pt',terrain_seed=REGISTERED_DEVELOPMENT_SEEDS[0],
        terrain_split='registered-development',seed=123,deadline_steps=2000,target_warmup_steps=0,device='cuda:0',worker_options=None,session_dir=str(tmp_path))
    command=worker_arguments(args,output=tmp_path/'trial',steps=2000,trial_id='trial-00000')
    parsed=build_play_parser().parse_args(command)
    assert parsed.terrain_split=='registered-development' and parsed.mode=='base' and parsed.evaluation_only
    assert parsed.session_ledger==str((tmp_path/'ledger').resolve())


def test_registered_runtime_rejects_selection_phase_and_releases_ledger(monkeypatch):
    import rl_training.online_adaptation.managed_trial as module
    closed=[]
    budget=SimpleNamespace(manifest={},state={'trials':{'trial':{'status':'ready','phase':'selection'}}},close=lambda:closed.append(True))
    monkeypatch.setattr(module,'InteractionBudget',lambda path:budget)
    args=SimpleNamespace(session_ledger='ledger',session_run_id='trial',terrain_split='registered-development')
    with pytest.raises(ValueError,match='selection phases'):module.ManagedTrial(args,'sac')
    assert closed==[True]


def test_registered_manifest_roundtrip_and_ledger_phase_guard(tmp_path):
    from rl_training.online_adaptation.interaction_budget import InteractionBudget,validate_manifest
    import copy
    base=tmp_path/'base.pt';base.write_bytes(b'base')
    manifest={'schema':'lite3_session_v1','session_id':'registered-test','algorithm':'sac',
              'base':{'path':str(base),'sha256':sha256_file(base)},'task_objective_sha256':'a'*64,
              'terrain_split':'registered-development','terrain_seed':REGISTERED_DEVELOPMENT_SEEDS[0],'learner_seed':3,
              'worker_configuration':{'arguments':{'terrain_split':'registered-development','terrain_stage':'rough_4cm'}},
              'limits':{'adaptation':2,'selection':0,'evaluation':2,'retention':2},'seal':None}
    directory=tmp_path/'ledger'
    with InteractionBudget(directory,manifest=manifest) as ledger:
        with pytest.raises(ValueError,match='forbids learning'):
            ledger.prepare_trial(phase='adaptation',max_steps=1,output=tmp_path/'bad',input_checkpoint=manifest['base'])
        ledger.lock_evaluation()
        trial=ledger.prepare_trial(phase='evaluation',max_steps=1,output=tmp_path/'eval',input_checkpoint=manifest['base'])
        assert trial['phase']=='evaluation'
    with InteractionBudget(directory) as reopened:
        assert reopened.manifest==manifest and sum(reopened.state['spent'].values())==0
    for seed in (101,1009):
        bad=copy.deepcopy(manifest);bad['terrain_seed']=seed
        with pytest.raises(ValueError):validate_manifest(bad)
