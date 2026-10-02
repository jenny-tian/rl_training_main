import json
from types import SimpleNamespace

import pytest

from scripts.reinforcement_learning.online_adaptation.cli import build_play_parser
from scripts.reinforcement_learning.online_adaptation.run_adaptation_session import worker_arguments, parser_for
from rl_training.online_adaptation.managed_trial import worker_configuration


def base_args():
    return ['--base-checkpoint','base.pt','--terrain-split','development','--terrain-seed','101',
            '--mode','residual_frozen','--evaluation-only','--fixed-code','.5','-.5']


def test_fixed_code_parses_and_is_bound_to_source_configuration():
    a=build_play_parser().parse_args(base_args())
    assert a.fixed_code == [.5,-.5]
    b=build_play_parser().parse_args(base_args()[:-2]+['0','0'])
    assert worker_configuration(a) != worker_configuration(b)


@pytest.mark.parametrize('extra', [
    ['--mode','online','--experiment-checkpoint-out','out.pt'], ['--stochastic-control'],
    ['--experiment-checkpoint-in','old.pt'], ['--residual-checkpoint','old.pt'],
    ['--target-warmup-steps','4000'], ['--fixed-code','nan','0'], ['--fixed-code','1.01','0'],
])
def test_fixed_code_rejects_learning_randomness_and_imported_checkpoints(extra):
    with pytest.raises(SystemExit):build_play_parser().parse_args(base_args()+extra)


def test_supervisor_preserves_vector_options_and_freezes_without_imported_state(tmp_path):
    options=tmp_path/'options.json';options.write_text(json.dumps({'fixed-code':[.5,-.5]}))
    args=SimpleNamespace(algorithm='sac',phase='evaluation',base_checkpoint='base.pt',terrain_seed=101,
                         seed=71,deadline_steps=2000,target_warmup_steps=0,device='cuda:0',worker_options=str(options))
    parsed=parser_for('sac').parse_args(worker_arguments(args,output=tmp_path,steps=2000))
    assert parsed.mode=='residual_frozen' and parsed.evaluation_only
    assert parsed.fixed_code==[.5,-.5]
    assert parsed.experiment_checkpoint_in is None
    with pytest.raises(ValueError,match='frozen'):
        worker_arguments(args,output=tmp_path,steps=2000,phase='adaptation')
    with pytest.raises(ValueError,match='frozen'):
        worker_arguments(args,output=tmp_path,steps=2000,checkpoint={'path':'old.pt'})


def test_managed_fixed_code_trial_is_frozen_and_code_change_is_rejected(tmp_path):
    from rl_training.online_adaptation.interaction_budget import InteractionBudget
    from rl_training.online_adaptation.managed_trial import attach_session
    from rl_training.online_adaptation.provenance import sha256_file
    from rl_training.online_adaptation.task_objective import FiniteHorizonObjective
    base=tmp_path/'base.pt';base.write_bytes(b'fixture')
    args=build_play_parser().parse_args(base_args()+['--base-checkpoint',str(base),
        '--task-objective','finite-horizon-v1','--require-track-completion',
        '--update-schedule','control-step','--max-steps','4','--run-dir',str(tmp_path/'run')])
    manifest={'schema':'lite3_session_v1','session_id':'fixed-code-test','algorithm':'sac',
        'base':{'path':str(base),'sha256':sha256_file(base)},
        'task_objective_sha256':FiniteHorizonObjective().sha256(),'terrain_split':'development',
        'terrain_seed':101,'learner_seed':7,'worker_configuration':worker_configuration(args),
        'limits':{'adaptation':8,'selection':2,'evaluation':8,'retention':2},'seal':None}
    with InteractionBudget(tmp_path/'ledger',manifest=manifest) as budget:
        budget.lock_evaluation()
        trial=budget.prepare_trial(phase='evaluation',max_steps=4,output=tmp_path/'run',input_checkpoint=manifest['base'])
    args.session_ledger,args.session_run_id=str(tmp_path/'ledger'),trial['id']
    args.fixed_code=[0.,0.]
    with pytest.raises(ValueError,match='configuration'):
        attach_session(args,'sac')
    args.fixed_code=[.5,-.5]
    managed=attach_session(args,'sac')
    try:
        assert managed.frozen
        managed.reserve(1);managed.commit(1)
        with pytest.raises(ValueError,match='mutated'):
            managed.finish({'steps':1,'updates':1})
        managed.finish({'steps':1,'updates':0})
    finally:managed.close()
    with pytest.raises(ValueError,match='duplicate'):
        attach_session(args,'sac')
