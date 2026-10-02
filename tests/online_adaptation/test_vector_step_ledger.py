import json
import pytest
from rl_training.vector_step_ledger import VectorStepLedger


def test_vector_steps_charge_every_environment_and_cannot_exceed_budget(tmp_path):
    p=tmp_path/'ledger.json';ledger=VectorStepLedger(p,num_envs=8,max_transitions=16,metadata={})
    for _ in range(2):ledger.reserve();ledger.confirm()
    with pytest.raises(ValueError,match='exceeds'):ledger.reserve()
    ledger.finish()
    s=json.loads(p.read_text());assert s['charged']==s['confirmed']==16 and s['uncertain_transitions']==0
    with pytest.raises(ValueError,match='exists'):VectorStepLedger(p,num_envs=8,max_transitions=16,metadata={})


def test_failed_step_keeps_whole_batch_charged_as_uncertain(tmp_path):
    ledger=VectorStepLedger(tmp_path/'ledger.json',num_envs=8,max_transitions=16,metadata={})
    ledger.reserve()
    with pytest.raises(ValueError):ledger.reserve()
    with pytest.raises(ValueError):ledger.finish()
    ledger.finish(aborted=True)
    assert ledger.state['charged']==8 and ledger.state['confirmed']==0 and ledger.state['uncertain_transitions']==8
