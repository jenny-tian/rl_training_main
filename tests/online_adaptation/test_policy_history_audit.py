import json
import numpy as np
import pytest
from docs.analysis.audit_policy_action_history import main


@pytest.mark.parametrize('defect', [None, 'reset', 'critic', 'truncated', 'nonfinite'])
def test_history_audit_rejects_incomplete_or_inconsistent_history(tmp_path, monkeypatch, defect):
    trajectory = tmp_path/'trajectory'
    trajectory.mkdir()
    (tmp_path/'report.json').write_text(json.dumps({'status':'complete','ledger':{'confirmed':3}}))
    (trajectory/'manifest.json').write_text(json.dumps({'status':'complete','steps':3,'chunks':[{'file':'chunk.npz','steps':3}]}))
    requested = np.zeros((3,1,12),dtype=np.float32)
    requested[0,0,0] = 4
    emitted = requested.copy();emitted[0,0,0]=3.6
    actor = np.zeros((3,1,45),dtype=np.float32);actor[0,0,-12]=3.6
    critic = np.zeros((3,1,48),dtype=np.float32);critic[0,0,-12]=3.6
    if defect=='reset': actor[2,0,-1]=1
    if defect=='critic': critic[0,0,-1]=1
    if defect=='nonfinite':actor[0,0,-1]=np.nan
    data = dict(requested_action=requested,emitted_action=emitted,next_policy_observation=actor,
                next_critic_observation=critic,done=np.array([[False],[False],[True]]))
    if defect=='truncated':data={k:v[:2] for k,v in data.items()}
    np.savez(trajectory/'chunk.npz',**data)
    output=tmp_path/'audit.json'
    monkeypatch.setattr('sys.argv',['audit','--training',str(tmp_path),'--output',str(output)])
    if defect in ('truncated','nonfinite'):
        with pytest.raises(AssertionError):main()
    else:
        main()
        assert json.loads(output.read_text())['emitted_action_history_compatible'] is (defect is None)
