import copy
import json

import pytest

from reference_low_stairs_zero_preservation_evaluate import verify_run, wait_training, PARENT_RUNS


@pytest.mark.parametrize('direction',['up','down'])
@pytest.mark.parametrize('field,value',[
    ('checkpoint_sha256','b'*64),('actor_unchanged',False),('probe',True),('direction','wrong'),
    ('action_mode','sampled'),('first_episodes_only',False),('height_set','reference'),
])
def test_rejects_wrong_model_or_protocol(direction,field,value,tmp_path):
    report=json.loads((PARENT_RUNS/f'{direction}_nominal/report.json').read_text())
    report.update(model='new',checkpoint_sha256='a'*64)
    for name,data in [('report',report),('execution',dict(status='completed',exit_code=0,physical_gpu=1)),
                      ('config',dict(reward_scoring='reference_frozen_common'))]:
        (tmp_path/f'{name}.json').write_text(json.dumps(data))
    verify_run(tmp_path,'stairs','nominal','a'*64,'new',direction)
    report[field]=value;(tmp_path/'report.json').write_text(json.dumps(report))
    with pytest.raises(AssertionError):verify_run(tmp_path,'stairs','nominal','a'*64,'new',direction)


@pytest.mark.parametrize('status',['failed','unknown'])
def test_wait_exits_on_failed_training_without_retry(status,monkeypatch,tmp_path):
    import reference_low_stairs_zero_preservation_evaluate as module
    monkeypatch.setattr(module,'directory',lambda _:tmp_path)
    (tmp_path/'execution.json').write_text(json.dumps(dict(status=status)))
    monkeypatch.setattr(module.time,'sleep',lambda _:pytest.fail('不能继续等待失败任务'))
    with pytest.raises(RuntimeError):wait_training()
