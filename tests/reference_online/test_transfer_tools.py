"""参数迁移入口里不依赖仿真的部分：任务表、输出位置、评估设置与原目标一致、平均参数。"""
import hashlib
import importlib
import sys
from pathlib import Path

import pytest
import torch

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')


def load(monkeypatch, variant, target, source=None):
    monkeypatch.setenv('C1A2_VARIANT', variant); monkeypatch.setenv('C1A2_TARGET', target)
    if source: monkeypatch.setenv('C1A2_SOURCE', source)
    else: monkeypatch.delenv('C1A2_SOURCE', raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    for name in ('reference_online_c1a2', 'reference_online_transfer'): sys.modules.pop(name, None)
    return importlib.import_module('reference_online_c1a2')


@pytest.mark.parametrize('variant,target,source', [('s2n', 'stairs_up_065', 's2n:stairs_up_06:60000'),
                                                   ('bs', 'boxes_06', 's2n:stairs_up_06:60000:pooled'),
                                                   ('s2n', 'stairs_up_06', 'bs:boxes_06:40000')])
def test_transfer_only_evaluates_and_keeps_the_target_evaluation_setup(monkeypatch, variant, target, source):
    native = load(monkeypatch, variant, target); reference = native.protocol()
    native_rows = [native.rows('target', part=p) for p in range(4)]
    assert native.SOURCE is None and native.jobs()[0]['name'] == 'smoke'
    module = load(monkeypatch, variant, target, source); protocol = module.protocol()
    mark = int(source.split(':')[2])
    assert [(j['name'], j['kind'], j['mark'], j['part']) for j in module.jobs()] == [(f'target_{mark}_{p:02d}', 'target', mark, p) for p in range(4)]
    assert protocol.pop('transfer_source') == source and protocol == reference  # 起点命名空间、安全过滤、残差上限都与原目标相同
    assert [module.rows('target', part=p) for p in range(4)] == native_rows
    assert module.OUTPUT.parts[-3:] == ('online_transfer_20261120', target, 'from_' + source.replace(':', '_'))
    assert module.SOURCE_FILE.parts[-3:] == (source.split(':')[1], 'train', f'at_{mark}.pt') and module.SOURCE_FILE.parts[-4] == module.RUNS[source.split(':')[0]]
    assert module.SOURCE_POOLED == source.endswith(':pooled')


def test_transfer_rejects_the_same_terrain_and_unknown_variants(monkeypatch):
    for variant, target, source in (('s2n', 'stairs_up_06', 's2n:stairs_up_06:60000'), ('s2f', 'stairs_up_06', 's2n:stairs_up_065:60000'),
                                    ('s2n', 'stairs_up_06', 's2n:stairs_up_065:60000:mean')):
        with pytest.raises(AssertionError): load(monkeypatch, variant, target, source)


def test_pooled_heads_give_every_updated_session_the_mean_and_leave_frozen_alone(monkeypatch):
    load(monkeypatch, 's2n', 'stairs_up_065', 's2n:stairs_up_06:60000:pooled')
    transfer = importlib.import_module('reference_online_transfer')
    digest = lambda w: hashlib.sha256(w.numpy().tobytes()).hexdigest()
    heads = []
    for k in range(6):
        w = torch.tensor([0., 0., .5 * k, -.5 * k], dtype=torch.float64) if k % 2 else torch.zeros(4, dtype=torch.float64)
        heads.append(dict(w=w, updated=bool(k % 2), parameter_sha256=digest(w)))
    pooled = transfer.pooled_heads(heads)
    mean = torch.tensor([0., 0., 1.5, -1.5], dtype=torch.float64)
    assert all(torch.equal(h['w'], mean) and h['parameter_sha256'] == digest(mean) for h in pooled[1::2])
    assert all(a is b for a, b in zip(pooled[::2], heads[::2])) and torch.equal(heads[1]['w'], torch.tensor([0., 0., .5, -.5], dtype=torch.float64))
    # 学习器载入时按参数哈希核对，载入后评估回合用的就是这组参数
    from rl_training.ea_lpl import c1b_s
    session = c1b_s.ParallelSession(False, replicas=1); session.load_evaluation_mean(pooled[1])
    assert session.begin(0).tolist() == mean.tolist()
