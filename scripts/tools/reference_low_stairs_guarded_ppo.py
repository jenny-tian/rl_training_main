"""将能力保持损失接入原生 PPO 的同一次反向传播和梯度裁剪。"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import types

import numpy as np
import torch
from tensordict import TensorDict

from reference_low_stairs_preservation import GROUPS, joint_target_preservation_loss
from reference_stairs_eval import sha

PPO_UPDATE_SHA = '507b4a88853dc6eda17c7c2e79bd081430c97ec8ca2e2efc24572729c94907b0'
KEY = 'capability_preservation_v1'
COEFFICIENT = .5
PER_GROUP = 128


class Preserver:
    def __init__(self, path, device, *, coefficient=COEFFICIENT):
        self.path = Path(path)
        self.digest = sha(self.path)
        metadata = json.loads(self.path.with_suffix('.json').read_text())
        if metadata['bank_sha256'] != self.digest or metadata['seed'] != 20261116 or metadata['diagnostic_bank_used']:
            raise ValueError('训练样本库来源不符')
        if coefficient < 0:
            raise ValueError('能力保持系数不能为负')
        with np.load(self.path) as data:
            self.obs = torch.tensor(data['observation'], device=device)
            self.target = torch.tensor(data['parent_action'], device=device).detach()
            self.groups = torch.tensor(data['group'], device=device, dtype=torch.long)
        if self.obs.shape != (len(self.groups), 45) or self.target.shape != (len(self.groups), 12):
            raise ValueError('样本库维度错误')
        if not torch.isfinite(self.obs).all() or not torch.isfinite(self.target).all():
            raise ValueError('样本库数值异常')
        self.indices = [torch.where(self.groups == group)[0] for group in range(len(GROUPS))]
        if any(len(ids) == 0 for ids in self.indices):
            raise ValueError('缺少能力类别')
        self.coefficient = coefficient
        self.calls = 0
        self.loss_sum = 0.

    def state(self):
        return dict(schema=1, bank_sha256=self.digest, seed=20261116,
                    coefficient=self.coefficient, per_group=PER_GROUP, calls=self.calls)

    def restore(self, state):
        expected = self.state()
        if set(state) != set(expected) or any(state[k] != expected[k] for k in expected if k != 'calls'):
            raise ValueError('能力保持快照与当前配置不符')
        if not isinstance(state['calls'], int) or state['calls'] < 0:
            raise ValueError('样本游标错误')
        self.calls = state['calls']

    def sample(self):
        start = self.calls * PER_GROUP
        ids = torch.cat([values[torch.arange(start, start + PER_GROUP, device=values.device) % len(values)]
                         for values in self.indices])
        return TensorDict({'policy': self.obs[ids]}, batch_size=[len(ids)]), self.target[ids], self.groups[ids]

    def loss(self, actor):
        obs, target, groups = self.sample()
        prediction = actor(obs)
        loss = joint_target_preservation_loss(prediction, target, groups) * self.coefficient
        if not torch.isfinite(loss):
            raise FloatingPointError('能力保持损失异常')
        self.calls += 1
        self.loss_sum += float(loss.detach())
        return loss


def adapted_update():
    """校验完整原方法，只插入一行；库文件和 PPO 类本身均保持不变。"""
    from rsl_rl.algorithms import PPO
    original = inspect.getsource(PPO.update)
    if hashlib.sha256(original.encode()).hexdigest() != PPO_UPDATE_SHA:
        raise ValueError('原生 PPO.update 版本变化，需要重新审查')
    source = textwrap.dedent(original)
    marker = '        self.optimizer.zero_grad()\n'
    if source.count(marker) != 1:
        raise ValueError('PPO 插入点不唯一')
    adapted = source.replace(marker, '        loss = loss + self.capability_preserver.loss(self.actor)\n' + marker)
    namespace = dict(PPO.update.__globals__)
    exec(compile(adapted, '<guarded_native_ppo_update>', 'exec'), namespace)
    return namespace['update'], source, adapted


def attach(algorithm, preserver):
    if algorithm.actor.is_recurrent or algorithm.actor.obs_normalization or algorithm.actor.obs_groups != ['policy']:
        raise ValueError('本阶段只支持已审查的无观测归一化 MLP actor')
    if algorithm.rnd or algorithm.symmetry or algorithm.is_multi_gpu:
        raise ValueError('本阶段不使用额外 PPO 插件或多卡')
    if hasattr(algorithm, 'capability_preserver'):
        raise ValueError('辅助损失不能重复安装')
    update, _, _ = adapted_update()
    algorithm.capability_preserver = preserver
    bound = types.MethodType(update, algorithm)
    def guarded_update():
        start_calls, start_loss = preserver.calls, preserver.loss_sum
        result = bound()
        count = preserver.calls - start_calls
        if count != algorithm.num_learning_epochs * algorithm.num_mini_batches:
            raise RuntimeError('辅助损失与原生 PPO 小批次数不一致')
        result['capability_preservation'] = (preserver.loss_sum - start_loss) / count
        return result
    algorithm.update = guarded_update


def attach_saving(runner, preserver):
    original = runner.save
    def save(path, infos=None):
        metadata = dict(infos or {})
        if KEY in metadata:
            raise ValueError('调用者不能覆盖辅助损失快照')
        metadata[KEY] = preserver.state()
        original(path, infos=metadata)
    runner.save = save
