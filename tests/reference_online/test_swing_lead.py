"""提前起抬规则里不依赖仿真的部分。方案见 docs/lite3-online-lead-20261123/PLAN.md。"""
import importlib
import sys
from pathlib import Path

import pytest
import torch

from rl_training.ea_lpl import c1b_s
from rl_training.ea_lpl.swing_lead import SwingLead

TOOLS = str(Path(__file__).resolve().parents[2] / 'scripts/tools')
ALL = torch.ones(1, 4, dtype=torch.bool)


def action(knee=0.):
    a = torch.zeros(1, 12); a[:, 2::3] = knee
    return a


def test_none_returns_the_real_stance_and_swing_is_never_changed():
    stance = torch.tensor([[True, False, True, False]])
    for rule in ('none', 'knee', 'partner', 'age'):
        lead = SwingLead(rule, 1)
        for _ in range(12): out = lead.step(stance, action())
        assert not out[0, 1] and not out[0, 3]  # 摆动的腿仍是摆动
        if rule == 'none': assert torch.equal(out, stance)


def test_knee_rule_marks_a_stance_leg_when_its_base_knee_action_rises():
    lead = SwingLead('knee', 1)
    assert torch.equal(lead.step(ALL, action(.5)), ALL)  # 第一步没有上一步的动作
    a = action(.5); a[0, 2] = .7  # 只有 FL 的膝关节动作增加 0.2
    assert lead.step(ALL, a).tolist() == [[False, True, True, True]]
    assert torch.equal(lead.step(ALL, a), ALL)  # 不再增加
    a2 = a.clone(); a2[0, 5] = .6  # FR 只增加 0.1
    assert torch.equal(lead.step(ALL, a2), ALL)
    lead.reset(torch.tensor([0])); a3 = a2.clone(); a3[0, 8] = 2.
    assert torch.equal(lead.step(ALL, a3), ALL)  # 重置后的第一步不比较


def test_partner_rule_marks_legs_just_after_the_other_diagonal_lands():
    lead = SwingLead('partner', 1)
    swing = torch.tensor([[True, False, False, True]])  # FR、HL 在摆动
    for _ in range(5): out = lead.step(swing, action())
    assert torch.equal(out, swing)
    for k in range(5):
        out = lead.step(ALL, action())
        assert out.tolist() == ([[False, True, True, False]] if k < 3 else [[True] * 4])  # FL、HR 在另一对角着地后的 3 步内被标记


def test_age_rule_marks_legs_after_eight_stance_steps_and_feeds_the_residual():
    lead = SwingLead('age', 1)
    for k in range(9):
        out = lead.step(ALL, action())
        assert out.all() == (k < 7)
    w = torch.tensor([[0., 0., 1.45, 1.05]])
    assert torch.count_nonzero(c1b_s.residual_request(w, ALL, torch.ones(1))) == 0
    assert torch.allclose(c1b_s.residual_request(w, out, torch.ones(1)), c1b_s.residual_request(w, ~ALL, torch.ones(1)))
    with pytest.raises(ValueError): SwingLead('other', 1)
    with pytest.raises(ValueError): lead.step(ALL.float(), action())


@pytest.mark.parametrize('target,lead', [('stairs_up_06', 'knee:0'), ('stairs_up_07', 'none:0'), ('boxes_07', 'age:2')])
def test_entry_registers_the_lead_check(monkeypatch, target, lead):
    monkeypatch.setenv('C1A2_VARIANT', 's2n'); monkeypatch.setenv('C1A2_TARGET', target); monkeypatch.setenv('C1A2_LEAD', lead)
    monkeypatch.delenv('C1A2_SOURCE', raising=False); monkeypatch.delenv('C1A2_CANDIDATE', raising=False)
    monkeypatch.syspath_prepend(TOOLS)
    sys.modules.pop('reference_online_c1a2', None)
    module = importlib.import_module('reference_online_c1a2')
    assert f"online_lead_20261123/{target}/{lead.replace(':', '_')}" in str(module.OUTPUT)
    assert [j['name'] for j in module.jobs()] == [f'target_0_{p:02d}' for p in range(4)]
    p = module.protocol()
    assert p['lead'] == lead and p['residual_rate_limit'] == .02 and p['namespaces']['target'] == 62
    sys.modules.pop('reference_online_c1a2', None)
