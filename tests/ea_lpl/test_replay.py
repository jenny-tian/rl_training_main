import numpy as np
import torch

from rl_training.ea_lpl.basis import basis
from rl_training.ea_lpl.replay import fixed_code_execution_stats, replay_fidelity, switching_diagnostics
from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits

LIMITS = SafetyLimits(residual_limits=torch.tensor([0.24, 0.20, 0.20] * 4))


def synthetic_recording(steps=40, n=3, contact_rows=()):
    rec = dict(base_action=np.zeros((steps, n, 12), np.float32), gravity_xy=np.zeros((steps, n, 2), np.float32),
               base_height=np.full((steps, n, 1), 0.35, np.float32), joint_vel=np.zeros((steps, n, 12), np.float32),
               nonfoot_contact=np.zeros((steps, n, 1), bool), control_timed_out=np.zeros((steps, n, 1), bool),
               joint_lower=np.full((n, 12), -10.0, np.float32), joint_upper=np.full((n, 12), 10.0, np.float32),
               default_joint_pos=np.zeros((n, 12), np.float32), action_scale=np.full((n, 12), 0.25, np.float32),
               basis_index=np.zeros(n, np.int64), u=np.tile(np.array([[1.0, 0.0]], np.float32), (n, 1)),
               episode_end=np.full(n, steps))
    for row in contact_rows:
        rec["nonfoot_contact"][:, row] = True
    request = torch.einsum("njk,nk->nj", basis("sag").expand(n, 12, 2), torch.tanh(torch.as_tensor(rec["u"])))
    rec["request"] = request.numpy()
    live = SafetyFilter(LIMITS)
    from rl_training.ea_lpl.replay import state_at
    rec["executed_action"] = np.stack([live.project(torch.zeros(n, 12), request, state_at(rec, t)).action.numpy()
                                       for t in range(steps)])
    return rec


def test_fidelity_and_fixed_code_statistics():
    rec = synthetic_recording(contact_rows=(2,))
    assert replay_fidelity(rec, LIMITS) <= 1e-6
    stats = fixed_code_execution_stats(rec)
    assert abs(stats["p_guard"] - 1 / 3) < 1e-9 and abs(stats["p_proj"] - 1 / 3) < 1e-9


def test_switching_diagnostics_detect_rate_limits():
    result = switching_diagnostics(synthetic_recording(steps=60), basis("sag"), LIMITS, seed=1)
    assert result["decisions"] > 0
    assert 0.0 <= result["informative_fraction_macro"] <= 1.0
    assert result["projection_activation"] > 0.0
