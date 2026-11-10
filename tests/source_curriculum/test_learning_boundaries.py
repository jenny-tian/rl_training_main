"""Exercise real reward and PPO boundaries without launching Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def height_reward():
    path = ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py'
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'base_height_l2')
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                           function], type_ignores=[])
    namespace = {'torch': torch, 'SceneEntityCfg': lambda name: SimpleNamespace(name=name),
                 'get_gait_level_tensor': lambda env: 1.0}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace['base_height_l2']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), 1.1e6])
def test_one_invalid_scan_does_not_disable_other_environment_height_rewards(bad):
    roots = torch.tensor([[0., 0., .4], [0., 0., .5], [0., 0., .6]])
    scan = torch.zeros(3, 9, 3)
    scan[2, :, 2] = .2
    env = SimpleNamespace(scene={
        'robot': SimpleNamespace(data=SimpleNamespace(root_pos_w=roots, root_link_pos_w=roots)),
        'scan': SimpleNamespace(data=SimpleNamespace(ray_hits_w=scan))})
    reward = height_reward()
    normal = reward(env, .375, sensor_cfg=SimpleNamespace(name='scan'))
    scan[1, 4, 2] = bad
    actual = reward(env, .375, sensor_cfg=SimpleNamespace(name='scan'))
    torch.testing.assert_close(actual[[0, 2]], normal[[0, 2]])
    assert actual[1] == 0
    assert torch.isfinite(actual).all()


def test_plane_height_reward_does_not_require_a_scanner():
    roots = torch.tensor([[0., 0., .4], [0., 0., .5]])
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(root_pos_w=roots))})
    torch.testing.assert_close(height_reward()(env, .375), (roots[:, 2] - .375).square())


class FakeVecEnv:
    def __init__(self, *, timeouts=True):
        self.reset_terminated = torch.tensor([False, False, True, True])
        self.reset_time_outs = torch.tensor([False, True, False, True])
        self.num_envs = 4
        self.observation_space = 'preserved'
        self.rewards = torch.ones(4)
        self.dones = (self.reset_terminated | self.reset_time_outs).long()
        self.extras = {'time_outs': self.reset_time_outs} if timeouts else {}
        self.unwrapped = self

    def step(self, actions):
        return {'policy': actions}, self.rewards, self.dones, self.extras


def test_actual_ppo_bootstraps_only_nonfailure_timeouts():
    from rsl_rl.algorithms import PPO
    from rl_training.source_curriculum.learning import FailurePriorityVecEnv

    raw = FakeVecEnv()
    wrapped = FailurePriorityVecEnv(raw)
    observations, rewards, dones, extras = wrapped.step(torch.zeros(4, 12))
    actor = SimpleNamespace(update_normalization=lambda obs: None, reset=lambda done: None)
    stored = []
    learner = SimpleNamespace(actor=actor, critic=actor, rnd=None, gamma=.99, device='cpu',
                              transition=SimpleNamespace(values=torch.full((4, 1), 5.), clear=lambda: None),
                              storage=SimpleNamespace(add_transition=lambda t: stored.append(t.rewards.clone())))
    PPO.process_env_step(learner, observations, rewards, dones, extras)
    torch.testing.assert_close(stored[0], torch.tensor([1., 5.95, 1., 1.]))
    # Preserve the original Gym termination masks, reward, and episode ending.
    assert raw.reset_time_outs.tolist() == [False, True, False, True]
    assert dones.tolist() == [0, 1, 1, 1]
    torch.testing.assert_close(rewards, torch.ones(4))
    assert wrapped.num_envs == 4 and wrapped.observation_space == 'preserved'
    assert wrapped.unwrapped is raw


def test_finite_horizon_does_not_gain_timeout_bootstrapping():
    from rl_training.source_curriculum.learning import FailurePriorityVecEnv
    raw = FakeVecEnv(timeouts=False)
    _, _, _, extras = FailurePriorityVecEnv(raw).step(torch.zeros(4, 12))
    assert 'time_outs' not in extras


def test_failure_mask_shape_mismatch_is_rejected():
    from rl_training.source_curriculum.learning import FailurePriorityVecEnv
    raw = FakeVecEnv()
    raw.reset_terminated = raw.reset_terminated[:1]
    with pytest.raises(ValueError, match='shape'):
        FailurePriorityVecEnv(raw).step(torch.zeros(4, 12))


def test_learning_correction_requires_task_opt_in():
    from rl_training.source_curriculum.learning import FailurePriorityVecEnv, wrap_source_curriculum_learning
    raw = FakeVecEnv()
    raw.cfg = SimpleNamespace()
    assert wrap_source_curriculum_learning(raw) is raw
    raw.cfg.source_failure_priority_bootstrap = True
    assert isinstance(wrap_source_curriculum_learning(raw), FailurePriorityVecEnv)
    raw.cfg.source_failure_priority_bootstrap = 'false'
    with pytest.raises(ValueError, match='boolean'):
        wrap_source_curriculum_learning(raw)
