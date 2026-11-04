from dataclasses import replace
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from rl_training.online_adaptation.safety import SafetyFilter, SafetyLimits
from rl_training.online_adaptation.types import SafetyState
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.training_env import SourceTrainingRuntime, SourceTrainingVecEnv
from rl_training.source_scan.training_lifecycle import SourceEpisodeLifecycle, SourceRunSeeds
from scripts.tools.source_training_preflight import runner_config


def state(batch=3):
    return SafetyState(
        projected_gravity_xy=torch.zeros(batch, 2), base_height=torch.full((batch, 1), .4),
        joint_pos=torch.zeros(batch, 12), joint_vel=torch.zeros(batch, 12),
        joint_lower=torch.full((batch, 12), -1.), joint_upper=torch.full((batch, 12), 1.),
        default_joint_pos=torch.zeros(batch, 12), action_scale=torch.ones(1, 12),
        nonfoot_contact=torch.zeros(batch, 1, dtype=torch.bool),
        control_timed_out=torch.zeros(batch, 1, dtype=torch.bool))


def test_completion_requires_consecutive_stable_hold_and_resets_only_ended_rows():
    lifecycle = SourceEpisodeLifecycle(3, "cpu", SafetyLimits(torch.ones(12)))
    progress = torch.full((3,), 9.)
    for step in range(1, 50):
        lifecycle.update(state(), progress, torch.full((3,), step))
    assert lifecycle.hold.tolist() == [49, 49, 49]
    assert not lifecycle.completed.any()
    contact = state()
    contact.nonfoot_contact[1] = True
    lifecycle.update(contact, progress, torch.full((3,), 50))
    assert lifecycle.completed.tolist() == [True, False, True]
    assert lifecycle.hold.tolist() == [50, 0, 50]
    lifecycle.reset(torch.tensor([0]))
    assert lifecycle.hold.tolist() == [0, 0, 50]


def test_safety_and_timeout_are_distinct_and_invalid_state_does_not_mutate():
    lifecycle = SourceEpisodeLifecycle(3, "cpu", SafetyLimits(torch.ones(12)))
    sample = state()
    sample.base_height[1] = .1
    lifecycle.update(sample, torch.zeros(3), torch.tensor([1999, 100, 2000]))
    assert lifecycle.unsafe.tolist() == [False, True, False]
    assert lifecycle.timeout.tolist() == [False, False, True]
    before = lifecycle.contact_steps.clone()
    sample.joint_vel[2, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        lifecycle.update(sample, torch.ones(3), torch.full((3,), 2000))
    assert torch.equal(before, lifecycle.contact_steps)


def test_contact_grace_and_interrupted_progress_cannot_count_as_completion():
    lifecycle = SourceEpisodeLifecycle(3, "cpu", SafetyLimits(torch.ones(12)))
    sample = state()
    sample.nonfoot_contact[0] = True
    for step in range(1, 12):
        lifecycle.update(sample, torch.full((3,), 9.), torch.full((3,), step))
        assert bool(lifecycle.unsafe[0]) == (step > 10)
    lifecycle.update(state(), torch.tensor([9., 8.99, 9.]), torch.full((3,), 12))
    assert lifecycle.hold.tolist() == [1, 0, 12]


def test_selective_filter_reset_preserves_other_pause_contact_and_residual_history():
    safety = SafetyFilter(SafetyLimits(torch.ones(12)))
    sample = state()
    sample.base_height[1] = .1
    sample.nonfoot_contact[2] = True
    safety.project(torch.zeros(3, 12), torch.full((3, 12), .01), sample)
    before = safety.temporal_snapshot(torch.zeros(3, 12))
    safety.reset_temporal_state(torch.tensor([0, 2]))
    after = safety.temporal_snapshot(torch.zeros(3, 12))
    for key in before:
        assert torch.equal(before[key][1], after[key][1])
        assert not after[key][[0, 2]].any()
    with pytest.raises(ValueError):
        safety.reset_temporal_state(torch.tensor([3]))


def test_training_seeds_are_disjoint_and_global_torch_rng_is_untouched():
    before = torch.random.get_rng_state().clone()
    runs = [SourceRunSeeds(i) for i in (1, 2, 3)]
    assert [r.geometry for r in runs] == [30000, 33000, 36000]
    values = [value for r in runs for value in (r.ppo, r.geometry, r.sampler, r.reset_state_id)]
    assert len(values) == len(set(values))
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(ValueError):
        SourceRunSeeds(True)


class Scene(dict):
    pass


class FakeManagerEnv:
    """Exercise production callbacks around a small synchronous CPU transition."""
    def __init__(self):
        self.num_envs, self.device = 3, "cpu"
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(3, dtype=torch.long)
        self.cfg = NS(source_training_run=1, source_terrain_level="L1", seed=130001,
                      source_parallel_sampler_seed=131001,
                      scene=NS(terrain=NS(terrain_generator=NS(
                          seed=30000, sub_terrains=dict.fromkeys(SOURCE_RESET_BLOCK_COUNTS)))))
        self.scene = Scene(robot=NS(data=NS(root_link_pos_w=torch.zeros(3, 3))))
        self.scene.terrain = NS(env_origins=torch.zeros(3, 3), terrain_types=torch.tensor([0, 1, 2]))
        self.executed = torch.zeros(3, 12)
        self.observation_manager = NS(compute=self.observations)
        self.callbacks = []
        self.safety_state = state()
        self.terms = {}
        self.termination_manager = NS(get_term=lambda name: self.terms[name], active_terms=[
            "source_training_safety", "source_training_completion", "source_training_timeout"])

    def add_pre_reset_callback(self, callback):
        self.callbacks.append(callback)

    def observations(self, update_history=False):
        return {"policy": torch.cat([torch.zeros(3, 33), self.executed], dim=1),
                "critic": self.scene["robot"].data.root_link_pos_w[:, :1].clone()}


class FakeRslEnv:
    def __init__(self):
        self.unwrapped = FakeManagerEnv()
        self.num_envs, self.num_actions, self.device, self.clip_actions = 3, 12, "cpu", None
        self.cfg = self.unwrapped.cfg
        env = self.unwrapped
        adapter = NS(safety_state=lambda timeout: replace(env.safety_state, control_timed_out=timeout),
                     contact_sensor=NS(data=NS(net_forces_w=torch.zeros(3, 17, 3))))
        self.runtime = SourceTrainingRuntime(env, adapter)

    def get_observations(self):
        return TensorDict(self.unwrapped.observations(), batch_size=[3])

    def step(self, actions):
        env = self.unwrapped
        env.executed = actions.clone()
        env.episode_length_buf += 1
        env.common_step_counter += 1
        env.scene["robot"].data.root_link_pos_w[:, 0] += 2.
        lifecycle = self.runtime.evaluate()
        # Three manager terms may request the same lifecycle in one step.
        assert self.runtime.evaluate() is lifecycle
        env.terms = {"source_training_safety": lifecycle.unsafe.clone(),
                     "source_training_completion": lifecycle.completed.clone(),
                     "source_training_timeout": lifecycle.timeout.clone(),
                     "source_terrain_tile_bounds": torch.zeros(3, dtype=torch.bool),
                     "illegal_contact": torch.zeros(3, dtype=torch.bool),
                     "bad_orientation_2": torch.zeros(3, dtype=torch.bool)}
        env.termination_manager.active_terms = list(env.terms)
        env.reset_terminated = lifecycle.unsafe | lifecycle.completed
        env.reset_time_outs = lifecycle.timeout.clone()
        done = env.reset_terminated | env.reset_time_outs
        ids = done.nonzero().flatten()
        if len(ids):
            for callback in env.callbacks:
                callback(ids)
            env.episode_length_buf[ids] = 0
            env.scene.terrain.terrain_types[ids] = 14
            env.scene["robot"].data.root_link_pos_w[ids, 0] = 1000.
            env.executed[ids] = 0.
        return self.get_observations(), torch.ones(3), done.long(), {"time_outs": env.reset_time_outs}


def wrapper():
    base = FakeRslEnv()
    return SourceTrainingVecEnv(base, runtime=base.runtime)


def make_runner(env):
    cfg = runner_config()
    cfg["algorithm"]["class_name"] = "rl_training.source_scan.training_ppo:SourcePPO"
    cfg["num_steps_per_env"] = 2
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    return OnPolicyRunner(env, cfg, log_dir=None, device="cpu")


def test_real_rsl_storage_keeps_gaussian_request_and_timeout_uses_pre_reset_critic():
    torch.manual_seed(23)
    env = wrapper()
    runner = make_runner(env)
    with torch.no_grad():
        runner.alg.actor.distribution.log_std_param.fill_(3.)
        for parameter in runner.alg.critic.parameters():
            parameter.zero_()
        runner.alg.critic.mlp[-1].bias.fill_(7.)
    observations = env.get_observations()
    env.unwrapped.episode_length_buf[0] = 1999
    actions = runner.alg.act(observations)
    sampled, old_logp = actions.clone(), runner.alg.transition.actions_log_prob.clone()
    new_obs, rewards, dones, extras = env.step(actions)
    assert torch.equal(actions, sampled)
    assert not torch.equal(env.last_trace.executed_action, sampled)
    assert extras["source_terminal"]["observations"]["critic"][0].item() == 2.
    assert new_obs["critic"][0].item() == 1000.
    assert torch.equal(new_obs["policy"][1, -12:], env.last_trace.executed_action[1])
    # Make value depend on the observation so terminal(2) and post-reset(1000)
    # and cached current-state value(7) are observably different.
    runner.alg.critic.forward = lambda obs: obs["critic"] * 3.
    runner.alg.process_env_step(new_obs, rewards, dones, extras)
    storage = runner.alg.storage
    assert torch.equal(storage.actions[0], sampled)
    assert torch.equal(storage.actions_log_prob[0].squeeze(-1), old_logp)
    assert storage.rewards[0, 0].item() == pytest.approx(1. + .99 * 6.)
    assert storage.rewards[0, 1].item() == 1.
    ledger = env.exposure.report()
    assert ledger["flat"] == {"episode_assignments": 1, "transitions": 1}
    assert ledger["friction_alternating_regions"] == {"episode_assignments": 1, "transitions": 0}
    assert env.runtime.lifecycle.hold[0] == 0
    assert env.unwrapped.episode_length_buf.tolist() == [0, 1, 1]


def test_terminated_plus_timeout_gets_no_bootstrap_and_real_update_is_finite():
    torch.manual_seed(24)
    env = wrapper()
    runner = make_runner(env)
    obs = env.get_observations()
    for step in range(2):
        actions = runner.alg.act(obs)
        if step == 0:
            env.unwrapped.episode_length_buf[1] = 1999
            # A failure occurring after the action (not an unsafe pre-state).
            original_step = env.env.step
            def step_with_fall(action):
                env.unwrapped.safety_state.base_height[1] = .1
                return original_step(action)
            env.env.step = step_with_fall
        obs, rewards, dones, extras = env.step(actions)
        env.env.step = original_step
        env.unwrapped.safety_state.base_height[1] = .4
        runner.alg.process_env_step(obs, rewards, dones, extras)
        if step == 0:
            assert not extras["time_outs"].any()
            assert runner.alg.storage.rewards[0, 1].item() == 1.
    with torch.no_grad():
        runner.alg.compute_returns(obs)
    losses = runner.alg.update()
    assert losses and all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert runner.alg.optimizer.state


def test_pre_action_pause_or_invalid_observation_stops_without_physics_step():
    env = wrapper()
    env.unwrapped.safety_state.base_height[2] = .1
    with pytest.raises(RuntimeError, match="pre-action safety pause"):
        env.step(torch.zeros(3, 12))
    assert env.unwrapped.common_step_counter == 0
    env.unwrapped.safety_state.base_height[2] = .4
    env.unwrapped.executed[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        env.get_observations()


def test_raw_nan_contact_force_cannot_be_hidden_by_boolean_contact_conversion():
    env = wrapper()
    env.runtime.adapter.contact_sensor.data.net_forces_w[1, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="contact forces"):
        env.step(torch.zeros(3, 12))
    assert env.unwrapped.common_step_counter == 0
    assert env.runtime.safety._previous_residual is None


@pytest.mark.parametrize("run", [1, 2, 3])
def test_training_config_executes_seed_overrides_and_registers_manager_endings(run):
    # Execute the actual subclass constructor with a record-only base config.
    # This is CPU wiring evidence, not an Isaac configclass/PhysX runtime test.
    path = Path(__file__).resolve().parents[2] / (
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/"
        "config/quadruped/deeprobotics_lite3/source_training_env_cfg.py")
    class Base:
        source_training_run = 1
        source_parallel_sampler_enabled = True
        source_terrain_level = "L1"
        def __post_init__(self):
            self.base_sampler_seed = self.source_parallel_sampler_seed
            self.scene = NS(terrain=NS(terrain_generator=NS(seed=10001)))
            self.observations = NS(critic=NS(enable_corruption=True))
            self.commands = NS(base_velocity=NS())
            self.terminations = NS(source_terrain_tile_bounds=NS(params={}))
    from rl_training.source_scan.training_env import (
        source_training_completion, source_training_safety, source_training_timeout)
    namespace = dict(DeeproboticsLite3SourceTerrainEnvCfg=Base, configclass=lambda cls: cls,
                     SourceRunSeeds=SourceRunSeeds, DoneTerm=lambda **kw: NS(**kw),
                     source_training_completion=source_training_completion,
                     source_training_safety=source_training_safety,
                     source_training_timeout=source_training_timeout)
    tree = ast.parse(path.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)],
                           type_ignores=[]), str(path), "exec"), namespace)
    config = namespace["DeeproboticsLite3SourceTrainingEnvCfg"]()
    config.source_training_run = run
    config.__post_init__()
    seeds = SourceRunSeeds(run)
    assert config.base_sampler_seed == seeds.sampler
    assert config.seed == seeds.ppo
    assert config.scene.terrain.terrain_generator.seed == seeds.geometry
    assert config.episode_length_s == 40. and config.is_finite_horizon is False
    assert config.terminations.source_training_timeout.time_out is True
    assert config.terminations.source_training_safety.func is source_training_safety
    assert config.terminations.source_terrain_tile_bounds.params['scanner_footprint_margin_m'] == .1
    assert not config.observations.critic.enable_corruption
    config.source_terrain_level = "L2"
    with pytest.raises(ValueError, match="L1 only"):
        config.__post_init__()


def test_missing_terminal_capture_and_malformed_timeout_fail_closed():
    env = wrapper()
    runner = make_runner(env)
    obs = env.get_observations()
    actions = runner.alg.act(obs)
    with pytest.raises(RuntimeError, match="requires"):
        runner.alg.process_env_step(obs, torch.ones(3), torch.zeros(3), {})
    with pytest.raises(ValueError, match="without a done"):
        runner.alg.process_env_step(obs, torch.ones(3), torch.zeros(3), {
            "source_terminal": None, "time_outs": torch.ones(3, dtype=torch.bool)})
    env.unwrapped.callbacks.clear()
    env.unwrapped.episode_length_buf[0] = 1999
    with pytest.raises(RuntimeError, match="capture exactly"):
        env.step(actions)


@pytest.mark.parametrize("fault", ["diagnostic_seed", "missing_endings"])
def test_adapter_rejects_scan_configuration_before_first_action(fault):
    base = FakeRslEnv()
    if fault == "diagnostic_seed":
        base.unwrapped.cfg.scene.terrain.terrain_generator.seed = 10001
    else:
        base.unwrapped.termination_manager.active_terms = []
    with pytest.raises(ValueError, match="unregistered|episode endings"):
        SourceTrainingVecEnv(base, runtime=base.runtime)
    assert base.unwrapped.common_step_counter == 0
