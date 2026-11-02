import hashlib
import json
from pathlib import Path

import pytest
import torch
from rsl_rl.algorithms import PPO

from scripts.tools.source_training_numerics_check import make_runner
from rl_training.source_scan.source_training import TransitionBudget
from rl_training.source_scan.parallel_terrain_allocation import SOURCE_RESET_BLOCK_COUNTS
from rl_training.source_scan.training_runner import (
    SourceTrainingRunner, TRAINING_NODES, _state_digest, validate_training_authorization)


def training(tmp_path, *, updates=1, length=4):
    runner = make_runner(compact=True, rollout_length=length)
    return SourceTrainingRunner(runner, run=1, budget=TransitionBudget(8, length, 8 * length * updates),
        output_dir=tmp_path / 'run', checkpoint_sha256='a' * 64, checkpoint_nodes=(1,))


def collect(runner):
    observations = runner.env.get_observations()
    with torch.inference_mode():
        for _ in range(runner.cfg['num_steps_per_env']):
            actions = runner.alg.act(observations)
            observations, rewards, dones, extras = runner.env.step(actions)
            runner.alg.process_env_step(observations, rewards, dones, extras)
        runner.alg.compute_returns(observations)


def report(run):
    return json.loads((run.output_dir / 'run_report.json').read_text())


def test_complete_updates_stop_at_budget_and_checkpoint_matches_live_actor(tmp_path):
    run = training(tmp_path, updates=2)
    initial = _state_digest(run.algorithm.actor)
    result = run.run()
    assert result['status'] == 'completed'
    assert result['completed_updates'] == 2
    assert result['attempted_transition_upper_bound'] == result['stored_transitions'] == 64
    assert result['synthetic_transitions'] == 64 and result['real_source_training_transitions'] == 0
    assert result['real_source_ppo_updates'] == 0
    assert result['source_backward_checks'] == result['source_optimizer_steps_completed'] == 2
    assert initial != _state_digest(run.algorithm.actor)
    assert sum(v['transitions'] for v in result['exposure'].values()) == 64
    assert len((run.output_dir / 'update_receipts.jsonl').read_text().splitlines()) == 2
    saved = torch.load(run.output_dir / 'actor_update_1.pt', weights_only=True)
    assert set(saved) == set(run.algorithm.actor.state_dict())
    checkpoint_actor = make_runner(compact=True, rollout_length=4).alg.actor
    checkpoint_actor.load_state_dict(saved)
    assert _state_digest(checkpoint_actor) == result['update_receipts'][0]['actor_sha256']
    assert not (run.output_dir / 'actor_update_2.pt').exists()


def test_default_checkpoint_nodes_are_bounded_by_numerics_budget(tmp_path):
    runner = make_runner(compact=True, num_envs=8, rollout_length=4)
    run = SourceTrainingRunner(runner, run=1, budget=TransitionBudget(8, 4, 32 * 91),
                               output_dir=tmp_path / 'run', checkpoint_sha256='a' * 64)
    assert run.checkpoint_nodes == (91,)
    for node, (updates, ceiling) in TRAINING_NODES.items():
        budget = TransitionBudget(120, 24, ceiling)
        assert budget.max_updates == updates
        assert budget.committed_transitions in (262080, 1048320, 2096640)


@pytest.mark.parametrize('fault', ['interrupt', 'reward_nan', 'state_nan', 'initial_width'])
def test_partial_collection_is_durable_and_never_counted_as_completed_update(tmp_path, fault):
    run = training(tmp_path)
    original = run.env.step
    calls = 0
    def step(actions):
        nonlocal calls
        calls += 1
        if fault == 'interrupt' and calls == 2:
            raise KeyboardInterrupt()
        obs, rewards, dones, extras = original(actions)
        if fault == 'reward_nan':
            rewards[0] = float('nan')
        if fault == 'state_nan':
            obs['critic'][0, 100] = float('nan')
        return obs, rewards, dones, extras
    run.env.step = step
    if fault == 'initial_width':
        get_obs = run.env.get_observations
        def malformed():
            observations = get_obs()
            observations['critic'] = observations['critic'][:, :-1]
            return observations
        run.env.get_observations = malformed
    with pytest.raises((KeyboardInterrupt, FloatingPointError, ValueError)):
        run.run()
    result = report(run)
    assert result['completed_updates'] == 0 and result['source_optimizer_steps_completed'] == 0
    if fault == 'interrupt':
        assert result['status'] == 'interrupted'
        assert result['attempted_transition_upper_bound'] == 16
        assert result['returned_transitions'] == result['stored_transitions'] == 8
    elif fault == 'initial_width':
        assert result['attempted_transition_upper_bound'] == result['stored_transitions'] == 0
    else:
        assert result['returned_transitions'] == 8 and result['stored_transitions'] == 0
    assert result['automatic_resume'] is False
    with pytest.raises(FileExistsError):
        SourceTrainingRunner(run.runner, run=1, budget=run.budget,
            output_dir=run.output_dir, checkpoint_sha256='a' * 64)


def test_overflowed_finite_returns_stop_before_backward_and_optimizer_mutation():
    runner = make_runner(compact=True, rollout_length=4)
    collect(runner)
    before = _state_digest(runner.alg.actor), _state_digest(runner.alg.critic)
    runner.alg.storage.returns.fill_(1e30)
    with pytest.raises(FloatingPointError, match='loss before backward'):
        runner.alg.update()
    assert runner.alg.source_backward_checks == runner.alg.source_optimizer_steps_started == 0
    assert not runner.alg.optimizer.state
    assert before == (_state_digest(runner.alg.actor), _state_digest(runner.alg.critic))


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_invalid_gradients_stop_before_first_optimizer_step(value):
    runner = make_runner(compact=True, rollout_length=4)
    collect(runner)
    before = _state_digest(runner.alg.actor)
    next(runner.alg.actor.parameters()).register_hook(lambda grad: torch.full_like(grad, value))
    with pytest.raises(FloatingPointError, match='gradient'):
        runner.alg.update()
    assert runner.alg.source_backward_checks == 1
    assert runner.alg.source_optimizer_steps_started == 0 and not runner.alg.optimizer.state
    assert before == _state_digest(runner.alg.actor)


def test_partial_minibatch_update_is_reported_when_later_hook_fails(tmp_path):
    run = training(tmp_path)
    def fault(optimizer, args, kwargs):
        raise RuntimeError('injected after one Adam step')
    run.algorithm.optimizer.register_step_post_hook(fault)
    before = _state_digest(run.algorithm.actor)
    with pytest.raises(RuntimeError, match='one Adam step'):
        run.run()
    result = report(run)
    assert result['completed_updates'] == 0
    assert result['source_optimizer_steps_started'] == result['source_optimizer_steps_completed'] == 1
    assert _state_digest(run.algorithm.actor) != before


@pytest.mark.parametrize('fault', ['parameter', 'optimizer'])
def test_nonfinite_model_or_optimizer_state_rejected(fault):
    runner = make_runner(compact=True, rollout_length=4)
    collect(runner)
    if fault == 'parameter':
        with torch.no_grad():
            next(runner.alg.critic.parameters()).flatten()[0] = float('nan')
    else:
        runner.alg.update()
        collect(runner)
        next(iter(runner.alg.optimizer.state.values()))['exp_avg'].flatten()[0] = float('inf')
    count = runner.alg.source_optimizer_steps_completed
    with pytest.raises(FloatingPointError):
        runner.alg.update()
    assert runner.alg.source_optimizer_steps_completed == count


def test_guarded_update_has_exact_parity_with_installed_ppo():
    guarded = make_runner(compact=True, rollout_length=4)
    collect(guarded)
    reference = make_runner(compact=True, rollout_length=4)
    collect(reference)
    torch.manual_seed(123)
    actual = guarded.alg.update()
    torch.manual_seed(123)
    expected = PPO.update(reference.alg)
    assert actual == expected
    assert _state_digest(guarded.alg.actor) == _state_digest(reference.alg.actor)
    assert _state_digest(guarded.alg.critic) == _state_digest(reference.alg.critic)


def authorization(tmp_path):
    def receipt(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    inputs = {'fixture': 'b' * 64}
    runtime = receipt('runtime.json', {'status': 'parallel_runtime_qualified', 'num_envs': 120,
        'device': 'cuda:0', 'g3_passed': True, 'g4_passed': True, 'input_sha256': inputs})
    selection = receipt('selection.json', {'status': 'locked', 'scope': 'source_only',
        'registered_cells': list(SOURCE_RESET_BLOCK_COUNTS),
        'development_or_sealed_feedback_allowed': False, 'quality_gates': {'fixture': True},
        'selection_rule': 'fixture', 'seed_splits': {'fixture': True}, 'evaluation_nodes': [91, 364, 728]})
    dependencies = {}
    for i in range(35):
        path = tmp_path / f'dependency_{i}'
        path.write_text(str(i))
        dependencies[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(schema='lite3_source_ppo_authorization_v1', user_approved=True, status='approved',
        action='source_ppo_training', run=1, ppo_seed=130001, geometry_seed=30000, sampler_seed=131001,
        reset_state_id=40000, num_envs=120, rollout_length=24, level='L1', registered_cells=15,
        six_families_retained=True, budget_node='numerics', max_updates=91, max_transitions=262144,
        device='cuda:0', minimum_free_vram_mib=12288, retry=False, resume=False,
        automatic_scale_increase=False, development_or_sealed_feedback_allowed=False,
        base_revision='c' * 40, remote_revision_verified='c' * 40, checkpoint_sha256='a' * 64,
        input_sha256=inputs, output_dir=str((tmp_path / 'new_run').resolve()),
        runtime_qualification=runtime, source_selection_lock=selection, dependency_sha256=dependencies)
    kwargs = dict(run=1, budget_node='numerics', output_dir=tmp_path / 'new_run',
                  checkpoint_sha256='a' * 64, revision='c' * 40, input_sha256=inputs)
    return manifest, kwargs


@pytest.mark.parametrize('fault', [None, 'approval', 'scale', 'budget', 'missing_runtime',
    'changed_receipt', 'dependency', 'missing_selection', 'continuation'])
def test_runtime_authorization_refuses_unready_or_changed_contract(tmp_path, fault):
    manifest, kwargs = authorization(tmp_path)
    if fault == 'approval': manifest['user_approved'] = False
    if fault == 'scale': manifest['num_envs'] = 8
    if fault == 'budget': manifest['max_transitions'] += 2880
    if fault == 'missing_runtime': del manifest['runtime_qualification']
    if fault == 'missing_selection': del manifest['source_selection_lock']
    if fault == 'changed_receipt': Path(manifest['runtime_qualification']['path']).write_text('{}')
    if fault == 'dependency': Path(next(iter(manifest['dependency_sha256']))).write_text('changed')
    if fault == 'continuation': kwargs['budget_node'] = 'midpoint'
    if fault is None:
        assert validate_training_authorization(manifest, **kwargs)['max_updates'] == 91
    else:
        with pytest.raises((ValueError, PermissionError, NotImplementedError)):
            validate_training_authorization(manifest, **kwargs)


def test_entrypoint_validation_refuses_approval_without_isaac_or_cuda_start(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    from scripts.tools import source_training_entrypoint as entrypoint
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.write_bytes(b'fixture')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(entrypoint, 'CHECKPOINT_SHA256', digest)
    monkeypatch.setattr(entrypoint, '_git_state', lambda: ('a' * 40, 'a' * 40, ''))
    monkeypatch.setattr(entrypoint, '_input_hashes', lambda: {'fixture': 'b' * 64})
    args = NS(run=1, budget_node='numerics', output_dir=tmp_path / 'output',
              checkpoint=checkpoint, device='cuda:0')
    before_modules = set(sys.modules)
    initialized = torch.cuda.is_initialized()
    with pytest.raises(PermissionError):
        entrypoint.validate_before_simulation(args, {'schema': 'lite3_source_ppo_authorization_v1',
                                                     'user_approved': False})
    assert not any(name.startswith(('isaaclab', 'isaacsim', 'omni.')) for name in set(sys.modules) - before_modules)
    assert torch.cuda.is_initialized() == initialized
    assert not args.output_dir.exists()
