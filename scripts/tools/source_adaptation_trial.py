"""One bounded, separately charged two-update trial of the corrected Source profile."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source/rl_training'), str(ROOT/'scripts/tools')]
from source_adaptation_runtime import (
    OLD, HASHES, digest, write_json, make_environment, environment_identity, collect_case,
)
from rl_training.source_scan.training_budget_ledger import BudgetLedger, canonical_budget_path

BASELINE = ROOT/'docs/lite3-source-reward-live-verification-20261105'
EXPECTED_BUDGET_SHA256 = 'f171d9fdbe380a20c827453c52bb30ac30c8f6ed431f45c01f796625be303435'
PLAN = dict(profile='lite3_source_lowspeed_v1', num_envs=120, rollout_length=24,
    max_updates=2, reserved_transitions=5760, baseline_charged_transitions=259584,
    cumulative_ceiling=265344, initialization='registered_actor_checkpoint_std_fresh_critic_optimizer')


def evaluation_checks(case, thresholds):
    value = case['after_100']
    return dict(speed=value['vx_mps'] >= thresholds['min_vx_mps'],
        velocity_error=value['velocity_rms_error_mps'] <= thresholds['max_velocity_rms_error_mps'],
        clipping=value['clipped_joint_fraction'] <= thresholds['max_clipped_joint_fraction'],
        endings=case['done_count'] <= thresholds['max_done_count'],
        reward=value['reward_per_control'] >= thresholds['min_reward_per_control'])


class TrialObserver:
    """Observe existing transitions and fixed probes without drawing extra random actions."""
    def __init__(self, folder):
        self.folder = folder
        self.rows = []

    def initial(self, training, observations):
        import torch
        self.probes = observations.clone()
        with torch.no_grad():
            self.initial_mean = training.algorithm.actor(self.probes).detach().clone()

    def before_step(self, training, step):
        if (step-1) % 24 == 0:
            self.reward = self.clipping = self.vx = 0.0
            self.endings = self.steps = 0

    def after_step(self, training, step, old, observations, actions, rewards, dones, extras):
        from source_delivery_light import check_scans
        env = training.env
        trace = env.last_trace
        self.reward += float(rewards.mean())
        self.clipping += float((trace.raw_policy_action != trace.base_action).float().mean())
        self.vx += float(env.unwrapped.scene['robot'].data.root_lin_vel_b[:, 0].mean())
        self.endings += int(dones.sum())
        self.steps += 1
        check_scans(env.unwrapped)

    def before_update(self, training, update):
        storage = training.algorithm.storage
        self.rollout = dict(value_mean=float(storage.values.mean()), return_mean=float(storage.returns.mean()),
            value_mse=float((storage.values-storage.returns).square().mean()),
            advantage_mean=float(storage.advantages.mean()), advantage_std=float(storage.advantages.std()))

    def after_update(self, training, update, receipt):
        import torch
        from rl_training.source_scan.delivery_metrics import append_json
        actor = training.algorithm.actor
        with torch.no_grad():
            shift = actor(self.probes) - self.initial_mean
            std = actor.state_dict()['distribution.log_std_param'].exp()
        row = dict(update=update, training_transitions=update*2880,
            mean_step_reward=self.reward/self.steps, sampled_action_clipped_fraction=self.clipping/self.steps,
            post_step_vx_mps=self.vx/self.steps, episode_endings=self.endings,
            learning_rate=float(training.algorithm.optimizer.param_groups[0]['lr']),
            action_std=std.cpu().tolist(), mean_action_std=float(std.mean()),
            initial_probe_mean_shift_rms=float(shift.square().mean().sqrt()),
            initial_probe_mean_shift_max=float(shift.abs().max()), rollout=self.rollout,
            losses=receipt['losses'], elapsed_s=receipt['elapsed_seconds'])
        self.rows.append(row)
        append_json(self.folder/'updates.jsonl', row)
        print(f'[adaptation-trial] update={update}/2 transitions={update*2880} '
              f'reward={row["mean_step_reward"]:.6f} std={row["mean_action_std"]:.6f}', flush=True)


def child(folder):
    from source_delivery_light import record_phase, close_comparison_environment
    manifest = json.loads((folder/'manifest.json').read_text())
    report = dict(status='starting', diagnostic_transitions=0, cases=[])
    app = wrapped = None
    stack_file = (folder/'python_stacks.log').open('w')
    faulthandler.enable(file=stack_file)

    def phase(name, limit=120):
        faulthandler.cancel_dump_traceback_later()
        record_phase(folder, name, None, limit)
        if limit:
            faulthandler.dump_traceback_later(max(1, limit-5), file=stack_file)

    try:
        if os.getppid() != manifest['supervisor_pid'] or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('trial child must be launched by its GPU0 supervisor')
        for path, sha in manifest['input_sha256'].items():
            if digest(path) != sha:
                raise ValueError('trial dependency changed: '+path)
        reservation = json.loads((folder/'reservation.json').read_text())
        rows = BudgetLedger(canonical_budget_path(ROOT)).read()['entries']
        if reservation not in rows or reservation['state'] != 'reserved':
            raise ValueError('trial reservation is not active')
        if any(reservation.get(k) != v for k, v in PLAN.items()) or reservation['output_dir'] != str(folder.resolve()):
            raise ValueError('trial reservation differs from the bounded plan')
        baseline = json.loads((BASELINE/'raw/runtime/result.json').read_text())['cases'][0]
        phase('application_start')
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device='cuda:0').app
        import numpy as np
        import random
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from rl_training.source_scan.adaptation_profile import initialize_source_adaptation_actor
        from rl_training.source_scan.frozen_actor import FrozenSourceActor
        from rl_training.source_scan.source_training import TransitionBudget
        from rl_training.source_scan.training_runner import SourceTrainingRunner, _state_digest, _atomic_torch_save
        from source_training_preflight import runner_config
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        phase('training_environment_start')
        wrapped = make_environment()
        report['training_initial'] = environment_identity(wrapped.unwrapped)
        if report['training_initial'] != baseline['initial']:
            raise ValueError('training initial state differs from the fixed baseline')
        torch.manual_seed(130001)
        np.random.seed(130001)
        random.seed(130001)
        config = runner_config()
        config.update(max_iterations=2, save_interval=3, check_for_nan=True)
        config['algorithm']['class_name'] = 'rl_training.source_scan.training_ppo:SourcePPO'
        runner = OnPolicyRunner(wrapped, config, log_dir=None, device='cuda:0')
        critic_before = _state_digest(runner.alg.critic)
        report['actor_initialization'] = initialize_source_adaptation_actor(runner.alg.actor, OLD, wrapped.unwrapped.cfg)
        original = torch.load(OLD, map_location='cpu', weights_only=True)['actor_state_dict']
        report['actor_tensors_equal_checkpoint'] = all(
            torch.equal(v.cpu(), original[k]) for k, v in runner.alg.actor.state_dict().items())
        report['critic_unchanged_by_loading'] = critic_before == _state_digest(runner.alg.critic)
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            obs = wrapped.get_observations()
            difference = runner.alg.actor(obs) - FrozenSourceActor(original).to('cuda:0')(obs)
            report['initial_mean_max_abs_error'] = float(difference.abs().max())
        if not report['actor_tensors_equal_checkpoint'] or not report['critic_unchanged_by_loading'] or report['initial_mean_max_abs_error'] != 0:
            raise ValueError('live actor initialization failed parity')
        report['runner_config'] = config
        write_json(folder/'result.json', report)
        training = SourceTrainingRunner(runner, run=1, budget=TransitionBudget(120,24,5760),
            output_dir=folder/'training', checkpoint_sha256=HASHES[0], checkpoint_nodes=(1,2))
        training.report.update(profile=PLAN['profile'], bounded_adaptation_trial=True,
            manifest_sha256=digest(folder/'manifest.json'), actor_initialization=report['actor_initialization'])
        phase('two_ppo_updates')
        observer = TrialObserver(folder)
        def before_step():
            if not app.is_running():
                raise RuntimeError('simulator stopped')
        report['training'] = training.run(observer=observer, before_step=before_step)
        if report['training']['stored_transitions'] != 5760 or report['training']['completed_updates'] != 2:
            raise RuntimeError('trial did not complete exactly two updates')
        saved = runner.alg.save()
        saved.update(iter=runner.current_learning_iteration, infos=dict(profile=PLAN['profile'], bounded_trial=True))
        _atomic_torch_save(folder/'training/model_update_2.pt', saved)
        report['updates'] = observer.rows
        report['checkpoint_sha256'] = {p.name:digest(p) for p in sorted((folder/'training').glob('*.pt'))}
        write_json(folder/'result.json', report)
        phase('environment_close', 15)
        close_comparison_environment(wrapped)
        wrapped = None
        del training, runner, saved, observer
        phase('stage_replace')
        from isaaclab.sim import create_new_stage
        create_new_stage()
        phase('evaluation_environment_start')
        wrapped = make_environment()
        if environment_identity(wrapped.unwrapped) != baseline['initial']:
            raise ValueError('evaluation geometry or initial state differs from baseline')
        candidate_path = folder/'training/actor_update_2.pt'
        actor = FrozenSourceActor(torch.load(candidate_path, map_location='cpu', weights_only=True)).to('cuda:0')
        phase('deterministic_evaluation')
        case = collect_case(wrapped, actor, 'actor_update_2', folder, app, report)
        report['cases'].append(case)
        report['evaluation_checks'] = evaluation_checks(case, manifest['evaluation_thresholds'])
        report['short_trial_behavior_passed'] = all(report['evaluation_checks'].values())
        if digest(OLD) != HASHES[0] or digest(candidate_path) != report['checkpoint_sha256'][candidate_path.name]:
            raise ValueError('checkpoint changed during evaluation')
        report['status'] = 'passed' if report['short_trial_behavior_passed'] else 'behavior_regressed'
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(folder/'result.json', report)
        if wrapped is not None:
            phase('environment_close', 15)
            close_comparison_environment(wrapped)
        if app is not None:
            phase('application_close', 15)
            app.close()
        phase('finished', None)
        faulthandler.disable()
        stack_file.close()
    return 0 if report['status'] in ('passed','behavior_regressed') else 1


def supervise(folder):
    from source_delivery_light import lifecycle_deadline
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('trial is restricted to GPU0')
    ledger = BudgetLedger(canonical_budget_path(ROOT))
    reference = ledger.reference()
    if reference['sha256'] != EXPECTED_BUDGET_SHA256:
        raise ValueError('historical budget changed; this one-off trial cannot be retried')
    if digest(OLD) != HASHES[0]:
        raise ValueError('original checkpoint changed')
    baseline = json.loads((BASELINE/'deliverables/summary.json').read_text())
    if baseline['eligible_for_two_update_trial'] is not True:
        raise ValueError('reward profile has not passed live verification')
    if subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('commit trial changes before launch')
    revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    upstream = subprocess.check_output(['git','rev-parse','@{upstream}'], cwd=ROOT, text=True).strip()
    if revision != upstream:
        raise ValueError('push trial revision before launch')
    inputs = [OLD, Path(__file__), ROOT/'scripts/tools/source_adaptation_runtime.py',
        ROOT/'scripts/tools/source_delivery_light.py', ROOT/'scripts/tools/source_training_preflight.py',
        BASELINE/'deliverables/summary.json', BASELINE/'raw/runtime/result.json']
    inputs += list((ROOT/'source/rl_training/rl_training/source_scan').glob('*.py'))
    inputs += list((ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3').glob('*.py'))
    inputs += list((ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp').glob('*.py'))
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'budget_before.json').write_bytes(ledger.path.read_bytes())
    manifest = dict(schema='source_adaptation_two_update_trial_v1', source_revision=revision,
        authorization_basis='user agreed to the two-update plan and incremental 5760-transition budget',
        plan=PLAN, budget_before=reference, original_checkpoint_sha256=HASHES[0],
        input_sha256={str(p):digest(p) for p in sorted(set(inputs))},
        evaluation_thresholds=baseline['two_update_evaluation_thresholds'],
        diagnostic_transition_cap=48000, wall_limit_s=300, startup_limit_s=120, close_limit_s=15,
        retry=False, resume=False, automatic_extension=False, python=sys.executable, supervisor_pid=os.getpid())
    write_json(folder/'manifest.json', manifest)
    reservation = ledger.reserve_adaptation_trial(run=1, attempt_id='source-reward-correction-two-update-20261105-v1',
        output_dir=folder, revision=revision, expected_reference=reference, expected_plan=PLAN)
    write_json(folder/'reservation.json', reservation)
    started = time.monotonic()
    reason = None
    code = 1
    remaining = False
    proc = None
    try:
        with (folder/'runtime.log').open('w') as log:
            proc = subprocess.Popen([sys.executable,'-u',__file__,'--child','--output',str(folder)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                now = time.monotonic()
                reason = '300s wall deadline exceeded' if now-started > 300 else lifecycle_deadline(folder,now)
                if reason:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                    break
                time.sleep(1)
            code = proc.wait()
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid,0)
                remaining = True
                os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        counts_path = folder/'training/run_report.json'
        counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
        result_path = folder/'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        accepted = code == 0 and reason is None and not remaining and result.get('status') == 'passed'
        finalized = ledger.finish(reservation['attempt_id'], accepted=accepted,
            attempted=counts.get('attempted_transition_upper_bound',0), returned=counts.get('returned_transitions',0),
            stored=counts.get('stored_transitions',0), updates=counts.get('completed_updates',0))
        write_json(folder/'budget_receipt.json', finalized)
        (folder/'budget_after.json').write_bytes(ledger.path.read_bytes())
        receipt = dict(exit_code=code, deadline_failure=reason, elapsed_s=time.monotonic()-started,
            process_group_had_remaining_children=remaining, budget_after=ledger.reference(),
            cumulative_charged_transitions=ledger.charged(1,ledger.read()), accepted_short_trial=accepted,
            training_transitions=counts.get('stored_transitions',0), ppo_updates=counts.get('completed_updates',0),
            diagnostic_transitions=result.get('diagnostic_transitions',0))
        write_json(folder/'execution.json',receipt)
        print(json.dumps(receipt),flush=True)
    return code if code else int(reason is not None or remaining)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    args = parser.parse_args()
    raise SystemExit(child(args.output) if args.child else supervise(args.output))
