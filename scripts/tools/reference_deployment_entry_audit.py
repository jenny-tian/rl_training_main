"""Audit actual C++ policy entry offline before permitting any deployment rollout."""
from __future__ import annotations

import difflib
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess

import numpy as np
import onnx
import onnxruntime as ort
import torch

import reference_capability_mujoco as original
import reference_capability_mujoco_deployment as previous
from reference_capability_mujoco import ROOT, DEPLOY, MODELS, read, save, sha, actor, DEFAULT, SCALE

OUTPUT = ROOT/'logs/reference_six/deployment_entry_20261116'
DOCS = ROOT/'docs/lite3-deployment-entry-20261116'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    previous.verify_sources(read(previous.OUTPUT/'plan.json'))
    build = OUTPUT/'build'; build.mkdir()
    files = [DEPLOY/'run_policy/lite3_test_policy_runner_onnx.cpp',
        DEPLOY/'run_policy/lite3_test_policy_runner_onnx.h', DEPLOY/'run_policy/policy_runner_base.hpp',
        DEPLOY/'types/common_types.h', DEPLOY/'types/custom_types.h', DEPLOY/'utils/basic_function.hpp',
        DEPLOY/'interface/robot/simulation/simulation_interface.hpp',
        DEPLOY/'interface/robot/simulation/mujoco_simulation.py',
        DEPLOY/'state_machine/rl_control_state_onnx.hpp', DEPLOY/'state_machine/state_machine.hpp',
        DEPLOY/'state_machine/standup_state.hpp', DEPLOY/'state_machine/parameters/lite3_control_parameters.cpp',
        DEPLOY/'CMakeLists.txt', DEPLOY/'third_party/onnxruntime/x86/lib/libonnxruntime.so.1.22.0',
        Path(__file__), Path(__file__).with_name('reference_deployment_entry_probe.cpp'), DOCS/'PLAN.md']
    sources = [dict(path=str(p), sha256=sha(p)) for p in files]
    cpp = files[0].read_text(); header = files[1].read_text()
    expression = 'model_path_ = GetAbsPath() + "/../policy/ppo/policy.onnx";'
    assert cpp.count(expression) == 1
    patched_cpp = cpp.replace(expression, 'model_path_ = std::getenv("LITE3_ENTRY_AUDIT_MODEL");')
    patched_cpp = '#include <cstdlib>\n'+patched_cpp
    assert header.count('private:') == 1
    patched_header = header.replace('private:',
        'const VecXf& AuditObservation() const { return current_obs_; }\n'
        '    const VecXf& AuditLastAction() const { return last_action; }\n\nprivate:')
    # Generated instrumented copies; the upstream source is read-only.
    (build/files[0].name).write_text(patched_cpp)
    (build/files[1].name).write_text(patched_header)
    changes = ''.join(difflib.unified_diff(cpp.splitlines(True), patched_cpp.splitlines(True),
        fromfile='original/runner.cpp', tofile='probe/runner.cpp'))
    changes += ''.join(difflib.unified_diff(header.splitlines(True), patched_header.splitlines(True),
        fromfile='original/runner.h', tofile='probe/runner.h'))
    (DOCS/'instrumentation.patch').write_text(changes)
    includes = [build, DEPLOY/'run_policy', DEPLOY/'types', DEPLOY/'utils', DEPLOY/'third_party/eigen',
        DEPLOY/'third_party/onnxruntime/x86/include']
    library = DEPLOY/'third_party/onnxruntime/x86/lib'
    compile_commands = []
    for profile, flags in (('debug', ['-O0', '-g']), ('release', ['-O2', '-DNDEBUG'])):
        command = ['g++', '-std=c++17', *flags, *[f'-I{p}' for p in includes],
            str(build/files[0].name), str(Path(__file__).with_name('reference_deployment_entry_probe.cpp')),
            f'-L{library}', f'-Wl,-rpath,{library}', '-lonnxruntime', '-o', str(build/profile)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        (OUTPUT/f'compile_{profile}.log').write_text(result.stdout+result.stderr)
        result.check_returncode(); compile_commands.append(command)
    records = read(previous.OUTPUT/'complete.json')['records']+read(previous.OUTPUT/'smoke.json')['records']
    model_results = []
    # Disable core dumps from the intentionally isolated assertion diagnostic.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for name in MODELS:
        policy = actor(name)
        exported = OUTPUT/f'{name}.onnx'
        torch.onnx.export(policy, torch.zeros(1, 45), str(exported), dynamo=False,
            input_names=['obs'], output_names=['actions'], opset_version=17)
        onnx.checker.check_model(str(exported))
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
        session = ort.InferenceSession(str(exported), sess_options=opts, providers=['CPUExecutionProvider'])
        observations = np.concatenate([previous.load_npz(previous.episode_path(r, previous.OUTPUT)/'trace.npz')['observation']
            for r in records if r['policy'] == name])
        ids = np.linspace(0, len(observations)-1, 1024, dtype=int)
        batch = observations[ids]
        with torch.inference_mode():
            expected = policy(torch.from_numpy(batch)).numpy()
        actual = np.concatenate([session.run(['actions'], {'obs': x[None]})[0] for x in batch])
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
        entry_results = {}
        environment = dict(os.environ, LITE3_ENTRY_AUDIT_MODEL=str(exported))
        for profile in ('debug', 'release'):
            destination = OUTPUT/f'{name}_{profile}.csv'
            process = subprocess.run([str(build/profile), str(destination)], env=environment,
                text=True, capture_output=True, timeout=30)
            (OUTPUT/f'{name}_{profile}.log').write_text(process.stdout+process.stderr)
            entry_results[profile] = dict(returncode=process.returncode,
                first_entry_last_action_size_zero='LAST_ACTION_SIZE_AT_FIRST_ENTRY=0' in process.stdout,
                eigen_dimension_assertion='Too few coefficients passed to comma initializer' in process.stderr)
        if entry_results['release']['returncode'] != 0:
            raise RuntimeError('Release runner failed offline; preserve evidence, no physics')
        values = np.loadtxt(OUTPUT/f'{name}_release.csv', delimiter=',')
        assert values.shape == (6, 70)
        obs, actions, targets = values[:, 1:46].astype(np.float32), values[:, 46:58], values[:, 58:70]
        python_actions = np.concatenate([session.run(['actions'], {'obs': x[None]})[0] for x in obs])
        np.testing.assert_allclose(actions, python_actions, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(targets, DEFAULT+actions*SCALE, atol=3e-7, rtol=1e-6)
        np.testing.assert_allclose(obs[:, 6:9], np.tile([.4, .1, -.5], (6, 1)), atol=1e-7, rtol=0)
        np.testing.assert_allclose(obs[1:4, 33:], actions[:3], atol=2e-7, rtol=1e-7)
        reset_expected = obs[4].copy(); reset_expected[33:] = 0.
        reset_action = session.run(['actions'], {'obs': reset_expected[None]})[0][0]
        model_results.append(dict(policy=name, checkpoint_sha256=MODELS[name][1], onnx_sha256=sha(exported),
            export_samples=1024, pytorch_onnx_max_abs_difference=float(np.abs(actual-expected).max()),
            cpp_python_onnx_max_abs_difference=float(np.abs(actions-python_actions).max()),
            command_scaling_matches=True, continuous_previous_action_matches=True,
            reentry_previous_action_max_abs=float(np.abs(obs[4, 33:]).max()),
            reentry_vs_zero_history_action_max_difference=float(np.abs(actions[4]-reset_action).max()),
            reentry_vs_zero_history_joint_target_max_difference_rad=float(np.abs((actions[4]-reset_action)*SCALE).max()),
            joint_velocity_150_input_actual=float(obs[5, 21]), training_expected=5., entry=entry_results))
        print('AUDITED', name, model_results[-1], flush=True)
    receiver = files[6].read_text()
    assert 'std::memcpy(data, buffer, 46*sizeof(float));' in receiver
    # Replicate exactly the existing byte copy to demonstrate the final-field loss.
    packet = struct.pack('d45f', 1.25, *range(1, 46))
    destination = bytearray(47*4); destination[:46*4] = packet[:46*4]
    decoded = struct.unpack('d45f', destination)
    assert len(packet) == 188 and decoded[-1] == 0. and decoded[-2] == 44.
    assert all(sha(item['path']) == item['sha256'] for item in sources)
    previous.verify_sources(read(previous.OUTPUT/'plan.json'))
    result = dict(status='stopped_before_rollouts_on_interface_failure', physics_steps=0,
        startup_checks_run=0, startup_checks_limit=2, formal_episodes_run=0, formal_episode_limit=12,
        optimizer_updates=0, original_sources_unchanged=True, models=model_results,
        packet_probe=dict(payload_bytes=len(packet), copied_bytes=184,
            last_joint_torque_sent=45., last_joint_torque_decoded=float(decoded[-1]),
            actor_uses_torque=False),
        scheduling_review='state counter counts observed timestamp changes, not elapsed simulation milliseconds; actual timing unmeasured',
        sources=sources, compile_commands=compile_commands,
        versions=dict(torch=torch.__version__, onnx=onnx.__version__, python_ort=ort.__version__, cpp_ort='1.22.0'))
    save(OUTPUT/'audit.json', result); save(DOCS/'audit.json', result)
    for path in OUTPUT.glob('*.log'):
        shutil.copy2(path, DOCS/path.name)
    for path in OUTPUT.glob('*.csv'):
        shutil.copy2(path, DOCS/path.name)
    save(DOCS/'raw_manifest.json', [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(OUTPUT.rglob('*')) if p.is_file()])
    print('OFFLINE AUDIT COMPLETE; interface gate failed, zero startup/formal episodes', flush=True)


if __name__ == '__main__':
    main()
