import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_training.online_adaptation import provenance as provenance_module
from rl_training.online_adaptation.provenance import (
    build_flat_baseline_manifest,
    sha256_file,
    snapshot_file,
    write_json_atomic,
)
from scripts.reinforcement_learning.online_adaptation import build_flat_baseline_manifest as manifest_script


STRICT_FLAT_TASK = "Flat-Deeprobotics-Lite3-v0"
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
VERSIONS = {
    "python": "3.11.8",
    "isaaclab": "0.54.3",
    "rsl-rl-lib": "5.0.1",
    "torch": "2.5.1",
}


def _write_required_configs(tmp_path: Path) -> tuple[Path, Path]:
    env_config = tmp_path / "env.yaml"
    env_config.write_text(
        "seed: 7\n"
        "scene:\n"
        "  num_envs: 4096\n"
        "  terrain:\n"
        "    terrain_type: plane\n"
        "    terrain_generator: null\n",
        encoding="utf-8",
    )
    agent_config = tmp_path / "agent.yaml"
    agent_config.write_text("seed: 7\nmax_iterations: 1500\n", encoding="utf-8")
    return env_config, agent_config


def _build_manifest(
    checkpoint: Path,
    configs: list[Path],
    **overrides,
) -> dict[str, object]:
    arguments = {
        "checkpoint": checkpoint,
        "config_paths": configs,
        "training_command": [
            "python",
            "train.py",
            "--task",
            STRICT_FLAT_TASK,
            "--seed",
            "7",
            "--num-envs",
            "4096",
            "--max_iterations",
            "1500",
        ],
        "git_sha": GIT_SHA,
        "task": STRICT_FLAT_TASK,
        "seed": 7,
        "num_envs": 4096,
        "max_iterations": 1500,
        "versions": VERSIONS,
    }
    arguments.update(overrides)
    return build_flat_baseline_manifest(**arguments)


def test_build_flat_baseline_manifest_records_reproducible_provenance(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"strict-flat-policy\x00\x01")
    env_config, agent_config = _write_required_configs(tmp_path)
    extra_config = tmp_path / "runner.yaml"
    extra_config.write_text("experiment_name: strict_flat\n", encoding="utf-8")

    manifest = _build_manifest(checkpoint, [env_config, agent_config, extra_config])

    checkpoint_key = str(checkpoint.resolve())
    assert manifest["schema_version"] == 1
    assert manifest["terrain_assertion"] == "plane-only"
    assert manifest["task"] == STRICT_FLAT_TASK
    assert manifest["git_sha"] == GIT_SHA
    assert manifest["training"] == {
        "seed": 7,
        "num_envs": 4096,
        "max_iterations": 1500,
        "command": [
            "python",
            "train.py",
            "--task",
            STRICT_FLAT_TASK,
            "--seed",
            "7",
            "--num-envs",
            "4096",
            "--max_iterations",
            "1500",
        ],
    }
    assert manifest["versions"] == dict(sorted(VERSIONS.items()))
    assert list(manifest["files"]) == sorted(manifest["files"])
    assert set(manifest["files"]) == {
        str(checkpoint.resolve()),
        str(env_config.resolve()),
        str(agent_config.resolve()),
        str(extra_config.resolve()),
    }
    assert manifest["files"][checkpoint_key] == {
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "bytes": checkpoint.stat().st_size,
    }


def test_build_manifest_accepts_real_isaac_yaml_tags_without_constructing_objects(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    env_config, agent_config = _write_required_configs(tmp_path)
    env_config.write_text(
        "seed: 7\n"
        "scene:\n"
        "  num_envs: 4096\n"
        "  terrain:\n"
        "    terrain_type: plane\n"
        "    terrain_generator: null\n"
        "sim:\n"
        "  gravity: !!python/tuple\n"
        "    - 0.0\n"
        "    - 0.0\n"
        "    - -9.81\n",
        encoding="utf-8",
    )

    manifest = _build_manifest(checkpoint, [env_config, agent_config])

    assert manifest["terrain_assertion"] == "plane-only"
    assert str(env_config.resolve()) in manifest["files"]


def test_build_flat_baseline_manifest_rejects_non_flat_task(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    configs = list(_write_required_configs(tmp_path))

    with pytest.raises(ValueError, match="Flat-Deeprobotics-Lite3-v0"):
        build_flat_baseline_manifest(
            checkpoint=checkpoint,
            config_paths=configs,
            training_command=["python", "train.py"],
            git_sha=GIT_SHA,
            task="OnlineAdapt-Deeprobotics-Lite3-v0",
            seed=7,
            num_envs=64,
            max_iterations=1,
            versions=VERSIONS,
        )


@pytest.mark.parametrize(
    "terrain_yaml",
    [
        "scene:\n  terrain:\n    terrain_type: generator\n    terrain_generator: {}\n",
        "scene:\n  terrain:\n    terrain_type: plane\n    terrain_generator:\n      class_type: rough\n",
        'scene:\n  terrain:\n    terrain_type: plane\n    terrain_generator: "null"\n',
        'scene:\n  terrain:\n    terrain_type: plane\n    terrain_generator: "None"\n',
        "scene:\n  terrain:\n    terrain_type: plane\n    terrain_generator: {}\n",
        "scene:\n  terrain:\n    terrain_type: plane\n",
    ],
)
def test_build_manifest_rejects_non_plane_or_generated_terrain(tmp_path, terrain_yaml):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    env_config, agent_config = _write_required_configs(tmp_path)
    env_config.write_text(terrain_yaml, encoding="utf-8")

    with pytest.raises(ValueError, match="strict plane-only provenance"):
        _build_manifest(checkpoint, [env_config, agent_config])


@pytest.mark.parametrize("missing_name", ["env.yaml", "agent.yaml", "both"])
def test_build_manifest_requires_env_and_agent_config_pair(tmp_path, missing_name):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    env_config, agent_config = _write_required_configs(tmp_path)
    configs = (
        []
        if missing_name == "both"
        else [path for path in (env_config, agent_config) if path.name != missing_name]
    )

    with pytest.raises(ValueError, match="strict plane-only provenance"):
        _build_manifest(checkpoint, configs)


@pytest.mark.parametrize("duplicate_name", ["env.yaml", "agent.yaml"])
def test_build_manifest_rejects_duplicate_required_config_names(tmp_path, duplicate_name):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    env_config, agent_config = _write_required_configs(tmp_path)
    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    duplicate = duplicate_dir / duplicate_name
    duplicate.write_text(
        env_config.read_text(encoding="utf-8") if duplicate_name == "env.yaml" else "seed: 11\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict plane-only provenance"):
        _build_manifest(checkpoint, [env_config, agent_config, duplicate])


@pytest.mark.parametrize(
    "env_yaml",
    [
        "scene: [\n",
        "- scene\n- terrain\n",
        "scene: []\n",
        "scene:\n  terrain: []\n",
        "scene: {}\n",
    ],
)
def test_build_manifest_rejects_malformed_or_non_mapping_env_structure(tmp_path, env_yaml):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    env_config, agent_config = _write_required_configs(tmp_path)
    env_config.write_text(env_yaml, encoding="utf-8")

    with pytest.raises(ValueError, match="strict plane-only provenance"):
        _build_manifest(checkpoint, [env_config, agent_config])


def test_build_manifest_rejects_missing_required_version(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    configs = list(_write_required_configs(tmp_path))

    with pytest.raises(ValueError, match="required version.*torch"):
        build_flat_baseline_manifest(
            checkpoint=checkpoint,
            config_paths=configs,
            training_command=["python", "train.py"],
            git_sha=GIT_SHA,
            task=STRICT_FLAT_TASK,
            seed=7,
            num_envs=64,
            max_iterations=1,
            versions={key: value for key, value in VERSIONS.items() if key != "torch"},
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"num_envs": 0}, "num_envs"),
        ({"num_envs": 1.5}, "num_envs"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": False}, "max_iterations"),
        ({"git_sha": "abc123"}, "git_sha"),
        ({"git_sha": "g" * 40}, "git_sha"),
        ({"training_command": []}, "training_command"),
        ({"training_command": ("python", "train.py")}, "training_command"),
        ({"training_command": ["python", ""]}, "training_command"),
        ({"training_command": ["python", " "]}, "training_command"),
    ],
)
def test_build_manifest_rejects_invalid_semantic_values(tmp_path, overrides, message):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    configs = list(_write_required_configs(tmp_path))

    with pytest.raises(ValueError, match=message):
        _build_manifest(checkpoint, configs, **overrides)


@pytest.mark.parametrize("invalid_version", ["", "  ", "unavailable", None, 5])
def test_build_manifest_rejects_non_concrete_required_versions(tmp_path, invalid_version):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    configs = list(_write_required_configs(tmp_path))
    versions = dict(VERSIONS)
    versions["torch"] = invalid_version

    with pytest.raises(ValueError, match="required version.*torch"):
        _build_manifest(checkpoint, configs, versions=versions)


def test_build_manifest_normalizes_full_git_sha_to_lowercase(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    configs = list(_write_required_configs(tmp_path))

    manifest = _build_manifest(checkpoint, configs, git_sha=GIT_SHA.upper())

    assert manifest["git_sha"] == GIT_SHA


def test_build_manifest_rejects_duplicate_resolved_input_paths(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    env_config, agent_config = _write_required_configs(tmp_path)
    extra = tmp_path / "extra.yaml"
    extra.write_text("value: extra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate resolved input"):
        _build_manifest(checkpoint, [env_config, agent_config, extra, extra])


@pytest.mark.parametrize(
    ("config_name", "old", "new"),
    [
        ("env.yaml", "seed: 7", "seed: 8"),
        ("env.yaml", "num_envs: 4096", "num_envs: 2048"),
        ("agent.yaml", "seed: 7", "seed: 8"),
        ("agent.yaml", "max_iterations: 1500", "max_iterations: 1499"),
    ],
)
def test_build_manifest_rejects_yaml_argument_mismatches(tmp_path, config_name, old, new):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    configs = list(_write_required_configs(tmp_path))
    config = next(path for path in configs if path.name == config_name)
    config.write_text(config.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        _build_manifest(checkpoint, configs)


@pytest.mark.parametrize(
    "command",
    [
        ["python", "train.py", "--seed", "7", "--num-envs", "4096", "--max_iterations", "1500"],
        [
            "python",
            "train.py",
            "--task",
            "OnlineAdapt-Deeprobotics-Lite3-v0",
            "--seed",
            "7",
            "--num-envs",
            "4096",
            "--max_iterations",
            "1500",
        ],
        [
            "python",
            "train.py",
            "--task",
            STRICT_FLAT_TASK,
            "--seed",
            "7",
            "--seed",
            "7",
            "--num-envs",
            "4096",
            "--max_iterations",
            "1500",
        ],
        [
            "python",
            "train.py",
            "--task",
            STRICT_FLAT_TASK,
            "--seed",
            "8",
            "--num-envs",
            "4096",
            "--max_iterations",
            "1500",
        ],
        [
            "python",
            "train.py",
            "--task",
            STRICT_FLAT_TASK,
            "--seed",
            "7",
            "--num-envs",
            "2048",
            "--max_iterations",
            "1500",
        ],
        [
            "python",
            "train.py",
            "--task",
            STRICT_FLAT_TASK,
            "--seed",
            "7",
            "--num-envs",
            "4096",
            "--max_iterations",
            "1499",
        ],
    ],
)
def test_build_manifest_rejects_missing_duplicate_or_conflicting_command_flags(tmp_path, command):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    configs = list(_write_required_configs(tmp_path))

    with pytest.raises(ValueError, match="training command"):
        _build_manifest(checkpoint, configs, training_command=command)


def test_build_manifest_rejects_path_replacement_during_snapshot(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    env_config, agent_config = _write_required_configs(tmp_path)
    replacement = tmp_path / "rough-env.yaml"
    replacement.write_text(
        env_config.read_text(encoding="utf-8")
        .replace("terrain_type: plane", "terrain_type: generator")
        .replace("terrain_generator: null", "terrain_generator: {}"),
        encoding="utf-8",
    )
    rough_digest = hashlib.sha256(replacement.read_bytes()).hexdigest()
    env_inode = env_config.stat().st_ino
    real_fstat = provenance_module.os.fstat
    env_fstat_calls = 0

    def replace_after_env_read(descriptor):
        nonlocal env_fstat_calls
        status = real_fstat(descriptor)
        if status.st_ino == env_inode:
            env_fstat_calls += 1
            if env_fstat_calls == 2:
                replacement.replace(env_config)
        return status

    monkeypatch.setattr(provenance_module.os, "fstat", replace_after_env_read)

    with pytest.raises(ValueError, match="changed while snapshotting"):
        _build_manifest(checkpoint, [env_config, agent_config])

    assert hashlib.sha256(env_config.read_bytes()).hexdigest() == rough_digest


def test_snapshot_rejects_same_size_multichunk_mutation_with_restored_mtime(tmp_path, monkeypatch):
    chunk_size = 1024 * 1024
    checkpoint = tmp_path / "policy.pt"
    original = b"A" * chunk_size + b"B" * chunk_size
    mutated = b"C" * chunk_size + b"D" * chunk_size
    checkpoint.write_bytes(original)
    original_status = checkpoint.stat()
    real_sha256 = hashlib.sha256
    update_count = 0

    class MutatingDigest:
        def __init__(self):
            self._digest = real_sha256()

        def update(self, chunk):
            nonlocal update_count
            self._digest.update(chunk)
            update_count += 1
            if update_count == 1:
                with checkpoint.open("r+b") as stream:
                    stream.write(mutated)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.utime(
                    checkpoint,
                    ns=(original_status.st_atime_ns, original_status.st_mtime_ns),
                )

        def hexdigest(self):
            return self._digest.hexdigest()

    monkeypatch.setattr(provenance_module.hashlib, "sha256", MutatingDigest)

    with pytest.raises(ValueError, match="changed while snapshotting"):
        snapshot_file(checkpoint, chunk_size=chunk_size)

    assert update_count == 2
    assert checkpoint.stat().st_size == original_status.st_size
    assert checkpoint.stat().st_mtime_ns == original_status.st_mtime_ns
    assert checkpoint.read_bytes() == mutated


@pytest.mark.parametrize(
    ("yaml_integer", "expected_num_envs"),
    [
        ("010000", 10_000),
        ("0x1000", 4096),
        ("!!int 010000", 10_000),
        ("4_096", 4096),
        ("+4096", 4096),
    ],
)
def test_build_manifest_rejects_noncanonical_yaml_integers(tmp_path, yaml_integer, expected_num_envs):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    env_config, agent_config = _write_required_configs(tmp_path)
    env_config.write_text(
        env_config.read_text(encoding="utf-8").replace("num_envs: 4096", f"num_envs: {yaml_integer}"),
        encoding="utf-8",
    )
    command = [
        "python",
        "train.py",
        "--task",
        STRICT_FLAT_TASK,
        "--seed",
        "7",
        "--num-envs",
        str(expected_num_envs),
        "--max_iterations",
        "1500",
    ]

    with pytest.raises(ValueError, match="canonical decimal"):
        _build_manifest(
            checkpoint,
            [env_config, agent_config],
            num_envs=expected_num_envs,
            training_command=command,
        )


@pytest.mark.parametrize("missing_kind", ["checkpoint", "config"])
def test_build_flat_baseline_manifest_rejects_missing_input(tmp_path, missing_kind):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    env_config, agent_config = _write_required_configs(tmp_path)
    if missing_kind == "checkpoint":
        checkpoint.unlink()
    else:
        agent_config.unlink()

    with pytest.raises(FileNotFoundError):
        _build_manifest(checkpoint, [env_config, agent_config])


def test_sha256_file_matches_hashlib_and_rejects_invalid_chunk_size(tmp_path):
    artifact = tmp_path / "artifact.bin"
    content = bytes(range(256)) * 17
    artifact.write_bytes(content)

    assert sha256_file(artifact, chunk_size=31) == hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="chunk_size"):
        sha256_file(artifact, chunk_size=0)


def test_write_json_atomic_round_trips_deterministically_without_temp_file(tmp_path):
    output = tmp_path / "manifest.json"
    payload = {"z": [3, 2, 1], "a": {"value": "flat"}}

    write_json_atomic(output, payload)

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert output.read_text(encoding="utf-8") == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_atomic_rejects_non_finite_values_without_replacing_output(tmp_path):
    output = tmp_path / "manifest.json"
    original = '{\n  "valid": true\n}\n'
    output.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError):
        write_json_atomic(output, {"metric": float("nan")})

    assert output.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("failed_operation", ["fsync", "replace"])
def test_write_json_atomic_cleans_temp_and_preserves_output_on_io_failure(
    tmp_path,
    monkeypatch,
    failed_operation,
):
    output = tmp_path / "manifest.json"
    original = '{\n  "valid": true\n}\n'
    output.write_text(original, encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise OSError(f"forced {failed_operation} failure")

    monkeypatch.setattr(provenance_module.os, failed_operation, fail)

    with pytest.raises(OSError, match=f"forced {failed_operation} failure"):
        write_json_atomic(output, {"valid": False})

    assert output.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_manifest_is_deterministic_regardless_of_config_order(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    env_config, agent_config = _write_required_configs(tmp_path)
    extra = tmp_path / "extra.yaml"
    extra.write_text("name: extra\n", encoding="utf-8")

    forward = _build_manifest(checkpoint, [env_config, agent_config, extra])
    reverse = _build_manifest(checkpoint, [extra, agent_config, env_config])

    assert forward == reverse
    assert list(forward["files"]) == sorted(forward["files"])


def test_training_command_file_requires_exactly_one_non_empty_line(tmp_path):
    command_file = tmp_path / "command.txt"
    command_file.write_text('python train.py --run-name "strict flat"\n', encoding="utf-8")

    assert manifest_script.read_training_command(command_file) == [
        "python",
        "train.py",
        "--run-name",
        "strict flat",
    ]

    command_file.write_text("python first.py\npython second.py\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one non-empty command line"):
        manifest_script.read_training_command(command_file)


def test_write_manifest_refuses_different_existing_content_and_preserves_equal_file(tmp_path):
    output = tmp_path / "manifest.json"
    payload = {"schema_version": 1, "terrain_assertion": "plane-only"}
    write_json_atomic(output, payload)
    original_stat = output.stat()

    manifest_script.write_manifest_if_unchanged(output, payload)
    assert output.stat().st_mtime_ns == original_stat.st_mtime_ns

    with pytest.raises(FileExistsError, match="differs"):
        manifest_script.write_manifest_if_unchanged(output, {"schema_version": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_write_manifest_preserves_concurrent_different_winner(tmp_path, monkeypatch):
    output = tmp_path / "manifest.json"
    requested = {"schema_version": 1, "git_sha": "a" * 40}
    winner = {"schema_version": 1, "git_sha": "b" * 40}

    def install_winner_then_fail(_temporary, destination):
        Path(destination).write_text(
            json.dumps(winner, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise FileExistsError("concurrent winner")

    monkeypatch.setattr(os, "link", install_winner_then_fail)

    with pytest.raises(FileExistsError, match="differs"):
        manifest_script.write_manifest_if_unchanged(output, requested)

    assert json.loads(output.read_text(encoding="utf-8")) == winner
    assert list(tmp_path.glob("*.tmp")) == []


def test_collect_versions_raises_when_package_metadata_is_missing(monkeypatch):
    def missing_metadata(_distribution):
        raise manifest_script.metadata.PackageNotFoundError

    monkeypatch.setattr(manifest_script.metadata, "version", missing_metadata)

    with pytest.raises(RuntimeError, match="package metadata.*isaaclab"):
        manifest_script.collect_versions()


def test_current_git_sha_invokes_git_with_reproducible_flags_and_trims_output(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f" {GIT_SHA}\n")

    monkeypatch.setattr(manifest_script.subprocess, "run", fake_run)

    project_root = Path(manifest_script.__file__).resolve().parents[3]
    assert manifest_script.current_git_sha() == GIT_SHA
    assert calls == [
        (
            ["git", "-C", str(project_root), "rev-parse", "--verify", "HEAD^{commit}"],
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_current_git_sha_rejects_empty_output(monkeypatch):
    monkeypatch.setattr(
        manifest_script.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=" \n"),
    )

    with pytest.raises(RuntimeError, match="empty SHA"):
        manifest_script.current_git_sha()


@pytest.mark.parametrize("cwd_kind", ["non_repo", "other_repo"])
def test_current_git_sha_is_independent_of_caller_cwd(tmp_path, cwd_kind):
    project_root = Path(__file__).resolve().parents[2]
    caller_cwd = tmp_path / cwd_kind
    caller_cwd.mkdir()
    if cwd_kind == "other_repo":
        subprocess.run(["git", "init", "-q", str(caller_cwd)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(caller_cwd),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "other repository",
            ],
            check=True,
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(project_root / "source/rl_training"), str(project_root))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.reinforcement_learning.online_adaptation."
            "build_flat_baseline_manifest import current_git_sha; print(current_git_sha())",
        ],
        cwd=caller_cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    expected = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "--verify", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected


def test_manifest_parser_and_help_import_without_isaac_or_pxr():
    program = r'''\
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"isaaclab", "pxr"}:
        raise AssertionError(f"forbidden simulator import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from scripts.reinforcement_learning.online_adaptation.cli import build_flat_baseline_manifest_parser
import scripts.reinforcement_learning.online_adaptation.build_flat_baseline_manifest

parser = build_flat_baseline_manifest_parser()
try:
    parser.parse_args(["--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("--help did not exit")
assert "isaaclab" not in sys.modules
assert "pxr" not in sys.modules
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "source/rl_training:."

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Record strict-flat Lite3 PPO baseline provenance." in completed.stdout
