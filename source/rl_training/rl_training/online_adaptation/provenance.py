from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode


STRICT_FLAT_BASELINE_TASK = "Flat-Deeprobotics-Lite3-v0"
REQUIRED_VERSION_KEYS = frozenset(("python", "isaaclab", "rsl-rl-lib", "torch"))
YAML_NULL_TAG = "tag:yaml.org,2002:null"
YAML_STRING_TAG = "tag:yaml.org,2002:str"
YAML_INT_TAG = "tag:yaml.org,2002:int"
_FULL_GIT_SHA = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")
_CANONICAL_YAML_INT = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
_COMMAND_FLAG_ALIASES = {
    "task": ("--task",),
    "seed": ("--seed",),
    "num_envs": ("--num_envs", "--num-envs"),
    "max_iterations": ("--max_iterations", "--max-iterations"),
}


@dataclass(frozen=True)
class FileSnapshot:
    sha256: str
    bytes: int
    content: bytes | None = None


def snapshot_file(
    path: str | Path,
    *,
    capture_bytes: bool = False,
    chunk_size: int = 1024 * 1024,
) -> FileSnapshot:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    input_path = Path(path)
    digest = hashlib.sha256()
    byte_count = 0
    captured = bytearray() if capture_bytes else None
    with input_path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
            byte_count += len(chunk)
            if captured is not None:
                captured.extend(chunk)
        after = os.fstat(stream.fileno())

    try:
        path_after = input_path.stat()
    except OSError as error:
        raise ValueError(f"input changed while snapshotting: {input_path}") from error

    def identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )

    if identity(before) != identity(after) or identity(before) != identity(path_after):
        raise ValueError(f"input changed while snapshotting: {input_path}")
    if byte_count != before.st_size:
        raise ValueError(f"input changed while snapshotting: {input_path}")
    return FileSnapshot(
        sha256=digest.hexdigest(),
        bytes=byte_count,
        content=bytes(captured) if captured is not None else None,
    )


def _mapping_value(mapping: MappingNode, key: str, path: str, context: str) -> Node:
    values = [
        value_node
        for key_node, value_node in mapping.value
        if isinstance(key_node, ScalarNode)
        and key_node.tag == YAML_STRING_TAG
        and key_node.value == key
    ]
    if len(values) != 1:
        raise ValueError(f"{context} requires exactly one {path}.{key}")
    return values[0]


def _required_config_paths(config_paths: list[Path]) -> tuple[Path, Path]:
    env_configs = [path for path in config_paths if path.name == "env.yaml"]
    agent_configs = [path for path in config_paths if path.name == "agent.yaml"]
    if len(env_configs) != 1 or len(agent_configs) != 1:
        raise ValueError(
            "strict plane-only provenance requires exactly one env.yaml and exactly one agent.yaml"
        )
    return env_configs[0], agent_configs[0]


def _compose_mapping(content: bytes, name: str, context: str) -> MappingNode:
    try:
        document = yaml.compose(content, Loader=yaml.SafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{context} requires a valid {name} mapping") from error
    if not isinstance(document, MappingNode):
        raise ValueError(f"{context} requires {name} to contain a mapping")
    return document


def _node_int(mapping: MappingNode, key: str, path: str) -> int:
    node = _mapping_value(mapping, key, path, "semantic provenance")
    if (
        not isinstance(node, ScalarNode)
        or node.tag != YAML_INT_TAG
        or _CANONICAL_YAML_INT.fullmatch(node.value) is None
    ):
        raise ValueError(
            f"semantic provenance requires {path}.{key} to use canonical decimal integer text"
        )
    return int(node.value, 10)


def _validate_config_snapshots(
    env_content: bytes,
    agent_content: bytes,
    *,
    seed: int,
    num_envs: int,
    max_iterations: int,
) -> None:
    env_config = _compose_mapping(env_content, "env.yaml", "strict plane-only provenance")
    agent_config = _compose_mapping(agent_content, "agent.yaml", "semantic provenance")

    scene = _mapping_value(env_config, "scene", "env.yaml", "strict plane-only provenance")
    if not isinstance(scene, MappingNode):
        raise ValueError("strict plane-only provenance requires scene to be a mapping")
    terrain = _mapping_value(scene, "terrain", "scene", "strict plane-only provenance")
    if not isinstance(terrain, MappingNode):
        raise ValueError("strict plane-only provenance requires scene.terrain to be a mapping")
    terrain_type = _mapping_value(
        terrain,
        "terrain_type",
        "scene.terrain",
        "strict plane-only provenance",
    )
    terrain_generator = _mapping_value(
        terrain,
        "terrain_generator",
        "scene.terrain",
        "strict plane-only provenance",
    )
    if not (
        isinstance(terrain_type, ScalarNode)
        and terrain_type.tag == YAML_STRING_TAG
        and terrain_type.value == "plane"
        and isinstance(terrain_generator, ScalarNode)
        and terrain_generator.tag == YAML_NULL_TAG
    ):
        raise ValueError(
            "strict plane-only provenance requires terrain_type 'plane' and terrain_generator null"
        )

    configured_values = {
        "env.yaml seed": _node_int(env_config, "seed", "env.yaml"),
        "env.yaml scene.num_envs": _node_int(scene, "num_envs", "env.yaml.scene"),
        "agent.yaml seed": _node_int(agent_config, "seed", "agent.yaml"),
        "agent.yaml max_iterations": _node_int(agent_config, "max_iterations", "agent.yaml"),
    }
    expected_values = {
        "env.yaml seed": seed,
        "env.yaml scene.num_envs": num_envs,
        "agent.yaml seed": seed,
        "agent.yaml max_iterations": max_iterations,
    }
    for name, configured in configured_values.items():
        if configured != expected_values[name]:
            raise ValueError(f"semantic provenance {name} does not match builder arguments")


def _validate_scalar_contracts(
    *,
    seed: int,
    num_envs: int,
    max_iterations: int,
    git_sha: str,
    training_command: list[str],
    versions: dict[str, str],
) -> str:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(num_envs) is not int or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not isinstance(git_sha, str) or _FULL_GIT_SHA.fullmatch(git_sha) is None:
        raise ValueError("git_sha must be a full 40- or 64-character hexadecimal commit SHA")
    if type(training_command) is not list or not training_command:
        raise ValueError("training_command must be a nonempty list")
    if any(not isinstance(token, str) or not token.strip() for token in training_command):
        raise ValueError("training_command entries must be nonempty strings")
    if not isinstance(versions, dict):
        raise ValueError("versions must be a mapping")

    missing_versions = sorted(REQUIRED_VERSION_KEYS.difference(versions))
    if missing_versions:
        raise ValueError(f"required version keys missing: {', '.join(missing_versions)}")
    invalid_versions = sorted(
        key
        for key in REQUIRED_VERSION_KEYS
        if not isinstance(versions[key], str)
        or not versions[key].strip()
        or versions[key].strip().lower() == "unavailable"
    )
    if invalid_versions:
        raise ValueError(f"required version values must be concrete strings: {', '.join(invalid_versions)}")
    return git_sha.lower()


def _command_flag_values(training_command: list[str]) -> dict[str, str]:
    found = {name: [] for name in _COMMAND_FLAG_ALIASES}
    for index, token in enumerate(training_command):
        for name, aliases in _COMMAND_FLAG_ALIASES.items():
            for flag in aliases:
                if token == flag:
                    if index + 1 >= len(training_command):
                        raise ValueError(f"training command flag {flag} has no value")
                    found[name].append(training_command[index + 1])
                elif token.startswith(f"{flag}="):
                    found[name].append(token.split("=", 1)[1])
    invalid_counts = sorted(name for name, values in found.items() if len(values) != 1)
    if invalid_counts:
        raise ValueError(
            "training command requires exactly one value for: " + ", ".join(invalid_counts)
        )
    return {name: values[0] for name, values in found.items()}


def _validate_training_command(
    training_command: list[str],
    *,
    task: str,
    seed: int,
    num_envs: int,
    max_iterations: int,
) -> None:
    values = _command_flag_values(training_command)
    if values["task"] != task:
        raise ValueError("training command --task does not match builder arguments")
    for name, expected in (
        ("seed", seed),
        ("num_envs", num_envs),
        ("max_iterations", max_iterations),
    ):
        try:
            configured = int(values[name], 10)
        except ValueError as error:
            raise ValueError(f"training command {name} must be an integer") from error
        if configured != expected:
            raise ValueError(f"training command {name} does not match builder arguments")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_flat_baseline_manifest(
    *,
    checkpoint: str | Path,
    config_paths: list[str | Path],
    training_command: list[str],
    git_sha: str,
    task: str,
    seed: int,
    num_envs: int,
    max_iterations: int,
    versions: dict[str, str],
) -> dict[str, object]:
    if task != STRICT_FLAT_BASELINE_TASK:
        raise ValueError(f"task must be exactly {STRICT_FLAT_BASELINE_TASK!r}")
    normalized_git_sha = _validate_scalar_contracts(
        seed=seed,
        num_envs=num_envs,
        max_iterations=max_iterations,
        git_sha=git_sha,
        training_command=training_command,
        versions=versions,
    )
    _validate_training_command(
        training_command,
        task=task,
        seed=seed,
        num_envs=num_envs,
        max_iterations=max_iterations,
    )

    input_paths = [Path(checkpoint), *(Path(path) for path in config_paths)]
    resolved_paths = [path.resolve() for path in input_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("duplicate resolved input paths are not allowed")
    for path in resolved_paths:
        if not path.is_file():
            raise FileNotFoundError(f"provenance input is not a file: {path}")

    resolved_configs = resolved_paths[1:]
    env_config, agent_config = _required_config_paths(resolved_configs)
    snapshots = {
        path: snapshot_file(path, capture_bytes=path in (env_config, agent_config))
        for path in resolved_paths
    }
    env_content = snapshots[env_config].content
    agent_content = snapshots[agent_config].content
    if env_content is None or agent_content is None:
        raise RuntimeError("required config snapshots did not capture bytes")
    _validate_config_snapshots(
        env_content,
        agent_content,
        seed=seed,
        num_envs=num_envs,
        max_iterations=max_iterations,
    )

    files = {
        str(path): {
            "sha256": snapshots[path].sha256,
            "bytes": snapshots[path].bytes,
        }
        for path in sorted(resolved_paths, key=str)
    }
    return {
        "schema_version": 1,
        "terrain_assertion": "plane-only",
        "task": task,
        "git_sha": normalized_git_sha,
        "training": {
            "seed": seed,
            "num_envs": num_envs,
            "max_iterations": max_iterations,
            "command": list(training_command),
        },
        "versions": dict(sorted(versions.items())),
        "files": files,
    }


def write_json_atomic(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)

    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
