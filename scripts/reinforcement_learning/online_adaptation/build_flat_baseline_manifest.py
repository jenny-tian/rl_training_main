"""Build a reproducible provenance manifest for a strict-flat Lite3 PPO baseline."""

from __future__ import annotations

import errno
import json
import os
import shlex
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from rl_training.online_adaptation.provenance import (
    REQUIRED_VERSION_KEYS,
    build_flat_baseline_manifest,
)
from scripts.reinforcement_learning.online_adaptation.cli import build_flat_baseline_manifest_parser


REQUIRED_DISTRIBUTIONS = tuple(sorted(REQUIRED_VERSION_KEYS - {"python"}))
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_training_command(path: str | Path) -> list[str]:
    command_path = Path(path)
    lines = [line.strip() for line in command_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("training command file must contain exactly one non-empty command line")
    command = shlex.split(lines[0])
    if not command:
        raise ValueError("training command file must contain exactly one non-empty command line")
    return command


def collect_versions() -> dict[str, str]:
    versions = {"python": sys.version}
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"package metadata unavailable for {distribution}") from error
    return versions


def current_git_sha() -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--verify", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    )
    git_sha = completed.stdout.strip()
    if not git_sha:
        raise RuntimeError("git rev-parse HEAD returned an empty SHA")
    return git_sha


def write_manifest_if_unchanged(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)

    descriptor: int | None = None
    temporary_path: Path | None = None
    installed = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, output)
            installed = True
        except OSError as error:
            if not isinstance(error, FileExistsError) and error.errno != errno.EEXIST:
                raise
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
            except json.JSONDecodeError as decode_error:
                raise ValueError(f"existing output is not valid JSON: {output}") from decode_error
            if existing != payload:
                raise FileExistsError(
                    f"existing manifest differs from requested manifest: {output}"
                ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if installed:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(output.parent, flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


def main(argv: list[str] | None = None) -> int:
    args = build_flat_baseline_manifest_parser().parse_args(argv)
    manifest = build_flat_baseline_manifest(
        checkpoint=args.checkpoint,
        config_paths=args.config,
        training_command=read_training_command(args.training_command_file),
        git_sha=current_git_sha(),
        task=args.task,
        seed=args.seed,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        versions=collect_versions(),
    )
    write_manifest_if_unchanged(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
