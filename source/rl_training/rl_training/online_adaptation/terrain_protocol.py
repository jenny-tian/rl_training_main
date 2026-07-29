from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real


MAX_PROTOCOL_INTEGER = 2**63 - 1


def _normalize_nonnegative_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    try:
        is_negative = value < 0
        is_nan = value != value
        is_infinite = value == math.inf or value == -math.inf
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonnegative real number") from error
    if is_negative or is_nan or is_infinite:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must convert to a finite Python float") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must convert to a finite Python float")
    if normalized == 0.0:
        return 0.0
    return normalized


def _validate_positive_integer(value: object, name: str) -> None:
    if type(value) is not int or not 1 <= value <= MAX_PROTOCOL_INTEGER:
        raise ValueError(
            f"{name} must be a positive integer between 1 and {MAX_PROTOCOL_INTEGER}"
        )


def _validate_seed(seed: object, name: str = "seed") -> None:
    if type(seed) is not int or not 0 <= seed <= MAX_PROTOCOL_INTEGER:
        raise ValueError(
            f"{name} must be an exact integer between 0 and {MAX_PROTOCOL_INTEGER}"
        )


@dataclass(frozen=True)
class TerrainGeometry:
    step_height: float
    rough_amplitude: float
    seed: int

    def __post_init__(self) -> None:
        step_height = _normalize_nonnegative_real(self.step_height, "step_height")
        rough_amplitude = _normalize_nonnegative_real(self.rough_amplitude, "rough_amplitude")
        _validate_seed(self.seed)
        object.__setattr__(self, "step_height", step_height)
        object.__setattr__(self, "rough_amplitude", rough_amplitude)


def apply_geometry_to_env_cfg(env_cfg: object, geometry: TerrainGeometry) -> None:
    if not isinstance(geometry, TerrainGeometry):
        raise TypeError("geometry must be a TerrainGeometry instance")

    try:
        terrain_generator = env_cfg.scene.terrain.terrain_generator
    except AttributeError as error:
        raise ValueError("env_cfg must define scene.terrain.terrain_generator") from error
    if terrain_generator is None:
        raise ValueError("env_cfg scene terrain_generator must not be None")
    if not hasattr(terrain_generator, "seed"):
        raise ValueError("terrain_generator must define a mutable seed")

    sub_terrains = getattr(terrain_generator, "sub_terrains", None)
    if not isinstance(sub_terrains, Mapping) or set(sub_terrains) != {"flat_online"}:
        raise ValueError("terrain_generator must contain the sole 'flat_online' subterrain")

    flat_online = sub_terrains["flat_online"]
    missing_fields = [
        name
        for name in ("seed", "step_height", "rough_amplitude")
        if not hasattr(flat_online, name)
    ]
    if missing_fields:
        raise ValueError(
            "flat_online subterrain must define mutable fields: "
            + ", ".join(missing_fields)
        )

    assignments = (
        ("flat_online.seed", flat_online, "seed", geometry.seed),
        (
            "flat_online.step_height",
            flat_online,
            "step_height",
            geometry.step_height,
        ),
        (
            "flat_online.rough_amplitude",
            flat_online,
            "rough_amplitude",
            geometry.rough_amplitude,
        ),
        ("terrain_generator.seed", terrain_generator, "seed", geometry.seed),
    )
    snapshots = [
        (name, target, attribute, getattr(target, attribute), value)
        for name, target, attribute, value in assignments
    ]
    applied = []
    for name, target, attribute, original, value in snapshots:
        try:
            setattr(target, attribute, value)
        except Exception as error:
            rollback_failures = []
            rollback_targets = []
            try:
                if getattr(target, attribute) != original:
                    rollback_targets.append((name, target, attribute, original))
            except Exception as inspection_error:
                rollback_failures.append((name, inspection_error))
            rollback_targets.extend(reversed(applied))

            for rollback_name, rollback_target, rollback_attribute, rollback_original in (
                rollback_targets
            ):
                try:
                    setattr(rollback_target, rollback_attribute, rollback_original)
                except Exception as rollback_error:
                    try:
                        restored = (
                            getattr(rollback_target, rollback_attribute) == rollback_original
                        )
                    except Exception as inspection_error:
                        rollback_failures.append(
                            (
                                rollback_name,
                                f"{rollback_error}; unable to verify restored value: "
                                f"{inspection_error}",
                            )
                        )
                    else:
                        if not restored:
                            rollback_failures.append((rollback_name, rollback_error))

            message = f"failed to apply terrain geometry at {name}"
            if rollback_failures:
                details = ", ".join(
                    f"{rollback_name}: {rollback_error}"
                    for rollback_name, rollback_error in rollback_failures
                )
                message += f"; rollback failed for {details}"
            else:
                message += "; previous values restored"
            raise ValueError(message) from error
        applied.append((name, target, attribute, original))


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    step_height: float
    rough_amplitude: float
    max_episodes: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("stage name must be a nonempty string")
        step_height = _normalize_nonnegative_real(self.step_height, "step_height")
        rough_amplitude = _normalize_nonnegative_real(self.rough_amplitude, "rough_amplitude")
        _validate_positive_integer(self.max_episodes, "max_episodes")
        object.__setattr__(self, "step_height", step_height)
        object.__setattr__(self, "rough_amplitude", rough_amplitude)


@dataclass(frozen=True)
class FlatOnlineProtocol:
    stages: tuple[CurriculumStage, ...]
    development_seeds: tuple[int, ...]
    sealed_seeds: tuple[int, ...]
    max_total_episodes: int = 100
    max_total_steps: int = 200_000
    max_steps_per_episode: int = 2_000

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.stages, tuple) or not self.stages:
            raise ValueError("stages must be a nonempty tuple")
        if any(not isinstance(stage, CurriculumStage) for stage in self.stages):
            raise ValueError("stages must contain only CurriculumStage instances")
        stage_names = tuple(stage.name for stage in self.stages)
        if len(stage_names) != len(set(stage_names)):
            raise ValueError("stage names must be unique")

        self._validate_seed_tuple(self.development_seeds, "development_seeds")
        self._validate_seed_tuple(self.sealed_seeds, "sealed_seeds")
        if not set(self.development_seeds).isdisjoint(self.sealed_seeds):
            raise ValueError("development_seeds and sealed_seeds must be disjoint")

        _validate_positive_integer(self.max_total_episodes, "max_total_episodes")
        _validate_positive_integer(self.max_total_steps, "max_total_steps")
        _validate_positive_integer(self.max_steps_per_episode, "max_steps_per_episode")
        if sum(stage.max_episodes for stage in self.stages) > self.max_total_episodes:
            raise ValueError("stage episode budgets must not exceed max_total_episodes")

    @staticmethod
    def _validate_seed_tuple(seeds: object, name: str) -> None:
        if not isinstance(seeds, tuple) or not seeds:
            raise ValueError(f"{name} must be a nonempty tuple")
        for seed in seeds:
            _validate_seed(seed, f"{name} entry")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{name} entries must be unique")

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


_DEFAULT_STAGES = (
    CurriculumStage("step_2cm", 0.02, 0.0),
    CurriculumStage("step_4cm", 0.04, 0.0),
    CurriculumStage("step_6cm", 0.06, 0.0),
    CurriculumStage("rough_2cm", 0.0, 0.02),
    CurriculumStage("rough_4cm", 0.0, 0.04),
)
_DEVELOPMENT_SEEDS = (101, 103, 107, 109, 113)
_SEALED_SEEDS = (
    1009,
    1013,
    1019,
    1021,
    1031,
    1033,
    1039,
    1049,
    1051,
    1061,
    1063,
    1069,
    1087,
    1091,
    1093,
    1097,
    1103,
    1109,
    1117,
    1123,
)


def default_flat_online_protocol() -> FlatOnlineProtocol:
    return FlatOnlineProtocol(
        stages=_DEFAULT_STAGES,
        development_seeds=_DEVELOPMENT_SEEDS,
        sealed_seeds=_SEALED_SEEDS,
    )


def _resolve_protocol(protocol: FlatOnlineProtocol | None) -> FlatOnlineProtocol:
    if protocol is None:
        return default_flat_online_protocol()
    if not isinstance(protocol, FlatOnlineProtocol):
        raise ValueError("protocol must be a FlatOnlineProtocol instance")
    protocol._validate()
    return protocol


def development_geometry(
    stage_name: str,
    seed: int,
    protocol: FlatOnlineProtocol | None = None,
) -> TerrainGeometry:
    resolved = _resolve_protocol(protocol)
    try:
        stage = next(stage for stage in resolved.stages if stage.name == stage_name)
    except StopIteration as error:
        raise ValueError(f"unknown curriculum stage: {stage_name!r}") from error
    _validate_seed(seed, "development seed")
    if seed not in resolved.development_seeds:
        raise ValueError(f"seed {seed} is not a development seed")
    return TerrainGeometry(stage.step_height, stage.rough_amplitude, seed)


def sealed_geometry(
    seed: int,
    protocol: FlatOnlineProtocol | None = None,
) -> TerrainGeometry:
    resolved = _resolve_protocol(protocol)
    _validate_seed(seed, "sealed seed")
    if seed not in resolved.sealed_seeds:
        raise ValueError(f"seed {seed} is not a sealed seed")
    index = resolved.sealed_seeds.index(seed)
    return TerrainGeometry(
        step_height=(0.04, 0.05, 0.06)[index % 3],
        rough_amplitude=0.04,
        seed=seed,
    )
