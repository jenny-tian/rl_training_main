import hashlib
import json
import math
import random
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import numpy as np
import pytest

import rl_training.online_adaptation as online_adaptation
import rl_training.online_adaptation.terrain_protocol as terrain_protocol_module
from rl_training.online_adaptation.terrain_protocol import (
    CurriculumStage,
    FlatOnlineProtocol,
    TerrainGeometry,
    default_flat_online_protocol,
    development_geometry,
    sealed_geometry,
)


EXPECTED_MAX_PROTOCOL_INTEGER = 2**63 - 1
DEVELOPMENT_SEEDS = (101, 103, 107, 109, 113)
SEALED_SEEDS = (
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
EXPECTED_STAGES = (
    CurriculumStage("step_2cm", 0.02, 0.0),
    CurriculumStage("step_4cm", 0.04, 0.0),
    CurriculumStage("step_6cm", 0.06, 0.0),
    CurriculumStage("rough_2cm", 0.0, 0.02),
    CurriculumStage("rough_4cm", 0.0, 0.04),
)


def test_terrain_protocol_is_part_of_the_public_package_api():
    expected = {
        "TerrainGeometry": TerrainGeometry,
        "CurriculumStage": CurriculumStage,
        "FlatOnlineProtocol": FlatOnlineProtocol,
        "default_flat_online_protocol": default_flat_online_protocol,
        "development_geometry": development_geometry,
        "sealed_geometry": sealed_geometry,
    }

    for name, value in expected.items():
        assert getattr(online_adaptation, name) is value
        assert name in online_adaptation.__all__


def test_protocol_integer_bound_is_stable_across_platforms():
    assert terrain_protocol_module.MAX_PROTOCOL_INTEGER == EXPECTED_MAX_PROTOCOL_INTEGER


def test_default_protocol_has_exact_ordered_stages_and_budgets():
    protocol = default_flat_online_protocol()

    assert protocol.stages == EXPECTED_STAGES
    assert all(stage.max_episodes == 20 for stage in protocol.stages)
    assert protocol.max_total_episodes == 100
    assert protocol.max_total_steps == 200_000
    assert protocol.max_steps_per_episode == 2_000


def test_default_protocol_has_exact_disjoint_seed_sets():
    protocol = default_flat_online_protocol()

    assert protocol.development_seeds == DEVELOPMENT_SEEDS
    assert protocol.sealed_seeds == SEALED_SEEDS
    assert len(protocol.sealed_seeds) == 20
    assert set(protocol.development_seeds).isdisjoint(protocol.sealed_seeds)


def test_sealed_geometry_uses_deterministic_step_cycle_and_fixed_roughness():
    geometries = tuple(sealed_geometry(seed) for seed in SEALED_SEEDS)

    assert geometries == tuple(
        TerrainGeometry(
            step_height=(0.04, 0.05, 0.06)[index % 3],
            rough_amplitude=0.04,
            seed=seed,
        )
        for index, seed in enumerate(SEALED_SEEDS)
    )
    heights = tuple(geometry.step_height for geometry in geometries)
    assert set(heights) == {0.04, 0.05, 0.06}
    assert tuple(heights.count(height) for height in (0.04, 0.05, 0.06)) == (7, 7, 6)


def test_sealed_geometry_repeats_expected_first_six_heights():
    expected = (0.04, 0.05, 0.06, 0.04, 0.05, 0.06)

    first = tuple(sealed_geometry(seed).step_height for seed in SEALED_SEEDS[:6])
    second = tuple(sealed_geometry(seed).step_height for seed in SEALED_SEEDS[:6])

    assert first == expected
    assert second == expected


@pytest.mark.parametrize("stage", EXPECTED_STAGES, ids=lambda stage: stage.name)
def test_development_geometry_maps_each_stage(stage):
    assert development_geometry(stage.name, DEVELOPMENT_SEEDS[0]) == TerrainGeometry(
        step_height=stage.step_height,
        rough_amplitude=stage.rough_amplitude,
        seed=DEVELOPMENT_SEEDS[0],
    )


def test_development_geometry_rejects_unknown_stage_and_non_development_seed():
    with pytest.raises(ValueError, match="unknown curriculum stage"):
        development_geometry("missing", DEVELOPMENT_SEEDS[0])
    with pytest.raises(ValueError, match="development seed"):
        development_geometry("step_2cm", SEALED_SEEDS[0])


@pytest.mark.parametrize("seed", [DEVELOPMENT_SEEDS[0], 999_999, True])
def test_sealed_geometry_rejects_non_sealed_seed(seed):
    with pytest.raises(ValueError, match="sealed seed"):
        sealed_geometry(seed)


def test_canonical_json_is_compact_deterministic_and_complete():
    first = default_flat_online_protocol()
    second = default_flat_online_protocol()
    expected = {
        "stages": [
            {
                "name": stage.name,
                "step_height": stage.step_height,
                "rough_amplitude": stage.rough_amplitude,
                "max_episodes": stage.max_episodes,
            }
            for stage in EXPECTED_STAGES
        ],
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "sealed_seeds": list(SEALED_SEEDS),
        "max_total_episodes": 100,
        "max_total_steps": 200_000,
        "max_steps_per_episode": 2_000,
    }

    canonical = first.canonical_json()

    assert json.loads(canonical) == expected
    assert canonical == json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )
    assert first.sha256() == second.sha256()
    assert first.sha256() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert len(first.sha256()) == 64
    assert first.sha256() == first.sha256().lower()


def test_one_field_change_changes_protocol_sha256():
    protocol = default_flat_online_protocol()
    changed = replace(protocol, max_total_steps=protocol.max_total_steps + 1)

    assert changed.sha256() != protocol.sha256()


@pytest.mark.parametrize(
    ("instance", "field", "value"),
    [
        (TerrainGeometry(0.02, 0.0, 101), "seed", 103),
        (CurriculumStage("step", 0.02, 0.0), "name", "other"),
        (default_flat_online_protocol(), "max_total_episodes", 101),
    ],
)
def test_protocol_dataclasses_are_frozen(instance, field, value):
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field, value)


@pytest.mark.parametrize(
    "value",
    [True, False, 1 + 0j, float("nan"), float("inf"), -0.01, "0.01"],
)
def test_geometry_rejects_invalid_heights_and_amplitudes(value):
    with pytest.raises(ValueError):
        TerrainGeometry(value, 0.0, 101)
    with pytest.raises(ValueError):
        TerrainGeometry(0.0, value, 101)
    with pytest.raises(ValueError):
        CurriculumStage("stage", value, 0.0)
    with pytest.raises(ValueError):
        CurriculumStage("stage", 0.0, value)


@pytest.mark.parametrize(
    "value",
    [0, 0.25, Fraction(1, 3), Fraction(1, 10**10000), np.float32(0.5), np.float64(0.75)],
    ids=["int", "float", "fraction", "underflowing-fraction", "float32", "float64"],
)
def test_every_accepted_protocol_numeric_value_normalizes_and_serializes(value):
    stage = CurriculumStage("stage", value, value, max_episodes=1)
    protocol = FlatOnlineProtocol(
        stages=(stage,),
        development_seeds=(101,),
        sealed_seeds=(1009,),
        max_total_episodes=1,
    )

    assert type(stage.step_height) is float
    assert type(stage.rough_amplitude) is float
    assert json.loads(protocol.canonical_json())["stages"][0] == {
        "max_episodes": 1,
        "name": "stage",
        "rough_amplitude": float(value),
        "step_height": float(value),
    }


def test_fraction_and_numpy_reals_normalize_and_hash_deterministically():
    first_stage = CurriculumStage("mixed", Fraction(1, 8), np.float32(0.25), max_episodes=1)
    second_stage = CurriculumStage("mixed", 0.125, 0.25, max_episodes=1)
    first = FlatOnlineProtocol((first_stage,), (101,), (1009,), max_total_episodes=1)
    second = FlatOnlineProtocol((second_stage,), (101,), (1009,), max_total_episodes=1)
    geometry = TerrainGeometry(np.float64(0.5), Fraction(1, 16), 101)

    assert type(first_stage.step_height) is float
    assert type(first_stage.rough_amplitude) is float
    assert type(geometry.step_height) is float
    assert type(geometry.rough_amplitude) is float
    assert first.canonical_json() == second.canonical_json()
    assert first.sha256() == second.sha256()


def test_signed_zero_normalizes_to_positive_zero_with_identical_protocol_hash():
    negative_zero_stage = CurriculumStage(
        "zero",
        -0.0,
        np.longdouble("-0.0"),
        max_episodes=1,
    )
    positive_zero_stage = CurriculumStage("zero", 0.0, 0.0, max_episodes=1)
    negative_zero = FlatOnlineProtocol(
        (negative_zero_stage,),
        (101,),
        (1009,),
        max_total_episodes=1,
    )
    positive_zero = FlatOnlineProtocol(
        (positive_zero_stage,),
        (101,),
        (1009,),
        max_total_episodes=1,
    )

    assert math.copysign(1.0, negative_zero_stage.step_height) == 1.0
    assert math.copysign(1.0, negative_zero_stage.rough_amplitude) == 1.0
    assert negative_zero_stage == positive_zero_stage
    assert negative_zero.canonical_json() == positive_zero.canonical_json()
    assert "-0.0" not in negative_zero.canonical_json()
    assert negative_zero.sha256() == positive_zero.sha256()


@pytest.mark.parametrize(
    "value",
    [10**10000, Fraction(10**10000, 3)],
    ids=["enormous-int", "enormous-fraction"],
)
def test_geometry_rejects_finite_reals_that_cannot_become_finite_python_floats(value):
    with pytest.raises(ValueError, match="finite Python float"):
        TerrainGeometry(value, 0.0, 101)
    with pytest.raises(ValueError, match="finite Python float"):
        CurriculumStage("stage", 0.0, value)


def test_source_precision_validation_rejects_negative_fraction_that_underflows_to_zero():
    tiny_negative = Fraction(-1, 10**10000)
    assert float(tiny_negative) == -0.0

    with pytest.raises(ValueError, match="nonnegative"):
        TerrainGeometry(tiny_negative, 0.0, 101)
    with pytest.raises(ValueError, match="nonnegative"):
        CurriculumStage("stage", 0.0, tiny_negative)


@pytest.mark.parametrize(
    "seed",
    [True, 1.5, "101", None, -1, EXPECTED_MAX_PROTOCOL_INTEGER + 1],
)
def test_geometry_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        TerrainGeometry(0.0, 0.0, seed)


@pytest.mark.parametrize("name", ["", "   ", None, 7])
def test_stage_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="name"):
        CurriculumStage(name, 0.0, 0.0)


@pytest.mark.parametrize(
    "max_episodes",
    [True, False, 0, -1, 1.0, "20", EXPECTED_MAX_PROTOCOL_INTEGER + 1],
)
def test_stage_rejects_invalid_episode_budget(max_episodes):
    with pytest.raises(ValueError, match="max_episodes"):
        CurriculumStage("stage", 0.0, 0.0, max_episodes=max_episodes)


def _protocol(**overrides):
    arguments = {
        "stages": EXPECTED_STAGES,
        "development_seeds": DEVELOPMENT_SEEDS,
        "sealed_seeds": SEALED_SEEDS,
    }
    arguments.update(overrides)
    return FlatOnlineProtocol(**arguments)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stages": ()}, "stages"),
        ({"stages": list(EXPECTED_STAGES)}, "stages"),
        ({"stages": (EXPECTED_STAGES[0], "bad")}, "CurriculumStage"),
        (
            {"stages": (EXPECTED_STAGES[0], replace(EXPECTED_STAGES[0], step_height=0.03))},
            "unique",
        ),
        ({"development_seeds": ()}, "development_seeds"),
        ({"development_seeds": list(DEVELOPMENT_SEEDS)}, "development_seeds"),
        ({"sealed_seeds": ()}, "sealed_seeds"),
        ({"sealed_seeds": list(SEALED_SEEDS)}, "sealed_seeds"),
        ({"development_seeds": (101, 101)}, "unique"),
        ({"sealed_seeds": (1009, 1009)}, "unique"),
        ({"development_seeds": (True,)}, "integer"),
        ({"sealed_seeds": (1.0,)}, "integer"),
        ({"development_seeds": (-1,)}, str(EXPECTED_MAX_PROTOCOL_INTEGER)),
        ({"sealed_seeds": (-1,)}, str(EXPECTED_MAX_PROTOCOL_INTEGER)),
        (
            {"development_seeds": (EXPECTED_MAX_PROTOCOL_INTEGER + 1,)},
            str(EXPECTED_MAX_PROTOCOL_INTEGER),
        ),
        (
            {"sealed_seeds": (EXPECTED_MAX_PROTOCOL_INTEGER + 1,)},
            str(EXPECTED_MAX_PROTOCOL_INTEGER),
        ),
        ({"development_seeds": (101,), "sealed_seeds": (101,)}, "disjoint"),
        ({"max_total_episodes": 99}, "stage episode budgets"),
    ],
)
def test_protocol_rejects_invalid_stage_and_seed_collections(overrides, message):
    with pytest.raises(ValueError, match=message):
        _protocol(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_total_episodes": True},
        {"max_total_episodes": 0},
        {"max_total_episodes": 100.0},
        {"max_total_episodes": EXPECTED_MAX_PROTOCOL_INTEGER + 1},
        {"max_total_steps": False},
        {"max_total_steps": -1},
        {"max_total_steps": 200_000.0},
        {"max_total_steps": EXPECTED_MAX_PROTOCOL_INTEGER + 1},
        {"max_steps_per_episode": True},
        {"max_steps_per_episode": 0},
        {"max_steps_per_episode": 2_000.0},
        {"max_steps_per_episode": EXPECTED_MAX_PROTOCOL_INTEGER + 1},
    ],
)
def test_protocol_rejects_invalid_total_budgets(overrides):
    with pytest.raises(ValueError, match="positive integer"):
        _protocol(**overrides)


def test_protocol_allows_episode_capacity_to_exceed_total_step_budget():
    protocol = _protocol(max_total_steps=1)

    assert protocol.max_steps_per_episode * protocol.max_total_episodes > protocol.max_total_steps


def test_protocol_integer_upper_bound_is_accepted_where_constraints_permit():
    stage = CurriculumStage("stage", 0.0, 0.0, max_episodes=EXPECTED_MAX_PROTOCOL_INTEGER)
    protocol = FlatOnlineProtocol(
        stages=(stage,),
        development_seeds=(0,),
        sealed_seeds=(EXPECTED_MAX_PROTOCOL_INTEGER,),
        max_total_episodes=EXPECTED_MAX_PROTOCOL_INTEGER,
        max_total_steps=EXPECTED_MAX_PROTOCOL_INTEGER,
        max_steps_per_episode=EXPECTED_MAX_PROTOCOL_INTEGER,
    )

    assert protocol.max_total_episodes == EXPECTED_MAX_PROTOCOL_INTEGER
    assert protocol.max_total_steps == EXPECTED_MAX_PROTOCOL_INTEGER
    assert protocol.max_steps_per_episode == EXPECTED_MAX_PROTOCOL_INTEGER
    assert protocol.development_seeds == (0,)
    assert protocol.sealed_seeds == (EXPECTED_MAX_PROTOCOL_INTEGER,)
    serialized = json.loads(protocol.canonical_json())
    assert serialized["stages"][0]["max_episodes"] == EXPECTED_MAX_PROTOCOL_INTEGER
    assert serialized["max_total_episodes"] == EXPECTED_MAX_PROTOCOL_INTEGER
    assert serialized["max_total_steps"] == EXPECTED_MAX_PROTOCOL_INTEGER
    assert serialized["max_steps_per_episode"] == EXPECTED_MAX_PROTOCOL_INTEGER


def test_out_of_range_seed_and_budget_errors_include_stable_bound():
    invalid = EXPECTED_MAX_PROTOCOL_INTEGER + 1

    with pytest.raises(ValueError, match=str(EXPECTED_MAX_PROTOCOL_INTEGER)):
        TerrainGeometry(0.0, 0.0, invalid)
    with pytest.raises(ValueError, match=str(EXPECTED_MAX_PROTOCOL_INTEGER)):
        CurriculumStage("stage", 0.0, 0.0, max_episodes=invalid)


@pytest.mark.parametrize(
    ("helper", "arguments"),
    [
        (sealed_geometry, (-1,)),
        (sealed_geometry, (EXPECTED_MAX_PROTOCOL_INTEGER + 1,)),
        (development_geometry, ("step_2cm", -1)),
        (development_geometry, ("step_2cm", EXPECTED_MAX_PROTOCOL_INTEGER + 1)),
    ],
)
def test_geometry_helpers_reject_out_of_range_seeds_with_stable_bound(helper, arguments):
    with pytest.raises(ValueError, match=str(EXPECTED_MAX_PROTOCOL_INTEGER)):
        helper(*arguments)


@pytest.mark.parametrize("helper", [sealed_geometry, development_geometry])
def test_geometry_helpers_reject_malformed_protocol(helper):
    arguments = (SEALED_SEEDS[0],) if helper is sealed_geometry else ("step_2cm", DEVELOPMENT_SEEDS[0])

    with pytest.raises(ValueError, match="protocol"):
        helper(*arguments, protocol="malformed")


def test_geometry_generation_is_repeatable_without_global_rng_access():
    rng_state = random.getstate()

    first_development = development_geometry("rough_4cm", DEVELOPMENT_SEEDS[-1])
    first_sealed = sealed_geometry(SEALED_SEEDS[-1])
    second_development = development_geometry("rough_4cm", DEVELOPMENT_SEEDS[-1])
    second_sealed = sealed_geometry(SEALED_SEEDS[-1])

    assert first_development == second_development
    assert first_sealed == second_sealed
    assert random.getstate() == rng_state
