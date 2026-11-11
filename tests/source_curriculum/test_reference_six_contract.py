"""Guard the user's constraint: replace terrain, retain reference training."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/tools"))
from reference_six_train import configuration_audit


def configurations():
    data = {"scene": {"terrain": {"class_type": "native", "max_init_terrain_level": 5,
             "terrain_generator": {"num_rows": 10, "num_cols": 20, "sub_terrains": {"rough": .4}}}},
            "rewards": {"stand_still": -.5}, "commands": {"vx": [-1.5, 1.5]},
            "observations": {"critic": 393}, "terminations": {"illegal_contact": None}}
    candidate = copy.deepcopy(data)
    candidate["scene"]["terrain"]["class_type"] = "material_aware"
    candidate["scene"]["terrain"]["terrain_generator"]["sub_terrains"] = {"six_families": 1.}
    agent = SimpleNamespace(resume=False, algorithm=SimpleNamespace(learning_rate=.001),
                            max_iterations=10000, to_dict=lambda: {"learning_rate": .001})
    return data, candidate, agent


def config(data):
    return SimpleNamespace(to_dict=lambda: copy.deepcopy(data))


def test_terrain_only_change_is_allowed():
    before, after, agent = configurations()
    assert configuration_audit(config(before), config(after), agent)["non_terrain_environment_exact"]


@pytest.mark.parametrize("section", ["rewards", "commands", "observations", "terminations"])
def test_non_terrain_changes_are_rejected(section):
    before, after, agent = configurations()
    after[section] = {"unexpected": True}
    with pytest.raises(ValueError, match="non-terrain"):
        configuration_audit(config(before), config(after), agent)


def test_fixed_difficulty_cannot_replace_native_curriculum():
    before, after, agent = configurations()
    after["scene"]["terrain"]["terrain_generator"]["num_rows"] = 1
    with pytest.raises(ValueError, match="layout or curriculum"):
        configuration_audit(config(before), config(after), agent)


def test_initial_row_limit_is_preserved():
    before, after, agent = configurations()
    after["scene"]["terrain"]["max_init_terrain_level"] = 9
    with pytest.raises(ValueError, match="importer setting"):
        configuration_audit(config(before), config(after), agent)


def test_old_checkpoint_resume_is_rejected():
    before, after, agent = configurations()
    agent.resume = True
    with pytest.raises(ValueError, match="PPO defaults"):
        configuration_audit(config(before), config(after), agent)
