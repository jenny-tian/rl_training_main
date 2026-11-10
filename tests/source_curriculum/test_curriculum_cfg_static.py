"""Isaac-free static checks of the Source curriculum cfg, registration and runner (source text + AST only)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3"
ENV_CFG = C / "source_curriculum_env_cfg.py"
REGISTRY = C / "__init__.py"
RUNNER = C / "agents/rsl_rl_ppo_cfg.py"


def _source(path):
    text = path.read_text()
    ast.parse(text)
    return text


def test_env_cfg_pins_grid_seed_and_termination_values():
    text = _source(ENV_CFG)
    for needle in ("seed=20_261_109", "num_rows=NUM_ROWS", "max_init_terrain_level=4", '"threshold": 5.0',
                   '"consecutive_steps": 3', "make_consistent"):
        assert needle in text, needle


def _registered_ids(path):
    ids = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "register":
            ids += [kw.value.value for kw in node.keywords if kw.arg == "id"]
    return ids


def test_task_is_registered():
    assert "SourceCurriculum-Deeprobotics-Lite3-v0" in _registered_ids(REGISTRY)


def test_runner_sets_learning_rate_and_experiment_name():
    text = _source(RUNNER)
    assert "class DeeproboticsLite3SourceCurriculumPPORunnerCfg" in text
    assert "3.844e-4" in text
    assert "deeprobotics_lite3_source_curriculum" in text
