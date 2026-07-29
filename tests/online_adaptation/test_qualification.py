import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.reinforcement_learning.online_adaptation import qualify_base_policy
from rl_training.online_adaptation.qualification import build_qualification_report


ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION_SOURCE = (
    ROOT / "scripts/reinforcement_learning/online_adaptation/qualify_base_policy.py"
).read_text(encoding="utf-8")
QUALIFICATION_TREE = ast.parse(QUALIFICATION_SOURCE)


def _function_node(name: str) -> ast.FunctionDef:
    return next(
        node for node in QUALIFICATION_TREE.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _expression_path(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_path(node.value)
        return f"{parent}.{node.attr}" if parent is not None else None
    return None


def _statement_calls(statement: ast.stmt, function_path: str) -> bool:
    return any(
        isinstance(node, ast.Call) and _expression_path(node.func) == function_path
        for node in ast.walk(statement)
    )


def _validation_index(function: ast.FunctionDef) -> int:
    matches = [
        (index, statement)
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.UnaryOp)
        and isinstance(statement.test.op, ast.Not)
        and isinstance(statement.test.operand, ast.Call)
        and _expression_path(statement.test.operand.func) == "is_strict_flat_baseline_task"
    ]
    assert len(matches) == 1
    index, validation = matches[0]
    task_check = validation.test.operand
    assert len(task_check.args) == 1
    assert _expression_path(task_check.args[0]) == "args.task"
    assert len(validation.body) == 1
    assert isinstance(validation.body[0], ast.Raise)
    exception = validation.body[0].exc
    assert isinstance(exception, ast.Call)
    assert _expression_path(exception.func) == "ValueError"
    return index


def test_qualification_report_requires_zero_falls_and_error_gate():
    records = [
        {"seed": 7, "falls": 0, "steps": 1000, "mean_velocity_error": 0.30},
        {"seed": 11, "falls": 0, "steps": 1000, "mean_velocity_error": 0.34},
    ]

    report = build_qualification_report("source-task", "base.pt", records, 0.35)

    assert report["aggregate"] == {
        "falls": 0,
        "mean_velocity_error": 0.32,
        "max_mean_velocity_error": 0.35,
        "passed": True,
    }


def test_qualification_report_fails_if_any_seed_falls():
    records = [
        {"seed": 7, "falls": 1, "steps": 91, "mean_velocity_error": 0.20},
        {"seed": 11, "falls": 0, "steps": 1000, "mean_velocity_error": 0.20},
    ]

    report = build_qualification_report("source-task", "base.pt", records, 0.35)

    assert report["aggregate"]["falls"] == 1
    assert report["aggregate"]["passed"] is False


@pytest.mark.parametrize(
    ("function_name", "side_effect_calls"),
    [
        ("main", ("_run_multi_seed", "_run_single_seed")),
        ("_run_multi_seed", ("subprocess.run",)),
        ("_run_single_seed", ("AppLauncher",)),
    ],
)
def test_qualification_validation_raises_before_side_effects(function_name, side_effect_calls):
    function = _function_node(function_name)
    validation_index = _validation_index(function)
    side_effect_indices = [
        index
        for index, statement in enumerate(function.body)
        if any(_statement_calls(statement, call) for call in side_effect_calls)
    ]

    assert side_effect_indices
    assert validation_index < min(side_effect_indices)


def test_invalid_multi_seed_task_does_not_spawn_worker(monkeypatch):
    run = Mock(side_effect=AssertionError("subprocess.run must not be called"))
    monkeypatch.setattr(qualify_base_policy.subprocess, "run", run)

    with pytest.raises(ValueError, match="Flat-Deeprobotics-Lite3-v0"):
        qualify_base_policy._run_multi_seed(SimpleNamespace(task="OnlineAdapt-Deeprobotics-Lite3-v0"))

    run.assert_not_called()


def test_invalid_single_seed_task_does_not_launch_app(monkeypatch):
    launcher = Mock(side_effect=AssertionError("AppLauncher must not be called"))
    monkeypatch.setattr(qualify_base_policy, "AppLauncher", launcher)

    with pytest.raises(ValueError, match="Flat-Deeprobotics-Lite3-v0"):
        qualify_base_policy._run_single_seed(
            SimpleNamespace(task="HeldoutBoxes-Deeprobotics-Lite3-v0", seeds=[7])
        )

    launcher.assert_not_called()


def test_invalid_main_task_does_not_dispatch_qualification(monkeypatch):
    parser = Mock()
    parser.parse_args.return_value = SimpleNamespace(task="Rough-Deeprobotics-Lite3-v0", seeds=[7, 11])
    multi_seed = Mock(side_effect=AssertionError("multi-seed qualification must not be called"))
    single_seed = Mock(side_effect=AssertionError("single-seed qualification must not be called"))
    monkeypatch.setattr(qualify_base_policy, "build_qualify_parser", Mock(return_value=parser))
    monkeypatch.setattr(qualify_base_policy, "_run_multi_seed", multi_seed)
    monkeypatch.setattr(qualify_base_policy, "_run_single_seed", single_seed)

    with pytest.raises(ValueError, match="Flat-Deeprobotics-Lite3-v0"):
        qualify_base_policy.main()

    multi_seed.assert_not_called()
    single_seed.assert_not_called()


def test_single_seed_qualification_guards_parsed_terrain_before_gym_make():
    worker = _function_node("_run_single_seed")
    try_body = next(statement for statement in worker.body if isinstance(statement, ast.Try)).body
    parse_index = next(
        index
        for index, statement in enumerate(try_body)
        if _statement_calls(statement, "parse_env_cfg")
    )
    make_index = next(
        index
        for index, statement in enumerate(try_body)
        if _statement_calls(statement, "gym.make")
    )

    guard_statement = try_body[parse_index + 1]
    assert isinstance(guard_statement, ast.Expr)
    guard_call = guard_statement.value
    assert isinstance(guard_call, ast.Call)
    assert _expression_path(guard_call.func) == "assert_plane_only_terrain"
    assert len(guard_call.args) == 1
    assert _expression_path(guard_call.args[0]) == "env_cfg.scene.terrain"
    assert guard_call.keywords == []
    assert parse_index + 1 < make_index
