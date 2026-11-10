"""Stage-1 specifications and offline CLI/runner checks; no simulator is launched."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rl_training.ea_lpl.headroom import CONFIRM_NAMESPACE, GRID_NAMESPACE
from rl_training.ea_lpl.specs import EQUAL8, P0_TABLE_SEED, p0_specs, spec_actor, stage1_specs, validate_spec

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3"


def policies(count):
    return {f"policy{i}": {"path": f"/actors/policy{i}.pt", "sha256": f"{i:064x}"} for i in range(count)}


@pytest.mark.parametrize("kind,count,total,levels,prefix,offset", [
    ("qualify", 3, 24, ("L1", "L2"), "s1q", 0),
    ("preview", 2, 8, ("L3",), "s1p", 0),
    ("extend", 2, 16, ("L1", "L2"), "s1x", 480),
])
def test_stage1_spec_counts_pairing_and_actor_identity(kind, count, total, levels, prefix, offset):
    actors = policies(count)
    specs = stage1_specs(actors, kind=kind)
    assert len(specs) == total
    assert {spec["label"] for spec in specs} == {
        f"{prefix}_{name}_{level.lower()}_{k}" for name in actors for level in levels for k in range(4)}
    groups = defaultdict(list)
    for spec in specs:
        assert validate_spec(spec) is spec
        name = spec["label"].split("_")[1]
        assert spec["actor"] == actors[name] and spec["actor"] is not actors[name]
        assert spec_actor(spec, default=None) == (actors[name]["path"], actors[name]["sha256"])
        assert spec["columns"] == list(EQUAL8)
        assert spec["namespace"] == 4 and spec["reset_table_seed"] == P0_TABLE_SEED
        assert spec["heading_hold"] and spec["codes"] is None
        assert (spec["episodes_per_env"], spec["max_episode_steps"], spec["wall_limit_s"]) == (1, 2000, 900)
        groups[name, spec["level"]].append(spec["slots"])
    for batches in groups.values():
        assert batches == [list(range(offset + 120 * k, offset + 120 * (k + 1))) for k in range(4)]
        assert len(set(slot for batch in batches for slot in batch)) == 480
    assert 4 not in {spec["namespace"] for spec in p0_specs()} | {GRID_NAMESPACE, CONFIRM_NAMESPACE}


def test_extend_uses_fresh_slots_after_qualification():
    actor = policies(1)
    qualify = stage1_specs(actor, kind="qualify")
    extend = stage1_specs(actor, kind="extend")
    assert {slot for spec in qualify for slot in spec["slots"]}.isdisjoint(
        slot for spec in extend for slot in spec["slots"])


def test_stage1_unknown_kind_and_default_actor():
    with pytest.raises(ValueError, match="unknown stage-1 spec kind"):
        stage1_specs(policies(1), kind="unknown")
    default = (Path("/registered/actor.pt"), "a" * 64)
    assert spec_actor(p0_specs()[0], default=default) is default


@pytest.mark.parametrize("actor", [
    None, [], "actor.pt", {}, {"path": "/actor.pt"},
    {"path": "relative.pt", "sha256": "a" * 64},
    {"path": "/actor.pt", "sha256": "A" * 64},
    {"path": "/actor.pt", "sha256": "a" * 63},
    {"path": "/actor.pt", "sha256": "a" * 64 + "\n"},
    {"path": "/actor.pt", "sha256": None},
    {"path": Path("/actor.pt"), "sha256": "a" * 64},
    {"path": "/actor.pt", "sha256": "a" * 64, "extra": True},
])
def test_stage1_malformed_actor_is_a_value_error(actor):
    with pytest.raises(ValueError, match="actor must define"):
        validate_spec(dict(p0_specs()[1], actor=actor))
    with pytest.raises(ValueError, match="actor must define"):
        stage1_specs({"bad": actor}, kind="qualify")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_spec_wall_deadline_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="wall_limit_s"):
        validate_spec(dict(p0_specs()[1], wall_limit_s=value))


def test_stage1_cli_writes_hashed_actor_specs_and_list(tmp_path):
    actors = [tmp_path / "first.pt", tmp_path / "second.pt"]
    for i, actor in enumerate(actors):
        actor.write_bytes(f"fake checkpoint {i}".encode())
    folder = tmp_path / "specs"
    subprocess.run([sys.executable, str(ROOT / "scripts/tools/stage1_make_specs.py"),
                    "--policy", f"first={actors[0]}", "--policy", f"second={actors[1]}",
                    "--kind", "preview", "--out-dir", str(folder)], check=True, capture_output=True, text=True)
    paths = (folder / "specs.txt").read_text().splitlines()
    assert len(paths) == 8 and len(set(paths)) == 8
    assert {Path(path) for path in paths} == set(folder.glob("*.json"))
    for path in paths:
        spec = validate_spec(json.loads(Path(path).read_text()))
        actor = actors[0] if "_first_" in spec["label"] else actors[1]
        assert spec["actor"] == {"path": str(actor), "sha256": hashlib.sha256(actor.read_bytes()).hexdigest()}


def test_l3_source_terrain_factory_and_evaluator_allow_list():
    path = CONFIG / "source_terrain_env_cfg.py"
    nodes = [node for node in ast.parse(path.read_text()).body
             if (isinstance(node, ast.FunctionDef) and node.name == "_source_sub_terrains")
             or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
                 "SOURCE_TERRAIN_PROPORTIONS", "_LEVEL_CANDIDATES"} for t in node.targets))]
    namespace = {"SourceTerrainTrackCfg": lambda **kwargs: SimpleNamespace(**kwargs)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    assert "L3" in namespace["_LEVEL_CANDIDATES"]
    terrains = namespace["_source_sub_terrains"]("L3")
    assert len(terrains) == 15 and terrains["stairs_ascending"].step_height_m == 0.05
    tree = ast.parse((CONFIG / "ea_lpl_env_cfg.py").read_text())
    allowed = next(node.value for node in ast.walk(tree) if isinstance(node, ast.AnnAssign)
                   and isinstance(node.target, ast.Name) and node.target.id == "source_allowed_levels")
    assert ast.literal_eval(allowed) == ("L1", "L2", "L3")


@pytest.mark.parametrize("override", [False, True])
def test_runner_supervisor_uses_selected_actor_and_records_manifest(tmp_path, monkeypatch, override):
    from scripts.tools import ea_lpl_run as runner
    actor = tmp_path / "actor.pt"
    actor.write_bytes(b"selected actor")
    sha = hashlib.sha256(actor.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "OLD", actor)
    monkeypatch.setattr(runner, "ACTOR_SHA256", sha)
    spec = p0_specs()[1]
    if override:
        spec["actor"] = {"path": str(actor), "sha256": sha}
        monkeypatch.setattr(runner, "OLD", tmp_path / "unused.pt")
        monkeypatch.setattr(runner, "ACTOR_SHA256", "0" * 64)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda cmd, **kw: "" if "status" in cmd else "revision")
    monkeypatch.setattr(runner, "choose_gpu", lambda request: 1)
    monkeypatch.setattr(runner, "gpu_free_mib", lambda: ({1: 20000}, ""))
    monkeypatch.setattr(runner, "group_exists", lambda pid: False)
    folder = tmp_path / "run"

    def fake_process(*args, **kwargs):
        (folder / "result.json").write_text(json.dumps({"status": "passed"}))
        return SimpleNamespace(pid=123, poll=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_process)
    assert runner.supervise(spec_path, folder, "auto") == 0
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["actor_path"] == str(actor) and manifest["actor_sha256"] == sha
    actor.write_bytes(b"changed")
    with pytest.raises(ValueError, match="base actor checkpoint changed"):
        runner.supervise(spec_path, tmp_path / "rejected", "auto")


@pytest.mark.parametrize("override", [False, True])
def test_runner_child_passes_selected_actor_identity_to_loader(tmp_path, monkeypatch, override):
    from scripts.tools import ea_lpl_run as runner
    from rl_training.source_scan import frozen_actor
    spec = p0_specs()[1]
    if override:
        spec["actor"] = {"path": "/candidate/actor.pt", "sha256": "a" * 64}
    expected = spec_actor(spec, default=(runner.OLD, runner.ACTOR_SHA256))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    app = SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=lambda **kw: SimpleNamespace(app=app)))
    for name in ("enable", "disable", "cancel_dump_traceback_later", "dump_traceback_later"):
        monkeypatch.setattr(runner.faulthandler, name, lambda *a, **kw: None)
    wrapped, actor = object(), object()
    monkeypatch.setattr(runner, "make_ea_environment", lambda spec: wrapped)
    monkeypatch.setattr(runner, "close_comparison_environment", lambda env: None)
    loaded = []

    def load(path, *, device, expected_sha256):
        loaded.append((path, expected_sha256, device))
        return actor

    def collect(env, selected_actor, *args):
        assert env is wrapped and selected_actor is actor
        return {"collected": True}

    monkeypatch.setattr(frozen_actor, "load_frozen_source_actor", load)
    monkeypatch.setattr(runner, "collect_fixed", collect)
    assert runner.child(spec_path, tmp_path) == 0
    assert loaded == [(expected[0], expected[1], "cuda:0")]
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "passed"
