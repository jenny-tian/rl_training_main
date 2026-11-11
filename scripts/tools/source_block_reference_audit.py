"""Read reference Lite3 configs in an isolated process; never change that checkout."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import types

REFERENCE = Path("/home/tjx/rl_training")
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "logs/stage1/block_b4/reference_audit"


def audit():
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    if any(name == "rl_training" or name.startswith("rl_training.") for name in sys.modules):
        raise RuntimeError("reference audit requires a fresh package namespace")
    package_root = REFERENCE / "source/rl_training/rl_training"
    # Avoid project-wide eager registration, which imports unrelated humanoid
    # assets absent from this reference checkout. Execute the real Lite3 config
    # classes and its exact asset assignment, without running DR02 conversion.
    paths = ("", "tasks", "tasks/manager_based", "tasks/manager_based/locomotion",
        "tasks/manager_based/locomotion/velocity", "tasks/manager_based/locomotion/velocity/config",
        "tasks/manager_based/locomotion/velocity/config/quadruped",
        "tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3")
    for path in paths:
        name = "rl_training" + ("." + path.replace("/", ".") if path else "")
        module = types.ModuleType(name)
        module.__path__ = [str(package_root / path)]
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)
    asset_path = package_root / "assets/deeprobotics.py"
    tree = ast.parse(asset_path.read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "DEEPROBOTICS_LITE3_CFG" for t in n.targets))
    asset = types.ModuleType("rl_training.assets.deeprobotics")
    exec("import isaaclab.sim as sim_utils\nfrom isaaclab.actuators import DelayedPDActuatorCfg\n"
         "from isaaclab.assets.articulation import ArticulationCfg\n"
         "from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR\n", asset.__dict__)
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(asset_path), "exec"), asset.__dict__)
    sys.modules[asset.__name__] = asset
    sys.modules["rl_training.assets"].deeprobotics = asset
    prefix = "rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3"
    rough = importlib.import_module(prefix + ".rough_env_cfg").DeeproboticsLite3RoughEnvCfg()
    agent = importlib.import_module(prefix + ".agents.rsl_rl_ppo_cfg").DeeproboticsLite3RoughPPORunnerCfg()
    from isaaclab.sensors.ray_caster.patterns import grid_pattern
    rays, _ = grid_pattern(rough.scene.height_scanner.pattern_cfg, device="cpu")
    env_dict = rough.to_dict()
    # Both observation groups use only scalar/vector base signals, joints,
    # actions and (critic only) this actual grid. No environment is simulated.
    dimensions = dict(policy=45, critic=48 + len(rays), height_scan=len(rays),
                      method="config-derived sizes; grid_pattern executed on CPU, no simulated observations")
    from inspect import getsourcefile
    reward = rough.rewards.stand_still
    sources = []
    names = ("assets/deeprobotics.py", "tasks/manager_based/locomotion/velocity/velocity_env_cfg.py",
        "tasks/manager_based/locomotion/velocity/mdp/rewards.py",
        "tasks/manager_based/locomotion/velocity/mdp/commands.py",
        "tasks/manager_based/locomotion/velocity/mdp/curriculums.py",
        "tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py",
        "tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/agents/rsl_rl_ppo_cfg.py")
    for name in names:
        path = package_root / name
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(status="passed", reference=str(REFERENCE),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REFERENCE, text=True).strip(),
        source_status=subprocess.check_output(["git", "status", "--short"], cwd=REFERENCE, text=True),
        method="Real reference config classes and exact Lite3 asset assignment; package-wide humanoid registration skipped",
        limitation="Configuration audit only. Reference checkout has no logs/rsl_rl; no claimed trained stair performance.",
        observation_dimensions=dimensions, configuration=env_dict, agent=agent.to_dict(),
        active_rewards={k: v for k, v in env_dict["rewards"].items() if v is not None and v["weight"] != 0},
        stand_still_implementation=dict(module=reward.func.__module__, name=reward.func.__name__,
            path=getsourcefile(reward.func), weight=reward.weight, params=reward.params),
        source_files=sources)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")
    print(json.dumps(dict(status="passed", dimensions=dimensions,
                         standing_function=result["stand_still_implementation"]), default=str), flush=True)
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if not args.child:
        from source_block_runtime import run
        return run([sys.executable, str(Path(__file__).resolve()), "--child"], gpu=0, output=OUTPUT, timeout=240)
    try:
        audit()
    except Exception:
        traceback.print_exc()
        (OUTPUT / "error.json").write_text(json.dumps(dict(error=traceback.format_exc()), indent=2))
        sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    sys.exit(main())
