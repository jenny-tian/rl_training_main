"""Load the reference Lite3 implementation without unrelated humanoid assets.

The reference checkout stays read-only. Only the terrain implementation is
provided by the private project; environment, rewards and PPO come from the
reference files. Call after AppLauncher, in a fresh Python process.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path("/home/tjx/rl_training")
REFERENCE_COMMIT = "127b4ff1081b1c7a2f08842298d06dcc5e9460e4"
PREFIX = "rl_training.tasks.manager_based.locomotion.velocity.config.quadruped.deeprobotics_lite3"


def namespace(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)
    return module


def load_reference():
    if any(k == "rl_training" or k.startswith("rl_training.") for k in sys.modules):
        raise RuntimeError("reference loading requires a fresh rl_training namespace")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REFERENCE, text=True).strip()
    if commit != REFERENCE_COMMIT:
        raise RuntimeError("reference revision changed; repeat the configuration audit first")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "source/rl_training"], cwd=REFERENCE,
                   check=True, stdout=subprocess.DEVNULL)
    package = REFERENCE / "source/rl_training/rl_training"
    for path in ("", "tasks", "tasks/manager_based", "tasks/manager_based/locomotion",
                 "tasks/manager_based/locomotion/velocity", "tasks/manager_based/locomotion/velocity/config",
                 "tasks/manager_based/locomotion/velocity/config/quadruped",
                 "tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3"):
        namespace("rl_training" + ("." + path.replace("/", ".") if path else ""), package / path)
    # The exact Lite3 assignment is evaluated; the same file's unrelated DR02
    # conversion would otherwise require humanoid assets absent in this checkout.
    asset_path = package / "assets/deeprobotics.py"
    tree = ast.parse(asset_path.read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "DEEPROBOTICS_LITE3_CFG" for t in n.targets))
    asset = types.ModuleType("rl_training.assets.deeprobotics")
    asset.__file__ = str(asset_path)
    exec("import isaaclab.sim as sim_utils\nfrom isaaclab.actuators import DelayedPDActuatorCfg\n"
         "from isaaclab.assets.articulation import ArticulationCfg\n"
         "from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR\n", asset.__dict__)
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(asset_path), "exec"), asset.__dict__)
    sys.modules[asset.__name__] = asset
    sys.modules["rl_training.assets"].deeprobotics = asset
    env = importlib.import_module(PREFIX + ".rough_env_cfg").DeeproboticsLite3RoughEnvCfg()
    agent = importlib.import_module(PREFIX + ".agents.rsl_rl_ppo_cfg").DeeproboticsLite3RoughPPORunnerCfg()
    # Reuse geometry and material-binding helpers only. No main-project
    # environment, reset scheduler, reward or training wrapper is imported.
    main_package = ROOT / "source/rl_training/rl_training"
    namespace("rl_training.source_scan", main_package / "source_scan")
    namespace("lite3_terrain_reuse", main_package / "tasks/manager_based/locomotion/velocity/terrains")
    return env, agent


def source_manifest():
    paths = set()
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("rl_training", "lite3_terrain_reuse", "reference_lite3", "reference_six")):
            path = getattr(module, "__file__", None)
            if path and Path(path).is_file():
                paths.add(Path(path).resolve())
    return [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)]


def resolve_lite3_asset(cfg):
    """Use the existing local Lite3 asset when the reference submodule is empty."""
    requested = Path(cfg.scene.robot.spawn.usd_path)
    if requested.is_file():
        return dict(changed=False, requested=str(requested), resolved=str(requested))
    relative = Path("source/rl_training/rl_training/assets/deeprobotics.py")

    def lite3_assignment(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == "DEEPROBOTICS_LITE3_CFG" for t in n.targets)))

    if lite3_assignment(REFERENCE / relative) != lite3_assignment(ROOT / relative):
        raise RuntimeError("reference and local Lite3 robot configurations differ")
    resolved = ROOT / "deep_robotics_model/Lite3/Lite3_usd/Lite3.usd"
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    cfg.scene.robot.spawn.usd_path = str(resolved)
    directory = resolved.parent
    asset_repo = ROOT / "deep_robotics_model"
    ref_asset_commit = subprocess.check_output(["git", "rev-parse", "HEAD:deep_robotics_model"],
                                               cwd=REFERENCE, text=True).strip()
    files = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(directory.rglob("*")) if p.is_file()]
    for row in files:
        path = Path(row["path"])
        original = "Lite3/usd/" + str(path.relative_to(directory))
        blob = subprocess.check_output(["git", "rev-parse", f"{ref_asset_commit}:{original}"],
                                       cwd=asset_repo, text=True).strip()
        actual = subprocess.check_output(["git", "hash-object", str(path)], cwd=asset_repo, text=True).strip()
        if blob != actual:
            raise RuntimeError(f"local robot asset differs from the reference gitlink: {path}")
        row.update(reference_path=original, git_blob=blob, reference_blob_exact=True)
    return dict(changed=True, reason="reference robot asset submodule is not populated", requested=str(requested),
                resolved=str(resolved), lite3_asset_configuration_exact=True,
                reference_asset_commit=ref_asset_commit, all_asset_blobs_exact=True, files=files)
