"""Execute the installed and local reward bodies on identical synthetic states."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from source_block_b3_protocol import ROOT


def function(path, name, namespace):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name], dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  line=node.lineno, source=ast.get_source_segment(source, node))


def reproduce():
    core = Path("/home/tjx/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py")
    installed = Path("/home/tjx/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py")
    local = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py"
    entity = lambda name: SimpleNamespace(name=name, joint_ids=slice(None))
    namespace = dict(torch=torch, SceneEntityCfg=entity, ManagerBasedRLEnv=object, Articulation=object,
                     RigidObject=object)
    deviation, dev_source = function(core, "joint_deviation_l1", namespace)
    namespace["mdp"] = SimpleNamespace(joint_deviation_l1=deviation)
    old, old_source = function(installed, "stand_still_joint_deviation_l1", namespace)
    candidate, candidate_source = function(local, "stand_still_without_cmd", namespace)
    commands = torch.tensor([[0., 0., 0.], [0., 0., .5], [0., 0., -.5], [0., 0., .15],
                             [0., 0., .05], [.25, 0., 0.], [0., .3, 0.]])
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.full((len(commands), 12), .1),
                                                  default_joint_pos=torch.zeros(len(commands), 12)))
    env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_command=lambda _: commands))
    params = dict(command_name="base_velocity", command_threshold=.1, asset_cfg=entity("robot"))
    original, corrected = old(env, **params), candidate(env, **params)
    assert torch.allclose(original[:5], torch.full((5,), 1.2))
    assert torch.allclose(corrected, torch.tensor([1.2, 0., 0., 0., 1.2, 0., 0.]))
    return dict(status="passed", method="AST-extracted actual function bodies; mock state, no simulator or training",
        sources=[dev_source, old_source, candidate_source], weight=-.2, command_threshold=.1,
        joint_deviation_l1=1.2, commands=commands.tolist(), original_weighted_rate=(-.2*original).tolist(),
        candidate_weighted_rate=(-.2*corrected).tolist(),
        conclusion="Current stand_still penalizes pure turning. Existing local all-command gate suppresses it at abs(wz)>=0.1.",
        limitation="This proves reward gating, not that changing the reward will repair a learned policy.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(reproduce(), indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.write_text(result)
    print(result)
