"""Exercise outcome ordering at the actual reset boundary without a simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from rl_training.source_curriculum.block import BlockStats, block_axis


def test_failure_stays_l2_and_is_recorded_before_replica_changes():
    root = Path(__file__).resolve().parents[2]
    path = root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/source_block.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "SourceBlockResets")
    class Term:
        def __init__(self, *_):
            pass
    ns = dict(torch=torch, ManagerTermBase=Term, NUM_ROWS=10, block_axis=block_axis)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    origin_grid = torch.arange(10 * 20 * 3).reshape(10, 20, 3).float()
    terrain = NS(terrain_levels=torch.tensor([9, 0]), terrain_types=torch.tensor([4, 5]),
                 terrain_origins=origin_grid, env_origins=origin_grid[[9, 0], [4, 5]].clone())
    stats = BlockStats("cpu")
    state = NS(stats=stats, axes=torch.tensor([0, 1]), episode_numbers=torch.tensor([0, 0]),
               support_progress=torch.zeros(2, 4), crossed_edges=torch.tensor([2, 4]))
    tm = NS(dones=torch.tensor([True, True]), terminated=torch.tensor([True, False]),
            time_outs=torch.tensor([True, True]), active_terms=["source_curriculum_completion", "safety"])
    tm.get_term = lambda name: torch.tensor([True, False])
    env = NS(source_block_state=state, scene=NS(terrain=terrain), termination_manager=tm,
             episode_length_buf=torch.tensor([15, 10]), device="cpu")
    term = ns["SourceBlockResets"](None, env)
    term(env, torch.arange(2))
    assert terrain.terrain_levels.tolist() == [0, 1]  # replicas cycle; both still L2
    assert torch.equal(terrain.env_origins, origin_grid[[0, 1], [4, 5]])
    assert stats.outcomes[4, 9, 0, 1] == 1  # original replica, failure beats completion/timeout
    assert stats.outcomes[5, 0, 1, 2] == 1
    assert stats.starts[4, 0, state.axes[0]] == 1
    assert stats.starts[5, 1, state.axes[1]] == 1
    assert state.crossed_edges.tolist() == [0, 0]
    assert torch.isneginf(state.support_progress).all()


def test_nonfinite_training_lines_stop_supervisor():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/tools"))
    from source_block_runtime import fatal_line
    assert fatal_line("Mean value function loss: nan")
    assert fatal_line("Contacts have been dropped")
    assert not fatal_line("Mean value function loss: 0.15")
