"""Fixed-L2 block geometry constants and simulator-independent episode logic."""
from __future__ import annotations

import torch

from .logic import OUTCOME_NAMES, resolve_outcomes
from .rows import COLUMN_SPECS, NUM_COLUMNS, NUM_ROWS

BLOCK_SIZE = (8.0, 8.0)
BLOCK_SEED = 20_261_111
COMPLETION_M = 3.4
LAST_EDGE_M = 3.0
STAIR_EDGES_M = (1.0, 1.5, 2.0, 2.5, 3.0)
STAIR_COLUMNS = (4, 5)


def replica_rows(columns):
    """Balance each column across all ten spatial replicas, including small scenes."""
    counts = torch.bincount(columns.long(), minlength=NUM_COLUMNS)
    offsets = counts.cumsum(0) - counts
    return (torch.arange(columns.numel(), device=columns.device) - offsets[columns]) % NUM_ROWS


def block_axis(rows, columns, episode_numbers, env_ids):
    # Non-stair meshes are rotated with their spatial row. Native stairs are
    # symmetric: cycle through all four exits without changing the geometry.
    stair = (columns == 4) | (columns == 5)
    return torch.where(stair, (env_ids + episode_numbers) % 4, rows % 4).long()


def axis_vectors(axis):
    return torch.tensor(((1., 0.), (0., 1.), (-1., 0.), (0., -1.)), device=axis.device)[axis]


def block_progress(root_xy, foot_xy, origins_xy, axis):
    direction = axis_vectors(axis)
    root = ((root_xy - origins_xy) * direction).sum(-1)
    feet = ((foot_xy - origins_xy[:, None]) * direction[:, None]).sum(-1)
    return root, feet


def block_completed(root_progress, foot_progress, omni):
    finite = torch.isfinite(root_progress) & torch.isfinite(foot_progress).all(-1)
    return finite & ~omni & (root_progress >= COMPLETION_M) & (foot_progress.amin(-1) >= LAST_EDGE_M + .025)


class BlockStats:
    """Physical rows are replicas; reference roughness has its own difficulty label."""
    def __init__(self, device, rough_profile="source_l2"):
        if rough_profile not in ("source_l2", "reference_uniform"):
            raise ValueError(f"unknown block rough profile: {rough_profile}")
        self.rough_profile = rough_profile
        self.column_names = [name for name, _ in COLUMN_SPECS]
        self.column_difficulties = ["L2"] * NUM_COLUMNS
        if rough_profile == "reference_uniform":
            for column in (6, 15):
                self.column_names[column] = self.column_names[column].replace("multi_scale", "reference_uniform")
                self.column_difficulties[column] = "reference_rough"
        self.starts = torch.zeros(NUM_COLUMNS, NUM_ROWS, 4, dtype=torch.long, device=device)
        self.outcomes = torch.zeros(NUM_COLUMNS, NUM_ROWS, 4, 3, dtype=torch.long, device=device)
        self.transitions = torch.zeros(NUM_COLUMNS, 4, dtype=torch.long, device=device)
        self.foot_contact_steps = torch.zeros_like(self.transitions)
        self.nonfoot_contact_steps = torch.zeros_like(self.transitions)
        self.stair_edges_at_end = torch.zeros(2, 6, dtype=torch.long, device=device)
        self.termination_counts = {}
        self.omni_command_steps = torch.zeros(5, dtype=torch.long, device=device)

    def record_starts(self, columns, rows, axes):
        index = (columns * NUM_ROWS + rows) * 4 + axes
        self.starts += torch.bincount(index, minlength=self.starts.numel()).reshape(self.starts.shape)

    def observe(self, columns, axes, foot_contact, nonfoot_contact, omni, commands):
        index = columns * 4 + axes
        for target, mask in ((self.transitions, torch.ones_like(columns, dtype=torch.bool)),
                             (self.foot_contact_steps, foot_contact),
                             (self.nonfoot_contact_steps, nonfoot_contact)):
            target += torch.bincount(index[mask], minlength=target.numel()).reshape(target.shape)
        cmd = commands[omni]
        # Categories can overlap (e.g. lateral and backward); standing/pure
        # turn use the same 0.1 m/s and 0.1 rad/s reporting thresholds.
        linear = cmd[:, :2].norm(dim=-1)
        self.omni_command_steps += torch.stack((
            (linear < .1) & (cmd[:, 2].abs() < .1),
            (linear < .1) & (cmd[:, 2].abs() >= .1),
            cmd[:, 0] < -.1, cmd[:, 1].abs() > .1, linear >= .1,
        )).sum(-1)

    def record_endings(self, columns, rows, axes, completed, failed, timed_out, crossed_edges, terms):
        flags = torch.stack(resolve_outcomes(completed, failed, timed_out), -1)
        base = ((columns * NUM_ROWS + rows) * 4 + axes)[:, None] * 3
        index = (base + torch.arange(3, device=columns.device))[flags]
        self.outcomes += torch.bincount(index, minlength=self.outcomes.numel()).reshape(self.outcomes.shape)
        for k, column in enumerate(STAIR_COLUMNS):
            self.stair_edges_at_end[k] += torch.bincount(crossed_edges[columns == column], minlength=6)
        for name, mask in terms.items():
            if name not in self.termination_counts:
                self.termination_counts[name] = torch.zeros(NUM_COLUMNS, dtype=torch.long, device=columns.device)
            self.termination_counts[name] += torch.bincount(columns[mask], minlength=NUM_COLUMNS)

    def report(self):
        starts = self.starts.sum((1, 2)).cpu().tolist()
        ends = self.outcomes.sum((1, 2)).cpu().tolist()
        transitions = self.transitions.sum(1).cpu().tolist()
        out = {}
        for i, name in enumerate(self.column_names):
            out[f"blocks/{name}/starts"] = starts[i]
            out[f"blocks/{name}/{self.column_difficulties[i]}_transitions"] = transitions[i]
            for j, outcome in enumerate(OUTCOME_NAMES):
                out[f"blocks/{name}/{outcome}"] = ends[i][j]
        return out

    def snapshot(self):
        reference = self.rough_profile == "reference_uniform"
        return dict(schema="source_block_reference_rough_counts_v1" if reference else "source_block_l2_counts_v1",
                    physical_difficulty="L2_except_reference_uniform_rough" if reference else "L2",
                    rough_profile=self.rough_profile, physical_difficulty_by_column=self.column_difficulties,
                    rows_are="spatial_replicas", columns=self.column_names, directions=["east", "north", "west", "south"],
                    outcomes=list(OUTCOME_NAMES), starts_by_column_replica_direction=self.starts.cpu().tolist(),
                    endings_by_column_replica_direction_outcome=self.outcomes.cpu().tolist(),
                    transitions_by_column_direction=self.transitions.cpu().tolist(),
                    foot_contact_steps_by_column_direction=self.foot_contact_steps.cpu().tolist(),
                    nonfoot_contact_steps_by_column_direction=self.nonfoot_contact_steps.cpu().tolist(),
                    stair_edges_at_end=self.stair_edges_at_end.cpu().tolist(),
                    stair_edge_metric="maximum number of edges passed by all four feet at observed contacts",
                    termination_counts={k: v.cpu().tolist() for k, v in self.termination_counts.items()},
                    omni_command_categories=["standing", "pure_turn", "backward", "lateral", "translation"],
                    omni_command_steps=self.omni_command_steps.cpu().tolist(), metrics=self.report())
