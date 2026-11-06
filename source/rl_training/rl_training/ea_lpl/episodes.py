"""Split per-step recordings into episodes using terminal events (terminal sample included)."""
from __future__ import annotations

import numpy as np

SAFETY_TERMS = {"source_training_safety", "illegal_contact", "bad_orientation_2"}
BOUNDS_TERM = "source_terrain_tile_bounds"
TIMEOUT_TERM = "source_training_timeout"


def episode_records(arrays, events, assignments, *, episodes_per_env=1, control_dt=0.02):
    done = np.asarray(arrays["done"], dtype=bool)
    state = np.asarray(arrays["state"], dtype=np.float64)
    rewards = np.asarray(arrays["rewards"], dtype=np.float64)
    actions = np.asarray(arrays["actions"], dtype=np.float64)
    columns = np.asarray(arrays["columns"])
    steps, n = done.shape
    if state.shape[:2] != (steps, n) or len(assignments) != n:
        raise ValueError("arrays and assignments have inconsistent dimensions")
    by_env = {}
    for event in events:
        by_env.setdefault(event["env_id"], []).append(event)
    records = []
    for env_id in range(n):
        ends = sorted(by_env.get(env_id, []), key=lambda e: e["control_step"])
        if [e["control_step"] for e in ends] != (np.flatnonzero(done[:, env_id]) + 1).tolist():
            raise ValueError(f"terminal events and done flags disagree for env {env_id}")
        if len(ends) < episodes_per_env:
            raise ValueError(f"env {env_id} finished {len(ends)} of {episodes_per_env} episodes")
        column = int(assignments[env_id]["column"])
        start = 0
        for index, event in enumerate(ends[:episodes_per_env]):
            end = event["control_step"]
            if event["episode_step"] != end - start:
                raise ValueError("episode length differs from the terminal snapshot")
            window = slice(start, end)
            if not (columns[window, env_id] == column).all():
                raise ValueError("terrain column changed inside an episode")
            values = state[window, env_id]
            reasons = list(event["terms"])
            completed = bool(event["completed"])
            if completed and (event["hold"] < 50 or values[-1, 5] < 9.0):
                raise ValueError("completion lacks progress and stable-hold evidence")
            records.append(dict(
                env_id=env_id, episode_index=index, column=column, terrain=assignments[env_id]["subterrain"],
                steps=end - start, seconds=(end - start) * control_dt, completed=completed, hold=event["hold"],
                termination_reasons=reasons, safety_ending=bool(SAFETY_TERMS.intersection(reasons)),
                bounds_ending=BOUNDS_TERM in reasons, timeout=TIMEOUT_TERM in reasons,
                forward_progress_m=float(values[-1, 5]), terminal_lateral_m=float(values[-1, 9]),
                lateral_peak_m=float(np.abs(values[:, 9]).max()), terminal_yaw_deg=float(np.degrees(values[-1, 12])),
                velocity_rms_error_mps=float(np.sqrt(values[:, 3].mean())), vx_mps=float(values[:, 0].mean()),
                reward_per_control=float(rewards[window, env_id].sum(-1).mean() * control_dt),
                projected_fraction=float(actions[window, env_id, 2].mean())))
            start = end
    return records
