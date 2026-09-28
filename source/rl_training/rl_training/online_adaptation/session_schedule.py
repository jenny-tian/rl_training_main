"""Deterministic development trial scheduling, including paid frozen probes."""


def build_schedule(*, trial_steps, evaluation_trials, retention_trials=5, retention_steps=1000,
                   probe_milestones=(10000, 20000, 40000), budget_steps=40000):
    probes = []
    markers = [marker for marker in probe_milestones if marker <= budget_steps]
    for index, marker in enumerate(markers):
        # Finish each probe group near its marker. The final group is terrain only.
        flat_steps = retention_steps if index < len(markers) - 1 else 0
        start = marker - trial_steps - flat_steps
        if start < 0 or (probes and start < probes[-1]["at"] + probes[-1]["max_steps"]):
            raise ValueError("probe milestones overlap or leave no collection room")
        probes.append({"at": start, "phase": "selection", "max_steps": trial_steps})
        if flat_steps:
            probes.append({"at": start + trial_steps, "phase": "flat_selection", "max_steps": flat_steps})
    return {"evaluation_trials": evaluation_trials, "trial_steps": trial_steps,
            "retention_trials": retention_trials, "retention_steps": retention_steps,
            "short_fall_steps": 100, "consecutive_short_falls": 5, "probes": probes}


def next_trial(budget, requested_phase):
    """An attempted start consumes a slot even if interrupted; never retry it for free."""
    schedule = budget.manifest["schedule"]
    trials = list(budget.state["trials"].values())
    remaining = budget.remaining(requested_phase)
    if remaining == 0:
        return None
    if requested_phase in {"evaluation", "retention"}:
        if sum(t["phase"] == requested_phase for t in trials) >= schedule[requested_phase + "_trials"]:
            return None
        horizon = schedule["retention_steps"] if requested_phase == "retention" else schedule["trial_steps"]
        return {"phase": requested_phase, "max_steps": min(horizon, remaining)}
    if budget.short_fall_stop():
        return None
    spent = budget.manifest["limits"]["adaptation"] - remaining
    attempted = sum(t["phase"] in {"selection", "flat_selection"} for t in trials)
    if attempted < len(schedule["probes"]):
        probe = schedule["probes"][attempted]
        if spent >= probe["at"]:
            return {"phase": probe["phase"], "max_steps": min(probe["max_steps"], budget.remaining(probe["phase"]))}
        remaining = min(remaining, probe["at"] - spent)
    return {"phase": "adaptation", "max_steps": min(schedule["trial_steps"], remaining)}
