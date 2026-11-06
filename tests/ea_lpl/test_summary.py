import math

from rl_training.ea_lpl.summary import clopper_pearson, condition_table, paired_mcnemar, select_target_conditions


def record(env_id, column, terrain, completed, progress=9.3, safety=False, lateral=0.1):
    return dict(env_id=env_id, episode_index=0, column=column, terrain=terrain, steps=1000 + env_id,
                completed=completed, safety_ending=safety, bounds_ending=not completed and not safety, timeout=False,
                forward_progress_m=progress, terminal_lateral_m=lateral, lateral_peak_m=abs(lateral),
                terminal_yaw_deg=1.0)


def test_clopper_pearson_known_values():
    assert math.isclose(clopper_pearson(0, 8)[1], 1 - 0.025 ** (1 / 8), rel_tol=1e-6)
    assert math.isclose(clopper_pearson(8, 8)[0], 0.025 ** (1 / 8), rel_tol=1e-6)
    low, high = clopper_pearson(4, 8)
    assert math.isclose(low, 0.15701, abs_tol=1e-4) and math.isclose(high, 0.84299, abs_tol=1e-4)


def test_condition_table_and_mcnemar():
    first = [record(i, 2, "slope_uphill", False, progress=4.0) for i in range(8)]
    second = [record(i, 2, "slope_uphill", i < 6, progress=9.3 if i < 6 else 4.0) for i in range(8)]
    row = condition_table(second, level="L1")[0]
    assert (row["n"], row["completed"], row["distinct_trajectories"]) == (8, 6, 8)
    test = paired_mcnemar(first, second)
    assert (test["only_first"], test["only_second"]) == (0, 6)
    assert math.isclose(test["p_value"], 2 * 0.5 ** 6)


def test_target_selection_rule():
    rows = [
        dict(level="L1", column=0, terrain="flat", n=8, completed=2, rate=0.25, safety=0, progress_mean_m=8.0),
        dict(level="L1", column=4, terrain="slope_cross_slope_left", n=8, completed=2, rate=0.25, safety=0, progress_mean_m=6.0),
        dict(level="L1", column=5, terrain="slope_cross_slope_right", n=8, completed=0, rate=0.0, safety=0, progress_mean_m=5.4),
        dict(level="L1", column=2, terrain="slope_uphill", n=8, completed=0, rate=0.0, safety=0, progress_mean_m=3.9),
        dict(level="L2", column=6, terrain="stairs_ascending", n=8, completed=3, rate=0.375, safety=0, progress_mean_m=7.0),
        dict(level="L2", column=7, terrain="stairs_descending", n=8, completed=2, rate=0.25, safety=2, progress_mean_m=7.0),
        dict(level="L2", column=3, terrain="slope_downhill", n=8, completed=7, rate=0.875, safety=0, progress_mean_m=9.3),
    ]
    eligible, chosen = select_target_conditions(rows)
    assert {(c["level"], c["column"]) for c in eligible} == {("L1", 4), ("L1", 5), ("L2", 6)}
    assert [(c["level"], c["column"]) for c in chosen] == [("L1", 4), ("L1", 5), ("L2", 6)]
