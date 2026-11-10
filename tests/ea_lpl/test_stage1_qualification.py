"""CPU qualification boundaries and real artifact-schema integrity checks."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from rl_training.ea_lpl.specs import stage1_specs
from rl_training.ea_lpl.stage1 import (
    CONDITIONS, FAMILIES, _family_verdict, choose_seed, condition_counts, family_of, family_table,
    validate_records, verdicts,
)
from scripts.tools import stage1_curves as curves
from scripts.tools import stage1_qualification as cli

ROOT = Path(__file__).resolve().parents[2]
METRICS = dict(rms_xy_error=.2, rms_yaw_error=.3, falls_per_episode=.02)


def records():
    return [dict(env_id=slot % 120, episode_index=0, namespace=4, slot=slot, column=column, terrain=terrain,
                 steps=100, completed=True, safety_ending=False, bounds_ending=False, timeout=False,
                 projected_fraction=.2, forward_progress_m=9.3, terminal_lateral_m=.1,
                 lateral_peak_m=.2, terminal_yaw_deg=-2.)
            for slot, (column, terrain) in enumerate((c, t) for _ in range(4)
                                                     for c, t in enumerate(CONDITIONS) for _ in range(8))]


def evaluations():
    return {level: records() for level in ("L1", "L2")}


@pytest.mark.parametrize("label,expected", list(zip(CONDITIONS, (
    "flat", "multi_scale_rough", *("slope",) * 4, *("multi_step_stairs",) * 2,
    *("slope_plus_rough",) * 4, *("friction_variation",) * 3))))
def test_all_real_condition_families(label, expected):
    assert family_of(label) == expected


@pytest.mark.parametrize("label", ["unknown", "slope_unknown", "slope_rough_unknown", "omni_flat_0"])
def test_unknown_condition_is_rejected(label):
    with pytest.raises(ValueError, match="unknown condition"):
        family_of(label)


def test_family_table_has_exact_intervals_and_step_weighted_projection():
    rows = records()
    flat = [r for r in rows if r["terrain"] == "flat"]
    for row in flat:
        row.update(steps=1, projected_fraction=0)
    flat[0].update(steps=31, projected_fraction=1, completed=False, safety_ending=True)
    table = {row["family"]: row for row in family_table(rows)}
    assert set(table) == set(FAMILIES)
    assert table["flat"]["n"] == 32
    assert table["flat"]["completed"] == 31
    assert table["flat"]["safety"] == 1
    assert table["flat"]["projected_fraction"] == .5  # 31/(31 + 31), not an episode mean
    assert 0 < table["flat"]["ci_low"] < table["flat"]["rate"] < table["flat"]["ci_high"] < 1
    assert condition_counts(rows)["flat"] == (31, 32)


def test_q1_inclusive_exact_rate_edges():
    # Exact .90/.85 cannot occur at these families' fixed multiples-of-32 sizes;
    # isolate the comparison while the full-design tests below verify validation.
    tables = {level: [dict(family=f, rate=rate) for f in FAMILIES]
              for level, rate in (("L1", .90), ("L2", .85))}
    assert _family_verdict(tables)["pass"]
    tables["L1"][0]["rate"] = .899999
    assert not _family_verdict(tables)["pass"]
    tables["L1"][0]["rate"] = .90
    tables["L2"][0]["rate"] = .849999
    assert not _family_verdict(tables)["pass"]


def test_complete_design_passes_and_q1_uses_separate_levels():
    data = evaluations()
    assert all(q["pass"] for q in verdicts(data, METRICS, METRICS).values())
    for row in [r for r in data["L1"] if r["terrain"] == "flat"][:4]:
        row["completed"] = False  # 28/32 < .90, even with perfect L2
    qs = verdicts(data, METRICS, METRICS)
    assert not qs["Q1"]["pass"]
    assert qs["Q2"]["pass"]
    data["L1"] = records()
    for row in [r for r in data["L2"] if r["terrain"] == "flat"][:5]:
        row["completed"] = False  # 27/32 < .85
    assert not verdicts(data, METRICS, METRICS)["Q1"]["pass"]


def test_q2_exactly_24_of_32_and_all_30_conditions_required():
    data = evaluations()
    for level in data:
        for terrain in CONDITIONS:
            for row in [r for r in data[level] if r["terrain"] == terrain][:8]:
                row["completed"] = False
    assert verdicts(data, METRICS, METRICS)["Q2"]["pass"]
    data["L2"][-1]["completed"] = False
    assert not verdicts(data, METRICS, METRICS)["Q2"]["pass"]


def test_q3_exactly_24_of_480_per_level_not_pooled():
    data = evaluations()
    for level in data:
        for row in data[level][:24]:
            row["safety_ending"] = True
    assert verdicts(data, METRICS, METRICS)["Q3"]["pass"]
    data["L1"][24]["safety_ending"] = True
    data["L2"][0]["safety_ending"] = False  # total remains 48, but L1 fails
    assert not verdicts(data, METRICS, METRICS)["Q3"]["pass"]


@pytest.mark.parametrize("key", list(METRICS))
def test_q4_each_inclusive_threshold_and_just_over(key):
    baseline = dict(rms_xy_error=1., rms_yaw_error=2., falls_per_episode=.1)
    actual = dict(rms_xy_error=1.1, rms_yaw_error=2.2, falls_per_episode=.1 + .01)
    assert verdicts(evaluations(), actual, baseline)["Q4"]["pass"]
    actual[key] += 1e-9
    assert not verdicts(evaluations(), actual, baseline)["Q4"]["pass"]


@pytest.mark.parametrize("bad", [None, {}, dict(METRICS, rms_xy_error=float("nan")),
                                 dict(METRICS, rms_yaw_error=float("inf")),
                                 dict(METRICS, falls_per_episode=-.1), dict(METRICS, rms_xy_error=True)])
def test_retention_missing_malformed_or_nonfinite_fails_closed(bad):
    assert not verdicts(evaluations(), bad, METRICS)["Q4"]["pass"]
    assert not verdicts(evaluations(), METRICS, bad)["Q4"]["pass"]


@pytest.mark.parametrize("bad", ["missing_record", "duplicate", "wrong_condition_count", "missing_level",
                                 "unknown_condition", "wrong_column", "missing_metric", "nonfinite",
                                 "non_boolean", "bad_steps", "missing_provenance", "later_episode"])
def test_terrain_integrity_failures_fail_all_terrain_verdicts(bad):
    data = evaluations()
    row = data["L1"][0]
    if bad == "missing_record":
        data["L1"].pop()
    elif bad == "duplicate":
        data["L1"][-1] = dict(row)
    elif bad == "wrong_condition_count":
        row.update(terrain=CONDITIONS[1], column=1)
    elif bad == "missing_level":
        del data["L2"]
    elif bad == "unknown_condition":
        row["terrain"] = "slope_new"
    elif bad == "wrong_column":
        row["column"] = 1
    elif bad == "missing_metric":
        del row["terminal_yaw_deg"]
    elif bad == "nonfinite":
        row["lateral_peak_m"] = float("nan")
    elif bad == "non_boolean":
        row["completed"] = "false"
    elif bad == "bad_steps":
        row["steps"] = 0
    elif bad == "missing_provenance":
        del row["slot"]
    elif bad == "later_episode":
        row["episode_index"] = 1
    qs = verdicts(data, METRICS, METRICS)
    assert all(not qs[f"Q{i}"]["pass"] for i in (1, 2, 3))


def test_repeating_env_id_across_four_runs_is_valid_with_unique_provenance():
    rows = records()
    assert len({r["env_id"] for r in rows}) == 120
    validate_records(rows)


def result(seed, *, rate=.95, safety=10, passed=True):
    return dict(seed=seed, verdicts={f"Q{i}": {"pass": passed} for i in range(1, 5)},
                l2_completion_rate=rate, total_safety_endings=safety)


def test_choose_seed_qualification_then_l2_then_safety_then_seed42():
    assert choose_seed([result(42, rate=1, safety=0, passed=False), result(43)])[0] == 43
    assert choose_seed([result(42, rate=.96, safety=20), result(43, rate=.95, safety=0)])[0] == 42
    assert choose_seed([result(42, safety=11), result(43, safety=10)])[0] == 43
    assert choose_seed([result(43), result(42)])[0] == 42
    seed, reason = choose_seed([result(42, passed=False), result(43, passed=False)])
    assert seed is None and "No qualified seed" in reason
    assert choose_seed([])[0] is None
    incomplete = result(42)
    del incomplete["verdicts"]["Q4"]
    assert choose_seed([incomplete])[0] is None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_policy(evidence, name, actor, kind="qualify"):
    rows = records()
    for spec in stage1_specs({name: actor}, kind=kind):
        folder = evidence / spec["label"]
        folder.mkdir(parents=True)
        write_json(folder / "spec.json", spec)
        write_json(folder / "manifest.json", dict(schema="ea_lpl_run_manifest_v1", label=spec["label"],
                   actor_sha256=actor["sha256"], actor_path=actor["path"], spec_sha256=cli.sha256(folder / "spec.json")))
        chunk = []
        for env_id, slot in enumerate(spec["slots"]):
            row = dict(rows[slot % 480], env_id=env_id)
            for key in ("namespace", "slot"):
                del row[key]
            chunk.append(row)
        assignments = [dict(env_id=i, column=r["column"], subterrain=r["terrain"]) for i, r in enumerate(chunk)]
        case = dict(label=spec["label"], codes=None, records=chunk, initial=dict(assignments=assignments,
                    initial_state_sha256=hashlib.sha256(str(spec["slots"]).encode()).hexdigest()))
        write_json(folder / "result.json", dict(status="passed", label=spec["label"], level=spec["level"],
                                               slots=spec["slots"], case=case))
        write_json(folder / "execution.json", dict(status="passed", exit_code=0, deadline_failure=None,
                                                   process_group_empty=True))
        (folder / "runtime.log").write_text("[ea-lpl] evaluation and cleanup finished\n")


@pytest.fixture
def evidence(tmp_path):
    actors, retentions = {}, {}
    for name in ("baseline", "seed42", "seed43"):
        path = tmp_path / f"{name}_final.pt"
        path.write_bytes(name.encode())
        actors[name] = dict(path=str(path), sha256=cli.sha256(path))
        write_policy(tmp_path, name, actors[name])
        retention = tmp_path / f"{name}_retention.json"
        write_json(retention, dict(status="passed", checkpoint=str(path), checkpoint_sha256=cli.sha256(path),
                                   task="Flat-Deeprobotics-Lite3-v0", num_envs=1024, seed=20261109,
                                   requested_steps=3000, completed_steps=3000, metrics=METRICS,
                                   clip_actions=100, policy_observation_corruption=False, safety_filter=False,
                                   termination_terms=["bad_orientation_2", "illegal_contact", "time_out"],
                                   cleanup=dict(environment="returned", application="returned"),
                                   runtime=dict(child_exit=0, fatal_messages=[], deadline_failure=None)))
        retentions[name] = retention
    return tmp_path, actors, retentions


@pytest.mark.parametrize("kind", ["qualify", "extend"])
def test_cli_loads_complete_design_with_real_schema_and_diagnostics(tmp_path, kind):
    actor = dict(path="/actors/final.pt", sha256="a" * 64)
    write_policy(tmp_path, "final", actor, kind=kind)
    loaded = cli.load_policy(tmp_path, "final", actor, kind=kind)
    assert all(len(rows) == 480 for rows in loaded["records"].values())
    assert {r["slot"] for r in loaded["records"]["L1"]} == set(range(480 if kind == "extend" else 0,
                                                                 960 if kind == "extend" else 480))
    assert len(loaded["runs"]) == 8


def test_retry_resolves_one_first_passed_attempt_and_does_not_pool(evidence):
    path, actors, _ = evidence
    label = "s1q_seed42_l1_0"
    original = path / label
    retry = path / (label + "_retry1")
    retry.mkdir()
    for filename in ("spec.json", "manifest.json", "result.json", "execution.json", "runtime.log"):
        (retry / filename).write_bytes((original / filename).read_bytes())
    # Both passed: original wins. Corrupt retry would fail if it were loaded.
    write_json(retry / "manifest.json", {})
    data = cli.load_policy(path, "seed42", actors["seed42"], kind="qualify")
    assert data["runs"][label]["folder"] == str(original)
    # Failed first attempt: the intact retry wins, and still contributes only 120.
    (retry / "manifest.json").write_bytes((original / "manifest.json").read_bytes())
    write_json(original / "result.json", dict(status="failed"))
    data = cli.load_policy(path, "seed42", actors["seed42"], kind="qualify")
    assert data["runs"][label]["folder"] == str(retry)
    assert len(data["records"]["L1"]) == 480


def corrupt_attempt(folder, corruption):
    if corruption == "missing_receipt":
        (folder / "execution.json").unlink()
    elif corruption == "malformed_receipt":
        (folder / "execution.json").write_text("{unfinished")
    elif corruption == "nonobject_receipt":
        write_json(folder / "execution.json", [])
    elif corruption in ("nonzero_exit", "boolean_exit", "float_exit", "deadline", "process_group", "receipt_status",
                        "missing_deadline"):
        receipt = cli.read_json(folder / "execution.json")
        field, value = {
            "nonzero_exit": ("exit_code", -15), "boolean_exit": ("exit_code", False),
            "float_exit": ("exit_code", 0.0), "deadline": ("deadline_failure", "environment_close exceeded 30s"),
            "process_group": ("process_group_empty", False), "receipt_status": ("status", "failed"),
            "missing_deadline": ("deadline_failure", None),
        }[corruption]
        receipt[field] = value
        if corruption == "missing_deadline":
            del receipt[field]
        write_json(folder / "execution.json", receipt)
    elif corruption == "missing_runtime":
        (folder / "runtime.log").unlink()
    elif corruption == "malformed_runtime":
        (folder / "runtime.log").write_bytes(b"\xff")
    elif corruption.startswith("native:"):
        (folder / "runtime.log").write_text("cleanup\n" + corruption.removeprefix("native:") + "\n")
    else:
        part, field, value = {
            "spec": ("spec", "namespace", 3), "identity": ("manifest", "actor_sha256", "b" * 64),
            "samples": ("result", None, None), "malformed_result": ("result", None, None),
        }[corruption]
        report = cli.read_json(folder / f"{part}.json")
        if corruption == "samples":
            report["case"]["records"][0]["forward_progress_m"] = float("nan")
        elif corruption == "malformed_result":
            (folder / "result.json").write_text("{unfinished")
            return
        else:
            report[field] = value
        write_json(folder / f"{part}.json", report)


ADMISSION_FAILURES = [
    "missing_receipt", "malformed_receipt", "nonobject_receipt", "nonzero_exit", "boolean_exit", "float_exit",
    "deadline", "process_group", "receipt_status", "missing_deadline", "missing_runtime", "malformed_runtime",
    *["native:" + marker for marker in ("PhysX error: stale state", "Scene state is corrupted", "CUDA error",
                                       "CUDA_ERROR_ILLEGAL_ADDRESS", "failed to allocate", "out of memory",
                                       "buffer overflow", "buffer capacity exceeded")],
]


@pytest.mark.parametrize("corruption", ADMISSION_FAILURES)
def test_unfinalized_or_unhealthy_terrain_evidence_prevents_selection(evidence, corruption):
    path, actors, retentions = evidence
    corrupt_attempt(path / "s1q_baseline_l1_0", corruption)
    report = cli.summarize(path, actors, retentions, kind="qualify")
    assert report["selected_seed"] is None
    assert report["selected_actor"] is None
    assert any("terrain evidence:" in error for error in report["policies"]["baseline"]["errors"])


@pytest.mark.parametrize("corruption", ADMISSION_FAILURES + ["spec", "identity", "samples", "malformed_result"])
def test_first_fully_valid_retry_wins_over_invalid_original(evidence, corruption):
    path, actors, _ = evidence
    label = "s1q_seed42_l1_0"
    original, retry = path / label, path / (label + "_retry1")
    shutil.copytree(original, retry)
    corrupt_attempt(original, corruption)
    data = cli.load_policy(path, "seed42", actors["seed42"], kind="qualify")
    assert data["runs"][label]["folder"] == str(retry)
    assert len(data["records"]["L1"]) == 480
    assert len({(r["namespace"], r["slot"], r["episode_index"]) for r in data["records"]["L1"]}) == 480


@pytest.mark.parametrize("part,field,value", [
    ("manifest", "actor_sha256", "b" * 64), ("manifest", "actor_path", "/wrong/final.pt"),
    ("manifest", "spec_sha256", "b" * 64), ("manifest", "label", "wrong"),
    ("spec", "namespace", 3), ("result", "label", "wrong"), ("result", "level", "L2"),
])
def test_loader_rejects_mismatched_artifact_identity(evidence, part, field, value):
    path, actors, _ = evidence
    filename = path / "s1q_seed42_l1_0" / f"{part}.json"
    document = cli.read_json(filename)
    document[field] = value
    write_json(filename, document)
    with pytest.raises(ValueError, match="identity|differs|mismatch"):
        cli.load_policy(path, "seed42", actors["seed42"], kind="qualify")


@pytest.mark.parametrize("corruption", ["duplicate_env", "duplicate_chunk", "nonfinite", "assignment", "missing_record"])
def test_loader_rejects_partial_or_corrupt_run_samples(evidence, corruption):
    path, actors, _ = evidence
    filename = path / "s1q_seed42_l1_0" / "result.json"
    report = cli.read_json(filename)
    if corruption == "duplicate_env":
        report["case"]["records"][-1] = report["case"]["records"][0]
    elif corruption == "duplicate_chunk":
        filename = path / "s1q_seed42_l1_1" / "spec.json"
        report = cli.read_json(filename)
        report["slots"] = list(range(120))
    elif corruption == "nonfinite":
        report["case"]["records"][0]["forward_progress_m"] = float("nan")
    elif corruption == "assignment":
        report["case"]["initial"]["assignments"][0]["subterrain"] = "slope_uphill"
    elif corruption == "missing_record":
        report["case"]["records"].pop()
    write_json(filename, report)
    with pytest.raises(ValueError):
        cli.load_policy(path, "seed42", actors["seed42"], kind="qualify")


def test_summary_verifies_paired_starts_and_emits_explicit_none(evidence):
    path, actors, retentions = evidence
    report = cli.summarize(path, actors, retentions, kind="qualify")
    assert report["selected_seed"] == 42
    assert report["selected_actor"] == actors["seed42"]
    condition = report["policies"]["seed42"]["conditions"]["L1"][0]
    assert condition["n"] == 32 and condition["progress_mean_m"] == 9.3
    assert condition["lateral_peak_mean_m"] == .2
    assert condition["terminal_abs_yaw_mean_deg"] == 2
    assert "ci_low" in condition and "ci_high" in condition
    for name in ("seed42", "seed43"):
        filename = path / f"s1q_{name}_l1_0" / "result.json"
        document = cli.read_json(filename)
        document["case"]["initial"]["initial_state_sha256"] = "f" * 64
        write_json(filename, document)
    report = cli.summarize(path, actors, retentions, kind="qualify")
    assert report["selected_seed"] is None
    assert report["selected_actor"] is None
    assert "No qualified seed" in report["selection_reason"]
    assert not report["policies"]["seed42"]["paired_starts_verified"]
    text = cli.markdown(report)
    assert "Selected seed: none" in text
    assert text.index("| seed43 |") < text.index("seed42: paired baseline evidence:")


def test_bad_baseline_evidence_prevents_qualification_of_both_seeds(evidence):
    path, actors, retentions = evidence
    write_json(path / "s1q_baseline_l1_0" / "result.json", {"status": "failed"})
    report = cli.summarize(path, actors, retentions, kind="qualify")
    assert report["selected_seed"] is None
    assert all(not report["policies"][name]["verdicts"]["Q1"]["pass"] for name in ("seed42", "seed43"))


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "c" * 64), ("checkpoint", "/wrong.pt"),
                                         ("status", "failed"), ("completed_steps", 2999)])
def test_retention_identity_and_completion_are_required(evidence, field, value):
    _, actors, retentions = evidence
    filename = retentions["seed42"]
    report = cli.read_json(filename)
    report[field] = value
    write_json(filename, report)
    with pytest.raises(ValueError, match="incomplete or differs"):
        cli.load_retention(filename, actors["seed42"])


@pytest.mark.parametrize("field,value", [
    ("cleanup", None), ("cleanup", {}),
    ("cleanup", dict(environment="closing", application="returned")),
    ("cleanup", dict(environment="returned", application="failed")),
    ("runtime", None), ("runtime", {}),
    ("runtime", dict(child_exit=-15, fatal_messages=[], deadline_failure=None)),
    ("runtime", dict(child_exit=False, fatal_messages=[], deadline_failure=None)),
    ("runtime", dict(child_exit=0.0, fatal_messages=[], deadline_failure=None)),
    ("runtime", dict(child_exit=0, fatal_messages=["PhysX error: cleanup"], deadline_failure=None)),
    ("runtime", dict(child_exit=0, fatal_messages=None, deadline_failure=None)),
    ("runtime", dict(child_exit=0, fatal_messages=[], deadline_failure="application_close exceeded 30s")),
    ("runtime", dict(child_exit=0, fatal_messages=[])),
    ("clip_actions", None), ("clip_actions", 1),
    ("policy_observation_corruption", True), ("policy_observation_corruption", 0),
    ("safety_filter", True), ("safety_filter", 0),
    ("termination_terms", ["time_out", "illegal_contact"]),
    ("termination_terms", ["bad_orientation_2", "illegal_contact", "time_out", "source_curriculum_safety"]),
    ("termination_terms", ["bad_orientation_2", "illegal_contact", "time_out", "time_out"]),
])
def test_retention_requires_parent_finalization_and_exact_flat_settings(evidence, field, value):
    path, actors, retentions = evidence
    # A bad baseline must deny both candidate seeds even when all terrain evidence passes.
    filename = retentions["baseline"]
    report = cli.read_json(filename)
    if value is None:
        del report[field]
    else:
        report[field] = value
    write_json(filename, report)
    summary = cli.summarize(path, actors, retentions, kind="qualify")
    assert summary["selected_seed"] is None and summary["selected_actor"] is None
    assert all(not summary["policies"][name]["verdicts"]["Q4"]["pass"] for name in ("seed42", "seed43"))
    assert any("retention evidence:" in error for error in summary["policies"]["baseline"]["errors"])


def test_finalized_retention_producer_schema_is_accepted(evidence):
    _, actors, retentions = evidence
    for name in actors:
        assert cli.load_retention(retentions[name], actors[name]) == METRICS


def test_cli_writes_reviewable_json_and_markdown_from_explicit_final_actors(evidence):
    path, actors, retentions = evidence
    command = [sys.executable, str(ROOT / "scripts/tools/stage1_qualification.py"),
               "--evidence", str(path), "--out", str(path / "out")]
    for name, actor in actors.items():
        command += ["--policy", f"{name}={actor['path']}", "--retention", f"{name}={retentions[name]}"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["selected_seed"] == 42
    report = cli.read_json(path / "out/qualification.json")
    assert report["selected_actor"]["path"].endswith("seed42_final.pt")
    assert "Step-weighted projection" in (path / "out/qualification.md").read_text()


def event_file(path, *, missing=False, nonfinite=False):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(path))
    try:
        for step in (0, 10):
            values = [Summary.Value(tag=f"{curves.PREFIX}/{family}/{metric}",
                                    simple_value=float("nan") if nonfinite else (step / 10 if metric == "completion_ema" else step / 2))
                      for family in FAMILIES for metric in ("mean_level", "completion_ema")
                      if not (missing and family == "flat" and metric == "mean_level")]
            writer.add_event(Event(step=step, summary=Summary(value=values)))
    finally:
        writer.close()


def test_curves_reads_all_scalars_and_generates_png_svg(tmp_path):
    event_file(tmp_path / "seed42")
    data = curves.load_curves(tmp_path / "seed42")
    assert data["flat"]["mean_level"] == [(0, 0), (10, 5)]
    assert data["flat"]["completion_ema"] == [(0, 0), (10, 1)]
    curves.plot_curves({42: data, 43: data}, tmp_path / "out/curves")
    assert (tmp_path / "out/curves.png").read_bytes().startswith(b"\x89PNG")
    svg = (tmp_path / "out/curves.svg").read_text()
    assert "#2a78d6" in svg and "#eb6834" in svg
    assert "Completion EMA" in svg and "Mean level" in svg
    assert "mastery" not in svg.lower()


@pytest.mark.parametrize("missing,nonfinite", [(True, False), (False, True)])
def test_curves_rejects_missing_or_nonfinite_scalar(tmp_path, missing, nonfinite):
    event_file(tmp_path, missing=missing, nonfinite=nonfinite)
    with pytest.raises(ValueError, match="missing|nonfinite"):
        curves.load_curves(tmp_path)
