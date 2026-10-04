"""Summarize the declared v3/v4 incremental gates without choosing checkpoints."""
import argparse
import json
from pathlib import Path
import statistics

from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    protocol = json.loads(Path(args.protocol).read_text())
    report = json.loads((root / "report.json").read_text())
    audit = json.loads((root / "trajectory_audit.json").read_text())
    assert report["status"] == "complete" and audit["status"] == "verified"
    assert report["protocol_sha256"] == sha256_file(args.protocol)
    modes = ("teacher", "flow", "previous", "adapted")
    if protocol.get("evaluate_initial_adapter"):
        modes = (*modes[:-1], "initial", "adapted")
    evaluations = {}
    for mode in modes:
        phase = json.loads((root / f"evaluation-{mode}" / "report.json").read_text())
        rough = [e for e in phase["episodes"] if e["terrain"] != "flat"]
        evaluations[mode] = dict(by_terrain=phase["by_terrain"],
            total_success=sum(e["success"] for e in phase["episodes"]), total=len(phase["episodes"]),
            rough_success=sum(e["success"] for e in rough), rough_total=len(rough),
            rough_mean_trial_rms_m=statistics.mean(e["body_lateral_rms_m"] for e in rough),
            rough_max_peak_m=max(e["max_body_lateral_m"] for e in rough),
            rough_median_peak_m=statistics.median(e["max_body_lateral_m"] for e in rough))
    candidate, base, previous = (evaluations[m] for m in ("adapted", "flow", "previous"))
    gates = dict(
        actual_learning=report["actor_changed"] and report["adapter_updates"] > 0,
        completion_vs_both=all(candidate["by_terrain"][t]["success"] >= evaluations[m]["by_terrain"][t]["success"]
                               for m in ("flow", "previous") for t in ("flat", "101", "107")),
        rough_rms_vs_flow=candidate["rough_mean_trial_rms_m"] <= .95*base["rough_mean_trial_rms_m"],
        worst_peak_vs_flow=candidate["rough_max_peak_m"] <= base["rough_max_peak_m"])
    if protocol.get("warmstart_comparison"):
        gates.update(rough_rms_vs_previous=candidate["rough_mean_trial_rms_m"] <= .98*previous["rough_mean_trial_rms_m"],
                     worst_peak_vs_previous=candidate["rough_max_peak_m"] <= previous["rough_max_peak_m"])
    if protocol.get("evaluate_initial_adapter"):
        initial = evaluations["initial"]
        gates.update(completion_vs_initial=all(candidate["by_terrain"][t]["success"] >= initial["by_terrain"][t]["success"] for t in ("flat","101","107")),
                     rough_rms_vs_initial=candidate["rough_mean_trial_rms_m"] <= .98*initial["rough_mean_trial_rms_m"],
                     worst_peak_vs_initial=candidate["rough_max_peak_m"] <= initial["rough_max_peak_m"])
    output = dict(status="audited_development_result", development_gate_passed=all(gates.values()), gates=gates,
        declared_criteria=protocol["acceptance"], evaluation=evaluations, training_rounds=report["rounds"],
        adapter_updates=report["adapter_updates"], ledger=report["ledger"],
        qualification=audit["paired_qualification"], deployment_replay_rows=audit["cpu_deployment_replay_rows"],
        final_checkpoint_sha256=report["final_checkpoint_sha256"],
        evidence={str(f.resolve()):sha256_file(f) for f in (root/"report.json",root/"trajectory_audit.json",Path(args.protocol))},
        scope="One development run with simulator route localization and paid paired training rollouts; not independent replication or hardware evidence.")
    write_json_atomic(args.output, output)
    print(json.dumps(dict(development_gate_passed=output["development_gate_passed"], gates=gates, evaluation=evaluations), indent=2))


if __name__ == "__main__":
    main()
