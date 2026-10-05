"""Independently reconstruct the FlowDAgger pilot from persisted step chunks."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rl_training.lite3_joint_conventions import clip_lite3_policy_action
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.flowdagger import ConditionalFlow, NoisePolicy


def region_from_numpy(obs, options):
    route = ((np.abs(obs[...,9])*np.float32(1.65) > np.float32(options["lateral_m"]))
        | (obs[...,11] < np.float32(math.cos(math.radians(options["yaw_deg"])))))
    tilted = -obs[...,5] < np.float32(math.cos(math.radians(options["tilt_deg"])))
    return route & ~tilted if options.get("tilt_role") == "preservation" else route | tilted


def read_pair_fields(folder):
    fields = ("policy_observation", "post_root_state", "intervention")
    rows = {key: [] for key in fields}
    for file in sorted((folder / "trajectory").glob("chunk-*.npz")):
        with np.load(file, allow_pickle=False) as data:
            for key in fields:
                rows[key].append(data[key])
    return {key: np.concatenate(value) for key, value in rows.items()}


def audit_qualification(directory, protocol, report):
    """Recompute the gate from raw prefixes and independently verified episodes."""
    q = protocol["paired_correction"]["qualification"]
    total_accepted = 0
    accepted_labels = 0
    autonomous_labels = 0
    summary = []
    for round_id in range(protocol["rounds"]):
        ref_dir, cor_dir = directory / f"reference-{round_id:02d}", directory / f"round-{round_id:02d}"
        reference = json.loads((ref_dir / "report.json").read_text())
        corrected = json.loads((cor_dir / "report.json").read_text())
        assert reference["initial_state_sha256"] == corrected["initial_state_sha256"]
        ref, cor = read_pair_fields(ref_dir), read_pair_fields(cor_dir)
        for episode in reference["episodes"]:
            if episode["success"]:
                observations = ref["policy_observation"][:episode["steps"],episode["env_id"]]
                autonomous_labels += (int((~region_from_numpy(observations, protocol["correction_region"])).sum())
                    if protocol.get("correction_region") else len(observations))
        decisions = json.loads((cor_dir / "qualification.json").read_text())
        eligible_indices = set()
        round_accepted = 0
        for row in decisions:
            i = row["env_id"]
            a, b = reference["episodes"][i], corrected["episodes"][i]
            assert row["reference"] == a and row["corrected"] == b
            interventions = np.flatnonzero(cor["intervention"][:b["steps"], i])
            assert row["intervention_count"] == len(interventions)
            matched = False
            prefix = 0.
            if len(interventions):
                first = interventions[0]
                if first < a["steps"]:
                    prefix = float(np.max(np.abs(cor["policy_observation"][:first+1, i]-ref["policy_observation"][:first+1, i])))
                    if first:
                        prefix = max(prefix, float(np.max(np.abs(cor["post_root_state"][:first, i]-ref["post_root_state"][:first, i]))))
                    matched = prefix <= q["prefix_tolerance"]
                    np.testing.assert_allclose(prefix, row["prefix_max_abs"], rtol=0., atol=1.e-8)
            recovered = (not a["success"] and not a.get("collector_cutoff", False) and (a["failure"] or a["deadline"]))
            lower_drift = (a["success"]
                and a["body_lateral_rms_m"]-b["body_lateral_rms_m"] >= max(q["minimum_rms_gain"], (1-q["rms_ratio"])*a["body_lateral_rms_m"])
                and b["max_body_lateral_m"] <= a["max_body_lateral_m"]
                and b["steps"] <= q["maximum_duration_ratio"]*a["steps"])
            eligible = bool(matched and b["success"] and (recovered or lower_drift))
            assert eligible == row["qualified"]
            if eligible:
                round_accepted += 1
                eligible_indices.update((i, int(step)) for step in interventions)
        inversion_path = cor_dir / "inversions.npz"
        if inversion_path.exists():
            with np.load(inversion_path, allow_pickle=False) as inverse:
                actual = set(zip(map(int, inverse["env_id"]), map(int, inverse["step"])))
                assert actual == eligible_indices
                accepted_labels += int(inverse["accepted"].sum())
        else:
            assert not eligible_indices
        assert report["rounds"][round_id]["qualified_episodes"] == round_accepted
        assert report["rounds"][round_id]["buffer_counts"]["correction"] == accepted_labels
        assert report["rounds"][round_id]["buffer_counts"]["autonomous"] == autonomous_labels
        if protocol.get("correction_region"):
            checkpoint = torch.load(cor_dir/"adapter.pt", map_location="cpu", weights_only=False)
            for role, rows in checkpoint["buffer"].items():
                for observations, _ in rows:
                    region = region_from_numpy(observations.numpy(), protocol["correction_region"])
                    assert bool(region.all()) if role == "correction" else not bool(region.any())
            if protocol.get("recovery_sampling") and "recovery_sampling" in report["rounds"][round_id]:
                obs = torch.cat([o for o, _ in checkpoint["buffer"]["autonomous"]]).numpy()
                threshold = np.float32(math.cos(math.radians(protocol["recovery_sampling"]["tilt_deg"])))
                count = int((-obs[:,5] < threshold).sum())
                assert report["rounds"][round_id]["recovery_sampling"]["recovery_rows"] == count
        total_accepted += round_accepted
        summary.append(dict(round=round_id, qualified_episodes=round_accepted, attempted_inversions=len(eligible_indices)))
    return dict(verified=True, qualified_episodes=total_accepted, accepted_labels=accepted_labels, rounds=summary)


def audit_phase(folder, protocol):
    report = json.loads((folder / "report.json").read_text())
    assert report["status"] == "complete"
    initial = np.load(folder / "initial_state.npz", allow_pickle=False)
    manifest = json.loads((folder / "trajectory/manifest.json").read_text())
    assert manifest["status"] == "complete"
    width = protocol["num_envs"]
    active = np.ones(width, dtype=bool)
    last_history = np.zeros((width, 12), dtype=np.float32)
    peak = np.zeros(width); peak_from_start = np.zeros(width); squares = np.zeros(width); samples = np.zeros(width)
    successes = np.zeros(width, dtype=bool)
    terminal_failure_terms = {k: np.zeros(width, dtype=bool) for k in ("contact", "orientation", "lateral")}
    ends = np.zeros(width, dtype=int)
    intervention_count = 0
    steps = 0
    hashes = {}
    for chunk in manifest["chunks"]:
        file = folder / "trajectory" / chunk["file"]
        hashes[chunk["file"]] = sha256_file(file)
        with np.load(file, allow_pickle=False) as data:
            obs = data["policy_observation"]
            assert np.isfinite(obs).all()
            if protocol.get("teacher_route_lookahead_m") is not None:
                yaw = np.arctan2(obs[...,10], obs[...,11])
                error = yaw + np.arctan2(obs[...,9]*np.float32(1.65), np.float32(protocol["teacher_route_lookahead_m"]))
                expected_route = np.stack((obs[...,9], np.sin(error), np.cos(error)), axis=-1)
                np.testing.assert_allclose(data["teacher_route_features"], expected_route, atol=2.e-6, rtol=1.e-6)
            np.testing.assert_allclose(obs[:, :, 6:9], np.broadcast_to([.3, 0., 0.], obs[:, :, 6:9].shape), rtol=0., atol=1.e-7)
            chosen = np.where(data["intervention"][:, :, None], data["teacher_requested_action"], data["student_requested_action"])
            expected_action = clip_lite3_policy_action(torch.from_numpy(chosen)).numpy()
            np.testing.assert_array_equal(data["emitted_action"], expected_action)
            if protocol.get("correction_region"):
                region = region_from_numpy(obs, protocol["correction_region"])
                np.testing.assert_array_equal(data["correction_region"], region)
                student = clip_lite3_policy_action(torch.from_numpy(data["student_requested_action"])).numpy()
                teacher = clip_lite3_policy_action(torch.from_numpy(data["teacher_requested_action"])).numpy()
                disagreement = np.sqrt(np.mean((student-teacher)**2, axis=-1))
                np.testing.assert_allclose(data["action_disagreement"], disagreement, atol=2.e-6, rtol=1.e-6)
                expected_intervention = (data["active"] & region & (data["action_disagreement"] > protocol["intervention_action_rmse"]))
                if report["mode"] != "collect":
                    expected_intervention[:] = False
                np.testing.assert_array_equal(data["intervention"], expected_intervention)
            for row in range(len(obs)):
                np.testing.assert_array_equal(data["active"][row], active)
                np.testing.assert_allclose(obs[row, :, -12:], last_history, atol=1.e-6, rtol=0.)
                np.testing.assert_array_equal(data["done"][row].astype(bool),
                    data["success"][row] | data["failure"][row] | data["deadline"][row])
                if protocol.get("record_failure_terms"):
                    np.testing.assert_array_equal(data["failure"][row], data["failure_contact"][row]
                        | data["failure_orientation"][row] | data["failure_lateral"][row])
                lateral = data["post_root_state"][row, :, 1] - initial["origins"][:, 1]
                offset = data["post_root_state"][row, :, 1] - initial["root_state"][:, 1]
                peak = np.maximum(peak, np.where(active, abs(lateral), 0.))
                peak_from_start = np.maximum(peak_from_start, np.where(active, abs(offset), 0.))
                squares += np.where(active, lateral**2, 0.)
                samples += active
                ending = active & data["done"][row].astype(bool)
                if protocol.get("record_failure_terms"):
                    for key in terminal_failure_terms:
                        terminal_failure_terms[key][ending] = data[f"failure_{key}"][row, ending]
                successes[ending] = data["success"][row, ending]
                ends[ending] = steps + 1
                intervention_count += int(data["intervention"][row].sum())
                assert not (data["intervention"][row] & ~active).any()
                active &= ~data["done"][row].astype(bool)
                last_history = data["emitted_action"][row].copy()
                last_history[data["done"][row].astype(bool)] = 0.
                steps += 1
    assert steps * width == report["confirmed_controls"]
    assert intervention_count == report["interventions"]
    if report["mode"] != "collect":
        assert intervention_count == 0
    for episode in report["episodes"]:
        i = episode["env_id"]
        assert bool(successes[i]) == episode["success"]
        assert (ends[i] or steps) == episode["steps"]
        if protocol.get("record_failure_terms") and not episode.get("collector_cutoff", False):
            assert episode["failure_terms"] == {k: bool(v[i]) for k,v in terminal_failure_terms.items()}
            assert episode["failure"] == any(episode["failure_terms"].values())
        np.testing.assert_allclose(peak[i], episode["max_body_lateral_m"], atol=1.e-6)
        np.testing.assert_allclose(peak_from_start[i], episode["max_body_offset_from_start_m"], atol=1.e-6)
        np.testing.assert_allclose(np.sqrt(squares[i]/samples[i]), episode["body_lateral_rms_m"], atol=1.e-5)
    return dict(status="verified", transitions=steps*width, interventions=intervention_count,
                max_body_lateral_m=float(peak.max()), median_trial_peak_m=float(np.median(peak)),
                max_body_offset_from_start_m=float(peak_from_start.max()),
                mean_trial_body_rms_m=float(np.sqrt(squares/samples).mean()),
                by_terrain=report["by_terrain"], trace_sha256=hashes)


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.run)
    report = json.loads((directory / "report.json").read_text())
    p = json.loads(Path(args.protocol).read_text())
    assert report["status"] == "complete"
    assert report["protocol_sha256"] == sha256_file(args.protocol)
    modes = ("teacher", "flow", "previous", "adapted") if p.get("comparison_checkpoint") else ("teacher", "flow", "adapted")
    if p.get("evaluate_initial_adapter"):
        modes = (*modes[:-1], "initial", "adapted")
    phases = [f"round-{i:02d}" for i in range(p["rounds"])]
    if p.get("paired_correction"):
        phases += [f"reference-{i:02d}" for i in range(p["rounds"])]
    phases += [f"evaluation-{m}" for m in modes]
    results = {name: audit_phase(directory / name, p) for name in phases}
    qualification = audit_qualification(directory, p, report) if p.get("paired_correction") else None
    assert sum(r["transitions"] for r in results.values()) == report["ledger"]["confirmed"]
    assert report["ledger"]["uncertain_transitions"] == 0
    assert report["initial_states_paired"] and report["frozen_flow_and_teacher_unchanged"] and report["evaluation_actor_unchanged"]
    assert report["actor_changed"] or (p.get("paired_correction") and report["adapter_updates"] == 0)
    assert sha256_file(directory / "final.pt") == report["final_checkpoint_sha256"]
    # Reconstruct deployment from the standalone saved networks on CPU. The
    # expert is never loaded in this check, so it cannot mask a teacher fallback.
    torch.set_num_threads(4)
    payload = torch.load(directory / "final.pt", map_location="cpu", weights_only=False)
    flow = ConditionalFlow(**payload["flow_config"]).eval().requires_grad_(False)
    flow.load_state_dict(payload["flow"])
    actor = NoisePolicy(bound=payload["noise_bound"], hidden=tuple(payload.get("noise_hidden", (256,128)))).eval().requires_grad_(False)
    actor.load_state_dict(payload["actor"])
    replay_checks = 0
    replay_modes = ("flow", "previous", "adapted") if p.get("comparison_checkpoint") else ("flow", "adapted")
    initial = None
    if p.get("evaluate_initial_adapter"):
        assert sha256_file(directory/"initial_adapter.pt") == report["initial_adapter_sha256"]
        initial_payload = torch.load(directory/"initial_adapter.pt", map_location="cpu", weights_only=False)
        initial = NoisePolicy(bound=initial_payload["noise_bound"], hidden=tuple(initial_payload["noise_hidden"])).eval().requires_grad_(False)
        initial.load_state_dict(initial_payload["actor"])
        replay_modes = (*replay_modes[:-1], "initial", "adapted")
    previous = None
    if p.get("comparison_checkpoint"):
        assert sha256_file(p["comparison_checkpoint"]) == p["comparison_sha256"]
        previous_payload = torch.load(p["comparison_checkpoint"], map_location="cpu", weights_only=False)
        previous = NoisePolicy(bound=previous_payload["noise_bound"], hidden=tuple(previous_payload.get("noise_hidden", (256,128)))).eval().requires_grad_(False)
        previous.load_state_dict(previous_payload["actor"])
    for mode in replay_modes:
        files = sorted((directory / f"evaluation-{mode}" / "trajectory").glob("chunk-*.npz"))
        for file in (files[0], files[len(files)//2], files[-1]):
            with np.load(file, allow_pickle=False) as data:
                active = data["active"].reshape(-1)
                obs = torch.from_numpy(data["policy_observation"].reshape(-1, 48)[active][::64][:32])
                recorded = torch.from_numpy(data["student_requested_action"].reshape(-1, 12)[active][::64][:32])
                noise = torch.zeros((len(obs), 12)) if mode == "flow" else {"previous":previous,"initial":initial,"adapted":actor}[mode](obs)
                torch.testing.assert_close(flow(obs, noise), recorded, atol=3.e-5, rtol=1.e-5)
                replay_checks += len(obs)
    audit = dict(status="verified", run=str(directory.resolve()), protocol_sha256=sha256_file(args.protocol),
                 phases=results, ledger=report["ledger"], final_checkpoint_sha256=report["final_checkpoint_sha256"],
                 cpu_deployment_replay_rows=replay_checks, replay_without_teacher=True,
                 paired_qualification=qualification,
                 scope="development evaluation on paired initial states; no independent training replication or new-terrain claim")
    write_json_atomic(args.output, audit)
    print(json.dumps({k: {n:v for n,v in r.items() if n != "trace_sha256"} for k,r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
