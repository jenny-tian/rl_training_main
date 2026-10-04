"""Budgeted scripted-expert FlowDAgger with frozen paired deployment evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
import time

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--flow-directory", required=True)
    parser.add_argument("--output", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
    from rl_training.vector_step_ledger import VectorStepLedger
    p = json.loads(Path(args.protocol).read_text())
    if p["schema"] != "lite3_flowdagger_pilot_v1":
        raise ValueError("unknown protocol")
    paired = p.get("paired_correction")
    if p.get("correction_region") and not paired:
        raise ValueError("disjoint supervision regions require paired reference collection")
    evaluation_modes = ("teacher", "flow", "previous", "adapted") if p.get("comparison_checkpoint") else ("teacher", "flow", "adapted")
    if p.get("evaluate_initial_adapter"):
        evaluation_modes = (*evaluation_modes[:-1], "initial", "adapted")
    expected_cap = (p["rounds"] * (2 if paired else 1) + len(evaluation_modes)) * p["steps_per_phase"] * p["num_envs"]
    if expected_cap != p["max_physics_controls"]:
        raise ValueError("phase budget mismatch")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "source/rl_training/rl_training").rglob("*.py")) + sorted((root / "scripts/reinforcement_learning").rglob("*.py"))
    sources = {str(f.relative_to(root)): sha256_file(f) for f in paths}
    protocol_hash = sha256_file(args.protocol)
    ledger = VectorStepLedger(out / "ledger.json", num_envs=p["num_envs"], max_transitions=expected_cap,
                             metadata={"protocol_sha256": protocol_hash})
    report = dict(status="running", protocol_sha256=protocol_hash, sources=sources, rounds=[], evaluation={},
                  algorithm="single-action conditional flow + discrete inversion + dual-buffer DAgger",
                  scope="development pilot with PPO expert and simulator route localization")
    write_json_atomic(out / "report.json", report)
    app = None
    started = time.perf_counter()
    try:
        flow_dir = Path(args.flow_directory)
        base_report = json.loads((flow_dir / "report.json").read_text())
        if base_report["status"] != "complete" or not base_report["engineering_gate_passed"]:
            raise ValueError("flow foundation failed its engineering gate")
        foundation_protocol = p.get("foundation_protocol_sha256", protocol_hash)
        if base_report["protocol_sha256"] != foundation_protocol or sha256_file(flow_dir / "final.pt") != base_report["checkpoint_sha256"]:
            raise ValueError("flow checkpoint/protocol mismatch")
        if p.get("foundation_checkpoint_sha256", base_report["checkpoint_sha256"]) != base_report["checkpoint_sha256"]:
            raise ValueError("registered foundation checkpoint differs")
        if sha256_file(p["teacher_checkpoint"]) != p["teacher_sha256"]:
            raise ValueError("teacher checksum mismatch")
        app = AppLauncher(args)
        stop_requested = []
        def request_stop(signum, frame):
            # Kit's default SIGINT handler can exit0 without finalizing Python
            # reports. Finish the current charged transition, then abort cleanly.
            stop_requested.append(signum)
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        import gymnasium as gym
        import numpy as np
        import torch
        from isaacsim.core.utils.stage import create_new_stage
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        import rl_training.tasks  # noqa: F401
        from rl_training.policy_route import configure_route_observations
        from rl_training.lite3_joint_conventions import clip_lite3_policy_action
        from rl_training.online_adaptation.step_trajectory import StepTrajectory
        from rl_training.online_adaptation.flowdagger import ConditionalFlow, FrozenPPOExpert, NoisePolicy, DualBuffer, balanced_batch, invert_euler, qualify_paired_correction, preservation_loss, save_adapter_checkpoint, widen_noise_policy, correction_region_mask, lookahead_teacher_observation
        from rl_training.online_adaptation.runner_contract import configure_forward_command, actuator_delay_metadata

        torch.set_num_threads(4)
        torch.manual_seed(p["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        payload = torch.load(flow_dir / "final.pt", map_location=args.device, weights_only=False)
        flow = ConditionalFlow(**payload["config"]).to(args.device)
        flow.load_state_dict(payload["flow"], strict=True)
        flow.requires_grad_(False).eval()
        teacher = FrozenPPOExpert(torch.load(p["teacher_checkpoint"], map_location="cpu", weights_only=False)["actor_state_dict"]).to(args.device)
        actor = NoisePolicy(bound=p["adapter"]["noise_bound"], hidden=tuple(p["adapter"].get("hidden", (256,128)))).to(args.device)
        previous_actor = None
        if p.get("comparison_checkpoint"):
            if sha256_file(p["comparison_checkpoint"]) != p["comparison_sha256"]:
                raise ValueError("comparison checkpoint mismatch")
            previous_payload = torch.load(p["comparison_checkpoint"], map_location=args.device, weights_only=False)
            if previous_payload["foundation_sha256"] != base_report["checkpoint_sha256"]:
                raise ValueError("comparison must use the same frozen flow")
            previous_actor = NoisePolicy(bound=previous_payload["noise_bound"], hidden=tuple(previous_payload.get("noise_hidden", (256,128)))).to(args.device)
            previous_actor.load_state_dict(previous_payload["actor"], strict=True)
            previous_actor.eval().requires_grad_(False)
            previous_initial = {k: v.clone() for k, v in previous_actor.state_dict().items()}
        if p.get("warmstart_comparison"):
            if previous_actor is None or previous_actor.bound != actor.bound:
                raise ValueError("warmstart requires a compatible pinned comparison adapter")
            if actor.hidden == previous_actor.hidden:
                actor.load_state_dict(previous_actor.state_dict(), strict=True)
            else:
                copies = actor.hidden[0] // previous_actor.hidden[0]
                if actor.hidden != tuple(v*copies for v in previous_actor.hidden):
                    raise ValueError("only uniform function-preserving widening is supported")
                widened = widen_noise_policy(previous_actor, copies, seed=p["seed"])
                actor.load_state_dict(widened.state_dict(), strict=True)
                probe = torch.linspace(-1,1,48*128,device=args.device).reshape(128,48)
                with torch.no_grad():
                    error = float((actor(probe)-previous_actor(probe)).abs().max())
                if error > 1.e-5:
                    raise ValueError("initial widened function mismatch")
                report["widening_initial_max_error"] = error
            report["adapter_initialization"] = dict(kind="pinned_previous_adapter", sha256=p["comparison_sha256"],
                                                     optimizer_and_buffer_restored=False)
        flow_initial = {k: v.clone() for k, v in flow.state_dict().items()}
        teacher_initial = {k: v.clone() for k, v in teacher.state_dict().items()}
        actor_initial = {k: v.clone() for k, v in actor.state_dict().items()}
        initial_actor = None
        if p.get("evaluate_initial_adapter"):
            initial_actor = NoisePolicy(bound=actor.bound, hidden=actor.hidden).to(args.device)
            initial_actor.load_state_dict(actor_initial)
            initial_actor.eval().requires_grad_(False)
            torch.save(dict(actor=actor_initial, noise_hidden=list(actor.hidden), noise_bound=actor.bound), out/"initial_adapter.pt")
            report["initial_adapter_sha256"] = sha256_file(out/"initial_adapter.pt")
        optimizer = torch.optim.Adam(actor.parameters(), lr=p["adapter"]["learning_rate"])
        buffer = DualBuffer()
        training_generator = torch.Generator(device=args.device).manual_seed(p["seed"] + 2)

        def run_phase(name, mode, seed):
            """One predeclared first episode per vector slot; padding is charged."""
            folder = out / name
            folder.mkdir(exist_ok=False)
            env = None
            trajectory = None
            phase_start = ledger.state["confirmed"]
            phase = dict(status="running", mode=mode, seed=seed, teacher_query_rows=0, eligible_teacher_label_rows=0, interventions=0)
            write_json_atomic(folder / "report.json", phase)
            saved_rows = []
            try:
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                create_new_stage()
                cfg = parse_env_cfg(p["task"], device=args.device, num_envs=p["num_envs"])
                cfg.seed = seed
                configure_route_observations(cfg)
                configure_forward_command(cfg.commands.base_velocity, .3)
                env = gym.make(p["task"], cfg=cfg)
                raw = env.unwrapped
                wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
                obs = wrapped.get_observations()["policy"]
                if obs.shape != (p["num_envs"], 48):
                    raise ValueError("route observation contract mismatch")
                expected_command = obs.new_tensor([.3, 0., 0.]).expand(len(obs), 3)
                torch.testing.assert_close(obs[:, 6:9], expected_command, rtol=0., atol=1.e-7)
                robot = raw.scene["robot"]
                origin = raw.scene.env_origins.clone()
                initial_y = robot.data.root_pos_w[:, 1].clone()
                initial = dict(root_state=robot.data.root_state_w.cpu().numpy().copy(),
                    joint_pos=robot.data.joint_pos.cpu().numpy().copy(), joint_vel=robot.data.joint_vel.cpu().numpy().copy(),
                    observation=obs.cpu().numpy().copy(), origins=origin.cpu().numpy().copy(),
                    terrain_types=raw.scene.terrain.terrain_types.cpu().numpy().copy())
                np.savez_compressed(folder / "initial_state.npz", **initial)
                delays = actuator_delay_metadata(robot)
                write_json_atomic(folder / "initial_actuator_delays.json", delays)
                phase["initial_state_sha256"] = hashlib.sha256(b"".join(v.tobytes() for v in initial.values())
                    + json.dumps(delays, sort_keys=True).encode()).hexdigest()
                active = torch.ones(p["num_envs"], device=args.device, dtype=torch.bool)
                first_end = [None] * p["num_envs"]
                peak = torch.zeros(p["num_envs"], device=args.device)
                peak_initial = peak.clone(); lateral_squared = peak.clone(); samples = peak.clone()
                captured = {}
                original_compute = raw.termination_manager.compute

                def capture_termination():
                    result = original_compute()
                    # This callback runs after physics and BEFORE automatic resets.
                    captured["root"] = robot.data.root_state_w.detach().clone()
                    return result

                raw.termination_manager.compute = capture_termination
                trajectory = StepTrajectory(folder / "trajectory", metadata={"phase": name, "mode": mode,
                    "active": "first episode only; inactive physical padding still charged", "root": "post-action, pre-reset"},
                    chunk_steps=128, vector_dtype="float32")
                inference_ms = []
                for step in range(p["steps_per_phase"]):
                    if stop_requested:
                        raise InterruptedError("stop requested; no physical transition pending")
                    before = obs.detach().clone()
                    before_active = active.clone()
                    with torch.no_grad():
                        teacher_observation = (lookahead_teacher_observation(obs, lookahead_m=p["teacher_route_lookahead_m"])
                            if p.get("teacher_route_lookahead_m") is not None else obs)
                        teacher_route_features = teacher_observation[:,9:12].clone()
                        # End-to-end inference timing excludes teacher querying and inverse fitting.
                        torch.cuda.synchronize(args.device)
                        tick = time.perf_counter()
                        if mode == "flow":
                            noise = torch.zeros((len(obs), 12), device=args.device)
                        elif mode == "previous":
                            noise = previous_actor(obs)
                        elif mode == "initial":
                            noise = initial_actor(obs)
                        else:
                            noise = actor(obs)
                        if mode == "teacher":
                            requested = teacher(teacher_observation)
                        else:
                            requested = flow(obs, noise)
                        torch.cuda.synchronize(args.device)
                        inference_ms.append((time.perf_counter() - tick) * 1000)
                        expert_action = teacher(teacher_observation)
                        student_emitted = clip_lite3_policy_action(requested)
                        expert_emitted = clip_lite3_policy_action(expert_action)
                        disagreement = (student_emitted-expert_emitted).square().mean(-1).sqrt()
                        region = correction_region_mask(obs, **p["correction_region"]) if p.get("correction_region") else torch.ones_like(active)
                        intervene = before_active & region & (disagreement > p["intervention_action_rmse"]) if mode == "collect" else torch.zeros_like(active)
                        executed_request = torch.where(intervene[:, None], expert_action, requested)
                        emitted = clip_lite3_policy_action(executed_request)
                    if not torch.isfinite(emitted).all():
                        raise ValueError("nonfinite deployment action")
                    phase["teacher_query_rows"] += len(obs) * (2 if mode == "teacher" else 1)
                    phase["eligible_teacher_label_rows"] += int(before_active.sum())
                    phase["interventions"] += int(intervene.sum())
                    ledger.reserve()
                    next_obs, reward, done, extras = wrapped.step(emitted)
                    ledger.confirm()
                    torch.testing.assert_close(next_obs["policy"][:, 6:9], expected_command, rtol=0., atol=1.e-7)
                    expected_history = emitted.clone(); expected_history[done.bool()] = 0
                    torch.testing.assert_close(next_obs["policy"][:, -12:], expected_history, atol=1.e-6, rtol=0)
                    torch.testing.assert_close(next_obs["critic"][:, -12:], expected_history, atol=1.e-6, rtol=0)
                    manager = raw.termination_manager
                    failure = manager.get_term("illegal_contact") | manager.get_term("bad_orientation_2") | manager.get_term("course_lateral")
                    success = manager.get_term("course_success") & ~failure
                    deadline = manager.get_term("time_out") & ~success & ~failure
                    if not torch.equal(done.bool(), success | failure | deadline):
                        raise ValueError("termination contract differs")
                    post_root = captured["root"]
                    lateral = post_root[:, 1] - origin[:, 1]
                    peak = torch.maximum(peak, torch.where(before_active, lateral.abs(), 0.))
                    peak_initial = torch.maximum(peak_initial, torch.where(before_active, (post_root[:, 1]-initial_y).abs(), 0.))
                    lateral_squared += torch.where(before_active, lateral.square(), 0.)
                    samples += before_active
                    snapshot = trajectory.snapshot(policy_observation=before, noise=noise, student_requested_action=requested,
                        teacher_requested_action=expert_action, emitted_action=emitted, intervention=intervene,
                        correction_region=region, action_disagreement=disagreement,
                        teacher_route_features=teacher_route_features,
                        active=before_active, done=done, success=success, failure=failure, deadline=deadline,
                        post_root_state=post_root, reward=reward)
                    trajectory.append(**snapshot)
                    if mode in ("collect", "reference"):
                        saved_rows.append({k: snapshot[k] for k in ("policy_observation", "noise", "teacher_requested_action", "intervention", "active", "post_root_state")})
                    for index in torch.where(before_active & done.bool())[0].tolist():
                        terrain_type = int(raw.scene.terrain.terrain_types[index])
                        first_end[index] = dict(env_id=index, steps=step+1, success=bool(success[index]),
                            failure=bool(failure[index]), deadline=bool(deadline[index]),
                            terrain="flat" if terrain_type < 2 else ("101" if terrain_type < 5 else "107"),
                            progress_m=float(post_root[index, 0]-origin[index, 0]),
                            max_body_lateral_m=float(peak[index]), max_body_offset_from_start_m=float(peak_initial[index]),
                            body_lateral_rms_m=float((lateral_squared[index]/samples[index]).sqrt()))
                    active &= ~done.bool()
                    obs = next_obs["policy"]
                    if (step + 1) % 250 == 0:
                        print(json.dumps(dict(phase=name, step=step+1, remaining=int(active.sum()),
                            interventions=phase["interventions"], charged=ledger.state["charged"])), flush=True)
                        phase.update(vector_steps=step+1, completed_episodes=sum(v is not None for v in first_end))
                        write_json_atomic(folder / "report.json", phase)
                    if not active.any():
                        break
                for index, item in enumerate(first_end):
                    if item is None:
                        terrain_type = int(raw.scene.terrain.terrain_types[index])
                        first_end[index] = dict(env_id=index, steps=step+1, success=False, failure=False, deadline=False,
                            collector_cutoff=True, terrain="flat" if terrain_type < 2 else ("101" if terrain_type < 5 else "107"),
                            progress_m=float(post_root[index, 0]-origin[index, 0]), max_body_lateral_m=float(peak[index]),
                            max_body_offset_from_start_m=float(peak_initial[index]),
                            body_lateral_rms_m=float((lateral_squared[index]/samples[index]).sqrt()))
                trajectory.close(complete=True)
                phase.update(status="complete", episodes=first_end, vector_steps=step+1,
                    confirmed_controls=ledger.state["confirmed"]-phase_start,
                    inference_ms_median=float(np.median(inference_ms)), inference_ms_p99=float(np.percentile(inference_ms, 99)),
                    inference_ms_max=float(max(inference_ms)), inference_over_20ms=int(np.sum(np.array(inference_ms)>20)),
                    by_terrain={name: {"success": sum(r["success"] for r in first_end if r["terrain"] == name),
                                      "total": sum(r["terrain"] == name for r in first_end),
                                      "max_body_lateral_m": max(r["max_body_lateral_m"] for r in first_end if r["terrain"] == name)}
                                for name in ("flat", "101", "107")})
                write_json_atomic(folder / "report.json", phase)
                return phase, saved_rows
            except BaseException as error:
                phase.update(status="failed", error=repr(error), confirmed_controls=ledger.state["confirmed"]-phase_start)
                write_json_atomic(folder / "report.json", phase)
                if trajectory is not None:
                    trajectory.close(complete=False)
                raise
            finally:
                if env is not None:
                    env.close()

        for round_id in range(p["rounds"]):
            reference_phase, reference_data = None, None
            if paired:
                reference_phase, reference_rows = run_phase(f"reference-{round_id:02d}", "reference", p["collection_seed"] + round_id)
                reference_data = {key: torch.from_numpy(np.stack([row[key] for row in reference_rows])) for key in reference_rows[0]}
                for episode in reference_phase["episodes"]:
                    i, length = episode["env_id"], episode["steps"]
                    valid_reference = torch.ones(length, dtype=torch.bool)
                    if p.get("correction_region"):
                        valid_reference &= ~correction_region_mask(reference_data["policy_observation"][:length, i], **p["correction_region"])
                    buffer.add_episode(f"{round_id}:reference:{i}", successful=episode["success"],
                        observations=reference_data["policy_observation"][:length, i],
                        noises=reference_data["noise"][:length, i], interventions=torch.zeros(length, dtype=torch.bool),
                        valid=valid_reference)
            phase, rows = run_phase(f"round-{round_id:02d}", "collect", p["collection_seed"] + round_id)
            if paired and phase["initial_state_sha256"] != reference_phase["initial_state_sha256"]:
                raise ValueError("corrected/reference initial state or actuator delay differs")
            data = {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]}
            inverse_rows, rejected, eligible = [], 0, 0
            qualifications = []
            for episode in phase["episodes"]:
                i, length = episode["env_id"], episode["steps"]
                obs = data["policy_observation"][:length, i]
                labels = data["noise"][:length, i].clone()
                interventions = data["intervention"][:length, i]
                valid = torch.ones(length, dtype=torch.bool)
                qualified = episode["success"]
                if paired:
                    # No correction labels are admitted from a branch without a
                    # measurable completion/drift benefit over its paired student.
                    reference_episode = reference_phase["episodes"][i]
                    first = int(torch.where(interventions)[0][0]) if interventions.any() else 0
                    prefix = 0.
                    if first >= reference_episode["steps"]:
                        prefix = float("inf")
                    elif interventions.any():
                        prefix = float((data["policy_observation"][:first+1, i]-reference_data["policy_observation"][:first+1, i]).abs().max())
                        if first:
                            prefix = max(prefix, float((data["post_root_state"][:first, i]-reference_data["post_root_state"][:first, i]).abs().max()))
                    if not np.isfinite(prefix):
                        qualified, reason = False, "reference_ended_before_intervention"
                    else:
                        qualified, reason = qualify_paired_correction(reference_episode, episode,
                            prefix_error=prefix, intervention_count=int(interventions.sum()), **paired["qualification"])
                    qualifications.append(dict(env_id=i, qualified=qualified, reason=reason,
                        prefix_max_abs=prefix if np.isfinite(prefix) else None, intervention_count=int(interventions.sum()),
                        reference=reference_episode, corrected=episode))
                    # All preservation data come from complete unassisted reference
                    # episodes. The corrected branch contributes corrections only.
                    valid &= interventions
                if qualified and interventions.any():
                    indices = torch.where(interventions)[0]
                    for block in indices.split(512):
                        result = invert_euler(flow, obs[block].to(args.device),
                            data["teacher_requested_action"][block, i].to(args.device), **p["inversion"])
                        labels[block] = result.noise.cpu()
                        valid[block] = result.accepted.cpu()
                        rejected += int((~result.accepted).sum()); eligible += len(block)
                        inverse_rows.append(dict(env_id=np.full(len(block), i), step=block.numpy(), noise=result.noise.cpu().numpy(),
                            max_error=result.max_error.cpu().numpy(), accepted=result.accepted.cpu().numpy()))
                buffer.add_episode(f"{round_id}:{i}", successful=qualified, observations=obs,
                                   noises=labels, interventions=interventions, valid=valid)
            if paired:
                write_json_atomic(out / f"round-{round_id:02d}" / "qualification.json", qualifications)
            inverse_path = out / f"round-{round_id:02d}" / "inversions.npz"
            if inverse_rows:
                np.savez_compressed(inverse_path, **{k: np.concatenate([r[k] for r in inverse_rows]) for k in inverse_rows[0]})
            info = dict(round=round_id, rollout=phase["by_terrain"], intervention_rows=phase["interventions"],
                        inversion_attempted=eligible, inversion_rejected=rejected, buffer_counts=buffer.counts(), updates=0)
            if paired:
                info.update(reference=reference_phase["by_terrain"],
                    qualified_episodes=sum(v["qualified"] for v in qualifications),
                    qualification_reasons={reason: sum(v["reason"] == reason for v in qualifications)
                                           for reason in sorted({v["reason"] for v in qualifications})})
            report["rounds"].append(info)
            write_json_atomic(out / "report.json", report)
            if paired and (min(buffer.counts().values()) < paired["minimum_buffer_rows"]
                           or sum(v["qualified_episodes"] for v in report["rounds"]) < paired["minimum_qualified_episodes"]):
                info["training_skipped_reason"] = "insufficient_verified_corrections; do not invent teacher benefit"
                save_adapter_checkpoint(out / f"round-{round_id:02d}" / "adapter.pt", actor, optimizer, buffer,
                    training_generator, round_id=round_id, updates=0, protocol_sha256=protocol_hash,
                    skipped_reason=info["training_skipped_reason"])
                write_json_atomic(out / "report.json", report)
                continue
            training = buffer.tensors(args.device)  # Refuses to invent an absent half of the mixture.
            if p.get("correction_region"):
                if not correction_region_mask(training["correction"][0], **p["correction_region"]).all():
                    raise ValueError("correction label outside corrective region")
                if correction_region_mask(training["autonomous"][0], **p["correction_region"]).any():
                    raise ValueError("preservation label conflicts with corrective region")
            for update in range(p["adapter"]["updates_per_round"]):
                batch_obs, batch_noise = balanced_batch(training, p["adapter"]["batch_size"], training_generator)
                loss, noise_loss, action_loss = preservation_loss(flow, actor, batch_obs, batch_noise,
                    weight=p["adapter"].get("action_preservation_weight", 0.))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite adapter objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                info["updates"] = update + 1
                if (update + 1) % 250 == 0:
                    event = dict(round=round_id, update=update+1, noise_mse=float(noise_loss),
                                 autonomous_action_mse=float(action_loss), total_loss=float(loss))
                    with (out / "adapter_updates.jsonl").open("a") as stream:
                        stream.write(json.dumps(event)+"\n")
                    print(json.dumps(event), flush=True)
            info["final_minibatch_noise_mse"] = float(noise_loss)
            info["final_minibatch_autonomous_action_mse"] = float(action_loss)
            save_adapter_checkpoint(out / f"round-{round_id:02d}" / "adapter.pt", actor, optimizer, buffer,
                training_generator, round_id=round_id, updates=info["updates"], protocol_sha256=protocol_hash)
            write_json_atomic(out / "report.json", report)
        actor.eval().requires_grad_(False)
        frozen_actor = {k: v.clone() for k, v in actor.state_dict().items()}
        torch.save(dict(schema="lite3_flowdagger_v1", flow_config=flow.config, flow=flow.state_dict(),
                        actor=actor.state_dict(), noise_bound=actor.bound, protocol_sha256=protocol_hash,
                        noise_hidden=list(actor.hidden),
                        expert_sha256=p["teacher_sha256"], foundation_sha256=base_report["checkpoint_sha256"],
                        teacher_route_lookahead_m=p.get("teacher_route_lookahead_m"),
                        observation_schema="sim-localization-v1", action_schema="lite3_joint_target_offsets_v1"), out / "final.pt")
        for mode in evaluation_modes:
            phase, _ = run_phase(f"evaluation-{mode}", mode, p["evaluation_seed"])
            report["evaluation"][mode] = {k: phase[k] for k in ("by_terrain", "initial_state_sha256", "inference_ms_p99", "inference_ms_max", "inference_over_20ms")}
            write_json_atomic(out / "report.json", report)
        if len({v["initial_state_sha256"] for v in report["evaluation"].values()}) != 1:
            raise ValueError("paired evaluation initial states differ")
        for initial, model in ((flow_initial, flow), (teacher_initial, teacher), (frozen_actor, actor)):
            if not all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items()):
                raise ValueError("frozen weights changed")
        if previous_actor is not None and not all(torch.equal(v, previous_actor.state_dict()[k]) for k, v in previous_initial.items()):
            raise ValueError("comparison policy changed")
        if initial_actor is not None and not all(torch.equal(v, initial_actor.state_dict()[k]) for k,v in actor_initial.items()):
            raise ValueError("untrained widened comparison changed")
        if sources != {str(f.relative_to(root)): sha256_file(f) for f in paths}:
            raise ValueError("runtime source changed during managed run")
        if sha256_file(p["teacher_checkpoint"]) != p["teacher_sha256"] or sha256_file(args.protocol) != protocol_hash:
            raise ValueError("protocol or original teacher changed")
        ledger.finish()
        report.update(status="complete", ledger=ledger.state, seconds=time.perf_counter()-started,
            frozen_flow_and_teacher_unchanged=True, evaluation_actor_unchanged=True, initial_states_paired=True,
            actor_changed=any(not torch.equal(v, actor.state_dict()[k]) for k, v in actor_initial.items()),
            adapter_updates=sum(v["updates"] for v in report["rounds"]),
            final_checkpoint_sha256=sha256_file(out / "final.pt"))
        write_json_atomic(out / "report.json", report)
    except BaseException as error:
        if ledger.state["status"] == "running":
            ledger.finish(aborted=True)
        report.update(status="failed", error=repr(error), ledger=ledger.state, seconds=time.perf_counter()-started)
        write_json_atomic(out / "report.json", report)
        raise
    finally:
        if app is not None:
            app.app.close()


if __name__ == "__main__":
    main()
