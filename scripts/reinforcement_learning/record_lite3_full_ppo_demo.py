"""New full-PPO vs original-flat demonstration; no training or model selection.

Derived from the audited recovery evaluation runner. FrozenFlatMean supports
canonical 45D/48D policies with exact route removal for the original actor.
"""
import argparse
import hashlib
import json
import random
import signal
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
    from rl_training.vector_step_ledger import VectorStepLedger
    p = json.loads(Path(args.protocol).read_text())
    video_options=p.get("video")
    if video_options:
        if not 0 <= video_options["env_id"] < p["num_envs"] or video_options["stride"] < 1:
            raise ValueError("invalid video environment or frame stride")
        args.enable_cameras=True
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = {str(f.relative_to(root)): sha256_file(f) for folder in (root/"source/rl_training/rl_training", root/"scripts/reinforcement_learning") for f in sorted(folder.rglob("*.py"))}
    cap = len(p["models"])*len(p["conditions"])*p["num_envs"]*p["steps"]
    assert cap == p["max_controls"]
    ledger = VectorStepLedger(out/"ledger.json", num_envs=p["num_envs"], max_transitions=cap,
                             metadata={"protocol_sha256":sha256_file(args.protocol)})
    report = dict(status="running", protocol_sha256=sha256_file(args.protocol), sources=sources, phases={})
    write_json_atomic(out/"report.json", report)
    launcher = None; env = None; trace = None; video = None
    try:
        launcher = AppLauncher(args)
        stop_requested=[]
        def request_stop(signum,frame):stop_requested.append(signum)
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        import numpy as np
        import torch
        import gymnasium as gym
        from isaacsim.core.utils.stage import create_new_stage
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        import rl_training.tasks  # noqa
        from rl_training.policy_route import configure_route_observations
        from rl_training.policy_route_history import RouteHistoryVecEnv, strip_route_history
        from rl_training.recovery_teacher import configure_recovery_resets, check_recovery_event_order
        from rl_training.online_adaptation.runner_contract import configure_forward_command, actuator_delay_metadata
        from rl_training.online_adaptation.flowdagger import ConditionalFlow, NoisePolicy
        from rl_training.policy_flat_anchor import FrozenFlatMean
        from rl_training.lite3_joint_conventions import clip_lite3_policy_action
        from rl_training.online_adaptation.step_trajectory import StepTrajectory
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        prefix=None
        if p.get("action_prefix"):
            from rl_training.recovery_sequence_search import prefix_at_step
            spec=p["action_prefix"]
            assert sha256_file(spec["path"])==spec["sha256"]
            prefix=np.load(spec["path"],allow_pickle=False)
            assert prefix.shape==(p["num_envs"],spec["knots"],12) and np.isfinite(prefix).all()
            assert np.abs(prefix).max()<=spec["bound"] and spec["hold_steps"]>=1
            assert p["recovery_resets"]["split"]=="train" and p["recovery_resets"].get("fixed_train_case") is not None
        models = {}
        for name, spec in p["models"].items():
            expected = spec.get("sha256")
            if "training_report" in spec:
                trained = json.loads(Path(spec["training_report"]).read_text())
                assert trained["status"] == "complete" and trained["protocol_sha256"] == spec["training_protocol_sha256"]
                assert Path(trained["final_checkpoint"]["path"]).resolve() == Path(spec["path"]).resolve()
                expected = trained["final_checkpoint"]["sha256"]
            assert sha256_file(spec["path"]) == expected
            payload = torch.load(spec["path"], map_location="cpu", weights_only=False)
            if spec["kind"] == "ppo":
                model = FrozenFlatMean(payload).to(args.device).eval().requires_grad_(False)
            else:
                flow = ConditionalFlow(**payload["flow_config"]).to(args.device).eval().requires_grad_(False)
                flow.load_state_dict(payload["flow"])
                actor = NoisePolicy(bound=payload["noise_bound"], hidden=tuple(payload.get("noise_hidden",(256,128)))).to(args.device).eval().requires_grad_(False)
                actor.load_state_dict(payload["actor"])
                model = (flow, actor)
            models[name] = model
            spec["actual_sha256"] = expected
        report["models"] = p["models"]
        for condition in p["conditions"]:
            for name, model in models.items():
                torch.manual_seed(p["evaluation_seed"]); np.random.seed(p["evaluation_seed"]); random.seed(p["evaluation_seed"])
                create_new_stage()
                cfg = parse_env_cfg(p["task"], device=args.device, num_envs=p["num_envs"])
                cfg.seed = p["evaluation_seed"]
                configure_route_observations(cfg); configure_forward_command(cfg.commands.base_velocity,.3)
                if video_options:
                    cfg.viewer.origin_type="asset_root";cfg.viewer.asset_name="robot"
                    cfg.viewer.env_index=video_options["env_id"]
                    cfg.viewer.eye=tuple(video_options.get("eye",(-2.2,-2.2,1.5)))
                    cfg.viewer.lookat=tuple(video_options.get("lookat",(.45,0.,0.)))
                    cfg.viewer.resolution=(video_options["width"],video_options["height"])
                if condition != "course":
                    configure_recovery_resets(cfg,p["recovery_resets"])
                env = gym.make(p["task"], cfg=cfg,render_mode="rgb_array" if video_options else None); raw = env.unwrapped
                if condition != "course":
                    check_recovery_event_order(raw)
                wrapped = RslRlVecEnvWrapper(env,clip_actions=None)
                history_options=p.get('route_history')
                if history_options is not None:
                    assert all(s['kind']=='ppo' for s in p['models'].values())
                    wrapped=RouteHistoryVecEnv(wrapped,history_options)
                obs = wrapped.get_observations()["policy"]
                robot = raw.scene["robot"]; origins = raw.scene.env_origins.clone()
                folder = out/f"{condition}-{name}"; folder.mkdir()
                if video_options:
                    from pxr import UsdGeom
                    from full_ppo_demo_video import ComparisonVideo as SimulationVideo
                    # Only visual visibility changes. Other actors continue
                    # their registered physical trials and remain in the ledger.
                    for i in range(p["num_envs"]):
                        if i!=video_options["env_id"]:
                            UsdGeom.Imageable(raw.sim.stage.GetPrimAtPath(f"/World/envs/env_{i}")).MakeInvisible()
                    # Cloned environments inherit env_0 properties. Override
                    # visibility locally after hiding the source environment.
                    selected_prim=raw.sim.stage.GetPrimAtPath(f"/World/envs/env_{video_options['env_id']}")
                    UsdGeom.Imageable(selected_prim).MakeVisible()
                    assert UsdGeom.Imageable(selected_prim).ComputeVisibility()!=UsdGeom.Tokens.invisible
                    for _ in range(10):raw.render()
                    video=SimulationVideo(folder/"demo.mp4",width=video_options["width"],height=video_options["height"],
                        fps=1/(raw.step_dt*video_options["stride"]),label=video_options["label"])
                cases = raw._recovery_case.clone() if condition != "course" else torch.full((p["num_envs"],),-1,device=args.device)
                initial = dict(root=robot.data.root_state_w.cpu().numpy().copy(), origins=origins.cpu().numpy(),
                    observation=strip_route_history(obs).cpu().numpy(),joint_pos=robot.data.joint_pos.cpu().numpy(),
                    joint_vel=robot.data.joint_vel.cpu().numpy(),terrain=raw.scene.terrain.terrain_types.cpu().numpy(),case=cases.cpu().numpy())
                np.savez_compressed(folder/"initial.npz",**initial)
                if history_options is not None:
                    np.savez_compressed(folder/'initial_history.npz',policy_observation=obs.cpu().numpy(),
                        **{k:v.cpu().numpy() for k,v in wrapped.history_snapshot().items()})
                delays=actuator_delay_metadata(robot); write_json_atomic(folder/"delays.json",delays)
                initial_hash=hashlib.sha256(b"".join(v.tobytes() for v in initial.values())+json.dumps(delays,sort_keys=True).encode()).hexdigest()
                if condition != "course":
                    chosen=cases>=0
                    minimum_tilt = p.get("minimum_initial_tilt_deg", 19.9)
                    assert chosen.any() and (obs[chosen,5] > -np.cos(np.deg2rad(minimum_tilt))).all()
                    assert not bool(getattr(raw,"startup_active",False))
                if prefix is not None:
                    assert np.all(prefix[cases.cpu().numpy()<0]==0)
                active=torch.ones(p["num_envs"],device=args.device,dtype=torch.bool)
                peak=torch.zeros(p["num_envs"],device=args.device); squares=peak.clone(); count=peak.clone()
                streak=torch.zeros(p["num_envs"],device=args.device,dtype=torch.long); recovered=torch.zeros_like(active)
                diagnostic = bool(p.get("contact_diagnostics"))
                if diagnostic:
                    feet = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
                    foot_ids, foot_names = robot.find_bodies(feet, preserve_order=True)
                    sensor = raw.scene["contact_forces"]
                    contact_ids, contact_names = sensor.find_bodies(feet, preserve_order=True)
                    assert list(foot_names) == list(contact_names) == feet
                    from rl_training.lite3_joint_conventions import LITE3_POLICY_JOINT_NAMES
                    torque_ids, torque_names=robot.find_joints(list(LITE3_POLICY_JOINT_NAMES),preserve_order=True)
                    assert tuple(torque_names)==LITE3_POLICY_JOINT_NAMES
                    write_json_atomic(folder/"diagnostic_metadata.json", dict(
                        foot_order=feet, joint_order=list(torque_names), step_dt=raw.step_dt, timing="post-control, before automatic reset",
                        contact_measurement="last physics substep net force, not a static support polygon",
                        root_com="base-link COM, not whole-robot COM"))
                episodes=[None]*p["num_envs"]; captured={}; compute=raw.termination_manager.compute
                def capture():
                    value=compute();captured["root"]=robot.data.root_state_w.detach().clone()
                    if diagnostic:
                        captured["diagnostics"] = dict(
                            foot_position_w=robot.data.body_pos_w[:,foot_ids].detach().clone(),
                            foot_contact_force_w=sensor.data.net_forces_w[:,contact_ids].detach().clone(),
                            base_com_position_w=robot.data.root_com_pos_w.detach().clone(),
                            joint_computed_torque=robot.data.computed_torque[:,torque_ids].detach().clone(),
                            joint_applied_torque=robot.data.applied_torque[:,torque_ids].detach().clone())
                    return value
                raw.termination_manager.compute=capture
                trace=StepTrajectory(folder/"trajectory",metadata={"condition":condition,"model":name,"root":"post-step pre-reset"},chunk_steps=128,vector_dtype="float32")
                phase_start=ledger.state["confirmed"]
                for step in range(p["steps"]):
                    if stop_requested:raise InterruptedError("stop requested before next physical step")
                    torch.testing.assert_close(obs[:,6:9],obs.new_tensor([.3,0.,0.]).expand(len(obs),3),atol=1e-7,rtol=0)
                    with torch.no_grad():
                        requested=model[0](obs,model[1](obs)) if isinstance(model,tuple) else model(obs)
                        base_requested=requested
                        if prefix is not None:
                            offset=torch.as_tensor(prefix_at_step(prefix,step,p["action_prefix"]["hold_steps"]),device=args.device)
                            requested=base_requested+offset
                        emitted=clip_lite3_policy_action(requested)
                    before=obs.clone(); before_active=active.clone()
                    before_history=wrapped.history_snapshot() if history_options is not None else {}
                    if video is not None and bool(active[video_options["env_id"]]) and step%video_options["stride"]==0:
                        i=video_options["env_id"];t=int(initial["terrain"][i])
                        video.append(raw.render(),step=step,time_s=step*raw.step_dt,
                            lateral_m=float(robot.data.root_pos_w[i,1]-origins[i,1]),
                            tilt_deg=float(torch.rad2deg(torch.acos((-obs[i,5]).clamp(-1,1)))),
                            terrain="flat" if t<2 else (101 if t<5 else 107),env_id=i)
                    ledger.reserve(); next_obs,reward,done,_=wrapped.step(emitted);ledger.confirm()
                    history=emitted.clone();history[done.bool()]=0
                    torch.testing.assert_close(next_obs["policy"][:,-12:],history,atol=1e-6,rtol=0)
                    manager=raw.termination_manager
                    terms={k:manager.get_term(v).clone() for k,v in dict(contact="illegal_contact",orientation="bad_orientation_2",lateral="course_lateral").items()}
                    failure=terms["contact"]|terms["orientation"]|terms["lateral"]
                    success=manager.get_term("course_success")&~failure
                    deadline=manager.get_term("time_out")&~failure&~success
                    assert torch.equal(done.bool(),success|failure|deadline)
                    state=captured["root"]; lateral=state[:,1]-origins[:,1]
                    peak=torch.maximum(peak,torch.where(active,lateral.abs(),0.));squares+=torch.where(active,lateral.square(),0.);count+=active
                    upright=1-2*state[:,4:6].square().sum(-1) >= np.cos(np.deg2rad(10.))
                    streak=torch.where(active&upright,streak+1,0);recovered|=streak>=25
                    trace.append(**trace.snapshot(policy_observation=before,requested_action=requested,emitted_action=emitted,
                        active=before_active,done=done,success=success,failure=failure,deadline=deadline,post_root_state=state,
                        **{f"failure_{k}":v for k,v in terms.items()},
                        **(captured["diagnostics"] if diagnostic else {}),
                        **before_history,
                        **(dict(base_requested_action=base_requested,search_offset=offset) if prefix is not None else {})))
                    for i in torch.where(active&done.bool())[0].tolist():
                        t=int(initial["terrain"][i]);episodes[i]=dict(env_id=i,terrain="flat" if t<2 else str(101 if t<5 else 107),
                            case=int(cases[i]),steps=step+1,success=bool(success[i]),recovery_success=bool(success[i]&recovered[i]),
                            failure_terms={k:bool(v[i]) for k,v in terms.items()},deadline=bool(deadline[i]),
                            max_body_lateral_m=float(peak[i]),body_lateral_rms_m=float((squares[i]/count[i]).sqrt()))
                    active&=~done.bool();obs=next_obs["policy"]
                    if (step+1)%250==0:
                        print(json.dumps(dict(phase=f"{condition}-{name}",step=step+1,remaining=int(active.sum()),charged=ledger.state["charged"])),flush=True)
                    if not active.any():break
                for i,e in enumerate(episodes):
                    if e is None:
                        t=int(initial["terrain"][i]);episodes[i]=dict(env_id=i,terrain="flat" if t<2 else str(101 if t<5 else 107),case=int(cases[i]),
                            steps=step+1,success=False,recovery_success=False,collector_cutoff=True,
                            max_body_lateral_m=float(peak[i]),body_lateral_rms_m=float((squares[i]/count[i]).sqrt()))
                trace.close(complete=True);trace=None
                phase=dict(status="complete",initial_sha256=initial_hash,episodes=episodes,confirmed_controls=ledger.state["confirmed"]-phase_start)
                if video is not None:
                    video.close();phase["video"]=dict(path=str(video.path),sha256=sha256_file(video.path),
                        frames=len(video.samples),env_id=video_options["env_id"],
                        scope="New frozen-policy simulation run; not footage of the original evaluation. Pre-action frames, first episode only.")
                    video=None
                write_json_atomic(folder/"report.json",phase);report["phases"][f"{condition}-{name}"]=phase
                write_json_atomic(out/"report.json",report)
                env.close();env=None
        for condition in p["conditions"]:
            assert len({report["phases"][f"{condition}-{m}"]["initial_sha256"] for m in models})==1
        assert all(sha256_file(s["path"])==s["actual_sha256"] for s in p["models"].values())
        assert all(sha256_file(root/f)==h for f,h in sources.items())
        if prefix is not None:assert sha256_file(p["action_prefix"]["path"])==p["action_prefix"]["sha256"]
        ledger.finish();report.update(status="complete",ledger=ledger.state)
        write_json_atomic(out/"report.json",report)
    except BaseException as error:
        if ledger.state["status"]=="running":ledger.finish(aborted=True)
        if trace is not None:trace.close(complete=False)
        report.update(status="failed",error=repr(error),ledger=ledger.state);write_json_atomic(out/"report.json",report)
        raise
    finally:
        if video is not None:video.close()
        if env is not None:env.close()
        if launcher is not None:launcher.app.close()


if __name__=="__main__":main()
