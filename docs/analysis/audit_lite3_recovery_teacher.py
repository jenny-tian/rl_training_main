"""Independent trajectory and split audits for the recovery teacher experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from rl_training.lite3_joint_conventions import LITE3_SAFE_ACTION_LOWER, LITE3_SAFE_ACTION_UPPER
from rl_training.online_adaptation.provenance import sha256_file, write_json_atomic
from rl_training.online_adaptation.flowdagger import FrozenPPOExpert, ConditionalFlow, NoisePolicy


def audit_training(root, protocol):
    report=json.loads((root/"report.json").read_text())
    assert report["status"]=="complete" and report["ppo_updates"]==protocol["iterations"]
    assert report["mean_actor_changed"] and not report["warmstart"]["critic_loaded"] and not report["warmstart"]["optimizer_loaded"]
    bank=np.load(protocol["recovery_resets"]["bank_path"],allow_pickle=False)
    assert sha256_file(protocol["recovery_resets"]["bank_path"])==protocol["recovery_resets"]["bank_sha256"]
    total=0; selected_rows=0; resets=0; exposed=set(); counts={k:0 for k in ("success","failure","deadline")};hashes={}
    previous_done=np.ones(protocol["num_envs"],bool);previous_cases=None
    for file in sorted((root/"trajectory").glob("chunk-*.npz")):
        hashes[file.name]=sha256_file(file)
        with np.load(file,allow_pickle=False) as d:
            obs=d["policy_observation"];case=d["recovery_case"];valid=case>=0
            if protocol["recovery_resets"].get("dedicated_per_terrain") is not None:
                # This registered128-slot layout has32flat,48rough101,48rough107.
                assert protocol['num_envs']==128 and protocol['recovery_resets']['dedicated_per_terrain']==16
                expected=np.zeros(128,bool);expected[32:48]=True;expected[80:96]=True
                np.testing.assert_array_equal(valid,np.broadcast_to(expected,valid.shape))
            np.testing.assert_allclose(obs[...,6:9],np.broadcast_to([.3,0,0],obs[...,6:9].shape),atol=1e-7,rtol=0)
            assert not bank["validation"][case[valid]].any()
            assert np.isfinite(obs).all() and np.isfinite(d["requested_action"]).all()
            emitted=np.clip(d["requested_action"],np.array(LITE3_SAFE_ACTION_LOWER,np.float32),np.array(LITE3_SAFE_ACTION_UPPER,np.float32))
            np.testing.assert_array_equal(d["emitted_action"],emitted)
            history=emitted.copy();history[d["done"].astype(bool)]=0
            for group in ("next_policy_observation","next_critic_observation"):
                np.testing.assert_allclose(d[group][...,-12:],history,atol=1e-6,rtol=0)
            for j in range(len(obs)):
                if previous_cases is not None:
                    np.testing.assert_array_equal(case[j,~previous_done],previous_cases[~previous_done])
                reset=previous_done&valid[j]
                ids=case[j,reset]
                if len(ids):
                    exposed.update(ids.tolist());resets+=len(ids)
                    np.testing.assert_allclose(d["pre_root_state"][j,reset,3:],bank["root"][ids,3:],atol=2e-6,rtol=1e-6)
                    np.testing.assert_allclose(obs[j,reset,12:24]+np.array([0,-.65,1.3]*4,np.float32),bank["joint_pos"][ids],atol=2e-6,rtol=0)
                    np.testing.assert_allclose(obs[j,reset,24:36]/.05,bank["joint_vel"][ids],atol=2e-5,rtol=1e-6)
                    assert (-obs[j,reset,5] <= np.cos(np.deg2rad(19.9))).all()
                previous_done=d["done"][j].astype(bool);previous_cases=case[j]
            for key in counts:counts[key]+=int((d[f"task_{key}"]&valid).sum())
            total+=case.size;selected_rows+=int(valid.sum())
    assert counts==report["recovery_episode_counts"]
    assert total==report["ledger"]["confirmed"]==protocol["training_control_transitions"]
    assert report["ledger"]["uncertain_transitions"]==0
    assert sha256_file(root/"final.pt")==report["final_checkpoint"]["sha256"]
    return dict(status="verified",controls=total,recovery_controls=selected_rows,recovery_resets=resets,
        exposed_training_cases=sorted(exposed),validation_cases_used=0,recovery_episode_counts=counts,
        final_sha256=report["final_checkpoint"]["sha256"],trace_sha256=hashes)


def audit_evaluation(root,protocol):
    report=json.loads((root/"report.json").read_text());assert report["status"]=="complete"
    models={}
    for name,spec in report["models"].items():
        assert sha256_file(spec["path"])==spec["actual_sha256"]
        payload=torch.load(spec["path"],map_location="cpu",weights_only=False)
        if spec["kind"]=="ppo":model=FrozenPPOExpert(payload["actor_state_dict"])
        else:
            flow=ConditionalFlow(**payload["flow_config"]).eval().requires_grad_(False);flow.load_state_dict(payload["flow"])
            actor=NoisePolicy(bound=payload["noise_bound"],hidden=tuple(payload.get("noise_hidden",(256,128)))).eval().requires_grad_(False);actor.load_state_dict(payload["actor"])
            model=(flow,actor)
        models[name]=model
    summaries={};total=0;replay_rows=0
    for phase_name,phase in report["phases"].items():
        condition,name=phase_name.split("-",1);folder=root/phase_name;initial=np.load(folder/"initial.npz",allow_pickle=False)
        delays=json.loads((folder/"delays.json").read_text())
        initial_hash=hashlib.sha256(b"".join(initial[k].tobytes() for k in initial.files)+json.dumps(delays,sort_keys=True).encode()).hexdigest()
        assert initial_hash==phase["initial_sha256"]
        if condition!="course":
            bank=np.load(protocol["recovery_resets"]["bank_path"],allow_pickle=False);cases=initial["case"].astype(int);selected=cases>=0
            assert (bank["validation"][cases[selected]]==(protocol["recovery_resets"]["split"]=="validation")).all()
            np.testing.assert_allclose(initial["root"][selected,:3]-initial["origins"][selected],bank["root"][cases[selected],:3],atol=5e-6,rtol=0)
            np.testing.assert_allclose(initial["root"][selected,3:],bank["root"][cases[selected],3:],atol=2e-6,rtol=1e-6)
        width=protocol["num_envs"];active=np.ones(width,bool);history=np.zeros((width,12),np.float32)
        peak=np.zeros(width);squares=peak.copy();count=peak.copy();ends=np.zeros(width,int)
        streak=np.zeros(width,int);recovered=np.zeros(width,bool);successes=np.zeros(width,bool)
        terms={k:np.zeros(width,bool) for k in ("contact","orientation","lateral")};steps=0;hashes={}
        for file in sorted((folder/"trajectory").glob("chunk-*.npz")):
            hashes[file.name]=sha256_file(file)
            with np.load(file,allow_pickle=False) as d:
                obs=d["policy_observation"]
                assert np.isfinite(obs).all()
                np.testing.assert_allclose(obs[...,6:9],np.broadcast_to([.3,0,0],obs[...,6:9].shape),atol=1e-7,rtol=0)
                expected=np.clip(d["requested_action"],np.array(LITE3_SAFE_ACTION_LOWER,np.float32),np.array(LITE3_SAFE_ACTION_UPPER,np.float32))
                np.testing.assert_array_equal(d["emitted_action"],expected)
                sampled=obs.reshape(-1,48)[d["active"].reshape(-1)][::128][:8]
                actions=d["requested_action"].reshape(-1,12)[d["active"].reshape(-1)][::128][:8]
                with torch.no_grad():
                    o=torch.from_numpy(sampled);model=models[name]
                    prediction=model[0](o,model[1](o)) if isinstance(model,tuple) else model(o)
                    torch.testing.assert_close(prediction,torch.from_numpy(actions),atol=3e-5,rtol=1e-5)
                replay_rows+=len(sampled)
                for j in range(len(obs)):
                    np.testing.assert_array_equal(d["active"][j],active)
                    np.testing.assert_allclose(obs[j,:,-12:],history,atol=1e-6,rtol=0)
                    failure=d["failure_contact"][j]|d["failure_orientation"][j]|d["failure_lateral"][j]
                    np.testing.assert_array_equal(failure,d["failure"][j])
                    np.testing.assert_array_equal(d["done"][j].astype(bool),d["success"][j]|failure|d["deadline"][j])
                    state=d["post_root_state"][j];lateral=state[:,1]-initial["origins"][:,1]
                    peak=np.maximum(peak,np.where(active,np.abs(lateral),0));squares+=np.where(active,lateral*lateral,0);count+=active
                    upright=1-2*np.sum(state[:,4:6]**2,axis=-1)>=np.cos(np.deg2rad(10.))
                    streak=np.where(active&upright,streak+1,0);recovered|=streak>=25
                    ending=active&d["done"][j].astype(bool);ends[ending]=steps+1;successes[ending]=d["success"][j,ending]
                    for key in terms:terms[key][ending]=d[f"failure_{key}"][j,ending]
                    active&=~d["done"][j].astype(bool);history=d["emitted_action"][j].copy();history[d["done"][j].astype(bool)]=0;steps+=1
        assert steps*width==phase["confirmed_controls"];total+=steps*width
        for e in phase["episodes"]:
            i=e["env_id"];assert e["steps"]==(ends[i] or steps) and e["success"]==bool(successes[i])
            assert e["recovery_success"]==bool(successes[i]&recovered[i])
            if not e.get("collector_cutoff"):assert e["failure_terms"]=={k:bool(v[i]) for k,v in terms.items()}
            np.testing.assert_allclose(e["max_body_lateral_m"],peak[i],atol=1e-6)
            np.testing.assert_allclose(e["body_lateral_rms_m"],np.sqrt(squares[i]/count[i]),atol=1e-5)
        rough=[e for e in phase["episodes"] if e["terrain"]!="flat"]
        summaries[phase_name]=dict(by_terrain={t:sum(e["success"] for e in phase["episodes"] if e["terrain"]==t) for t in ("flat","101","107")},
            rough_mean_rms_m=statistics.mean(e["body_lateral_rms_m"] for e in rough),rough_peak_m=max(e["max_body_lateral_m"] for e in rough),
            recovery_success=sum(e["recovery_success"] for e in rough),
            recovery_by_terrain={t:sum(e["recovery_success"] for e in rough if e["terrain"]==t) for t in ("101","107")},
            recovery_by_case={str(i):sum(e["recovery_success"] for e in rough if e["case"]==i) for i in sorted({e["case"] for e in rough})},trace_sha256=hashes)
    assert total==report["ledger"]["confirmed"] and report["ledger"]["uncertain_transitions"]==0
    gates={}
    if "candidate" in models:
        c=summaries["course-candidate"];r=summaries["recovery-candidate"];previous=summaries["course-previous"]
        gates=dict(course_completion=all(c["by_terrain"][t]>=summaries[f"course-{m}"]["by_terrain"][t] for m in ("base","previous") for t in ("flat","101","107")),
            course_rms=c["rough_mean_rms_m"]<=.98*previous["rough_mean_rms_m"],course_peak=c["rough_peak_m"]<=previous["rough_peak_m"],
            recovery_completion=all(r["recovery_by_terrain"][t]>=summaries[f"recovery-{m}"]["recovery_by_terrain"][t] for m in ("base","previous") for t in ("101","107")),
            recovery_80percent=r["recovery_success"]>=.8*48,every_heldout_case=all(v>0 for v in r["recovery_by_case"].values()))
    return dict(status="verified",controls=total,cpu_replay_rows=replay_rows,phases=summaries,gates=gates,
        teacher_gate_passed=bool(gates) and all(gates.values()),
        scope=(f"Kinematic-reset development on bank split={protocol['recovery_resets']['split']}; repeated starts are not independent source episodes or hardware/unseen-terrain evidence."
            if any(c!='course' for c in protocol['conditions']) else
            "Frozen full-course verification only; no recovery qualification, independent training replication, or unseen-terrain evidence."))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True);parser.add_argument("--protocol",required=True);parser.add_argument("--output",required=True);parser.add_argument("--training",action="store_true")
    args=parser.parse_args();torch.set_num_threads(4)
    p=json.loads(Path(args.protocol).read_text());root=Path(args.run)
    assert json.loads((root/"report.json").read_text())["protocol_sha256"]==sha256_file(args.protocol)
    result=audit_training(root,p) if args.training else audit_evaluation(root,p)
    result["evidence"]={str(f):sha256_file(f) for f in (root/"report.json",Path(args.protocol))}
    write_json_atomic(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("trace_sha256","phases")},indent=2))


if __name__=="__main__":main()
