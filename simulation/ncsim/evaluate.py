from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import platform
import time
import numpy as np
import torch
from .data import ROOT,configuration,load_scene,demo_rows,sha256
from .environment import SurgicalCameraEnv
from .policies import policy_set,LABELS
from .native_backend import ENGINE,provenance
from .robot_model import prepare_robot_assets,render_viewer


def make_env(scenes,config):
    return SurgicalCameraEnv(scenes,**{k:config[k] for k in ("threshold","max_steps","action_step_m","action_mode")})


def write_csv(path,rows):
    if not rows:
        return
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_evaluation(per_session=3,starts=2,output=None):
    config=configuration()
    torch.set_num_threads(1)
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out=Path(output) if output else ROOT/"outputs"/("run_"+timestamp)
    out.mkdir(parents=True,exist_ok=False)
    rows=demo_rows(per_session)
    scenes=[load_scene(row) for row in rows]
    env=make_env([scenes[0]],config)
    policies=policy_set(config,env)
    episodes,step_rows,cases,obs_records,obs_mapping=[],[],[],[],[]
    for scene_i,scene in enumerate(scenes):
        for start_i in range(starts):
            seed=config["seed"]+scene_i*100+start_i
            case_id=f"scene_{scene_i+1:03d}_start_{start_i+1:02d}"
            case=dict(case_id=case_id,scene_id=scene.scene_id,session=scene.session,seed=seed,
                      source=scene.source,source_sha256=scene.source_sha256,runs={})
            start_scores=[]
            for method,policy in policies.items():
                env=make_env([scene],config)
                obs,info=env.reset(seed=seed)
                initial_xy=env.xy.copy()
                start_scores.append(info["score"])
                if "points" not in case:
                    case["points"]=np.round(env.points,5).tolist()
                trace=[env.snapshot()]
                score_sum=0.0
                inference_times=[]
                if info["initially_clear"]:
                    _,_,_,_,info=env.step(np.zeros(2))
                    trace[0]["reason"]=info["termination_reason"]
                else:
                    for _ in range(config["max_steps"]):
                        old_score=env.score
                        before_step=env.steps
                        if method=="fusion" and before_step<3:
                            obs_records.append({k:v.copy() for k,v in obs.items()})
                            obs_mapping.append(dict(case_id=case_id,step=before_step,scene_id=scene.scene_id,
                                source_sha256=scene.source_sha256,xy=env.xy.tolist()))
                        begin=time.perf_counter_ns()
                        action=policy.predict(obs,env)
                        elapsed=(time.perf_counter_ns()-begin)/1e6
                        inference_times.append(elapsed)
                        obs,reward,terminated,truncated,info=env.step(action)
                        score_sum+=(old_score+env.score)/2
                        snap=env.snapshot()
                        snap.update(action=np.asarray(action).tolist(),reward=reward)
                        trace.append(snap)
                        step_rows.append(dict(case_id=case_id,scene_id=scene.scene_id,method=method,
                              step=env.steps,camera_x_m=env.xy[0],camera_y_m=env.xy[1],
                              score=env.score,action_x=float(action[0]),action_y=float(action[1]),
                              reward=reward,reason=info["termination_reason"]))
                        if terminated or truncated:
                            break
                # Fixed horizon burden is a static-scene hold-last-position estimate,
                # explicitly separated from observed pre-termination score-step area.
                fixed_area=score_sum+env.score*(config["max_steps"]-env.steps)
                row=dict(case_id=case_id,scene_id=scene.scene_id,capture_group=scene.capture_group,
                         session=scene.session,seed=seed,method=method,initial_x_m=initial_xy[0],initial_y_m=initial_xy[1],
                         initial_score=env.initial_score,final_score=env.score,initially_clear=info["initially_clear"],
                         criterion_met=bool(info["success"]),new_recovery=bool(info["success"] and not info["initially_clear"]),
                         decision_steps=env.steps,path_length_m=env.path_length,termination_reason=info["termination_reason"],
                         score_step_area_observed=score_sum,score_step_area_fixed_horizon_hold=fixed_area,
                         policy_call_ms_mean=float(np.mean(inference_times)) if inference_times else None)
                episodes.append(row)
                case["runs"][method]=dict(trace=trace,summary=row)
            if not np.allclose(start_scores,start_scores[0],rtol=0,atol=1e-10):
                raise AssertionError("Methods did not start in the same state")
            cases.append(case)
            print(f"Completed {len(cases)}/{len(scenes)*starts}: {case_id}, initial={start_scores[0]:.2f}",flush=True)
    summary=[]
    for method in policies:
        values=[r for r in episodes if r["method"]==method]
        occluded=[r for r in values if not r["initially_clear"]]
        recovered=[r for r in occluded if r["new_recovery"]]
        summary.append(dict(method=method,cases=len(values),initially_occluded_cases=len(occluded),
                            newly_recovered_cases=len(recovered),
                            recovery_fraction=len(recovered)/len(occluded) if occluded else None,
                            mean_final_score=float(np.mean([r["final_score"] for r in values])),
                            mean_steps_all=float(np.mean([r["decision_steps"] for r in values])),
                            mean_steps_recovered=float(np.mean([r["decision_steps"] for r in recovered])) if recovered else None,
                            mean_path_length_m=float(np.mean([r["path_length_m"] for r in values])),
                            self_collision_cases=sum(r["termination_reason"]=="self_collision" for r in values)))
    write_csv(out/"episodes.csv",episodes)
    write_csv(out/"steps.csv",step_rows)
    write_csv(out/"summary.csv",summary)
    if obs_records:
        np.savez_compressed(out/"simulation_observations.npz",**{k:np.stack([o[k] for o in obs_records]) for k in obs_records[0]})
    robot_model=prepare_robot_assets()
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),engine=ENGINE,native_backend=provenance(),robot_model=robot_model,
                  viewer_files_sha256={name:sha256(ROOT/"viewer"/name) for name in ("robot-viewer.js","robot-viewer.bundle.js","template.html","package-lock.json")},
                  purpose="environment_validation_not_publication_results",config=config,
                  python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                  policies={k:v.metadata for k,v in policies.items()},sources=rows,
                  code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/"ncsim").glob("*.py"))},
                  scope="Static captured point clouds; exposure to archived training unknown; no physical time or clinical outcomes.",
                  observations=obs_mapping,
                  motion_note="Archived Orocos KDL 1.5.1 solver and NC collision.cpp execute natively without algorithm edits; binding preserves original simulation chain and solver settings. Historical Python collision module identity remains unverified.",
                  fixes=["reset clears unreachable counter","6-state declared shape matches output","max_steps truncates at exactly 40",
                         "separate success/failure reasons","independent local random generators"],
                  baseline_note="greedy is a one-step geometric search for platform diagnostics; no simulated human baseline",
                  policy_time_note="Diagnostic Python call time only, no warmup; different controllers include different computations; not a timing benchmark.")
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    payload=dict(title="开放手术自主视角仿真",threshold=config["threshold"],horizon=config["max_steps"],
                 labels=LABELS,summary=summary,cases=cases,manifest_path="manifest.json",robot_model=robot_model)
    (out/"replay.json").write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    render_viewer(payload,out/"仿真回放.html")
    (ROOT/"outputs"/"latest.json").write_text(json.dumps({"directory":str(out),"viewer":str(out/"仿真回放.html")},ensure_ascii=False),encoding="utf-8")
    return out,summary
