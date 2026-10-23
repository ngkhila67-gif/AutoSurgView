"""Paired pilot of the new CBF equipment extension, isolated from 398 x 40."""
from pathlib import Path
import csv, json, time, gzip
import numpy as np
import torch
from .data import ROOT, configuration, load_scene, sha256
from . import historical_backend as hb, native_backend as nb
from .manuscript_evaluation import HISTORY
from .manuscript_viewer import assembly
from .policies import ArchivedPolicy
from .paper_evaluation import dump, write_rows
from .cbf_safety import (RobotCapsules, Clearance, equipment_scene, forward,
                        filter_velocity, DT, DECISION_TIME, V_MAX, D_SAFE, ALPHA)

BASE=ROOT/'outputs/manuscript_aligned_20261002'
OUT=ROOT/'outputs/cbf_equipment_20261002'


def frame(env,clearance,t,decision,diag=None):
    inside,score,guide,geom=hb.classify(env.points,env.xy,env.scene.target,env.q)
    distance,pair=clearance.closest(env.q)
    return dict(step=t,decision=decision,simulation_time_s=round(t*DT,6),q=env.q.tolist(),xy=env.xy.tolist(),
                score=score,camera=geom[1].tolist(),corners=geom[3].tolist(),target=env.scene.target.tolist(),
                inside_indices=np.flatnonzero(inside).tolist(),robot_transforms=assembly(env.q,env.xy,env.scene.target),
                clearance_m=distance,closest_pair=pair,self_collision=bool(nb.self_collision(env.q)),
                path_length_m=env.path_length,filter=diag)


def rollout(scene,start,equipment,capsules,policy,enabled):
    env=hb.HistoricalEnv([scene],max_steps=40);obs=env.reset_start(start)
    # Actual forward pose is used in BOTH arms of this new comparison.
    env.xy=forward(env.q);env._measure();env.sampling_rng=np.random.RandomState(start['seed']);obs=env._observation()
    clearance=Clearance(capsules,equipment,scene.target)
    trace=[frame(env,clearance,0,0)];substeps=[];reason='step_limit';decisions=0;unreachable=0
    begin=time.perf_counter()
    if trace[0]['clearance_m']<D_SAFE or trace[0]['self_collision']:raise ValueError('Unsafe pilot start')
    if env.score<20:return dict(trace=trace,summary=dict(method='cbf' if enabled else 'nominal',termination='initially_clear'))
    for decision in range(1,41):
        action=policy.predict(obs,env);delta=env.displacement(action)
        goal,ok=hb.inverse(env.xy+delta,env.q);decisions=decision
        unreachable=0 if ok else unreachable+1
        if not ok:goal=env.q.copy()
        # Do not wrap joint angles: the archived IK's returned branch is retained.
        for sub in range(round(DECISION_TIME/DT)):
            remaining=DECISION_TIME-sub*DT
            u_nom=(goal-env.q)/remaining
            u_nom=u_nom/max(1.,np.max(np.abs(u_nom))/V_MAX)
            diag=dict(status='unfiltered',correction=0.,residual=None,backtrack=0)
            if enabled:u,diag=filter_velocity(env.q,u_nom,clearance)
            else:u=u_nom
            old_xy=env.xy.copy();old_q=env.q.copy()
            # Record the same five within-step checks for both arms.
            ds=[float(clearance.distances(old_q+fraction*DT*u).min()) for fraction in [.2,.4,.6,.8,1.]]
            self_checks=[bool(nb.self_collision(old_q+fraction*DT*u)) for fraction in [.2,.4,.6,.8,1.]]
            env.q=old_q+DT*u;env.xy=forward(env.q);env.path_length+=float(np.linalg.norm(env.xy-old_xy))
            k=len(trace);diag.update(min_sampled_clearance_m=min(ds),nominal_speed_rad_s=float(np.linalg.norm(u_nom)),
                                    command_rad_s=u.tolist())
            trace.append(frame(env,clearance,k,decision,diag))
            substeps.append(dict(substep=k,decision=decision,time_s=k*DT,clearance_m=trace[-1]['clearance_m'],
                       min_sampled_clearance_m=min(ds),score=trace[-1]['score'],correction_rad_s=diag['correction'],
                       cbf_residual=diag['residual'],backtrack=diag['backtrack'],solver=diag['status']))
            if any(self_checks):reason='self_collision';break
            if min(ds)<0:reason='equipment_envelope_overlap';break
        env.steps=decision;env.unreachable_count=unreachable;env._measure();obs=env._observation()
        if reason!='step_limit':break
        if env.score<20:reason='criterion_met';break
        if unreachable>5:reason='repeated_unreachable';break
    minimum=min([trace[0]['clearance_m']]+[x['min_sampled_clearance_m'] for x in substeps])
    corrections=[x['correction_rad_s'] for x in substeps]
    summary=dict(method='cbf' if enabled else 'nominal',initial_score=trace[0]['score'],final_score=trace[-1]['score'],
       decisions=decisions,integration_steps=len(substeps),simulation_time_s=len(substeps)*DT,termination=reason,
       occlusion_success=env.score<20,clearance_satisfied=minimum>=D_SAFE-1e-7,min_clearance_m=minimum,
       safety_margin_violated=minimum<D_SAFE-1e-7,envelope_overlap=minimum<0,
       original_self_collision=any(f['self_collision'] for f in trace),path_length_m=env.path_length,
       corrected_steps=sum(x>1e-5 for x in corrections),max_correction_rad_s=max(corrections,default=0.),
       qp_failure_steps=sum(x['solver']=='QP_failure_hold' for x in substeps),
       nonlinear_backtrack_steps=sum(x['backtrack']>0 for x in substeps),wall_seconds=time.perf_counter()-begin)
    return dict(trace=trace,summary=summary,substeps=substeps)


def run(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    hb.configure(HISTORY);torch.set_num_threads(1)
    rows=json.loads((BASE/'ordered_dataset.json').read_text(encoding='utf-8'))[1592:]
    starts=json.loads((BASE/'starts.json').read_text(encoding='utf-8'))
    capsules=RobotCapsules();selected=[];exclusions=[]
    # Select first safe and initially occluded start per predeclared scene/layout.
    # No selection based on policy success or CBF intervention is performed.
    for scene_index in [0,99,198,297]:
        scene=load_scene(rows[scene_index])
        for layout in range(3):
            equipment=equipment_scene(scene.target,layout);clearance=Clearance(capsules,equipment,scene.target)
            for start in [s for s in starts if s['scene_index']==scene_index]:
                env=hb.HistoricalEnv([scene]);env.reset_start(start);d=clearance.distances(env.q).min()
                reasons=[]
                if nb.self_collision(env.q):reasons.append('original_self_collision')
                if d<D_SAFE+.005:reasons.append('equipment_margin_under_45mm')
                if env.score<20:reasons.append('initially_clear')
                if reasons:exclusions.append(dict(scene_index=scene_index,start_index=start['start_index'],layout=layout,reasons=reasons));continue
                selected.append(dict(scene_index=scene_index,start=start,equipment=equipment));break
            else:exclusions.append(dict(scene_index=scene_index,layout=layout,reasons=['no_eligible_start']))
    if not selected:raise RuntimeError('No eligible predeclared pilot starts')
    model=Path(configuration()['archive_root'])/'model/RL_Point/ppo_AnzhenRobot_20.zip'
    protocol=dict(version='cbf_equipment_pilot_v1',scope='New simulation-only pilot. No physical validation, no replacement of the 398 x 40 benchmark.',
        paired_cases=len(selected),predeclared_scene_indices=[0,99,198,297],layouts=3,selection_rule='First safe initially occluded archived start per scene/layout; no outcome-based selection',
        selected=[dict(scene_index=x['scene_index'],start_index=x['start']['start_index'],seed=x['start']['seed'],layout=x['equipment']['layout']) for x in selected],
        exclusions=exclusions,dt_s=DT,nominal_decision_time_s=DECISION_TIME,velocity_limit_rad_s=V_MAX,
        safe_distance_m=D_SAFE,alpha_per_s=ALPHA,max_decisions=40,model_sha256=sha256(model),
        source_sha256={str(p):sha256(p) for p in [Path(__file__),ROOT/'ncsim/cbf_safety.py']},
        baseline='Same archived policy, native KDL, velocity bound and integration as CBF arm; external-equipment CBF disabled',
        cbf='qdot=u; h_ij=distance(capsule_i(q),OBB_j)-0.04; min 0.5*||u-u_nom||^2 s.t. grad(h_ij)*u+2*h_ij>=0, |u_k|<=0.35',
        safeguards='No safety slack; failed solve holds; original self-collision checked; nonlinear backtracking checks five points per 0.1 s interval',
        capsule_geometry=capsules.manifest,
        limitations=['Control periods, velocity limits, 40 mm clearance and all equipment poses/sizes are new simulation settings, not measured hardware parameters',
           'Seven conservative capsules enclose original STL meshes; envelope overlap is not proof of physical mesh contact',
           'Numerical gradients and sampled nonlinear guards are not a proof of continuous-time forward invariance',
           'Static equipment only; no moving-person, contact dynamics, actuator or sensor-delay model',
           'Recorded scene points alone determine the existing occlusion objective; equipment surfaces are not added to the point cloud',
           'No empirical hardware joint limits or FOV CBF added; camera still uses archived target-pointing geometry',
           'Source display camera offset is preserved, as in the archived playback'])
    dump(out/'protocol.json',protocol) # Freeze settings before any comparison.
    policy=ArchivedPolicy(model,hb.HistoricalEnv([load_scene(rows[0])]))
    catalog=[];summaries=[];allsteps=[]
    for case_index,spec in enumerate(selected):
        scene=load_scene(rows[spec['scene_index']]);start=spec['start'];equipment=spec['equipment']
        case_id=f"scene_{spec['scene_index']+1:03d}_start_{start['start_index']+1:02d}_layout_{equipment['layout']}"
        print(f'Running {case_index+1}/{len(selected)} {case_id}',flush=True)
        runs={}
        for enabled in (False,True):
            result=rollout(scene,start,equipment,capsules,policy,enabled);name='cbf' if enabled else 'nominal'
            for row in result.pop('substeps'):allsteps.append(dict(case_id=case_id,method=name,**row))
            result['summary'].update(case_id=case_id,scene_index=spec['scene_index'],start_index=start['start_index'],layout=equipment['layout'])
            summaries.append(result['summary']);runs[name]=result
            print(f" {name}: {result['summary']['termination']}; clearance {result['summary']['min_clearance_m']:.4f} m; score {result['summary']['final_score']:.2f}",flush=True)
        np.testing.assert_allclose(runs['nominal']['trace'][0]['q'],runs['cbf']['trace'][0]['q'],rtol=0,atol=0)
        assert runs['nominal']['trace'][0]['score']==runs['cbf']['trace'][0]['score']
        points=scene.points[(scene.points[:,2]>=scene.target[2]+.2)&(scene.points[:,2]<=1.75)]
        case=dict(case_id=case_id,scene_id=scene.scene_id,points=points.tolist(),equipment=equipment,runs=runs)
        dump(out/'cases'/f'{case_id}.json',case)
        catalog.append(dict(case_id=case_id,label=f"场景 {spec['scene_index']+1} · 起点 {start['start_index']+1} · {equipment['label']}",file=f'cases/{case_id}.json',summaries={k:v['summary'] for k,v in runs.items()}))
        dump(out/'progress.json',dict(completed=case_index+1,total=len(selected)))
    dump(out/'catalog.json',dict(cases=catalog,settings={k:protocol[k] for k in ('dt_s','safe_distance_m','velocity_limit_rad_s','alpha_per_s')}))
    write_rows(out/'paired_results.csv',summaries);write_rows(out/'control_steps.csv',allsteps)
    report={}
    for method in ('nominal','cbf'):
        ss=[s for s in summaries if s['method']==method]
        report[method]=dict(trials=len(ss),clearance_satisfied=sum(s['clearance_satisfied'] for s in ss),
          occlusion_success=sum(s['occlusion_success'] for s in ss),equipment_envelope_overlap=sum(s['envelope_overlap'] for s in ss),
          original_self_collision=sum(s['original_self_collision'] for s in ss),minimum_clearance_m=min(s['min_clearance_m'] for s in ss),
          corrected_steps=sum(s['corrected_steps'] for s in ss),qp_failure_steps=sum(s['qp_failure_steps'] for s in ss))
    dump(out/'summary.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


if __name__=='__main__':run()
