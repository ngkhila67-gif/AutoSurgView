"""New human-aware simulation; never overwrites the recorded 398 x 40 study."""
from pathlib import Path
import json,time
import numpy as np
import torch
from .data import ROOT,configuration,load_scene,sha256
from . import historical_backend as hb,native_backend as nb
from .manuscript_evaluation import HISTORY
from .manuscript_viewer import assembly
from .policies import ArchivedPolicy
from .paper_evaluation import dump,write_rows
from .cbf_evaluation import BASE
from .cbf_safety import RobotCapsules,forward,DT,V_MAX,D_SAFE,ALPHA
from .human_scene import HumanScene,HumanClearance,HUMAN_MARGIN,SCENARIOS
from .human_cbf import filter_velocity

OUT=ROOT/'outputs/cbf_human_20261002'
HORIZON=12


def measure(env,world,t):
    env.xy=forward(env.q);env.points=world.points(t);env._measure()


def frame(env,world,clearance,k,decision,diag=None):
    t=round(k*DT,6);measure(env,world,t);geometry=env.geometry
    return dict(step=k,decision=decision,simulation_time_s=t,q=env.q.tolist(),xy=env.xy.tolist(),
        score=float(env.score),camera=geometry[1].tolist(),corners=geometry[3].tolist(),target=world.target.tolist(),
        points=np.round(env.points,7).tolist(),humans=world.pose(t),inside_indices=np.flatnonzero(env.inside).tolist(),
        robot_transforms=assembly(env.q,env.xy,world.target),self_collision=bool(nb.self_collision(env.q)),
        path_length_m=env.path_length,filter=diag,**clearance.metrics(env.q,t),**world.occlusion(geometry[1],t))


def rollout(scene,start,world,robot,policy,enabled):
    env=hb.HistoricalEnv([scene]);env.reset_start(start);env.sampling_rng=np.random.RandomState(start['seed'])
    cl=HumanClearance(robot,world);trace=[frame(env,world,cl,0,0)];records=[];reason='horizon_complete'
    if cl.h(env.q,0).min()<.005:raise ValueError('Unsafe or marginal initial state')
    for decision in range(1,HORIZON+1):
        measure(env,world,(decision-1));obs=env._observation()
        # Original policy and source score threshold retained, but this is a
        # synthetic out-of-distribution observation, not benchmark recovery.
        delta=env.displacement(policy.predict(obs,env)) if env.score>=20 else np.zeros(2)
        goal,ok=hb.inverse(env.xy+delta,env.q)
        env.unreachable_count=0 if ok else env.unreachable_count+1
        if not ok:goal=env.q.copy()
        for sub in range(10):
            t=(len(trace)-1)*DT;remaining=1-sub*DT
            nominal=(goal-env.q)/remaining;nominal/=max(1,np.max(np.abs(nominal))/V_MAX)
            diag=dict(status='unfiltered',correction=0.,residual=None,backtrack=0,feasible=None)
            u,diag=filter_velocity(env.q,nominal,cl,t) if enabled else (nominal,diag)
            oldq=env.q.copy();oldxy=env.xy.copy();samples=[];self_checks=[]
            for f in (.2,.4,.6,.8,1.):
                ds=cl.distances(oldq+f*DT*u,t+f*DT)
                samples.append([float(ds[:cl.n_equipment].min()),float(ds[cl.n_equipment:].min())])
                self_checks.append(bool(nb.self_collision(oldq+f*DT*u)))
            env.q=oldq+DT*u;env.xy=forward(env.q);env.path_length+=float(np.linalg.norm(env.xy-oldxy))
            de,dh=map(float,np.min(samples,axis=0))
            diag.update(command_rad_s=u.tolist(),nominal_rad_s=nominal.tolist(),min_sampled_equipment_m=float(de),min_sampled_human_m=float(dh))
            trace.append(frame(env,world,cl,len(trace),decision,diag))
            records.append(dict(time_s=trace[-1]['simulation_time_s'],human_clearance_m=dh,equipment_clearance_m=de,
                field_occlusion_pct=trace[-1]['field_occlusion_pct'],head_occlusion_pct=trace[-1]['head_occlusion_pct'],
                correction_rad_s=diag['correction'],solver=diag['status'],residual=diag['residual'],feasible=diag['feasible']))
            if enabled and not diag['feasible']:reason='dynamic_safety_unverified';break
            if any(self_checks):reason='original_self_collision';break
            if min(de,dh)<0:reason='envelope_overlap';break
        if reason!='horizon_complete':break
    minhuman=min([trace[0]['human_clearance_m']]+[r['human_clearance_m'] for r in records])
    minequip=min([trace[0]['equipment_clearance_m']]+[r['equipment_clearance_m'] for r in records])
    summary=dict(method='cbf' if enabled else 'nominal',termination=reason,simulation_time_s=trace[-1]['simulation_time_s'],
        min_human_clearance_m=minhuman,min_equipment_clearance_m=minequip,
        human_margin_satisfied=minhuman>=HUMAN_MARGIN-1e-7,equipment_margin_satisfied=minequip>=D_SAFE-1e-7,
        envelope_overlap=min(minhuman,minequip)<0,original_self_collision=any(x['self_collision'] for x in trace),
        full_horizon=reason=='horizon_complete',initial_field_occlusion_pct=trace[0]['field_occlusion_pct'],
        final_field_occlusion_pct=trace[-1]['field_occlusion_pct'],final_head_occlusion_pct=trace[-1]['head_occlusion_pct'],
        initial_score=trace[0]['score'],final_score=trace[-1]['score'],path_length_m=env.path_length,
        corrected_steps=sum(r['correction_rad_s']>1e-5 for r in records),
        unverified_steps=sum(r['feasible'] is False for r in records),integration_steps=len(records))
    summary['complete_safe_trial']=bool(summary['full_horizon'] and summary['human_margin_satisfied'] and summary['equipment_margin_satisfied'] and not summary['original_self_collision'] and summary['unverified_steps']==0)
    return dict(trace=trace,summary=summary),records


def run():
    OUT.mkdir(exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    hb.configure(HISTORY);torch.set_num_threads(1);robot=RobotCapsules()
    rows=json.loads((BASE/'ordered_dataset.json').read_text(encoding='utf-8'))[1592:]
    starts=json.loads((BASE/'starts.json').read_text(encoding='utf-8'));selected=[];excluded=[]
    for si in (0,99,198,297):
        scene=load_scene(rows[si])
        for scenario in range(3):
            world=HumanScene(scene.target,scenario);cl=HumanClearance(robot,world)
            for start in [s for s in starts if s['scene_index']==si]:
                q=np.array(start['q']);geom=hb.camera_geometry(q,forward(q),scene.target)
                reasons=[]
                if nb.self_collision(q):reasons.append('original_self_collision')
                if cl.h(q,0).min()<.005:reasons.append('initial_margin_below_required_plus_5mm')
                if world.occlusion(geom[1],0)['head_occlusion_pct']<25:reasons.append('head_ray_occlusion_below_25pct')
                if reasons:excluded.append(dict(scene_index=si,scenario=scenario,start_index=start['start_index'],reasons=reasons));continue
                selected.append(dict(scene_index=si,scenario=scenario,start=start));break
            else:excluded.append(dict(scene_index=si,scenario=scenario,reasons=['no_eligible_start']))
    if not selected:raise RuntimeError('No eligible cases')
    model=Path(configuration()['archive_root'])/'model/RL_Point/ppo_AnzhenRobot_20.zip'
    protocol=dict(version='human_equipment_cbf_v1',scope='Synthetic human-aware pilot; no physical experiment, not the 398 x 40 recorded point-cloud benchmark',
        paired_cases=len(selected),selected=selected,exclusions=excluded,predeclared_scene_indices=[0,99,198,297],
        selection='First archived start with head-ray occlusion >=25%, both initial margins plus 5mm, and no native self-collision; outcomes never used',
        dt_s=DT,horizon_s=HORIZON,decision_time_s=1.,human_margin_m=HUMAN_MARGIN,equipment_margin_m=D_SAFE,
        alpha_per_s=ALPHA,velocity_limit_rad_s=V_MAX,capsule_geometry=robot.manifest,
        model_sha256=sha256(model),source_sha256={str(p):sha256(p) for p in [Path(__file__),ROOT/'ncsim/human_scene.py',ROOT/'ncsim/human_cbf.py',ROOT/'ncsim/cbf_safety.py']},
        original_word_sha256=sha256(ROOT.parent/'JFR0511.docx'),scenarios=SCENARIOS,
        cbf='qdot=u; h_ij(q,t)=distance_ij(q,t)-margin_ij; min .5||u-u_nom||^2 s.t. grad_q(h)u+partial_t(h)+2h>=0, |u_k|<=0.35',
        failure='No slack. If QP fails or sampled future states fail, log unverified safety, advance one hold interval against the MOVING human, and terminate; a hold is not a safety certificate.',
        sampling='Five future-pose checks within every 0.1s interval; finite differences 1e-5; full horizon continues after visual clearance',
        human_motion='8s smooth period, w=(1-cos(2*pi*t/8))/2; surgeon head displacement=(-.025,-.065,.025)*w m, chest .45 times; assistant=(.04,-.055,.025)*w; legs fixed',
        human_models_at_t0={str(i):HumanScene(load_scene(rows[0]).target,i).pose(0) for i in range(3)},
        surgical_field=dict(width_m=.20,height_m=.12,ray_grid=[21,15],metric='Percentage of 315 target-plane rays intersecting human capsules or equipment OBBs'),
        observations='Original archived target coordinates, initial joint states, KDL, meshes, self-collision and policy. Anonymous archived cloud omitted; deterministic sampled human capsule and equipment OBB surfaces replace it. z ROI preserved. 72 points per human capsule; 24 per equipment box before ROI.',
        limitations=['All human and equipment geometry, placement, movement, margins and control periods are new declared simulation assumptions, not clinical measurements.',
            'Original policy is frozen and receives out-of-distribution synthetic points. Source score threshold20 is density-dependent and not calibrated here; no direct comparison to archived success rates.',
            'Human capsules and robot enclosing capsules are simplified, conservative geometry. Their overlap is not proof of mesh contact or injury.',
            'Sampled nonlinear checks and nonsmooth distance finite differences are not a continuous-time safety proof.',
            'No contact dynamics, measured human tracking, uncertainty/latency model or actual hardware CBF.',
            'Baseline terminates on envelope overlap/self-collision; shortened runs must not be treated as faster successful trials.',
            'Patient and clinicians are all included in clearance; head-only and total ray occlusion are measured separately.'])
    dump(OUT/'protocol.json',protocol)
    policy=ArchivedPolicy(model,hb.HistoricalEnv([load_scene(rows[0])]))
    catalog=[];summaries=[];allsteps=[]
    for ci,spec in enumerate(selected):
        scene=load_scene(rows[spec['scene_index']]);world=HumanScene(scene.target,spec['scenario'])
        caseid=f"scene_{spec['scene_index']+1:03d}_start_{spec['start']['start_index']+1:02d}_human_{spec['scenario']}"
        print(f'{ci+1}/{len(selected)} {caseid}',flush=True);runs={}
        for enabled in (False,True):
            result,records=rollout(scene,spec['start'],world,robot,policy,enabled);m='cbf' if enabled else 'nominal'
            result['summary'].update(case_id=caseid,scenario=spec['scenario']);runs[m]=result;summaries.append(result['summary'])
            allsteps.extend(dict(case_id=caseid,method=m,**r) for r in records)
            print(m,json.dumps(result['summary'],ensure_ascii=False),flush=True)
        case=dict(case_id=caseid,scene_id=scene.scene_id,scenario=spec['scenario'],target=scene.target.tolist(),points=runs['cbf']['trace'][0]['points'],equipment=world.equipment,runs=runs)
        # Compact JSON avoids unnecessarily large repeated frame arrays.
        (OUT/'cases'/f'{caseid}.json').write_text(json.dumps(case,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
        catalog.append(dict(case_id=caseid,label=f"场景 {spec['scene_index']+1} · 起点 {spec['start']['start_index']+1} · {SCENARIOS[spec['scenario']]}",file=f'cases/{caseid}.json',summaries={k:v['summary'] for k,v in runs.items()}))
        dump(OUT/'progress.json',dict(completed=ci+1,total=len(selected)))
    dump(OUT/'catalog.json',dict(cases=catalog));write_rows(OUT/'paired_results.csv',summaries);write_rows(OUT/'control_steps.csv',allsteps)
    report={}
    for m in ('nominal','cbf'):
        ss=[s for s in summaries if s['method']==m]
        report[m]=dict(trials=len(ss),complete_safe_trials=sum(s['complete_safe_trial'] for s in ss),
            human_margin_satisfied=sum(s['human_margin_satisfied'] for s in ss),equipment_margin_satisfied=sum(s['equipment_margin_satisfied'] for s in ss),
            full_horizon_trials=sum(s['full_horizon'] for s in ss),envelope_overlap=sum(s['envelope_overlap'] for s in ss),
            minimum_human_clearance_m=min(s['min_human_clearance_m'] for s in ss),corrected_steps=sum(s['corrected_steps'] for s in ss),
            unverified_steps=sum(s['unverified_steps'] for s in ss))
    dump(OUT/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':run()
