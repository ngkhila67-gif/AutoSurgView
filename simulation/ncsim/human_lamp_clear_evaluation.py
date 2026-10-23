"""Rerun identical human trials after separating lamp and camera and joining monitor supports."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from . import human_evaluation as previous
from .data import ROOT,configuration,load_scene,sha256
from .paper_evaluation import dump,write_rows
from .human_lamp_clear_scene import ClearLampHumanScene
from .human_scene import HumanClearance
from .cbf_safety import RobotCapsules
from . import historical_backend as hb
from .manuscript_evaluation import HISTORY
from .policies import ArchivedPolicy

OUT=ROOT/'outputs/cbf_human_lamp_clear_20261008'


def run():
    OUT.mkdir(exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    hb.configure(HISTORY);torch.set_num_threads(1);robot=RobotCapsules()
    rows=json.loads((previous.BASE/'ordered_dataset.json').read_text(encoding='utf-8'))[1592:]
    oldprotocol=ROOT/'outputs/cbf_human_lamp_close_20261002/protocol.json';protocol=copy.deepcopy(json.loads(oldprotocol.read_text(encoding='utf-8')))
    protocol.update(version='human_equipment_cbf_lamp_clear_v4',previous_protocol_sha256=sha256(oldprotocol),
        selection='Same twelve starts and human motions as v3; no outcome-based selection.',
        lamp_change=dict(head_center_z_m=1.90,previous_head_center_z_m=1.90,lowered_m=0.,
            head_xy='G.xy+(0,0.60) m; previous G.xy+(0,0.20) m.',horizontal_move_m=.40,
            light_bottom_above_target_m=1.0075,light_bottom_center_to_target_m=float(np.hypot(1.0075,.60)),
            upper_arm='Reconnected original pole to moved vertical neck, updating arm length and yaw.',
            scope='User-requested simulation layout; no clinical recommendation or photometric model'),
        support_change=dict(monitor='Extend upright to screen centre and add overlapping rear mount.',
            anaesthesia='Add support overlapping body top and screen bottom.',
            calculation='All changed/added solids enter equipment clearance, synthetic observations and occlusion.'))
    protocol['source_sha256'].update({str(p):sha256(p) for p in [Path(__file__),ROOT/'ncsim/human_lamp_clear_scene.py']})
    # Check predeclared original starts, and independently check prescribed
    # humans against the static lamp at all 20 ms sampling instants.
    checks=[]
    for spec in protocol['selected']:
        scene=load_scene(rows[spec['scene_index']]);world=ClearLampHumanScene(scene.target,spec['scenario'])
        cl=HumanClearance(robot,world)
        if cl.h(np.array(spec['start']['q']),0).min()<.005:raise ValueError('Original start no longer safe with moved lamp')
        gap,pair=min(world.lamp_human_clearance(t) for t in np.linspace(0,12,601))
        if gap<.08:raise ValueError('Prescribed human motion too close to moved lamp')
        checks.append(dict(scene_index=spec['scene_index'],scenario=spec['scenario'],min_lamp_human_clearance_m=gap,closest_pair=pair))
    protocol['prescribed_human_lamp_checks']=checks
    dump(OUT/'protocol.json',protocol)
    model=Path(configuration()['archive_root'])/'model/RL_Point/ppo_AnzhenRobot_20.zip'
    policy=ArchivedPolicy(model,hb.HistoricalEnv([load_scene(rows[0])]))
    catalog=[];summaries=[];allsteps=[]
    for i,spec in enumerate(protocol['selected']):
        scene=load_scene(rows[spec['scene_index']]);world=ClearLampHumanScene(scene.target,spec['scenario'])
        caseid=f"scene_{spec['scene_index']+1:03d}_start_{spec['start']['start_index']+1:02d}_human_{spec['scenario']}"
        print(f'{i+1}/12 {caseid}',flush=True);runs={}
        for enabled in (False,True):
            result,records=previous.rollout(scene,spec['start'],world,robot,policy,enabled);name='cbf' if enabled else 'nominal'
            result['summary'].update(case_id=caseid,scenario=spec['scenario']);runs[name]=result;summaries.append(result['summary'])
            allsteps.extend(dict(case_id=caseid,method=name,**r) for r in records)
            print(name,result['summary']['termination'],'equipment',result['summary']['min_equipment_clearance_m'],'human',result['summary']['min_human_clearance_m'],flush=True)
        case=dict(case_id=caseid,scene_id=scene.scene_id,scenario=spec['scenario'],target=scene.target.tolist(),points=runs['cbf']['trace'][0]['points'],equipment=world.equipment,runs=runs)
        (OUT/'cases'/f'{caseid}.json').write_text(json.dumps(case,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
        catalog.append(dict(case_id=caseid,label=f"场景 {spec['scene_index']+1} · 起点 {spec['start']['start_index']+1} · {previous.SCENARIOS[spec['scenario']]}",file=f'cases/{caseid}.json',summaries={k:v['summary'] for k,v in runs.items()}))
        dump(OUT/'progress.json',dict(completed=i+1,total=12))
    dump(OUT/'catalog.json',dict(cases=catalog));write_rows(OUT/'paired_results.csv',summaries);write_rows(OUT/'control_steps.csv',allsteps)
    report={}
    for name in ('nominal','cbf'):
        ss=[s for s in summaries if s['method']==name]
        report[name]=dict(trials=len(ss),complete_safe_trials=sum(s['complete_safe_trial'] for s in ss),
            human_margin_satisfied=sum(s['human_margin_satisfied'] for s in ss),equipment_margin_satisfied=sum(s['equipment_margin_satisfied'] for s in ss),
            full_horizon_trials=sum(s['full_horizon'] for s in ss),envelope_overlap=sum(s['envelope_overlap'] for s in ss),
            minimum_human_clearance_m=min(s['min_human_clearance_m'] for s in ss),minimum_equipment_clearance_m=min(s['min_equipment_clearance_m'] for s in ss),
            corrected_steps=sum(s['corrected_steps'] for s in ss),unverified_steps=sum(s['unverified_steps'] for s in ss))
    dump(OUT/'summary.json',report)
    print(json.dumps(report,indent=2),flush=True)
    # The same plotting implementation reads NEW records, not the old chart.
    from . import human_results
    human_results.OUT=OUT;human_results.draw()


if __name__=='__main__':run()
