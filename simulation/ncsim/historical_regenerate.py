"""Independent policy inference with the recovered archived geometry version."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import gzip,json,random
import numpy as np
import torch
from .data import sha256,load_scene,configuration
from .geometry import INITIAL_Q
from .historical_backend import configure,HistoricalEnv,inverse
from .historical_reproduction import FOLDERS,extract_episodes,mat_path
from .paper_evaluation import dump,write_rows
from .policies import ArchivedPolicy


def run_method(out_string,method,number):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out=Path(out_string);configure(out)
    rows=json.loads((out/'recovered_dataset_order.json').read_text(encoding='utf-8'))
    model_path=Path(configuration()['archive_root'])/f'model/RL_Point/ppo_AnzhenRobot_{number}.zip'
    policy=ArchivedPolicy(model_path,HistoricalEnv([load_scene(rows[0])]))
    sampling_rng=np.random.RandomState(0)
    results=[];macro=[];saved_frames=0;shared_frames=0;matched_actions=0;matched_xy=0;matched_scores=0;exact_length_episodes=0
    with gzip.open(out/'regenerated'/f'{method}_traces.json.gz','wt',encoding='utf-8') as f:
        f.write('[');first=True
        for batch in range(398):
            scene=load_scene(rows[-batch]);env=HistoricalEnv([scene],max_steps=41)
            q=INITIAL_Q.copy();unreach=0
            eps,_=extract_episodes(mat_path(out/'original_mat',method,batch))
            reference=dict(eps);batch_rows=[]
            for j in range(40):
                seed=batch*40+j+1;rng=random.Random(seed)
                for _ in range(10000):
                    xy=np.array([rng.uniform(.15-.35,.15+.35),rng.uniform(1.31-.4,1.31+.4)])
                    qnew,ok=inverse(xy,q)
                    if ok:q=qnew;break
                else:raise RuntimeError('No reachable seeded start')
                env.reset_start(dict(xy=xy.tolist(),q=q.tolist(),seed=seed))
                env.sampling_rng=sampling_rng;env.unreachable_count=unreach
                obs=env._observation();obs['robot_state'][-1]=0
                trace=[[0,*env.xy,0.,0.,env.score,0.]]
                while not env.finished:
                    action=policy.predict(obs,env)
                    obs,reward,_=env.step(action)
                    trace.append([env.steps,*env.xy.tolist(),*action.astype(float).tolist(),env.score,reward])
                q=env.q.copy();unreach=env.unreachable_count
                eligible=env.initial_score>=20
                r=dict(method=method,batch_index=batch,seed=seed,episode_id=j,scene_id=scene.scene_id,
                    eligible=eligible,initial_score=env.initial_score,final_score=env.score,
                    success=bool(eligible and env.score<20),improved=bool(eligible and env.score<env.initial_score),
                    actions=env.steps,plotted_steps=env.steps+1,reason=env.reason,
                    reduction_percent=(env.initial_score-env.score)/env.initial_score*100 if eligible else None)
                old=reference.get(j)
                if old is not None:
                    now=np.array(trace,dtype=float);n=min(len(now),len(old));shared_frames+=n
                    matched_xy+=int(np.sum(np.max(np.abs(now[:n,1:3]-old[:n,2:4]),axis=1)<1e-6))
                    matched_scores+=int(np.sum(np.abs(now[:n,5]-old[:n,6])<1e-4))
                    matched_actions+=int(np.sum(np.max(np.abs(now[1:n,3:5]-old[1:n,4:6]),axis=1)<1e-5))
                    same_length=len(now)==len(old);exact_length_episodes+=int(same_length)
                    r.update(historical_present=True,historical_same_length=same_length,
                        historical_success=bool(old[-1,6]<20),historical_final_score=float(old[-1,6]))
                else:
                    r.update(historical_present=False,historical_same_length=False,historical_success=None,historical_final_score=None)
                results.append(r)
                if eligible:batch_rows.append(r)
                record=dict(summary=r,trace=trace)
                if not first:f.write(',')
                first=False;json.dump(record,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
                saved_frames+=len(trace)
            if batch_rows:
                macro.append([np.mean([r['success'] for r in batch_rows]),np.mean([r['improved'] for r in batch_rows])])
            if (batch+1)%80==0:print(f'{method} regeneration {batch+1}/398',flush=True)
        f.write(']')
    eligible=[r for r in results if r['eligible']];improved=[r for r in eligible if r['improved']]
    stats=dict(method=method,candidates=len(results),eligible=len(eligible),
        macro_success_percent=float(np.mean(np.array(macro)[:,0])*100),
        macro_improvement_percent=float(np.mean(np.array(macro)[:,1])*100),
        mean_final_improved=float(np.mean([r['final_score'] for r in improved])),
        mean_reduction_percent_improved=float(np.mean([r['reduction_percent'] for r in improved])),
        mean_plotted_steps_improved=float(np.mean([r['plotted_steps'] for r in improved])),
        recorded_reference_episodes=sum(r['historical_present'] for r in results),
        episodes_same_record_count=exact_length_episodes,compared_common_prefix_frames=shared_frames,
        frames_xy_within_1um=matched_xy,frames_score_within_1e_4=matched_scores,
        noninitial_actions_within_1e_5=matched_actions,generated_frames=saved_frames)
    write_rows(out/'regenerated'/f'{method}_episodes.csv',results)
    dump(out/'regenerated'/f'{method}_summary.json',stats)
    return stats


def run(out):
    out=Path(out).resolve();(out/'regenerated').mkdir(exist_ok=False)
    archive=Path(configuration()['archive_root'])
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Independent model inference; original MAT actions are compared after generation, never fed back to the policy.',
        scene_order='dataset[-j] for j=0..397, exactly the archived evaluation loop',
        steps='Original >40 stopping condition, allowing 41 actions',
        reset='Carry prior joint guess and unreachable counter within scene; reset observation retry feature to zero, as source does',
        geometry='Archived Myclass_修改模型备份.py: chain 0.5405 m; camera .4+.1405 m; same native KDL and deployment collision code',
        sampling='Newly fixed NumPy MT19937 seed 0 once per model, continuous across scenes. Historical state unavailable. No seed search.',
        cpu_inference='Stored deterministic policy mean; CPU; numerical cross-platform identity is not assumed',
        models={m:dict(path=str(archive/f'model/RL_Point/ppo_AnzhenRobot_{n}.zip'),sha256=sha256(archive/f'model/RL_Point/ppo_AnzhenRobot_{n}.zip')) for m,n in zip(FOLDERS,[20,21,22])},
        source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('historical_backend.py'),archive/'myenv/Myclass_修改模型备份.py']},
        backend_manifest_sha256=sha256(out/'historical_backend_manifest.json'))
    dump(out/'regenerated/protocol.json',protocol)
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(run_method,str(out),m,n) for m,n in zip(FOLDERS,[20,21,22])]
        stats=[f.result() for f in as_completed(jobs)]
    stats.sort(key=lambda s:list(FOLDERS).index(s['method']))
    dump(out/'regenerated/summary.json',stats)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();run(a.output)
