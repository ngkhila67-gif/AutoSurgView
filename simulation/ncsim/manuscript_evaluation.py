"""Controlled 398 x 40 evaluation aligned with section 4.1, without changing archives."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import gzip, json, random, time
import numpy as np
import torch
from .data import ROOT, configuration, load_scene, sha256
from .geometry import INITIAL_Q, self_collision
from .historical_backend import configure, inverse, HistoricalEnv
from .paper_evaluation import METHODS, dump, write_rows, ordered_archive_rows, read_scene, validate_scene
from .policies import ArchivedPolicy

HISTORY=ROOT/'outputs/historical_reproduction_20261002'
VERSION='manuscript_40_paired_05405_v1'
_POLICIES={}


def freeze(out):
    out.mkdir(parents=True,exist_ok=False);(out/'scenes').mkdir()
    configure(HISTORY)
    rows=ordered_archive_rows()
    for r in rows:
        assert sha256(r['source'])==r['source_sha256'] and sha256(r['target_source'])==r['target_sha256']
    dump(out/'ordered_dataset.json',rows)
    starts=[]
    for i,row in enumerate(rows[1592:]):
        for j in range(40):
            seed=i*40+j+1;rng=random.Random(seed)
            for attempt in range(1,10001):
                xy=np.array([rng.uniform(.15-.35,.15+.35),rng.uniform(1.31-.4,1.31+.4)])
                q,ok=inverse(xy,INITIAL_Q)
                if ok:break
            else:raise RuntimeError('No reachable seeded start')
            starts.append(dict(seed=seed,xy=xy.tolist(),q=q.tolist(),attempts=attempt,
                initial_self_collision=self_collision(q),scene_index=i,start_index=j,
                case_id=f'scene_{i+1:03d}_start_{j+1:02d}',scene_id=row['scene_id']))
    dump(out/'starts.json',starts)
    archive=Path(configuration()['archive_root'])
    source=[Path(__file__),*[ROOT/'ncsim'/f for f in ('historical_backend.py','paper_evaluation.py','geometry.py','native_backend.py','environment.py','data.py','policies.py')],archive/'myenv/Myclass_修改模型备份.py',ROOT.parent/'JFR0511.docx']
    train_groups={r['capture_group'] for r in rows[:1592]};test_groups={r['capture_group'] for r in rows[1592:]}
    protocol=dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        scene_count=398,starts_per_scene=40,models=3,max_decision_steps=40,threshold=20.,fast_zone='Successful and actual decisions < 12',
        geometry='Original archived backup: 0.5405 m chain; original camera transform .4+.1405; native KDL, no substitute solver',
        scene_order='Current archived loader order [1592:1990], all 398 files once; fixes dataset[-0]',
        pairing='Identical reachable position, joint seed, retry counter zero and NumPy sampling seed across all three models',
        starts='Python Random(seed); original workspace; fixed original joint guess for independent episodes',
        sampling='Newly fixed per-start legacy MT19937 stream; not the unrecorded historical RNG',
        counts='Initial frame is step 0; only policy calls count as decisions. Initially clear starts are recorded separately.',
        metrics='Scene-equal macro success/improvement; all initially occluded means and Fig14 improved-only means reported separately',
        historical_metrics='Historical figures remain tied to original MAT files, not replaced by this new controlled run',
        limitations=['Historical checkpoint training exposure unverified; file-level split is not proof of independent surgery-level generalization',
            'Archived NR+pinv has no explicit joint bounds; no documented bounds invented',
            'Collision is original NC deployment source; historical Python binary identity unverified',
            'Static recorded point clouds and kinematic decisions, not physical time or robot dynamics'],
        capture_groups_crossing_split=sorted(train_groups&test_groups),
        source_sha256={str(p):sha256(p) for p in source},
        model_files={m:dict(path=str(archive/f'model/RL_Point/ppo_AnzhenRobot_{n}.zip'),sha256=sha256(archive/f'model/RL_Point/ppo_AnzhenRobot_{n}.zip')) for m,n in METHODS.items()},
        historical_backend_manifest_sha256=sha256(HISTORY/'historical_backend_manifest.json'),
        historical_backend_dll_sha256=sha256(HISTORY/'historical_backend_05405.dll'),
        original_collision_dll_sha256=sha256(ROOT/'native/original_backend.dll'),
        dataset_sha256=sha256(out/'ordered_dataset.json'),starts_sha256=sha256(out/'starts.json'))
    dump(out/'protocol.json',protocol)
    return protocol


def init_worker(models,first):
    global _POLICIES
    torch.set_num_threads(1);torch.set_num_interop_threads(1);configure(HISTORY)
    env=HistoricalEnv([load_scene(first)],max_steps=40)
    _POLICIES={m:ArchivedPolicy(s['path'],env) for m,s in models.items()}


def evaluate_scene(out_string,index,row,starts,digest):
    scene=load_scene(row);episodes=[];begin=time.perf_counter()
    for start in starts:
        initial=[]
        for method,policy in _POLICIES.items():
            env=HistoricalEnv([scene],max_steps=40);obs=env.reset_start(start)
            initial.append((env.score,*env.q,*env.xy));trace=[env.compact_frame()];total=0.
            while not env.finished:
                action=policy.predict(obs,env);obs,reward,ok=env.step(action)
                total+=reward;trace.append(env.compact_frame(action,reward,ok))
            eligible=env.initial_score>=20
            r=dict(case_id=start['case_id'],scene_index=index,scene_id=scene.scene_id,
                start_index=start['start_index'],seed=start['seed'],method=method,
                initial_x_m=start['xy'][0],initial_y_m=start['xy'][1],initial_score=env.initial_score,
                initially_clear=not eligible,eligible=eligible,final_score=env.score,
                success=bool(eligible and env.score<20),strict_success=bool(eligible and env.score<20 and not env.collided),
                improved=bool(eligible and env.score<env.initial_score),
                reduction_fraction=(env.initial_score-env.score)/env.initial_score if eligible else None,
                decision_steps=env.steps,path_length_m=env.path_length,termination_reason=env.reason,
                self_collision=env.collided,initial_self_collision=start['initial_self_collision'],total_reward=total)
            episodes.append(dict(summary=r,trace=trace))
        assert all(x==initial[0] for x in initial)
    record=dict(protocol_sha256=digest,scene_index=index,scene_id=scene.scene_id,episodes=episodes,
        elapsed_seconds=time.perf_counter()-begin)
    validate_scene(record,index,digest)
    path=Path(out_string)/'scenes'/f'scene_{index+1:03d}.json.gz';temp=path.with_suffix('.tmp')
    with gzip.open(temp,'wt',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    temp.replace(path)
    return index


def summarize(out,protocol):
    rows=[];hashes={};frames=0;digest=sha256(out/'protocol.json')
    with gzip.open(out/'steps.csv.gz','wt',encoding='utf-8-sig',newline='') as f:
        import csv
        writer=csv.writer(f);writer.writerow(['case_id','method','step','score','x','y','q1','q2','q3','action_x','action_y','reward','reason'])
        for i in range(398):
            path=out/'scenes'/f'scene_{i+1:03d}.json.gz';rec=read_scene(path);validate_scene(rec,i,digest);hashes[path.name]=sha256(path)
            for ep in rec['episodes']:
                r=ep['summary'];rows.append(r)
                for t in ep['trace']:
                    writer.writerow([r['case_id'],r['method'],t['step'],t['score'],*t['xy'],*t['q'],*(t['action'] or [None,None]),t['reward'],t['reason']]);frames+=1
    write_rows(out/'episodes.csv',rows)
    stats=[];per_scene=[]
    for m in METHODS:
        a=[r for r in rows if r['method']==m];e=[r for r in a if r['eligible']];im=[r for r in e if r['improved']]
        macro=[]
        for i in range(398):
            v=[r for r in e if r['scene_index']==i]
            entry=dict(method=m,scene_index=i,eligible=len(v),success_percent=float(np.mean([x['success'] for x in v])*100) if v else None,improved_percent=float(np.mean([x['improved'] for x in v])*100) if v else None)
            per_scene.append(entry)
            if v:macro.append(entry)
        mean=lambda subset,key:float(np.mean([r[key] for r in subset])) if subset else None
        stats.append(dict(method=m,candidates=len(a),eligible=len(e),initially_clear=len(a)-len(e),
            macro_success_percent=mean(macro,'success_percent'),macro_improved_percent=mean(macro,'improved_percent'),
            pooled_success_percent=mean(e,'success')*100,strict_success_percent=mean(e,'strict_success')*100,
            mean_final_all=mean(e,'final_score'),mean_reduction_percent_all=mean(e,'reduction_fraction')*100,
            mean_actions_all=mean(e,'decision_steps'),mean_final_improved=mean(im,'final_score'),
            mean_reduction_percent_improved=mean(im,'reduction_fraction')*100,mean_actions_improved=mean(im,'decision_steps'),
            fast_successes=sum(r['success'] and r['decision_steps']<12 for r in e),
            max_actions=max(r['decision_steps'] for r in a),self_collision_cases=sum(r['self_collision'] for r in e),
            step_limit_cases=sum(r['termination_reason']=='step_limit' for r in e)))
    pairs={}
    for r in rows:pairs.setdefault(r['case_id'],[]).append((r['initial_x_m'],r['initial_y_m'],r['initial_score']))
    assert len(rows)==47760 and len(pairs)==15920 and all(len(v)==3 and len(set(v))==1 for v in pairs.values())
    write_rows(out/'summary.csv',stats);write_rows(out/'per_scene_summary.csv',per_scene);dump(out/'summary.json',stats)
    dump(out/'verification.json',dict(candidates=47760,paired_starts=15920,scene_files=398,frames=frames,
        maximum_actions=max(r['decision_steps'] for r in rows),all_pairs_identical=True,
        source_hashes_unchanged=all(sha256(p)==h for p,h in protocol['source_sha256'].items()),scene_sha256=hashes))
    dump(out/'progress.json',dict(status='complete',completed_scenes=398,completed_model_trials=47760))
    return stats


def run(output,workers=4,limit=None):
    out=Path(output).resolve()
    protocol=freeze(out) if not (out/'protocol.json').exists() else json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['version']==VERSION
    for p,h in protocol['source_sha256'].items():assert sha256(p)==h,p
    assert sha256(out/'ordered_dataset.json')==protocol['dataset_sha256'] and sha256(out/'starts.json')==protocol['starts_sha256']
    for s in protocol['model_files'].values():assert sha256(s['path'])==s['sha256']
    assert sha256(HISTORY/'historical_backend_05405.dll')==protocol['historical_backend_dll_sha256']
    assert sha256(ROOT/'native/original_backend.dll')==protocol['original_collision_dll_sha256']
    rows=json.loads((out/'ordered_dataset.json').read_text(encoding='utf-8'))[1592:];starts=json.loads((out/'starts.json').read_text(encoding='utf-8'))
    digest=sha256(out/'protocol.json');todo=[];done=0
    for i in range(limit or 398):
        p=out/'scenes'/f'scene_{i+1:03d}.json.gz'
        if p.exists():validate_scene(read_scene(p),i,digest);done+=1
        else:todo.append(i)
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(protocol['model_files'],rows[0])) as pool:
        jobs=[pool.submit(evaluate_scene,str(out),i,rows[i],starts[i*40:(i+1)*40],digest) for i in todo]
        for job in as_completed(jobs):
            job.result();done+=1
            dump(out/'progress.json',dict(status='running',completed_scenes=done,total_scenes=398,completed_model_trials=done*120))
            if done%40==0:print(f'Completed {done}/398',flush=True)
    if limit is None:print(json.dumps(summarize(out,protocol),indent=2),flush=True)
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    a=p.parse_args();run(a.output,a.workers,a.limit)
