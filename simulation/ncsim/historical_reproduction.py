"""Recompute the paper figures from the recovered historical MAT batches.

These are historical observations, not newly generated policy trajectories.
Kinematic replay is audited separately and never replaces the recorded scores.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import csv,gzip,json,random,re,shutil
import numpy as np
from scipy.io import loadmat
from .data import ROOT,configuration,sha256,load_scene
from .paper_evaluation import dump,write_rows,ordered_archive_rows,PaperEnv
from .geometry import INITIAL_Q,inverse
from .policies import LABELS

FOLDERS={"fusion":"Matlab_Data","pointcloud":"Matlab_Data_only_PC","robotstate":"Matlab_Data_only_state"}
PUBLISHED={"fusion":[96.8,99.7,11.2,93.3,7.0],"pointcloud":[85.7,95.4,26.1,89.3,7.8],"robotstate":[92.4,99.0,19.3,91.2,9.0]}


def mat_path(root,method,batch):
    first=batch*40+1
    return Path(root)/FOLDERS[method]/f"EvaResult_Seed_{first}_to_{first+39}.mat"


def extract_episodes(path):
    content=loadmat(path)
    data=np.asarray(content["data"],dtype=float)
    if data.ndim!=2 or data.shape[1]!=9 or not np.isfinite(data).all():
        raise ValueError(f"Invalid historical matrix: {path}")
    episodes=[]
    for eid in np.unique(data[:,0]):
        if eid!=int(eid) or not 0<=eid<40:
            raise ValueError("Invalid episode ID")
        e=data[data[:,0]==eid]
        e=e[np.argsort(e[:,1],kind="stable")]
        if e[0,1]!=0 or len(set(e[:,1]))!=len(e):
            raise ValueError("Invalid step ordering")
        if not np.all(e[:,8]==e[-1,8]):
            raise ValueError("Historical success flags disagree within episode")
        episodes.append((int(eid),e))
    return episodes,content["__header__"].decode("ascii",errors="replace")


def collect(out):
    archive=Path(configuration()["archive_root"])
    sources=[];episodes=[];batch_rows=[]
    with gzip.open(out/"historical_steps.csv.gz","wt",encoding="utf-8-sig",newline="") as sf:
        w=csv.writer(sf)
        w.writerow(["method","batch_index","seed","episode_id","row_index","stored_step_id","robot_x","robot_y","action_x","action_y","occlusion","reward","success_flag"])
        for method,folder in FOLDERS.items():
            paths=list((archive/folder).glob("*.mat"))
            expected={mat_path(archive,method,i).name for i in range(398)}
            if {p.name for p in paths}!=expected:
                raise ValueError(f"Historical 398-batch coverage incomplete: {folder}")
            snapshot=out/"original_mat"/folder
            snapshot.mkdir(parents=True,exist_ok=True)
            for batch in range(398):
                path=mat_path(archive,method,batch)
                eps,header=extract_episodes(path)
                target=snapshot/path.name
                shutil.copy2(path,target)
                if sha256(path)!=sha256(target):
                    raise AssertionError("Source snapshot checksum mismatch")
                sources.append(dict(method=method,batch_index=batch,source=str(path),snapshot=str(target.relative_to(out)),
                    sha256=sha256(path),header=header,episodes=len(eps)))
                records=[]
                for eid,e in eps:
                    initial,final=e[0,6],e[-1,6]
                    r=dict(method=method,batch_index=batch,episode_id=eid,seed=batch*40+eid+1,
                        initial_x=float(e[0,2]),initial_y=float(e[0,3]),initial_score=float(initial),final_score=float(final),
                        success_flag=bool(e[-1,8]>.5),threshold_success=bool(final<20),improved=bool(final<initial),
                        reduction_percent=float((initial-final)/initial*100),
                        plotted_steps=len(e),actual_actions=len(e)-1,stored_final_step_id=int(e[-1,1]),
                        stored_step_ids_double_actions=bool(np.array_equal(e[:,1],2*np.arange(len(e)))))
                    if r["success_flag"]!=r["threshold_success"] or initial<20:
                        raise ValueError("Historical success/population differs from source evaluator")
                    for k,v in enumerate(e):
                        w.writerow([method,batch,r["seed"],eid,k,*v[1:].tolist()])
                    episodes.append(r);records.append(r)
                batch_rows.append(dict(method=method,batch_index=batch,logged_episodes=len(records),
                    successes=sum(r["success_flag"] for r in records),improvements=sum(r["improved"] for r in records),
                    success_percent=100*np.mean([r["success_flag"] for r in records]),
                    improvement_percent=100*np.mean([r["improved"] for r in records])))
    stats=[]
    for method in FOLDERS:
        er=[r for r in episodes if r["method"]==method]
        br=[r for r in batch_rows if r["method"]==method]
        improved=[r for r in er if r["improved"]]
        stat=dict(method=method,batches=len(br),candidate_seeds=15920,logged_occluded_episodes=len(er),
            unlogged_candidate_seeds=15920-len(er),macro_success_percent=float(np.mean([r["success_percent"] for r in br])),
            macro_improvement_percent=float(np.mean([r["improvement_percent"] for r in br])),
            micro_success_percent=float(100*np.mean([r["success_flag"] for r in er])),
            mean_final_improved=float(np.mean([r["final_score"] for r in improved])),
            mean_reduction_percent_improved=float(np.mean([r["reduction_percent"] for r in improved])),
            mean_plotted_steps_improved=float(np.mean([r["plotted_steps"] for r in improved])),
            mean_actual_actions_improved=float(np.mean([r["actual_actions"] for r in improved])),
            zero_success_batches=sum(r["successes"]==0 for r in br),improved_episodes=len(improved),
            maximum_actual_actions=max(r["actual_actions"] for r in er),maximum_plotted_steps=max(r["plotted_steps"] for r in er),
            all_stored_step_ids_double_actions=all(r["stored_step_ids_double_actions"] for r in er))
        metrics=[stat[k] for k in ["macro_success_percent","macro_improvement_percent","mean_final_improved","mean_reduction_percent_improved","mean_plotted_steps_improved"]]
        stat["rounded_values"]=[float(f"{v:.1f}") for v in metrics]
        stat["paper_values"]=PUBLISHED[method]
        stat["all_five_rounded_values_match"]=stat["rounded_values"]==PUBLISHED[method]
        stats.append(stat)
    write_rows(out/"historical_episodes.csv",episodes)
    write_rows(out/"historical_batches.csv",batch_rows)
    dump(out/"historical_metrics.json",stats)
    dump(out/"original_files.json",sources)
    dump(out/"recovered_dataset_order.json",ordered_archive_rows())
    scripts={}
    for name in ["xaiorong.m","plot_cdf_heatmap_gai.m","plot_cdf_heatmap.m"]:
        src=archive.parent/"主动规避遮挡/SCI对应"/name
        if src.exists():
            dst=out/"original_scripts"/name
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(src,dst)
            scripts[name]=dict(source=str(src),sha256=sha256(src))
    dump(out/"source_provenance.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        original_mat_files=len(sources),recorded_episodes=len(episodes),candidate_seeds_per_model=15920,
        scripts=scripts,paper_sha256=sha256(ROOT.parent/"JFR0511.docx"),
        source_note="All three MAT folders were recovered in the original project root, not mytest. Earlier conclusions based only on text logs are superseded by these records.",
        metrics_note="Faithful xaiorong.m calculations: per-batch mean rates, improved-only distributions, number of rows INCLUDING initial state.",
        missing_fields=["sampled point clouds","NumPy RNG state","joint angles","policy checksum","scene pathname","historical collision binary identity"]))
    return episodes,stats


def replay_batch(args):
    """Replay recorded actions using current archived-source KDL; test, not fit."""
    out_string,batch,row,*options=args
    out=Path(out_string);scene=load_scene(row);results=[]
    version=options[0] if options else 'current'
    env_type,ik_function=PaperEnv,inverse
    replay_dir='action_replay'
    if version=='backup':
        from .historical_backend import configure,HistoricalEnv,inverse as backup_inverse
        configure(out)
        env_type,ik_function=HistoricalEnv,backup_inverse
        replay_dir='action_replay_backup'
    for method in FOLDERS:
        env=env_type([scene],max_steps=41)
        q=INITIAL_Q.copy();unreachable=0
        eps,_=extract_episodes(mat_path(out/"original_mat",method,batch))
        lookup=dict(eps)
        score_errors=[];xy_errors=[];states=[];missing_occluded=0;premature=0;actions=0
        for j in range(40):
            rng=random.Random(batch*40+j+1)
            for _ in range(10000):
                xy=np.array([rng.uniform(.15-.35,.15+.35),rng.uniform(1.31-.4,1.31+.4)])
                solved,ok=ik_function(xy,q)
                if ok:
                    q=solved;break
            else:
                raise RuntimeError("Historical start reconstruction failed")
            env.reset_start(dict(xy=xy.tolist(),q=q.tolist(),seed=batch*40+j+1))
            env.unreachable_count=unreachable
            ep=lookup.get(j)
            if ep is None:
                missing_occluded+=int(env.score>=20)
            else:
                scores=[env.score-float(ep[0,6])]
                xys=[float(np.max(np.abs(env.xy-ep[0,2:4])))]
                for k,v in enumerate(ep[1:],1):
                    # Continue exactly the recorded number of actions, even if
                    # this backend would terminate earlier; retain mismatch count.
                    if env.finished:
                        premature+=1
                    env.finished=False;env.reason="running"
                    _,_,ok=env.step(v[4:6].astype(np.float32))
                    actions+=1
                    scores.append(env.score-float(v[6]))
                    xys.append(float(np.max(np.abs(env.xy-v[2:4]))))
                score_errors.extend(scores);xy_errors.extend(xys)
                states.append(dict(episode_id=j,seed=batch*40+j+1,frames=len(ep),
                    maximum_score_error=max(map(abs,scores)),maximum_xy_error_m=max(xys),
                    matching_score_frames=sum(abs(v)<1e-6 for v in scores)))
            q=env.q.copy();unreachable=env.unreachable_count
        results.append(dict(method=method,batch_index=batch,scene_id=scene.scene_id,
            archive_index=row["archive_index_zero_based"],logged_episodes=len(eps),frames=len(score_errors),actions=actions,
            max_score_error=max(map(abs,score_errors)),max_xy_error_m=max(xy_errors),
            matching_score_frames=sum(abs(v)<1e-6 for v in score_errors),
            matching_xy_frames=sum(abs(v)<1e-10 for v in xy_errors),
            originally_unlogged_but_now_occluded=missing_occluded,continued_after_new_backend_done=premature,
            episodes=states))
    with gzip.open(out/replay_dir/f"batch_{batch+1:03d}.json.gz","wt",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,allow_nan=False)
    return batch


def audit_actions(out,workers=4,version='current'):
    rows=json.loads((out/"recovered_dataset_order.json").read_text(encoding="utf-8"))
    replay_dir='action_replay_backup' if version=='backup' else 'action_replay'
    (out/replay_dir).mkdir(exist_ok=True)
    # Source evaluator uses dataset[-j], including dataset[0] for j=0.
    tasks=[(str(out),j,rows[-j],version) for j in range(398)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(replay_batch,t) for t in tasks]
        for n,f in enumerate(as_completed(futures),1):
            f.result()
            if n%40==0 or n==398:
                print(f"Historical action audit {n}/398 batches",flush=True)
    all_rows=[]
    for j in range(398):
        with gzip.open(out/replay_dir/f"batch_{j+1:03d}.json.gz","rt",encoding="utf-8") as f:
            all_rows.extend(json.load(f))
    summary=[]
    for method in FOLDERS:
        rs=[r for r in all_rows if r["method"]==method]
        summary.append(dict(method=method,batches=len(rs),frames=sum(r["frames"] for r in rs),
            matching_score_frames=sum(r["matching_score_frames"] for r in rs),
            matching_xy_frames=sum(r["matching_xy_frames"] for r in rs),max_score_error=max(r["max_score_error"] for r in rs),
            max_xy_error_m=max(r["max_xy_error_m"] for r in rs),
            originally_unlogged_but_now_occluded=sum(r["originally_unlogged_but_now_occluded"] for r in rs),
            continued_after_new_backend_done=sum(r["continued_after_new_backend_done"] for r in rs)))
    dump(out/(replay_dir+"_audit.json"),dict(version=version,protocol="Archived negative-index scene order; Python seeded starts; carried joint guess and retry counter; recorded actions fed to source-backed native backend; no policy inference and no fitting",
        scope="Checks consistency of recorded actions with current reconstructed native geometry. Does NOT verify regeneration of policy actions or historical joint angles.",summary=summary))
    return summary


def run(out=None,workers=4):
    out=Path(out) if out else ROOT/"outputs"/"historical_reproduction_20261002"
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    episodes,stats=collect(out)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
    audit_actions(out,workers)
    return out


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output");p.add_argument("--workers",type=int,default=4)
    p.add_argument('--audit-backup',action='store_true')
    a=p.parse_args()
    if a.audit_backup:
        audit_actions(Path(a.output).resolve(),a.workers,'backup')
    else:
        run(a.output,a.workers)
