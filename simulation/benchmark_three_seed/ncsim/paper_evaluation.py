"""Frozen, paired reconstruction of JFR0511 section 4.1 (398 x 40 x 3).

Uses the archived trained weights and native KDL/collision; never trains or tunes.
Historical ambiguities and corrected bookkeeping are part of the output protocol.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import csv
import gzip
import hashlib
import json
import os
import platform
import random
import time
import numpy as np
import torch
from .data import ROOT, configuration, scene_index, load_scene, sha256
from .environment import SurgicalCameraEnv
from .geometry import INITIAL_Q, HEIGHT, inverse, self_collision, sample_points
from .native_backend import provenance
from .policies import ArchivedPolicy, LABELS

METHODS = {"fusion":20, "pointcloud":21, "robotstate":22}
VERSION = "paper398_paired_v1"


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ordered_archive_rows():
    """Repeat read_all_pcd's CURRENT os.listdir order; save it, never infer old order."""
    config = configuration()
    lookup = {str(Path(r["source"]).resolve()).casefold():r for r in scene_index()["scenes"]}
    root = Path(config["archive_root"])/"mytest"/"pointdata"
    rows = []
    for session, groups in zip(config["scene_sessions"], [50,70,81,77]):
        for group in range(1, groups+1):
            folder = root/session/f"data{group}"
            for name in os.listdir(folder):
                if name.endswith(".pcd"):
                    row = dict(lookup[str((folder/name).resolve()).casefold()])
                    row["archive_index_zero_based"] = len(rows)
                    row["paper_split"] = "train" if len(rows)<1592 else "test"
                    rows.append(row)
    if len(rows)!=1990 or len({r["source"] for r in rows})!=1990:
        raise ValueError("The paper requires exactly 1990 unique scene files")
    return rows


def reachable_start(seed):
    rng = random.Random(seed)
    for attempt in range(1,10001):
        xy = np.array([rng.uniform(.15-.35,.15+.35), rng.uniform(1.31-.4,1.31+.4)])
        q, ok = inverse(xy, INITIAL_Q)
        if ok:
            return dict(seed=seed, xy=xy.tolist(), q=q.tolist(), attempts=attempt,
                        initial_self_collision=self_collision(q))
    raise RuntimeError(f"Original KDL could not find start for seed {seed}")


class PaperEnv(SurgicalCameraEnv):
    """Paper horizon, source float32 action normalization and requested XY state.

    The starting IK guess and retry counter are reset for paired independent
    episodes. A frozen legacy NumPy RNG makes previously unseeded sampling explicit.
    """
    def reset_start(self, start):
        self.scene = self.scenes[0]
        self.points = self.scene.points[(self.scene.points[:,2]>=self.scene.target[2]+.2)&
                                       (self.scene.points[:,2]<=HEIGHT-.05)].copy()
        self.xy = np.array(start["xy"], dtype=float)
        self.q = np.array(start["q"], dtype=float)
        self.sampling_rng = np.random.RandomState(start["seed"])
        self.steps, self.unreachable_count, self.path_length = 0,0,0.
        self.collided, self.finished, self.reason = False,False,"running"
        self._measure()
        self.initial_score = self.score
        if self.score<self.threshold:
            self.finished,self.reason = True,"initially_clear"
        return self._observation()

    def _observation(self):
        T = self.geometry[0]
        return dict(robot_state=np.array([(self.xy[0]+1)/2,(self.xy[1]-.8)/.95,
                    min(self.score/500,1),*self.guide,self.unreachable_count/5], dtype=np.float32),
                    point_cloud_in=sample_points(self.points[self.inside],self.scene.target,T,256,self.sampling_rng),
                    point_cloud_out=sample_points(self.points[~self.inside],self.scene.target,T,512,self.sampling_rng))

    def displacement(self, action):
        delta = np.asarray(action, dtype=np.float32).copy()
        if delta.shape!=(2,) or not np.isfinite(delta).all():
            raise ValueError("Invalid archived policy action")
        delta = np.clip(delta,-1,1)
        norm = np.linalg.norm(delta)
        if norm>self.action_step:
            delta /= norm
            delta *= self.action_step
        return delta

    def step(self, action):
        if self.finished:
            raise RuntimeError("Cannot step a finished trial")
        old = self.score
        delta = self.displacement(action)
        candidate = self.xy+delta
        q,ok = inverse(candidate,self.q)
        self.steps += 1
        if ok:
            self.q,self.xy = q,candidate
            self.path_length += float(np.linalg.norm(delta))
            self.unreachable_count = 0
            self._measure()
            reward = -self.score/max(old,1e-9)+(0.5 if self.score<old else 0.)
        else:
            self.unreachable_count += 1
            reward = -float(self.unreachable_count)
        self.collided = self_collision(self.q)
        if self.collided:
            self.reason,self.finished,reward = "self_collision",True,reward-5
        elif ok and self.score<self.threshold:
            self.reason,self.finished,reward = "criterion_met",True,reward+5
        elif self.unreachable_count>5:
            self.reason,self.finished,reward = "repeated_unreachable",True,reward-5
        elif self.steps>=self.max_steps:
            self.reason,self.finished,reward = "step_limit",True,reward-5
        return self._observation(),float(reward),ok

    def compact_frame(self, action=None, reward=None, ik_ok=None):
        return dict(step=self.steps, score=self.score, xy=self.xy.tolist(), q=self.q.tolist(),
                    reason=self.reason, self_collision=self.collided, unreachable_count=self.unreachable_count,
                    path_length_m=self.path_length, action=None if action is None else np.asarray(action).tolist(),
                    reward=reward, ik_ok=ik_ok)


def freeze(out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    (out/"scenes").mkdir()
    rows = ordered_archive_rows()
    # Hash both labels and point clouds, including the reconstructed train split.
    for r in rows:
        if sha256(r["source"])!=r["source_sha256"] or sha256(r["target_source"])!=r["target_sha256"]:
            raise ValueError(f"Archived source changed: {r['scene_id']}")
    dump(out/"ordered_dataset.json",rows)
    starts=[]
    for i,row in enumerate(rows[1592:]):
        for j in range(40):
            start=reachable_start(i*40+j+1)
            start.update(scene_index=i,start_index=j,case_id=f"scene_{i+1:03d}_start_{j+1:02d}",scene_id=row["scene_id"])
            starts.append(start)
    dump(out/"starts.json",starts)
    train_groups={r["capture_group"] for r in rows[:1592]}
    test_groups={r["capture_group"] for r in rows[1592:]}
    archive=Path(configuration()["archive_root"])
    source_paths=[ROOT/"ncsim"/name for name in ("paper_evaluation.py","data.py","geometry.py","native_backend.py","policies.py","environment.py")]
    source_paths += [archive/"myenv/Myclass.py",archive/"mytest/train_AnzhenRobotPoint.py",ROOT.parent/"JFR0511.docx"]
    protocol=dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        status="frozen_before_full_evaluation",purpose="full_scale_reconstruction_not_verified_historical_identity",
        scene_count=398,starts_per_scene=40,candidates_per_model=15920,total_model_trials=47760,
        threshold=20.,max_decision_steps=40,action_max_norm_m=.08,
        primary_population="initial_score >= 20; initially clear candidates recorded separately and not called recoveries",
        primary_success="final_score < 20 (paper definition); collision-free success also reported",
        primary_means="all initially occluded trials; Figure 14 improved-only means reported separately",
        dataset_order="Current archive os.listdir order, four sessions and ascending numbered folders; take [1592:1990]",
        seed_assignment="Ascending heldout index i: Python random.Random(i*40+j+1), j=0..39",
        initial_pose="Original KDL with fixed [-50,100,-50] degree seed for each start; accept reachable poses, no extra collision rejection",
        sampling="NumPy RandomState(start seed), original choice/replacement algorithm; reset identically for each model",
        policy="Archived weights, deterministic clipped mean action, no observation normalization or retraining",
        test_capture_groups=len(test_groups),capture_groups_crossing_train_test=sorted(train_groups&test_groups),
        native_backend=provenance(),models={m:dict(path=str(archive/f'model/RL_Point/ppo_AnzhenRobot_{n}.zip'),
            sha256=sha256(archive/f'model/RL_Point/ppo_AnzhenRobot_{n}.zip')) for m,n in METHODS.items()},
        versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),
        source_sha256={str(p):sha256(p) for p in source_paths},
        ordered_dataset_sha256=sha256(out/"ordered_dataset.json"),starts_sha256=sha256(out/"starts.json"),
        disclosed_differences=[
            "Original commented loop dataset[-j] selects dataset[0] at j=0 and drops heldout index1592. This run uses all 398 heldout files once, ascending.",
            "Historical filesystem enumeration and checkpoint training exposure were not saved. Current file-level split is reconstructed, not proven historically unseen.",
            "Original reset retains previous theta and unreachable counter. This run resets both and freezes identical starts across the three methods.",
            "Original NumPy point sampling was unseeded. This run freezes a legacy MT19937 seed per start; exact historical trajectories cannot be asserted.",
            "Original termination interaction>40 allows 41 actions. Primary results stop at40; one extra action only for step-limit trials is retained as a separate sensitivity result.",
            "Original log appends/increments twice per action; actual actions are counted once here. Old logs may come from different script versions.",
            "Original scene aggregation excludes zero-success scenes. Here no completed eligible trial or zero-success scene is discarded.",
            "Native archived KDL and deployment collision.cpp are executed, with no replacement solver. Historical Python collision binary identity remains unverified.",
            "Paper says joint limits; retrieved original NR+pinv solver does not enforce NR_JL joint limits. Original solver is retained.",
            "Static recorded point clouds and decision steps do not reproduce moving staff, robot dynamics, physical seconds or clinical outcomes."])
    dump(out/"protocol.json",protocol)
    dump(out/"progress.json",dict(status="frozen",completed_scenes=0,total_scenes=398))
    return protocol


_POLICIES = None

def init_worker(model_specs, first_row):
    global _POLICIES
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    env=PaperEnv([load_scene(first_row)])
    _POLICIES={m:ArchivedPolicy(spec["path"],env) for m,spec in model_specs.items()}


def evaluate_scene(out_string, index, row, starts, protocol_hash):
    out=Path(out_string)
    scene=load_scene(row)
    episodes=[]
    begin=time.perf_counter()
    for start in starts:
        initial_states=[]
        for method,policy in _POLICIES.items():
            env=PaperEnv([scene])
            obs=env.reset_start(start)
            initial_states.append((env.initial_score,*env.q,*env.xy))
            trace=[env.compact_frame()]
            total_reward=0.
            while not env.finished:
                action=policy.predict(obs,env)
                obs,reward,ok=env.step(action)
                total_reward+=reward
                trace.append(env.compact_frame(action,reward,ok))
            eligible=env.initial_score>=20
            result=dict(case_id=start["case_id"],scene_index=index,scene_id=scene.scene_id,
                capture_group=scene.capture_group,start_index=start["start_index"],seed=start["seed"],method=method,
                initial_x_m=start["xy"][0],initial_y_m=start["xy"][1],initial_score=env.initial_score,
                initial_self_collision=start["initial_self_collision"],initially_clear=not eligible,
                eligible=eligible,final_score=env.score,success=eligible and env.score<20,
                strict_success=eligible and env.score<20 and not env.collided,
                improved=eligible and env.score<env.initial_score,
                reduction_fraction=(env.initial_score-env.score)/env.initial_score if eligible else None,
                decision_steps=env.steps,path_length_m=env.path_length,termination_reason=env.reason,
                self_collision=env.collided,total_reward=total_reward,
                final_score_horizon41=env.score,success_horizon41=eligible and env.score<20,
                steps_horizon41=env.steps,reason_horizon41=env.reason)
            sensitivity=None
            if env.reason=="step_limit":
                env.max_steps=41
                env.finished,env.reason=False,"running"
                action=policy.predict(obs,env)
                _,reward,ok=env.step(action)
                sensitivity=env.compact_frame(action,reward,ok)
                result.update(final_score_horizon41=env.score,success_horizon41=env.score<20,
                              steps_horizon41=env.steps,reason_horizon41=env.reason)
            episodes.append(dict(summary=result,trace=trace,horizon41_extra=sensitivity))
        if not all(s==initial_states[0] for s in initial_states):
            raise AssertionError("Unpaired starts")
    record=dict(protocol_sha256=protocol_hash,scene_index=index,scene_id=scene.scene_id,
                elapsed_seconds=time.perf_counter()-begin,episodes=episodes)
    path=out/"scenes"/f"scene_{index+1:03d}.json.gz"
    tmp=path.with_suffix(".tmp")
    with gzip.open(tmp,"wt",encoding="utf-8") as f:
        json.dump(record,f,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    tmp.replace(path)
    return dict(scene_index=index,episode_records=len(episodes),seconds=record["elapsed_seconds"])


def read_scene(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        return json.load(f)


def validate_scene(record,index,protocol_hash):
    if record["protocol_sha256"]!=protocol_hash or record["scene_index"]!=index or len(record["episodes"])!=120:
        raise ValueError(f"Incomplete or incompatible checkpoint: scene {index+1}")
    keys={(e["summary"]["start_index"],e["summary"]["method"]) for e in record["episodes"]}
    if keys!={(j,m) for j in range(40) for m in METHODS}:
        raise ValueError("Missing or repeated model trial")
    for e in record["episodes"]:
        r,t=e["summary"],e["trace"]
        if len(t)!=r["decision_steps"]+1 or r["decision_steps"]>40 or t[-1]["score"]!=r["final_score"]:
            raise ValueError("Trace and summary disagree")


def run(out=None,workers=4,scene_limit=None):
    torch.set_num_threads(1)
    if out is None:
        out=ROOT/"outputs"/("paper398_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    out=Path(out).resolve()
    if not (out/"protocol.json").exists():
        freeze(out)
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    if protocol["version"]!=VERSION:
        raise ValueError("Protocol version mismatch")
    for p,digest in protocol["source_sha256"].items():
        if sha256(p)!=digest:
            raise ValueError(f"Code/source changed after protocol freeze: {p}")
    for spec in protocol["models"].values():
        if sha256(spec["path"])!=spec["sha256"]:
            raise ValueError("Archived model changed")
    if sha256(out/"ordered_dataset.json")!=protocol["ordered_dataset_sha256"] or sha256(out/"starts.json")!=protocol["starts_sha256"]:
        raise ValueError("Frozen dataset/starts modified")
    if provenance()["dll_sha256"]!=protocol["native_backend"]["dll_sha256"]:
        raise ValueError("Native engine changed")
    rows=json.loads((out/"ordered_dataset.json").read_text(encoding="utf-8"))[1592:]
    starts=json.loads((out/"starts.json").read_text(encoding="utf-8"))
    digest=sha256(out/"protocol.json")
    indices=list(range(398 if scene_limit is None else scene_limit))
    completed=[]
    for i in indices:
        p=out/"scenes"/f"scene_{i+1:03d}.json.gz"
        if p.exists():
            validate_scene(read_scene(p),i,digest)
            completed.append(i)
    remaining=[i for i in indices if i not in completed]
    print(f"OUTPUT {out}\nCompleted {len(completed)}/{len(indices)} scenes; workers={workers}",flush=True)
    begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(protocol["models"],rows[0])) as pool:
        futures={pool.submit(evaluate_scene,str(out),i,rows[i],starts[i*40:(i+1)*40],digest):i for i in remaining}
        for future in as_completed(futures):
            result=future.result()
            completed.append(result["scene_index"])
            progress=dict(status="running",completed_scenes=len(completed),total_scenes=398,
                completed_model_trials=len(completed)*120,elapsed_this_invocation_seconds=time.perf_counter()-begin,
                last_completed=result,updated_utc=datetime.now(timezone.utc).isoformat())
            dump(out/"progress.json",progress)
            print(f"Completed {len(completed)}/{len(indices)} scenes ({len(completed)*120} model trials)",flush=True)
    if scene_limit is None:
        from .paper_report import summarize
        summarize(out)
    else:
        dump(out/"progress.json",dict(status="partial_pilot",completed_scenes=len(completed),total_scenes=398))
    return out
