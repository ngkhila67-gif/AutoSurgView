"""Fresh, session-held-out multimodal PPO/SAC/TD3/TQC study; never edits archived results.

Run ``python -m ncsim.learning_benchmark --help``. A smoke run checks software,
not scientific performance. The interaction-budget pilot is not a SOTA claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
import stable_baselines3 as sb3
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC
import sb3_contrib
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.noise import NormalActionNoise

from .data import ROOT, configuration, scene_index, load_scene, sha256
from .learning_features import LearningPointNet


METHODS = {"ppo": (PPO, "MM-PPO-v2", "shaped"),
           "sac": (SAC, "MM-SAC", "shaped"),
           "td3": (TD3, "MM-TD3", "shaped"),
           "tqc": (TQC, "MM-TQC", "shaped"),
           "ppo_legacy": (PPO, "MM-PPO reward ablation", "legacy")}
CODE_FILES = ["learning_benchmark.py", "learning_env.py", "learning_evaluation.py",
              "learning_features.py",
              "policies.py", "historical_backend.py", "geometry.py", "native_backend.py",
              "paper_evaluation.py", "environment.py", "data.py"]


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # Readers may briefly hold files open on Windows.
    for i in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(.05 * (i + 1))
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {name: sha256(ROOT / "ncsim" / name) for name in CODE_FILES}


def check_protocol(out):
    p = read(out / "protocol.json")
    if source_hashes() != p["source_sha256"]:
        raise RuntimeError("Study source changed after freezing; create a new study instead of mixing implementations")
    for name, expected in p["input_sha256"].items():
        if sha256(out / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    for name, expected in p.get('backend_sha256',{}).items():
        if sha256(ROOT/name) != expected:
            raise RuntimeError(f"Frozen native backend changed: {name}")
    return p


def prepare(out, stage, steps, seeds, methods, full_test=False):
    from .learning_evaluation import freeze_cases, select_group_scenes
    from .learning_env import HistoricalLearningEnv
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(METHODS):
        raise ValueError('Methods must be unique recognized algorithms')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Training seeds must be nonempty and unique')
    if steps < 512 or steps % 512:
        raise ValueError('Training budget must be positive and divisible by 512')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = [r for r in scene_index()["scenes"] if r.get("duplicate_of") is None]
    splits = {s: [r for r in rows if r["new_training_split"] == s]
              for s in ("train", "validation", "test")}
    groups = {s: {r["capture_group"] for r in rr} for s, rr in splits.items()}
    hashes = {s: {r["source_sha256"] for r in rr} for s, rr in splits.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[a] & groups[b] or hashes[a] & hashes[b]:
            raise ValueError(f"Cross-split duplicate capture groups or point clouds: {a}/{b}")
    for s, rr in splits.items():
        dump(out / f"{s}_rows.json", rr)
    validation_rows = select_group_scenes(splits["validation"])
    test_rows = splits["test"] if full_test else select_group_scenes(splits["test"])
    if stage == "smoke":
        # Spatial/source-order sampling, unrelated to any policy outcome.
        validation_rows = [validation_rows[i] for i in np.linspace(0, len(validation_rows)-1, 4, dtype=int)]
        test_rows = [test_rows[i] for i in np.linspace(0, len(test_rows)-1, 4, dtype=int)]
    freeze_cases(validation_rows, 2, 8102026, out / "validation_cases.json")
    freeze_cases(test_rows, 40 if full_test else 2, 8202026, out / "test_cases.json")
    if steps % 512:
        raise ValueError("Training budget must be divisible by 512 to match PPO rollouts exactly")
    p = dict(created_utc=utc(), stage=stage, claim="software_smoke_only" if stage == "smoke" else "interaction_budget_pilot_not_sota",
        steps_per_method_seed=steps, seeds=seeds, methods=methods, labels={m:METHODS[m][1] for m in methods},
        shared=dict(observation="joint12 state + in-view point cloud256x3 + out-of-view point cloud512x3",
            encoder="LearningPointNet(fusion): same point encoder; state12, including unwrapped joint angles, executed previous displacement and remaining horizon",
            head_layers=[64,64], head_activation="Tanh", physical_action="0.08 * a / max(1, norm(a)); a clipped to [-1,1]^2",
            max_steps=40, success_threshold=20., reward="shaped for PPO/SAC/TD3/TQC; legacy PPO is a separate reward ablation",
            geometry="archived 0.5405 m native KDL + historical camera geometry",
            safety="Training is unfiltered; each selected checkpoint is tested both raw and with the SAME sampled path collision filter",
            normalization="No observation or reward running normalization; PPO uses its standard advantage normalization"),
        optimizer_settings=dict(ppo=dict(learning_rate=3e-4,n_steps=512,batch_size=32,n_epochs=4,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.01,target_kl=.03),
            sac=dict(learning_rate=3e-4,batch_size=32,buffer_size=2048 if stage=='smoke' else 20000,learning_starts=256 if stage=="smoke" else 2048,train_freq=8,gradient_steps=1,gamma=.99,tau=.005,ent_coef="auto"),
            tqc=dict(learning_rate=3e-4,batch_size=32,buffer_size=2048 if stage=="smoke" else 20000,learning_starts=256 if stage=="smoke" else 2048,train_freq=8,gradient_steps=1,gamma=.99,tau=.005,ent_coef="auto",top_quantiles_to_drop_per_net=2),
            td3=dict(learning_rate=3e-4,batch_size=32,buffer_size=2048 if stage=='smoke' else 20000,learning_starts=256 if stage=="smoke" else 2048,train_freq=8,gradient_steps=1,gamma=.99,tau=.005,policy_delay=2,target_policy_noise=.2,target_noise_clip=.5,action_noise_sigma=.1)),
        tqc_policy_settings=dict(n_critics=2,n_quantiles=25),
        validation_interval_steps=steps if stage=="smoke" else 16384,
        checkpoint_selection="Lexicographic validation collision-free success, then lower all-eligible final score, then lower all-eligible path; no test-based tuning",
        split_counts={s:dict(scenes=len(rr),capture_groups=len(groups[s]),sessions=sorted({r['session'] for r in rr})) for s,rr in splits.items()},
        evaluation_scope="all636 test scenes x40 starts" if full_test else "one source-order scene per held-out capture group x2 starts; smoke uses4 scenes",
        hardware=dict(platform=platform.platform(),logical_cpus=os.cpu_count(),device=os.environ.get("NC_TRAIN_DEVICE","cuda:0"),evaluation_device="cpu",gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,cuda=torch.version.cuda,torch_threads_per_job=1),
        versions=dict(python=platform.python_version(),torch=torch.__version__,stable_baselines3=sb3.__version__,sb3_contrib=sb3_contrib.__version__,numpy=np.__version__),
        limitations=["Fresh training from random initialization; historical checkpoints are not part of this ranking.",
            "Equal environment interactions are not equal FLOPs, wall time, update counts, or guaranteed convergence.",
            "One held-out acquisition session. Frames and starts are correlated; not independent patients or clinical trials.",
            "Collision checks use original planar geometry and sampled joint interpolation; not continuous safety, human collision validation, or a new CBF.",
            "Lower path length alone may reflect early failures; inspect success, collisions, and common-success paired paths together.",
            "No test-driven selection of winning seeds, scenarios, or final checkpoints. All failed jobs and results are retained."],
        source_sha256=source_hashes(),input_sha256={name:sha256(out/name) for name in ["train_rows.json","validation_rows.json","test_rows.json","validation_cases.json","test_cases.json"]})
    probe = HistoricalLearningEnv([load_scene(splits['train'][0])],state_mode='joint12')
    p['environment_parameters'] = probe.configuration
    p['backend_sha256'] = {str(path.relative_to(ROOT)):sha256(path) for path in
        [ROOT/'native/original_backend.so',ROOT/'outputs/historical_reproduction_20261002/historical_backend_05405.so',
         ROOT/'outputs/historical_reproduction_20261002/historical_backend_manifest.json']}
    probe.close()
    # Retain reviewable source versions even if future work changes the workspace.
    import shutil
    snapshot=out/'source_snapshot'/'ncsim';snapshot.mkdir(parents=True)
    for name, expected in p['source_sha256'].items():
        shutil.copy2(ROOT/'ncsim'/name,snapshot/name)
        if sha256(snapshot/name)!=expected:
            raise RuntimeError('Source changed while the study was being frozen')
    for path in [ROOT/'run_learning_study.py',ROOT/'config.json',ROOT/'requirements.txt']:
        shutil.copy2(path,out/'source_snapshot'/path.name)
    dump(out / "protocol.json", p)
    dump(out / "status.json", dict(status="prepared",created_utc=utc(),jobs=[]))
    write_dashboard(out)
    return p


def make_model(method, env, seed, p):
    klass = METHODS[method][0]
    kw = dict(features_extractor_class=LearningPointNet,
              features_extractor_kwargs=dict(ablation_mode="fusion"), net_arch=[64,64], activation_fn=nn.Tanh)
    if method == "tqc":
        kw.update(p["tqc_policy_settings"])
    common = dict(policy="MultiInputPolicy",env=env,seed=seed,device=p["hardware"]["device"],verbose=1,policy_kwargs=kw)
    settings = dict(p['optimizer_settings']['ppo' if method=='ppo_legacy' else method])
    if method in ('ppo','ppo_legacy'):
        return klass(**common, **settings)
    if method == 'td3':
        sigma = settings.pop('action_noise_sigma')
        settings['action_noise'] = NormalActionNoise(np.zeros(2), sigma*np.ones(2))
    return klass(**common, **settings)


def optimizer_steps(model):
    optimizers={}
    if hasattr(model.policy,'optimizer'):
        optimizers['policy']=model.policy.optimizer
    for name in ('actor','critic'):
        module=getattr(model,name,None)
        if module is not None and hasattr(module,'optimizer'):
            optimizers[name]=module.optimizer
    if getattr(model,'ent_coef_optimizer',None) is not None:
        optimizers['entropy_temperature']=model.ent_coef_optimizer
    return {name:max([int(state.get('step',0)) for state in opt.state.values()],default=0)
            for name,opt in optimizers.items()}


def selection_from_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    eligible = [r for r in rows if float(r['initial_score']) >= 20]
    def truth(value): return str(value).lower() in ('true','1','1.0')
    if not eligible:
        return dict(success=0., score=1e10, path=1e10)
    successes = [truth(r['strict_success']) for r in eligible]
    return dict(success=float(np.mean(successes)),score=float(np.mean([float(r['final_score']) for r in eligible])),
                path=float(np.mean([float(r['path_length_m']) for r in eligible])))


class StudyCallback(BaseCallback):
    def __init__(self, out, job, method, seed, p):
        super().__init__()
        self.out,self.job,self.method,self.seed,self.p=out,job,method,seed,p
        self.start=time.perf_counter();self.last_progress=0;self.last_val=0;self.best_rank=None;self.history=[]
        self.episodes=0;self.collisions=0

    def _on_step(self):
        self.episodes += int(sum(self.locals.get('dones',[])))
        self.collisions += sum(bool(i.get('self_collision') or i.get('swept_self_collision')) for i in self.locals.get('infos',[]) if i.get('episode'))
        if self.num_timesteps-self.last_progress >= 128:
            self.last_progress=self.num_timesteps
            dump(self.job/'progress.json',dict(status='training',updated_utc=utc(),method=self.method,seed=self.seed,
                steps=self.num_timesteps,budget=self.p['steps_per_method_seed'],elapsed_seconds=time.perf_counter()-self.start,
                episodes=self.episodes,training_any_sampled_collision_episodes=self.collisions,gradient_updates=getattr(self.model,'_n_updates',None)))
        # Evaluating inside rollout collection complicates exact state/RNG matching.
        # Checkpoints at rollout ends or off-policy collect boundaries are handled below.
        return True

    def _on_rollout_end(self):
        if self.num_timesteps-self.last_val >= self.p['validation_interval_steps'] and self.num_timesteps < self.p['steps_per_method_seed']:
            self.validate()

    def validate(self):
        from .learning_evaluation import evaluate_checkpoint
        step=self.num_timesteps
        self.last_val=step
        model_path=self.job/f'checkpoint_{step:07d}.zip'
        self.model.save(model_path)
        # Evaluation uses an independent env and restored RNG state. No test data here.
        py_random=__import__('random');py_state=py_random.getstate();np_state=np.random.get_state();th_state=torch.get_rng_state();cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            validation=self.job/f'validation_{step:07d}'
            summary=evaluate_checkpoint(model_path,'ppo' if self.method=='ppo_legacy' else self.method,
                                read(self.out/'validation_cases.json'),validation,trace_limit=0)
            if not summary['complete']:
                raise RuntimeError('Incomplete validation cannot select a checkpoint')
            selection=selection_from_csv(validation/'episodes.csv')
            rank=(-selection['success'],selection['score'],selection['path'])
            record=dict(step=step,checkpoint=model_path.name,**selection)
            self.history.append(record)
            if self.best_rank is None or rank < self.best_rank:
                self.best_rank=rank
                # Copy verified local file; does not alter the training instance.
                import shutil
                shutil.copy2(model_path,self.job/'best_model.zip')
                dump(self.job/'best_validation.json',record)
            dump(self.job/'validation_history.json',self.history)
        finally:
            py_random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(th_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


def worker(out, method, seed):
    from .learning_env import HistoricalLearningEnv
    from .learning_evaluation import evaluate_checkpoint
    out=Path(out).resolve();p=check_protocol(out)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    job=out/f'{method}_seed_{seed}'
    job.mkdir(exist_ok=False)
    started=time.perf_counter()
    try:
        dump(job/'progress.json',dict(status='loading_training_data',method=method,seed=seed,pid=os.getpid(),updated_utc=utc()))
        scenes=[load_scene(r) for r in read(out/'train_rows.json')]
        env=Monitor(HistoricalLearningEnv(scenes,reward_mode=METHODS[method][2],seed=seed,safety_filter=False,state_mode='joint12'),filename=str(job/'monitor'))
        model=make_model(method,env,seed,p)
        if str(model.device) != p['hardware']['device']:
            raise RuntimeError('Unexpected training device')
        torch.cuda.synchronize() if model.device.type == 'cuda' else None
        model.set_logger(configure_logger(str(job/'learning_curves'),['stdout','csv']))
        callback=StudyCallback(out,job,method,seed,p)
        model.learn(total_timesteps=p['steps_per_method_seed'],callback=callback)
        torch.cuda.synchronize() if model.device.type == 'cuda' else None
        training_end=time.perf_counter()
        model.save(job/'final_model.zip')
        callback.validate()
        training_manifest=dict(method=method,label=METHODS[method][1],algorithm=type(model).__name__,seed=seed,
            actual_env_steps=model.num_timesteps,requested_env_steps=p['steps_per_method_seed'],
            training_device=str(model.device),parameter_device=str(next(model.policy.parameters()).device),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if model.device.type == 'cuda' else 0,
            optimizer_updates=model._n_updates,optimizer_update_units='PPO completed epochs; SAC/TD3/TQC critic update iterations',
            optimizer_adam_steps=optimizer_steps(model),policy_parameters=sum(x.numel() for x in model.policy.parameters()),
            learning_wall_seconds=training_end-started,validation_included_in_wall_time=True,
            final_model_sha256=sha256(job/'final_model.zip'),selected_model_sha256=sha256(job/'best_model.zip'),
            selection=read(job/'best_validation.json'),protocol_sha256=sha256(out/'protocol.json'),checkpoint_kind='validation_selected')
        dump(job/'training_manifest.json',training_manifest)
        env.close();del callback,model,env,scenes
        import gc;gc.collect()
        for shield in (False,True):
            mode='shared_filter' if shield else 'raw'
            def progress(*args,**kwargs):
                dump(job/'progress.json',dict(status=f'evaluating_test_{mode}',method=method,seed=seed,steps=p['steps_per_method_seed'],
                    budget=p['steps_per_method_seed'],updated_utc=utc(),elapsed_seconds=time.perf_counter()-started))
            progress()
            evaluate_checkpoint(job/'best_model.zip','ppo' if method=='ppo_legacy' else method,
                read(out/'test_cases.json'),job/f'test_{mode}',safety_filter=shield,progress_callback=progress,trace_limit=2)
        dump(job/'progress.json',dict(status='completed',method=method,seed=seed,steps=p['steps_per_method_seed'],budget=p['steps_per_method_seed'],updated_utc=utc(),elapsed_seconds=time.perf_counter()-started))
    except Exception:
        dump(job/'progress.json',dict(status='failed',method=method,seed=seed,updated_utc=utc(),error=traceback.format_exc()))
        raise


def launch(out, parallel=2):
    if not 1 <= parallel <= 4:
        raise ValueError('parallel must be between 1 and 4 for this memory-bounded CPU study')
    out=Path(out).resolve();p=check_protocol(out)
    jobs=[(m,s) for s in p['seeds'] for m in p['methods']]
    pending=[]
    for m,s in jobs:
        folder=out/f'{m}_seed_{s}'
        if folder.exists():
            status=read(folder/'progress.json').get('status')
            if status!='completed':
                raise RuntimeError(f'An existing incomplete attempt must be inspected, not silently restarted: {folder}')
        else: pending.append((m,s))
    running={};failures=[]
    dump(out/'runner.json',dict(pid=os.getpid(),started_utc=utc(),parallel=parallel))
    try:
        while pending or running:
            while pending and len(running)<parallel:
                m,s=pending.pop(0)
                log=(out/f'{m}_seed_{s}.log').open('w',encoding='utf-8')
                proc=subprocess.Popen([sys.executable,'-X','utf8','-X','faulthandler','-u','-m','ncsim.learning_benchmark','worker','--out',str(out),'--method',m,'--seed',str(s)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                running[(m,s)]=(proc,log)
            for key,(proc,log) in list(running.items()):
                code=proc.poll()
                if code is not None:
                    log.close();del running[key]
                    if code: failures.append(dict(method=key[0],seed=key[1],exit_code=code))
            statuses=[]
            for m,s in jobs:
                status_path=out/f'{m}_seed_{s}'/'progress.json'
                st=read(status_path) if status_path.exists() else dict(status='queued',method=m,seed=s)
                st['log']=f'{m}_seed_{s}.log'
                statuses.append(st)
            dump(out/'status.json',dict(status='running' if pending or running else ('completed_with_errors' if failures else 'completed'),
                 updated_utc=utc(),stage=p['stage'],jobs=statuses,failures=failures))
            if pending or running: time.sleep(5)
        try:
            aggregate(out)
        except Exception:
            dump(out/'status.json',dict(status='aggregation_failed',updated_utc=utc(),stage=p['stage'],jobs=statuses,error=traceback.format_exc()))
            raise
    finally:
        # A user interrupt stops only this study's own workers, never unrelated Python.
        for proc,log in running.values():
            proc.terminate();log.close()
    if failures: raise RuntimeError(f'{len(failures)} failed jobs; see retained logs')


def complete_evaluation(folder, model_path, cases, algorithm, safety_filter):
    """Accept only a complete, checksum-verified journal with the exact frozen cases."""
    from .learning_evaluation import _digest, _source_checksums
    folder=Path(folder)
    required=['summary.json','progress.json','evaluation_protocol.json','episodes.jsonl']
    if not all((folder/name).exists() for name in required):
        return None
    summary,progress,spec=(read(folder/name) for name in required[:3])
    if not summary.get('complete') or progress.get('status')!='complete':
        return None
    if (spec['checkpoint_sha256']!=sha256(model_path) or spec['cases_sha256']!=_digest(cases)
            or spec['algorithm']!=algorithm.upper() or spec['safety_filter']!=safety_filter
            or spec['source_sha256']!=_source_checksums()):
        raise ValueError(f'Evaluation identity mismatch: {folder}')
    digest=_digest(spec);records=[]
    for line in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines():
        item=json.loads(line);record=item['episode']
        if item['protocol_sha256']!=digest or item['episode_sha256']!=_digest(record):
            raise ValueError(f'Evaluation journal checksum mismatch: {folder}')
        records.append(record)
    expected={c['case_id']:c for c in cases}
    if (len(records)!=len(expected) or len({r['case_id'] for r in records})!=len(expected)
            or {r['case_id'] for r in records}!=set(expected)
            or summary['candidates']!=len(expected) or progress['completed_cases']!=len(expected)):
        raise ValueError(f'Complete evaluation has missing or duplicate cases: {folder}')
    for r in records:
        c=expected[r['case_id']]
        if (r['initial_score']!=c['initial_score'] or r['eligible']!=c['eligible']
                or r['start_seed']!=c['start']['seed'] or r['initial_x_m']!=c['start']['xy'][0]
                or r['initial_y_m']!=c['start']['xy'][1] or r['safety_filter']!=safety_filter):
            raise ValueError(f'Paired evaluation start mismatch: {folder}')
    return records


def aggregate(out):
    out=Path(out);p=check_protocol(out);all_rows=[];paired=[];incomplete=[]
    cases=read(out/'test_cases.json')
    for m in p['methods']:
        for seed in p['seeds']:
            for mode in ('raw','shared_filter'):
                job=out/f'{m}_seed_{seed}'
                model=job/'best_model.zip'
                records=complete_evaluation(job/f'test_{mode}',model,cases,
                    'ppo' if m=='ppo_legacy' else m,mode=='shared_filter') if model.exists() else None
                if records is None:
                    incomplete.append(dict(method=m,seed=seed,mode=mode))
                    continue
                for r in records: r.update(method=m,training_seed=seed,mode=mode)
                all_rows.extend(records)
    if not all_rows:
        dump(out/'comparison.json',dict(stage=p['stage'],claim=p['claim'],table=[],paired_comparisons=[],incomplete=incomplete))
        return
    keys=list(dict.fromkeys(k for r in all_rows for k in r))
    with (out/'all_test_episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(all_rows)
    table=[]
    truth=lambda v:str(v).lower() in ('true','1','1.0')
    for m in p['methods']:
        for mode in ('raw','shared_filter'):
            per_seed=[]
            for seed in p['seeds']:
                rr=[r for r in all_rows if r['method']==m and r['training_seed']==seed and r['mode']==mode and float(r['initial_score'])>=20]
                if not rr:continue
                good=lambda r:truth(r['strict_success'])
                per_seed.append(dict(seed=seed,n=len(rr),success_pct=100*np.mean([good(r) for r in rr]),
                    self_collision_episodes=sum(truth(r['self_collision']) for r in rr),sampled_path_collision_episodes=sum(truth(r.get('sampled_path_self_collision','false')) for r in rr),
                    mean_final_score=np.mean([float(r['final_score']) for r in rr]),mean_path_m=np.mean([float(r['path_length_m']) for r in rr]),
                    successful_path_m=float(np.mean([float(r['path_length_m']) for r in rr if good(r)])) if any(good(r) for r in rr) else None))
            if per_seed:
                table.append(dict(method=m,label=METHODS[m][1],mode=mode,seeds_completed=len(per_seed),per_seed=per_seed,
                    mean_success_pct=float(np.mean([s['success_pct'] for s in per_seed])),
                    mean_final_score=float(np.mean([s['mean_final_score'] for s in per_seed])),
                    mean_path_m=float(np.mean([s['mean_path_m'] for s in per_seed])),
                    total_self_collision_episodes=int(sum(s['self_collision_episodes'] for s in per_seed))))
    for mode in ('raw','shared_filter'):
        for baseline in [m for m in p['methods'] if m!='ppo']:
            for seed in p['seeds']:
                def indexed(method):
                    return {r['case_id']:r for r in all_rows if r['method']==method and r['mode']==mode and r['training_seed']==seed and float(r['initial_score'])>=20}
                a,b=indexed('ppo'),indexed(baseline)
                if a and b and a.keys()!=b.keys():
                    raise ValueError('Paired algorithm evaluations must have identical eligible case sets')
                common=sorted(a) if a and b else []
                good=lambda r:truth(r['strict_success'])
                success=[k for k in common if good(a[k]) and good(b[k])]
                if common:
                    paired.append(dict(mode=mode,baseline=baseline,seed=seed,paired_cases=len(common),common_success_cases=len(success),
                        ppo_minus_baseline_success_pp=100*float(np.mean([good(a[k])-good(b[k]) for k in common])),
                        ppo_minus_baseline_path_on_common_success_m=float(np.mean([float(a[k]['path_length_m'])-float(b[k]['path_length_m']) for k in success])) if success else None,
                        ppo_minus_baseline_final_score_all=float(np.mean([float(a[k]['final_score'])-float(b[k]['final_score']) for k in common]))))
    dump(out/'comparison.json',dict(stage=p['stage'],claim=p['claim'],updated_utc=utc(),table=table,paired_comparisons=paired,incomplete=incomplete,
         warning='Smoke results validate execution only. Pilot results are budget-specific, not a SOTA or clinical safety claim. Lower path must be read together with success and collisions.'))


def write_dashboard(out):
    html='''<!doctype html><meta charset="utf-8"><title>多模态算法：训练与对比</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#17304b;margin:32px auto;max-width:1100px}h1{font-size:26px}.card{background:white;padding:22px;margin:18px 0;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;border-bottom:1px solid #e3e9ed;text-align:left}small{color:#667}a{color:#246cac}progress{width:140px}code{font-size:12px}#notice{color:#944516}</style>
<h1>多模态算法比较 · 新增 MM-TQC</h1><div class="card"><b id="stage"></b><p>原始 0.5405 m 运动学；按采集会话划分训练／验证／测试；相同多模态输入和环境交互预算。验证集选模型，测试集单独评估。</p><p id="notice">新结果尚不能预先称为 SOTA。小规模运行只验证程序，正式结论需多种子学习曲线与独立测试。</p><a href="protocol.json">冻结协议</a> · <a href="comparison.json">比较结果</a> · <a href="all_test_episodes.csv">逐次测试数据</a></div>
<div class="card"><table><thead><tr><th>算法／种子</th><th>状态</th><th>环境交互</th><th>耗时</th><th>日志</th></tr></thead><tbody id="jobs"></tbody></table><small id="updated"></small></div>
<div class="card"><h2>已完成测试</h2><p>原始策略与共享碰撞过滤分别列出。路径为全部初始遮挡试次的均值；共同成功起点的配对路径见比较结果文件。</p><table><thead><tr><th>算法</th><th>模式</th><th>已完成种子</th><th>无碰撞成功率</th><th>最终遮挡</th><th>路径 m</th><th>终点自碰撞数</th></tr></thead><tbody id="results"></tbody></table></div>
<script>const esc=x=>String(x??'').replace(/[&<>"']/g,s=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));async function refresh(){try{let p=await(await fetch('protocol.json',{cache:'no-store'})).json();stage.textContent=p.stage==='smoke'?'当前：程序验证（不可作性能结论）':'当前：固定交互预算试验（不等于收敛或 SOTA）';let s=await(await fetch('status.json',{cache:'no-store'})).json();jobs.innerHTML=s.jobs.map(j=>`<tr><td>${esc(j.method)} / ${j.seed}</td><td>${esc(j.status)}</td><td><progress max="${j.budget||1}" value="${j.steps||0}"></progress> ${j.steps||0}/${j.budget||'—'}</td><td>${j.elapsed_seconds?(j.elapsed_seconds/60).toFixed(1)+' min':'—'}</td><td><a href="${esc(j.log)}">日志</a></td></tr>`).join('');updated.textContent='更新：'+s.updated_utc;let r=await fetch('comparison.json',{cache:'no-store'});if(r.ok){let c=await r.json();results.innerHTML=c.table.map(v=>`<tr><td>${esc(v.label)}</td><td>${esc(v.mode)}</td><td>${v.seeds_completed}</td><td>${v.mean_success_pct.toFixed(2)}%</td><td>${v.mean_final_score.toFixed(2)}</td><td>${v.mean_path_m.toFixed(3)}</td><td>${v.total_self_collision_episodes}</td></tr>`).join('')}}catch(e){updated.textContent=e.message}}refresh();setInterval(refresh,10000);</script>'''
    (out/'实验进度.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--out',required=True);prep.add_argument('--stage',choices=['smoke','pilot'],default='pilot');prep.add_argument('--steps',type=int,default=65536);prep.add_argument('--seeds',default='29,47,83');prep.add_argument('--methods',default='ppo,sac,td3,tqc,ppo_legacy');prep.add_argument('--full-test',action='store_true')
    run=sub.add_parser('run');run.add_argument('--out',required=True);run.add_argument('--parallel',type=int,default=2)
    work=sub.add_parser('worker');work.add_argument('--out',required=True);work.add_argument('--method',choices=METHODS,required=True);work.add_argument('--seed',type=int,required=True)
    agg=sub.add_parser('aggregate');agg.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        print(json.dumps(prepare(args.out,args.stage,args.steps,list(map(int,args.seeds.split(','))),args.methods.split(','),args.full_test),ensure_ascii=False,indent=2))
    elif args.command=='run':launch(args.out,args.parallel)
    elif args.command=='worker':worker(args.out,args.method,args.seed)
    else:aggregate(args.out)


if __name__=='__main__':main()
