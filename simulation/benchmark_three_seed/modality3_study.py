"""Reproduce the three reported input configurations with seeds 29, 47 and 83.
The current reported analysis is specified in results/analysis_protocol.json.
`prepare` creates a fresh protocol for a new reproduction.
"""
import argparse, csv, gc, hashlib, json, os, platform, random, shutil, subprocess, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure as logger_configure
from ncsim.data import ROOT, load_scene, sha256
from ncsim.historical_backend import HistoricalEnv, configure, inverse
from ncsim.geometry import INITIAL_Q, self_collision
from ncsim.policies import AblationAttentionPointNet

NAME = os.environ.get('NC_STUDY', 'modality3_reproduction')
MODES = {'mmppo': 'fusion', 'pc': 'only_pc', 'state': 'only_state'}
LABELS = {'mmppo': 'MM-PPO (point clouds + task state)', 'pc': 'Point clouds only', 'state': 'Task state only'}
SEEDS = [29, 47, 83]
POINTDATA = Path(os.environ.get('NC_POINTDATA', Path(__file__).resolve().parents[2] / 'data' / 'pointdata'))
HISTORY = Path(os.environ.get('NC_HISTORY', Path(__file__).resolve().parents[1] / 'outputs' / 'historical_reproduction_20261002'))
THREADS = int(os.environ.get('NC_THREADS', '4'))


def utc(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8'); os.replace(tmp, p)
def digest(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def rng_save(): return random.getstate(), np.random.get_state(), torch.get_rng_state()
def rng_restore(v): random.setstate(v[0]); np.random.set_state(v[1]); torch.set_rng_state(v[2])


def rel(path):
    """path relative to the pointdata folder (machine independent)"""
    parts = str(path).replace('\\', '/').split('/'); return '/'.join(parts[parts.index('pointdata') + 1:])


def resolve(row):
    r = dict(row); r['source'] = str(POINTDATA / row['source_rel']); r['target_source'] = str(POINTDATA / row['target_rel']); return r


class OriginalTrainingEnv(gym.Env):
    """Gymnasium adapter: delegate every non-clear transition to HistoricalEnv (as in original6_study.py)."""
    def __init__(self, scenes):
        super().__init__(); self.scenes = scenes; self.core = HistoricalEnv([scenes[0]], max_steps=40)
        self.observation_space = self.core.observation_space; self.action_space = self.core.action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); options = options or {}
        i = options['scene_index'] if 'scene_index' in options else int(self.np_random.integers(len(self.scenes)))
        self.core.scenes = [self.scenes[i]]
        for _ in range(10000):
            xy = self.np_random.uniform([-.2, .91], [.5, 1.71]); q, ok = inverse(xy, INITIAL_Q)
            if ok: break
        else: raise RuntimeError('No reachable reset')
        start = dict(xy=xy.tolist(), q=q.tolist(), seed=int(self.np_random.integers(1, 2 ** 31 - 1)))
        self.obs = self.core.reset_start(start)
        return self.obs, self.core._info()

    def step(self, action):
        if self.core.finished:
            if self.core.reason != 'initially_clear': raise RuntimeError('Reset a finished episode')
            return self.obs, 0., True, False, self.core._info()
        self.obs, r, ok = self.core.step(action); info = self.core._info(); info['ik_ok'] = ok
        return self.obs, r, self.core.finished, False, info


def make_env(scenes, seed):
    env = DummyVecEnv([lambda: Monitor(OriginalTrainingEnv(scenes))])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10., gamma=.99, epsilon=1e-8); env.seed(seed); return env


def make_model(method, env, seed, p):
    kw = dict(features_extractor_class=AblationAttentionPointNet, features_extractor_kwargs=dict(ablation_mode=MODES[method]),
              net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh)
    return PPO('MultiInputPolicy', env, seed=seed, device=os.environ.get('NC_DEVICE', 'cpu'), verbose=0, policy_kwargs=kw, **p['ppo_settings'])


def make_case(row, start):
    core = HistoricalEnv([load_scene(resolve(row))], max_steps=40); core.reset_start(start)
    return dict(case_id=start['case_id'], scene_id=row['scene_id'], scene_index=start['scene_index'], capture_group=row['capture_group'],
                row=row, start=start, initial_score=core.score, eligible=bool(core.score >= 20), initial_self_collision=self_collision(core.q))


def prepare(out, smoke=False):
    out.mkdir(parents=True, exist_ok=False); configure(HISTORY)
    src = read(ROOT / 'inputs/ordered_dataset.json'); rows = []
    for r in src:
        n = {k: r[k] for k in ('scene_id', 'session', 'capture_group', 'source_sha256', 'target_sha256', 'target')}
        n['source_rel'], n['target_rel'] = rel(r['source']), rel(r['target_source'])
        rows.append(n)
    test = rows[1592:]; tg = {r['capture_group'] for r in test}
    validation = [r for r in rows[:1592] if r['session'] == 'TR0821train_filtered' and r['capture_group'] not in tg]
    vg = {r['capture_group'] for r in validation}; vh = {r['source_sha256'] for r in validation}; th = {r['source_sha256'] for r in test}
    training = [r for r in rows[:1592] if r['capture_group'] not in tg | vg and r['source_sha256'] not in th | vh]
    assert not ({r['capture_group'] for r in training} & (tg | vg)) and not (tg & vg)
    assert len(rows) == 1990 and len(test) == 398 and validation and training
    for n in training + validation + test:                                # the split uses metadata only; verify every file that is used
        rr = resolve(n); assert sha256(rr['source']) == n['source_sha256'] and sha256(rr['target_source']) == n['target_sha256'], n['scene_id']
    valcases = []
    for i, g in enumerate(sorted(vg)):
        rr = [r for r in validation if r['capture_group'] == g]; row = rr[len(rr) // 2]
        for j in range(2):
            seed = 300000 + i * 2 + j; rng = random.Random(seed)
            for _ in range(10000):
                xy = [rng.uniform(-.2, .5), rng.uniform(.91, 1.71)]; q, ok = inverse(xy, INITIAL_Q)
                if ok: break
            else: raise RuntimeError('Validation start unreachable')
            valcases.append(make_case(row, dict(seed=seed, xy=xy, q=q.tolist(), scene_index=i, start_index=j, case_id=f'val_{i:03d}_{j}')))
    testcases = []
    for start in read(ROOT / 'inputs/starts.json'):
        row = test[start['scene_index']]; assert row['scene_id'] == start['scene_id']; testcases.append(make_case(row, start))
    assert len(testcases) == 15920 and sum(c['eligible'] for c in testcases) == 11647
    if smoke: training = training[:3]; valcases = [c for c in valcases if c['eligible']][:4]; testcases = testcases[:12]
    for name, value in [('train_rows.json', training), ('validation_rows.json', validation), ('test_rows.json', test),
                        ('validation_cases.json', valcases), ('test_cases.json', testcases)]: dump(out / name, value)
    p = dict(
        created_utc=utc(), stage='software_smoke' if smoke else 'modality3_reproduction', methods=list(MODES), labels=LABELS, ablation_modes=MODES,
        seeds=[29] if smoke else SEEDS, steps_per_method_seed=4096 if smoke else 1273856, validation_interval_steps=2048 if smoke else 131072,
        ppo_settings=dict(learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10, gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=0., vf_coef=.5, max_grad_norm=.5),
        reward_normalization=dict(norm_obs=False, norm_reward=True, clip_reward=10., gamma=.99, epsilon=1e-8, fresh_statistics_per_seed=True),
        network='AblationAttentionPointNet (manuscript Methods): point branches 3-64-128-128 with attention pooling, state 6-64-64; actor/critic 2x64 Tanh',
        environment='HistoricalEnv (manuscript benchmark): original 0.5405 m kinematics, original reward, action clip [-1,1] and norm cap 0.08 m, threshold 20, 40 decisions, original self-collision',
        data_counts=dict(train=len(training), validation=len(validation), test=398, validation_cases=len(valcases), test_cases=len(testcases)),
        split='as original6_study.py: TR0821 sessions of the first 1,592 files reserved for validation; every capture group and file hash of the 398 test files removed from training',
        selection='validation only (every 131,072 steps and at the end): scene-macro collision-free recovery, then mean final score, then mean path; first checkpoint wins ties',
        analysis_plan=dict(
            primary='scene-macro collision-free recovery (V<20 within 40 decisions) on the 11,647 initially occluded starts of the fixed 398 x 40 test starts, validation-selected checkpoint',
            secondary=['scene-macro recovery including collisions', 'mean final score', 'mean path of successful episodes', 'self-collision episodes',
                       'recovery with the image occlusion rate O <= 20 % at the final state (occlusion_calibration.py definition, computed after training)',
                       'the same metrics for the final checkpoint'],
            summary='mean and sample s.d. over the three seeds and every per-seed value; MM-PPO minus each single-input configuration paired by seed (mean difference, 95 % t-interval over seeds, and per-seed differences with a bootstrap over the 47 acquisition folders)',
            archived='the archived checkpoints of the manuscript remain a separate historical reference and are not averaged with the new seeds'),
        threads_per_worker=THREADS, device=os.environ.get('NC_DEVICE', 'cpu'),
        execution_plan=dict(reproduction=dict(seeds=SEEDS, configurations='all three', device=os.environ.get('NC_DEVICE', 'cpu'))),
        limitations=['Controlled retraining of the original method, not a replay of the unrecorded historical training run.',
                     'Recorded static clouds; capture-folder separation does not establish patient-level independence.',
                     'The task state contains the occlusion score and guidance vector, so the state-only policy is not image-free.'],
        source_sha256={Path(f).relative_to(ROOT).as_posix(): sha256(f) for f in [ROOT / 'modality3_study.py', *sorted((ROOT / 'ncsim').glob('*.py'))]},
        input_sha256={n: sha256(out / n) for n in ('train_rows.json', 'validation_rows.json', 'test_rows.json', 'validation_cases.json', 'test_cases.json')},
        split_digest=digest(dict(train=[r['source_sha256'] for r in training], validation=[r['source_sha256'] for r in validation], test=[r['source_sha256'] for r in test])))
    if os.environ.get('NC_AMENDMENT'):                                   # protocol revision decided before any result was inspected
        am = read(os.environ['NC_AMENDMENT']); p['amendment'] = am; p['execution_plan'] = am.get('execution_plan', p['execution_plan'])
    dump(out / 'protocol.json', p); dump(out / 'protocol_digest.json', dict(protocol_sha256=sha256(out / 'protocol.json'), frozen_utc=utc()))
    print(json.dumps(dict(prepared=True, out=str(out), counts=p['data_counts'], seeds=p['seeds'], protocol_sha256=sha256(out / 'protocol.json'))), flush=True)


def verify(out):
    p = read(out / 'protocol.json')
    for n, h in p['input_sha256'].items(): assert sha256(out / n) == h, n
    for f, h in p['source_sha256'].items(): assert sha256(ROOT / f.replace(chr(92), '/')) == h, f'source changed: {f}'   # chr(92): backslash of Windows keys
    return p


def metrics(records):
    el = [r for r in records if r['eligible']]
    if not el: return dict(n=0)
    groups = sorted({r['scene_id'] for r in el}); good = [r for r in el if r['strict_success']]
    macro = lambda key: 100 * float(np.mean([np.mean([r[key] for r in el if r['scene_id'] == g]) for g in groups]))
    return dict(n=len(el), macro_strict_success_pct=macro('strict_success'), macro_success_pct=macro('success'),
                pooled_strict_success_pct=100 * float(np.mean([r['strict_success'] for r in el])),
                mean_final_score=float(np.mean([r['final_score'] for r in el])), mean_path_m=float(np.mean([r['path_length_m'] for r in el])),
                successful_path_m=float(np.mean([r['path_length_m'] for r in good])) if good else None, self_collision_episodes=int(sum(r['self_collision'] for r in el)))


def evaluate(model, cases, folder, progress=None):
    folder.mkdir(parents=True, exist_ok=False); records = []; scene = None; core = None; beg = time.perf_counter()
    with (folder / 'episodes.jsonl').open('w', encoding='utf-8') as f:
        for i, c in enumerate(cases):
            if scene is None or scene.scene_id != c['scene_id']:
                scene = load_scene(resolve(c['row'])); core = HistoricalEnv([scene], max_steps=40)
            obs = core.reset_start(c['start']); assert abs(core.score - c['initial_score']) <= 1e-6 * max(1., c['initial_score']), c['case_id']   # cross-platform native backends
            while not core.finished:
                obs, reward, ok = core.step(model.predict(obs, deterministic=True)[0])
            r = dict(case_id=c['case_id'], scene_id=c['scene_id'], capture_group=c['capture_group'], scene_index=c['scene_index'],
                     start_seed=c['start']['seed'], initial_xy=c['start']['xy'], initial_score=core.initial_score, eligible=c['eligible'],
                     final_score=core.score, success=bool(c['eligible'] and core.score < 20), strict_success=bool(c['eligible'] and core.score < 20 and not core.collided),
                     self_collision=core.collided, path_length_m=core.path_length, decision_steps=core.steps, reason=core.reason,
                     final_xy=[float(v) for v in core.xy], final_q=[float(v) for v in core.q])
            records.append(r); f.write(json.dumps(dict(record=r, sha256=digest(r)), allow_nan=False) + '\n')
            if (i + 1) % 200 == 0:
                f.flush()
                if progress: progress(i + 1, len(cases))
    summary = dict(complete=True, candidates=len(records), case_sha256=digest(cases), elapsed_seconds=time.perf_counter() - beg, **metrics(records))
    dump(folder / 'summary.json', summary); return summary


class Callback(BaseCallback):
    def __init__(self, out, job, method, seed, p):
        super().__init__(); self.out, self.job, self.method, self.seed, self.p = out, job, method, seed, p
        self.start = time.perf_counter(); self.last_status = 0; self.last_val = 0; self.best = None; self.history = []
    def report(self, status, **kw):
        dump(self.job / 'progress.json', dict(status=status, method=self.method, seed=self.seed, updated_utc=utc(), steps=self.num_timesteps,
                                             budget=self.p['steps_per_method_seed'], elapsed_seconds=time.perf_counter() - self.start, pid=os.getpid(), host=platform.node(), **kw))
    def _on_step(self):
        if self.num_timesteps - self.last_status >= 512: self.last_status = self.num_timesteps; self.report('training')
        return not (self.out / 'STOP').exists()
    def _on_rollout_start(self):
        if self.num_timesteps - self.last_val >= self.p['validation_interval_steps'] and self.num_timesteps < self.p['steps_per_method_seed']: self.validate()
    def validate(self):
        step = self.num_timesteps; self.last_val = step; self.report('validation')
        ck = self.job / f'checkpoint_{step:07d}.zip'; self.model.save(ck); env = self.model.get_vec_normalize_env(); env.save(self.job / f'normalization_{step:07d}.pkl')
        state = rng_save()
        try:
            m = evaluate(PPO.load(ck, device='cpu'), read(self.out / 'validation_cases.json'), self.job / f'validation_{step:07d}')
            rank = (-m['macro_strict_success_pct'], m['mean_final_score'], m['mean_path_m']); rec = dict(step=step, checkpoint=ck.name, **m); self.history.append(rec)
            if self.best is None or rank < self.best:
                self.best = rank; shutil.copy2(ck, self.job / 'best_model.zip'); dump(self.job / 'best_validation.json', rec)
            dump(self.job / 'validation_history.json', self.history)
        finally: rng_restore(state)
        self.model.save(self.job / 'latest_model.zip'); env.save(self.job / 'latest_normalization.pkl'); self.report('training')


def worker(out, method, seed):
    torch.set_num_threads(THREADS); torch.set_num_interop_threads(1); configure(HISTORY); p = verify(out)
    job = out / f'{method}_seed_{seed}'; job.mkdir(exist_ok=False); started = time.perf_counter()
    try:
        dump(job / 'progress.json', dict(status='loading', method=method, seed=seed, updated_utc=utc(), pid=os.getpid(), host=platform.node()))
        scenes = [load_scene(resolve(r)) for r in read(out / 'train_rows.json')]; env = make_env(scenes, seed); model = make_model(method, env, seed, p)
        model.set_logger(logger_configure(str(job / 'learning_curves'), ['csv']))
        cb = Callback(out, job, method, seed, p); model.learn(total_timesteps=p['steps_per_method_seed'], callback=cb)
        model.save(job / 'final_model.zip'); env.save(job / 'final_normalization.pkl')
        if model.num_timesteps != p['steps_per_method_seed']: cb.report('stopped_incomplete'); raise RuntimeError('Training stopped before the fixed budget')
        cb.validate()
        dump(job / 'training_manifest.json', dict(method=method, ablation_mode=MODES[method], seed=seed, actual_env_steps=model.num_timesteps,
            requested_env_steps=p['steps_per_method_seed'], learning_wall_seconds=time.perf_counter() - started, optimizer_updates=model._n_updates,
            policy_parameters=sum(x.numel() for x in model.policy.parameters()), selected_step=read(job / 'best_validation.json')['step'],
            selected_model_sha256=sha256(job / 'best_model.zip'), final_model_sha256=sha256(job / 'final_model.zip'), protocol_sha256=sha256(out / 'protocol.json'),
            device=str(model.device), threads=THREADS, host=platform.node(), python=sys.version.split()[0], torch=torch.__version__))
        env.close(); del model, cb, env, scenes; gc.collect()
        tc = read(out / 'test_cases.json')
        def progress(n, total): dump(job / 'progress.json', dict(status='evaluating_test', method=method, seed=seed, completed_cases=n, total_cases=total, updated_utc=utc(), elapsed_seconds=time.perf_counter() - started))
        evaluate(PPO.load(job / 'best_model.zip', device='cpu'), tc, job / 'test', progress=progress)
        evaluate(PPO.load(job / 'final_model.zip', device='cpu'), tc, job / 'test_final', progress=progress)
        dump(job / 'progress.json', dict(status='completed', method=method, seed=seed, updated_utc=utc(), elapsed_seconds=time.perf_counter() - started, host=platform.node()))
    except Exception:
        dump(job / 'progress.json', dict(status='failed', method=method, seed=seed, updated_utc=utc(), error=traceback.format_exc())); raise


def launch(out, parallel, jobs):
    p = verify(out); jobs = jobs or [(m, s) for s in p['seeds'] for m in p['methods']]
    pending = list(jobs); running = {}; failures = []
    tag = os.environ.get('NC_TAG', platform.node()); dump(out / f'runner_{tag}.json', dict(pid=os.getpid(), started_utc=utc(), parallel=parallel, jobs=jobs, threads=THREADS))
    while pending or running:
        while pending and len(running) < parallel:
            m, s = pending.pop(0); log = (out / f'{m}_seed_{s}.log').open('w')
            proc = subprocess.Popen([sys.executable, '-u', str(ROOT / 'modality3_study.py'), 'worker', '--out', str(out), '--method', m, '--seed', str(s)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            running[(m, s)] = (proc, log)
        for key, (proc, log) in list(running.items()):
            code = proc.poll()
            if code is not None:
                log.close(); del running[key]
                if code: failures.append(dict(method=key[0], seed=key[1], exit_code=code))
        dump(out / f'status_{tag}.json', dict(updated_utc=utc(), running=[f'{m}:{s}' for m, s in running], pending=[f'{m}:{s}' for m, s in pending], failures=failures))
        if pending or running: time.sleep(15)
    print(json.dumps(dict(finished=True, failures=failures)), flush=True)


def aggregate(out):
    p = verify(out); expected = {c['case_id']: c for c in read(out / 'test_cases.json')}; table = []; allrows = []
    for method in p['methods']:
        per = []
        for seed in p['seeds']:
            job = out / f'{method}_seed_{seed}'; m = read(job / 'training_manifest.json'); assert m['actual_env_steps'] == p['steps_per_method_seed']
            assert sha256(job / 'best_model.zip') == m['selected_model_sha256'] and m['protocol_sha256'] == sha256(out / 'protocol.json')
            for kind in ('test', 'test_final'):
                rr = []
                for line in (job / kind / 'episodes.jsonl').read_text().splitlines():
                    item = json.loads(line); r = item['record']; assert digest(r) == item['sha256']; c = expected[r['case_id']]
                    assert abs(r['initial_score'] - c['initial_score']) <= 1e-6 * max(1., c['initial_score']) and r['start_seed'] == c['start']['seed']; rr.append(dict(r, method=method, training_seed=seed, checkpoint=kind))
                assert len(rr) == len(expected); allrows.extend(rr)
                if kind == 'test': per.append(dict(seed=seed, selected_step=m['selected_step'], host=m['host'], **metrics(rr)))
        vals = lambda k: [x[k] for x in per]
        table.append(dict(method=method, label=LABELS[method], per_seed=per, mean_macro_strict_success_pct=float(np.mean(vals('macro_strict_success_pct'))),
                          sd_macro_strict_success_pct=float(np.std(vals('macro_strict_success_pct'), ddof=1)) if len(per) > 1 else None, mean_final_score=float(np.mean(vals('mean_final_score')))))
    with (out / 'all_test_episodes.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
    dump(out / 'comparison.json', dict(complete=True, table=table, rows=len(allrows), protocol_sha256=sha256(out / 'protocol.json'),
                                       note='Validation-selected checkpoints; final-checkpoint episodes are in all_test_episodes.csv (checkpoint=test_final). Paired and O-based analyses are computed separately.'))
    print(json.dumps([dict(method=t['method'], mean=round(t['mean_macro_strict_success_pct'], 2), sd=t['sd_macro_strict_success_pct']) for t in table]), flush=True)


def main():
    a = argparse.ArgumentParser(); a.add_argument('command', choices=['prepare', 'run', 'worker', 'aggregate'])
    a.add_argument('--out', type=Path, default=ROOT / 'outputs' / NAME); a.add_argument('--smoke', action='store_true')
    a.add_argument('--parallel', type=int, default=3); a.add_argument('--method', choices=MODES); a.add_argument('--seed', type=int)
    a.add_argument('--jobs', default='', help='comma-separated method:seed list, e.g. mmppo:29,pc:29 (default: all)')
    args = a.parse_args(); out = args.out.resolve()
    if args.command == 'prepare': prepare(out, args.smoke)
    elif args.command == 'worker': worker(out, args.method, args.seed)
    elif args.command == 'run': launch(out, args.parallel, [(j.split(':')[0], int(j.split(':')[1])) for j in args.jobs.split(',') if j])
    elif args.command == 'aggregate': aggregate(out)


if __name__ == '__main__':
    main()
