"""Main simulation benchmark recomputed with the three reported policies of each input configuration (protocol v2,
validation-selected checkpoints) on the 398 x 40 starts of the original benchmark.

Statistics are those of the original analyses: ncsim.publication_figures (47 acquisition folders, 2,000 resamples,
seed 20261002) for recovery, curves and strata. Episodes of the three seeds are pooled; because every seed has the same starts, pooled scene-level rates
equal the seed means of the per-seed rates. Seed-level values are written separately.
Output: 仿真平台/outputs/benchmark_three_seed."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIM = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(SIM))
from ncsim import publication_figures as pf

V2 = Path(__file__).resolve().parent / 'outputs/modality3_reproduction'
OUT = SIM / 'outputs/benchmark_three_seed'
NAME = {'mmppo': 'fusion', 'pc': 'pointcloud', 'state': 'robotstate'}
SEEDS = [29, 47, 83]
B_ = lambda s: s.astype(str) == 'True'


def episodes():
    A = pd.read_csv(SIM / 'outputs/manuscript_aligned_20261002/episodes.csv')
    key = A[A.method == 'fusion'][['case_id', 'scene_id', 'scene_index', 'start_index', 'initial_score', 'eligible']]
    R = pd.read_csv(V2 / 'all_test_episodes.csv'); R = R[(R.checkpoint == 'test') & R.training_seed.isin(SEEDS)]
    R = R.assign(method=R.method.map(NAME))
    d = R[['case_id', 'method', 'training_seed', 'initial_score', 'final_score', 'success', 'strict_success', 'self_collision', 'path_length_m',
           'decision_steps', 'reason']].merge(key, on='case_id', suffixes=('', '_archived'), validate='many_to_one')
    assert len(d) == 15920 * 9 and (abs(d.initial_score - d.initial_score_archived) <= 1e-6 * d.initial_score.clip(lower=1)).all()
    for k in ('success', 'strict_success', 'self_collision'): d[k] = B_(d[k])
    d['eligible'] = d.eligible.astype(bool); d['improved'] = d.final_score < d.initial_score
    d['reduction_fraction'] = np.where(d.eligible, (d.initial_score - d.final_score) / d.initial_score.where(d.initial_score > 0, 1), np.nan)
    d['termination_reason'] = d.reason; d['capture_group'] = d.scene_id.str.rsplit('/', n=1).str[0]
    groups = sorted(d.capture_group.unique()); assert len(groups) == 47
    d['group_index'] = d.capture_group.map({g: i for i, g in enumerate(groups)})
    assert ((d.final_score < 20) == d.success)[d.eligible].all()
    return d.drop(columns=['initial_score_archived', 'reason']), groups


def tint(x):
    x = np.asarray(x, float); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return [float(x.mean() - h), float(x.mean() + h)]


def seed_level(e):
    rows = []
    for (m, s), g in e.groupby(['method', 'training_seed']):
        rows.append(dict(method=m, seed=s, n=len(g), macro_success=100 * g.groupby('scene_index').success.mean().mean(),
                         macro_improved=100 * g.groupby('scene_index').improved.mean().mean(),
                         macro_strict_success=100 * g.groupby('scene_index').strict_success.mean().mean(),
                         mean_final=g.final_score.mean(), mean_reduction_pct=100 * g.reduction_fraction.mean(), mean_decisions=g.decision_steps.mean(),
                         self_collision=int((g.termination_reason == 'self_collision').sum()), step_limit=int((g.termination_reason == 'step_limit').sum()),
                         repeated_unreachable=int((g.termination_reason == 'repeated_unreachable').sum()), mean_path_all=g.path_length_m.mean(),
                         mean_path_success=g[g.success].path_length_m.mean(), improved_episodes=int(g.improved.sum())))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True); (OUT / 'source_data').mkdir(exist_ok=True)
    d, groups = episodes(); e = d[d.eligible].copy(); assert len(e) == 11647 * 9
    d.to_csv(OUT / 'episodes.csv.gz', index=False, compression='gzip')
    W = np.random.default_rng(pf.SEED).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=pf.B)
    pf.OUT = OUT; pf.style()
    for fn in (pf.fig15, pf.fig16, pf.fig17):                          # same estimators, intervals and CSV layout as the original benchmark
        plt.close(fn(e, W))
    # per-scene seed-averaged rates (Supplementary Fig. 1b)
    per = e.groupby(['method', 'scene_index']).agg(eligible=('case_id', 'nunique'), success_percent=('success', 'mean'), improved_percent=('improved', 'mean')).reset_index()
    per[['success_percent', 'improved_percent']] *= 100; per.to_csv(OUT / 'per_scene_summary.csv', index=False)
    S = seed_level(e); S.to_csv(OUT / 'seed_level.csv', index=False)
    summ = {m: {k: dict(mean=float(g[k].mean()), sd=float(g[k].std(ddof=1))) for k in S.columns if k not in ('method', 'seed', 'n')} for m, g in S.groupby('method')}
    gains = {}
    for m in ('pointcloud', 'robotstate'):
        x = S[S.method == 'fusion'].set_index('seed').macro_success - S[S.method == m].set_index('seed').macro_success
        gains[m] = dict(per_seed=x.round(3).tolist(), mean=float(x.mean()), t95=tint(x), seeds_favouring_mmppo=int((x > 0).sum()))
    # collision-free recovery (pre-specified primary end point) with the same folder bootstrap as macro_estimates
    agg = e.groupby(['method', 'scene_index', 'group_index'])[['strict_success', 'success']].mean().reset_index(); cf, bt = {}, {}
    for m in pf.METHODS:
        a_ = agg[agg.method == m]; num = a_.groupby('group_index').strict_success.sum().reindex(range(47), fill_value=0).to_numpy()
        n = a_.groupby('group_index').size().reindex(range(47), fill_value=0).to_numpy(); cf[m] = 100 * a_.strict_success.mean(); bt[m] = 100 * (W @ num) / (W @ n)
    strict = {m: dict(estimate=float(cf[m]), ci95=pf.ci(bt[m]).tolist(), seeds=S[S.method == m].macro_strict_success.tolist()) for m in pf.METHODS}
    for m in ('pointcloud', 'robotstate'):
        x = S[S.method == 'fusion'].set_index('seed').macro_strict_success - S[S.method == m].set_index('seed').macro_strict_success
        strict['fusion_minus_' + m] = dict(estimate=float(cf['fusion'] - cf[m]), ci95=pf.ci(bt['fusion'] - bt[m]).tolist(), t95=tint(x), per_seed=x.tolist(), seeds_favouring_mmppo=int((x > 0).sum()))
    pooled = {m: dict(success=float(100 * g.success.mean()), strict=float(100 * g.strict_success.mean())) for m, g in e.groupby('method')}
    res = dict(summary=summ, gains_over_single_inputs=gains, collision_free=strict, pooled_rates=pooled,
               fig15=pd.read_csv(OUT / 'source_data/Fig15_estimates_and_intervals.csv').to_dict('records'))
    (OUT / 'benchmark_retrained.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    for m in ('pointcloud', 'robotstate', 'fusion'):
        s = summ[m]; print(m, {k: f"{s[k]['mean']:.3f}±{s[k]['sd']:.3f}" for k in ('macro_success', 'mean_final', 'mean_reduction_pct', 'mean_decisions', 'self_collision', 'step_limit', 'mean_path_all')})
    print(pd.read_csv(OUT / 'source_data/Fig15_estimates_and_intervals.csv').round(2).to_string())
    print('gains', json.dumps(gains));
    print('collision_free', json.dumps(strict)); print('pooled', json.dumps(pooled))


if __name__ == '__main__':
    main()
