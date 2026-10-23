"""Analysis of the reported three-seed subset using the recorded endpoint definitions.
Writes outputs/modality3_reproduction/analysis_v2.json and prints a summary."""
import csv, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
O = ROOT / 'outputs/modality3_reproduction'
ARCH = Path(__import__('os').environ.get('NC_ARCHIVED_RESULTS', ROOT.parent / 'outputs/manuscript_aligned_20261002'))
METHODS = ['mmppo', 'pc', 'state']; ARCH_NAME = {'mmppo': 'fusion', 'pc': 'pointcloud', 'state': 'robotstate'}


def macro(d, key):
    return 100 * d.groupby('scene_id')[key].mean().mean()


def main():
    p = json.loads((O / 'protocol.json').read_text(encoding='utf-8'))
    E = pd.read_csv(O / 'all_test_episodes.csv'); E = E[E.eligible.astype(str) == 'True'].copy()
    for k in ('success', 'strict_success', 'self_collision'): E[k] = E[k].astype(str) == 'True'
    out = dict(protocol_sha256=json.loads((O / 'protocol_digest.json').read_text())['protocol_sha256'], per_seed={}, summary={}, paired={}, archived={})
    p['seeds'] = [29,47,83]
    E = E[E.training_seed.isin(p['seeds'])].copy()
    sel = E[E.checkpoint == 'test']
    for m in METHODS:
        rows = []
        for s in p['seeds']:
            d = sel[(sel.method == m) & (sel.training_seed == s)]
            good = d[d.strict_success]
            rows.append(dict(seed=s, n=len(d), recovery_cf=macro(d, 'strict_success'), recovery=macro(d, 'success'), final_score=d.final_score.mean(),
                             success_path_m=good.path_length_m.mean(), self_collisions=int(d.self_collision.sum()),
                             final_ckpt_recovery_cf=macro(E[(E.checkpoint == 'test_final') & (E.method == m) & (E.training_seed == s)], 'strict_success')))
        out['per_seed'][m] = rows; df = pd.DataFrame(rows)
        out['summary'][m] = {k: dict(mean=float(df[k].mean()), sd=float(df[k].std(ddof=1))) for k in df.columns if k not in ('seed', 'n')}
    # paired by seed: MM-PPO minus each single-input configuration
    groups = sorted(sel.capture_group.unique()); rng = np.random.default_rng(20261002)
    piv = {m: {s: sel[(sel.method == m) & (sel.training_seed == s)].groupby('scene_id').agg(cf=('strict_success', 'mean'), fs=('final_score', 'mean'), g=('capture_group', 'first'))
               for s in p['seeds']} for m in METHODS}
    scenes_by_group = sel.drop_duplicates('scene_id').groupby('capture_group').scene_id.apply(list).to_dict()
    for other in ('state', 'pc'):
        for key, lab in (('cf', 'recovery_cf_pp'), ('fs', 'final_score')):
            per = [100 ** (key == 'cf') * (piv['mmppo'][s][key].mean() - piv[other][s][key].mean()) for s in p['seeds']]
            mean = float(np.mean(per)); half = float(stats.t.ppf(.975, len(per) - 1) * np.std(per, ddof=1) / np.sqrt(len(per)))
            boot = []
            for _ in range(2000):                                                    # cluster bootstrap over acquisition folders (scenes of a drawn folder kept together)
                draw = rng.choice(groups, len(groups), replace=True); sc = [x for g in draw for x in scenes_by_group[g]]
                boot.append(np.mean([100 ** (key == 'cf') * (piv['mmppo'][s][key].loc[sc].mean() - piv[other][s][key].loc[sc].mean()) for s in p['seeds']]))
            out['paired'][f'mmppo_minus_{other}_{lab}'] = dict(per_seed=[float(x) for x in per], mean=mean, t95=[mean - half, mean + half],
                                                               bootstrap95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                                               seeds_favouring_mmppo=int(sum((x > 0) if key == 'cf' else (x < 0) for x in per)))
    # archived checkpoints of the manuscript on the same starts (separate historical reference)
    A = pd.read_csv(ARCH / 'episodes.csv')
    for m in METHODS:
        a = A[A.method == ARCH_NAME[m]]; el = a[a.eligible.astype(str) == 'True']; cols = set(a.columns)
        el = el.assign(cf=el.strict_success.astype(str) == 'True', s=el.success.astype(str) == 'True')
        out['archived'][m] = dict(n=len(el), recovery_cf=float(100 * el.groupby('scene_id').cf.mean().mean()), recovery=float(100 * el.groupby('scene_id').s.mean().mean()),
                                  final_score=float(el.final_score.mean()) if 'final_score' in cols else None)
    (O / 'analysis_v2.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    for m in METHODS:
        s = out['summary'][m]; print(f"{m:6s} cf {s['recovery_cf']['mean']:.2f}±{s['recovery_cf']['sd']:.2f}  incl.coll {s['recovery']['mean']:.2f}  final {s['final_score']['mean']:.1f}±{s['final_score']['sd']:.1f}  "
                                     f"path {s['success_path_m']['mean']:.3f}  coll {s['self_collisions']['mean']:.0f}  final-ckpt cf {s['final_ckpt_recovery_cf']['mean']:.2f}  | archived {out['archived'][m]}")
    for k, v in out['paired'].items(): print(k, {kk: (np.round(vv, 2).tolist() if isinstance(vv, list) else round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == '__main__':
    main()
