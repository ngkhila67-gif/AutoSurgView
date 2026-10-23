"""Image-based end point of the three-seed subset analysis, from final_state_O.csv (analyse_O.py).
secondary_O_le_20: pre-specified secondary end point (protocol.json), O <= 20 % at the final state of every episode,
                   however the episode ended.
posthoc_O_le_20_collision_free: post hoc sensitivity analysis, episodes ending in self-collision counted as failures.
Both are written into analysis_v2.json; the pre-specified values are checked against those already stored."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

O = Path(__file__).resolve().parent / 'outputs/modality3_reproduction'
SEEDS = [29, 47, 83]; METHODS = ['mmppo', 'pc', 'state']


def endpoint(R, key, rng):
    sc = {m: {s: R[(R.source == 'retrained') & (R.method == m) & (R.training_seed == s)].groupby('scene_id')[key].mean() for s in SEEDS} for m in METHODS}
    per = {m: [round(100 * sc[m][s].mean(), 2) for s in SEEDS] for m in METHODS}
    raw = {m: [100 * sc[m][s].mean() for s in SEEDS] for m in METHODS}
    groups = R.drop_duplicates('scene_id').groupby('capture_group').scene_id.apply(list).to_dict(); G = sorted(groups)
    paired = {}
    for o in ('state', 'pc'):
        d = np.array([100 * (sc['mmppo'][s].mean() - sc[o][s].mean()) for s in SEEDS]); h = stats.t.ppf(.975, len(SEEDS)-1) * d.std(ddof=1) / np.sqrt(len(SEEDS))
        boot = []
        for _ in range(2000):                                         # bootstrap over acquisition folders, averaged over seeds
            ids = [x for g in rng.choice(G, len(G), replace=True) for x in groups[g]]
            boot.append(np.mean([100 * (sc['mmppo'][s].loc[ids].mean() - sc[o][s].loc[ids].mean()) for s in SEEDS]))
        paired[o] = dict(per_seed=d.round(2).tolist(), mean=round(d.mean(), 2), t95=[round(d.mean() - h, 2), round(d.mean() + h, 2)],
                         boot95=np.percentile(boot, [2.5, 97.5]).round(2).tolist(), favour=int((d > 0).sum()))
    arch = {m: round(100 * R[(R.source == 'archived') & (R.method == m)].groupby('scene_id')[key].mean().mean(), 2) for m in METHODS}
    return dict(paired=paired, per_seed=per, archived=arch,
                summary={m: dict(mean=round(float(np.mean(raw[m])), 2), sd=round(float(np.std(raw[m], ddof=1)), 2)) for m in METHODS})


def main():
    R = pd.read_csv(O / 'final_state_O.csv')
    R = R[(R.source == 'archived') | R.training_seed.isin(SEEDS)]
    E = pd.read_csv(O / 'all_test_episodes.csv'); E = E[(E.eligible.astype(str) == 'True') & (E.checkpoint == 'test')]
    E = E.assign(sc=E.self_collision.astype(str) == 'True')[['case_id', 'method', 'training_seed', 'sc', 'capture_group']]
    A = pd.read_csv(Path(__import__('os').environ.get('NC_ARCHIVED_RESULTS', Path(__file__).resolve().parents[1] / 'outputs/manuscript_aligned_20261002')) / 'episodes.csv')
    grp = E.merge(R[['case_id', 'scene_id']].drop_duplicates(), on='case_id').drop_duplicates('scene_id').set_index('scene_id').capture_group
    A = A[A.eligible.astype(str) == 'True'].assign(method=lambda d: d.method.map({'fusion': 'mmppo', 'pointcloud': 'pc', 'robotstate': 'state'}),
                                                   training_seed=0, sc=lambda d: d.self_collision.astype(str) == 'True', capture_group=lambda d: d.scene_id.map(grp))
    R = R.merge(pd.concat([E, A[E.columns]]), on=['case_id', 'method', 'training_seed'], how='left', validate='one_to_one')
    assert R.capture_group.notna().all()
    R['rec'] = R.O_final <= 20; R['rec_cf'] = R.rec & ~R.sc.astype(bool)
    J = json.loads((O / 'analysis_v2.json').read_text(encoding='utf-8')); rng = np.random.default_rng(20261002)
    sec = endpoint(R, 'rec', rng)
    if 'secondary_O_le_20' in J:                                       # stored values must be reproduced
        old = J['secondary_O_le_20']
        assert old['per_seed'] == sec['per_seed'] and old['archived'] == sec['archived'], (old, sec)
        for o in ('state', 'pc'): assert old['paired'][o]['per_seed'] == sec['paired'][o]['per_seed'] and old['paired'][o]['t95'] == sec['paired'][o]['t95']
    J['secondary_O_le_20'] = sec
    ph = endpoint(R, 'rec_cf', np.random.default_rng(20261002)); ph['note'] = 'post hoc sensitivity analysis: O <= 20 % at the final state and no self-collision'
    J['posthoc_O_le_20_collision_free'] = ph
    (O / 'analysis_v2.json').write_text(json.dumps(J, indent=1), encoding='utf-8')
    for k in ('secondary_O_le_20', 'posthoc_O_le_20_collision_free'): print(k, json.dumps({x: J[k][x] for x in ('summary', 'paired', 'archived')}))


if __name__ == '__main__':
    main()
