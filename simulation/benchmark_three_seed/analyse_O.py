"""Secondary endpoint of protocol v2: recovery with the image occlusion rate O <= 20 % at the final state
(definition of occlusion_calibration.py: region 26.5 % of the image, points drawn as discs of the voxel area), for the
validation-selected checkpoints of the three-seed subset analysis and, as a separate reference, the archived checkpoints."""
import ast, csv, gzip, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # simulation/ (ncsim package)
from ncsim.data import load_scene
from ncsim.historical_backend import configure, classify

SIM = Path(__file__).resolve().parents[1]; O = Path(__file__).resolve().parent / 'outputs/modality3_reproduction'
ARCH = Path(__import__('os').environ.get('NC_ARCHIVED_RESULTS', SIM / 'outputs/manuscript_aligned_20261002')); RW = 0.0185 / np.sqrt(np.pi); F = .265; NG = 161


def occ_rate(P, t, xy, q):
    ins, V, _, (T, c, ax, cor) = classify(P, np.asarray(xy, float), t, np.asarray(q, float))
    def uvz(Q):
        v = Q - np.r_[xy, 1.8]; Z = v @ ax; d = v / Z[:, None] - ax; return np.c_[d @ T[:3, 0], d @ T[:3, 1]], Z
    uc, _ = uvz(cor); U, W = np.abs(uc[:, 0]).max(), np.abs(uc[:, 1]).max()
    if not ins.any(): return 0.0, V
    uv, Z = uvz(P[ins]); g = np.linspace(-1, 1, NG); gu, gv = np.meshgrid(g * U, g * W); k = np.sqrt(4 * F / np.pi)
    m = (gu / (k * U)) ** 2 + (gv / (k * W)) ** 2 <= 1; pts = np.c_[gu[m], gv[m]]
    return float((np.linalg.norm(pts[:, None] - uv[None], axis=2) <= (RW / Z)[None]).any(1).mean() * 100), V


def main():
    configure(SIM / 'outputs/historical_reproduction_20261002')
    rows = {r['scene_id']: r for r in json.loads((ARCH / 'ordered_dataset.json').read_text(encoding='utf-8'))}
    E = pd.read_csv(O / 'all_test_episodes.csv'); E = E[(E.eligible.astype(str) == 'True') & (E.checkpoint == 'test') & E.training_seed.isin([29,47,83])]
    E = E.assign(fx=E.final_xy.map(ast.literal_eval), fq=E.final_q.map(ast.literal_eval), source='retrained')
    # archived checkpoints: final state = last logged step of every case
    S = pd.read_csv(ARCH / 'steps.csv.gz'); last = S.sort_values('step').groupby(['case_id', 'method']).tail(1)
    A = pd.read_csv(ARCH / 'episodes.csv'); A = A[A.eligible.astype(str) == 'True'].merge(last[['case_id', 'method', 'x', 'y', 'q1', 'q2', 'q3', 'score']], on=['case_id', 'method'])
    A = A.assign(fx=list(zip(A.x, A.y)), fq=list(zip(A.q1, A.q2, A.q3)), training_seed=0, source='archived',
                 method=A.method.map({'fusion': 'mmppo', 'pointcloud': 'pc', 'robotstate': 'state'}))
    cols = ['source', 'method', 'training_seed', 'case_id', 'scene_id', 'final_score', 'strict_success', 'fx', 'fq']
    X = pd.concat([E[cols], A[cols]], ignore_index=True); out = []
    for k, (sid, d) in enumerate(X.groupby('scene_id')):
        sc = load_scene(rows[sid]); t = sc.target; P = sc.points[(sc.points[:, 2] >= t[2] + .2) & (sc.points[:, 2] <= 1.8 - .05)]
        for r in d.itertuples():
            o, V = occ_rate(P, t, r.fx, r.fq)
            assert abs(V - r.final_score) <= 1e-6 * max(1, V), (r.case_id, r.method, V, r.final_score)
            out.append(dict(source=r.source, method=r.method, training_seed=r.training_seed, case_id=r.case_id, scene_id=sid, final_score=V, O_final=o,
                            strict_success=str(r.strict_success) == 'True'))
        if (k + 1) % 50 == 0: print(k + 1, 'scenes', flush=True)
    R = pd.DataFrame(out); R.to_csv(O / 'final_state_O.csv', index=False)
    R['rec_O'] = R.O_final <= 20
    summ = R.groupby(['source', 'method', 'training_seed']).apply(lambda d: pd.Series(dict(recovery_O_macro=100 * d.groupby('scene_id').rec_O.mean().mean(),
                                                                                          recovery_V_cf_macro=100 * d.groupby('scene_id').strict_success.mean().mean(),
                                                                                          mean_O_final=d.O_final.mean())))
    summ.to_csv(O / 'final_state_O_summary.csv'); print(summ.round(2).to_string())


if __name__ == '__main__':
    main()
