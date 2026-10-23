"""Calibration of the geometric occlusion score V against an image occlusion rate O on the measured scenes (2026-10-16).

For every logged MM-PPO state of the manuscript evaluation (outputs/manuscript_aligned_20261002, 398 scenes x 40 starts)
the scene is reloaded, the archived camera geometry is rebuilt with the historical backend, and on the same inside points
  V = sum_i 1/|d_i|                      (original score, d_i = normalised image coordinate of point i)
  O = 100 |Omega ∩ S| / |Omega|          (silhouette S of the inside points in the task-camera image)
are computed. Each point stands for one voxel of the down-sampled cloud and is drawn as a disc of physical radius
r_w = s/sqrt(pi) (equal area, s = median point spacing), i.e. r_w / Z_i in normalised image units at depth Z_i.
Omega is a centred ellipse with the aspect of the field of view and an area fraction f of the image, f taken from the
annotated target regions of the operating-room frames (median 0.265; IQR 0.157-0.354 used for sensitivity).
"""
from pathlib import Path
import csv, gzip, json, sys
import numpy as np
from ncsim.data import load_scene
from ncsim.historical_backend import configure, classify

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'outputs/manuscript_aligned_20261002'
OUT = ROOT / 'outputs/occlusion_calibration_20261016'
SPACING = 0.0185                  # median nearest-neighbour spacing of the retained points (m), measured on the scenes
F_OMEGA = {'f265': .265, 'f157': .157, 'f354': .354}
R_SCALE = {'r100': 1.0, 'r070': .7, 'r140': 1.4}
NGRID = 161


def image_coords(P, cam_xy, axis, T):
    v = P - np.r_[cam_xy, 1.8]; Z = v @ axis; d = v / Z[:, None] - axis
    return np.c_[d @ T[:3, 0], d @ T[:3, 1]], Z


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure(ROOT / 'outputs/historical_reproduction_20261002')
    rows = {r['scene_id']: r for r in json.loads((RUN / 'ordered_dataset.json').read_text(encoding='utf-8'))}
    starts = json.loads((RUN / 'starts.json').read_text(encoding='utf-8')); case_scene = {s['case_id']: s['scene_id'] for s in starts}
    states = {}
    with gzip.open(RUN / 'steps.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['method'] == 'fusion':
                states.setdefault(case_scene[r['case_id']], []).append(r)
    rw0 = SPACING / np.sqrt(np.pi)
    out = []; nscene = 0
    for sid, st in sorted(states.items()):
        sc = load_scene(rows[sid]); t = sc.target
        P = sc.points[(sc.points[:, 2] >= t[2] + .2) & (sc.points[:, 2] <= 1.8 - .05)]
        for r in st:
            xy = np.array([float(r['x']), float(r['y'])]); q = np.array([float(r['q1']), float(r['q2']), float(r['q3'])])
            ins, V, _, (T, c, ax, cor) = classify(P, xy, t, q)
            uv_c, _ = image_coords(cor, xy, ax, T)
            U, W = np.abs(uv_c[:, 0]).max(), np.abs(uv_c[:, 1]).max()          # half extents of the field of view
            uv, Z = image_coords(P[ins], xy, ax, T)
            rec = dict(case_id=r['case_id'], scene_id=sid, step=int(r['step']), V=V, V_logged=float(r['score']), n_inside=int(ins.sum()),
                       fov_half_u=U, fov_half_v=W, median_depth_m=float(np.median(Z)) if len(Z) else np.nan)
            g = np.linspace(-1, 1, NGRID); gu, gv = np.meshgrid(g * U, g * W)
            for fk, f in F_OMEGA.items():
                k = np.sqrt(4 * f / np.pi); inside_omega = (gu / (k * U)) ** 2 + (gv / (k * W)) ** 2 <= 1
                pts = np.c_[gu[inside_omega], gv[inside_omega]]
                for rk, s in R_SCALE.items():
                    if fk != 'f265' and rk != 'r100':
                        continue
                    if len(uv):
                        dist = np.linalg.norm(pts[:, None, :] - uv[None, :, :], axis=2)
                        covered = (dist <= (s * rw0 / Z)[None, :]).any(1).mean() * 100
                    else:
                        covered = 0.0
                    rec[f'O_{fk}_{rk}'] = covered
            out.append(rec)
        nscene += 1
        if nscene % 40 == 0:
            print(nscene, 'scenes', len(out), 'states', flush=True)
    keys = list(out[0].keys())
    with open(OUT / 'states.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(out)
    print('done', nscene, 'scenes', len(out), 'states')


if __name__ == '__main__':
    main()
