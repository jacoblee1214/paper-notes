"""토이 A: 촉각 접촉 위치 오차 vs taxel 간격(pitch)과 잡음 (numpy, 1스레드, 수 초).

설정 (논문에 없는 값은 가정으로 표시)
- 40 x 40 mm 평면 패치, taxel 격자 간격 p in {2,3,4,6,8} mm  (실제 간격은 논문에 없음: 가정)
- 접촉: 가우시안 압력 분포, 표준편차 3 mm, 중심은 [8, 32]^2 mm에서 균일 샘플링 (가정)
- 판독: 격자점에서 압력 샘플 + 가우시안 잡음 sigma_n (피크 대비 비율) , 음수는 0으로 자름
- 추정: 최대값의 20%를 넘는 taxel의 (r - t) 가중 질량중심
- 1000회 반복, 시드 0
측정값: 중심 오차의 중앙값과 90분위수 (mm)
실행: python3 analysis/taxel_loc_toy.py -> analysis/taxel_loc_results.json
"""
import json
import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH = 40.0
SIG_C = 3.0
PITCHES = [2.0, 3.0, 4.0, 6.0, 8.0]
NOISE = [0.0, 0.02, 0.05, 0.10]
TRIALS = 1000
rng = np.random.default_rng(0)


def grid(p):
    g = np.arange(0.0, PATCH + 1e-9, p)
    xx, yy = np.meshgrid(g, g)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def trial(nodes, c, sn):
    d2 = ((nodes - c) ** 2).sum(1)
    r = np.exp(-d2 / (2 * SIG_C ** 2))
    r = r + sn * rng.standard_normal(r.shape)
    r = np.clip(r, 0.0, None)
    t = 0.2 * r.max()
    w = np.clip(r - t, 0.0, None)
    if w.sum() <= 0:
        return np.nan
    est = (w[:, None] * nodes).sum(0) / w.sum()
    return float(np.linalg.norm(est - c))


res = {"config": {"patch_mm": PATCH, "contact_sigma_mm": SIG_C, "trials": TRIALS, "seed": 0,
                  "threshold": "0.2*max", "note": "taxel 간격과 접촉 크기는 가정. 논문 미기재"},
       "pitch_mm": PITCHES, "noise_frac": NOISE, "median_mm": {}, "p90_mm": {}}
for p in PITCHES:
    nodes = grid(p)
    med_row, p90_row = [], []
    for sn in NOISE:
        errs = []
        for _ in range(TRIALS):
            c = rng.uniform(8.0, 32.0, size=2)
            errs.append(trial(nodes, c, sn))
        e = np.array(errs)
        e = e[~np.isnan(e)]
        med_row.append(round(float(np.median(e)), 3))
        p90_row.append(round(float(np.percentile(e, 90)), 3))
    res["median_mm"][str(p)] = med_row
    res["p90_mm"][str(p)] = p90_row

# 간격 대비 오차 비율: 중앙값 / 간격 (잡음 0)
res["median_over_pitch_noise0"] = {str(p): round(res["median_mm"][str(p)][0] / p, 3) for p in PITCHES}
(ROOT / "analysis" / "taxel_loc_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))
