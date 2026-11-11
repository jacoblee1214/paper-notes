"""추가 검증 (ART-Glove): taxel 간격과 판독 잡음에 따른 접촉 위치 오차, 목표 2 mm를 만족하는 범위.

03 §5의 토이 A(taxel_loc_toy.py)를 간격 2–8 mm를 0.5 mm 단위로, 접촉 크기를 3가지로 넓혀 다시 돌린다 (numpy, 단일 스레드).
논문에는 taxel 실제 간격이 없다 (01 §4 "taxel 간격은 없다"). 따라서 간격은 모두 가정이다.
  - 40 x 40 mm 패치, 격자 간격 p in {2, 2.5, ..., 8} mm.
  - 접촉: 가우시안, 표준편차 sigma_c in {1.5, 3, 5} mm. 중심은 [8, 32]^2 mm 균일 (기존 토이와 같음).
  - 판독: 격자점 압력 + 잡음 (피크 대비 sigma_n in {0, 0.02, 0.05, 0.10}), 음수는 0.
  - 추정: 최댓값의 20%를 넘는 taxel의 (가중치 r - t) 질량중심 (기존 토이와 같음).
  - 시행 1000회 per 셀. 시드는 셀마다 고정 (seed = 셀 인덱스 기반).
  - 목표: 중앙 오차 <= 2 mm 과 90분위 <= 2 mm 두 기준을 함께 본다. 만족하는 최대 간격을 셀별로 찾는다.
  - 대조: 기존 토이의 값(taxel_loc_results.json)과 같은 격자 점에서 비교해 MC 오차 내에서 맞는지 본다.

실행: python3 analysis/extra_taxel_grid.py -> analysis/extra_taxel_grid_results.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_taxel_grid_results.json"
PATCH = 40.0
PITCHES = [round(x, 1) for x in np.arange(2.0, 8.0 + 1e-9, 0.5)]
SIG_CS = [1.5, 3.0, 5.0]
NOISES = [0.0, 0.02, 0.05, 0.10]
TRIALS = 1000
TARGET_MM = 2.0
REF_PITCHES = [2.0, 3.0, 4.0, 6.0, 8.0]


def nodes_for(p):
    g = np.arange(0.0, PATCH + 1e-9, p)
    xx, yy = np.meshgrid(g, g)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def run_cell(p, sig_c, sn, seed):
    rng = np.random.default_rng(seed)
    nodes = nodes_for(p)
    c = rng.uniform(8.0, 32.0, size=(TRIALS, 2))
    d2 = ((nodes[None, :, :] - c[:, None, :]) ** 2).sum(-1)
    r = np.exp(-d2 / (2 * sig_c ** 2))
    r = r + sn * rng.standard_normal(r.shape)
    r = np.clip(r, 0.0, None)
    t = 0.2 * r.max(axis=1, keepdims=True)
    w = np.clip(r - t, 0.0, None)
    ws = w.sum(axis=1)
    est = np.einsum("tn,nd->td", w, nodes) / np.where(ws > 0, ws, 1.0)[:, None]
    err = np.linalg.norm(est - c, axis=1)
    err = err[ws > 0]
    return float(np.median(err)), float(np.percentile(err, 90))


def main():
    res = {"settings": {"patch_mm": PATCH, "pitches_mm": PITCHES, "contact_sigma_mm": SIG_CS,
                        "noise_frac": NOISES, "trials": TRIALS, "target_mm": TARGET_MM,
                        "estimator": "20% of max threshold, weighted centroid",
                        "note": "taxel 간격은 논문에 없음 (가정). 실제 간격 확인 필요"},
           "median_mm": {}, "p90_mm": {}, "max_pitch_meeting_target": {}}
    seed = 0
    for sig_c in SIG_CS:
        for sn in NOISES:
            med_row, p90_row = [], []
            for p in PITCHES:
                seed += 1
                m, q = run_cell(p, sig_c, sn, seed)
                med_row.append(round(m, 3))
                p90_row.append(round(q, 3))
            key = f"sigma_c={sig_c:g},noise={sn:g}"
            res["median_mm"][key] = med_row
            res["p90_mm"][key] = p90_row
            ok = [p for p, m, q in zip(PITCHES, med_row, p90_row) if m <= TARGET_MM and q <= TARGET_MM]
            res["max_pitch_meeting_target"][key] = max(ok) if ok else None
    res["pitch_index"] = PITCHES
    # 기존 토이 값과 비교 (sigma_c=3, 같은 격자 점)
    old = json.loads((ROOT / "analysis" / "taxel_loc_results.json").read_text())
    cmp = {}
    for p in REF_PITCHES:
        for j, sn in enumerate(NOISES):
            new_m = res["median_mm"][f"sigma_c=3,noise={sn:g}"][PITCHES.index(p)]
            old_m = old["median_mm"][f"{p:g}"][j] if f"{p:g}" in old["median_mm"] else old["median_mm"][str(p)][j]
            cmp[f"p={p:g},noise={sn:g}"] = {"old_median": old_m, "new_median": new_m,
                                             "abs_diff": round(abs(new_m - old_m), 3)}
    res["comparison_with_original_toy_sigma3"] = cmp
    res["max_abs_diff_vs_original"] = max(v["abs_diff"] for v in cmp.values())
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("max pitch meeting 2mm (median and p90):")
    for k, v in res["max_pitch_meeting_target"].items():
        print(" ", k, v)
    print("max abs diff vs original toy (sigma 3):", res["max_abs_diff_vs_original"])


if __name__ == "__main__":
    main()
