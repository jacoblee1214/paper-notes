"""추가 검증 1: H_G 민감도의 시드 반복 (CPU numpy 1스레드, 시드 10개).

entropy_toy.py의 자료 생성(make_demos)과 정렬·H_G(align, h_g)를 그대로 가져다 쓴다.
기존 검사(03 §3.1)는 시드 3개 또는 1개였으므로 다음을 10시드로 다시 본다.
  (a) T1b 순서: 시작–목표 거리 0.3, N=1000에서 k=0과 k=1의 r 의존성, k=0−k=1 차이의 표준편차
  (b) N 편향: r=0.10, k=0, 시작 거리 0.3, 노이즈 0.002와 0.03에서 M = 50–1000의 편향(M=6000 참값 대비), 시드 간 표준편차
  (c) 재표집 점 수 T=25, 50, 100의 H 차이가 ln 2 배수와 맞는지 (M=500, 노이즈 0.002)
실행: python3 analysis/extra_hg_replicates.py  (1스레드, 수 분 이내)
출력: analysis/extra_hg_replicates_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import entropy_toy as et  # noqa: E402

OUT = HERE / "extra_hg_replicates_results.json"
SEEDS = range(10)
t0 = time.time()


def H(M, r, k, seed, noise, start_radius=1.0, T_pts=50, tag=0):
    rng = np.random.default_rng(10_000 * seed + 100 * k + int(r * 1000) + 7 * tag + int(M))
    W, R, g = et.make_demos(M, r, k, rng, noise=noise, T_pts=T_pts, start_radius=start_radius)
    return float(et.h_g(et.align(W, R, g)))


def ms(v):
    v = np.asarray(v, float)
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4)}


def main():
    res = {"seeds": len(SEEDS), "entropy_toy": "entropy_toy.py의 make_demos·align·h_g 사용"}

    # (a) T1b 순서, 시작 거리 0.3, N=1000
    rs = [0.0, 0.01, 0.03, 0.05, 0.10]
    a = {}
    for k in [0, 1]:
        a[f"k={k}"] = {}
        for r in rs:
            vals = [H(1000, r, k, s, 0.002, start_radius=0.3, tag=1) for s in SEEDS]
            a[f"k={k}"][f"r={r}"] = ms(vals)
    diff = [H(1000, 0.10, 0, s, 0.002, start_radius=0.3, tag=1) - H(1000, 0.10, 1, s, 0.002, start_radius=0.3, tag=1)
            for s in SEEDS]
    a["k0_minus_k1_at_r0.10"] = ms(diff)
    res["a_order_startdist0.3_N1000"] = a
    print("a done", round(time.time() - t0, 1), flush=True)

    # (b) N 편향, r=0.10, k=0, 시작 거리 0.3. 참값: 같은 설정의 M=6000 (시드별로 같은 자료군 사용)
    b = {}
    for noise in [0.002, 0.03]:
        ref = [H(6000, 0.10, 0, s, noise, start_radius=0.3, tag=2) for s in SEEDS]
        row = {}
        for M in [50, 100, 200, 500, 1000]:
            vals = [H(M, 0.10, 0, s, noise, start_radius=0.3, tag=2) for s in SEEDS]
            dev = np.array(vals) - np.array(ref)
            row[f"M={M}"] = {"H": ms(vals), "bias_vs_M6000": ms(dev)}
        b[f"noise={noise}"] = row
    res["b_N_bias_r0.10_k0"] = b
    print("b done", round(time.time() - t0, 1), flush=True)

    # (c) 재표집 점 수 T (M=500, 노이즈 0.002, 거리 1.0)
    c = {}
    per_T = {}
    for T in [25, 50, 100]:
        per_T[T] = [H(500, 0.10, 0, s, 0.002, T_pts=T, tag=3) for s in SEEDS]
        c[f"T={T}"] = ms(per_T[T])
    c["diff_T100_minus_T50"] = ms(np.array(per_T[100]) - np.array(per_T[50]))
    c["diff_T50_minus_T25"] = ms(np.array(per_T[50]) - np.array(per_T[25]))
    c["ln2"] = round(float(np.log(2)), 4)
    res["c_resample_points"] = c
    print("c done", round(time.time() - t0, 1), flush=True)

    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("wrote", OUT.name, res["runtime_s"], "s")


if __name__ == "__main__":
    main()
