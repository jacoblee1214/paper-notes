"""추가 검증 1: 같은 평균 NFE에서 일관 중점 분기와 정적 Euler·RK2 비교 (10 시드, CPU numpy 1스레드).

실행: python3 analysis/extra_flow_nfe.py   (약 1–2분)
flow_toy.py의 GMM, reference, euler, midpoint, adavla를 그대로 가져다 쓴다.
출력: analysis/extra_flow_nfe_results.json
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
import flow_toy as ft  # noqa: E402

OUT = HERE / "extra_flow_nfe_results.json"
SEEDS = list(range(10))
N = 2000
CTHS = [0.15, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 2.0, 5.0]
STATIC_N = [2, 3, 4, 5, 10, 20, 30]  # 정적 Euler 기준선 (NFE = n)


def configs():
    return {
        "B_two_modes_a2": ft.GMM([[-2.0, 0.0], [2.0, 0.0]], [0.2, 0.2], [0.5, 0.5]),
        "C_ring4_r3": ft.GMM([[3, 0], [0, 3], [-3, 0], [0, -3]], [0.25] * 4, [0.25] * 4),
    }


def summarize(vals):
    a = np.asarray(vals, float)
    return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4)}


def main():
    t0 = time.time()
    res = {"seeds": SEEDS, "N": N, "tol": ft.TOL, "configs": {}}
    for name, g in configs().items():
        rec = {"static": {}, "paper": {}, "consistent": {}}
        per_seed = {"static": {}, "paper": {}, "consistent": {}}
        for s in SEEDS:
            x0 = np.random.default_rng(s).normal(size=(N, 2))
            ref = ft.reference(g, x0)
            centers = g.mu
            for n in STATIC_N:
                end, nfe = ft.euler(g, x0, n)
                r, _ = ft.metrics(end, ref, np.full(N, nfe), centers)
                per_seed["static"].setdefault(f"euler_{n}", []).append((nfe, r["err_mean"]))
            for n in [1, 2]:
                end, nfe = ft.midpoint(g, x0, n)
                r, _ = ft.metrics(end, ref, np.full(N, nfe), centers)
                per_seed["static"].setdefault(f"rk2_{n}", []).append((nfe, r["err_mean"]))
            for cth in CTHS:
                end, nfe, _, _, _ = ft.adavla(g, x0, cth)
                r, _ = ft.metrics(end, ref, nfe, centers)
                per_seed["paper"].setdefault(f"cth_{cth}", []).append((r["nfe_mean"], r["err_mean"]))
                end2, nfe2, _, _, _ = ft.adavla(g, x0, cth, consistent=True)
                r2, _ = ft.metrics(end2, ref, nfe2, centers)
                per_seed["consistent"].setdefault(f"cth_{cth}", []).append((r2["nfe_mean"], r2["err_mean"]))
        for kind in per_seed:
            for key, pairs in per_seed[kind].items():
                nfes = [p[0] for p in pairs]
                errs = [p[1] for p in pairs]
                rec[kind][key] = {"nfe": summarize(nfes), "err": summarize(errs)}
        # 같은 NFE 비교: 시드마다 일관 중점의 NFE를 정적 Euler 곡선(n=2,3,4,5,10)에 선형 보간한 오차와 비교
        comp = {}
        for cth in CTHS:
            cons = per_seed["consistent"][f"cth_{cth}"]
            wins, gaps, nfes = 0, [], []
            for s_i, (nfe_c, err_c) in enumerate(cons):
                eu = sorted((per_seed["static"][f"euler_{n}"][s_i] for n in STATIC_N))
                xs = [e[0] for e in eu]
                ys = [e[1] for e in eu]
                err_ref = float(np.interp(nfe_c, xs, ys))
                wins += int(err_c < err_ref)
                gaps.append(err_c - err_ref)
                nfes.append(nfe_c)
            comp[f"cth_{cth}"] = {
                "consistent_nfe_mean": round(float(np.mean(nfes)), 3),
                "gap_err_minus_euler_interp_mean": round(float(np.mean(gaps)), 4),
                "seeds_consistent_better": f"{wins}/{len(SEEDS)}",
            }
        rec["matched_nfe_comparison"] = comp
        res["configs"][name] = rec
        print("done", name, f"{time.time() - t0:.0f}s", flush=True)

    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("wrote", OUT.name, res["runtime_s"], "s")


if __name__ == "__main__":
    main()
