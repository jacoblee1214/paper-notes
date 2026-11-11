"""추가 검증 2: 스케일 불변 곡률 정의와 Cth 전이성 (CPU numpy 1스레드, 10 시드).

원래 C = ‖v_τ − v_mid‖ / ‖v_mid‖ 는 C ≈ ‖v′‖(1−τ)/(2‖v‖) 이므로 τ에 따라 (1−τ)만큼 변한다.
제안 정의: C_new = ‖v_τ − v_mid‖ / ((1−τ)·(‖v_τ‖+‖v_mid‖)/2)   (τ 보정 + 두 노름 평균)
목표 스케일 c ∈ {0.1, 1, 10}(flow_toy.py의 D 설정과 같은 2봉우리)에서
  (a) 첫 반복 분기 비율: Cth를 c=1의 C 중앙값으로 고정했을 때 c별 낮은 곡률 비율
  (b) 첫 반복 C와 Euler 1단계 오차의 Spearman 상관
  (c) 전체 분기 규칙(flow_toy.adavla와 같은 Eq.5,6,8,9 로직, 곡률만 교체)의 NFE·오차
실행: python3 analysis/extra_flow_scale.py (약 1분)
출력: analysis/extra_flow_scale_results.json
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

OUT = HERE / "extra_flow_scale_results.json"
SEEDS = list(range(10))
N = 2000
SCALES = [0.1, 1.0, 10.0]
CTH_PAPER = 0.15


def curv_paper(va, vm, tau):
    return np.linalg.norm(va - vm, axis=1) / (np.linalg.norm(vm, axis=1) + ft.EPS)


def curv_new(va, vm, tau):
    avg = 0.5 * (np.linalg.norm(va, axis=1) + np.linalg.norm(vm, axis=1))
    return np.linalg.norm(va - vm, axis=1) / ((1 - tau) * avg + ft.EPS)


def adavla_c(g, x0, cth, curv):
    """flow_toy.adavla(논문 식 5,6,8,9)와 같은 로직. 곡률 함수만 바꾼다."""
    n = x0.shape[0]
    x = x0.copy()
    tau = np.zeros(n)
    nfe = np.zeros(n)
    done = np.zeros(n, bool)
    c_first = np.full(n, np.nan)
    for it in range(ft.MAX_IT):
        idx = np.where(~done)[0]
        if idx.size == 0:
            break
        xa, ta = x[idx], tau[idx]
        va = g.v(xa, ta)
        h = (1 - ta) / 2
        vm = g.v(xa + va * h[:, None], ta + h)
        nfe[idx] += 2
        C = curv(va, vm, ta)
        if it == 0:
            c_first[idx] = C
        low = C < cth
        li = idx[low]
        x[li] = xa[low] + vm[low] * (1 - ta[low])[:, None]
        done[li] = True
        hi = ~low
        hi_idx = idx[hi]
        if hi_idx.size:
            th = ta[hi]
            shat = np.minimum((1 - th) / 2, np.maximum(ft.S, cth / C[hi] * 2 * ft.S))
            x[hi_idx] = xa[hi] + vm[hi] * shat[:, None]
            tau[hi_idx] = th + shat
            done[hi_idx[tau[hi_idx] >= 1 - 1e-9]] = True
    rest = np.where(~done)[0]
    if rest.size:
        ta = tau[rest]
        vm = g.v(x[rest], ta + (1 - ta) / 2)
        x[rest] = x[rest] + vm * (1 - ta)[:, None]
        nfe[rest] += 1
    return x, nfe, c_first


def first_iter(g, x0):
    va = g.v(x0, 0.0)
    h = 0.5
    vm = g.v(x0 + va * h, h)
    return va, vm


def main():
    t0 = time.time()
    res = {"seeds": SEEDS, "N": N, "scales": SCALES, "cth_paper": CTH_PAPER, "per_scale": {}}
    gs = {c: ft.GMM(np.array([[-2.0, 0.0], [2.0, 0.0]]) * c, [0.2 * c, 0.2 * c], [0.5, 0.5]) for c in SCALES}

    # (a) 전이성: Cth를 c=1에서 고른 값으로 고정
    cth_new = {}
    for name, curv in [("orig", curv_paper), ("new", curv_new)]:
        vals = []
        for s in SEEDS:
            x0 = np.random.default_rng(s).normal(size=(N, 2))
            va, vm = first_iter(gs[1.0], x0)
            vals.append(curv(va, vm, 0.0))
        cth_new[name] = float(np.median(np.concatenate(vals)))
    res["cth_fixed_from_c1_median"] = {k: round(v, 4) for k, v in cth_new.items()}

    for c in SCALES:
        g = gs[c]
        acc = {d: {"low_frac_first": [], "spearman_first_vs_euler1": [], "nfe": [], "err": [],
                   "low_frac_at_paper_cth": []} for d in ("orig", "new")}
        for s in SEEDS:
            x0 = np.random.default_rng(s).normal(size=(N, 2))
            ref = ft.reference(g, x0)
            e1, _ = ft.euler(g, x0, 1)
            e1_err = np.linalg.norm(e1 - ref, axis=1)
            for name, curv in [("orig", curv_paper), ("new", curv_new)]:
                va, vm = first_iter(g, x0)
                C = curv(va, vm, 0.0)
                acc[name]["low_frac_first"].append(float((C < cth_new[name]).mean()))
                acc[name]["low_frac_at_paper_cth"].append(float((C < CTH_PAPER).mean()))
                acc[name]["spearman_first_vs_euler1"].append(ft.spearman(C, e1_err))
                end, nfe, _ = adavla_c(g, x0, cth_new[name], curv)
                acc[name]["nfe"].append(float(nfe.mean()))
                acc[name]["err"].append(float(np.linalg.norm(end - ref, axis=1).mean()))
        out = {}
        for d in acc:
            out[d] = {k: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4)}
                      for k, v in acc[d].items()}
        res["per_scale"][f"c={c:g}"] = out
        print("done c", c, f"{time.time() - t0:.0f}s", flush=True)

    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("wrote", OUT.name, res["runtime_s"], "s")


if __name__ == "__main__":
    main()
