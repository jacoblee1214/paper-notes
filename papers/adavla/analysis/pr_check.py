"""AdaVLA Participation Ratio(식 13–15) 검산 (CPU, numpy 1스레드, 수 초). 실행: python3 analysis/pr_check.py

(i)   PR = (tr G)^2 / ||G||_F^2 (G = A^T A) 가 고윳값 기반 정의 exp(H2)와 일치하는지
(ii)  Eq.11–12의 실효 랭크(σ 정규화 Shannon)와 PR(λ=σ² 정규화 collision)의 차이
(iii) 층별 PR의 최소–최대 정규화(Eq.15)가 활성 스케일에 불변인지
(iv)  Gram 계산 경로(M×M 대 N×N)의 시간 차이 (단일 실행, 참고용)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "pr_check_results.json"


def pr_collision_eig(A):
    lam = np.clip(np.linalg.eigvalsh(A.T @ A), 0, None)
    p = lam / lam.sum()
    return float(1.0 / np.sum(p ** 2))  # exp(H2) with H2 = -ln Σp²


def pr_trace(A):
    return float(np.trace(A.T @ A) ** 2 / np.sum((A.T @ A) ** 2))


def pr_trace_fast(A):
    # tr(G) = ||A||_F², ||G||_F² = ||A Aᵀ||_F² (N×N, N<M일 때 싸다)
    tr = np.sum(A * A)
    K = A @ A.T
    return float(tr ** 2 / np.sum(K * K))


def eff_rank(A):
    s = np.linalg.svd(A, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def main():
    rng = np.random.default_rng(0)
    res = {}
    # (i)
    cases = {}
    for M in [64, 512]:
        A = rng.normal(size=(256, M))
        cases[f"gauss_N256_M{M}"] = A
    low = rng.normal(size=(256, 8)) @ rng.normal(size=(8, 512)) + 0.05 * rng.normal(size=(256, 512))
    cases["lowrank8_plus_noise_N256_M512"] = low
    ident = {}
    for k, A in cases.items():
        p_trace = pr_trace(A)
        p_eig = pr_collision_eig(A)
        p_fast = pr_trace_fast(A)
        ident[k] = {
            "PR_trace_formula": round(p_trace, 4),
            "PR_collision_entropy_eig": round(p_eig, 4),
            "rel_err_trace_vs_eig": float(f"{abs(p_trace - p_eig) / p_eig:.2e}"),
            "rel_err_NxN_gram_vs_eig": float(f"{abs(p_fast - p_eig) / p_eig:.2e}"),
            "PR_over_M": round(p_trace / A.shape[1], 4),
            "effective_rank_eq11_12": round(eff_rank(A), 4),
        }
    res["i_identity_and_effrank"] = ident

    # (ii) 실효 랭크와 PR은 다른 양: 다양한 스펙트럼에서 비교
    comp = {}
    for name, sv in {
        "flat_8": np.r_[np.ones(8), np.zeros(56)],
        "power_0.5": np.arange(1, 65) ** -0.5,
        "power_1.0": np.arange(1, 65) ** -1.0,
        "one_big_63_small": np.r_[10.0, np.ones(63) * 0.5],
    }.items():
        lam = sv ** 2
        p_lam = lam / lam.sum()
        pr = 1.0 / np.sum(p_lam ** 2)
        p_s = sv / sv.sum()
        er = np.exp(-np.sum(p_s[p_s > 0] * np.log(p_s[p_s > 0])))
        comp[name] = {"PR_on_lambda": round(float(pr), 3), "effrank_on_sigma": round(float(er), 3)}
    res["ii_PR_vs_effrank_spectra"] = comp

    # (iii) 스케일 불변성과 층간 min–max 정규화
    layers = [rng.normal(size=(256, 256)) * (1 + 0.3 * l) @ np.diag(np.linspace(1, 0.05, 256) ** (l % 4 + 0.5))
              for l in range(18)]
    prs = np.array([pr_trace(L) for L in layers])
    def mm(x):
        return (x - x.min()) / (x.max() - x.min())
    scaled = np.array([pr_trace(7.3 * L) for L in layers])
    res["iii_scale"] = {
        "PR_scale_invariant_max_abs_diff": float(f"{np.max(np.abs(scaled - prs)):.3g}"),
        "minmax_delta_scale_invariant_max_abs_diff": float(f"{np.max(np.abs(mm(scaled) - mm(prs))):.3g}"),
        "delta_range": [round(float(mm(prs).min()), 3), round(float(mm(prs).max()), 3)],
        "note": "Eq.15 min–max는 층 수 18에서 가장 작은/큰 층을 항상 0/1로 보냄. 상대 순위만 반영.",
    }

    # (iv) 시간 (단일 실행, 참고)
    A = rng.normal(size=(256, 2048))
    t0 = time.perf_counter(); pr_trace(A); t_tr = time.perf_counter() - t0
    t0 = time.perf_counter(); pr_trace_fast(A); t_fast = time.perf_counter() - t0
    t0 = time.perf_counter(); pr_collision_eig(A); t_eig = time.perf_counter() - t0
    t0 = time.perf_counter(); np.linalg.svd(A, compute_uv=False); t_svd = time.perf_counter() - t0
    res["iv_timing_ms_single_run_N256_M2048"] = {
        "trace_MxM_gram": round(t_tr * 1e3, 1), "trace_NxN_gram": round(t_fast * 1e3, 1),
        "eigvalsh_MxM": round(t_eig * 1e3, 1), "svd_values": round(t_svd * 1e3, 1),
        "note": "단일 CPU 1스레드, 참고용. 논문의 Jetson 측정이 아님.",
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
