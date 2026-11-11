"""추가 검증 1·2: (1) 실행 불가능 시연의 섞임이 꼬리 지수 a를 만드는가, (2) 같은 manifold toy에서
검색 대 결정론적 회귀 대 가우시안 혼합(EM) 조건부 밀도를 비교.

(1) 꼬리 지수 기전 (해석 계산, 몬테카를로 없음)
  시연 집단의 p(시연 하나를 GNR 표본 하나로 풀 확률)을 다음과 같이 둔다.
    f 비율: 실행 불가능 (p = 0),  1 − f 비율: p ~ Beta(α, β)
  pass@S = (1 − f) · (1 − B(α, β+S)/B(α, β))   (f만큼은 어떤 S에서도 실패)
  α, f를 바꾸고, S = 1 성공률 6.87%(HOT3D MPC-failure, Table 1)에 맞도록 β를 정한 뒤,
  S = 64, 128, 256의 관측값 34.89, 38.05, 42.58과 비교한다(RMSE, %p).
  추가로 03 §2와 같은 3-모수 모형(π0, a, b)을 자유롭게 맞춘 a와 π0 = 0으로 고정해 맞춘 a를 본다.

(2) 조건부 생성기 비교 (manifold toy, manifold_toy.py의 설정 그대로)
  - 검색: 최근접 시드 잔차 + 국소 선형 보정 (manifold_toy.gnr_proxy, 기존 기준선)
  - 결정론적 회귀: 대상 조건에서 모드 평균을 한 점으로 예측하고 잡음 σ만 더함
  - 가우시안 혼합 조건부 밀도: (c, Δ) 결합 공간에서 K = 3 가우시안 혼합을 EM으로 적합한 뒤 조건부 샘플링
  시드는 MPC로 푼 것만 쓰고(930개), 일부(q = 0, 0.2)를 실행 불가능한 무작위 Δ로 바꿔 오염시킨다.
  시험: 학습 범위 안 400개(in-dist), 빈 원판 400개(보간 OOD), 각 S = 256.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import betaln

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manifold_toy as mt  # noqa: E402  (같은 폴더의 기존 toy: 설정·함수 재사용)

OUT_JSON = HERE / "extra_tail_and_generators_results.json"
S_GRID = [1, 64, 128, 256]
OBS = np.array([6.87, 34.89, 38.05, 42.58]) / 100


def pass_mix(S, f, a, b):
    S = np.asarray(S, float)
    return (1 - f) * (1 - np.exp(betaln(a, b + S) - betaln(a, b)))


def fit_free(S, y):
    """03 §2와 같은 3-모수 적합 (π0 = f 자유)."""
    def res(t):
        f = 1 / (1 + np.exp(-t[0]))
        return pass_mix(S, f, np.exp(t[1]), np.exp(t[2])) - y
    best = None
    for t0 in ([-2, -1, 0], [0, -1, 1], [-4, 0, 2], [1, -2, 0], [-3, -0.5, 3]):
        r = least_squares(res, t0, bounds=([-12, -6, -6], [12, 4, 8]))
        if best is None or r.cost < best.cost:
            best = r
    t = best.x
    return {"pi0": round(float(1 / (1 + np.exp(-t[0]))), 3), "a": round(float(np.exp(t[1])), 3),
            "b": round(float(np.exp(t[2])), 3), "rmse_pp": round(float(np.sqrt(np.mean(best.fun ** 2)) * 100), 2)}


def fit_pi0_zero(S, y):
    """π0 = 0으로 고정하고 (a, b)만 적합."""
    def res(t):
        return pass_mix(S, 0.0, np.exp(t[0]), np.exp(t[1])) - y
    best = None
    for t0 in ([-2, 0], [0, 1], [-1, -1], [1, 2]):
        r = least_squares(res, t0, bounds=([-6, -6], [4, 8]))
        if best is None or r.cost < best.cost:
            best = r
    return {"a": round(float(np.exp(best.x[0])), 3), "b": round(float(np.exp(best.x[1])), 3),
            "rmse_pp": round(float(np.sqrt(np.mean(best.fun ** 2)) * 100), 2)}


def part1():
    """(f, α)마다 S=1 값에 맞춰 β를 정하고 S = 64·128·256을 관측과 비교."""
    rows = {}
    for f in (0.0, 0.2, 0.4, 0.5):
        for alpha in (0.14, 0.5, 2.0):
            beta = (1 - f) * alpha / OBS[0] - alpha
            if beta <= 0:
                rows[f"f={f},alpha={alpha}"] = "infeasible (beta<=0)"
                continue
            pred = pass_mix(S_GRID, f, alpha, beta) * 100
            rmse = float(np.sqrt(np.mean((pred[1:] - OBS[1:] * 100) ** 2)))
            rows[f"f={f},alpha={alpha}"] = {
                "beta": round(beta, 3), "pred_pass_64_128_256": [round(float(v), 1) for v in pred[1:]],
                "rmse_pp_on_64_128_256": round(rmse, 2)}
    # 관측 곡선을 자유 3-모수로 적합하면 얻는 a (π0 고정 0과 비교)
    obs_fit = {"free_3param": fit_free(S_GRID, OBS),
               "pi0_forced_0": fit_pi0_zero(S_GRID, OBS)}
    return {"match_S1_then_compare": rows, "paper_curve_fit": obs_fit}


# ---------- (2) 생성기 비교 ----------
def fit_gmm(Z, K, rng, iters=300, reg=1e-3):
    n, d = Z.shape
    mu = Z[rng.choice(n, K, replace=False)].copy()
    cov = np.array([np.cov(Z.T) + reg * np.eye(d)] * K)
    pi = np.full(K, 1.0 / K)
    prev = -np.inf
    for _ in range(iters):
        logp = np.empty((n, K))
        for k in range(K):
            L = np.linalg.cholesky(cov[k])
            sol = np.linalg.solve(L, (Z - mu[k]).T)
            logp[:, k] = np.log(pi[k]) - 0.5 * (np.sum(sol ** 2, 0) + 2 * np.sum(np.log(np.diag(L))) + d * np.log(2 * np.pi))
        m = logp.max(1, keepdims=True)
        ll = float(np.sum(m[:, 0] + np.log(np.exp(logp - m).sum(1))))
        r = np.exp(logp - m)
        r /= r.sum(1, keepdims=True)
        Nk = r.sum(0) + 1e-10
        pi = Nk / n
        mu = (r.T @ Z) / Nk[:, None]
        for k in range(K):
            diff = Z - mu[k]
            cov[k] = (r[:, k, None] * diff).T @ diff / Nk[k] + reg * np.eye(d)
        if abs(ll - prev) < 1e-6 * abs(ll):
            break
        prev = ll
    return pi, mu, cov, ll


def gmm_sample(pi, mu, cov, c_test, S, rng):
    """조건부 샘플: p(Δ | c) = Σ_k w_k(c) N(Δ | m_k(c), C_k). 입력 c_test (n, 2), 출력 (n, S, D)."""
    K = len(pi)
    n = len(c_test)
    D = mu.shape[1] - 2
    out = np.empty((n, S, D))
    for i, c in enumerate(c_test):
        logw = np.empty(K)
        cond_m, cond_C = [], []
        for k in range(K):
            Scc = cov[k][:2, :2]
            Sdc = cov[k][2:, :2]
            Sdd = cov[k][2:, 2:]
            Si = np.linalg.inv(Scc)
            dc = c - mu[k][:2]
            logw[k] = np.log(pi[k]) - 0.5 * (dc @ Si @ dc + np.log(np.linalg.det(Scc)))
            cond_m.append(mu[k][2:] + Sdc @ Si @ dc)
            cond_C.append(Sdd - Sdc @ Si @ Sdc.T + 1e-6 * np.eye(D))
        w = np.exp(logw - logw.max())
        w /= w.sum()
        comp = rng.choice(K, size=S, p=w)
        for k in range(K):
            idx = np.where(comp == k)[0]
            if idx.size == 0:
                continue
            Lk = np.linalg.cholesky(cond_C[k])
            out[i, idx] = cond_m[k] + rng.normal(size=(idx.size, D)) @ Lk.T
    return out


def part2(n_rep=5):
    rng = np.random.default_rng(7)
    # 시드: manifold_toy.main과 같은 방식으로 만든다 (풀 1600, MPC 3,000표본)
    c_pool = rng.uniform(-1, 1, size=(1600, 2))
    ok, _, sol = mt.mpc(c_pool, 300, 10, rng)
    center, rad = np.array([0.3, -0.2]), 0.35
    keep = ok & (np.linalg.norm(c_pool - center, axis=1) > rad)
    c_seed, d_seed = c_pool[keep], sol[keep]
    # 시험 집합
    c_in = rng.uniform(-1, 1, size=(600, 2))
    c_in = c_in[np.linalg.norm(c_in - center, axis=1) > rad][:400]
    ang, rr = rng.uniform(0, 2 * np.pi, 400), rad * np.sqrt(rng.uniform(0, 1, 400))
    c_hole = center + np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1)
    tests = {"in_dist": c_in, "interp_ood": c_hole}
    methods = ["search_pass", "deterministic_regression_pass", "gmm_conditional_pass"]

    acc = {}  # (q, test, method) -> list of pass curves
    for rep in range(n_rep):
        r_rng = np.random.default_rng(100 + rep)
        for q in (0.0, 0.2):
            d_use = d_seed.copy()
            n_bad = int(round(q * len(d_use)))
            bad_idx = r_rng.choice(len(d_use), n_bad, replace=False)
            d_use[bad_idx] = r_rng.uniform(-2.5, 2.5, size=(n_bad, mt.D))  # 실행 불가능한 무작위 Δ
            pi, mu, cov, ll = fit_gmm(np.hstack([c_seed, d_use]), 3, np.random.default_rng(11 + rep))
            for name, cc in tests.items():
                succ_s = mt.feasible(cc, mt.gnr_proxy(c_seed, d_use, cc, 256, np.random.default_rng(3 + rep), residual=True))
                tt = mt.targets(cc)
                pred = np.broadcast_to(tt.mean(1)[:, None, :], (len(cc), 256, mt.D))
                pred = pred + 0.06 * np.random.default_rng(5 + rep).normal(size=pred.shape)
                succ_r = mt.feasible(cc, pred)
                samp = gmm_sample(pi, mu, cov, cc, 256, np.random.default_rng(9 + rep))
                succ_g = mt.feasible(cc, samp)
                for m, succ in zip(methods, (succ_s, succ_r, succ_g)):
                    acc.setdefault((q, name, m), []).append(mt.pass_curve(succ))
                acc.setdefault((q, name, "n_contaminated"), []).append(n_bad)
    res = {"n_seeds": int(keep.sum()), "n_replicates": n_rep, "summary": {}}
    for (q, name, m), vals in acc.items():
        key = f"q={q}/{name}/{m}"
        if m == "n_contaminated":
            res["summary"][key] = vals[0]
            continue
        row = {}
        for s in (1, 4, 16, 64, 256):
            v = np.array([x[s] for x in vals])
            row[f"pass@{s}_mean"] = round(float(v.mean()), 1)
            row[f"pass@{s}_sd"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 1)
        res["summary"][key] = row
    return res


def main():
    t0 = time.time()
    out = {"part1_tail_mechanism": part1()}
    print(json.dumps(out["part1_tail_mechanism"], ensure_ascii=False, indent=1), flush=True)
    out["part2_generators"] = part2()
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("runtime", out["runtime_s"])


if __name__ == "__main__":
    main()
