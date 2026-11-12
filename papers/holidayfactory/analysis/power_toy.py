"""Reset-free 평가의 검정 설계 토이: 작업(job) 단위 검정 대 실행(run) 단위 검정.

논문 24/7-Bench는 작업이 이전 작업의 장면 상태를 물려받으므로, 한 실행 안의 작업은 독립이 아니라고 말한다(§5.3, §6.6).
이 토이는 실행마다 잠재 효과 u_r ~ N(0, s^2)가 로짓 성공률을 옮기는 단순 모형으로 다음을 잰다.
  1) 작업을 독립으로 보는 이항 z 검정(naive)과 실행 평균에 대한 Welch t 검정(run)의 1종 오류
  2) 같은 두 검정의 검정력 (실행 수 R에 따른 변화)
  3) 작업 독립 가정의 Wilson 95% 구간과 실행 군집 t 구간의 포함 확률(coverage)
논문은 실행 수, 실행당 작업 수, 정책 비교 결과를 보고하지 않는다. m = 20, p0 = 0.8, 실행 효과 s는 모두 가정이다.

실행: .venv/bin/python -I analysis/power_toy.py → analysis/power_toy_results.json (수 초, numpy·scipy 단일 스레드)
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as tdist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "power_toy_results.json"

M = 20            # 실행당 작업 수 (가정)
P0 = 0.80         # 기준 정책의 작업 성공률 (가정)
SIGMAS = [0.0, 0.5, 1.0]        # 실행 효과의 로짓 표준편차 (가정)
DELTAS = [0.0, 0.05, 0.10]      # 성공률 차이 (0은 1종 오류)
RUNS = [3, 5, 10, 20, 40, 80]   # 정책당 실행 수
NSIM = 4000
ALPHA = 0.05
SEED = 20261112
Z = norm.ppf(1 - ALPHA / 2)


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def simulate_policy(rng, p, sigma, runs, nsim):
    """실행별 성공 수 (nsim, runs). 로짓 p + sigma * u_r, u_r ~ N(0,1)을 실행마다 뽑고 이항(M, .)로 작업 결과를 뽑는다."""
    u = rng.standard_normal((nsim, runs))
    pr = expit(logit(p) + sigma * u)
    return rng.binomial(M, pr), pr


def naive_test(ka, kb, runs):
    """작업 독립 가정의 두 비율 z 검정(합동 분산). 양측 p-값 (nsim,)."""
    n = runs * M
    pa = ka.sum(1) / n
    pb = kb.sum(1) / n
    pool = (ka.sum(1) + kb.sum(1)) / (2 * n)
    se = np.sqrt(pool * (1 - pool) * (2 / n))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (pb - pa) / se
    z = np.where(se > 0, z, 0.0)
    return 2 * norm.sf(np.abs(z))


def run_test(ka, kb):
    """실행별 성공률 평균에 대한 Welch t 검정. 양측 p-값 (nsim,)."""
    fa = ka / M
    fb = kb / M
    ra, rb = fa.shape[1], fb.shape[1]
    va = fa.var(1, ddof=1)
    vb = fb.var(1, ddof=1)
    se2 = va / ra + vb / rb
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = (fb.mean(1) - fa.mean(1)) / np.sqrt(se2)
        df = se2**2 / ((va / ra) ** 2 / (ra - 1) + (vb / rb) ** 2 / (rb - 1))
    tstat = np.where(np.isfinite(tstat), tstat, 0.0)
    df = np.where(np.isfinite(df) & (df > 0), df, 1.0)
    return 2 * tdist.sf(np.abs(tstat), df)


def wilson_cover(k, n, theta):
    """Wilson 95% 구간이 theta를 포함하는 비율 (nsim 기준 평균)."""
    p = k / n
    den = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return float(np.mean((c - h <= theta) & (theta <= c + h)))


def cluster_t_cover(f, theta):
    """실행 평균 f (nsim, runs)에 대한 t 구간 포함 비율."""
    runs = f.shape[1]
    m = f.mean(1)
    s = f.std(1, ddof=1)
    h = tdist.ppf(0.975, runs - 1) * s / np.sqrt(runs)
    return float(np.mean((m - h <= theta) & (theta <= m + h)))


def main():
    rng = np.random.default_rng(SEED)
    R = {"settings": {"jobs_per_run_M": M, "p0": P0, "sigmas_logit": SIGMAS, "deltas": DELTAS,
                      "runs_per_policy": RUNS, "nsim": NSIM, "alpha": ALPHA, "seed": SEED,
                      "note": "논문에 없는 가정이다. 실행당 작업 수, 실행 효과 크기는 모두 설계 가정."}}

    # 1) 1종 오류와 검정력
    power = {}
    for sigma in SIGMAS:
        power[f"sigma={sigma}"] = {}
        for runs in RUNS:
            row = {}
            for delta in DELTAS:
                p1 = P0  # 정책 A의 로짓 기준값
                p2 = P0 + delta  # 정책 B (로짓 기준에서 성공률 차이만큼 이동)
                ka, pa = simulate_policy(rng, p1, sigma, runs, NSIM)
                kb, pb = simulate_policy(rng, p2, sigma, runs, NSIM)
                pn = naive_test(ka, kb, runs)
                pr = run_test(ka, kb)
                key = f"delta={delta}"
                row[key] = {"naive_reject": round(float(np.mean(pn < ALPHA)), 4),
                            "run_reject": round(float(np.mean(pr < ALPHA)), 4)}
            power[f"sigma={sigma}"][f"R={runs}"] = row
    R["type1_and_power"] = power

    # 2) 80% 검정력을 얻는 최소 실행 수 (run 검정, 격자 값 중 첫 값)
    need = {}
    for sigma in SIGMAS:
        for delta in DELTAS[1:]:
            key = f"sigma={sigma},delta={delta}"
            need[key] = None
            for runs in RUNS:
                if power[f"sigma={sigma}"][f"R={runs}"][f"delta={delta}"]["run_reject"] >= 0.8:
                    need[key] = runs
                    break
    R["runs_for_80pct_power_run_test"] = need

    # 3) 포함 확률: 작업 독립 Wilson 구간 대 실행 군집 t 구간 (정책 A, R = 10)
    cov = {}
    for sigma in SIGMAS:
        big = 400000
        pr_big = expit(logit(P0) + sigma * rng.standard_normal(big))
        theta = float(pr_big.mean())  # 주변 성공률 (모형의 참값)
        runs = 10
        ka, pa = simulate_policy(rng, P0, sigma, runs, NSIM)
        n = runs * M
        k = ka.sum(1)
        cov[f"sigma={sigma}"] = {
            "true_marginal_rate": round(theta, 4),
            "wilson_naive_jobs_coverage": round(wilson_cover(k, n, theta), 4),
            "run_cluster_t_coverage": round(cluster_t_cover(ka / M, theta), 4),
            "design_effect_1_plus_(M-1)ICC": round(1 + (M - 1) * float(pa.var() / (theta * (1 - theta))), 3),
        }
    R["ci_coverage_R10"] = cov

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
