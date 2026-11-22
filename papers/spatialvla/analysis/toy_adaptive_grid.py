"""토이 T1: 적응형 행동 격자(등확률 가우시안 경계)와 균일 격자의 양자화 오차 비교 (numpy·scipy, 수 초).

공식 코드 scripts/gs_*.json의 평균·표준편차를 읽어 경계를 재구성한다(코드를 import하거나 실행하지 않음).
  - 격자: 코드의 get_bin_policy와 같은 방식 (범위 안에서 잘린 가우시안의 등확률 경계)
  - 비교: 같은 빈 수의 균일 격자
  - 데이터: (a) 같은 분포에서 샘플, (b) 다른 데이터셋 분포에서 샘플 (전이 상황)
실행: python3 analysis/toy_adaptive_grid.py   (저장소 복제본은 spatialvla/repo 또는 SPATIALVLA_REPO 환경변수)
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import norm, truncnorm

HERE = Path(__file__).resolve().parent
GS_DIR = Path(os.environ.get("SPATIALVLA_REPO", HERE.parent / "repo")) / "scripts"  # 얕은 복제본 위치
OUT = HERE / "toy_adaptive_grid_results.json"

RANGE = {"theta": (0.0, np.pi), "phi": (-np.pi, np.pi), "r": (0.0, np.sqrt(3)),
         "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)}
BINS = {"theta": 16, "phi": 32, "r": 8, "roll": 16, "pitch": 16, "yaw": 16}
rng = np.random.default_rng(0)


def gs(name):
    d = json.loads((GS_DIR / name).read_text(encoding="utf-8"))
    return {k: (v["mu"], v["sigma"]) for k, v in d.items() if k in RANGE}


def adaptive_edges(var, mu, sigma):
    lo, hi = RANGE[var]
    n = BINS[var]
    p = np.linspace(norm.cdf(lo, mu, sigma), norm.cdf(hi, mu, sigma), n + 1)
    e = norm.ppf(p, mu, sigma)
    return np.clip(e, lo, hi)


def uniform_edges(var):
    lo, hi = RANGE[var]
    return np.linspace(lo, hi, BINS[var] + 1)


def quant_err(x, edges, mu, sigma):
    """MSE 전체, MAE 전체, 중심 2시그마 안의 MSE (논문 부록 G: 대부분 샘플이 2시그마 안)."""
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    mid = 0.5 * (edges[idx] + edges[idx + 1])
    err = x - mid
    inner = np.abs(x - mu) <= 2 * sigma
    return float(np.mean(err ** 2)), float(np.mean(np.abs(err))), float(np.mean(err[inner] ** 2))


def sample(var, mu, sigma, n=200_000):
    lo, hi = RANGE[var]
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n, random_state=rng)


fractal = gs("gs_fractal.json")
bridge = gs("gs_bridge.json")
out = {"config": {"bins": BINS, "grid_source": "gs_fractal.json", "shift_source": "gs_bridge.json", "n": 200000}, "vars": {}}
tot = {}
for var in RANGE:
    mu_f, s_f = fractal[var]
    mu_b, s_b = bridge[var]
    ea, eu = adaptive_edges(var, mu_f, s_f), uniform_edges(var)
    x_in = sample(var, mu_f, s_f)
    x_sh = sample(var, mu_b, s_b)
    m_ai, a_ai, i_ai = quant_err(x_in, ea, mu_f, s_f)
    m_ui, a_ui, i_ui = quant_err(x_in, eu, mu_f, s_f)
    m_as, a_as, i_as = quant_err(x_sh, ea, mu_f, s_f)
    m_us, a_us, i_us = quant_err(x_sh, eu, mu_f, s_f)
    # 등확률 확인: 격자 경계 사이의 같은 분포 확률 질량 (모두 1/N이어야 함)
    lo, hi = RANGE[var]
    cdf_e = norm.cdf(ea, mu_f, s_f)
    mass = np.diff(cdf_e) / (norm.cdf(hi, mu_f, s_f) - norm.cdf(lo, mu_f, s_f))
    out["vars"][var] = {
        "mse_in_adaptive": m_ai, "mse_in_uniform": m_ui,
        "mse_shift_adaptive": m_as, "mse_shift_uniform": m_us,
        "mae_in_adaptive": a_ai, "mae_in_uniform": a_ui,
        "mae_shift_adaptive": a_as, "mae_shift_uniform": a_us,
        "mse_in2sig_adaptive": i_ai, "mse_in2sig_uniform": i_ui,
        "mse_shift_in2sig_adaptive": i_as, "mse_shift_in2sig_uniform": i_us,
        "mass_per_bin_min": round(float(mass.min()), 4), "mass_per_bin_max": round(float(mass.max()), 4),
        "ideal_mass": round(1 / BINS[var], 4),
    }
    for key, val in [("in_adaptive", m_ai), ("in_uniform", m_ui), ("shift_adaptive", m_as), ("shift_uniform", m_us),
                     ("mae_in_adaptive", a_ai), ("mae_in_uniform", a_ui), ("mae_shift_adaptive", a_as), ("mae_shift_uniform", a_us),
                     ("in2sig_adaptive", i_ai), ("in2sig_uniform", i_ui), ("shift2sig_adaptive", i_as), ("shift2sig_uniform", i_us)]:
        tot.setdefault(key, []).append(val)
out["mean_mse_over_vars"] = {k: float(np.mean(v)) for k, v in tot.items()}
M = out["mean_mse_over_vars"]
out["ratio_adaptive_over_uniform"] = {
    "mse_in_distribution": round(M["in_adaptive"] / M["in_uniform"], 3),
    "mse_shift": round(M["shift_adaptive"] / M["shift_uniform"], 3),
    "mae_in_distribution": round(M["mae_in_adaptive"] / M["mae_in_uniform"], 3),
    "mae_shift": round(M["mae_shift_adaptive"] / M["mae_shift_uniform"], 3),
    "mse_within_2sigma_in_distribution": round(M["in2sig_adaptive"] / M["in2sig_uniform"], 3),
    "mse_within_2sigma_shift": round(M["shift2sig_adaptive"] / M["shift2sig_uniform"], 3),
}
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out["ratio_adaptive_over_uniform"], indent=1))
for v, d in out["vars"].items():
    print(v, {k: (round(x, 5) if isinstance(x, float) else x) for k, x in d.items()})
print("wrote", OUT)
