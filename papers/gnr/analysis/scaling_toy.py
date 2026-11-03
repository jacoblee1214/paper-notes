"""test-time scaling 곡선의 구조: 시연마다 성공 확률이 다르다는 모델(Beta + 0 질량)로 pass@S를 맞춘다.

pass@S = (1 - pi0) * (1 - B(a, b + S) / B(a, b))      p ~ pi0·δ0 + (1 - pi0)·Beta(a, b)
- 같은 p를 갖는 독립 시행이라면 pass@S = 1 - (1 - p1)^S 이어야 하는데, 실제 곡선은 훨씬 느리게 오른다.
- 맞춘 분포로 (i) 점근선 1 - pi0, (ii) 조기 종료 시 기대 표본 수 E[min(S, T)]를 계산한다.
데이터: Table 1, 부록 D.4(360k), Table 11(ShapeFilter State), Table 12에서 역산한 N별 시험 성공률.
실행: python3 analysis/scaling_toy.py  (CPU, 단일 스레드, 수 초)
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import betaln

ROOT = Path(__file__).resolve().parents[1]

CURVES = {
    "HOT3D test MPC-failure (n=728)": ([1, 64, 128, 256], [6.87, 34.89, 38.05, 42.58]),
    "HOT3D test MPC-success (n=272)": ([1, 64, 128, 256], [48.16, 88.97, 90.07, 92.65]),
    "HOT3D 360k MPC-failure (D.4)": ([1, 64, 128, 256], [7.32, 38.19, 43.43, 48.01]),
    "HOT3D 360k MPC-success (D.4)": ([1, 64, 256], [43.22, 84.75, 89.30]),
    "ShapeFilter State (Table 11)": ([1, 2, 4, 8, 16, 32, 64, 128, 256],
                                     [50.42, 67.06, 78.44, 87.10, 92.36, 95.93, 97.45, 98.47, 99.32]),
    "ShapeFilter N=50 (Table 12)": ([1, 2, 4, 8, 16, 32, 64, 128, 256],
                                    [11.5, 15.8, 32.3, 46.5, 65.8, 82.8, 93.1, 96.6, 99.5]),
    "ShapeFilter N=2448 (Table 12)": ([1, 2, 4, 8, 16, 32, 64, 128, 256],
                                      [52.6, 68.0, 79.4, 87.1, 92.3, 95.4, 97.4, 98.5, 99.5]),
}


def pass_at(S, pi0, a, b):
    S = np.asarray(S, dtype=float)
    return (1 - pi0) * (1 - np.exp(betaln(a, b + S) - betaln(a, b)))


def expected_samples(budget, pi0, a, b):
    k = np.arange(budget, dtype=float)
    surv = pi0 + (1 - pi0) * np.exp(betaln(a, b + k) - betaln(a, b))  # P(처음 k개 모두 실패)
    return float(surv.sum())


def fit(S, sr):
    S, y = np.asarray(S, float), np.asarray(sr, float) / 100

    def res(t):
        pi0 = 1 / (1 + np.exp(-t[0]))
        return pass_at(S, pi0, np.exp(t[1]), np.exp(t[2])) - y

    best = None
    for t0 in ([-2, -1, 0], [0, -1, 1], [-4, 0, 2], [1, -2, 0], [-3, -0.5, 3]):
        r = least_squares(res, t0, bounds=([-12, -6, -6], [12, 4, 8]))
        if best is None or r.cost < best.cost:
            best = r
    t = best.x
    return 1 / (1 + np.exp(-t[0])), float(np.exp(t[1])), float(np.exp(t[2])), float(np.sqrt(np.mean(best.fun ** 2)) * 100)


def main() -> None:
    out = {}
    for name, (S, sr) in CURVES.items():
        pi0, a, b = 0.0, 0.0, 0.0
        pi0, a, b, rmse = fit(S, sr)
        p1 = sr[0] / 100
        iid = {s: round(100 * (1 - (1 - p1) ** s), 1) for s in (16, 64, 256)}
        e256 = expected_samples(256, pi0, a, b)
        mean_p = (1 - pi0) * a / (a + b)
        out[name] = {
            "pi0_never_solved": round(pi0, 3), "beta_a": round(a, 3), "beta_b": round(b, 2), "fit_rmse_pp": round(rmse, 2),
            "asymptote_pct": round(100 * (1 - pi0), 1),
            "mean_p_per_sample": round(mean_p, 3),
            "observed": dict(zip(S, sr)),
            "fitted": {s: round(100 * float(pass_at(s, pi0, a, b)), 1) for s in S},
            "iid_same_p_prediction": iid,
            "extrapolated_pass_at_1024": round(100 * float(pass_at(1024, pi0, a, b)), 1),
            "expected_samples_with_early_stop_S256": round(e256, 1),
            "early_stop_saving_vs_full_256_pct": round(100 * (1 - e256 / 256), 1),
        }
    # 360k MPC-failure: 논문 추정(354.1 core-s × 264,806, 조기 종료 없음) 대 조기 종료 기대 비용
    f = out["HOT3D 360k MPC-failure (D.4)"]
    per_sample = (354.1 - 48.9) / 255  # Table 1: S=1→S=256 증분을 표본 수로 나눈 근사 단가
    fixed = 48.9 - per_sample
    est_early = fixed + per_sample * f["expected_samples_with_early_stop_S256"]
    out["expansion_cost_estimate"] = {
        "per_sample_core_s_approx": round(per_sample, 3), "fixed_core_s_approx": round(fixed, 1),
        "paper_core_h_full_budget": round(264806 * 354.1 / 3600),
        "early_stop_core_h_estimate": round(264806 * est_early / 3600),
        "note": "Table 1의 S별 비용이 S에 거의 비례(1.37–1.42 core-s/표본)라 조기 종료 없이 잰 값으로 보고 근사",
    }
    # 실제 배치 시나리오(App. G.2): 시드 28,480 + 나머지 331,670개(MPC-failure 264,806 + 시드 밖 MPC-success 66,864)
    s = out["HOT3D 360k MPC-success (D.4)"]
    rest_succ = 331670 - 264806
    gnr_es = (264806 * est_early + rest_succ * (fixed + per_sample * s["expected_samples_with_early_stop_S256"])) / 3600
    out["expansion_cost_estimate"].update({
        "deploy_gnr_core_h_paper": 32623, "deploy_gnr_core_h_early_stop": round(gnr_es),
        "deploy_total_ratio_paper": round(390043 / (116112 + 32623), 2),
        "deploy_total_ratio_early_stop": round(390043 / (116112 + gnr_es), 2),
        "seed_share_of_paper_total_pct": round(100 * 116112 / (116112 + 32623), 1),
        "ratio_ceiling_if_gnr_free": round(390043 / 116112, 2),
    })
    path = ROOT / "analysis" / "scaling_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:420])
    print("wrote", path)


if __name__ == "__main__":
    main()
