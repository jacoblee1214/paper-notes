"""토이: 데이터 규모 점 수와 평가 잡음이 로그-선형 적합과 100k 외삽을 얼마나 흔드는지.

중심 주장: "5개 점(1k–20k)에서 R^2 0.998의 로그-선형 적합은 포화 곡선과 구별되지 않으며, 100k 외삽이 그만큼 불확실하다."

두 참 곡선 (scaling_refit_results.json에서 읽음):
  T1: 로그-선형 L = a - b ln D  (재판독 적합)
  T2: 포화 L = L_inf + A D^-alpha, L_inf = 0 (음수가 아닌 점근값 중 재판독에 가장 잘 맞는 것)
각 참 곡선에서 관측 설계(점 3개, 5개, 9개)와 평가 잡음 sigma(0.0001, 0.0003, 0.0010)로 2000번 시뮬레이션.
측정: 로그-선형 적합의 R^2 >= 0.99 비율, 기울기 편향, 100k 외삽 상대 오차, 100k 예측구간의 참값 포함률,
그리고 2차항(곡률) 검정의 검출력(p < 0.05). 참 곡선이 T2일 때 곡률 검정이 얼마나 잡아내는지가 핵심.

실행: python3 analysis/toy_scaling_fit.py → analysis/toy_scaling_fit_results.json (numpy, 수십 초)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
REF = json.loads((HERE / "scaling_refit_results.json").read_text(encoding="utf-8"))
OUT = HERE / "toy_scaling_fit_results.json"

A_LL = REF["loglinear_fit"]["intercept"]
B_LL = -REF["loglinear_fit"]["slope"]
SAT = next(p for p in REF["saturating_profile"] if p["L_inf_fixed"] == 0.0)
A_SAT, ALPHA_SAT = SAT["A"], SAT["alpha"]

DESIGNS = {
    "3pts_1_4_20k": [1.0, 4.0, 20.0],
    "5pts_1_2_4_10_20k": [1.0, 2.0, 4.0, 10.0, 20.0],
    "9pts_log_1_20k": list(np.geomspace(1.0, 20.0, 9)),
}
SIGMAS = [0.0001, 0.0003, 0.0010]
REPS = 2000
D_TARGET = 100.0  # 천 시간 (100k 시간)
rng = np.random.default_rng(20260218)


def truth(name, d):
    if name == "T1_loglinear":
        return A_LL - B_LL * np.log(d)
    return A_SAT * d ** (-ALPHA_SAT)


def run_cell(tname, dname, sigma):
    d = np.array(DESIGNS[dname], dtype=float)
    x = np.log(d)
    n = len(d)
    mu = truth(tname, d)
    mu_t = truth(tname, np.array([D_TARGET]))[0]
    xt = np.log(D_TARGET)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    tcrit = stats.t.ppf(0.975, n - 2)
    # 2차항 검정용 설계
    X2 = np.column_stack([np.ones(n), x, x ** 2])
    XtX2_inv = np.linalg.inv(X2.T @ X2)

    y = mu[None, :] + rng.normal(0, sigma, size=(REPS, n))
    beta = (np.linalg.pinv(X) @ y.T).T  # (REPS, 2)
    fitted = beta @ X.T
    resid = y - fitted
    sse = (resid ** 2).sum(1)
    sst = ((y - y.mean(1, keepdims=True)) ** 2).sum(1)
    r2 = 1 - sse / sst
    s2 = sse / (n - 2)
    slope_true = -B_LL
    slope_err = beta[:, 1] - (slope_true if tname == "T1_loglinear" else np.polyfit(x, mu, 1)[0])
    pred_t = beta[:, 0] + beta[:, 1] * xt
    se_pred = np.sqrt(s2 * (1 + XtX_inv[0, 0] + 2 * XtX_inv[0, 1] * xt + XtX_inv[1, 1] * xt * xt))
    lo, hi = pred_t - tcrit * se_pred, pred_t + tcrit * se_pred
    covered = ((lo <= mu_t) & (mu_t <= hi)).mean()
    ext_rel_err = (pred_t - mu_t) / mu_t

    # 곡률 검정: 2차항 계수의 t 검정 (자유도 n-3). n=3이면 자유도 0이므로 생략
    if n >= 4:
        b2 = (np.linalg.pinv(X2) @ y.T).T
        sse2 = ((y - b2 @ X2.T) ** 2).sum(1)
        s2b = sse2 / (n - 3)
        se_quad = np.sqrt(s2b * XtX2_inv[2, 2])
        tq = b2[:, 2] / se_quad
        p_curv = 2 * stats.t.sf(np.abs(tq), n - 3)
        curv_power = float((p_curv < 0.05).mean())
    else:
        curv_power = None

    return {
        "truth": tname, "design": dname, "sigma": sigma,
        "frac_R2_ge_0.99": round(float((r2 >= 0.99).mean()), 3),
        "median_R2": round(float(np.median(r2)), 4),
        "slope_bias_median": float("%.2g" % np.median(slope_err)),
        "slope_sd": float("%.2g" % np.std(beta[:, 1])),
        "extrap_100k_rel_err_median": round(float(np.median(ext_rel_err)), 3),
        "extrap_100k_rel_err_p05_p95": [round(float(np.quantile(ext_rel_err, 0.05)), 3), round(float(np.quantile(ext_rel_err, 0.95)), 3)],
        "pi95_coverage_of_true_100k": round(float(covered), 3),
        "curvature_test_power_p05": None if curv_power is None else round(curv_power, 3),
    }


def main():
    rows = []
    for tname in ["T1_loglinear", "T2_saturating"]:
        for dname in DESIGNS:
            for s in SIGMAS:
                rows.append(run_cell(tname, dname, s))
    out = {
        "truths": {"T1_loglinear": {"a": A_LL, "b": B_LL, "source": "scaling_refit_results.json loglinear_fit"},
                   "T2_saturating": {"L_inf": 0.0, "A": A_SAT, "alpha": ALPHA_SAT, "source": "saturating_profile L_inf=0"}},
        "truth_100k_loss": {"T1": round(float(truth("T1_loglinear", np.array([D_TARGET]))[0]), 5),
                            "T2": round(float(truth("T2_saturating", np.array([D_TARGET]))[0]), 5)},
        "reps": REPS, "sigma_values": SIGMAS, "rows": rows,
        "note": ("잡음은 점마다 독립 정규. 실제 검증 손실의 잡음은 논문에 없음(2000 에피소드 x 20 시점 x 16 샘플 평균으로 "
                 "줄어든 값이라 작을 가능성이 있으나 확인 불가). 시드 간 변동은 넣지 않음. "
                 "T2의 참 곡선은 재판독 데이터에 RMS 약 0.0003으로 맞는 것 중 하나이며, 가장 나은 적합은 아님"),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
    print("truth 100k:", out["truth_100k_loss"])
    for r in rows:
        print(r["truth"][:2], r["design"][:6], r["sigma"], "r2>=0.99:", r["frac_R2_ge_0.99"],
              "ext_err_med:", r["extrap_100k_rel_err_median"], "cov:", r["pi95_coverage_of_true_100k"],
              "curv_pow:", r["curvature_test_power_p05"])


if __name__ == "__main__":
    main()
