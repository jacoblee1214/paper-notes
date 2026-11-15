"""그림 5 중앙 패널(최적 검증 손실 vs 데이터 규모)을 다시 맞춘다.

논문은 L = 0.024 - 0.003 ln(D), R^2 = 0.9983 (D: 시간)이라고 쓴다. 표가 없으므로 그림 5 중앙 패널의
마커 중심을 픽셀에서 읽었다(300 dpi 래스터, y축 눈금 0.014–0.026 기준 선형 보정, 판독 오차 약 ±0.0001).
오른쪽 패널의 평균 과제 완료 점수는 막대 라벨을 그대로 쓴다.

확인하는 것: (1) 로그-선형 적합의 기울기, 절편, R^2, 기울기 95% 구간, (2) 논문 식과 판독값의 차이,
(3) 대안 곡선(거듭제곱 법칙, 포화 곡선 L = L_inf + A D^-alpha, 2차 로그식)과 AICc,
(4) 외삽(1–10k로 맞추고 20k 예측, 100k와 시간 축 외삽), (5) 검증 손실과 과제 점수의 상관(n=5).

실행: python3 analysis/scaling_refit.py → analysis/scaling_refit_results.json (CPU, 수 초)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

OUT = Path(__file__).with_name("scaling_refit_results.json")

# 픽셀 판독 (fig5 중앙 패널, 300 dpi 래스터). y: 마커 중심, x: 1k 882 px, 20k 1303 px, ln 간격 140.5 px
Y_TICK = {0.026: 50.5, 0.014: 468.5}  # 눈금 두 개로 선형 보정 (0.024, 0.022, ... 도 같은 간격)
MARKERS_Y = {1: 109.8, 2: 205.7, 4: 279.5, 10: 382.1, 20: 472.9}
D_K = np.array([1, 2, 4, 10, 20], dtype=float)  # 단위: 천 시간
SCORE = np.array([0.30, 0.45, 0.48, 0.57, 0.71])  # 그림 5 오른쪽 막대 라벨


def y_to_loss(y):
    slope = (0.014 - 0.026) / (Y_TICK[0.014] - Y_TICK[0.026])
    return 0.026 + (y - Y_TICK[0.026]) * slope


L = np.array([y_to_loss(MARKERS_Y[int(d)]) for d in D_K])
LN_HOURS = np.log(D_K * 1000.0)
LN = np.log(D_K)  # ln(천 시간)


def ols(x, y):
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    dof = n - 2
    s2 = sse / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    t = stats.t.ppf(0.975, dof)
    return {
        "intercept": float(beta[0]), "slope": float(beta[1]),
        "intercept_se": float(math.sqrt(cov[0, 0])), "slope_se": float(math.sqrt(cov[1, 1])),
        "slope_ci95": [float(beta[1] - t * math.sqrt(cov[1, 1])), float(beta[1] + t * math.sqrt(cov[1, 1]))],
        "r2": 1 - sse / sst, "sse": sse, "residuals": resid.tolist(), "dof": dof,
    }


def aicc(sse, n, k):
    aic = n * math.log(sse / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def saturating(d, linf, a, alpha):
    return linf + a * d ** (-alpha)


def fit_saturating(d, y):
    best = None
    for alpha0 in [0.05, 0.2, 0.5, 1.0]:
        for linf0 in [0.0, 0.005, 0.01, 0.012]:
            try:
                p, _ = optimize.curve_fit(saturating, d, y, p0=[linf0, 0.03, alpha0],
                                          bounds=([-0.05, 0.0, 0.0], [0.02, 1.0, 5.0]), maxfev=20000)
            except (RuntimeError, ValueError):
                continue
            sse = float(((y - saturating(d, *p)) ** 2).sum())
            if best is None or sse < best[1]:
                best = (p, sse)
    p, sse = best
    return {"L_inf": float(p[0]), "A": float(p[1]), "alpha": float(p[2]), "sse": sse,
            "r2": 1 - sse / float(((y - y.mean()) ** 2).sum()), "aicc": aicc(sse, len(y), 3)}


def main():
    res = {"digitized": {"D_thousand_hours": D_K.tolist(), "loss": [round(float(v), 5) for v in L],
                         "marker_y_px": MARKERS_Y, "tick_calibration": Y_TICK,
                         "note": "판독 오차 약 ±0.0001. 그림 5 중앙 패널만 사용"}}

    # (1) 논문 식 검사: D를 시간으로 쓰면 값이 0.003 ln(D) ~ 0.02 이상이 되어 말이 안 됨. 천 시간으로 읽으면 맞음
    paper_formula_hours = 0.024 - 0.003 * LN_HOURS
    paper_formula_kh = 0.024 - 0.003 * LN
    res["paper_formula"] = {
        "as_written_D_hours_pred": [round(float(v), 5) for v in paper_formula_hours],
        "D_in_thousand_hours_pred": [round(float(v), 5) for v in paper_formula_kh],
        "residual_kh_(digitized_minus_pred)": [round(float(v), 5) for v in (L - paper_formula_kh)],
        "rmse_kh": float(np.sqrt(np.mean((L - paper_formula_kh) ** 2))),
        "r2_kh": float(1 - ((L - paper_formula_kh) ** 2).sum() / ((L - L.mean()) ** 2).sum()),
        "note": "D를 시간으로 쓰면 ln(1000)=6.9가 곱해져 L이 0.003 근처가 됨. 천 시간 단위로 읽어야 1k 점과 맞음. 그래도 20k 점은 약 0.001 과대 예측",
    }

    # (2) 로그-선형 적합 (천 시간 단위; 기울기는 단위 변환과 무관하게 같음)
    lin = ols(LN, L)
    lin["slope_ci95"] = [float(v) for v in lin["slope_ci95"]]
    res["loglinear_fit"] = {k: (v if not isinstance(v, float) else round(v, 6)) for k, v in lin.items() if k != "residuals"}
    res["loglinear_fit"]["residuals"] = [round(float(v), 6) for v in lin["residuals"]]
    res["loglinear_fit"]["paper_reports"] = {"intercept": 0.024, "slope": -0.003, "r2": 0.9983}
    res["loglinear_fit"]["r2_paper_vs_refit_note"] = "논문 R^2 0.9983은 재판독 값에서 재현되는지 위 r2로 확인. 기울기는 재판독 기준 약 -0.0035"

    # (3) 대안 모델 (같은 5점)
    pw = ols(LN, np.log(L))  # 거듭제곱 법칙 L = c D^(-b)
    pw_sse_L = float(((L - np.exp(pw["intercept"] + pw["slope"] * LN)) ** 2).sum())
    quad = np.polyfit(LN, L, 2)
    quad_sse = float(((L - np.polyval(quad, LN)) ** 2).sum())
    sat = fit_saturating(D_K, L)
    sst = float(((L - L.mean()) ** 2).sum())
    models = {
        "loglinear_2p": {"sse": lin["sse"], "r2": lin["r2"], "aicc": aicc(lin["sse"], 5, 2)},
        "power_law_2p": {"sse": pw_sse_L, "r2": 1 - pw_sse_L / sst, "aicc": aicc(pw_sse_L, 5, 2),
                         "exponent_b": float(-pw["slope"])},
        "quadratic_in_lnD_3p": {"sse": quad_sse, "r2": 1 - quad_sse / sst, "aicc": aicc(quad_sse, 5, 3),
                                "coef": [float(c) for c in quad]},
        "saturating_3p": {"sse": sat["sse"], "r2": sat["r2"], "aicc": sat["aicc"], **{k: sat[k] for k in ["L_inf", "A", "alpha"]}},
    }
    best_aicc = min(m["aicc"] for m in models.values())
    for m in models.values():
        m["delta_aicc"] = round(m["aicc"] - best_aicc, 3)
    res["model_comparison"] = models

    # (3a) 곡률 검정 (2차항 t 검정, 자유도 2)과 잔차 표준편차
    X2 = np.column_stack([np.ones(5), LN, LN ** 2])
    b2 = np.linalg.lstsq(X2, L, rcond=None)[0]
    res2 = L - X2 @ b2
    s2q = float(res2 @ res2) / 2
    se_q = math.sqrt(s2q * np.linalg.inv(X2.T @ X2)[2, 2])
    tq = float(b2[2] / se_q)
    res["curvature_test_5pts"] = {
        "quad_coef": float(b2[2]), "t": round(tq, 3), "p": round(2 * stats.t.sf(abs(tq), 2), 3),
        "loglinear_residual_sd": round(math.sqrt(lin["sse"] / lin["dof"]), 6),
        "note": "p가 크다는 것은 곡률이 검출되지 않았다는 뜻일 뿐, 직선임을 입증하지 않음. 토이(toy_scaling_fit)에서 5점 검정의 검출력은 잡음에 따라 0.07–0.8",
    }

    # (3b) 포화 곡선 프로파일: 점근값 L_inf를 고정하고 A, alpha를 맞춘다. 같은 SSE에서 외삽이 얼마나 갈리는지
    profile = []
    for linf in [-0.05, 0.0, 0.005, 0.010, 0.012, 0.0135]:
        f = lambda d, a, al: linf + a * d ** (-al)
        best = None
        for al0 in [0.05, 0.3, 1.0, 2.0]:
            try:
                p, _ = optimize.curve_fit(f, D_K, L, p0=[0.03, al0], bounds=([0, 0], [10, 10]), maxfev=20000)
            except (RuntimeError, ValueError):
                continue
            sse_p = float(((L - f(D_K, *p)) ** 2).sum())
            if best is None or sse_p < best[1]:
                best = (p, sse_p)
        if best is None:
            continue
        p, sse_p = best
        profile.append({"L_inf_fixed": linf, "A": round(float(p[0]), 5), "alpha": round(float(p[1]), 4),
                        "sse": float("%.3g" % sse_p), "delta_sse_vs_loglinear": float("%.3g" % (sse_p - lin["sse"])),
                        "pred_100k": round(float(f(np.array([100.0]), *p)[0]), 5),
                        "pred_1000k": round(float(f(np.array([1000.0]), *p)[0]), 5)})
    res["saturating_profile"] = profile
    res["saturating_profile_note"] = ("L_inf를 고정해도 SSE 차이가 작으면 여러 점근값이 같은 데이터를 설명한다. "
                                      "이때 100k 외삽은 점근값에 따라 달라진다. 음수 점근값은 물리적으로 불가")

    # (4) 외삽. 로그-선형, 거듭제곱, 포화 모델이 같은 데이터에서 다른 값을 내는지
    grid = np.array([40, 100, 200], dtype=float)  # 천 시간
    ll_pred = lin["intercept"] + lin["slope"] * np.log(grid)
    pw_pred = np.exp(pw["intercept"] + pw["slope"] * np.log(grid))
    sat_pred = saturating(grid, sat["L_inf"], sat["A"], sat["alpha"])
    res["extrapolation_all5"] = {
        "D_thousand_hours": grid.tolist(),
        "loglinear": [round(float(v), 5) for v in ll_pred],
        "power_law": [round(float(v), 5) for v in pw_pred],
        "saturating": [round(float(v), 5) for v in sat_pred],
        "loglinear_predicts_zero_or_negative_at": float(np.exp(-lin["intercept"] / lin["slope"])) if lin["slope"] < 0 else None,
        "note": "천 시간 단위. 로그-선형 식은 절편/기울기 비에 해당하는 규모에서 0이 됨(외삽의 물리적 한계)",
    }

    # (5) 홀드아웃: 1–10k로 맞추고 20k 예측 (로그-선형)
    fit3 = ols(LN[:4], L[:4])
    pred20 = fit3["intercept"] + fit3["slope"] * LN[4]
    res["holdout_fit_1to10k_predict_20k"] = {
        "pred": round(float(pred20), 5), "observed": round(float(L[4]), 5),
        "error": round(float(L[4] - pred20), 5),
        "slope_1to10k": round(fit3["slope"], 5), "slope_all5": round(lin["slope"], 5),
    }
    # 1k와 2k만으로 맞추고 20k 예측 (두 점으로 직선 = 정확히 맞음, 검증 불가이지만 외삽 거리 보여줌)
    slope2 = (L[1] - L[0]) / (LN[1] - LN[0])
    res["holdout_fit_1to2k_predict_20k"] = {
        "pred": round(float(L[0] + slope2 * (LN[4] - LN[0])), 5),
        "slope": round(float(slope2), 5),
        "note": "두 점은 직선을 정확히 정의하므로 R^2 검사는 불가",
    }

    # (6) 검증 손실과 과제 점수의 상관 (n=5). 상관계수는 논문이 보고하지 않음
    r, p = stats.pearsonr(L, SCORE)
    rho, p_s = stats.spearmanr(L, SCORE)
    z = np.arctanh(r)
    se = 1 / math.sqrt(len(L) - 3)
    zc = stats.norm.ppf(0.975)
    res["loss_vs_score"] = {
        "pearson_r": round(float(r), 4), "p": float("%.3g" % p),
        "pearson_ci95_fisher": [round(float(np.tanh(z - zc * se)), 3), round(float(np.tanh(z + zc * se)), 3)],
        "spearman_rho": round(float(rho), 3), "spearman_p": float("%.3g" % p_s),
        "r2_linear": round(float(r * r), 4),
        "note": "n=5. 점 수가 적어 상관 구간이 넓음. 1k 점 하나를 빼면 달라지는지 아래 확인",
    }
    r4, p4 = stats.pearsonr(L[1:], SCORE[1:])
    # ln D를 통제한 부분 상관: 두 변수가 모두 D에 단조이므로 원상관은 데이터 규모의 공통 추세일 수 있음
    rl = L - np.polyval(np.polyfit(LN, L, 1), LN)
    rs = SCORE - np.polyval(np.polyfit(LN, SCORE, 1), LN)
    partial_r = float(np.corrcoef(rl, rs)[0, 1])
    res["loss_vs_score"]["partial_r_given_lnD"] = round(partial_r, 4)
    res["loss_vs_score"]["partial_r_dof"] = 2
    res["loss_vs_score"]["pearson_r_without_1k"] = round(float(r4), 4)
    res["loss_vs_score"]["p_without_1k"] = round(float(p4), 4)

    # (7) 최적 손실의 정의: 곡선 최솟값(체크포인트 선택은 검증 세트). 10k, 20k는 100k 스텝 끝까지 감소
    res["selection_note"] = ("'최적 검증 손실'은 학습 곡선의 최솟값이다(그림 5 왼쪽). 같은 검증 세트로 체크포인트를 고르므로 "
                             "값이 낙관적이다. 10k와 20k 곡선은 100k 스텝까지 줄어들어 수렴하지 않았다. 데이터 규모 비교는 같은 스텝 예산에서 이루어짐")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
    print("digitized L:", [round(float(v), 5) for v in L])
    print("loglinear slope %.5f  intercept %.5f  r2 %.4f  slope95 %s" % (lin["slope"], lin["intercept"], lin["r2"], np.round(lin["slope_ci95"], 5)))
    print("paper formula (kh) rmse %.5f r2 %.4f" % (res["paper_formula"]["rmse_kh"], res["paper_formula"]["r2_kh"]))
    for k, v in models.items():
        print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'coef'})
    print("extrap", res["extrapolation_all5"])
    print("holdout", res["holdout_fit_1to10k_predict_20k"])
    print("loss_vs_score", res["loss_vs_score"])


if __name__ == "__main__":
    main()
