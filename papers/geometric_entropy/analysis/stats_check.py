"""논문 그림에서 우리가 판독한 근사 값으로 주장을 검정한다 (CPU, numpy/scipy, 1스레드).

입력은 논문 본문과 Table I(정확한 값)와, Fig. 3–5에서 300dpi로 판독한 점들(±0.01–0.02 수준의 근사)이다.
논문은 시뮬레이션 평가 에피소드 수, 오차막대 정의(SD 또는 SE), 시드별 값을 보고하지 않는다.
그래서 아래 검정은 가정을 명시한 상한 근처의 판단용이다.
  - 시뮬 오차막대 반폭 ≈ 0.015를 표준오차로 가정(3 시드). 실물은 ≈ 0.05.
  - 실물 평가는 40회 × 3 시드 = 120회(가정 A), 또는 40회(가정 B)로 두 번 계산.

출력: analysis/stats_check_results.json
실행: python3 analysis/stats_check.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(12345)

# ---------- 1. 판독 데이터 (논문 Table I의 H^avg_G는 정확한 값, 나머지는 그림 판독) ----------
CONFIGS = ["0/0", "0.05/1", "0.01/0", "0.10/1", "0.03/0", "0.05/0", "0.10/0"]
H_STACK = np.array([2.901, 2.909, 2.917, 2.997, 3.016, 3.183, 3.667])   # Table I (정확)
R_STACK = np.array([0.0, 0.05, 0.01, 0.10, 0.03, 0.05, 0.10])
K_STACK = np.array([0, 1, 0, 1, 0, 0, 0])
DP_STACK = {  # Fig. 3a 판독, 행: N, 열: CONFIGS 순서
    100: [0.06, 0.065, 0.12, 0.215, 0.22, 0.325, 0.24],
    200: [0.16, 0.26, 0.285, 0.36, 0.375, 0.42, 0.27],
    500: [0.395, 0.42, 0.48, 0.575, 0.595, 0.58, 0.355],
    1000: [0.71, 0.735, 0.76, 0.72, 0.715, 0.635, 0.445],
}
SE_SIM = 0.015
PI05 = {  # Fig. 4 판독 (오차막대 없음). 구성: k=0 경우만, r = 0.01, 0.03, 0.05, 0.10
    "H": np.array([2.917, 3.016, 3.183, 3.667]),
    100: [0.905, 0.88, 0.875, 0.685],
    200: [0.932, 0.92, 0.93, 0.85],
    500: [0.97, 0.953, 0.947, 0.845],
    1000: [0.975, 0.94, 0.918, 0.86],
}
REAL = {  # Fig. 5 판독 (ACT, 실물). H, 성공률
    "StackCube": (np.array([3.00, 3.27, 3.50, 3.70, 3.94]), np.array([0.30, 0.62, 0.47, 0.37, 0.17])),
    "PlacePanda": (np.array([2.45, 2.53, 2.79, 3.27, 3.29]), np.array([0.87, 0.67, 0.47, 0.335, 0.20])),
    "OpenDrawer": (np.array([2.0, 4.0, 5.5, 5.9, 6.4]), np.array([0.93, 0.73, 0.40, 0.37, 0.23])),
}
SE_REAL = 0.05


def wls_fit(x, y, se, deg):
    """가중 최소제곱. 반환: 계수, RSS, 자유도."""
    X = np.vander(x - x.mean(), deg + 1, increasing=True)
    w = 1.0 / se
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    rss = float(np.sum(((y - X @ beta) * w) ** 2))
    return beta, rss, len(y) - (deg + 1)


def quad_vs_lin(x, y, se):
    """2차항 F검정(가중). 2차항 계수의 부호와 p값을 함께 낸다(역U면 음수)."""
    b1, rss1, df1 = wls_fit(x, y, se, 1)
    b2, rss2, df2 = wls_fit(x, y, se, 2)
    if df2 <= 0:
        return {"note": "자유도 부족"}
    F = (rss1 - rss2) / (rss2 / df2)
    p = 1 - stats.f.cdf(F, 1, df2)
    # 2차 계수 t값
    X = np.vander(x - x.mean(), 3, increasing=True)
    w = 1.0 / se
    cov = np.linalg.inv((X * w[:, None]).T @ (X * w[:, None])) * max(rss2 / df2, 1e-12)
    t = b2[2] / np.sqrt(cov[2, 2])
    vertex = x.mean() - b2[1] / (2 * b2[2]) if b2[2] != 0 else float("nan")
    return {"quad_coef": round(float(b2[2]), 3), "t_quad": round(float(t), 2), "F_quad_vs_lin": round(float(F), 3),
            "p_quad_vs_lin": round(float(p), 4), "df_resid": df2,
            "vertex_H": round(float(vertex), 3), "rss_lin": round(rss1, 4), "rss_quad": round(rss2, 4)}


def monotone_test(y):
    """경우의 수에 따른 완전 단조 순열 검정. 점이 n개면 단조 확률 2/n!."""
    n = len(y)
    tau, _ = stats.kendalltau(np.arange(n), y)
    return {"kendall_tau": round(float(tau), 3), "p_exact_two_sided_if_random": round(float(2 / math.factorial(n)), 4)}


def wilson(p, n, z=1.96):
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(float(centre - half), 3), round(float(centre + half), 3)


def fisher(p1, p2, n):
    a = round(p1 * n); b = round(p2 * n)
    tab = [[a, n - a], [b, n - b]]
    OR, p = stats.fisher_exact(tab)
    return {"counts": tab, "odds_ratio": round(float(OR), 3), "p_two_sided": float(p)}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return [round(float(a), 4) for a in adj]


def main():
    out = {"assumptions": {
        "sim_se": SE_SIM, "real_se": SE_REAL,
        "real_n_A": 120, "real_n_B": 40,
        "digitization": "300dpi 그림 판독, ±0.01–0.02(시뮬), ±0.01(실물 점 위치). 근사값이며 논문 원자료가 아님"}}

    # ---------- (a) Table I 순서와 H의 값 ----------
    out["A_table_I_order"] = {"H_sorted": [[c, float(h)] for c, h in sorted(zip(CONFIGS, H_STACK), key=lambda t: t[1])]}

    # ---------- (b) DP StackCube: 각 N의 최적점, argmax 불확실성, 2차 대 1차 ----------
    res_b = {}
    pvals_b = []
    for N, y in DP_STACK.items():
        y = np.array(y)
        j = int(np.argmax(y))
        second = int(np.argsort(y)[-2])
        z = (y[j] - y[second]) / (np.sqrt(2) * SE_SIM)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        pvals_b.append(p)
        # 판독 불확실성: 점마다 SE만큼 흔들어 argmax 확률
        sims = y[None, :] + rng.normal(0, SE_SIM, (20000, len(y)))
        argmax_prob = np.bincount(np.argmax(sims, axis=1), minlength=len(y)) / len(sims)
        quad = quad_vs_lin(H_STACK, y, np.full(len(y), SE_SIM))
        res_b[str(N)] = {
            "argmax_config": CONFIGS[j], "argmax_H": float(H_STACK[j]),
            "runner_up_config": CONFIGS[second], "z_top_vs_runner_up": round(float(z), 2),
            "p_top_vs_runner_up": round(float(p), 4),
            "P_argmax_under_noise": {CONFIGS[i]: round(float(argmax_prob[i]), 3) for i in range(len(y))},
            "quad_vs_linear": quad,
            "monotone_tail_after_peak": monotone_test(y[j:]) if len(y[j:]) >= 3 else "점 부족",
        }
    holm_b = holm(np.array(pvals_b))
    for (N, _), ha in zip(DP_STACK.items(), holm_b):
        res_b[str(N)]["holm_p_top_vs_runner_up"] = ha
    out["B_DP_StackCube_peak_and_curvature"] = res_b

    # ---------- (c) 피크 위치의 N 의존성: 2차 적합의 꼭짓점 H*, log10 N에 회귀 ----------
    Ns = np.array(list(DP_STACK.keys()), float)
    verts = []
    boot = []
    for N in DP_STACK:
        y = np.array(DP_STACK[N])
        _, rss, df = wls_fit(H_STACK, y, np.full(7, SE_SIM), 2)
        b, _, _ = wls_fit(H_STACK, y, np.full(7, SE_SIM), 2)
        xm = H_STACK.mean()
        v = xm - b[1] / (2 * b[2]) if b[2] != 0 else np.nan
        verts.append(v)
        # 꼭짓점 불확실성 부트스트랩(잡음 주입)
        vs = []
        for _ in range(2000):
            yy = y + rng.normal(0, SE_SIM, 7)
            bb, _, _ = wls_fit(H_STACK, yy, np.full(7, SE_SIM), 2)
            if bb[2] < 0:  # 역U만 채택
                vs.append(xm - bb[1] / (2 * bb[2]))
        boot.append(np.percentile(vs, [2.5, 97.5]).round(3).tolist() if len(vs) > 50 else "역U 아님")
    slope, intercept, r, p_slope, se_slope = stats.linregress(np.log10(Ns), verts)
    out["C_peak_H_vs_N"] = {
        "vertex_H_by_N": {str(int(N)): round(float(v), 3) for N, v in zip(Ns, verts)},
        "bootstrap_95CI_vertex_by_N": {str(int(N)): b for N, b in zip(Ns, boot)},
        "slope_per_decade_of_N": round(float(slope), 3), "se_slope": round(float(se_slope), 3),
        "p_slope": round(float(p_slope), 4), "r": round(float(r), 3),
        "note": "4개 점, 꼭짓점은 7개 격자 점의 2차 적합에서 나옴. 기울기 p값은 참고용."}

    # ---------- (d) H만으로 성공률을 설명하는가? r, k와 비교 (DP, N=1000, 7점) ----------
    y = np.array(DP_STACK[1000])
    se = np.full(7, SE_SIM)
    X_H = np.vander(H_STACK - H_STACK.mean(), 3, increasing=True)
    X_rk = np.column_stack([np.ones(7), R_STACK, R_STACK ** 2, K_STACK])
    def rss(X):
        w = 1 / se
        beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        return float(np.sum(((y - X @ beta) * w) ** 2)), beta
    rss_H, bH = rss(X_H)
    rss_rk, brk = rss(X_rk)
    out["D_H_vs_r_k_N1000"] = {
        "model_H_quadratic_RSS": round(rss_H, 5), "model_r_quadratic_plus_k_RSS": round(rss_rk, 5),
        "coef_r_quadratic_plus_k": [round(float(v), 3) for v in brk],
        "note": "k=0 경우 H는 r의 단조 함수(Table I). 두 모형의 차이는 k=1 두 점(0.05/1, 0.10/1)이 좌우한다. 7점, 자유도 3–4라 차이를 강하게 말할 수 없다."}

    # ---------- (e) π0.5 (Fig. 4, 오차막대 없음): 4점 ----------
    res_e = {}
    for N in [100, 200, 500, 1000]:
        y = np.array(PI05[N])
        res_e[str(N)] = {"monotone": monotone_test(y), "slope_per_H": round(float(np.polyfit(PI05["H"], y, 1)[0]), 3),
                         "drop_2.917_to_3.667": round(float(y[0] - y[-1]), 3)}
    out["E_pi05_StackCube_fig4"] = res_e

    # ---------- (f) 실물 ACT: 2차 검정, 단조성, 피크 대 최고·최저 쌍 ----------
    res_f = {}
    for task, (Hs, sr) in REAL.items():
        se_r = np.full(len(Hs), SE_REAL)
        res_f[task] = {"quad_vs_linear": quad_vs_lin(Hs, sr, se_r), "monotone": monotone_test(sr),
                       "Wilson95_nA_120": [wilson(p, 120) for p in sr], "Wilson95_nB_40": [wilson(p, 40) for p in sr]}
    out["F_real_ACT_curvature"] = res_f

    # 실물 쌍 비교 (Fisher, 가정 A: 120회, 가정 B: 40회) + Holm
    pairs = [
        ("StackCube 피크(3.27) vs 왼쪽 끝(3.00)", 0.62, 0.30),
        ("StackCube 피크(3.27) vs 오른쪽 이웃(3.50)", 0.62, 0.47),
        ("StackCube 피크(3.27) vs 최고 H(3.94)", 0.62, 0.17),
        ("PlacePanda 최고(2.45) vs 최고 H(3.29)", 0.87, 0.20),
        ("PlacePanda 3.27 vs 3.29 (거의 같은 H)", 0.335, 0.20),
        ("OpenDrawer 최고(2.0) vs 최고 H(6.4)", 0.93, 0.23),
        ("OpenDrawer 4.0 vs 5.5", 0.73, 0.40),
    ]
    rows = []
    pvA = []
    for name, p1, p2 in pairs:
        fA = fisher(p1, p2, 120)
        fB = fisher(p1, p2, 40)
        pvA.append(fA["p_two_sided"])
        rows.append({"pair": name, "A_120": {k: v for k, v in fA.items()}, "B_40": {k: v for k, v in fB.items()}})
    adj = holm(np.array(pvA))
    for row, a in zip(rows, adj):
        row["holm_p_A"] = a
    out["G_real_pairwise_fisher"] = rows

    # ---------- (g) CMH: 세 과제를 층으로, '최고 성공 대 최고 H' ----------
    from statsmodels.stats.contingency_tables import StratifiedTable
    strata = [(0.62, 0.17, 120), (0.87, 0.20, 120), (0.93, 0.23, 120)]  # 피크(또는 최고 성공)와 최고 H
    tabs = []
    for p1, p2, n in strata:
        a_ = round(p1 * n); b_ = round(p2 * n)
        tabs.append(np.array([[a_, n - a_], [b_, n - b_]]))
    st = StratifiedTable(tabs)
    cmh = st.test_null_odds(correction=True)
    out["H_CMH_best_vs_top_H_strata_tasks"] = {
        "MH_OR_best_over_top": round(float(st.oddsratio_pooled), 2),
        "OR_CI95": [round(float(v), 2) for v in st.oddsratio_pooled_confint()],
        "CMH_chi2": round(float(cmh.statistic), 2), "p": float(cmh.pvalue),
        "note": "층은 과제 3개(가정 A, 120회). 최고 성공 쌍 대 최고 H 쌍의 요약이며 H의 인과 효과를 뜻하지 않는다. "
                "동질성 검정 p는 StratifiedTable.test_equal_odds로 따로 계산했다."}
    out["H_CMH_best_vs_top_H_strata_tasks"]["homogeneity_p"] = float(st.test_equal_odds().pvalue)

    # ---------- (h) 논문 두 버전의 핵심 수치 대조는 텍스트 쪽에서 수행 ----------
    path = HERE / "stats_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float)[:2500])


if __name__ == "__main__":
    main()
