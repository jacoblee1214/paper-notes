"""논문 표 수치의 일관성 재계산 (arXiv:2611.17775 v1).

재현이 아니다. 논문이 보고한 표 값만으로 (1) 집계식이 맞는지, (2) 표끼리 맞는지,
(3) 보고된 구간과 검정이 보고된 개수로 다시 나오는지 확인한다.
원자료(에피소드별 점수, 잠재 캐시)는 논문이 공개하지 않으므로 bootstrap 자체는 다시 하지 않는다.
한 가지 합성 데이터 검사(절 7–8)만 numpy로 한다. 실행: python3 analysis/recompute.py (수 초)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
res = {}


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# 1. 표 4 (목표별, 5 folds) -> 표 2 (ALOHA, 10 pairs)
#    GeoLASR over 10 pairs = sqrt(Geo_act * Geo_delta), WorstLASR = max over targets of max over folds.
t4 = {  # (Geo_action, Worst_action, Geo_delta, Worst_delta)
    "DiLA": (0.735, 0.759, 0.767, 0.796),
    "LaWAM": (0.887, 0.934, 0.895, 0.947),
    "LAPA": (0.921, 0.966, 0.923, 0.956),
    "UniVLA": (1.011, 1.077, 1.007, 1.066),
    "MVP-LAM": (1.098, 1.157, 1.092, 1.137),
}
t2_aloha = {"DiLA": (0.750, 0.796), "LaWAM": (0.891, 0.947), "LAPA": (0.922, 0.966),
            "UniVLA": (1.009, 1.077), "MVP-LAM": (1.095, 1.157)}
rows = {}
for m, (ga, wa, gd, wd) in t4.items():
    g = math.sqrt(ga * gd)
    w = max(wa, wd)
    rows[m] = {"geo_from_t4": round(g, 4), "worst_from_t4": w, "geo_t2": t2_aloha[m][0], "worst_t2": t2_aloha[m][1],
               "ok": abs(g - t2_aloha[m][0]) <= 0.0015 and abs(w - t2_aloha[m][1]) <= 0.0015}
res["check1_table4_to_table2"] = rows

# 2. 표 5 (두 반쪽) : 고정된 예측기에서 전체 위험은 두 반쪽의 가중 평균이므로
#    전체 비율 <= 두 반쪽 비율 중 큰 값 (분모 min은 오목 함수). 두 반쪽 사이 또는 아래여야 한다.
t5 = {  # ord_geo, enr_geo, ord_worst, enr_worst ; overall from t2
    "DiLA": (0.750, 0.752, 0.803, 0.840, 0.750, 0.796),
    "LaWAM": (0.882, 0.897, 0.908, 0.981, 0.891, 0.947),
    "LAPA": (0.920, 0.924, 0.942, 0.999, 0.922, 0.966),
    "UniVLA": (1.014, 1.005, 1.128, 1.044, 1.009, 1.077),
    "MVP-LAM": (1.090, 1.100, 1.223, 1.244, 1.095, 1.157),
}
res["check2_table5_bound"] = {
    m: {"geo_pooled_le_max_half": g <= max(o, e) + 1e-9, "worst_pooled_le_max_half": w <= max(ow, ew) + 1e-9}
    for m, (o, e, ow, ew, g, w) in t5.items()
}

# 3. 표 3(b) 와 표 7 (같은 값이어야 함)
t3b = {"Identity": (0.750, 0.796), "LASR residual": (0.745, 0.785), "MSE residual": (0.759, 0.810),
       "Random residual": (0.762, 0.815), "Shuffled residual": (0.761, 0.834)}
t7 = {"Identity": (0.750, 0.796), "LASR residual": (0.745, 0.785), "MSE residual": (0.759, 0.810),
      "Random residual": (0.762, 0.815), "Shuffled residual": (0.761, 0.834)}
res["check3_table3b_vs_table7_equal"] = t3b == t7

# 4. 표 8 (쌍체 차이) 와 표 3(b) 점추정 차이 (반올림 0.0005 이내면 일치)
t8 = {  # comparator: (dGeo, dWorst) paired, adapter = LASR residual
    "Raw/Identity": (-0.00563, -0.00488),
    "Random residual": (-0.01804, -0.01713),
    "Shuffled residual": (-0.01617, -0.01456),
    "MSE residual": (-0.01466, -0.01404),
}
ad = t3b["LASR residual"]
diff = {}
for k, key in [("Raw/Identity", "Identity"), ("Random residual", "Random residual"),
               ("Shuffled residual", "Shuffled residual"), ("MSE residual", "MSE residual")]:
    dg = ad[0] - t3b[key][0]
    dw = ad[1] - t3b[key][1]
    diff[k] = {"geo_point_diff": round(dg, 4), "geo_paired": t8[k][0],
               "worst_point_diff": round(dw, 4), "worst_paired": t8[k][1],
               "worst_ratio_paired_over_point": round(t8[k][1] / dw, 2)}
res["check4_table8_vs_table3b"] = diff
res["check4b_identity_vs_raw_ci"] = {
    "raw_geo_ci": [-0.01852, 0.00713], "identity_geo_ci": [-0.01843, 0.00745],
    "note": "동일 표현이어야 하므로 CI가 같아야 하지만 다르다 (미확인)"}

# 5. 표 10 (matched panel, 부트스트랩 CI) 의 순위 겹침
ci = {"DiLA native": (0.688, 0.668, 0.708), "LAPA": (0.831, 0.810, 0.851), "LaWAM": (0.860, 0.838, 0.883),
      "MVP-LAM": (0.988, 0.981, 0.995), "UniVLA": (1.010, 0.998, 1.025)}
pairs = {}
names = list(ci)
for i in range(len(names) - 1):
    a, b = names[i], names[i + 1]
    overlap = ci[a][2] >= ci[b][1]
    pairs[f"{a} vs {b}"] = {"point_order_ok": ci[a][0] < ci[b][0], "ci_overlap": bool(overlap)}
res["check5_matched_ci_overlap"] = pairs

# 6. 표 11 (과제별 Geo) 평균 vs 표 10 pooled Geo (대략 일치해야 함)
t11 = {"DiLA": [0.632, 0.596, 0.597, 0.607, 1.015], "LAPA": [0.737, 0.735, 0.729, 0.749, 1.214],
       "LaWAM": [0.769, 0.734, 0.739, 0.763, 1.312], "MVP-LAM": [0.967, 0.970, 0.975, 0.994, 1.036],
       "UniVLA": [1.042, 1.008, 1.007, 0.998, 0.992]}
pooled = {"DiLA": 0.688, "LAPA": 0.831, "LaWAM": 0.860, "MVP-LAM": 0.988, "UniVLA": 1.010}
res["check6_task_mean_vs_pooled"] = {
    m: {"task_geo_mean": round(geo(v), 3), "pooled": pooled[m], "gap": round(geo(v) - pooled[m], 3)}
    for m, v in t11.items()}
res["check6b_peg_insertion_gt1"] = {m: v[4] > 1 for m, v in t11.items()}

# 7. 보고된 Wilson 95% 구간 (MVP-LAM 공식 실행 95/100)
n, x, z = 100, 95, 1.96
p = x / n
center = (p + z * z / (2 * n)) / (1 + z * z / n)
half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
res["check7_wilson_mvp"] = {"recomputed": [round(center - half, 4), round(center + half, 4)],
                            "reported": [0.8882, 0.9785]}

# 8. 표 14 차이 (점추정)
res["check8_table14_delta"] = {"camera": round(0.7625 - 0.7444, 4), "outcome": round(0.9888 - 0.7442, 4),
                               "reported_camera": 0.0181, "reported_outcome": 0.2446}

# 9. 스피어만 순위: 표 10의 5 모델 GeoLASR 순서와 표 2 분기 정확도 순서 (matched panel)
geo_order = sorted(pooled, key=lambda m: pooled[m])
branch = {"DiLA": 65.58, "LAPA": 50.39, "LaWAM": 48.59, "MVP-LAM": 26.75, "UniVLA": 25.44}
branch_order = sorted(branch, key=lambda m: -branch[m])
res["check9_rank_match_5models"] = {
    "geo_order": geo_order, "branch_order": branch_order, "identical": geo_order == branch_order,
    "prob_perfect_by_chance_one_sided": 1 / math.factorial(5)}

# 10. 보고된 스피어만 ρ 의 Fisher z 근사 구간과 보고된 구간 비교 (n: 25 cells, 50 block)
def fisher(r, n):
    z0 = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return [round(math.tanh(z0 - 1.96 * se), 3), round(math.tanh(z0 + 1.96 * se), 3)]

res["check10_spearman_ci"] = {
    "H4_TCP_mse_rho0.956_n25": {"fisher": fisher(0.956, 25), "reported": [0.916, 0.993]},
    "H4_branch_rho0.810_n25": {"fisher": fisher(0.810, 25), "reported": [0.270, 0.987]},
    "H5_native_rho-0.161_n50": {"fisher": fisher(-0.161, 50), "reported": [-0.6048, 0.3930]},
}
t_stat = 0.956 * math.sqrt(23 / (1 - 0.956 ** 2))
res["check10_spearman_p_TCP"] = float(2 * stats.t.sf(t_stat, 23))

# 11. 분할 크기와 에피소드 수
test_counts = [420, 418, 414, 420, 376]
ep_per_fold = [len([e for e in range(49) if e % 5 == f]) for f in range(5)]
res["check11_splits"] = {"sum_test": sum(test_counts), "reported_panel": 2048,
                         "episodes_per_fold": ep_per_fold, "eligible_transitions": 49 * 585,
                         "reported_eligible": 28665}

# 12. 기록된 행동 기준(표 2 Recorded action, 표 3 주석 1.19e-7)의 원인: 합성 데이터로 메커니즘만 확인
rng = np.random.default_rng(0)
n = 4000
S = rng.normal(size=(n, 14))
A = S @ rng.normal(size=(14, 14)) * 0.3 + rng.normal(size=(n, 14)) * 0.5
Y_act = A.copy()                     # 목표 = 기록된 행동 자체
Y_delta = 0.5 * A + rng.normal(size=(n, 14))


def ridge_mse(X, Y, lam=1e-6):
    """절편 비정규화 ridge (계수 1e-6, 논문 부록 C와 같은 설정). 학습 = 평가 데이터(검사용)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc)
    return float(np.mean((Yc - Xc @ W) ** 2))


for tag, Y in [("action_target", Y_act), ("delta_target", Y_delta)]:
    e_state = ridge_mse(S, Y)
    e_act = ridge_mse(np.hstack([S, A]), Y)
    res.setdefault("check12_recorded_action_mechanism", {})[tag] = {
        "E_state": round(e_state, 4), "E_action_route": e_act, "ratio_to_state": e_act / e_state}

res["check13_note"] = "action_target 비율이 0에 가까운 것은 Y=A를 입력으로 넣은 항등 매핑 때문이다 (논문 식 (6)의 목표)."

# 14. 추론 가정 확인: 선형 가우시안 채널 (부록 B.5, Corollary 8). 닫힌 식과 몬테카를로 비교
sA2, sE2, gam, sv2 = 1.0, 0.5, 0.7, 0.2
var_A_given_Z = sA2 * sE2 / (sA2 + sE2)
R_B = gam ** 2 * sA2 + sv2
R_BZ = sv2 + gam ** 2 * var_A_given_Z
R_BA = sv2
Q_closed = (R_B - R_BZ) / (R_B - R_BA)
Q_formula = sA2 / (sA2 + sE2)
res["check14_gaussian_Q"] = {"Q_closed_from_risks": round(Q_closed, 6),
                             "Q_formula_sA2_over_sA2_plus_sE2": round(Q_formula, 6)}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
