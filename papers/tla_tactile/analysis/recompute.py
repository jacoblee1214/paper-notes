"""TLA (arXiv:2503.08548 v1) 표 수치 재계산과 검정 (표준 라이브러리만 사용, 수 초).

입력: 논문 v1 표 I–IV 값 (paper/paper.txt의 pdftotext 결과와 대조해 옮김).
출력: analysis/recompute_results.json

가정(논문에 없는 값은 가정으로 표시):
- 표 III·IV의 Suc는 50회 시행의 성공률(%)로 본다. 본문은 "각 과제 50회 반복"이라고 쓴다.
- 표 I의 GCR 검정 표본 수는 단일 페그 시험셋 2k(본문 §IV-B), 표 II의 ID GCR은 시험 4k로 가정한다.
  논문은 이 표본 수를 표에 적지 않는다.
- 검정은 단일 시드 가정이다. 학습 시드 간 분산은 논문에 없어 반영하지 않는다.
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "recompute_results.json"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher_two_sided(a, b, c, d):
    """2x2 표 [[a,b],[c,d]]의 양측 Fisher 정확 검정 p값 (초기하분포)."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2

    def pmf(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)

    lo, hi = max(0, r1 - (n - c1)), min(r1, c1)
    p_obs = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-7)))


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def pct_to_count(pct, n):
    """성공률(%)을 n회 시행의 횟수로 바꾼다. 정수가 아니면 None."""
    k = pct * n / 100
    return int(round(k)) if abs(k - round(k)) < 1e-9 else None


def compare(label, pct1, pct2, n):
    """두 방법의 성공률 비교: 횟수가 정수면 Fisher, 아니면 pooled z 근사."""
    k1, k2 = pct_to_count(pct1, n), pct_to_count(pct2, n)
    z, pz = two_prop_z(pct1 * n / 100, n, pct2 * n / 100, n)
    row = {"label": label, "pct1": pct1, "pct2": pct2, "n_per_arm": n,
           "diff_pp": round(pct1 - pct2, 1), "z_pooled": round(z, 2), "p_pooled_z": float("%.3g" % pz)}
    if k1 is not None and k2 is not None:
        row["p_fisher"] = float("%.3g" % fisher_two_sided(k1, n - k1, k2, n - k2))
    else:
        row["p_fisher"] = None
        row["note"] = "성공 횟수가 정수가 아님 (값이 n=%d의 %%로 맞지 않음), z 근사만" % n
    return row


results = {}

# ---- V1: 표 III "Total"이 세 간격 평균과 같은지 (pooled 성공률과 같음) ----
T3 = {  # Suc (%) 2.0, 1.6, 1.0 mm, Total, 그리고 Step
    "BC": {"suc": [44, 32, 18], "total": 31, "step": [2.60, 4.00, 4.44], "step_total": 3.68},
    "DP": {"suc": [58, 43, 20], "total": 40, "step": [2.45, 4.35, 4.70], "step_total": 3.83},
    "SP-TLA": {"suc": [96, 94, 74], "total": 88, "step": [3.15, 3.77, 4.37], "step_total": 3.76},
    "MP-TLA": {"suc": [94, 86, 90], "total": 90, "step": [3.04, 3.30, 4.35], "step_total": 3.56},
}
results["V1_table3_total"] = {}
for m, d in T3.items():
    mean = sum(d["suc"]) / 3
    results["V1_table3_total"][m] = {"mean_of_3": round(mean, 2), "reported_total": d["total"],
                                     "match_rounding": round(mean) == d["total"]}

# ---- V2: 표 III·IV 성공률의 정수성 (n=50이면 짝수여야 함) ----
T3_all = [v for d in T3.values() for v in d["suc"]]
T4 = {  # ID: Square, Triangle / OOD: Round, Hexagon. Suc (%), Step
    "BC": {"suc": [60, 84, 62, 66], "step": [3.37, 2.67, 2.84, 2.30]},
    "DP": {"suc": [60, 60, 56, 54], "step": [3.60, 3.40, 2.89, 2.37]},
    "SP-TLA": {"suc": [96, 76, 92, 90], "step": [3.15, 2.82, 2.80, 2.78]},
    "MP-TLA": {"suc": [94, 92, 94, 96], "step": [3.04, 1.80, 2.66, 3.00]},
}
T4_all = [v for d in T4.values() for v in d["suc"]]
results["V2_integrality_n50"] = {
    "table3_odd_values": [v for v in T3_all if (v * 50 / 100) % 1 != 0],
    "table4_odd_values": [v for v in T4_all if (v * 50 / 100) % 1 != 0],
}

# ---- V3: 표 I의 L1 x 감소율 (본문 "78%") ----
L1x = {"BC": 0.803, "DP": 0.370, "SP-TLA": 0.079}
L1y = {"BC": 0.302, "DP": 0.382, "SP-TLA": 0.122}
L1rz = {"BC": 0.205, "DP": 0.568, "SP-TLA": 0.173}
best_other_x = min(L1x["BC"], L1x["DP"])
results["V3_table1_L1"] = {
    "x_reduction_vs_second_best_pct": round(100 * (1 - L1x["SP-TLA"] / best_other_x), 1),
    "second_best_x": "DP 0.370",
    "y_reduction_vs_best_other_pct": round(100 * (1 - L1y["SP-TLA"] / min(L1y["BC"], L1y["DP"])), 1),
    "rz_reduction_vs_best_other_pct": round(100 * (1 - L1rz["SP-TLA"] / min(L1rz["BC"], L1rz["DP"])), 1),
}

# ---- V4: 표 I GCR 차이 (표본 2k 가정) ----
gcr1 = {"BC": 10.4, "DP": 8.5, "SP-TLA": 12.5}
n_gcr1 = 2000
results["V4_table1_GCR"] = {
    "n_assumed": n_gcr1,
    "ci95_wilson_pct": {m: [round(100 * x, 1) for x in wilson(round(v * n_gcr1 / 100), n_gcr1)]
                        for m, v in gcr1.items()},
    "SP_vs_BC": compare("GCR SP-TLA vs BC (n=2000 가정)", 12.5, 10.4, n_gcr1),
    "SP_vs_DP": compare("GCR SP-TLA vs DP (n=2000 가정)", 12.5, 8.5, n_gcr1),
}
# GCR 비율은 정수 횟수로 바꿔 Fisher도 함께 본다 (반올림 오차 ±0.05%p 안에서)
results["V4_table1_GCR"]["SP_vs_BC_fisher_counts"] = round(
    fisher_two_sided(250, 1750, 208, 1792), 4)

# ---- V5: 표 II ID GCR (표본 4k 가정) 과 L1 감소율 ----
gcr2 = {"BC": 18.4, "DP": 7.4, "MP-TLA": 18.4}
n_gcr2 = 4000
L2 = {  # ID: L1 x, y, rz
    "BC": (0.260, 0.655, 0.186), "DP": (0.371, 0.348, 0.480), "MP-TLA": (0.102, 0.114, 0.135)}
L2o = {  # OOD: L1 x, y, rz
    "BC": (0.286, 0.722, 0.246), "DP": (0.386, 0.369, 0.544), "MP-TLA": (0.121, 0.102, 0.184)}


def red(d, idx):
    best = min(d["BC"][idx], d["DP"][idx])
    return round(100 * (1 - d["MP-TLA"][idx] / best), 1)


results["V5_table2"] = {
    "ID_gcr_MP_vs_DP": compare("GCR MP-TLA vs DP, ID (n=4000 가정)", 18.4, 7.4, n_gcr2),
    "ID_gcr_MP_vs_BC": compare("GCR MP-TLA vs BC, ID (n=4000 가정)", 18.4, 18.4, n_gcr2),
    "ID_L1_reduction_pct_xyrz": [red({"BC": L2["BC"], "DP": L2["DP"], "MP-TLA": L2["MP-TLA"]}, i) for i in range(3)],
    "OOD_L1_reduction_pct_xyrz": [red({"BC": L2o["BC"], "DP": L2o["DP"], "MP-TLA": L2o["MP-TLA"]}, i) for i in range(3)],
}

# ---- V6: 표 III 간격별 비교 (n=50 per cell) 와 합계 (n=150 pooled) ----
rows = []
for lab, idx in [("2.0 mm", 0), ("1.6 mm", 1), ("1.0 mm", 2)]:
    rows.append(compare(f"Suc SP-TLA vs DP, {lab}", T3["SP-TLA"]["suc"][idx], T3["DP"]["suc"][idx], 50))
    rows.append(compare(f"Suc MP-TLA vs DP, {lab}", T3["MP-TLA"]["suc"][idx], T3["DP"]["suc"][idx], 50))
    rows.append(compare(f"Suc SP-TLA vs BC, {lab}", T3["SP-TLA"]["suc"][idx], T3["BC"]["suc"][idx], 50))
rows.append(compare("Suc SP-TLA vs DP, 합계 (150회 pooled)", 88, 40, 150))
rows.append(compare("Suc MP-TLA vs DP, 합계 (150회 pooled)", 90, 40, 150))
rows.append(compare("Suc SP-TLA vs BC, 합계 (150회 pooled)", 88, 31, 150))
results["V6_table3_tests"] = rows
results["V6_table3_margins"] = {
    "rel_gain_vs_second_best_pct": {  # 2순위(DP) 대비 상대 증가
        "2.0mm": round(100 * (96 / 58 - 1), 1), "1.6mm": round(100 * (94 / 43 - 1), 1),
        "1.0mm": round(100 * (74 / 20 - 1), 1), "total": round(100 * (88 / 40 - 1), 1)},
    "abs_gain_pp_vs_second_best": {"2.0mm": 38, "1.6mm": 51, "1.0mm": 54, "total": 48},
}

# ---- V7: 표 IV 간격 (n=50 per cell) ----
rows4 = []
for j, lab in enumerate(["Square(ID)", "Triangle(ID)", "Round(OOD)", "Hexagon(OOD)"]):
    best_other = max(T4["BC"]["suc"][j], T4["DP"]["suc"][j])
    rows4.append({"peg": lab, "SP_TLA": T4["SP-TLA"]["suc"][j], "MP_TLA": T4["MP-TLA"]["suc"][j],
                  "best_baseline": best_other,
                  "MP_vs_best_baseline": compare(f"Suc MP-TLA vs best baseline, {lab}",
                                                 T4["MP-TLA"]["suc"][j], best_other, 50)})
results["V7_table4"] = rows4

# ---- V8: 표 III "Step" (성공 에피소드 평균) 와 본문 "효율" 주장 ----
results["V8_steps"] = {
    "2.0mm_SP_vs_DP_and_BC": [T3["SP-TLA"]["step"][0], T3["DP"]["step"][0], T3["BC"]["step"][0]],
    "1.6mm_SP_vs_DP_and_BC": [T3["SP-TLA"]["step"][1], T3["DP"]["step"][1], T3["BC"]["step"][1]],
    "1.0mm_SP_vs_DP_and_BC": [T3["SP-TLA"]["step"][2], T3["DP"]["step"][2], T3["BC"]["step"][2]],
}

# ---- V9: 초록의 "1.6 mm·1.0 mm에서 85% 이상" 주장 점검 (2.0 mm만 학습) ----
results["V9_abstract_85pct"] = {
    "SP_TLA_1.6": T3["SP-TLA"]["suc"][1], "SP_TLA_1.0": T3["SP-TLA"]["suc"][2],
    "MP_TLA_1.6": T3["MP-TLA"]["suc"][1], "MP_TLA_1.0": T3["MP-TLA"]["suc"][2],
    "SP_TLA_OOD_shapes": [T4["SP-TLA"]["suc"][2], T4["SP-TLA"]["suc"][3]],
    "MP_TLA_OOD_shapes": [T4["MP-TLA"]["suc"][2], T4["MP-TLA"]["suc"][3]],
}

# ---- V10: 시각 토큰 수 산술 (본문 §IV-A: 616×616, 패치 14, 2×2 병합) ----
side = 616 // 14
patches = side * side
results["V10_token_count"] = {
    "patch_grid": f"{side}x{side}", "vit_patches": patches,
    "tokens_after_2x2_merge": patches // 4,
    "paper_claim": 1936,
    "consistent": patches // 4 == 1936,
}

# ---- V11: 학습·평가 표본 수 합 (24k 주장) ----
results["V11_dataset_counts"] = {
    "single_peg_split": {"train": 6000, "test": 2000, "total": 8000},
    "multi_peg_train": 16000, "multi_peg_eval": 8000,
    "multi_plus_eval": 16000 + 8000,
    "claim_k": 24,
    "note": "16k 학습 + 8k 평가 = 24k로 맞는다. 단일 페그 8k가 이 안에 포함되는지는 본문에 없다.",
}

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
