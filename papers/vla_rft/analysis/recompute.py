"""VLA-RFT 표 값의 재계산과 일관성 검사 (numpy만 사용).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
가정(본문에 없음, 명시): LIBERO 표준 평가는 과제당 50 에피소드 x 10 과제 = 스위트당 500 에피소드라고 가정한다.
표 2의 SR 값은 모두 0.2% 간격(500의 배수)이라 이 가정과 맞는다. 학습 실행 시드 수는 보고되지 않았다.
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("recompute_results.json")
res = {}


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def two_prop_z(s1, n1, s2, n2):
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    return z, 2 * norm_sf(abs(z))


# 1. 표 1 평균과 표 2, 표 4, 표 10 산술
t1 = {"MSE": [0.0039, 0.0036, 0.0024, 0.0056], "PSNR": [24.98, 25.13, 26.99, 23.83],
      "SSIM": [0.896, 0.913, 0.929, 0.885], "LPIPS": [0.067, 0.054, 0.040, 0.074]}
t1_avg_reported = {"MSE": 0.0039, "PSNR": 25.23, "SSIM": 0.906, "LPIPS": 0.059}
res["table1_avg_recomputed"] = {k: round(float(np.mean(v)), 4) for k, v in t1.items()}
res["table1_avg_reported"] = t1_avg_reported

suites = ["Spatial", "Object", "Goal", "Long"]
base = np.array([88.4, 88.0, 92.8, 77.2])
rft = np.array([94.4, 94.4, 95.4, 80.2])
res["table2_base_avg"] = round(base.mean(), 3)  # 표 2 보고값 86.6
res["table2_rft_avg"] = round(rft.mean(), 3)    # 표 2 보고값 91.1, 본문 §4.3은 91.3
res["table2_delta_per_suite"] = dict(zip(suites, np.round(rft - base, 2).tolist()))  # 6.0 6.4 2.6 3.0 (표와 일치)
res["table2_delta_avg"] = round(rft.mean() - base.mean(), 2)  # 4.5 (본문은 +4.7)

# 500 에피소드 가정 하의 성공 수와 두 비율 검정 (같은 학습 실행 하나의 샘플링 잡음만 본다)
n = 500
sb = np.round(base / 100 * n).astype(int)
sr = np.round(rft / 100 * n).astype(int)
res["implied_successes_base"] = dict(zip(suites, sb.tolist()))
res["implied_successes_rft"] = dict(zip(suites, sr.tolist()))
res["per_suite_two_prop"] = {s: {"z": round(two_prop_z(sb[i], n, sr[i], n)[0], 2),
                                  "p": float(f"{two_prop_z(sb[i], n, sr[i], n)[1]:.3g}")}
                             for i, s in enumerate(suites)}
# 전체 평균: 4 x 500 = 2000 에피소드씩
S_b = int(sb.sum())
S_r = int(sr.sum())
z, p = two_prop_z(S_b, 2000, S_r, 2000)
res["pooled_two_prop"] = {"base_succ": S_b, "rft_succ": S_r, "n_per_arm": 2000, "z": round(z, 2), "p": float(f"{p:.3g}")}

# 표 2 스위트별 차이의 Wald 95% CI (에피소드 샘플링만, 시드 분산 없음)
def wald_ci(p1, p2, n1, n2):
    d = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return d - 1.96 * se, d + 1.96 * se
res["per_suite_wald95_ci_pp"] = {s: [round(100 * x, 2) for x in wald_ci(sb[i] / n, sr[i] / n, n, n)]
                                  for i, s in enumerate(suites)}

# 표 3, 표 8 (섭동)
pert = {  # 표 8 셀 (Task1 Dim1, Task1 Dim2, Task2 Dim1, Task2 Dim2)
    "Object ±2.5": ([87, 52, 78, 60], [94, 62, 80, 58], 69.3, 73.5, 4.2),
    "Object ±5": ([70, 44, 50, 28], [72, 52, 56, 30], 48.0, 52.5, 4.5),
    "Goal ±2.5": ([62, 58, 92, 86], [64, 68, 94, 90], 74.5, 79.0, 4.5),
    "Goal ±5": ([34, 46, 48, 54], [46, 42, 58, 60], 44.8, 51.5, 6.7),
    "RoboState ±20": ([60, 88, 54, 90], [62, 92, 58, 94], 73.0, 76.5, 2.5),
    "RoboState ±50": ([42, 82, 52, 78], [46, 86, 56, 80], 63.5, 67.0, 3.5),
    "Combined ±2.5/2.5/20": ([64, 82, 36, 72], [68, 92, 40, 80], 63.5, 70.0, 6.5),
    "Combined ±5/5/50": ([34, 64, 8, 30], [36, 60, 12, 40], 34.0, 37.0, 3.0),
}
pert_rows = {}
wins = ties = losses = 0
for k, (b, o, rb, ro, rd) in pert.items():
    mb, mo = np.mean(b), np.mean(o)
    d = np.array(o) - np.array(b)
    wins += int((d > 0).sum()); ties += int((d == 0).sum()); losses += int((d < 0).sum())
    pert_rows[k] = {"table8_mean_base": round(float(mb), 2), "reported_base": rb,
                    "table8_mean_ours": round(float(mo), 2), "reported_ours": ro,
                    "reported_delta": rd, "recomputed_delta": round(ro - rb, 2),
                    "cell_diffs": d.tolist()}
res["perturbation"] = pert_rows
res["perturbation_cell_sign_test"] = {"wins": wins, "ties": ties, "losses": losses,
                                      "note": "표 8의 32개 셀(8 설정 x 4 셀) 대응 비교. 셀별 n은 미보고"}

# 셀 수 n이 없어 얻을 수 있는 민감도: 평균 차이 d=4.2 (Object ±2.5)가 p<0.05가 되려면 설정당 n이 얼마여야 하는가
def n_needed(p1, p2, alpha_z=1.96):
    for nn in range(10, 20000, 10):
        z, _ = two_prop_z(p1 * nn, nn, p2 * nn, nn)
        if abs(z) >= alpha_z:
            return nn
    return None
res["n_per_arm_needed_for_p05_object_pm25"] = n_needed(0.693, 0.735)
res["n_per_arm_needed_for_p05_goal_pm5"] = n_needed(0.448, 0.515)

# 2. 표 9 평균 재계산 (행별 4 스위트 평균)
t9 = {"Diffusion Policy": [78.3, 92.5, 68.3, 50.5, 72.4], "Octo": [78.9, 85.7, 84.6, 51.1, 75.1],
      "MDT": [78.5, 87.5, 73.5, 64.8, 76.1], "OpenVLA": [84.7, 88.4, 79.2, 53.7, 76.5],
      "SpatialVLA": [88.2, 89.9, 78.6, 55.5, 78.1], "WorldVLA": [87.6, 96.2, 83.4, 60.0, 81.8],
      "CoT-VLA": [87.5, 91.6, 87.6, 69.0, 81.1], "TraceVLA": [84.6, 85.2, 75.1, 54.1, 74.8],
      "pi0": [91.2, 93.2, 93.8, 74.2, 88.1], "VLA-Adapter(Base)": [88.4, 88.0, 92.8, 77.2, 86.6],
      "VLA-RFT": [94.4, 94.4, 95.4, 80.2, 91.1]}
mism = {k: round(float(np.mean(v[:4])) - v[4], 3) for k, v in t9.items() if abs(np.mean(v[:4]) - v[4]) > 0.06}
res["table9_avg_mismatch_over_0.06"] = mism
# 1위 확인
res["table9_rank1_by_avg"] = max(t9, key=lambda k: t9[k][4])

# 표 10 산술
t10 = [("VLA-RL", 76.5, 81.0, 4.5), ("ARFM", 88.1, 92.1, 4.0), ("RWR", 88.1, 90.8, 2.7),
       ("ReinboT", 88.1, 91.2, 3.1), ("VLA-RFT", 86.6, 91.1, 4.5)]
res["table10_check"] = {a: round(s - b - d, 2) for a, b, s, d in t10}

# 표 4 Δ
res["table4_deltas"] = {"type1": round(87.7 - 86.6, 2), "type2": round(87.1 - 86.6, 2), "type3": round(91.1 - 86.6, 2)}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
               encoding="utf-8")
print(json.dumps({k: res[k] for k in ["table2_rft_avg", "table2_delta_avg", "pooled_two_prop",
                                       "n_per_arm_needed_for_p05_object_pm25", "perturbation_cell_sign_test",
                                       "table9_avg_mismatch_over_0.06", "table10_check"]}, ensure_ascii=False, indent=1, default=str))
print("RoboState ±20 reported vs recomputed:", pert_rows["RoboState ±20"]["reported_delta"], pert_rows["RoboState ±20"]["recomputed_delta"])
print("Goal ±5 base mean:", pert_rows["Goal ±5"]["table8_mean_base"], "reported", pert_rows["Goal ±5"]["reported_base"])
print("Per-suite:", res["per_suite_two_prop"], res["per_suite_wald95_ci_pp"])
