"""SpatialVLA 논문 표 수치 재계산과 검정 (수 초, numpy·scipy).

입력: 논문 v5 본문과 부록의 표 값 (pdftotext -layout으로 읽은 값을 손으로 옮김).
출력: recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher(k1, n1, k2, n2):
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


# ---------------------------------------------------------------- 표 X: Google Robot (SimplerEnv)
# 원 행: 9개 값 = 픽(수평, 수직, 서기, 평균) + Move Near + 서랍(open, close, 평균) + #Overall
# 셀(6개) = 수평, 수직, 서기, Move Near, open, close. 평균 열은 셀이 아니므로 제외.
rows_va = {  # Variant Aggregation (Table X 첫 블록)
    "RT-1 (Begin)": [2.2, 1.3, 3.1, 2.2, 4.0, 0.5, 13.2, 6.9, 4.2],
    "RT-1 (15%)": [92.0, 70.4, 81.3, 81.3, 44.6, 21.2, 32.3, 26.7, 56.2],
    "RT-1 (Converged)": [96.9, 76.0, 96.4, 89.8, 50.0, 27.0, 37.6, 32.3, 63.3],
    "RT-1-X": [56.9, 20.4, 69.8, 49.0, 32.3, 6.9, 51.9, 29.4, 39.6],
    "RT-2-X": [82.2, 75.4, 89.3, 82.3, 79.2, 33.3, 37.2, 35.3, 64.3],
    "Octo-Base": [0.5, 0.0, 1.3, 0.6, 3.1, 0.0, 2.1, 1.1, 1.1],
    "OpenVLA": [71.1, 27.1, 65.3, 54.5, 47.7, 15.8, 19.5, 17.7, 39.8],
    "RoboVLM (zero-shot)": [77.8, 48.0, 79.1, 68.3, 56.0, 1.6, 15.3, 8.5, 46.3],
    "RoboVLM (fine-tuning)": [93.8, 49.8, 83.1, 75.6, 60.0, 2.6, 18.5, 10.6, 51.3],
    "SpatialVLA (zero-shot)": [93.3, 83.1, 92.0, 89.5, 71.7, 23.3, 49.2, 36.2, 68.8],
    "SpatialVLA (fine-tuning)": [93.3, 78.2, 92.4, 88.0, 72.7, 28.6, 55.0, 41.8, 70.7],
}
rows_vm = {  # Visual Matching (Table X 둘째 블록)
    "RT-1 (Begin)": [5.0, 0.0, 3.0, 2.7, 5.0, 0.0, 27.8, 13.9, 6.8],
    "RT-1 (15%)": [86.0, 79.0, 48.0, 71.0, 35.4, 46.3, 66.7, 56.5, 60.2],
    "RT-1 (Converged)": [96.0, 90.0, 71.0, 85.7, 44.2, 60.1, 86.1, 73.0, 74.6],
    "RT-1-X": [82.0, 33.0, 55.0, 56.7, 31.7, 29.6, 89.1, 59.7, 53.4],
    "RT-2-X": [74.0, 74.0, 88.0, 78.7, 77.9, 15.7, 34.3, 25.0, 60.7],
    "Octo-Base": [21.0, 21.0, 9.0, 17.0, 4.2, 0.9, 44.4, 22.7, 16.8],
    "OpenVLA": [27.0, 3.0, 19.0, 16.3, 46.2, 19.4, 51.8, 35.6, 27.7],
    "RoboVLM (zero-shot)": [85.0, 43.0, 90.0, 72.7, 66.3, 28.7, 25.0, 26.8, 56.3],
    "RoboVLM (fine-tuning)": [94.0, 47.0, 91.0, 77.3, 61.7, 33.3, 53.1, 43.5, 63.4],
    "SpatialVLA (zero-shot)": [70.0, 82.0, 91.0, 81.0, 69.6, 49.1, 69.4, 59.3, 71.9],
    "SpatialVLA (fine-tuning)": [85.0, 76.0, 97.0, 86.0, 77.9, 50.0, 64.8, 57.4, 75.1],
}
google = {}
for name in rows_va:
    v, m = rows_va[name], rows_vm[name]
    cells_va = [v[0], v[1], v[2], v[4], v[5], v[6]]
    cells_vm = [m[0], m[1], m[2], m[4], m[5], m[6]]
    google[name] = {
        "VA_6cell_mean": round(float(np.mean(cells_va)), 2), "VA_reported": v[8],
        "VM_6cell_mean": round(float(np.mean(cells_vm)), 2), "VM_reported": m[8],
    }
res["simpler_google_overall"] = google

# ---------------------------------------------------------------- 표 II: WidowX (SimplerEnv)
# 셀 값의 분모는 72 (모든 셀이 1/72 격자에 맞음). 성공률 네 개의 평균 = #Overall.
w_succ = {  # 성공 횟수 (put spoon, carrot, block, eggplant)
    "RT-1-X": [0, 3, 0, 0],
    "Octo-Base": [9, 6, 0, 31],
    "Octo-Small": [34, 7, 3, 41],
    "OpenVLA": [0, 0, 0, 3],
    "RoboVLM (zero-shot)": [15, 18, 6, 0],
    "RoboVLM (fine-tuning)": [21, 18, 9, 42],
    "SpatialVLA (zero-shot)": [15, 15, 18, 51],
    "SpatialVLA (fine-tuning)": [12, 18, 21, 72],
}
w_rep = {"RT-1-X": 1.1, "Octo-Base": 16.0, "Octo-Small": 30.0, "OpenVLA": 1.0,
         "RoboVLM (zero-shot)": 13.5, "RoboVLM (fine-tuning)": 31.3,
         "SpatialVLA (zero-shot)": 34.4, "SpatialVLA (fine-tuning)": 42.7}
widowx_sim = {}
for k, v in w_succ.items():
    overall = 100 * sum(v) / (4 * 72)
    widowx_sim[k] = {"successes": v, "overall_recomputed": round(overall, 2), "overall_reported": w_rep[k],
                     "diff": round(overall - w_rep[k], 2)}
res["widowx_simpler_overall"] = widowx_sim

# ---------------------------------------------------------------- 표 XI: WidowX 실로봇 zero-shot (7 과제 x 11회)
# 열: (close microwave), (lift chili), (eggplant grasp, success), (purple cup grasp, success),
#     (green cup stove grasp, success), (purple cup sink grasp, success), (carrot grasp, success)
# 성공 횟수만 사용. chili는 열이 하나만 보임 (Grasp/Success 쌍 아님).
real_succ = {
    "SpatialVLA": {"microwave": 11, "chili": 8, "eggplant": 8, "purple_white": 5, "green_stove": 9, "purple_sink": 10, "carrot": 5},
    "OpenVLA": {"microwave": 11, "chili": 8, "eggplant": 7, "purple_white": 2, "green_stove": 4, "purple_sink": 6, "carrot": 2},
    "RoboVLM": {"microwave": 8, "chili": 0, "eggplant": 1, "purple_white": 3, "green_stove": 2, "purple_sink": 1, "carrot": 3},
    "Octo-Base": {"microwave": 6, "chili": 5, "eggplant": 1, "purple_white": 0, "green_stove": 1, "purple_sink": 2, "carrot": 0},
    "RT-1-X": {"microwave": 6, "chili": 4, "eggplant": 0, "purple_white": 1, "green_stove": 2, "purple_sink": 0, "carrot": 1},
}
N_REAL = 11
real = {}
for k, d in real_succ.items():
    tot = sum(d.values())
    real[k] = {"successes": tot, "n": 77, "rate": round(100 * tot / 77, 1), "wilson95": [round(100 * x, 1) for x in wilson(tot, 77)]}
real["p_SpatialVLA_vs_OpenVLA_fisher"] = fisher(56, 77, 40, 77)
real["p_SpatialVLA_vs_RoboVLM_fisher"] = fisher(56, 77, 18, 77)
real["p_SpatialVLA_vs_OpenVLA_mcnemar_note"] = "쌍 구조는 논문에 없음 (같은 시도 쌍 미보고), Fisher로만 계산"
real["per_task_successes"] = real_succ
res["widowx_real_zeroshot"] = real

# ---------------------------------------------------------------- 표 IX: 네 과제 깊이 어블레이션 (11회)
t9 = {
    "OpenVLA": [8, 8, 2, 6], "SpatialVLA w/o D": [7, 6, 1, 6],
    "SpatialVLA sensor D": [10, 9, 3, 9], "SpatialVLA zoe D": [10, 10, 3, 9],
}
t9_rep = {"OpenVLA": 54.5, "SpatialVLA w/o D": 45.4, "SpatialVLA sensor D": 70.5, "SpatialVLA zoe D": 72.7}
res["table9_depth"] = {k: {"avg_recomputed": round(100 * sum(v) / 44, 2), "avg_reported": t9_rep[k],
                           "successes": sum(v), "n": 44} for k, v in t9.items()}
res["table9_depth"]["p_zoeD_vs_OpenVLA_fisher"] = fisher(32, 44, 24, 44)
res["table9_depth"]["p_zoeD_vs_sensorD_fisher"] = fisher(32, 44, 31, 44)

# ---------------------------------------------------------------- 표 XII–XIV: Franka 실로봇 (11회)
franka = {
    "single": {  # teapot, pot(grasp,success), banana(grasp,success), drawer -> 성공만
        "Diffusion Policy": [8, 8, 8, 11], "Octo": [11, 7, 0, 11], "OpenVLA": [11, 6, 6, 9], "SpatialVLA": [9, 8, 11, 8]},
    "multi": {
        "Diffusion Policy": [2, 3, 3, 3], "Octo": [3, 4, 4, 3], "OpenVLA": [6, 6, 7, 4], "SpatialVLA": [7, 4, 9, 5]},
}
# 표 XIII의 pot 열은 (grasp, success) 쌍. 위 리스트는 (teapot, pot_success, banana_success, drawer)
franka_single_success = {k: v for k, v in franka["single"].items()}
franka_multi_success = {k: v for k, v in franka["multi"].items()}
inst = {  # 5 과제 성공 횟수 (red, green, blue, orange, croissant)
    "Diffusion Policy": [1, 0, 0, 2, 1], "Octo": [7, 0, 5, 7, 2],
    "OpenVLA": [6, 4, 3, 3, 4], "SpatialVLA": [6, 5, 7, 6, 6],
}
franka_out = {}
for label, dd in [("single", franka_single_success), ("multi", franka_multi_success), ("instruction", inst)]:
    k_list = {m: sum(v) for m, v in dd.items()}
    n_list = {m: (44 if label != "instruction" else 55) for m in dd}
    mean_rate = {m: round(100 * np.mean(np.array(v) / 11), 2) for m, v in dd.items()}
    franka_out[label] = {"successes_total": k_list, "n": n_list, "mean_of_task_rates": mean_rate}
franka_out["single"]["p_SpatialVLA_vs_OpenVLA_fisher"] = fisher(sum(franka["single"]["SpatialVLA"]), 44,
                                                                   sum(franka["single"]["OpenVLA"]), 44)
franka_out["multi"]["p_SpatialVLA_vs_OpenVLA_fisher"] = fisher(sum(franka["multi"]["SpatialVLA"]), 44,
                                                                 sum(franka["multi"]["OpenVLA"]), 44)
franka_out["instruction"]["p_SpatialVLA_vs_OpenVLA_fisher"] = fisher(sum(inst["SpatialVLA"]), 55,
                                                                       sum(inst["OpenVLA"]), 55)
franka_out["instruction"]["p_SpatialVLA_vs_OpenVLA_fisher_note"] = "같은 55회 기준"
res["franka"] = franka_out

# ---------------------------------------------------------------- 표 XV: 공간 이해 (11회)
spatial = {
    "SpatialVLA": {"plush_success": 7, "stove_success": 8, "sink_success": 9, "carrot_success": 7},
    "OpenVLA": {"plush_success": 5, "stove_success": 3, "sink_success": 5, "carrot_success": 6},
    "Octo-Base": {"plush_success": 3, "stove_success": 1, "sink_success": 2, "carrot_success": 1},
}
spatial_xi_stove = {"SpatialVLA_XI": 9, "OpenVLA_XI": 4}  # 표 XI의 같은 stove 과제 성공 횟수
res["spatial_table15"] = {
    "SpatialVLA_zero_shot_widowx_mean_success": round(100 * (8 + 9 + 7) / 33, 1),
    "OpenVLA_zero_shot_widowx_success_rates": [27.3, 45.5, 54.5],
    "xi_vs_xv_stove_mismatch": {"SpatialVLA": {"table_XI": 9, "table_XV": 8}, "OpenVLA": {"table_XI": 4, "table_XV": 3}},
    "text_73pct_is_grasp": "표 XV 플러시 과제 SpatialVLA 파지 72.72%, 성공 63.63% (본문 73%는 파지)",
}

# ---------------------------------------------------------------- 표 III: LIBERO
libero = {  # 스위트별 SR, ± (논문 표기; 표준오차로 가정)
    "Diffusion Policy": ([78.3, 92.5, 68.3, 50.5], [1.1, 0.7, 1.2, 1.3], 72.4),
    "Octo": ([78.9, 85.7, 84.6, 51.1], [1.0, 0.9, 0.9, 1.0], 75.1),
    "OpenVLA": ([84.7, 88.4, 79.2, 53.7], [0.9, 0.8, 1.0, 1.3], 76.5),
    "TraceVLA": ([84.6, 85.2, 75.1, 54.1], [0.2, 0.4, 0.3, 1.0], 74.8),
    "SpatialVLA": ([88.2, 89.9, 78.6, 55.5], [0.5, 0.7, 0.6, 1.0], 78.1),
}
lib = {}
for k, (sr, se, rep) in libero.items():
    lib[k] = {"avg_recomputed": round(float(np.mean(sr)), 2), "avg_reported": rep}
lib_test = {}
for suite_i, suite in enumerate(["Spatial", "Object", "Goal", "Long"]):
    a, sa = libero["SpatialVLA"][0][suite_i], libero["SpatialVLA"][1][suite_i]
    for other in ["OpenVLA", "Octo", "TraceVLA"]:
        b, sb = libero[other][0][suite_i], libero[other][1][suite_i]
        z = (a - b) / math.sqrt(sa ** 2 + sb ** 2)
        lib_test[f"{suite}_vs_{other}_z_(SE_assumed)"] = round(z, 2)
res["libero"] = {"averages": lib, "z_tests": lib_test,
                 "note": "± 를 표준오차로 가정한 근사. 3시드 평균이라 자유도가 작아 z 근사는 과대평가 가능"}

# ---------------------------------------------------------------- 표 IV: 사전학습 어블레이션 (SimplerEnv, 24회 셀 가정)
t4 = {  # [VA pick, VM pick, VA move, VM move, carrot grasp, carrot success, eggplant grasp, eggplant success]
    "1 SpatialVLA (8194)": [81.6, 70.7, 79.2, 85.4, 41.7, 33.3, 91.7, 87.5],
    "2 linear 256": [40.7, 19.0, 47.1, 52.9, 41.7, 33.3, 87.5, 70.8],
    "3 uniform 8194": [77.9, 28.0, 64.2, 55.0, 45.8, 12.5, 79.2, 54.2],
    "4 resolution 1026": [74.4, 67.3, 59.1, 54.2, 45.8, 25.0, 66.7, 54.2],
    "8 - ego3d": [68.9, 70.3, 66.7, 62.0, 54.2, 12.5, 75.0, 37.5],
    "9 - freeze LLM embed": [70.2, 50.7, 63.1, 62.5, 33.3, 20.8, 95.8, 79.2],
}
d_va = np.mean([t4["1 SpatialVLA (8194)"][0], t4["1 SpatialVLA (8194)"][2]]) - np.mean([t4["2 linear 256"][0], t4["2 linear 256"][2]])
d_vm = np.mean([t4["1 SpatialVLA (8194)"][1], t4["1 SpatialVLA (8194)"][3]]) - np.mean([t4["2 linear 256"][1], t4["2 linear 256"][3]])
res["table4_ablation"] = {
    "delta_VA_grid_vs_linear_mean_of_Google_cols": round(float(d_va), 1), "reported_VA": 36.5,
    "delta_VM_grid_vs_linear_mean_of_Google_cols": round(float(d_vm), 1), "reported_VM": 42.1,
    "delta_move_near_VM_1_vs_4": round(85.4 - 54.2, 1), "reported_move_near": 31.2,
    "delta_eggplant_success_1_vs_4": round(87.5 - 54.2, 1), "reported_eggplant": 33.3,
    "ego3d_drop_VA_pick_81.6_to_68.9": round(81.6 - 68.9, 1), "ego3d_drop_VA_move_79.2_to_66.7": round(79.2 - 66.7, 1),
    "note": "시행 수 24는 셀 값이 1/24 격자에 맞아서 가정. 단일 실행, 시드 분산 미보고",
}
# 셀 시행 수 24 가정 아래 단일 셀 차이의 최소 검출 차이 (80% 검정력, 양측 5%, 정규 근사)
n = 24
p0 = 0.5
mdd = 2.8 * math.sqrt(2 * p0 * (1 - p0) / n)
res["table4_ablation"]["MDE_pp_at_n24_p0.5_80pct"] = round(100 * mdd, 1)

# ---------------------------------------------------------------- 표 V: 미세조정 어블레이션
res["table5_ablation"] = {
    "move_near_VM_gain_fullFT_to_gaussian": round(77.9 - 75.0, 1), "reported": 2.9,
    "libero_gain_spatial_emb_adapt": {"Spatial": round(88.2 - 83.6, 1), "Object": round(89.9 - 84.8, 1),
                                      "Goal": round(78.6 - 76.4, 1), "Long": round(55.5 - 50.1, 1)},
    "libero_reported_gains": {"Spatial": 4.6, "Object": 5.1, "Goal": 2.2, "Long": 5.4},
}

# ---------------------------------------------------------------- 본문 주장 대조
res["claims"] = {
    "franka_single_SpatialVLA_avg_success": round(np.mean([9, 8, 11, 8]) / 11 * 100, 1),
    "franka_single_DP_avg_success": round(np.mean([8, 8, 8, 11]) / 11 * 100, 1),
    "franka_single_text": "SpatialVLA 82% 대 DP 81% (본문)",
    "franka_instruction_SpatialVLA_avg": round(np.mean([6, 5, 7, 6, 6]) / 11 * 100, 1),
    "franka_instruction_OpenVLA_avg": round(np.mean([6, 4, 3, 3, 4]) / 11 * 100, 1),
    "franka_instruction_DP_avg": round(np.mean([1, 0, 0, 2, 1]) / 11 * 100, 1),
    "franka_instruction_text": "본문: SpatialVLA가 OpenVLA보다 +12%, DP는 26%",
    "franka_multi_SpatialVLA_avg": round(np.mean([7, 4, 9, 5]) / 11 * 100 * 1.0, 1),
    "franka_multi_text": "본문 57% (부록 표 XIII 평균 56.8%와 일치)",
    "widowx_real_text_SOTA_all_7": "표 XI 전 과제 최고 주장, 그러나 close microwave는 OpenVLA와 동률(11/11)",
    "widowx_real_text_fig5": "7개 과제 x 11회 = 77회 (본문과 일치)",
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:6000])
print("wrote", OUT)
