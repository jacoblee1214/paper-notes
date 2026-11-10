"""DexTouch-WM 표 재계산 (Table I–V). numpy만 사용, 1스레드.

실행: python3 analysis/table_check.py  -> analysis/table_check_results.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("table_check_results.json")
res = {}


def t_two_sided_p(t, df):
    # df=1 (n=3 상관) 과 큰 df 근사 모두 쓰기 위해 수치 적분 없이 t 분포 CDF를 근사
    # df=1 은 Cauchy 해석식, 그 외는 정규 근사(보고용)
    if df == 1:
        return 1 - 2 / np.pi * np.arctan(abs(t))
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def pearson_p_n3(r):
    # 표본 n=3 의 Pearson 상관은 df=1. t = r*sqrt(df)/sqrt(1-r^2)
    if abs(r) >= 1:
        return 0.0
    t = r * sqrt(1) / sqrt(1 - r * r)
    return t_two_sided_p(t, 1)


# ---- Table I: 스케일링 (5h 로봇 고정) ----
table1 = {
    # 열: V-only, Cross-Attn, Robot-only, +10h, +50h, +100h
    "PSNR": [22.099, 22.130, 23.473, 24.032, 26.874, 27.096],
    "SSIM": [0.799, 0.801, 0.824, 0.838, 0.886, 0.890],
    "LPIPS": [0.117, 0.118, 0.098, 0.084, 0.048, 0.048],
    "Aesthetic": [0.406, 0.405, 0.400, 0.404, 0.414, 0.414],
    "ImageQ": [0.718, 0.717, 0.718, 0.717, 0.723, 0.724],
    "JEPA": [0.870, 0.860, 0.864, 0.894, 0.917, 0.924],
    "SubjCons": [0.689, 0.681, 0.751, 0.740, 0.748, 0.751],
    "SemAlign": [0.829, 0.788, 0.813, 0.835, 0.866, 0.860],
    "TrajAcc": [0.884, 0.850, 0.891, 0.900, 0.962, 0.962],
    "GeomErr": [0.128, 0.130, 0.131, 0.112, 0.102, 0.100],
    # 촉각 열: Cross-Attn, Robot-only, +10h, +50h, +100h (V-only는 '–')
    "PSNRtaxel": [24.132, 24.813, 24.349, 28.401, 29.179],
    "SSIMtaxel": [0.795, 0.810, 0.823, 0.876, 0.896],
    "LPIPSheat": [0.009, 0.008, 0.008, 0.003, 0.002],
    "MSE": [0.105, 0.093, 0.108, 0.038, 0.029],
    "MAE": [0.075, 0.069, 0.069, 0.040, 0.035],
    "ContactMAE": [0.880, 0.806, 0.819, 0.580, 0.516],
    "ContactIoU": [0.388, 0.415, 0.417, 0.534, 0.588],
    "ContactF1": [0.523, 0.551, 0.554, 0.659, 0.706],
}
t1 = {}
for k, v in table1.items():
    if len(v) == 6:
        robot, h100 = v[2], v[5]
    else:
        robot, h100 = v[1], v[4]
    t1[k] = {"robot_only": robot, "plus100h": h100, "delta": round(h100 - robot, 4),
             "rel_pct": round(100 * (h100 - robot) / abs(robot), 1)}
res["table1_robot_to_100h"] = t1

# 텍스트 주장 점검: "+10h는 촉각에서 거의 평평" -> PSNRtaxel, IoU
res["table1_plus10_vs_robot"] = {
    "PSNRtaxel": round(24.349 - 24.813, 3),
    "ContactIoU": round(0.417 - 0.415, 3),
    "ContactF1": round(0.554 - 0.551, 3),
}
# 포화 점검: +50h -> +100h 증분 대 robot -> +50h 증분
res["saturation"] = {
    "PSNR_robot_to_50": round(26.874 - 23.473, 3),
    "PSNR_50_to_100": round(27.096 - 26.874, 3),
    "ContactIoU_robot_to_50": round(0.534 - 0.415, 3),
    "ContactIoU_50_to_100": round(0.588 - 0.534, 3),
    "TrajAcc_50_to_100": round(0.962 - 0.962, 3),
    "Aesthetic_robot_to_100": round(0.414 - 0.400, 3),
}

# ---- Table III: 정책 평가 상관과 MMRV (n=3 정책) ----
real = {  # Table II 실제 점수 (정책 x 과제)
    "PlaceShoes": {"FTP": 0.725, "pi05": 0.450, "XVLA": 0.500},
    "PlacePhone": {"FTP": 0.800, "pi05": 0.700, "XVLA": 0.600},
    "StackBowls": {"FTP": 0.500, "pi05": 0.850, "XVLA": 0.300},
    "StandBottle": {"FTP": 0.900, "pi05": 0.500, "XVLA": 0.600},
}
wm_robot = {
    "PlaceShoes": {"FTP": 0.975, "pi05": 0.950, "XVLA": 0.600},
    "PlacePhone": {"FTP": 0.975, "pi05": 0.925, "XVLA": 0.725},
    "StackBowls": {"FTP": 0.750, "pi05": 0.950, "XVLA": 0.150},
    "StandBottle": {"FTP": 0.900, "pi05": 0.850, "XVLA": 0.150},
}
wm_mix = {
    "PlaceShoes": {"FTP": 0.963, "pi05": 0.900, "XVLA": 0.925},
    "PlacePhone": {"FTP": 0.975, "pi05": 0.925, "XVLA": 0.700},
    "StackBowls": {"FTP": 0.800, "pi05": 0.950, "XVLA": 0.250},
    "StandBottle": {"FTP": 1.000, "pi05": 0.800, "XVLA": 0.400},
}
paper_t3 = {
    "WM-Robot": {"pearson": [0.400, 0.945, 0.906, 0.334], "mmrv": [0.033, 0, 0, 0.067]},
    "WM-Mix": {"pearson": [0.972, 0.939, 0.889, 0.577], "mmrv": [0, 0, 0, 0.067]},
}


def mmrv(R, S):
    pols = list(R)
    total = 0.0
    for i in pols:
        best = 0.0
        for j in pols:
            if i == j:
                continue
            v = (R[i] - R[j]) * (S[i] - S[j]) < 0
            if v:
                best = max(best, abs(R[i] - R[j]))
        total += best
    return total / len(pols)


t3 = {}
for name, sim in [("WM-Robot", wm_robot), ("WM-Mix", wm_mix)]:
    pear, mm, pvals = [], [], []
    for task in real:
        R = real[task]
        S = sim[task]
        r = np.corrcoef([R[p] for p in R], [S[p] for p in S])[0, 1]
        pear.append(round(float(r), 3))
        pvals.append(round(pearson_p_n3(float(r)), 3))
        mm.append(round(mmrv(R, S), 4))
    t3[name] = {
        "pearson_recomputed": pear, "pearson_paper": paper_t3[name]["pearson"],
        "pearson_mean": round(float(np.mean(pear)), 3),
        "p_two_sided_df1": pvals,
        "mmrv_recomputed": mm, "mmrv_paper": paper_t3[name]["mmrv"],
        "mmrv_mean": round(float(np.mean(mm)), 4),
    }
res["table3"] = t3

# n=3 상관의 임계값: df=1 에서 p<0.05 가 되려면 |r| >= 0.997
res["n3_critical_r_p05"] = 0.997

# ---- 12쌍 pooled Pearson (WM-Robot vs WM-Mix) ----
a = np.array([wm_robot[t][p] for t in real for p in real[t]])
b = np.array([wm_mix[t][p] for t in real for p in real[t]])
res["pooled_r_robot_vs_mix"] = round(float(np.corrcoef(a, b)[0, 1]), 3)
# 과제 평균을 뺀(within-task) 상관: 과제 간 차이를 제거하면 얼마나 줄어드는지
ar = np.concatenate([np.array([wm_robot[t][p] for p in real[t]]) - np.mean([wm_robot[t][p] for p in real[t]]) for t in real])
br = np.concatenate([np.array([wm_mix[t][p] for p in real[t]]) - np.mean([wm_mix[t][p] for p in real[t]]) for t in real])
res["within_task_r_robot_vs_mix"] = round(float(np.corrcoef(ar, br)[0, 1]), 3)

# ---- Table IV: 정책 학습 실 데이터 대 합성 반반 ----
t4 = {
    "FTP": {"real": [0.725, 0.800, 0.500, 0.900], "+WM-Robot": [0.750, 0.850, 0.350, 0.800], "+WM-Mix": [0.775, 0.700, 0.500, 0.800]},
    "pi05": {"real": [0.450, 0.700, 0.850, 0.500], "+WM-Robot": [0.400, 0.550, 0.800, 0.500], "+WM-Mix": [0.375, 0.350, 0.850, 0.400]},
    "XVLA": {"real": [0.500, 0.600, 0.300, 0.600], "+WM-Robot": [0.675, 0.300, 0.450, 0.600], "+WM-Mix": [0.550, 0.600, 0.450, 0.000]},
}
res["table4_means"] = {p: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for p, d in t4.items()}

# 실 데이터 대 반반 차이의 불확실성: 롤아웃 10개, 점수 SD를 가정 (논문에 SD 보고 없음)
sens = {}
for sd in (0.15, 0.25, 0.35):
    se_cell = sd / np.sqrt(10)  # 셀 하나(정책 x 과제, 롤아웃 10개)의 표준오차
    se_mean = se_cell / np.sqrt(4)  # 4과제 평균
    se_diff = np.sqrt(2) * se_mean  # 두 정책 평균 차이
    sens[str(sd)] = {"se_mean_4task": round(float(se_mean), 3), "se_diff": round(float(se_diff), 3)}
res["table4_se_sensitivity"] = sens
res["table4_diff_pi05_real_minus_mix"] = round(0.625 - 0.494, 3)
res["table4_diff_ftp_real_minus_mix"] = round(0.731 - 0.694, 3)
res["table4_diff_xvla_real_minus_mix"] = round(0.500 - 0.400, 3)

# ---- 시뮬레이터 점수의 편향: 같은 정책·과제에서 imagined - real ----
bias = {}
for name, sim in [("WM-Robot", wm_robot), ("WM-Mix", wm_mix)]:
    d = [sim[t][p] - real[t][p] for t in real for p in real[t]]
    bias[name] = {"mean_signed": round(float(np.mean(d)), 3), "mean_abs": round(float(np.mean(np.abs(d))), 3),
                  "n_over": int(sum(x > 0 for x in d)), "n_cells": len(d)}
res["sim_minus_real_bias"] = bias

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
print(json.dumps(res, ensure_ascii=False, indent=2))
