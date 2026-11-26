"""WM-Craftnet (arXiv:2609.07002 v1) 표 수치 재계산과 검정.

입력은 논문 표(Table 1-6, 9, 10)에서 옮긴 값이다. 시행 수(n)는 논문이 말한 것만 쓰고,
말하지 않은 것은 문장 안에 "(n 가정)"으로 밝힌다.
실행: python3 analysis/recompute.py   (수 초, numpy와 scipy만 사용)
결과: analysis/recompute_results.json
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------------------------------------------------------------- 표 값 (평균, 95% CI 반폭)
# 열: Return, EpLen, RotR, OffAxis, AngVar
T1 = {
    "Open-loop Replay": [(306.3, 48.3), (320.2, 35.0), (0.332, 0.080), (1.789, 0.295), (2.241, 0.679)],
    "Blind RL": [(304.1, 3.8), (295.8, 2.3), (0.944, 0.004), (1.539, 0.007), (1.810, 0.011)],
    "Touch Dexterity": [(386.9, 3.1), (362.5, 2.3), (1.018, 0.004), (1.601, 0.008), (1.764, 0.053)],
    "RL (depth, no tactile)": [(362.2, 4.7), (351.0, 3.6), (0.978, 0.004), (1.806, 0.009), (2.222, 0.065)],
    "In-Hand Rotation": [(236.1, 2.7), (272.5, 2.8), (0.882, 0.003), (1.986, 0.009), (2.487, 0.074)],
    "WM scratch": [(414.3, 15.7), (264.2, 8.8), (0.742, 0.012), (1.285, 0.010), (1.735, 0.052)],
    "WM all-zero tactile": [(724.9, 4.4), (428.8, 2.8), (1.252, 0.004), (1.224, 0.006), (1.388, 0.031)],
    "WM 75% tactile dropout": [(735.2, 4.1), (429.8, 2.7), (1.272, 0.004), (1.224, 0.006), (1.390, 0.031)],
    "WM 50% tactile dropout": [(737.8, 4.4), (425.1, 3.6), (1.276, 0.004), (1.279, 0.006), (1.511, 0.061)],
    "WM 25% tactile dropout": [(743.4, 4.2), (427.9, 3.2), (1.287, 0.004), (1.243, 0.006), (1.434, 0.032)],
    "WM pretrained prop-only": [(684.5, 4.3), (424.0, 2.2), (1.191, 0.003), (1.310, 0.007), (1.234, 0.013)],
    "WM pretrained prop+tac": [(698.9, 4.3), (435.4, 1.9), (1.192, 0.002), (1.165, 0.006), (1.373, 0.024)],
    "WM full WSM": [(753.3, 3.6), (435.2, 1.9), (1.293, 0.004), (1.225, 0.006), (1.324, 0.027)],
}
T2 = {
    "Noisy-depth supervision": [(708.0, 3.5), (431.1, 2.0), (1.265, 0.005), (1.299, 0.006), (1.456, 0.045)],
    "No h_{t-1} recurrent input": [(705.4, 4.3), (434.3, 2.3), (1.282, 0.005), (1.125, 0.006), (1.185, 0.008)],
    "Encoder-decoder z_t policy": [(667.6, 4.7), (417.4, 2.6), (1.254, 0.003), (1.323, 0.006), (1.343, 0.018)],
    "Full WSM": T1["WM full WSM"],
}
T9 = {
    "LSTM": [(621.0, 4.3), (424.3, 2.7), (1.186, 0.004), (1.297, 0.006), (1.392, 0.028)],
    "GRU": [(497.8, 4.5), (377.1, 2.5), (1.095, 0.003), (1.226, 0.007), (1.357, 0.013)],
    "Full WSM": T1["WM full WSM"],
}
T10 = {
    "No aux heads": [(688.1, 3.7), (433.7, 2.0), (1.239, 0.006), (1.229, 0.005), (1.778, 0.138)],
    "Only value": [(737.0, 4.9), (435.7, 2.2), (1.291, 0.003), (1.258, 0.007), (1.346, 0.045)],
    "Only pose": [(690.0, 5.8), (432.3, 2.5), (1.221, 0.007), (1.228, 0.008), (1.231, 0.011)],
    "Only shape": [(689.2, 4.2), (428.1, 2.2), (1.261, 0.002), (1.255, 0.006), (1.535, 0.032)],
    "Full WSM": T1["WM full WSM"],
}
COLS = ["Return", "EpLen", "RotR", "OffAxis", "AngVar"]
Z = 1.96


def zdiff(a, b, n_a=128, n_b=128):
    """두 평균의 차이 검정. CI 반폭 h = 1.96*SE 로 보고 SE를 복원한다 (n은 CI 계산에 쓰인 시행 수와 무관, SE만 쓴다)."""
    (ma, ha), (mb, hb) = a, b
    se = math.sqrt((ha / Z) ** 2 + (hb / Z) ** 2)
    z = (ma - mb) / se
    return {"diff": round(ma - mb, 4), "se": round(se, 4), "z": round(z, 2), "p": float(2 * stats.norm.sf(abs(z)))}


# ---------------------------------------------------------------- V1: 본문 퍼센트 주장
v = {}
v["tab6_rotr_gain_pct"] = {
    "x": round((1.002 / 0.923 - 1) * 100, 2), "y": round((0.715 / 0.378 - 1) * 100, 2), "z": round((0.842 / 0.696 - 1) * 100, 2),
    "paper": {"x": 8.6, "y": 89.2, "z": 20.9},
}
v["tab6_return_gain_pct"] = {
    "x": round((333.6 / 246.0 - 1) * 100, 2), "y": round((203.5 / 193.0 - 1) * 100, 2), "z": round((533.1 / 301.7 - 1) * 100, 2),
    "paper": {"x": 35.6, "y": 5.4, "z": 76.7},
}
v["tab4_y"] = {"rotr_gain_pct": round((1.025 / 0.545 - 1) * 100, 2), "eplen_gain_pct": round((215.0 / 192.4 - 1) * 100, 2),
               "paper": {"rotr": 88.1, "eplen": 11.7}}
v["tab6_recovery"] = {"abs_gain_pp_from_rounded": round(14.1 - 6.2, 2), "abs_gain_pp_from_counts": round((18 - 8) / 128 * 100, 2),
                      "time_reduction_s": round(3.14 - 2.56, 2), "paper": {"pp": 7.9, "s": 0.58}}
v["tab2_clean_vs_noisy_return"] = round(753.3 - 708.0, 1)
res["V1_percent_claims"] = v

# ---------------------------------------------------------------- V2: 복구율의 정수 분모 (Table 6 Sim. Recovery)
rec = {"Blind RL": (4.7, 3.7), "Touch Dexterity": (6.2, 4.2), "RL (no tactile)": (5.5, 4.0), "In-Hand Rotation": (0.0, 0.0),
       "WM-Craftnet": (14.1, 6.0)}
rec_out = {}
for name, (pct, ci) in rec.items():
    hits = []
    for n in range(20, 600):
        k = round(pct / 100 * n)
        if abs(100 * k / n - pct) <= 0.051:
            p = k / n
            half = 100 * Z * math.sqrt(p * (1 - p) / n)
            if abs(half - ci) <= 0.051:
                hits.append((n, k))
    p128 = round(pct / 100 * 128)
    p = p128 / 128
    rec_out[name] = {
        "n_k_matching_both_pct_and_ci_(n<600)": hits[:12],
        "k_at_n128": p128, "pct_at_n128": round(100 * p128 / 128, 2),
        "normal_ci_half_at_n128": round(100 * Z * math.sqrt(p * (1 - p) / 128), 2), "paper_ci_half": ci,
    }
res["V2_recovery_integer_check"] = rec_out

# 복구율 검정 (n=128 가정, k = 18, 8, 7, 6, 0)
k = {"WM-Craftnet": 18, "Touch Dexterity": 8, "RL (no tactile)": 7, "Blind RL": 6, "In-Hand Rotation": 0}
tests = {}
for name, kk in k.items():
    if name == "WM-Craftnet":
        continue
    _, p = stats.fisher_exact([[18, 128 - 18], [kk, 128 - kk]])
    tests[f"WM vs {name}"] = {"k": f"18/128 vs {kk}/128", "fisher_p": float(p)}
res["V2_recovery_fisher_n128"] = tests


def wilson(kk, n, z=1.96):
    p = kk / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


res["V2_recovery_wilson_n128"] = {n_: wilson(kk, 128) for n_, kk in k.items()}

# 복구 시간: 복구된 시행 수(18, 8, 7, 6)만 평균에 들어간다. ±의 정의를 모르므로 두 해석을 모두 계산한다.
tm = {"WM-Craftnet": (2.56, 0.77, 18), "RL (no tactile)": (3.14, 0.79, 7), "Touch Dexterity": (3.33, 1.41, 8), "Blind RL": (3.62, 2.25, 6)}
t_out = {}
for name in ["RL (no tactile)", "Touch Dexterity", "Blind RL"]:
    m1, s1, n1 = tm["WM-Craftnet"]
    m2, s2, n2 = tm[name]
    for label, f1, f2 in (("pm_is_SD", s1, s2), ("pm_is_CI95", s1 * math.sqrt(n1) / Z, s2 * math.sqrt(n2) / Z)):
        t, p = stats.ttest_ind_from_stats(m1, f1, n1, m2, f2, n2, equal_var=False)
        t_out[f"WM vs {name} ({label})"] = {"t": round(float(t), 2), "p": round(float(p), 3)}
res["V3_recovery_time_welch"] = t_out

# ---------------------------------------------------------------- V4: 실로봇 Table 6 (200회) Wilson/Fisher
real = {"Blind RL": 41, "Touch Dexterity": 33, "RL (no tactile)": 45, "In-Hand Rotation": 53, "WM-Craftnet": 175}
r_out = {}
for name, kk in real.items():
    r_out[name] = {"rate": round(kk / 200, 4), "wilson95": wilson(kk, 200)}
    if name != "WM-Craftnet":
        _, p = stats.fisher_exact([[175, 25], [kk, 200 - kk]])
        r_out[name]["fisher_p_vs_WM"] = float(p)
res["V4_real_table6"] = r_out
_, p = stats.fisher_exact([[53, 147], [33, 167]])
res["V4_real_table6"]["IHR_vs_TD_fisher_p"] = float(p)
_, p = stats.fisher_exact([[45, 155], [33, 167]])
res["V4_real_table6"]["RLnoTac_vs_TD_fisher_p"] = float(p)

# ---------------------------------------------------------------- V5: 실로봇 Table 3 (10회 x 4 물체) : 정수 가능성과 Fisher
T3 = {  # (RR, k/10) 물체 순서: duck, cross, corner, double-notched
    "OlR": [(2.32, 6), (3.04, 6), (5.19, 7), (0.23, 0)],
    "TD": [(1.54, 4), (0.34, 2), (0.22, 0), (0.27, 0)],
    "IHR": [(1.83, 5), (0.45, 1), (0.36, 1), (0.39, 0)],
    "IHR+WSM-denoised": [(2.76, 8), (1.21, 8), (2.43, 10), (0.80, 0)],
    "Ours": [(16.179, 10), (8.01, 10), (8.48, 10), (4.32, 8)],
}
objs = ["duck", "cross", "corner", "double-notched(unseen)"]
feas = []
for m, row in T3.items():
    for o, (rr, kk) in zip(objs, row):
        # 성공을 '반 바퀴 이상 = pi rad 이상'으로 읽으면 평균 RR >= kk*pi/10 이어야 한다
        lo = kk * math.pi / 10
        feas.append({"method": m, "object": o, "RR": rr, "k": kk, "min_mean_RR_if_halfspin_is_pi": round(lo, 3),
                     "feasible": rr >= lo - 1e-9})
res["V5_table3_feasibility_halfspin_pi"] = feas
res["V5_table3_infeasible_cells"] = [f for f in feas if not f["feasible"]]
f_out = {}
for j, o in enumerate(objs):
    for m in ["OlR", "TD", "IHR", "IHR+WSM-denoised"]:
        kk = T3[m][j][1]
        _, p = stats.fisher_exact([[T3["Ours"][j][1], 10 - T3["Ours"][j][1]], [kk, 10 - kk]])
        f_out[f"{o}: Ours vs {m}"] = {"k": f"{T3['Ours'][j][1]}/10 vs {kk}/10", "fisher_p": round(float(p), 4)}
res["V5_table3_fisher_per_object"] = f_out
pool = {}
for m in ["OlR", "TD", "IHR", "IHR+WSM-denoised"]:
    kk = sum(x[1] for x in T3[m])
    _, p = stats.fisher_exact([[38, 2], [kk, 40 - kk]])
    pool[m] = {"k": f"38/40 vs {kk}/40", "fisher_p": float(p)}
res["V5_table3_pooled_4objects"] = pool
den = {}
for j, o in enumerate(objs[:3]):
    a, b = T3["IHR+WSM-denoised"][j][1], T3["IHR"][j][1]
    den[o] = {"k": f"{a}/10 vs {b}/10", "fisher_p": round(float(stats.fisher_exact([[a, 10 - a], [b, 10 - b]])[1]), 4)}
a = sum(T3["IHR+WSM-denoised"][j][1] for j in range(3))
b = sum(T3["IHR"][j][1] for j in range(3))
den["pooled_3_seen_objects"] = {"k": f"{a}/30 vs {b}/30", "fisher_p": float(stats.fisher_exact([[a, 30 - a], [b, 30 - b]])[1])}
a2 = sum(T3["Ours"][j][1] for j in range(3))
den["Ours_vs_IHR+WSM_pooled_3_seen_objects"] = {"k": f"{a2}/30 vs {a}/30", "fisher_p": round(float(stats.fisher_exact([[a2, 30 - a2], [a, 30 - a]])[1]), 4)}
res["V5_denoised_depth_effect"] = den
# Table 3 풀링 성공률과 Table 6 성공률의 일치
res["V5_table3_vs_table6"] = {
    "TD": {"table3_pooled": "6/40=15.0%", "table6": "33/200=16.5%"},
    "IHR": {"table3_pooled": "7/40=17.5%", "table6": "53/200=26.5%"},
    "Ours": {"table3_pooled": "38/40=95.0%", "table6": "175/200=87.5%",
             "rest_160_trials_if_table3_is_subset": "137/160=85.6% (Table 3의 네 물체가 Table 6의 부분집합이라고 가정)"},
}

# ---------------------------------------------------------------- V6: CI에서 복원한 시행 단위 SD와 변동계수
sd = {}
for name, row in T1.items():
    if name in ("Open-loop Replay",):
        continue
    ret_m, ret_h = row[0]
    ep_m, ep_h = row[1]
    for n in (128, 1024):
        sd.setdefault(name, {})[f"n{n}"] = {
            "Return_SD": round(ret_h * math.sqrt(n) / Z, 1), "Return_CV_pct": round(100 * ret_h * math.sqrt(n) / Z / ret_m, 1),
            "EpLen_SD": round(ep_h * math.sqrt(n) / Z, 1),
        }
res["V6_implied_SD_from_CI_table1"] = sd
res["V6_note"] = ("공식 코드의 평가 집계기는 1.96*SD/sqrt(n)로 CI를 만들고 기본 창(window)이 1024 에피소드다. "
                  "논문은 128 에피소드라고 쓴다. 두 가정에서 복원되는 에피소드 SD가 다르다.")

# ---------------------------------------------------------------- V7: 표 안 쌍 비교 (CI로 복원한 SE, 평가 에피소드 변동만 반영)
pairs = {}


def add(label, a, b, ci_idx):
    pairs[label] = {COLS[ci_idx]: zdiff(a[ci_idx], b[ci_idx])}


add("T1 full vs scratch", T1["WM full WSM"], T1["WM scratch"], 0)
add("T1 full vs all-zero tactile", T1["WM full WSM"], T1["WM all-zero tactile"], 0)
add("T1 full vs 25% dropout", T1["WM full WSM"], T1["WM 25% tactile dropout"], 0)
add("T1 25% vs all-zero", T1["WM 25% tactile dropout"], T1["WM all-zero tactile"], 0)
add("T1 prop+tac vs prop-only", T1["WM pretrained prop+tac"], T1["WM pretrained prop-only"], 0)
add("T1 full vs prop+tac", T1["WM full WSM"], T1["WM pretrained prop+tac"], 0)
add("T1 scratch vs Touch Dexterity (Return)", T1["WM scratch"], T1["Touch Dexterity"], 0)
add("T1 scratch vs Touch Dexterity (AngVar)", T1["WM scratch"], T1["Touch Dexterity"], 4)
add("T1 full vs all-zero tactile (OffAxis)", T1["WM full WSM"], T1["WM all-zero tactile"], 3)
add("T2 full vs noisy-depth", T2["Full WSM"], T2["Noisy-depth supervision"], 0)
add("T2 full vs no h_{t-1}", T2["Full WSM"], T2["No h_{t-1} recurrent input"], 0)
add("T2 full vs enc-dec z_t", T2["Full WSM"], T2["Encoder-decoder z_t policy"], 0)
add("T2 no h_{t-1} vs full (RotR)", T2["No h_{t-1} recurrent input"], T2["Full WSM"], 2)
add("T9 full vs LSTM", T9["Full WSM"], T9["LSTM"], 0)
add("T9 full vs GRU (AngVar)", T9["Full WSM"], T9["GRU"], 4)
add("T9 full vs LSTM (AngVar)", T9["Full WSM"], T9["LSTM"], 4)
add("T10 only value vs no-aux", T10["Only value"], T10["No aux heads"], 0)
add("T10 full vs only value", T10["Full WSM"], T10["Only value"], 0)
add("T10 only pose vs no-aux", T10["Only pose"], T10["No aux heads"], 0)
add("T10 only shape vs no-aux", T10["Only shape"], T10["No aux heads"], 0)
add("T10 only value vs full (RotR)", T10["Only value"], T10["Full WSM"], 2)
res["V7_pairwise_from_CI"] = pairs
res["V7_note"] = ("SE는 표의 95% CI 반폭/1.96으로 복원했다. 이 SE는 한 번 학습한 정책의 평가 에피소드 변동만 담는다. "
                  "학습 시드 수와 시드 간 변동은 논문에 없다.")

# ---------------------------------------------------------------- V8: 표 간 일관성 (같은 방법이 여러 표에 나올 때)
cons = {"full_WSM_identical_across_T1_T2_T9_T10": all(T1["WM full WSM"] == x["Full WSM"] for x in (T2, T9, T10))}
# 학습 물체 대 보류(held-out) 물체의 Return 비 (Table 1, 4, 5 대 Table 6)
tr = {  # (train Return, train RotR, held Return, held RotR)
    "x": {"Blind RL": (44.0, 0.867, 134.7, 0.852), "Touch Dexterity": (37.8, 1.023, 35.8, 0.923), "RL (no tactile)": (38.2, 1.079, 33.8, 0.853),
          "In-Hand Rotation": (295.1, 0.839, 246.0, 0.732), "WM-Craftnet": (432.8, 1.114, 333.6, 1.002)},
    "y": {"Blind RL": (177.8, 0.545, 135.7, 0.300), "Touch Dexterity": (105.3, 0.297, 78.7, 0.174), "RL (no tactile)": (154.8, 0.288, 168.9, 0.378),
          "In-Hand Rotation": (142.8, 0.377, 193.0, 0.309), "WM-Craftnet": (272.7, 1.025, 203.5, 0.715)},
    "z": {"Blind RL": (304.1, 0.944, 277.1, 0.665), "Touch Dexterity": (386.9, 1.018, 242.2, 0.605), "RL (no tactile)": (362.2, 0.978, 301.7, 0.696),
          "In-Hand Rotation": (236.1, 0.882, 156.1, 0.468), "WM-Craftnet": (753.3, 1.293, 533.1, 0.842)},
}
ratio = {}
for ax, d in tr.items():
    for m, (r0, q0, r1, q1) in d.items():
        ratio[f"{ax}:{m}"] = {"return_held/train": round(r1 / r0, 2), "rotr_held/train": round(q1 / q0, 2)}
cons["held_out_over_train_ratio"] = ratio
cons["held_out_return_ratio_gt_1"] = [k_ for k_, v_ in ratio.items() if v_["return_held/train"] > 1.0]
res["V8_cross_table"] = cons

# RotR 단위: 코드는 RotR = 에피소드 평균 [clip(20*dtheta, -6.28, 2.0) * spin_coef(1.0)] (dtheta: 제어 한 스텝의 축 회전 rad)
rot = {}
for name, val in [("Full WSM (T1)", 1.293), ("scratch (T1)", 0.742), ("y WM (T4)", 1.025), ("x WM (T5)", 1.114)]:
    rad_per_step = val / 20.0
    rot[name] = {"RotR": val, "rad_per_s_at_10Hz": round(rad_per_step * 10, 3), "rad_per_20s": round(rad_per_step * 10 * 20, 2),
                 "rad_per_500step_episode": round(rad_per_step * 500, 2)}
rot["note"] = ("RotR*0.5 = rad/s (제어 10 Hz, 20배 스케일 가정, 클립 영향 무시). 실로봇 Table 3의 'RR'은 20 s당 rad. "
               "49개 물체 결과 '9.37 rad per episode'는 에피소드 길이(20 s 또는 500 스텝=50 s)가 본문에 없어 비교 보류.")
res["V9_rotr_units"] = rot
res["V9_replay_memory"] = {
    "ring_rows": 256, "envs_paper": 1024, "depth_96x72_float32_bytes_each": 256 * 1024 * 72 * 96 * 4,
    "GiB_each(image, image_clean)": round(256 * 1024 * 72 * 96 * 4 / 2 ** 30, 2), "GiB_both": round(2 * 256 * 1024 * 72 * 96 * 4 / 2 ** 30, 2),
    "envs_256_GiB_both": round(2 * 256 * 256 * 72 * 96 * 4 / 2 ** 30, 2),
    "env_steps_per_PPO_epoch(1024x16)": 1024 * 16, "wm_frames_per_epoch(10x16x32)": 10 * 16 * 32,
    "3000_epochs_env_steps": 3000 * 1024 * 16,
}

# 보고된 49-물체 값: 9.37 ± 0.13 rad, 3.28 rad 대 비율 / 낙하율
res["V10_49obj"] = {"ratio_prior_over_noprior": round(9.37 / 3.28, 2), "fall_6pct_to_0.3pct_factor": round(6 / 0.3, 1),
                    "five_block_RotR_gain_pct": {"9obj_vs_scratch": round((1.34 / 1.10 - 1) * 100, 1),
                                                 "49obj_vs_scratch": round((1.46 / 1.10 - 1) * 100, 1)}}

# CoRL spotlight 비율 (conferences/corl2026/papers.json의 집계: 구두 21, 스포트라이트 72, 포스터 594, 총 687)
res["V11_corl_counts"] = {"spotlight": 72, "oral": 21, "poster": 594, "total": 687,
                          "spotlight_share_of_accepted_pct": round(72 / 687 * 100, 1),
                          "oral_plus_spotlight_share_pct": round(93 / 687 * 100, 1),
                          "claimed_top_pct": 4.4,
                          "implied_submissions_if_spotlight_is_top4.4pct": round(72 / 0.044),
                          "implied_submissions_if_oral_plus_spotlight_is_top4.4pct": round(93 / 0.044)}

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
for key in ("V1_percent_claims", "V2_recovery_fisher_n128", "V3_recovery_time_welch", "V4_real_table6",
            "V5_table3_infeasible_cells", "V5_table3_pooled_4objects", "V9_rotr_units", "V9_replay_memory", "V11_corl_counts"):
    print(key, json.dumps(res[key], ensure_ascii=False)[:900])
