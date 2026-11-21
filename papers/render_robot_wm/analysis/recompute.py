"""Doppel (arXiv:2611.24925 v1) 논문 표 수치를 다시 계산하고 일관성을 점검한다.

입력: 논문 Table 2, 4, 1, 10, 13, 16, 17과 부록 설정(Table 8, 9)의 값을 아래에 옮겨 적었다(출처 주석).
외부 데이터, 가중치, 재학습은 쓰지 않는다. 출력: analysis/recompute_results.json.
실행: python3 analysis/recompute.py   (numpy, scipy)

가정(논문에 없는 것은 표시한다)
- Table 10의 "±"는 표준오차(SE)로 읽는다(논문 표 캡션). 클립 단위인지 에피소드 단위인지는 밝히지 않는다.
- Table 10의 Better(= Doppel-8 PSNR이 더 높은 클립 비율)는 클립 200개 기준 이항 검정에 쓴다(동점 무시).
- 평균 이득은 3개 로봇 데이터셋의 단순 평균이다(논문 초록의 1.00 dB, 1.20 dB가 이 정의와 맞는지 확인).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}

# ---- Table 2 (validation = unseen trajectories, OOD test). PSNR only. -------------
# (action-only, Doppel single-dataset expert, Doppel-3, Doppel-8)
T2_PSNR = {
    "Droid":    {"val": (22.888, 24.357, 24.297, 24.819), "ood": (20.069, 21.276, 21.250, 21.460)},
    "Bridge":   {"val": (26.290, 26.618, 26.551, 26.544), "ood": (20.584, 21.722, 22.474, 22.238)},
    "RoboMind": {"val": (30.474, 31.104, 31.065, 31.302), "ood": (30.098, 30.290, 30.442, 30.659)},
}
# SSIM / LPIPS / SCS not needed for the headline check.

def mean_gain(split, idx):
    g = {d: T2_PSNR[d][split][idx] - T2_PSNR[d][split][0] for d in T2_PSNR}
    return {"per_dataset_db": {k: round(v, 3) for k, v in g.items()},
            "mean_db": round(float(np.mean(list(g.values()))), 3)}

R["headline_gain_vs_action_only"] = {
    "Doppel_single_expert": {"val": mean_gain("val", 1), "ood": mean_gain("ood", 1)},
    "Doppel-3":             {"val": mean_gain("val", 2), "ood": mean_gain("ood", 2)},
    "Doppel-8":             {"val": mean_gain("val", 3), "ood": mean_gain("ood", 3)},
}
# Abstract: 1.00 dB (val) and 1.20 dB (OOD) "across three robot-arm datasets". Matches Doppel-8 only.

# ---- Table 2 Doppel-3 vs Doppel-8 (final checkpoint), and Table 10 paired stats --
T10 = [  # dataset, split, d_psnr, se, better_k(of 200), d_arm, d_obj
    ("Droid", "val", 0.52, 0.03, 184, 0.70, 0.69),
    ("Droid", "ood", 0.21, 0.03, 146, 0.34, 0.13),
    ("Bridge", "val", -0.01, 0.06, 114, 0.09, 0.12),
    ("Bridge", "ood", -0.24, 0.05, 76, -0.01, -0.08),
    ("RoboMind", "val", 0.24, 0.04, 150, 0.42, 0.14),
    ("RoboMind", "ood", 0.22, 0.04, 144, 0.34, 0.16),
]
rows = []
for ds, sp, d, se, k, da, do in T10:
    z = d / se
    lo, hi = d - 1.96 * se, d + 1.96 * se
    p_sign = stats.binomtest(k, 200, 0.5).pvalue
    rows.append({"dataset": ds, "split": sp, "d_psnr": d, "se": se,
                 "ci95": [round(lo, 3), round(hi, 3)], "z": round(z, 1),
                 "better_share": round(k / 200, 3), "sign_test_p": float(f"{p_sign:.2g}"),
                 "d_arm": da, "d_obj": do})
R["table10_paired_doppel8_vs_doppel3"] = rows
R["table10_note"] = ("SE가 클립 단위면 같은 에피소드의 창(window)이 독립이 아니어서 과소 추정될 가능성이 크다. "
                     "논문 부록 A.10.3은 에피소드 단위 5000회 부트스트랩을 말하지만 표에는 SE만 있다.")

# ---- Table 13 consistency: region deltas vs Table 10 deltas (Droid val, Bridge OOD) --
T13 = {  # PSNRarm, PSNRobj for Doppel-3 and Doppel-8 (val / ood)
    ("Droid", "val"): ((20.89, 28.56), (21.59, 29.25)),
    ("Droid", "ood"): ((19.23, 26.41), (19.57, 26.54)),
    ("Bridge", "val"): ((21.49, 27.84), (21.57, 27.96)),
    ("Bridge", "ood"): ((19.05, 24.93), (19.04, 24.85)),
    ("RoboMind", "val"): ((26.81, 29.46), (27.23, 29.60)),
    ("RoboMind", "ood"): ((25.30, 28.91), (25.64, 29.07)),
}
R["table13_vs_table10_region_delta"] = {
    f"{k[0]}-{k[1]}": {"d_arm_recomputed": round(v[1][0] - v[0][0], 2),
                       "d_obj_recomputed": round(v[1][1] - v[0][1], 2)}
    for k, v in T13.items()
}
# Bridge OOD moving-object claim in Sec.6: action-only 21.50 -> Doppel-8 24.85 (PSNRobj), SCS 0.778 -> 0.831.
R["bridge_ood_moving_object_claim"] = {
    "psnr_obj_gain_vs_action_only": round(24.85 - 21.50, 2),
    "scs_obj_gain_vs_action_only": round(0.831 - 0.778, 3),
    "note": "Doppel 단일 전문가(0.838)와 Doppel-3(0.843)의 SCSobj가 Doppel-8(0.831)보다 높다. 주장은 action-only 대비만 맞다.",
}

# ---- Draw efficiency: Table 16 (PSNR) and Table 17 (SCS), batch-64 shared models --
T16 = {  # method -> list over 6 sets (Droid val, Droid ood, Bridge val, Bridge ood, RM val, RM ood)
    "action_only_3p4M": [20.959, 19.036, 24.070, 21.873, 27.564, 28.200],
    "doppel3_b64_1p2M": [22.070, 20.251, 24.098, 22.781, 28.504, 29.082],
    "doppel3_b64_3p4M": [23.188, 20.862, 25.431, 22.726, 29.937, 29.818],
}
T17 = {
    "action_only_3p4M": [0.901, 0.846, 0.925, 0.890, 0.920, 0.916],
    "doppel3_b64_1p2M": [0.919, 0.883, 0.927, 0.914, 0.927, 0.932],
}
a, b = np.array(T16["action_only_3p4M"]), np.array(T16["doppel3_b64_1p2M"])
R["draw_efficiency"] = {
    "reduction_pct_from_1216k_vs_3356k": round(100 * (1 - 1.216 / 3.356), 1),
    "reduction_pct_from_table_labels_1p2_vs_3p4": round(100 * (1 - 1.2 / 3.4), 1),
    "mean_psnr_doppel_1p2M": round(float(b.mean()), 3),
    "mean_psnr_actiononly_3p4M": round(float(a.mean()), 3),
    "all6_psnr_doppel_1p2_beats_ao_3p4": bool(np.all(b > a)),
    "min_psnr_margin_db": round(float((b - a).min()), 3),
    "mean_scs_doppel_1p2M": round(float(np.mean(T17["doppel3_b64_1p2M"])), 4),
    "mean_scs_actiononly_3p4M": round(float(np.mean(T17["action_only_3p4M"])), 4),
    "all6_scs_doppel_1p2_beats_ao_3p4": bool(np.all(np.array(T17["doppel3_b64_1p2M"]) > np.array(T17["action_only_3p4M"]))),
    "note": "Bridge val 여유는 +0.03 dB, +0.002 SCS로 매우 작다. 학습 실행은 seed 반복 없이 하나씩이다.",
}

# ---- Table 1 / Table 4 / config consistency ----------------------------------------
clips = {"Droid": 83511, "Bridge": 61935, "RoboMind": 74005, "ABC": 274523,
         "T-Rex": 58120, "Ego-Exo4D": 213292, "GigaHands": 31218, "OakInk2": 6998}
episodes = {"Droid": 16726, "Bridge": 12387, "RoboMind": 14801, "ABC": 9115,
            "T-Rex": 4784, "Ego-Exo4D": 2041, "GigaHands": 8491, "OakInk2": 403}
R["table1_sums"] = {
    "robot3_clips": clips["Droid"] + clips["Bridge"] + clips["RoboMind"],
    "all8_clips": sum(clips.values()),
    "robot3_episodes": episodes["Droid"] + episodes["Bridge"] + episodes["RoboMind"],
    "all8_episodes": sum(episodes.values()),
    "paper_states": {"robot3_clips": 219451, "all8_clips": 803602, "robot3_episodes": 43914, "all8_episodes": 68748},
}
# Doppel-8 sampler weights (release config emb8_render_action_4node.json) vs clip proportions.
cfg_w = {"abc": 0.341615, "bridge": 0.077072, "droid": 0.103921, "egoexo4d": 0.26542,
         "gigahands": 0.038848, "oakink2": 0.008708, "robomind": 0.092092, "trex": 0.072324}
tot = sum(clips.values())
prop = {"abc": clips["ABC"], "bridge": clips["Bridge"], "droid": clips["Droid"], "egoexo4d": clips["Ego-Exo4D"],
        "gigahands": clips["GigaHands"], "oakink2": clips["OakInk2"], "robomind": clips["RoboMind"], "trex": clips["T-Rex"]}
R["emb8_sampler_vs_clip_share_max_abs_diff"] = round(max(abs(cfg_w[k] - prop[k] / tot) for k in cfg_w), 6)
R["emb8_sampler_weight_sum"] = round(sum(cfg_w.values()), 6)

# Table 4 row sums (train + val + test) and totals
T4 = {"Droid": (98.9, 5.4, 11.6, 115.9), "Bridge": (26.4, 1.4, 3.2, 31.1), "RoboMind": (62.1, 3.2, 6.8, 72.1),
      "ABC": (242.6, 12.7, 31.4, 286.7), "T-Rex": (44.2, 2.4, 3.9, 50.5), "Ego-Exo4D": (88.3, 5.1, 10.4, 103.8),
      "GigaHands": (25.4, 1.2, 2.1, 28.6), "OakInk2": (5.2, 1.2, 1.5, 7.9)}
R["table4_row_check"] = {k: round(v[0] + v[1] + v[2] - v[3], 2) for k, v in T4.items()}
R["table4_totals"] = {"train": round(sum(v[0] for v in T4.values()), 1), "val": round(sum(v[1] for v in T4.values()), 1),
                      "test": round(sum(v[2] for v in T4.values()), 1), "total": round(sum(v[3] for v in T4.values()), 1),
                      "paper": {"train": 593.1, "val": 32.6, "test": 70.8, "total": 696.6}}

# ---- Training budget implied by configs (Table 8/9, release configs) ----------------
steps_doppel3 = 651 * 19456 // 256          # epochs * epoch_len / global batch
steps_doppel8 = 2490 * 19456 // 256
R["steps"] = {"doppel3_steps": steps_doppel3, "doppel8_steps": steps_doppel8,
              "paper": {"doppel3": 49500, "doppel8": 189000}}
draws3 = steps_doppel3 * 256
R["doppel3_draws_total"] = draws3
R["doppel3_implied_passes_uniform_sampling"] = {
    k: round(draws3 / 3 / clips[k], 1) for k in ["Droid", "Bridge", "RoboMind"]}
R["doppel3_implied_passes_note"] = ("균등 샘플링(각 1/3)이면 데이터셋마다 pass 수가 다르다. 논문의 '약 51 pass'와 "
                                    "Bridge(약 68)/Droid(약 51) 차이가 난다. 'pass'의 정의(중복 샘플 포함 여부)에 따라 달라진다.")
R["doppel8_passes_proportional"] = round(steps_doppel8 * 256 / tot, 1)

# ---- Parameter counts from the architecture described in App. A.4/A.5 (hand arithmetic) --
def linear(i, o):
    return i * o + o
def ln(d):
    return 2 * d
def mha(d):
    return 3 * d * d + 3 * d + d * d + d
def ffn(d, m=4):
    return linear(d, d * m) + linear(d * m, d)
d = 512
self_block = ln(d) + mha(d) + ln(d) + ffn(d)
cross_block = ln(d) + ln(d) + mha(d) + ln(d) + ffn(d)
render_enc = (16 * 256 * 1 * 3 * 3 + 256) + (256 * 512 * 27 + 512) + (512 * 512 * 1 * 3 * 3 + 512) \
             + (32 + 32 + 64) * d + 2 * self_block
action_enc = linear(159, d) + linear(d, d) + 128 * d
fusion = 2 * cross_block
to_temb = ln(d) + linear(d, 2048) + linear(2048, 5120)
adapter = render_enc + action_enc + fusion + to_temb
R["params"] = {"render_action_adapter_M": round(adapter / 1e6, 3), "paper_M": 30.5,
               "action_only_adapter_M_by_formula": round((linear(8, d) + linear(d, d) + linear(d, d)
                                                          + (512 * 512 * 4 + 512) + linear(d, 5120)) / 1e6, 3),
               "paper_action_only_M": 4.2}

# ---- Sampling check: 17-frame window -> 16 predicted frames; 25.6 s rollout ----------
R["rollout"] = {"windows_for_25p6s": round(25.6 / 3.2), "frames_out": 1 + 16 * 8,
                "fps_implied": round(16 / 3.2, 2)}

OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False))
print(json.dumps(R, indent=2, ensure_ascii=False))
