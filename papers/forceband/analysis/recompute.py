"""ForceBand (arXiv:2606.26093 v1) 보고 수치 재계산.

입력은 모두 논문 본문·표·그림에 인쇄된 값을 옮긴 것이다(그림에서 읽은 값은 "그림 판독"으로 표시).
결과는 analysis/recompute_results.json 에 저장한다.

실행: python3 analysis/recompute.py   (numpy, scipy; 수 초)
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def fisher(a, b, c, d):
    return float(stats.fisher_exact([[a, b], [c, d]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ---------------------------------------------------------------- 표 2 (V1-V5)
objects = ["Mustard", "FaceWash", "Toothpaste", "BBQSmall", "Chips", "BBQ", "MustardSmall", "Chocolate", "Ketchup"]
weight = [650, 280, 176, 290, 43, 580, 244, 210, 410]
# (pick-and-place 성공, squeeze 성공) 각 10회. squeeze가 필요 없는 칸은 None.
T2 = {
    "continuous": [(7, 0), (6, 4), (4, 2), (6, 0), (3, None), (8, 0), (3, 0), (0, 0), (7, 0)],
    "binary": [(10, 0), (10, 0), (8, 0), (9, 0), (10, None), (7, 0), (10, 0), (10, 0), (10, 0)],
    "forceband": [(10, 10), (10, 10), (9, 9), (8, 8), (7, None), (6, 6), (10, 10), (10, 9), (10, 10)],
}
N = 10
t2 = {}
for m, rows in T2.items():
    pick = sum(r[0] for r in rows)
    sq = [r[1] for r in rows if r[1] is not None]
    n_pick = N * len(rows)
    n_sq = N * len(sq)
    t2[m] = dict(
        pick_k=pick, pick_n=n_pick, pick_rate=round(pick / n_pick, 4), pick_wilson=wilson(pick, n_pick),
        squeeze_k=sum(sq), squeeze_n=n_sq, squeeze_rate=round(sum(sq) / n_sq, 4), squeeze_wilson=wilson(sum(sq), n_sq),
    )
res["V1_table2_totals"] = t2

# V2: "87%" 후보 정의
fb = T2["forceband"]
pick_total = sum(r[0] for r in fb)
sq_total = sum(r[1] for r in fb if r[1] is not None)
chips_pick = fb[4][0]
cands = {
    "pick_place_pooled_80/90": pick_total / 90,
    "squeeze_pooled_72/80": sq_total / 80,
    "squeeze_where_required_else_pick (72+7)/90": (sq_total + chips_pick) / 90,
    "mean_of_per_object_rates_same_definition": float(np.mean([(r[1] if r[1] is not None else r[0]) / 10 for r in fb])),
    "pick_and_squeeze_cells_pooled (80+72)/170": (pick_total + sq_total) / 170,
    "pick_squeeze_all_cells_both_counted_except_chips_once": (pick_total + sq_total) / 170,
}
res["V2_candidates_for_87pct"] = {k: round(v, 4) for k, v in cands.items()}
res["V2_integer_k_over_90_near_87"] = {k: round(k / 90, 4) for k in (77, 78, 79, 80)}
res["V2_integer_k_over_80_near_87"] = {k: round(k / 80, 4) for k in (69, 70, 71)}
res["V2_integer_k_over_170_near_87"] = {k: round(k / 170, 4) for k in (147, 148, 149)}

# V3: 방법 간 Fisher (합산, 90/80회)
def pooled_pair(a, b, kind):
    ka, na = t2[a][f"{kind}_k"], t2[a][f"{kind}_n"]
    kb, nb = t2[b][f"{kind}_k"], t2[b][f"{kind}_n"]
    return fisher(ka, na - ka, kb, nb - kb)

res["V3_fisher_pooled"] = {
    "pick_forceband_vs_binary": pooled_pair("forceband", "binary", "pick"),
    "pick_forceband_vs_continuous": pooled_pair("forceband", "continuous", "pick"),
    "squeeze_forceband_vs_binary": pooled_pair("forceband", "binary", "squeeze"),
    "squeeze_forceband_vs_continuous": pooled_pair("forceband", "continuous", "squeeze"),
}

# V4: 물체별 pick 성공 (ForceBand 대 이진 그리퍼) 와 squeeze (ForceBand 대 연속 그리퍼)
per_obj = []
for i, o in enumerate(objects):
    fbp, bip, cop = fb[i][0], T2["binary"][i][0], T2["continuous"][i][0]
    row = dict(obj=o, weight_g=weight[i], fb_pick=fbp, bin_pick=bip, cont_pick=cop,
               p_pick_fb_vs_bin=fisher(fbp, N - fbp, bip, N - bip))
    if fb[i][1] is not None:
        fbs, cos = fb[i][1], T2["continuous"][i][1]
        row.update(fb_sq=fbs, cont_sq=cos, p_sq_fb_vs_cont=fisher(fbs, N - fbs, cos, N - cos))
    per_obj.append(row)
sq_ps = [r["p_sq_fb_vs_cont"] for r in per_obj if "p_sq_fb_vs_cont" in r]
adj = holm(sq_ps)
j = 0
for r in per_obj:
    if "p_sq_fb_vs_cont" in r:
        r["p_sq_holm"] = adj[j]
        j += 1
res["V4_per_object"] = per_obj

# V5: 검정력. 이진 그리퍼 pick 84/90 대 ForceBand 80/90 를 가르려면 필요한 쌍당 시행 수 (정규 근사)
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    return ((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)) ** 2

res["V5_power"] = {
    "pick_binary_vs_forceband_n_per_arm_for_80pct_power": round(n_per_arm(84 / 90, 80 / 90)),
    "per_object_n10_wilson_width_at_p0.9": round(wilson(9, 10)[1] - wilson(9, 10)[0], 3),
    "per_object_10of10_wilson_lower": wilson(10, 10)[0],
}

# ---------------------------------------------------------------- 그림 5 (V6-V8)
fig5 = {
    "hand_level_force": {"MAE": (0.80, 1.58), "RMSE": (1.36, 2.80)},   # (Ours, FEEL) 그림 인쇄 값
    "finger_mae": {"Thumb": (1.45, 2.72), "Index": (1.06, 2.09), "Middle": (0.86, 1.57), "Ring": (0.34, 0.75), "Pinky": (0.26, 0.76)},
    "finger_prauc": {"Thumb": (0.93, 0.86), "Index": (0.96, 0.78), "Middle": (0.81, 0.63), "Ring": (0.76, 0.38), "Pinky": (0.59, 0.27)},
    "prevalence_random": {"Thumb": 0.45, "Index": 0.36, "Middle": 0.26, "Ring": 0.12, "Pinky": 0.07},
    "hand_bcd": {"ROC_AUC": (0.94, 0.85, 0.70, 0.61), "PR_AUC": (0.89, 0.86, 0.72, 0.61),
                 "F1": (0.85, 0.77, 0.66, 0.70), "BalAcc": (0.89, 0.76, 0.64, 0.55)},  # Ours, FEEL, VISOR ft, VISOR raw
}
rel = {k: round(1 - v[0] / v[1], 4) for k, v in fig5["hand_level_force"].items()}
rel.update({f"{k}_MAE": round(1 - v[0] / v[1], 4) for k, v in fig5["finger_mae"].items()})
res["V6_relative_error_reduction"] = rel
lo = 1 - 0.805 / 1.575
hi = 1 - 0.795 / 1.585
res["V6_hand_MAE_reduction_rounding_range"] = [round(lo, 4), round(hi, 4)]  # 인쇄 값 0.80, 1.58 의 반올림 범위
res["V6_mean_of_finger_MAE"] = {
    "ours": round(float(np.mean([v[0] for v in fig5["finger_mae"].values()])), 4),
    "feel": round(float(np.mean([v[1] for v in fig5["finger_mae"].values()])), 4),
    "hand_level_printed": (0.80, 1.58),
}
# 본문 PR AUC 문장
txt = {"ring": (0.763, 0.398), "pinky": (0.590, 0.314)}
prn = {"ring": (0.76, 0.38), "pinky": (0.59, 0.27)}
rnd = {"ring": 0.12, "pinky": 0.07}
res["V7_pr_auc_text_vs_figure"] = {
    f: dict(text_ratio=round(txt[f][0] / txt[f][1], 3), figure_label_ratio=round(prn[f][0] / prn[f][1], 3),
            text_vs_random=round(txt[f][0] / rnd[f], 2), figure_FEEL_label=prn[f][1], text_FEEL=txt[f][1])
    for f in txt
}
# 모든 막대에서 ours > FEEL ?
res["V8_all_fingers_ours_better"] = {
    "pr_auc": all(v[0] > v[1] for v in fig5["finger_prauc"].values()),
    "mae": all(v[0] < v[1] for v in fig5["finger_mae"].values()),
    "hand_bcd_all_metrics_ours_best": all(v[0] == max(v) for v in fig5["hand_bcd"].values()),
}
# 이진 접촉 기준 0.8 N 에서 PR AUC 기준선 = 유병률 (무작위 예측기의 기대 PR AUC)
res["V8_random_pr_auc_equals_prevalence_note"] = "무작위 예측기의 PR AUC 기대값은 양성 비율(유병률)이므로 점선 값 0.45–0.07 이 유병률과 같다"

# ---------------------------------------------------------------- 표 1 (V9-V10)
t1_left = {"1ch": (1.89, 3.69), "2ch": (1.76, 3.06), "4ch": (0.93, 2.00), "8ch_ours": (0.85, 1.92)}
t1_right = {"8ch_even": (0.94, 1.77), "8ch_ours": (0.77, 1.33)}
res["V9_table1"] = {
    "right_rel_reduction_MAE": round(1 - 0.77 / 0.94, 4),
    "right_rel_reduction_RMSE": round(1 - 1.33 / 1.77, 4),
    "left_1ch_to_8ch_MAE": round(1 - 0.85 / 1.89, 4),
    "left_4ch_to_8ch_MAE": round(1 - 0.85 / 0.93, 4),
    "left_4ch_to_8ch_RMSE": round(1 - 1.92 / 2.00, 4),
    "left_monotone_MAE": all(t1_left[a][0] > t1_left[b][0] for a, b in zip(list(t1_left)[:-1], list(t1_left)[1:])),
    "left_monotone_RMSE": all(t1_left[a][1] > t1_left[b][1] for a, b in zip(list(t1_left)[:-1], list(t1_left)[1:])),
    "RMSE_over_MAE_ratio": {k: round(v[1] / v[0], 2) for k, v in {**t1_left, **{"right_" + a: b for a, b in t1_right.items()}}.items()},
}
# 같은 모델(8ch ours)의 MAE: 문서 안의 서로 다른 값
res["V10_full_model_MAE_values_N"] = {
    "table1_left_8ch_ours": 0.85, "table1_right_8ch_ours_30min": 0.77, "fig5B_hand_level": 0.80,
    "fig9_full_model_val_MAE_epoch50": 0.92,
    "repo_reference_val_17.63_div_18": round(17.63 / 18, 3),
    "repo_reference_val_RMSE_39.85_div_18": round(39.85 / 18, 3),
    "table1_left_8ch_RMSE": 1.92, "fig5B_RMSE": 1.36, "table1_right_RMSE": 1.33,
}
res["V10_repo_per_finger_N"] = {
    f: round(v / 18, 3) for f, v in dict(Thumb=36.460079, Index=24.125414, Middle=15.055875, Ring=6.291829, Pinky=6.232687).items()
}
res["V10_paper_fig5D_ours_per_finger_N"] = {f: v[0] for f, v in fig5["finger_mae"].items()}

# ---------------------------------------------------------------- 그림 9 (V11)
f9 = {"full": 0.92, "no_imu": 1.02, "no_spec": 1.14}
res["V11_ablation"] = {
    "no_spec_vs_full_rel_increase": round(f9["no_spec"] / f9["full"] - 1, 4),
    "no_imu_vs_full_rel_increase": round(f9["no_imu"] / f9["full"] - 1, 4),
    "abs_gap_full_vs_no_imu": round(f9["no_imu"] - f9["full"], 2),
    "note": "그림 9는 epoch 50 의 검증 MAE 한 번의 학습 곡선이다. 마지막 10 epoch 에서 No-IMU 곡선이 약 ±0.05 N 진동하는 것이 그림에서 보이므로 0.10 N 차이는 시드 간 변동과 구분되지 않는다(그림 판독).",
}

# ---------------------------------------------------------------- 그림 4 / 데이터 (V12)
res["V12_dataset_fractions"] = {
    "fig4A_sum": 46 + 27 + 16 + 11,
    "fig4B_sum": 51 + 17 + 11 + 21,
    "text_assignment": {"2-finger": 51, "3-finger": 17, "5-finger": 11, "in_the_wild": 21},
    "figure_color_reading": {"2-finger": 17, "3-finger": 11, "5-finger": 21, "in_the_wild": 51},
    "atomic_total_text": 51 + 17 + 11,
    "atomic_total_figure": 17 + 11 + 21,
    "note": "그림 4B 의 범례 색과 조각 색으로 읽은 배정(그림 판독)이 본문 문장의 배정과 다르다. 합은 둘 다 100.",
}
hours = 10.0
res["V12_hours_by_text_fraction"] = {k: round(hours * v / 100, 2) for k, v in
                                       {"pinch/grasp": 46, "pick&place": 27, "open/close": 16, "pour": 11}.items()}

# ---------------------------------------------------------------- 저장소 수치 (V13)
# README: 센서 단위 ÷18 ≈ N. 영샷 값에서 이득 비를 역산 (동일 평균 가정, 근사)
def implied_std_ratio(r2, r):
    # r2 = 1 - (s^2 - 2 r s + 1)  (평균 동일, σ_gt = 1 로 정규화) -> s^2 - 2 r s + (r2) = 0
    disc = (2 * r) ** 2 - 4 * r2
    if disc < 0:
        return None
    return [round((2 * r + sg * math.sqrt(disc)) / 2, 3) for sg in (+1, -1)]

res["V13_repo_calibration"] = {
    "zero_shot": dict(MAE_units=46.0, MAE_N=round(46.0 / 18, 2), RMSE_N=round(59.3 / 18, 2), r=0.59, r2=-4.94,
                      implied_pred_over_gt_std=implied_std_ratio(-4.94, 0.59)),
    "calibrated": dict(MAE_units=9.5, MAE_N=round(9.5 / 18, 2), RMSE_N=round(16.3 / 18, 2), r=0.78, r2=0.58,
                       implied_pred_over_gt_std=implied_std_ratio(0.58, 0.78)),
    "val_released": dict(MAE_N=round(17.63 / 18, 2), RMSE_N=round(39.85 / 18, 2), r=0.698, r2=0.453),
    "r_squared_vs_r2_calibrated": round(0.78 ** 2, 3),
    "note": "평균 동일·σ 비만 다르다는 가정의 역산이며 지표는 세 손가락 평균이라 근사다.",
}
# 저장소 파라미터 수 대략 (README 28.5 M 주장 점검): ViT-S/14 reg4(약 22 M) + 변환기 + 합성곱
def tf_layer(d, ff=2048):
    return 4 * d * d + 4 * d + 2 * d * ff + ff + d + 4 * d
vit_s = 12 * (4 * 384 * 384 + 4 * 384 + 2 * 384 * 1536 + 1536 + 384 + 4 * 384) + 14 * 14 * 3 * 384 + 384 + 384 * (1 + 4)
conv = (384 + 18) * 512 * 5 + 512 + 512 * 256 * 5 + 256
tf = 4 * tf_layer(256) + 256 * 5 + 5 + (5 * 256 + 256 + 2 * 256)
res["V13_param_estimate"] = {"vit_s14_approx": vit_s, "conv": conv, "transformer_plus_head": tf,
                             "total_approx_M": round((vit_s + conv + tf + 8 * 3) / 1e6, 2), "readme_claim_M": 28.5}

# 검증 데이터 길이 (LFS 포인터 크기) 추정: 1 kHz, float32 (8+10+5)*4 + 8 byte = 100 byte/샘플, 압축 없음 가정
bps = (8 + 10 + 5) * 4 + 8
res["V14_repo_data_size_estimate"] = {
    "val_files": 7, "val_bytes_each": 12008224, "val_seconds_each_if_1kHz_uncompressed": round(12008224 / bps / 1000, 1),
    "val_total_min_if_1kHz_uncompressed": round(7 * 12008224 / bps / 60000, 1),
    "demos": 94, "demo_total_bytes": 35558423, "calib_36_bytes": 24202198,
    "calib_36_min_if_1kHz_uncompressed": round(24202198 / bps / 60000, 1),
    "paper_calibration_min": 15,
    "hf_hdf5_total_GB": 0.556, "readme_hours_claim": 8.6,
    "hf_hdf5_hours_if_1kHz_uncompressed": round(0.556e9 / bps / 1000 / 3600, 2),
    "hf_hdf5_hours_if_250Hz_uncompressed": round(0.556e9 / (bps * 250) / 3600, 2),
    "note": "저장 형식(압축, 저장 주파수)을 확인하지 못해 시간 환산은 가정이다.",
}

# ---------------------------------------------------------------- 정책 수치 (V15)
# Fig. 6 / 프로젝트 페이지 힘 값 (인쇄된 N)
res["V15_force_pairs_project_page_N"] = {
    "toothpaste_human_vs_robot": (10.2, 12.6), "chips_human_vs_robot": (5.2, 1.7), "bbq_human_vs_robot": (17.3, 18.4),
    "abs_gap": {"toothpaste": round(abs(12.6 - 10.2), 1), "chips": round(abs(1.7 - 5.2), 1), "bbq": round(abs(18.4 - 17.3), 1)},
    "rel_gap": {"toothpaste": round(abs(12.6 - 10.2) / 10.2, 2), "chips": round(abs(1.7 - 5.2) / 5.2, 2), "bbq": round(abs(18.4 - 17.3) / 17.3, 2)},
}
res["V15_force_range_note"] = "논문 그림 6: 정점 힘 3.2–19.3 N. 표 2 주석: 파악 폭은 초기 접촉부터 20 N 인가까지의 구간이라 최대 힘 20 N 이 상한으로 보인다"

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
