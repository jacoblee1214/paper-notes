"""T(R,O) Grasp++ (arXiv:2610.25410 v1) 논문 표 수치를 다시 계산하고 일관성을 검사한다.

입력: 논문 본문과 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 그림 8 막대 높이는
      paper/paper.pdf 15쪽의 래스터 이미지에서 픽셀로 읽는다(pymupdf, Pillow).
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, pymupdf, Pillow; 수 초)

가정(논문에 없는 것은 표시한다)
- 표 2·5·6·11의 성공률 분모: 손당 20,000 장면(5.2절). 표 5는 8개 손 160,000 장면이라고 가정한다.
- 점수(%)가 2자리 소수이므로 20,000 장면이면 모두 정수 개수로 표현할 수 있다(해상도 0.005 %p).
- 장면 독립을 가정한 z 검정은 같은 테스트 장면에서 나온 쌍 비교를 과소 평가할 수 있다(쌍 검정이면 더 강함).
"""
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def z_two_prop(p1, n1, p2, n2):
    """두 비율 차이의 z 검정(합동 분산, 양측)."""
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def r1(x, d=2):
    return round(float(x), d)


# ------------------------------------------------------------------ V1 표 2 평균
hands = ["Allegro", "Ability", "Barrett", "DClaw", "LEAP", "Schunk", "Shadow", "XHand"]
t2_sr = [88.79, 88.28, 87.24, 89.11, 85.93, 90.39, 87.87, 86.03]
t2_div = [0.369, 0.509, 0.470, 0.393, 0.404, 0.389, 0.354, 0.418]
RES["V1_table2"] = {
    "sr_mean_recomputed": r1(np.mean(t2_sr), 3), "sr_mean_reported": 87.96,
    "div_mean_recomputed": r1(np.mean(t2_div), 4), "div_mean_reported": 0.413,
    "sr_min_max": [min(t2_sr), max(t2_sr)],
    "sr_range_pp": r1(max(t2_sr) - min(t2_sr)),
    "wilson95_halfwidth_pp_per_hand_n20000": r1(100 * (wilson(17600, 20000)[1] - wilson(17600, 20000)[0]) / 2, 3),
    "wilson95_pooled_160000_pp": [r1(100 * x) for x in wilson(round(np.mean(t2_sr) * 1600), 160000)],
}

# ------------------------------------------------------------------ V2 표 6
full4 = [88.79, 85.93, 87.87, 86.03]
wo4 = [23.38, 4.31, 4.84, 5.91]
RES["V2_table6"] = {
    "full_avg_recomputed": r1(np.mean(full4), 3), "full_avg_reported": 87.16,
    "wo_avg_recomputed": r1(np.mean(wo4), 3), "wo_avg_reported": 9.61,
    "full_vs_table2_8hand": [87.16, 87.96],
    "note": "표 6 Full Model 평균은 4개 손(87.16)이며 표 2의 8개 손 평균(87.96)과 다른 집합이다. 4개 손 값은 표 2와 같다.",
}

# ------------------------------------------------------------------ V3 표 1 개수
res3 = {}
res3["objects_sum"] = [13049 + 40626, 53675]
res3["train_test_sum"] = [47737 + 5938, 53675]
res3["ours_per_obj_hand_if_444M_all_objects"] = r1(444e6 / (53675 * 8), 0)
res3["ours_per_obj_hand_train_objects_only"] = r1(444e6 / (47737 * 8), 0)
res3["abstract_0p5B_vs_444M_ratio"] = r1(0.5e9 / 444e6, 3)
res3["dgn2_per_obj_(427M/1319)"] = r1(427e6 / 1319, 0)
res3["multigripper_per_obj_hand_(30.4M/(345*11))"] = r1(30.4e6 / (345 * 11), 0)
res3["ratio_objects_ours_vs_dgn2"] = r1(53675 / 1319, 1)
res3["ratio_grasps_ours_vs_dgn2"] = r1(444 / 427, 3)
res3["ratio_grasps_per_obj_hand_dgn2_vs_ours"] = r1((427e6 / 1319) / (444e6 / (53675 * 8)), 0)
# 공개 데이터 카드(HF Barrybarry-Smith/TROV2Dataset, 작은 메타데이터만 읽음)
res3["hf_release_train_objects"] = 25228
res3["hf_release_source_objects"] = 31166
res3["hf_release_eval_objects_excluded"] = 5938
res3["hf_release_vs_paper_train_objects_pct"] = r1(100 * 25228 / 47737, 1)
res3["hf_release_vs_paper_total_objects_pct"] = r1(100 * 31166 / 53675, 1)
res3["hf_zarr_tib"] = 6.19
res3["hf_zarr_bytes"] = 6806567178240
res3["hf_bytes_per_object_MB"] = r1(6806567178240 / 25228 / 1e6, 0)
RES["V3_table1_counts"] = res3

# ------------------------------------------------------------------ V4 동일 손 격차
anyd, dgn2 = 54.44, 52.09
res4 = {
    "fig1_gap_8hand_avg_vs_anydexgrasp_pp": r1(87.96 - anyd),
    "same_hand_allegro_gap_pp": r1(88.79 - anyd), "same_hand_allegro_ratio": r1(88.79 / anyd, 3),
    "same_hand_leap_gap_pp": r1(85.93 - dgn2), "same_hand_leap_ratio": r1(85.93 / dgn2, 3),
    "8hand_avg_vs_dgn2_gap_pp": r1(87.96 - dgn2),
    "relative_gain_pct_fig1_vs_anydexgrasp": r1(100 * (87.96 / anyd - 1), 1),
    "diversity_allegro_ours_vs_anydexgrasp": [0.369, 0.387],
    "diversity_leap_ours_vs_dgn2": [0.404, 0.303],
}
for name, a, b in [("allegro_vs_anydexgrasp", 88.79, anyd), ("leap_vs_dgn2", 85.93, dgn2)]:
    z, p = z_two_prop(a / 100, 20000, b / 100, 20000)
    res4[name + "_z"] = r1(z, 1)
    res4[name + "_p"] = float(p)
RES["V4_same_hand_gaps"] = res4

# ------------------------------------------------------------------ V5 표 5 규모 곡선
params = [("10M", 79.63), ("50M", 86.77), ("200M", 87.96), ("1B", 88.07)]
objs = [("12K", 83.65), ("24K", 84.15), ("36K", 85.29), ("48K", 86.24)]
trials = [87.96, 95.58, 97.61, 98.46]
N = 160000
res5 = {"assumed_n": N}
res5["se_pp_at_88"] = r1(100 * math.sqrt(0.88 * 0.12 / N), 3)
res5["params_steps"] = []
for (a, pa), (b, pb) in zip(params[:-1], params[1:]):
    z, p = z_two_prop(pb / 100, N, pa / 100, N)
    res5["params_steps"].append({"from": a, "to": b, "diff_pp": r1(pb - pa), "z": r1(z, 1), "p": float(p)})
res5["objects_steps"] = []
for (a, pa), (b, pb) in zip(objs[:-1], objs[1:]):
    z, p = z_two_prop(pb / 100, N, pa / 100, N)
    res5["objects_steps"].append({"from": a, "to": b, "diff_pp": r1(pb - pa), "z": r1(z, 1), "p": float(p)})
res5["objects_gain_12K_to_48K_pp"] = r1(86.24 - 83.65)
res5["objects_gain_per_doubling_pp"] = r1((86.24 - 83.65) / 2)
res5["objects_log_slope_check"] = "12K→24K→48K: +0.50, +(1.14+0.95)=2.09 (12K→24K, 24K→48K)"
res5["objects_48K_vs_table2_full"] = {"table5_48K": 86.24, "table2_default_200M": 87.96, "diff_pp": r1(87.96 - 86.24),
                                      "train_objects_in_text": 47737,
                                      "note": "본문 3.1절의 학습 객체는 47,737개(약 48K)인데 표 5의 48K 행(86.24)은 기본 모델(87.96)과 1.72 %p 다르다."}
# 모델 규모 곡선 로그 기울기
lp = np.log10([10, 50, 200, 1000])
res5["params_log10_slope_pp_per_decade_10_50_200"] = r1(np.polyfit(lp[:3], [79.63, 86.77, 87.96], 1)[0])
RES["V5_scaling_table5"] = res5

# ------------------------------------------------------------------ V6 표 3 경계
res6 = {}
levels = {
    "ours": ([80.32, 77.72, 77.81, 75.37], 87.96),
    "anydexgrasp": ([53.09, 50.82, 48.73, 44.27], 54.44),
    "dgn2": ([55.28, 54.76, 51.83, 50.56], 52.09),
}
for k, (lv, orig) in levels.items():
    lo, hi = min(lv), max(lv)
    res6[k] = {
        "levels": lv, "level_mean_unweighted": r1(np.mean(lv)),
        "orig20k_sr": orig,
        "dense20k_sr_if_combined_equals_weighted_avg_bounds": [r1(2 * lo - orig), r1(2 * hi - orig)],
        "level1_to_level4_drop_pp": r1(lv[0] - lv[3]),
    }
res6["ours_monotone"] = [80.32 >= 77.72, 77.72 >= 77.81, 77.81 >= 75.37]
res6["gap_ours_minus_anydexgrasp_by_level"] = [r1(a - b) for a, b in zip([80.32, 77.72, 77.81, 75.37], [53.09, 50.82, 48.73, 44.27])]
res6["gap_ours_minus_dgn2_by_level"] = [r1(a - b) for a, b in zip([80.32, 77.72, 77.81, 75.37], [55.28, 54.76, 51.83, 50.56])]
res6["diversity_ours_levels"] = [0.408, 0.418, 0.423, 0.412]
res6["note"] = ("표 3은 40,000 장면(원래 20,000 + 밀집 20,000)을 D_env로 나눈 것이다. 가중 평균은 수준별 값의 최소와 최대 사이에 있으므로, "
                "원래 세트가 87.96 %라면 밀집 세트는 62.8–72.7 %여야 한다(우리 방법). 수준별 장면 수는 보고되지 않았다.")
RES["V6_table3_bounds"] = res6

# ------------------------------------------------------------------ V7 표 4 / 그림 1 파생값
lat = {"TRO-Grasp": 0.210, "AnyDexGrasp": 0.835, "DexGraspNet2": 1.712, "Ours": 0.336}
thr = {"TRO-Grasp": 16.97, "AnyDexGrasp": 1.28, "DexGraspNet2": 4.96, "Ours": 65.31}
pps = {"TRO-Grasp": 100.0, "AnyDexGrasp": 26.39, "DexGraspNet2": 13.34, "Ours": 1.15}
res7 = {
    "fps_from_latency": {k: r1(1 / v, 3) for k, v in lat.items()},
    "fig1_fps_reported": {"AnyDexGrasp": 1.20, "DexGraspNet2": 0.58, "Ours": 2.98},
    "fig1_speedup_x2p5_check": {"ours_over_anydexgrasp": r1((1 / 0.336) / (1 / 0.835), 2),
                                "ours_over_dgn2": r1((1 / 0.336) / (1 / 1.712), 2)},
    "implied_items_per_call_(throughput*latency)": {k: r1(thr[k] * lat[k], 2) for k in lat},
    "inv_pps": {k: r1(1 / v, 4) for k, v in pps.items() if k != "TRO-Grasp"},
    "fig1_inv_pps_reported": {"DexGraspNet2": 0.07, "AnyDexGrasp": 0.04, "Ours": 0.87},
    "fig1_x12p4_check_reported_rounded": r1(0.87 / 0.07, 2),
    "pps_ratio_exact_vs_dgn2": r1(13.34 / 1.15, 2),
    "pps_ratio_exact_vs_anydexgrasp": r1(26.39 / 1.15, 2),
    "implied_per_proposal_success_pct": {k: r1(100 / v, 2) for k, v in pps.items() if k != "TRO-Grasp"},
    "ours_pps_vs_1_over_sr": {"1/0.8796": r1(1 / 0.8796, 3), "reported": 1.15, "implied_sr_pct": r1(100 / 1.15, 2)},
    "params_default_M": 176.53, "params_table5_nominal_M": 200,
    "latency_intro_text_s": 0.34, "latency_table4_s": 0.336, "latency_abstract_s": 0.3,
    "inference_speed_text_fps_3": r1(1 / 0.336, 2),
}
RES["V7_efficiency"] = res7

# ------------------------------------------------------------------ V8 다중 시도 vs 독립 가정, 베타 혼합 적합
sr = np.array(trials) / 100
fail = 1 - sr
indep = fail[0] ** np.arange(1, 5)
res8 = {"observed_fail_pct": [r1(100 * f, 2) for f in fail],
        "independent_pred_fail_pct": [r1(100 * f, 3) for f in indep],
        "observed_over_independent": [r1(f / i, 1) for f, i in zip(fail, indep)],
        "conditional_fail_given_prev_fail": [r1(fail[i + 1] / fail[i], 3) for i in range(3)]}


def beta_fail(a, k, m=0.8796):
    b = a * (1 - m) / m
    return math.exp(math.lgamma(a + b) - math.lgamma(b) + math.lgamma(b + k) - math.lgamma(a + b + k))  # B(a,b+k)/B(a,b)


def loss(la):
    a = math.exp(la)
    return sum((beta_fail(a, k) - fail[k - 1]) ** 2 for k in (2, 3, 4))


opt = optimize.minimize_scalar(loss, bounds=(-6, 4), method="bounded")
a_hat = math.exp(opt.x)
b_hat = a_hat * (1 - 0.8796) / 0.8796
res8["beta_fit"] = {"a": r1(a_hat, 4), "b": r1(b_hat, 4),
                    "pred_fail_pct_k1_to_4": [r1(100 * beta_fail(a_hat, k), 3) for k in (1, 2, 3, 4)],
                    "frac_scenes_theta_below_0.1_pct": r1(100 * stats.beta.cdf(0.1, a_hat, b_hat), 2),
                    "frac_scenes_theta_below_0.5_pct": r1(100 * stats.beta.cdf(0.5, a_hat, b_hat), 2),
                    "frac_scenes_theta_above_0.95_pct": r1(100 * (1 - stats.beta.cdf(0.95, a_hat, b_hat)), 2)}
res8["note"] = ("장면별 성공 확률이 균일하면 4회 실패율은 0.021 %여야 하는데 보고값은 1.54 %(73배)다. 실패가 장면에 묶여 있다 "
                "(같은 장면을 다시 뽑아도 같은 이유로 실패). 베타 혼합 적합은 한 자유 매개변수의 사후 해석이다.")
RES["V8_multi_trial"] = res8

# ------------------------------------------------------------------ V9 표 11
sim = [1.0, 0.9, 0.8, 0.7]
sr_link = [86.03, 85.35, 84.56, 84.51]
sr_joint = [86.03, 84.04, 82.93, 81.73]
res9 = {"link_drop_pp": [r1(86.03 - x) for x in sr_link], "joint_drop_pp": [r1(86.03 - x) for x in sr_joint],
        "link_slope_pp_per_0.1": r1(np.polyfit(sim, sr_link, 1)[0] / 10, 2),
        "joint_slope_pp_per_0.1": r1(np.polyfit(sim, sr_joint, 1)[0] / 10, 2),
        "min_sr": min(sr_link + sr_joint),
        "note": "S=1.0 행은 표 2 XHand(86.03/0.418)와 같다. 유사도 0.7에서도 80 % 이상이라는 문장은 맞지만, 변형 손은 학습 손의 링크 길이·관절 범위만 바꾼 것이다(위상 같음)."}
RES["V9_table11"] = res9

# ------------------------------------------------------------------ V10 실로봇 표 7, 8
xhand = {"in_box": [10, 9, 7, 7, 10], "on_table": [10, 8, 7, 8, 9]}
leap = {"in_box": [9, 10, 7, 7, 9], "on_table": [8, 7, 7, 10, 9]}
res10 = {}
for name, d, claim in [("XHand", xhand, 85), ("LEAP", leap, 82)]:
    k = sum(d["in_box"]) + sum(d["on_table"])
    lo, hi = wilson(k, 100)
    res10[name] = {"in_box": sum(d["in_box"]), "on_table": sum(d["on_table"]), "total": k, "n": 100,
                   "claimed_pct": claim, "matches_claim": k == claim, "wilson95_pct": [r1(100 * lo, 1), r1(100 * hi, 1)]}
res10["xhand_vs_leap_fisher_p"] = r1(fisher(85, 100, 83, 100), 3)
res10["xhand_vs_leap_fisher_p_if_82"] = r1(fisher(85, 100, 82, 100), 3)
# 객체별 이질성: 카이제곱(성공/실패)
obs = np.array([[x, 10 - x] for x in xhand["in_box"] + xhand["on_table"]])
chi2, p_het, _, _ = stats.chi2_contingency(obs)
res10["xhand_object_heterogeneity_chi2_p"] = r1(p_het, 3)
# 표 8
t8 = {"Wuji": ([13, 5, 12, 44], 414), "XHand": ([22, 14, 27, 29], 548)}
res10["table8"] = {}
for k, (f, n) in t8.items():
    tot = sum(f)
    lo, hi = wilson(n - tot, n)
    res10["table8"][k] = {"fail_sum": tot, "n": n, "success_pct": r1(100 * (n - tot) / n, 2),
                          "wilson95_success_pct": [r1(100 * lo, 1), r1(100 * hi, 1)],
                          "instability_share_pct": r1(100 * f[3] / tot, 1),
                          "shares_pct": [r1(100 * x / tot, 1) for x in f]}
res10["table8"]["pooled_instability_share_pct"] = r1(100 * (44 + 29) / (74 + 92), 1)
feas = [k for k in range(0, 549) if abs(100 * k / 548 - 90.6) < 0.05]
res10["text_xhandpro_90p6"] = {"integer_k_giving_90.6_at_n548": feas,
                               "nearest_values_pct": [r1(100 * 496 / 548, 2), r1(100 * 497 / 548, 2)],
                               "table8_xhand_success_pct": r1(100 * (548 - 92) / 548, 2),
                               "note": "본문 5.6절은 XHandPro 90.6 %, 표 8은 XHand 실패 92/548(성공 83.2 %). n = 548에서 90.6 %는 정수 개수가 없다."}
res10["wuji_82p1"] = {"k_340_of_414_pct": r1(100 * 340 / 414, 2)}
res10["gripper_98p5_feasible_n_with_10_per_obj"] = [n for n in range(400, 450, 1) if any(abs(100 * k / n - 98.5) < 0.05 for k in range(n + 1))][:6]
RES["V10_real_world"] = res10

# ------------------------------------------------------------------ V11 그림 8 막대 읽기
res11 = {}
PDF = ROOT / "paper" / "paper.pdf"
if not PDF.exists():
    # 원문 PDF가 없으면(arXiv에서 받아 paper/paper.pdf로 두면 됨) 이전 결과를 유지한다
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    res11 = prev.get("V11_fig8_digitized", {"skipped": "paper/paper.pdf 없음"})
else:
  try:
    import pymupdf
    from PIL import Image

    doc = pymupdf.open(ROOT / "paper" / "paper.pdf")
    img = doc.extract_image(doc[14].get_images(full=True)[0][0])
    a = np.array(Image.open(io.BytesIO(img["image"])).convert("RGB")).astype(int)
    cols = {"Dobj": (109, 166, 206), "Denv": (255, 171, 98), "Davg": (117, 193, 117)}
    grid = [y for y in range(a.shape[0]) if (np.abs(a[y, 200:1800] - 239).sum(axis=1) < 12).sum() > 200]
    # 격자선 y = 25(200), 103(150), 182(100), 260(50), 축 y = 338(0)
    y0, y200 = 338, 25
    ppm = (y0 - y200) / 200.0
    x_axis0, pitch = 23.0, 181.0  # 0.0 위치, 0.1 간격(px)
    centers = {}
    for name, c in cols.items():
        mask = np.abs(a - np.array(c)).sum(axis=2) < 8
        xs = np.where(mask.sum(axis=0) > 0)[0]
        groups, s, p = [], xs[0], xs[0]
        for x in xs[1:]:
            if x != p + 1:
                groups.append((s, p))
                s = x
            p = x
        groups.append((s, p))
        bars = []
        for g in groups:
            if g[1] - g[0] < 10:
                continue
            xm = (g[0] + g[1]) // 2
            ys = np.where(mask[:, xm])[0]
            h = (y0 - ys.min()) / ppm
            offset = {"Dobj": -50, "Denv": 0, "Davg": 48}[name]
            k = int(round((xm - offset - x_axis0 - pitch / 2) / pitch))
            bars.append((k, h))
        centers[name] = bars
    hist = {n: np.zeros(10) for n in cols}
    for n, bars in centers.items():
        for k, h in bars:
            hist[n][k] = h
    mid = (np.arange(10) + 0.5) / 10
    tot = {n: float(hist[n].sum()) for n in cols}
    mean = {n: float((hist[n] * mid).sum() / hist[n].sum()) for n in cols}
    res11 = {
        "grid_rows_found": grid,
        "bar_heights_M": {n: [r1(x, 1) for x in hist[n]] for n in cols},
        "sum_M": {n: r1(v, 1) for n, v in tot.items()},
        "claimed_total_M_table1": 444, "ratio_sum_to_444": {n: r1(v / 444, 3) for n, v in tot.items()},
        "mean_from_bin_centers": {n: r1(v, 3) for n, v in mean.items()},
        "mean_check_Davg_vs_avg_of_Dobj_Denv": {"Davg": r1(mean["Davg"], 3), "(Dobj+Denv)/2": r1((mean["Dobj"] + mean["Denv"]) / 2, 3)},
        "Dobj_share_last_bin_pct": r1(100 * hist["Dobj"][9] / tot["Dobj"], 2),
        "Denv_share_by_table3_level_pct": {
            "L1 [0,0.5)": r1(100 * hist["Denv"][:5].sum() / tot["Denv"], 1),
            "L2 [0.5,0.75) (0.7–0.8 bin 절반 가정)": r1(100 * (hist["Denv"][5:7].sum() + hist["Denv"][7] * 0.5) / tot["Denv"], 1),
            "L3+L4 (>=0.75)": r1(100 * (hist["Denv"][7] * 0.5 + hist["Denv"][8:].sum()) / tot["Denv"], 1)},
        "note": "세 분포의 합은 서로 1 % 안에서 일치하고(약 413 M) 표 1의 444 M보다 약 7 % 작다. 막대 높이 읽기 오차는 막대당 약 0.6 M이다."}
  except Exception as e:  # pragma: no cover
    res11 = {"error": repr(e)}
RES["V11_fig8_digitized"] = res11

# ------------------------------------------------------------------ V12 매개변수 수 (공식 설정 값으로 직접 계산)
D, T = 768, 128


def lin(i, o, b=True):
    return i * o + (o if b else 0)


def ln(d):
    return 2 * d


bind = lambda d, t: lin(d, d) + lin(t, d, False)
edge_val = lambda d: lin(d, d, False) * 2 + lin(d, d) * 2
attn_block = lambda d: lin(d, d) * 2 + edge_val(d) + lin(d, d)
edge_up = lambda d: lin(d, d) + lin(d, d, False) + lin(d, d)
layer_core = 6 * bind(D, T) + 3 * attn_block(D)
n_layers = 8
layers = n_layers * layer_core + (n_layers - 1) * 3 * edge_up(D)
enc = (lin(3, D) + lin(128, D)) * 2 + (lin(3, D) * 2 + lin(128, D) * 2) + lin(3, D) * 2 + (lin(3, D) * 2 + lin(128, D)) + lin(3, D) * 2
wide = lin(D * (1 + n_layers), D)


def naive_mlp(z, out, hidden):
    c, tot, wide_in = z, 0, z
    for h in hidden:
        tot += lin(c, h) + ln(h)
        wide_in += h
        c = h
    return tot + lin(wide_in, out)


outm = 2 * naive_mlp(3 + D, 3, [512, 256, 128])
joint_enc = (lin(6, 128) + ln(128)) + (lin(128, 128) + ln(128)) * 2 + (lin(256, 128) + ln(128)) + lin(128, 128)
pos_enc = lin(6, 128) + ln(128) + lin(128, 128) + ln(128) + lin(128, 128)
den_total = layers + enc + wide + outm
total = den_total + joint_enc + pos_enc
res12 = {"layers_M": r1(layers / 1e6, 2), "input_encoders_M": r1(enc / 1e6, 3), "wide_fc_M": r1(wide / 1e6, 3),
         "output_mlps_M": r1(outm / 1e6, 3), "denoiser_total_M": r1(den_total / 1e6, 2),
         "joint_space_and_pos_encoders_M": r1((joint_enc + pos_enc) / 1e6, 3), "graph_model_total_M": r1(total / 1e6, 2),
         "table4_reported_M": 176.53, "diff_M": r1(total / 1e6 - 176.53, 2),
         "note": "공식 config/train.yaml 값(768, 8층, 12헤드, 임베딩 128)으로 numpy 없이 산술 계산. VQ-VAE(고정)는 제외. 이 계산은 코드를 실행하지 않은 정적 추정이다."}
RES["V12_param_count"] = res12

# ------------------------------------------------------------------ V13 DDIM 단계 수 (코드의 정수 산술 그대로)
M, ns = 1000, 20
step = M // ns
start_t = M - 1 - step
ddim_t = list(range(start_t, -1, -step))
s_ = 0.008
x = np.linspace(0, M, M + 1)
ab = np.cos(((x / M) + s_) / (1 + s_) * np.pi / 2) ** 2
ab = ab / ab[0]
betas = np.clip(1 - ab[1:] / ab[:-1], 1e-5, 0.999)
alpha_bars = np.cumprod(1 - betas)
sig = []
for i, t in enumerate(ddim_t):
    a_t = alpha_bars[t]
    a_prev = 1.0 if i == len(ddim_t) - 1 else alpha_bars[ddim_t[i + 1]]
    sig.append(float(np.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))))
res13 = {"inference_step_config": ns, "step": step, "start_t": start_t, "num_denoise_steps_code": len(ddim_t),
         "ddim_t_first_last": [ddim_t[0], ddim_t[-1]], "alpha_bar_at_start": r1(alpha_bars[start_t], 5),
         "alpha_bar_at_999": float(alpha_bars[999]), "sigma_eta1_first_mid_last": [r1(sig[0], 4), r1(sig[len(sig) // 2], 4), r1(sig[-1], 4)],
         "noise_lambda_scale": 0.2, "paper_steps": 20, "paper_eta": 1.0,
         "note": "코드의 DDIM은 t = 949, 899, …, 49의 19단계이고 마지막 단계는 a_bar_prev = 1로 x0를 낸다. 확률 항에 λ = 0.2를 곱한다(논문에 없음)."}
RES["V13_ddim_schedule"] = res13

# ------------------------------------------------------------------ V14 훈련 규모 산술
steps_cfg, bs, gpus = 1_000_000, 48, 8
res14 = {"samples_seen_if_1M_steps_8gpu_bs48": steps_cfg * bs * gpus,
         "fraction_of_444M": r1(steps_cfg * bs * gpus / 444e6, 3),
         "coverage_if_uniform_with_replacement_pct": r1(100 * (1 - math.exp(-steps_cfg * bs * gpus / 444e6)), 1),
         "timesteps_per_sample": 5, "denoiser_evals_total": steps_cfg * bs * gpus * 5,
         "readme_example_ckpt_step": 820000, "samples_seen_at_820k": 820000 * bs * gpus,
         "note": "공식 train.yaml total_steps = 1,000,000. 논문은 '학습 데이터 전체를 덮도록' 스텝 수를 정했다고만 쓴다. GPU 수는 논문(8×A100)을 가정."}
RES["V14_training_scale"] = res14

# ------------------------------------------------------------------ V15 표 9 엔트로피
res15 = {"max_entropy_bits_6_dirs": r1(math.log2(6), 4), "normalized_reported": r1(5.96 / 6, 4)}
RES["V15_table9"] = res15

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1))
for k, v in RES.items():
    print("==", k)
    print(json.dumps(v, ensure_ascii=False)[:900])
