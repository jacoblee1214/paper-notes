"""SPIDER (arXiv:2511.09484 v3) 표 수치 재계산과 일관성 검사.

논문 본문·부록의 표 값을 그대로 옮겨 적고(PDF 텍스트 추출 기준), 평균·합계·가중 평균을
다시 계산해 보고된 값과 비교합니다. 외부 데이터 다운로드와 공식 코드 실행은 없습니다.
실행: python3 analysis/recompute.py  (수 초, numpy·scipy만 사용)
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}


def welch(m1, s1, n1, m2, s2, n2):
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return float(t), float(p)


# ---------------------------------------------------------------
# 1. Table 19 (ARCTIC, DexMachina 비교): 작업별 평균, 승패, 전체 평균
# ---------------------------------------------------------------
tasks = ["box", "ketchup", "ketchup-s01-u02", "laptop", "mixer", "notebook", "waffleiron"]
# (SPIDER mean, std, DexMachina mean, std)
t19 = {
    "dpos": [(0.0172, 0.0069, 0.0906, 0.0373), (0.8917, 0.1311, 1.1865, 0.3021),
             (0.7573, 0.3397, 1.1569, 0.2231), (0.0427, 0.0171, 0.0573, 0.0840),
             (0.0176, 0.0098, 0.1211, 0.1563), (0.9968, 0.0245, 0.4969, 0.0682),
             (0.2224, 0.1585, 0.5113, 0.0682)],
    "drot": [(0.0988, 0.0198, 0.3709, 0.1155), (1.7862, 0.1723, 2.0616, 0.1768),
             (1.5418, 0.3500, 2.0379, 0.2021), (0.2155, 0.0726, 0.1980, 0.2344),
             (0.1203, 0.0360, 0.7360, 0.2814), (1.9491, 0.2248, 1.1062, 0.2850),
             (1.0033, 0.2985, 1.1791, 0.0903)],
    "darti": [(0.0929, 0.0801, 0.2897, 0.0522), (0.7836, 0.0850, 0.8057, 0.0806),
              (1.1717, 0.0227, 1.1863, 0.0851), (1.0548, 0.5011, 0.4094, 0.3296),
              (0.1440, 0.0118, 0.3654, 0.1230), (0.7024, 0.0007, 0.7468, 0.1131),
              (0.5528, 0.3142, 0.5480, 0.0753)],
}
t19_overall_reported = {"dpos": (0.4208, 0.5172), "drot": (0.9593, 1.0985), "darti": (0.6432, 0.6216)}
res19 = {}
for k, rows in t19.items():
    s = np.array([r[0] for r in rows])
    d = np.array([r[2] for r in rows])
    wins = int(np.sum(s < d))
    res19[k] = {
        "spider_mean_of_task_means": round(float(s.mean()), 4),
        "dm_mean_of_task_means": round(float(d.mean()), 4),
        "reported_overall": t19_overall_reported[k],
        "spider_wins_of_7": wins,
        "tasks_spider_wins": [tasks[i] for i in range(7) if s[i] < d[i]],
    }
    # 작업별 Welch (5 seeds 가정; DexMachina는 결정론적이라 표준편차가 없다고 B.3에 적혀 있음)
    pv = []
    for (ms, ss, md, sd) in rows:
        t_, p_ = welch(ms, ss, 5, md, sd, 5)
        pv.append(round(p_, 4))
    res19[k]["welch_p_per_task_n5_assumed"] = dict(zip(tasks, pv))
out["table19"] = res19

# ---------------------------------------------------------------
# 2. Table 17 (Oakink, ManipTrans 비교): 16개 궤적 평균으로 전체 평균 재계산, 대응 t검정
# ---------------------------------------------------------------
sp_et = [34.81, 40.74, 44.27, 25.96, 31.09, 26.88, 5.99, 20.56, 45.45, 22.06, 14.00, 11.36, 12.64, 21.04, 6.25, 6.77]
mt_et = [34.50, 40.17, 44.31, 13.40, 34.43, 25.64, 21.33, 11.66, 45.41, 17.04, 14.03, 16.97, 18.61, 23.99, 3.93, 3.13]
sp_er = [101.76, 47.17, 80.72, 27.95, 29.26, 40.46, 56.57, 52.75, 51.92, 59.61, 85.78, 23.57, 33.06, 49.52, 85.93, 44.97]
mt_er = [112.67, 35.20, 80.54, 137.87, 57.47, 73.22, 57.51, 58.05, 51.81, 70.61, 85.77, 15.78, 99.13, 122.98, 50.81, 38.57]
sp_ej = [4.73, 5.03, 4.80, 5.09, 4.86, 4.79, 4.57, 4.45, 4.89, 4.94, 4.60, 5.82, 4.73, 4.64, 4.55, 4.84]
mt_ej = [25.98, 101.47, 293.28, 10.20, 11.12, 8.05, 9.93, 7.19, 54.05, 21.54, 6.40, 19.06, 19.24, 1618.07, 11.91, 6.61]
res17 = {}
for name, arr_s, arr_m, rep_s, rep_m in [
    ("et_cm", sp_et, mt_et, 23.12, 23.03),
    ("er_deg", sp_er, mt_er, 54.44, 71.75),
    ("ej", sp_ej, mt_ej, 4.84, 139.01),
]:
    a, b = np.array(arr_s), np.array(arr_m)
    t_, p_ = stats.ttest_rel(a, b)
    w_p = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
    res17[name] = {
        "spider_mean_16": round(float(a.mean()), 3),
        "maniptrans_mean_16": round(float(b.mean()), 3),
        "reported_spider_overall": rep_s,
        "reported_maniptrans_overall": rep_m,
        "paired_t_p": round(float(p_), 4),
        "wilcoxon_p": round(float(w_p), 4),
    }
out["table17"] = res17

# ---------------------------------------------------------------
# 3. Table 9/10 (per-hand AUC) 평균 vs Table 1 보고값
# ---------------------------------------------------------------
t9_auc = {  # Oakink, 7 tasks: kinematic, SPIDER, SPIDER+C
    "Allegro": (21.5, 53.4, 56.6), "Ability": (8.2, 43.7, 57.9), "Inspire": (7.7, 46.4, 63.4),
    "Schunk": (14.6, 46.7, 52.4), "XHand": (5.7, 53.4, 54.5),
}
t10_auc = {  # GigaHands, 3 trajectories
    "Allegro": (41.4, 58.7, 83.1), "Ability": (17.4, 55.5, 30.8), "Inspire": (12.5, 53.1, 73.4),
    "Schunk": (18.1, 60.8, 30.9), "XHand": (31.1, 56.6, 64.0),
}
k9 = [v[0] for v in t9_auc.values()]
s9 = [v[1] for v in t9_auc.values()]
c9 = [v[2] for v in t9_auc.values()]
k10 = [v[0] for v in t10_auc.values()]
s10 = [v[1] for v in t10_auc.values()]
c10 = [v[2] for v in t10_auc.values()]
out["table1_vs_table9_10"] = {
    "oakink_kin_mean_of_hands": round(float(np.mean(k9)), 2), "oakink_kin_reported_T1": 11.0,
    "oakink_spider_mean_of_hands": round(float(np.mean(s9)), 2), "oakink_spider_reported_T1": 48.8,
    "oakink_spiderC_mean_of_hands": round(float(np.mean(c9)), 2), "oakink_spiderC_reported_T1": 56.9,
    "gigahands_kin_mean_of_hands": round(float(np.mean(k10)), 2), "gigahands_kin_reported_T1": 24.1,
    "gigahands_spider_mean_of_hands": round(float(np.mean(s10)), 2), "gigahands_spider_reported_T1": 56.9,
    "gigahands_spiderC_mean_of_hands": round(float(np.mean(c10)), 2), "gigahands_spiderC_reported_T1": 57.7,
}

# Table 11 Oakink Avg 행(7개 작업)의 kinematic et와 Table 9 Allegro 행 비교
oak_kin_et = [3.8, 6.1, 23.3, 15.4, 3.3, 6.4, 7.5]
oak_spider_et = [1.9, 4.5, 23.6, 13.5, 2.6, 3.8, 3.7]
oak_spiderc_et = [1.8, 7.2, 5.7, 7.3, 3.1, 6.0, 0.9]
out["table11_oakink_avg_check"] = {
    "kin_et_mean_7": round(float(np.mean(oak_kin_et)), 2),
    "kin_et_pop_std": round(float(np.std(oak_kin_et)), 2),
    "reported_T11_avg_kin_et": "9.4 ± 6.8",
    "spider_et_mean_7": round(float(np.mean(oak_spider_et)), 2),
    "spiderC_et_mean_7": round(float(np.mean(oak_spiderc_et)), 2),
    "note": "Table 11의 Avg 표준편차는 작업 간 모집단 표준편차로 보임(9.4±6.8 재현)",
}

# ---------------------------------------------------------------
# 4. Table 7 (대규모 생성 보존 세트): 합계와 궤적 가중 평균
# ---------------------------------------------------------------
t7 = {  # trajectories, objects, frames, ADD-AUC
    "DexYCB": (870, 20, 151650, 73.27),
    "OakInk-Image": (241, 58, 36850, 76.52),
    "HRDexDB": (871, 80, 152425, 79.13),
    "HOT3D": (4538, 33, 950650, 79.19),
}
n_tot = sum(v[0] for v in t7.values())
obj_tot = sum(v[1] for v in t7.values())
fr_tot = sum(v[2] for v in t7.values())
auc_w = sum(v[0] * v[3] for v in t7.values()) / n_tot
out["table7"] = {
    "trajectories_sum": n_tot, "reported_total": 6520,
    "objects_sum": obj_tot, "reported_total_objects": 191,
    "frames_sum": fr_tot, "reported_total_frames": 1291575,
    "hours_sum": round(fr_tot / 100 / 3600, 3), "reported_hours": 3.59,
    "trajectory_weighted_add_auc": round(float(auc_w), 2), "reported_add_auc": 78.30,
}

# ---------------------------------------------------------------
# 5. Table 3 (demos 가중 평균) 와 본문 61.48%
# ---------------------------------------------------------------
t3 = {  # demos, Allegro, XHand, Inspire, Sharpa
    "DexYCB": (499, 55.70, 53.21, 54.47, 45.53),
    "OakInk-Image": (169, 59.17, 58.07, 59.14, 56.94),
    "HRDexDB": (441, 64.21, 64.44, 63.76, 62.14),
    "HOT3D": (2718, 71.16, 70.09, 69.41, 40.74),
}
demos = sum(v[0] for v in t3.values())
agg = {}
for j, hand in enumerate(["Allegro", "XHand", "Inspire", "Sharpa"]):
    agg[hand] = round(sum(v[0] * v[j + 1] for v in t3.values()) / demos, 2)
reported_agg = {"Allegro": 67.86, "XHand": 66.76, "Inspire": 66.41, "Sharpa": 44.46}
out["table3"] = {
    "demos_sum": demos, "reported_demos": 3827,
    "demo_weighted_by_hand": agg, "reported_by_hand": reported_agg,
    "mean_of_4_hand_aggregates": round(float(np.mean(list(reported_agg.values()))), 2),
    "text_completed_rollouts_ADD_AUC": 61.48,
}

# ---------------------------------------------------------------
# 6. 과제 수·실행 집계 (A.4)
# ---------------------------------------------------------------
planned = 3827 * 4
completed = 15041
ifail, pfail, missing = 161, 100, 6
out["funnel"] = {
    "planned_3827x4": planned, "reported_planned": 15308,
    "records_sum": completed + ifail + pfail, "reported_records": 15302,
    "eligible_after_lift_filter": completed - 437, "reported_eligible": 14604,
    "retained": 6520,
    "retained_fraction_of_completed": round(6520 / completed, 4),
}

# ---------------------------------------------------------------
# 7. 정책 학습 (4.2.1): 7 seed, 0.99±0.01 vs 0.07±0.11
# ---------------------------------------------------------------
tp, pp = welch(0.99, 0.01, 7, 0.07, 0.11, 7)
out["policy_4_2_1"] = {"welch_t": round(tp, 2), "p": float(f"{pp:.2e}"), "n_seeds_assumed": 7}

# ---------------------------------------------------------------
# 8. Table 1 ManipTrans vs SPIDER (ADD-AUC, 19.4±27.5 vs 22.3±28.1), n 가정 두 가지
# ---------------------------------------------------------------
res_t1 = {}
for n in (16, 35):
    t_, p_ = welch(22.3, 28.1, n, 19.4, 27.5, n)
    res_t1[f"n={n}"] = {"welch_t": round(t_, 2), "p": round(p_, 3)}
out["table1_maniptrans_auc_significance"] = res_t1
out["table1_dexmachina_auc"] = {
    "n=7_assumed": {k: round(v, 3) for k, v in zip(["t", "p"], welch(43.6, 29.8, 7, 41.0, 25.3, 7))}
}

# ---------------------------------------------------------------
# 9. 표 2 OMOMO: SPIDER의 관절 오차(ej)가 GMR보다 큼
# ---------------------------------------------------------------
out["table2_omomo_ej"] = {"GMR": 0.57, "SPIDER": 0.83, "spider_worse_on_ej": True}

# ---------------------------------------------------------------
# 10. FPS 비율 (표 1)
# ---------------------------------------------------------------
out["fps_ratio"] = {
    "spider_isaac_vs_maniptrans": round(2.5 / 0.1, 1),
    "spider_genesis_vs_dexmachina": round(1.5 / 0.05, 1),
}

path = os.path.join(HERE, "recompute_results.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
