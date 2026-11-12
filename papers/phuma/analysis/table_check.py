"""PHUMA 논문 표 재계산과 검정 (논문 수치만 사용, 데이터 다운로드 없음).

실행: python3 analysis/table_check.py  -> analysis/table_check_results.json
출력: 표 간 일치 여부, 실물 오차 감소율, 데이터 분할 개수, 프레임 기준 계산 시간,
      두 비율 z 검정(성공률), 시드 3개 Welch t 검정, 이항 95% 구간.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("table_check_results.json")
RESULTS = {}


def two_prop_z(p1, n1, p2, n2):
    """두 성공률(%)의 차이에 대한 pooled z 검정. 샘플은 독립 베르누이로 가정."""
    p1, p2 = p1 / 100, p2 / 100
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"diff_pts": round((p1 - p2) * 100, 2), "z": round(z, 2), "p_two_sided": float(f"{p:.3g}")}


def wilson(p, n, z=1.96):
    p = p / 100
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round((c - h) * 100, 1), round((c + h) * 100, 1)


def welch(m1, s1, m2, s2, n=3):
    """시드 평균 ± 표준편차(표본 표준편차로 가정), 시드 수 n."""
    t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    return {"diff": round(m1 - m2, 2), "t": round(float(t), 2), "p_two_sided": float(f"{p:.3g}")}


# ---------- 1. 표 간 일치: 같은 값이 여러 표에 있으면 확인 ----------
# Table 4 (PHUMA 열, 0.15 m 임계) 와 Table 20/21 (0.15 m 행)
t4 = {
    "LaFAN1": {"test": 46.1, "unseen": 28.4},
    "AMASS": {"test": 76.2, "unseen": 78.2},  # Table 4 원문 값
    "Humanoid-X": {"test": 50.6, "unseen": 39.1},
    "PHUMA": {"test": 92.7, "unseen": 82.9},
}
t21_015 = {  # Table 21 (a) G1, 0.15 m
    "LaFAN1": {"test": 46.1, "unseen": 28.4},
    "AMASS": {"test": 76.2, "unseen": 70.2},
    "Humanoid-X": {"test": 50.6, "unseen": 39.1},
    "PHUMA": {"test": 92.7, "unseen": 82.9},
}
t3_sink = {"test": 76.2, "unseen": 70.2}  # Table 3 SINK 행 (AMASS)
mismatch = []
for k in t4:
    for part in ("test", "unseen"):
        if t4[k][part] != t21_015[k][part]:
            mismatch.append({"dataset": k, "split": part, "table4": t4[k][part], "table21_015": t21_015[k][part]})
RESULTS["table4_vs_table21_mismatch"] = mismatch
RESULTS["table3_sink_vs_table21_amass"] = {"table3_sink_unseen": t3_sink["unseen"], "table21_amass_unseen": 70.2,
                                          "table4_amass_unseen": 78.2}

# Table 4 의 AMASS 카테고리 값 (90.7, 75.0, 62.7, 44.1)은 Table 21 0.15 m 값과 같다.
# 그러면 총합도 70.2여야 한다. 카테고리 평균(단순 평균)과 비교해 본다.
cat_unseen_amass = [90.7, 75.0, 62.7, 44.1]
RESULTS["amass_unseen_simple_mean_of_categories"] = round(float(np.mean(cat_unseen_amass)), 1)

# Table 3 PhySINK 행과 Table 9/34의 PhySINK 값이 다른 학습 데이터에서 나왔는지 확인 (AMASS 대 PHUMA 원천)
RESULTS["phusink_amass_test_table3"] = 79.5
RESULTS["phusink_phumasrc_test_table9"] = 89.9

# ---------- 2. 실물 오차 감소율 (Table 5, 본문 16.3%) ----------
emp_amass, emp_phuma = 41.0, 34.3
emp_vel_amass, emp_vel_phuma = 3.20, 2.69
emp_acc_amass, emp_acc_phuma = 1.50, 1.29
RESULTS["real_reduction_pct"] = {
    "empjpe_total": round((emp_amass - emp_phuma) / emp_amass * 100, 1),
    "dof_vel_total": round((emp_vel_amass - emp_vel_phuma) / emp_vel_amass * 100, 1),
    "dof_acc_total": round((emp_acc_amass - emp_acc_phuma) / emp_acc_amass * 100, 1),
}
# Table 5 행 평균이 Total과 맞는지 (단순 평균; 카테고리 수가 같을 때만 근사)
amass_cat = [27.3, 47.5, 49.9, 39.4]
phuma_cat = [23.2, 40.4, 42.6, 31.1]
RESULTS["table5_empjpe_simple_mean"] = {"AMASS": round(float(np.mean(amass_cat)), 2),
                                        "PHUMA": round(float(np.mean(phuma_cat)), 2)}

# ---------- 3. 데이터 구성 (Table 8, 분할 파일) ----------
clips = {"LocoMuJoCo": 0.78, "GRAB": 1.73, "EgoBody": 2.12, "LAFAN1": 2.18, "AMASS": 21.73,
         "HAA500": 1.76, "Motion-X Video": 33.04, "HuMMan": 0.50, "AIST": 1.75, "IDEA400": 9.94,
         "PHUMA Video": 0.50}
hours = {"LocoMuJoCo": 0.86, "GRAB": 1.88, "EgoBody": 2.19, "LAFAN1": 2.40, "AMASS": 20.86,
         "HAA500": 1.01, "Motion-X Video": 31.98, "HuMMan": 0.47, "AIST": 1.66, "IDEA400": 9.10,
         "PHUMA Video": 0.56}
RESULTS["table8_sum_clips_K"] = round(sum(clips.values()), 2)  # 표 값 76.01K
RESULTS["table8_sum_hours"] = round(sum(hours.values()), 2)  # 표 값 72.96h (표 본문 73.0h)
RESULTS["table8_video_hours"] = round(sum(v for k, v in hours.items() if k not in
                                          ("LocoMuJoCo", "GRAB", "EgoBody", "LAFAN1", "AMASS")), 2)
RESULTS["table8_mocap_hours"] = round(sum(v for k, v in hours.items() if k in
                                          ("LocoMuJoCo", "GRAB", "EgoBody", "LAFAN1", "AMASS")), 2)
RESULTS["split_counts"] = {"train": 67953, "test": 7557, "unseen_video": 504,
                           "train_plus_test": 67953 + 7557, "total_clips_listed": 76014}
RESULTS["test_fraction_of_total_pct"] = round(7557 / 76014 * 100, 1)  # 본문 "10%"

# ---------- 4. Humanoid-X 필터 후 비율 (Table 2, Table 31, 본문 27.1%) ----------
RESULTS["humanoidx_remaining_pct"] = {
    "62.2/231.4": round(62.2 / 231.4 * 100, 1),
    "62.2/237.2": round(62.2 / 237.2 * 100, 1),
    "text_claim": 27.1,
}
RESULTS["humanoidx_hours_reported"] = {"figure2a_feasible": 69.1, "table2_all_filters": 62,
                                       "table31_original": 237.2, "table4_original": 231.4}

# ---------- 5. 계산 시간 (Table 10, RTX 3090 기준 프레임/초) ----------
frames_total = 7.88e6  # Table 8
for name, fps_ in {"PhySINK": 0.92, "SINK": 0.95}.items():
    secs = frames_total / fps_
    RESULTS.setdefault("compute_days_single_gpu", {})[name] = round(secs / 86400, 1)
RESULTS["clip_frames_mean"] = round(7.88e6 / 76.01e3, 1)  # 클립당 프레임 (약 3.5 s at 30 Hz)

# ---------- 6. 성공률 비율 검정 (독립 베르누이 가정) ----------
N_TEST, N_UNSEEN = 7557, 504
RESULTS["ci95_wilson"] = {}
for label, p, n in [("PhySINK test 79.5", 79.5, N_TEST), ("PHUMA test 92.7", 92.7, N_TEST),
                    ("PhySINK unseen 72.8", 72.8, N_UNSEEN), ("SINK unseen 70.2", 70.2, N_UNSEEN),
                    ("PHUMA unseen 82.9", 82.9, N_UNSEEN), ("Humanoid-X unseen 39.1", 39.1, N_UNSEEN)]:
    RESULTS["ci95_wilson"][label] = {"n": n, "ci": wilson(p, n)}

RESULTS["z_tests"] = {
    "T3 PhySINK vs SINK, PHUMA test (n=7557)": two_prop_z(79.5, N_TEST, 76.2, N_TEST),
    "T3 PhySINK vs SINK, Unseen (n=504)": two_prop_z(72.8, N_UNSEEN, 70.2, N_UNSEEN),
    "T3 PhySINK vs GMR, Unseen (n=504)": two_prop_z(72.8, N_UNSEEN, 59.3, N_UNSEEN),
    "T4 PHUMA vs AMASS, PHUMA test (n=7557)": two_prop_z(92.7, N_TEST, 76.2, N_TEST),
    "T4 PHUMA vs AMASS, Unseen (n=504, table4 78.2)": two_prop_z(82.9, N_UNSEEN, 78.2, N_UNSEEN),
    "T4 PHUMA vs AMASS, Unseen (n=504, table21 70.2)": two_prop_z(82.9, N_UNSEEN, 70.2, N_UNSEEN),
    "T21 PHUMA vs AMASS 0.5 m, Unseen (n=504)": two_prop_z(93.7, N_UNSEEN, 92.3, N_UNSEEN),
    "T20 PHUMA vs AMASS 0.5 m, PHUMA test (n=7557)": two_prop_z(97.1, N_TEST, 90.2, N_TEST),
    "T9 PhySINK vs GMR, Unseen (n=504)": two_prop_z(81.7, N_UNSEEN, 75.2, N_UNSEEN),
}

# ---------- 7. 시드 3개 Welch t 검정 (표 25, 30, 32, 35; 표준편차는 표본 표준편차로 가정) ----------
RESULTS["seed_welch"] = {
    "T25 PhySINK vs SINK, PHUMA test (GMR) 0.5m": welch(83.4, 0.2, 81.6, 0.2),
    "T25 PhySINK vs SINK, Unseen (GMR) 0.5m": welch(83.7, 0.4, 79.8, 0.6),
    "T30 +Video 33% vs MoCap only, Unseen 0.5m": welch(80.1, 0.8, 86.0, 0.2),
    "T30 +Video 100% vs MoCap only, Unseen 0.5m": welch(90.2, 0.9, 86.0, 0.2),
    "T32 No FC filter vs all filters, Unseen": welch(80.0, 0.5, 90.2, 0.9),
    "T35 No Grounding vs PhySINK, Unseen": welch(84.5, 0.8, 90.2, 0.9),
    "T35 No Skating vs PhySINK, PHUMA test": welch(94.2, 0.1, 94.6, 0.1),
    "T35 No Skating vs PhySINK, Unseen": welch(88.2, 0.8, 90.2, 0.9),
}

# ---------- 8. 보고된 수치 검산 ----------
RESULTS["table2_all_filters_vs_unfiltered_pts"] = round(87.0 - 50.6, 1)
RESULTS["fig2_pct_gain_phuma_vs_amass_real"] = 16.3

OUT.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2))
print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
