"""PhysVAM (arXiv:2611.26105 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석, 논문 Table 1–8 및 본문 §4.2–4.4, 부록 B).
      외부 데이터와 원자료는 쓰지 않는다. 논문에 없는 값(시드별 결과, 에피소드별 값)은 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 시뮬레이션: 장면당 20회 × 시드 3개 = 60회 (§4.1의 "20 trials per scene"과 "three random seeds"로 읽었다).
  평균은 60회 풀링 값이다. 표의 SR이 모두 1/60(1.67%)의 배수인지로 확인한다(§1).
- "±"는 시드 3개의 표준편차로 가정한다. 모집단 SD(ddof=0)인지 표본 SD(ddof=1)인지는 논문이 밝히지 않는다.
  §1에서 정수 시드 결과로 어느 쪽과 맞는지 찾는다.
- 실로봇: 과제당 20회, 단일 시행 집계로 읽는다. 시드와 SD는 보고되지 않는다.
- 실로봇 "Overall Avg"는 장면 3개의 단순 평균이다(표 8 캡션). 풀링 비율과 다르다. 두 값을 모두 낸다.
- 검정은 시행을 독립으로 본 것이다. 같은 장면과 같은 시드 안의 시행은 독립이 아니므로 p 값은 과대평가일 수 있다.
  그래서 장면 단위 쌍체 검정(n = 10)을 함께 낸다.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
SCENES = ["Avocado", "Beer", "Potato", "Apple", "Lemon", "Can", "D-Bottle", "W-Bottle", "Egg", "Cup"]
NSEED, NTRIAL = 3, 20          # 시드당 시행 수
N_SCENE = NSEED * NTRIAL       # 장면당 60회


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher_p(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def comp(name, k1, n1, k2, n2):
    d, lo, hi = newcombe_diff(k1, n1, k2, n2)
    return {"comparison": name, "k1": k1, "n1": n1, "k2": k2, "n2": n2,
            "diff_pp": round(100 * d, 2), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
            "fisher_p": round(fisher_p(k1, n1, k2, n2), 6)}


def welch_summary(m1, s1, n1, m2, s2, n2):
    r = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return {"t": round(float(r.statistic), 3), "p": round(float(r.pvalue), 4)}


# ------------------------------------------------------------ 표 1 (시뮬, 장면별 SR %, 시간 s)
T1 = {
    "FastWAM":        [0.00, 0.00, 0.00, 11.67, 0.00, 0.00, 0.00, 1.67, 3.33, 0.00],
    "DynamicWAM":     [6.67, 5.00, 0.00, 5.00, 1.67, 0.00, 25.00, 8.33, 1.67, 0.00],
    "InternVLA-A1.5": [3.33, 0.00, 0.00, 3.33, 0.00, 0.00, 18.33, 0.00, 15.00, 0.00],
    "DynamicVLA":     [23.33, 10.00, 5.00, 8.33, 16.67, 1.67, 83.33, 10.00, 6.67, 8.33],
    "PUMA":           [10.00, 65.00, 41.67, 11.67, 85.00, 1.67, 58.33, 10.00, 0.00, 35.00],
    "DynamicVLA+":    [33.33, 21.67, 8.33, 16.67, 43.33, 0.00, 81.67, 28.33, 13.33, 5.00],
    "PUMA+":          [13.33, 85.00, 51.67, 26.67, 70.00, 0.00, 88.33, 43.33, 5.00, 31.67],
    "PhysVAM":        [60.00, 41.67, 65.00, 45.00, 88.33, 36.67, 86.67, 93.33, 28.33, 43.33],
}
T1_AVG = {"FastWAM": 1.67, "DynamicWAM": 5.33, "InternVLA-A1.5": 4.00, "DynamicVLA": 17.33,
          "PUMA": 31.83, "DynamicVLA+": 25.17, "PUMA+": 41.50, "PhysVAM": 58.83}
T1_TIME = {"FastWAM": 11.59, "DynamicWAM": 10.96, "InternVLA-A1.5": 10.82, "DynamicVLA": 8.92,
           "PUMA": 9.56, "DynamicVLA+": 9.30, "PUMA+": 8.83, "PhysVAM": 8.23}
HIGH = {"Can", "D-Bottle", "W-Bottle", "Egg", "Cup"}

# 1-1. 장면 평균 재계산과 정수 확인
chk = {}
for m, vals in T1.items():
    avg = float(np.mean(vals))
    k = [v * N_SCENE / 100 for v in vals]
    integer_ok = all(abs(x - round(x)) < 0.02 for x in k)  # 소수 둘째 자리 반올림 허용
    hi = float(np.mean([v for s, v in zip(SCENES, vals) if s in HIGH]))
    chk[m] = {"mean_recomputed": round(avg, 2), "reported": T1_AVG[m],
              "match": abs(avg - T1_AVG[m]) < 0.006, "counts_out_of_60_integer": integer_ok,
              "high_speed_mean": round(hi, 2)}
RES["t1_mean_check"] = chk
print("[표1] 평균 재계산")
for m, c in chk.items():
    print(f"  {m:15s} 재계산 {c['mean_recomputed']:6.2f}  보고 {c['reported']:6.2f}  일치 {c['match']}  "
          f"60회 정수 {c['counts_out_of_60_integer']}  고속 5장면 {c['high_speed_mean']}")

# 1-2. 최고 장면 수 (§4.2 "8 of 10")
best_wins = []
for i, s in enumerate(SCENES):
    others = {m: v[i] for m, v in T1.items() if m != "PhysVAM"}
    best_other = max(others.values())
    best_wins.append((s, T1["PhysVAM"][i] >= best_other, max(others, key=others.get)))
RES["t1_best_per_scene"] = [{"scene": s, "physvam_best": w, "top_other": o} for s, w, o in best_wins]
RES["t1_best_count"] = sum(1 for _, w, _ in best_wins if w)
print("[표1] PhysVAM 최고 장면 수:", RES["t1_best_count"], "(논문 §4.2: 8/10)",
      [s for s, w, _ in best_wins if not w], "에서 최고 아님")
print("[표1] 고속 5장면 평균: PhysVAM", chk["PhysVAM"]["high_speed_mean"],
      "DynamicVLA", chk["DynamicVLA"]["high_speed_mean"], "PUMA", chk["PUMA"]["high_speed_mean"],
      "(논문 §4.2: 57.67, 22.00, 21.00)")
print("[표1] 시간 최소 방법:", min(T1_TIME, key=T1_TIME.get), T1_TIME["PhysVAM"])

# 1-3. 시드 SD 정의 확인: 시드 3개 정수 결과로 평균과 ±를 동시에 맞춰 본다
def seed_sd_candidates(mean_pct, sd_rep, n_seed=3, n_trial=20):
    """장면 하나의 시드별 성공 횟수 k_i (0..20)를 찾아 평균과 SD가 맞는지 본다."""
    total = round(mean_pct * n_seed * n_trial / 100)  # 시드 합계 성공 횟수
    hits = {"ddof0": [], "ddof1": []}
    for ks in itertools.combinations_with_replacement(range(n_trial + 1), n_seed):
        if sum(ks) != total:
            continue
        s = np.array(ks) * 100 / n_trial
        for dd, key in ((0, "ddof0"), (1, "ddof1")):
            if abs(round(float(s.std(ddof=dd)), 2) - sd_rep) < 0.006:
                hits[key].append(ks)
    return hits


sd_checks = {}
for m, mean_rep, sd_rep in [("PhysVAM", 60.00, 10.80), ("PhysVAM", 41.67, 10.27), ("PhysVAM", 88.33, 6.24),
                            ("PUMA", 10.00, 0.00), ("PUMA+", 85.00, 12.25), ("DynamicVLA", 23.33, 2.36)]:
    h = seed_sd_candidates(mean_rep, sd_rep)
    sd_checks[f"{m} {mean_rep}±{sd_rep}"] = {"ddof0_candidates": len(h["ddof0"]), "ddof1_candidates": len(h["ddof1"]),
                                              "ddof0_example": list(h["ddof0"][0]) if h["ddof0"] else None,
                                              "ddof1_example": list(h["ddof1"][0]) if h["ddof1"] else None}
RES["seed_sd_checks"] = sd_checks
print("[표1] ± 정의 확인 (시드 3개 정수 해 존재 여부):")
for k, v in sd_checks.items():
    print("  ", k, v)

# 1-4. 장면 단위 쌍체 비교 (n = 10)
def paired(a, b):
    d = np.array(a) - np.array(b)
    t = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b) if np.any(d != 0) else None
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(len(d))
    tc = stats.t.ppf(0.975, len(d) - 1)
    return {"mean_diff_pp": round(float(m), 2), "ci95_pp": [round(float(m - tc * se), 2), round(float(m + tc * se), 2)],
            "paired_t_p": round(float(t.pvalue), 4), "wilcoxon_p": round(float(w.pvalue), 4) if w else None}


RES["paired_scene"] = {
    "PhysVAM vs PUMA": paired(T1["PhysVAM"], T1["PUMA"]),
    "PhysVAM vs PUMA+": paired(T1["PhysVAM"], T1["PUMA+"]),
    "PhysVAM vs DynamicVLA": paired(T1["PhysVAM"], T1["DynamicVLA"]),
    "PhysVAM vs DynamicVLA+": paired(T1["PhysVAM"], T1["DynamicVLA+"]),
}
print("[표1] 장면 쌍체:", json.dumps(RES["paired_scene"], ensure_ascii=False, indent=1))

# 1-5. 풀링 이항 비교 (장면 x 60회, 시행 독립 가정: 과대평가 가능)
def pooled(m):
    return int(round(sum(T1[m]) * N_SCENE / 100))

K = {m: pooled(m) for m in T1}
N_SIM = 10 * N_SCENE
RES["sim_pooled_counts"] = {m: [K[m], N_SIM] for m in K}
RES["sim_pooled_comparisons"] = [comp("PhysVAM vs PUMA", K["PhysVAM"], N_SIM, K["PUMA"], N_SIM),
                                  comp("PhysVAM vs PUMA+", K["PhysVAM"], N_SIM, K["PUMA+"], N_SIM),
                                  comp("PhysVAM vs DynamicVLA", K["PhysVAM"], N_SIM, K["DynamicVLA"], N_SIM)]
print("[표1] 풀링 횟수:", RES["sim_pooled_counts"])
for c in RES["sim_pooled_comparisons"]:
    print("  ", c)

# ------------------------------------------------------------ 표 2, 3 (시뮬 ablation, 평균 ± SD, 시드 3)
T2 = {
    "Without Value Branch": (50.50, 1.08), "Scalar Conditioning": (40.83, 1.43), "Binary Conditioning": (48.50, 2.55),
    "Single-Value (Single Query)": (37.83, 3.30), "Single-Value (Multiple Queries)": (50.50, 0.71),
    "Current-Value Prediction": (46.50, 2.27), "PhysVAM (Full)": (58.83, 1.65),
}
T3 = {
    "Without Object Motion History": (17.17, 2.39), "RGB History": (22.33, 1.55), "Object Position History": (30.67, 2.01),
    "Object Velocity History (Oracle)": (34.00, 3.27), "Object Velocity History (Estimated)": (33.67, 0.85),
    "Without Robot Motion History": (45.17, 2.05), "PhysVAM (Full)": (58.83, 1.65),
}
# 표 6, 7 장면별 값 (논문 부록 B.5)
T6 = {
    "Without Value Branch": [48.33, 36.67, 56.67, 41.67, 75.00, 40.00, 85.00, 71.67, 16.67, 33.33],
    "Scalar Conditioning": [53.33, 30.00, 45.00, 36.67, 46.67, 8.33, 73.33, 71.67, 21.67, 21.67],
    "Binary Conditioning": [68.33, 25.00, 46.67, 30.00, 53.33, 40.00, 83.33, 80.00, 13.33, 45.00],
    "Single-Value (Single Query)": [61.67, 18.33, 56.67, 31.67, 11.67, 3.33, 76.67, 58.33, 20.00, 40.00],
    "Single-Value (Multiple Queries)": [56.67, 46.67, 50.00, 23.33, 81.67, 30.00, 71.67, 96.67, 21.67, 26.67],
    "Current-Value Prediction": [65.00, 55.00, 41.67, 40.00, 38.33, 26.67, 75.00, 81.67, 8.33, 33.33],
    "PhysVAM (Full)": [60.00, 41.67, 65.00, 45.00, 88.33, 36.67, 86.67, 93.33, 28.33, 43.33],
}
T7 = {
    "Without Object Motion History": [0.00, 18.33, 1.67, 43.33, 0.00, 0.00, 78.33, 6.67, 10.00, 13.33],
    "RGB History": [5.00, 23.33, 15.00, 18.33, 6.67, 0.00, 91.67, 18.33, 21.67, 23.33],
    "Object Position History": [30.00, 23.33, 15.00, 35.00, 33.33, 1.67, 85.00, 16.67, 13.33, 53.33],
    "Object Velocity History (Oracle)": [6.67, 21.67, 41.67, 36.67, 26.67, 3.33, 88.33, 76.67, 23.33, 15.00],
    "Object Velocity History (Estimated)": [11.67, 41.67, 30.00, 28.33, 5.00, 10.00, 78.33, 83.33, 16.67, 31.67],
    "Without Robot Motion History": [51.67, 40.00, 35.00, 38.33, 41.67, 6.67, 88.33, 88.33, 28.33, 33.33],
    "PhysVAM (Full)": [60.00, 41.67, 65.00, 45.00, 88.33, 36.67, 86.67, 93.33, 28.33, 43.33],
}
T5 = {
    "Main (3-CAM)": [60.00, 41.67, 65.00, 45.00, 88.33, 36.67, 86.67, 93.33, 28.33, 43.33],
    "Reweighted (3-CAM)": [66.00, 49.33, 61.00, 42.67, 84.33, 37.67, 89.33, 91.00, 29.33, 64.33],
    "Main (2-CAM)": [55.00, 46.67, 46.67, 25.00, 45.00, 8.33, 93.33, 86.67, 28.33, 65.00],
}
T5_AVG = {"Main (3-CAM)": 58.83, "Reweighted (3-CAM)": 61.50, "Main (2-CAM)": 50.00}
T2_T6_AVG = {"Without Value Branch": 50.50, "Scalar Conditioning": 40.83, "Binary Conditioning": 48.50,
             "Single-Value (Single Query)": 37.83, "Single-Value (Multiple Queries)": 50.50,
             "Current-Value Prediction": 46.50, "PhysVAM (Full)": 58.83}
T3_T7_AVG = {"Without Object Motion History": 17.17, "RGB History": 22.33, "Object Position History": 30.67,
             "Object Velocity History (Oracle)": 34.00, "Object Velocity History (Estimated)": 33.67,
             "Without Robot Motion History": 45.17, "PhysVAM (Full)": 58.83}

cons = {}
for name, tab, avgs in [("표 6 대 표 2", T6, T2_T6_AVG), ("표 7 대 표 3", T7, T3_T7_AVG), ("표 5", T5, T5_AVG)]:
    rows = {}
    for m, vals in tab.items():
        rows[m] = {"mean_recomputed": round(float(np.mean(vals)), 2), "reported": avgs[m],
                   "match": abs(float(np.mean(vals)) - avgs[m]) < 0.006}
    cons[name] = rows
    bad = [m for m, r in rows.items() if not r["match"]]
    print(f"[{name}] 평균 불일치: {bad if bad else '없음'}")
RES["ablation_mean_consistency"] = cons

# 표 2, 3의 대표 값과 표 1/표 6/표 7의 전체 모형이 같은지
RES["full_model_same_across_tables"] = (T2["PhysVAM (Full)"] == T3["PhysVAM (Full)"] == T1_AVG["PhysVAM"] and
                                       T5_AVG["Main (3-CAM)"] == T1_AVG["PhysVAM"])

# 2-1. 표 2 대 전체 모형 (Welch, 시드 3)과 풀링 횟수 비교
full = T2["PhysVAM (Full)"]
abl = {}
for m in ["Without Value Branch", "Scalar Conditioning", "Binary Conditioning", "Single-Value (Single Query)",
          "Single-Value (Multiple Queries)", "Current-Value Prediction"]:
    mm, ss = T2[m]
    abl[m] = {"diff_pp_vs_full": round(mm - full[0], 2),
              "welch_seed3": welch_summary(full[0], full[1], 3, mm, ss, 3),
              "fisher_pooled_p": round(fisher_p(round(full[0] * 6), 600, round(mm * 6), 600), 6)}
RES["ablation_vs_full"] = abl
print("[표2] 전체 모형 대비 (Welch 시드 3, 풀링 Fisher는 과대평가):")
for k, v in abl.items():
    print("  ", k, v)

# 3-1. 표 3 객체/로봇 이력
abl3 = {}
for m in ["Without Object Motion History", "RGB History", "Object Position History", "Object Velocity History (Estimated)",
          "Without Robot Motion History"]:
    mm, ss = T3[m]
    abl3[m] = {"diff_pp_vs_full": round(mm - full[0], 2), "welch_seed3": welch_summary(full[0], full[1], 3, mm, ss, 3)}
RES["history_vs_full"] = abl3
print("[표3] 전체 대비:", json.dumps(abl3, ensure_ascii=False))
RES["oracle_vs_estimated_velocity"] = welch_summary(34.00, 3.27, 3, 33.67, 0.85, 3)

# 4. 재평가 (표 5): 2.67 pp 차이 검정
rw = {"welch_seed3": welch_summary(61.50, 1.08, 3, 58.83, 1.65, 3),
      "fisher_pooled": comp("Reweighted vs Main (3-CAM), 풀링", round(61.50 * 6), 600, round(58.83 * 6), 600)}
diff_rw = np.array(T5["Reweighted (3-CAM)"]) - np.array(T5["Main (3-CAM)"])
rw["scene_paired"] = {"n_up": int((diff_rw > 0).sum()), "n_down": int((diff_rw < 0).sum()),
                       "mean_diff_pp": round(float(diff_rw.mean()), 2)}
rw["scene_paired_t_p"] = round(float(stats.ttest_rel(T5["Reweighted (3-CAM)"], T5["Main (3-CAM)"]).pvalue), 4)
RES["reweight"] = rw
print("[표5] 재가중:", json.dumps(rw, ensure_ascii=False))
diff_view = np.array(T5["Main (3-CAM)"]) - np.array(T5["Main (2-CAM)"])
RES["view_paired"] = {"mean_diff_pp": round(float(diff_view.mean()), 2),
                       "scene_paired_t_p": round(float(stats.ttest_rel(T5["Main (3-CAM)"], T5["Main (2-CAM)"]).pvalue), 4),
                       "n_up": int((diff_view > 0).sum()), "n_down": int((diff_view < 0).sum())}
print("[표5] 3대 2 카메라:", RES["view_paired"])

# 2-2. 증강 기준선(+)과 원 기준선의 차이 (시드 3, Welch)와 시간 비교
RES["plus_variants_welch"] = {
    "DynamicVLA+ vs DynamicVLA": welch_summary(25.17, 0.85, 3, 17.33, 0.24, 3),
    "PUMA+ vs PUMA": welch_summary(41.50, 2.86, 3, 31.83, 1.70, 3),
    "PhysVAM vs PUMA+": welch_summary(58.83, 1.65, 3, 41.50, 2.86, 3),
    "PhysVAM vs PUMA+ (SD=0.5 가정 아님, 같은 SD)": None,
}
del RES["plus_variants_welch"]["PhysVAM vs PUMA+ (SD=0.5 가정 아님, 같은 SD)"]
print("[증강] 시드 Welch:", RES["plus_variants_welch"])
# 표 2-3 시간 값 교차 확인 (표 3의 시간 열과 표 7의 시간 열)
T3_TIME = {"Without Object Motion History": 9.65, "RGB History": 9.27, "Object Position History": 9.53,
           "Object Velocity History (Oracle)": 9.00, "Object Velocity History (Estimated)": 8.93,
           "Without Robot Motion History": 8.87}
T7_TIME = {"Without Object Motion History": 9.65, "RGB History": 9.27, "Object Position History": 9.53,
           "Object Velocity History (Oracle)": 9.00, "Object Velocity History (Estimated)": 8.93,
           "Without Robot Motion History": 8.86}
RES["time_table3_vs_table7_mismatch"] = [m for m in T3_TIME if T3_TIME[m] != T7_TIME[m]]
print("[일관성] 표 3과 표 7의 시간 불일치:", RES["time_table3_vs_table7_mismatch"])

# ------------------------------------------------------------ 실로봇 (표 8)
RW = {  # (성공, 시행) 논문 표 8
    "Orange": {"DynamicVLA": (6, 20), "PUMA": (7, 20), "PhysVAM": (14, 20)},
    "Cube": {"DynamicVLA": (10, 20), "PUMA": (12, 20), "PhysVAM": (15, 20)},
    "Bread": {"DynamicVLA": (5, 20), "PUMA": (7, 20), "PhysVAM": (12, 20)},
    "Cup": {"DynamicVLA": (4, 20), "PUMA": (6, 20), "PhysVAM": (14, 20)},
    "Bottle": {"DynamicVLA": (5, 20), "PUMA": (6, 20), "PhysVAM": (11, 20)},
}
SCN = {"Conveyor": ["Orange", "Cube"], "Turntable": ["Bread", "Cup"], "Rolling": ["Bottle"]}
rw_pct = {}
for m in ["DynamicVLA", "PUMA", "PhysVAM"]:
    scn_avg = {}
    for sc, objs in SCN.items():
        scn_avg[sc] = round(100 * np.mean([RW[o][m][0] / RW[o][m][1] for o in objs]), 2)
    macro = round(float(np.mean(list(scn_avg.values()))), 2)
    pooled_k = sum(RW[o][m][0] for o in RW)
    pooled_n = sum(RW[o][m][1] for o in RW)
    rw_pct[m] = {"scenario_avg": scn_avg, "macro_overall": macro, "pooled_k_n": [pooled_k, pooled_n],
                 "pooled_pct": round(100 * pooled_k / pooled_n, 2),
                 "wilson95_pooled": [round(100 * x, 1) for x in wilson(pooled_k, pooled_n)]}
RES["real_world"] = rw_pct
print("[실로봇] 장면 평균·풀링:")
for m, v in rw_pct.items():
    print("  ", m, v)
RES["real_world_comparisons"] = [
    comp("PhysVAM vs PUMA (풀링 100회)", 66, 100, 38, 100),
    comp("PhysVAM vs DynamicVLA (풀링 100회)", 66, 100, 30, 100),
]
print("[실로봇] 비교:", RES["real_world_comparisons"])
# 장면별 Fisher (PhysVAM 대 PUMA)
scene_fisher = {}
for sc, objs in SCN.items():
    kp = sum(RW[o]["PhysVAM"][0] for o in objs); np_ = sum(RW[o]["PhysVAM"][1] for o in objs)
    kb = sum(RW[o]["PUMA"][0] for o in objs); nb = sum(RW[o]["PUMA"][1] for o in objs)
    scene_fisher[sc] = {"physvam": [kp, np_], "puma": [kb, nb], "fisher_p": round(fisher_p(kp, np_, kb, nb), 4)}
RES["real_world_scene_fisher"] = scene_fisher
print("[실로봇] 장면별 Fisher 대 PUMA:", scene_fisher)
# 객체 단위 쌍체 (5개 객체, PhysVAM 대 PUMA)
objs = list(RW)
pv = [RW[o]["PhysVAM"][0] / 20 for o in objs]
pu = [RW[o]["PUMA"][0] / 20 for o in objs]
RES["real_world_object_paired"] = {"mean_diff_pp": round(100 * float(np.mean(np.array(pv) - np.array(pu))), 2),
                                    "paired_t_p": round(float(stats.ttest_rel(pv, pu).pvalue), 4), "n": 5}
print("[실로봇] 객체 쌍체 (n=5):", RES["real_world_object_paired"])
# 객체별 차이 범위 (논문 §B.6: 15–40 pp)
RES["real_world_object_gain_range"] = [round(100 * (RW[o]["PhysVAM"][0] - RW[o]["PUMA"][0]) / 20, 1) for o in objs]
print("[실로봇] 객체별 PUMA 대비 차이(pp):", RES["real_world_object_gain_range"])

# 헤드라인 차이 (§abstract)
RES["headline"] = {"sim_gain_vs_PUMA_pp": round(T1_AVG["PhysVAM"] - T1_AVG["PUMA"], 2),
                   "sim_gain_vs_PUMA_plus_pp": round(T1_AVG["PhysVAM"] - T1_AVG["PUMA+"], 2),
                   "real_gain_vs_PUMA_pp": round(rw_pct["PhysVAM"]["macro_overall"] - rw_pct["PUMA"]["macro_overall"], 2)}
print("[헤드라인]", RES["headline"])

# ------------------------------------------------------------ 시간 지표 교란 (§4.1: 드롭·타임아웃 포함 종료)
RES["time_note"] = {"PhysVAM": T1_TIME["PhysVAM"], "PUMA": T1_TIME["PUMA"], "PUMA+": T1_TIME["PUMA+"],
                    "pearson_r_SR_vs_time_8methods": None}
sr = [T1_AVG[m] for m in T1_TIME]
tm = [T1_TIME[m] for m in T1_TIME]
RES["time_note"]["pearson_r_SR_vs_time_8methods"] = round(float(stats.pearsonr(sr, tm)[0]), 3)
print("[시간] SR과 시간의 상관 (8개 방법):", RES["time_note"]["pearson_r_SR_vs_time_8methods"])

# ------------------------------------------------------------ 표 4 (Lift–Miss 예측 변화, 220회)
L = dict(n=119, m=0.254, s=0.370); Mi = dict(n=101, m=-0.185, s=0.237)
w = stats.ttest_ind_from_stats(L["m"], L["s"], L["n"], Mi["m"], Mi["s"], Mi["n"], equal_var=False)
gap_se = math.sqrt(L["s"] ** 2 / L["n"] + Mi["s"] ** 2 / Mi["n"])
dfw = (L["s"] ** 2 / L["n"] + Mi["s"] ** 2 / Mi["n"]) ** 2 / ((L["s"] ** 2 / L["n"]) ** 2 / (L["n"] - 1) + (Mi["s"] ** 2 / Mi["n"]) ** 2 / (Mi["n"] - 1))
tc = stats.t.ppf(0.975, dfw)
gap = L["m"] - Mi["m"]
RES["lift_miss_16"] = {"gap": round(gap, 3), "ci95": [round(gap - tc * gap_se, 3), round(gap + tc * gap_se, 3)],
                       "welch_p": float(w.pvalue)}
RES["lift_miss_16"]["welch_p"] = f"{float(w.pvalue):.2e}"
# 비율 확인
RES["lift_miss_16"]["frac_up_lift"] = [96, 119, round(96 / 119, 3)]
RES["lift_miss_16"]["frac_down_miss"] = [77, 101, round(77 / 101, 3)]
RES["lift_miss_8_gap"] = round(0.103 + 0.150, 3)
print("[표4] Lift–Miss(±16) 차이:", RES["lift_miss_16"], " ±8 차이:", RES["lift_miss_8_gap"])
RES["lift_miss_note"] = ("220회는 같은 에피소드 안에 묶여 있다. 에피소드 단위 군집 보정이 없어 Welch p는 과대평가일 수 있다. "
                         "33개 에피소드 쌍체 +0.319와 부트스트랩 CI [+0.125, +0.431]은 에피소드별 값이 없어 재계산하지 못했다.")

# 표 4의 ±8 구간 과 ±16 구간 비율 일관성은 본문과 같음

# ------------------------------------------------------------ 저장
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
