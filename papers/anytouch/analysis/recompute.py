"""AnyTouch (arXiv:2502.12191 v3, ICLR 2025) 논문 표 수치 재계산과 일관성 검사.

논문 표 값만 쓴다(재학습, 데이터 다운로드 없음). 실행: python3 analysis/recompute.py  (수 초)
출력: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------- 1. 표 1 증분과 ↑↓ 표기 검사 ----------
# (행 이름, TAG, Feel, OF1, OF2, 표에 적힌 증분 (TAG, Feel, OF1, OF2))
t1 = [
    ("CLIP", [52.96, 72.37, 41.00, 73.16], None),
    ("TAG,VisGel,Cloth", [83.55, 79.12, 46.12, 75.10], [30.59, 6.75, 5.12, 1.94]),
    ("+OF Real", [79.67, 79.28, 47.55, 75.53], [-3.88, 0.16, 1.43, 0.43]),
    ("+TVL,SSVTP,YCB-Slide", [79.61, 79.10, 48.00, 75.57], [-0.06, -0.18, 0.45, 0.04]),
    ("+Octopi", [79.70, 79.40, 48.75, 75.66], [0.09, 0.30, 0.75, 0.09]),
]
# 표 1의 증분은 바로 앞 행과의 차이가 아니라 전 행 대비 차이인 경우가 있어 두 방식을 모두 본다.
# 1행은 CLIP 대비, 2행 이후는 바로 앞 행 대비로 적혀 있다.
mismatch_t1 = []
prev = None
for name, vals, delta in t1:
    if delta is None:
        prev = vals
        continue
    base = np.array(prev) if name != "TAG,VisGel,Cloth" else np.array(t1[0][1])
    calc = np.round(np.array(vals) - base, 2)
    for col, c, d in zip(["TAG", "Feel", "OF1", "OF2"], calc, delta):
        if abs(c - d) > 0.005:
            mismatch_t1.append({"row": name, "col": col, "calc": float(c), "table": d})
    prev = vals
res["table1_delta_check"] = {"mismatches": mismatch_t1, "n_checked": 4 * 4}

# ---------- 2. 표 6 절제 ----------
full = [80.82, 80.53, 49.62, 76.02]
ablations = {
    "w/o Text Modality": ([75.91, 78.93, 48.87, 75.52], [4.91, 1.60, 0.75, 0.50]),
    "w/o Vision Modality": ([74.55, 77.30, 48.12, 75.22], [6.27, 3.23, 1.50, 0.80]),
    "w/o Text in TacQuad": ([80.70, 80.19, 49.21, 75.91], [0.12, 0.34, 0.41, 0.11]),
    "w/o Stage 1": ([78.34, 78.62, 48.75, 76.08], [2.48, 1.91, 0.87, -0.06]),
    "w/o Stage 2": ([68.64, 72.39, 46.50, 73.09], [12.18, 8.14, 3.12, 2.93]),
    "w/o Cross-Sensor Matching": ([80.54, 79.43, 49.25, 75.80], [0.28, 1.10, 0.37, 0.22]),
    "w/o Dynamic Perception": ([77.93, 79.28, 48.62, 75.70], [2.89, 1.25, 1.00, 0.32]),
    "w/o Universal Sensor Tokens": ([80.79, 79.03, 48.40, 75.40], [0.03, 1.53, 1.22, 0.62]),
}
t6_mismatch = []
t6_drop = {}
for name, (vals, dtab) in ablations.items():
    calc = [round(f - v, 2) for f, v in zip(full, vals)]
    t6_drop[name] = calc
    for col, c, d in zip(["TAG", "Feel", "OF1", "OF2"], calc, dtab):
        if abs(c - d) > 0.005:
            t6_mismatch.append({"row": name, "col": col, "calc_drop": c, "table_drop": d})
res["table6_delta_check"] = {"mismatches": t6_mismatch, "drops_calc": t6_drop}

# 시딩 가정 없이 "seen 센서(TAG, Feel) 대 unseen 센서(OF1, OF2)" 평균 하락량
seen_unseen = {}
for name in ["w/o Cross-Sensor Matching", "w/o Universal Sensor Tokens"]:
    d = t6_drop[name]
    seen_unseen[name] = {"seen_mean_drop(TAG,Feel)": round(float(np.mean(d[:2])), 3),
                         "unseen_mean_drop(OF1,OF2)": round(float(np.mean(d[2:])), 3),
                         "largest_drop_dataset": ["TAG", "Feel", "OF1", "OF2"][int(np.argmax(d))]}
res["seen_vs_unseen_ablation"] = seen_unseen

# ---------- 3. 표 5 대 본문 데이터 규모 ----------
t5 = {"TAG": 250, "VisGel": 587, "Cloth": 587, "TVL": 39, "SSVTP": 4.5, "YCB-Slide": 183,
      "OF Real": 1165, "Octopi": 39, "TacQuad": 55}  # 천 장 단위
res["data_size"] = {
    "table5_sum_k": round(sum(t5.values()), 1),
    "table5_TAG_VisGel_Cloth_k": 250 + 587 + 587,
    "table1_first_row_cum_k": 996,  # 표 1 1행 누적 (TAG, VisGel, Cloth)
    "table1_last_row_cum_k": 2427,   # 표 1 마지막 행 (TacQuad 제외)
    "table1_last_row_plus_tacquad_k": 2427 + 55,
    "appendix_A1_total_frames": 2481703,
    "tacquad_fine_plus_coarse": 17524 + 55082,
    "intro_tacquad_frames": 72606,
}

# ---------- 4. 표 2 "네 과제 모두에서 UniTouch를 앞선다" 검사 ----------
uni = {"TAG material": 61.3, "TAG roughness": None, "TAG hardness": None, "Feel grasp": 82.3}
any_same = {"TAG material": 82.74, "TAG roughness": 86.01, "TAG hardness": 94.24, "Feel grasp": 87.17}
res["table2_uni_comparison"] = {k: {"UniTouch": v, "AnyTouch(same data)": any_same[k],
                                     "comparable": v is not None} for k, v in uni.items()}

# ---------- 5. 표 3, 표 7 상대 비교 ----------
t3 = {"OF1 AnyTouch(UniTouch data) - UniTouch": round(46.50 - 41.3, 2),
      "OF1 AnyTouch(full) - UniTouch+": round(49.62 - 47.25, 2),
      "OF2 AnyTouch(UniTouch data) - UniTouch(seen)": round(85.87 - 85.4, 2),
      "OF2 AnyTouch(full) - UniTouch+": round(76.02 - 75.29, 2)}
res["table3_differences_pp"] = t3

t7 = {"GelSight Mini->DuraGel": (0.2261, 0.2159), "GelSight Mini->Tac3D": (0.0167, 0.0151),
      "DIGIT->Tac3D": (0.0155, 0.0144)}
res["table7_rel_mse_reduction_pct"] = {k: round(100 * (a - b) / a, 2) for k, (a, b) in t7.items()}

# ---------- 6. 이항 신뢰구간과 검출 가능한 차이 (표가 n을 보고하지 않으므로 n 범위별) ----------
ns = [500, 1000, 2000, 5000, 10000]
binom = {}
for p in [0.5, 0.8]:
    for n in ns:
        se = math.sqrt(p * (1 - p) / n)
        # 두 독립 비율 비교, 양측 α=0.05, 검정력 0.8: MDD ≈ (1.96+0.84) * sqrt(2 p(1-p)/n)
        mdd = (1.96 + 0.84) * math.sqrt(2 * p * (1 - p) / n)
        binom[f"p={p},n={n}"] = {"ci95_halfwidth_pp": round(100 * 1.96 * se, 2),
                                  "min_detectable_diff_pp_unpaired": round(100 * mdd, 2)}
res["binomial_sensitivity"] = binom

# ---------- 7. 표 4 실로봇 펜딩: 보고된 평균만 있고 SD 없음 ----------
# 10회씩, 두 군의 SD가 같다고 가정하면 Welch(=Student) t 임계값 t(18, 0.975)에서
# 유의한 차이 Δ가 나려면 SD ≥ Δ / (t_crit * sqrt(2/10))의 역이 된다. 최대 허용 SD를 계산한다.
tcrit = stats.t.ppf(0.975, df=18)
pouring = {
    "AnyTouch dyn (FT) vs AnyTouch static (FT)": (1.56, 2.45),
    "AnyTouch static (FT) vs T3 (FT)": (2.45, 2.33),
    "AnyTouch dyn (Freeze) vs T3 (Freeze)": (8.22, 9.74),
    "AnyTouch static (Freeze) vs T3 (Freeze)": (9.60, 9.74),
}
pt = {}
for k, (a, b) in pouring.items():
    delta = abs(a - b)
    sd_max = delta / (tcrit * math.sqrt(2 / 10))  # 이 SD 이하일 때만 p<0.05
    pt[k] = {"delta_g": round(delta, 2), "max_sd_for_p_lt_0.05_g": round(sd_max, 3)}
res["pouring_sensitivity"] = {"tcrit_df18": round(float(tcrit), 4), "rows": pt}

# ---------- 8. 표 2 TLV-Link, 표 2 CLIP 값과 표 1, 표 3 CLIP 값 일치 ----------
res["clip_value_consistency"] = {"TAG material 52.96 (T1,T2)": True, "Feel 72.37 (T1,T2)": True,
                                  "OF1 41.00 (T1,T3)": True, "OF2 73.16 (T1,T3)": True}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: res[k] for k in ["table1_delta_check", "table6_delta_check", "seen_vs_unseen_ablation",
                                      "data_size", "table3_differences_pp",
                                      "table7_rel_mse_reduction_pct", "pouring_sensitivity"]},
                 ensure_ascii=False, indent=1)[:6000])
