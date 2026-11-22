"""TapTacTouch (arXiv:2611.24207 v1) 표 재계산과 일관성 검사 (numpy, scipy). 실행: python3 analysis/recompute.py
결과: analysis/recompute_results.json

값은 paper/paper.txt(pdftotext -layout)의 표 I, II, III, IV에서 옮겼다. 원자료(대상별 반복 측정, 접촉 수, sP 값)는 논문에 없으므로
검사는 '보고된 값끼리' 맞는지와, 대상 여섯 개를 짝지은 비교로 제한된다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}
OBJS = ["O1", "O2", "O3", "O4", "O5", "O6"]

# ---------------------------------------------------------------- 표 I (Ours, 대상별): CD mm, EMD mm, F %, 시간 s
T1 = {
    "CD":   [1.589, 1.500, 1.602, 1.166, 1.204, 2.602],
    "EMD":  [7.993, 7.144, 6.817, 7.405, 7.038, 10.033],
    "F":    [96.98, 95.82, 95.71, 99.14, 97.86, 84.49],
    "time": [61.67, 13.01, 34.10, 47.94, 127.60, 15.64],
}
T1_REPORTED_AVG = {"CD": 1.610, "EMD": 7.738, "F": 95.00, "time": 49.99}

# ---------------------------------------------------------------- 표 IV (fill 후 전체 CD, 대상별)
T4 = {
    "Single-scale GPIS":          [4.456, 3.277, 4.324, 3.362, 2.915, 4.261],
    "Multi-scale GPIS":           [1.673, 1.828, 1.791, 1.193, 1.229, 2.521],
    "Multi-scale GPIS+PCG":       [1.674, 1.766, 1.758, 1.205, 1.213, 2.533],
    "Multi-scale GPIS+CSP":       [1.569, 1.597, 1.684, 1.150, 1.214, 2.594],
    "Ours (CSP+PCG)":             [1.590, 1.552, 1.680, 1.166, 1.204, 2.602],
}
T4_REPORTED_AVG = {"Single-scale GPIS": 3.766, "Multi-scale GPIS": 1.706, "Multi-scale GPIS+PCG": 1.691,
                   "Multi-scale GPIS+CSP": 1.635, "Ours (CSP+PCG)": 1.632}

# ---------------------------------------------------------------- 표 II (관측 영역, 여섯 대상 평균만 보고)
T2 = {  # CD, EMD, F
    "Single-scale GPIS": (3.448, 11.060, 75.580),
    "Multi-scale GPIS": (1.244, 6.491, 97.504),
    "Multi-scale GPIS+PCG": (1.248, 6.491, 97.504),
    "Multi-scale GPIS+CSP": (1.229, 6.818, 97.497),
    "Ours (CSP+PCG)": (1.236, 6.637, 97.509),
}
# ---------------------------------------------------------------- 표 III (미관측 영역, 평균만)
T3 = {
    "Single-scale GPIS": (4.076, 13.019, 71.253),
    "Multi-scale GPIS": (2.133, 8.968, 89.705),
    "Multi-scale GPIS+PCG": (2.102, 8.731, 90.124),
    "Multi-scale GPIS+CSP": (2.003, 8.573, 92.148),
    "Ours (CSP+PCG)": (1.992, 8.862, 92.605),
}


def mean_sd(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1))


def t_ci(x, level=0.95):
    x = np.asarray(x, float)
    n = len(x)
    m, s = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * s / np.sqrt(n)
    return float(m - h), float(m + h)


# ---------------------------------------------------------------- 1. 표 I 평균 재계산
res["table1_recomputed"] = {}
for k, v in T1.items():
    m, s = mean_sd(v)
    lo, hi = t_ci(v)
    res["table1_recomputed"][k] = {"mean": round(m, 4), "sd": round(s, 3), "ci95_t_n6": [round(lo, 3), round(hi, 3)],
                                   "reported_mean": T1_REPORTED_AVG[k],
                                   "match_at_reported_precision": bool(abs(m - T1_REPORTED_AVG[k]) <= 0.005 * 10 ** -(len(str(T1_REPORTED_AVG[k]).split('.')[1]) - 2) + 1e-9)}

# ---------------------------------------------------------------- 2. 표 IV 평균 재계산
res["table4_recomputed"] = {}
for k, v in T4.items():
    m, s = mean_sd(v)
    res["table4_recomputed"][k] = {"mean": round(m, 4), "sd": round(s, 3), "reported_mean": T4_REPORTED_AVG[k],
                                   "match_at_reported_precision": bool(abs(m - T4_REPORTED_AVG[k]) <= 0.0005 + 1e-9)}

# ---------------------------------------------------------------- 3. 표 I 대 표 IV (Ours 대상별 CD)
d14 = [round(a - b, 3) for a, b in zip(T1["CD"], T4["Ours (CSP+PCG)"])]
res["table1_vs_table4_ours_CD_diff"] = dict(zip(OBJS, d14))
res["table1_vs_table4_ours_CD_mean"] = {"table1": 1.610, "table4_ours": T4_REPORTED_AVG["Ours (CSP+PCG)"]}

# ---------------------------------------------------------------- 4. 표 II, III 와 표 I 의 관측 비율 역산
# 전체 CD/EMD/F 가 관측 영역 값과 미관측 영역 값의 가중 평균이라고 보면, 관측 비율 w는 다음과 같다(근사).
obs, unobs = T2["Ours (CSP+PCG)"], T3["Ours (CSP+PCG)"]
overall = {"CD": 1.610, "EMD": 7.738, "F": 95.00}
w = {}
w["CD"] = (unobs[0] - overall["CD"]) / (unobs[0] - obs[0])
w["EMD"] = (unobs[1] - overall["EMD"]) / (unobs[1] - obs[1])
w["F"] = (overall["F"] - unobs[2]) / (obs[2] - unobs[2])
res["implied_observed_fraction_ours"] = {k: round(v, 3) for k, v in w.items()}
res["overall_within_obs_unobs_range"] = {
    "CD": bool(obs[0] <= overall["CD"] <= unobs[0]),
    "EMD": bool(obs[1] <= overall["EMD"] <= unobs[1]),
    "F": bool(unobs[2] <= overall["F"] <= obs[2]),
}

# ---------------------------------------------------------------- 5. 표 II/III 주장 수치 확인
res["text_claims_tables23"] = {
    "unobs_CD_multi_to_ours_rel_change_pct": round(100 * (1.992 - 2.133) / 2.133, 2),
    "unobs_F_multi_to_ours_pt": round(92.605 - 89.705, 3),
    "unobs_EMD_csp_vs_ours_pct": round(100 * (8.862 - 8.573) / 8.573, 2),
    "unobs_CD_csp_vs_ours_mm": round(1.992 - 2.003, 3),
    "fill_CD_ours_vs_csp_mm": round(1.632 - 1.635, 4),
    "pcg_alone_unobs_CD_pct": round(100 * (2.102 - 2.133) / 2.133, 2),
    "pcg_alone_fill_CD_pct": round(100 * (1.691 - 1.706) / 1.706, 2),
}

# ---------------------------------------------------------------- 6. 대상 짝비교 (표 IV, n = 6 대상, 대상당 1회로 가정)
def paired(a, b):
    d = np.asarray(T4[a], float) - np.asarray(T4[b], float)
    m, s = d.mean(), d.std(ddof=1)
    t, p_t = stats.ttest_rel(T4[a], T4[b])
    lo, hi = t_ci(d)
    # 정확 부호 뒤집기 검정 (두 측 평균 차), 2^6 = 64 조합
    dd = np.abs(d).sum()
    cnt = 0
    tot = 0
    for signs in range(64):
        s_sum = 0.0
        for i in range(6):
            s_sum += (1 if (signs >> i) & 1 else -1) * d[i]
        tot += 1
        if abs(s_sum) >= abs(d.sum()) - 1e-12:
            cnt += 1
    p_exact = cnt / tot
    wins = int((d < 0).sum())
    return {"mean_diff_mm": round(m, 4), "ci95_t": [round(lo, 4), round(hi, 4)], "p_paired_t": round(float(p_t), 4),
            "p_signflip_exact": round(p_exact, 4), "objects_a_lower": f"{wins}/6", "diff_per_object": [round(x, 3) for x in d]}


res["paired_CD_fill_table4"] = {
    "Multi-scale vs Single-scale": paired("Multi-scale GPIS", "Single-scale GPIS"),
    "Ours vs Multi-scale": paired("Ours (CSP+PCG)", "Multi-scale GPIS"),
    "Ours vs CSP only": paired("Ours (CSP+PCG)", "Multi-scale GPIS+CSP"),
    "CSP only vs Multi-scale": paired("Multi-scale GPIS+CSP", "Multi-scale GPIS"),
    "PCG only vs Multi-scale": paired("Multi-scale GPIS+PCG", "Multi-scale GPIS"),
}

# ---------------------------------------------------------------- 7. 상대 크기 (객체 폭 80–100 mm 기준)
res["CD_relative_to_object_size"] = {
    "CD_mean_over_90mm_pct": round(100 * 1.610 / 90, 2),
    "CD_over_input_noise_sd_0.5mm": round(1.610 / 0.5, 2),
    "note": "입력 접촉에 더한 노이즈 표준편차 0.5 mm 대비 평균 CD 1.61 mm (3.2배). 노이즈 자체가 CD 바닥의 주된 원인인지는 논문이 검증하지 않음.",
}

# ---------------------------------------------------------------- 8. 시간 합계
res["time_check"] = {"sum_table1_s": round(sum(T1["time"]), 2), "mean": round(np.mean(T1["time"]), 3),
                     "note": "보고값 49.99 s와 일치"}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- 출력
print("[표 I 평균]")
for k, v in res["table1_recomputed"].items():
    print(f"  {k:5s} mean={v['mean']:.4f} sd={v['sd']:.3f} CI95={v['ci95_t_n6']} 보고={v['reported_mean']} 일치={v['match_at_reported_precision']}")
print("[표 IV 평균]")
for k, v in res["table4_recomputed"].items():
    print(f"  {k:24s} mean={v['mean']:.4f} 보고={v['reported_mean']} 일치={v['match_at_reported_precision']}")
print("[표 I vs 표 IV, Ours CD 차이 (표I - 표IV)]", res["table1_vs_table4_ours_CD_diff"])
print("[관측 비율 역산]", res["implied_observed_fraction_ours"], res["overall_within_obs_unobs_range"])
print("[짝비교, 표 IV CD, mm]")
for k, v in res["paired_CD_fill_table4"].items():
    print(f"  {k:28s} diff={v['mean_diff_mm']:+.4f} CI={v['ci95_t']} p_t={v['p_paired_t']} p_flip={v['p_signflip_exact']} a<b={v['objects_a_lower']}")
print("[텍스트 주장 수치]", res["text_claims_tables23"])
print("[상대 크기]", res["CD_relative_to_object_size"])
