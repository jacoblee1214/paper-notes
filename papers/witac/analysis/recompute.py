"""WiTac (arXiv:2611.21057 v2) 보고 수치 재계산. 표준 라이브러리만 사용한다.

입력: 논문 표의 값(paper.txt 및 pdftotext -layout으로 확인). 출력: recompute_results.json.
- 표 1, 2, 5의 성공 횟수 (20회 시행 × 4과제 = 과제별 20회)에서 평균·Wilson 95% 구간·Fisher 정확 검정
- 표 3과 표 S6·S7에서 상대 감소율 재계산
- 표 S4, S5에서 상대 오차 감소·macro-F1 차이 재계산
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "recompute_results.json")


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _hyp_pmf(a, r1, c1, n):
    # 2x2 표에서 셀 a(처치 성공)의 초기하 확률. r1=처치 행합, c1=성공 열합, n=전체
    return math.comb(r1, a) * math.comb(n - r1, c1 - a) / math.comb(n, c1)


def fisher_two_sided(k1, n1, k2, n2):
    """두 비율 k1/n1, k2/n2의 양측 Fisher 정확 검정 p값."""
    n = n1 + n2
    c1 = k1 + k2  # 전체 성공 수
    p_obs = _hyp_pmf(k1, n1, c1, n)
    lo, hi = max(0, c1 - n2), min(n1, c1)
    total = 0.0
    for a in range(lo, hi + 1):
        pa = _hyp_pmf(a, n1, c1, n)
        if pa <= p_obs * (1 + 1e-7):
            total += pa
    return min(1.0, total)


def pct(k, n):
    return round(100.0 * k / n, 2)


res = {}

# ---- 표 1: 성공률 (과제별 20회, 4과제) ----
# 과제 순서: battery, bowl, USB, cabinet. 퍼센트를 20회 기준 횟수로 환산.
T1 = {
    "FTP-1": [40, 55, 5, 60],
    "T-Rex": [45, 65, 15, 75],
    "GR00T base": [30, 40, 10, 70],
    "GR00T + WiTac-NW": [45, 60, 5, 80],
    "GR00T + VT-MUSE": [40, 70, 15, 75],
    "GR00T + WiTac": [55, 80, 40, 90],
    "pi05 base": [55, 65, 10, 45],
    "pi05 + WiTac-NW": [60, 65, 15, 65],
    "pi05 + WiTac": [70, 75, 20, 70],
}
N_TASK = 20
t1 = {}
for name, pcts in T1.items():
    counts = [int(round(p * N_TASK / 100)) for p in pcts]
    k = sum(counts)
    n = N_TASK * len(counts)
    lo, hi = wilson(k, n)
    t1[name] = {
        "task_counts": counts,
        "avg_pct_recomputed": pct(k, n),
        "avg_pct_reported": round(sum(pcts) / len(pcts), 2),
        "pooled_ci95_wilson_pct": [round(100 * lo, 1), round(100 * hi, 1)],
    }
res["table1_average_recomputed"] = t1


def fisher_vs(base, other, counts_base, counts_other):
    out = {}
    for i, task in enumerate(["battery", "bowl", "USB", "cabinet"]):
        out[task] = round(fisher_two_sided(counts_other[i], N_TASK, counts_base[i], N_TASK), 4)
    kb, kO = sum(counts_base), sum(counts_other)
    out["pooled_80_vs_80"] = round(fisher_two_sided(kO, 4 * N_TASK, kb, 4 * N_TASK), 6)
    return out


c = {k: [int(round(p * N_TASK / 100)) for p in v] for k, v in T1.items()}
res["fisher_p_twosided"] = {
    "GR00T WiTac vs GR00T base": fisher_vs("base", "WiTac", c["GR00T base"], c["GR00T + WiTac"]),
    "GR00T WiTac vs WiTac-NW": fisher_vs("NW", "WiTac", c["GR00T + WiTac-NW"], c["GR00T + WiTac"]),
    "GR00T WiTac vs VT-MUSE": fisher_vs("VT", "WiTac", c["GR00T + VT-MUSE"], c["GR00T + WiTac"]),
    "pi05 WiTac vs pi05 base": fisher_vs("base", "WiTac", c["pi05 base"], c["pi05 + WiTac"]),
    "pi05 WiTac vs WiTac-NW": fisher_vs("NW", "WiTac", c["pi05 + WiTac-NW"], c["pi05 + WiTac"]),
}

# ---- 표 2: 접촉 단계 결과 (20회) ----
T2 = {
    "GR00T": {"misalign": 7, "extract": 10},
    "GR00T + WiTac-NW": {"misalign": 4, "extract": 12},
    "GR00T + WiTac": {"misalign": 1, "extract": 19},
}
res["table2_contact_stage"] = {
    "fisher_extract_WiTac_vs_NW": round(fisher_two_sided(19, 20, 12, 20), 4),
    "fisher_extract_WiTac_vs_base": round(fisher_two_sided(19, 20, 10, 20), 4),
    "fisher_misalign_WiTac_vs_base_7_of_20_vs_1_of_20": round(fisher_two_sided(1, 20, 7, 20), 4),
    "fisher_misalign_WiTac_vs_NW_4_vs_1": round(fisher_two_sided(1, 20, 4, 20), 4),
    "wilson_extract_WiTac_pct": [round(100 * x, 1) for x in wilson(19, 20)],
    "wilson_extract_NW_pct": [round(100 * x, 1) for x in wilson(12, 20)],
}

# ---- 표 5: 설계 제거 실험 (GR00T, 20회 기준 과제별 횟수) ----
T5 = {
    "w/o pretraining": [45, 75, 15, 80],
    "w/o visual recovery": [45, 80, 25, 85],
    "separate injection": [40, 65, 15, 80],
    "joint (WiTac)": [55, 80, 40, 90],
}
t5 = {}
for name, pcts in T5.items():
    counts = [int(round(p * N_TASK / 100)) for p in pcts]
    t5[name] = {"avg_recomputed": pct(sum(counts), 80), "counts": counts}
res["table5_ablation"] = t5
joint = T5["joint (WiTac)"]
res["table5_diffs_joint_minus_variant_pp"] = {
    k: [joint[i] - v[i] for i in range(4)] for k, v in T5.items() if k != "joint (WiTac)"
}
res["table5_fisher_joint_vs_wo_visual"] = round(
    fisher_two_sided(53, 80, 47, 80), 4
)  # 66.3%=53/80 vs 58.8%=47/80
res["table5_fisher_joint_vs_wo_pretraining"] = round(fisher_two_sided(53, 80, 43, 80), 4)  # 53.8%=43/80
res["table5_fisher_joint_vs_separate"] = round(fisher_two_sided(53, 80, 40, 80), 4)  # 50.0%=40/80


# ---- 표 S4: 대조 오차 재구성 상대 감소 ----
S4 = {
    "area_all": (0.3948, 0.3649, 7.57),
    "area_contact": (0.7723, 0.7623, 1.30),
    "intensity_all": (0.3706, 0.3335, 10.00),
    "intensity_contact": (0.7406, 0.6997, 5.53),
    "center_px": (6.5204, 6.3481, 2.64),
}
res["tableS4_rel_reduction"] = {
    k: {"recomputed_pct": round(100 * (a - b) / a, 2), "reported_pct": r}
    for k, (a, b, r) in S4.items()
}

# ---- 표 S5: macro-F1 차이 (pp) ----
S5 = {
    "12/O": {"N": 3487, "DINO": .7140, "NW": .7166, "Concat": .7828, "WiTac": .7931},
    "12/C": {"N": 3233, "DINO": .7322, "NW": .7189, "Concat": .7787, "WiTac": .7761},
    "20/O": {"N": 3555, "DINO": .4894, "NW": .5600, "Concat": .5792, "WiTac": .5783},
    "20/C": {"N": 3288, "DINO": .4935, "NW": .5578, "Concat": .5463, "WiTac": .5949},
}
diffs = {}
for k, v in S5.items():
    diffs[k] = {
        "WiTac_minus_DINO_pp": round(100 * (v["WiTac"] - v["DINO"]), 2),
        "WiTac_minus_NW_pp": round(100 * (v["WiTac"] - v["NW"]), 2),
        "WiTac_minus_Concat_pp": round(100 * (v["WiTac"] - v["Concat"]), 2),
    }
res["tableS5_macroF1_diffs"] = diffs
res["tableS5_claim_check"] = {
    "DINO_range_text_4.39_10.14": [min(d["WiTac_minus_DINO_pp"] for d in diffs.values()),
                                    max(d["WiTac_minus_DINO_pp"] for d in diffs.values())],
    "NW_range_text_1.83_7.65": [min(d["WiTac_minus_NW_pp"] for d in diffs.values()),
                                 max(d["WiTac_minus_NW_pp"] for d in diffs.values())],
    "Concat_abs_max_small_three_settings": sorted(abs(d["WiTac_minus_Concat_pp"]) for d in diffs.values()),
}

# ---- 표 S6/S7/표 3: 노이즈 실험 ----
S6 = {  # (NW, WiTac) 정규화 드리프트, JS, 출력 드리프트 - 과제 순서 동일
    "GR00T insert": ((0.5078, 0.3177), (0.1626, 0.0658), (1.4321, 1.1506)),
    "GR00T wipe": ((0.5413, 0.3484), (0.1437, 0.0878), (1.1974, 0.8520)),
    "GR00T USB": ((0.4966, 0.3130), (0.1711, 0.0348), (1.6406, 1.2741)),
    "GR00T cabinet": ((0.5415, 0.3028), (0.2133, 0.1560), (1.4558, 1.1163)),
    "pi05 insert": ((0.5078, 0.3177), (0.1023, 0.0353), (1.0884, 0.8215)),
    "pi05 wipe": ((0.5413, 0.3484), (0.0584, 0.0289), (1.3652, 1.1510)),
    "pi05 USB": ((0.4966, 0.3130), (0.1059, 0.0384), (1.1275, 0.8561)),
    "pi05 cabinet": ((0.5415, 0.3028), (0.0789, 0.0248), (1.6630, 1.3620)),
}
rel = {}
for k, (drift, js, outd) in S6.items():
    rel[k] = {
        "feature_drift_red_pct": round(100 * (drift[0] - drift[1]) / drift[0], 2),
        "attn_JS_red_pct": round(100 * (js[0] - js[1]) / js[0], 2),
        "contact_out_drift_red_pct": round(100 * (outd[0] - outd[1]) / outd[0], 2),
    }
res["tableS6_relative_reductions"] = rel
res["tableS6_ranges"] = {
    "feature_drift": [min(r["feature_drift_red_pct"] for r in rel.values()),
                      max(r["feature_drift_red_pct"] for r in rel.values())],
    "attn_JS": [min(r["attn_JS_red_pct"] for r in rel.values()),
                max(r["attn_JS_red_pct"] for r in rel.values())],
    "contact_out_drift": [min(r["contact_out_drift_red_pct"] for r in rel.values()),
                          max(r["contact_out_drift_red_pct"] for r in rel.values())],
}

# S7: (clean NW, clean W, noisy NW, noisy W, CI diff)
S7 = {
    "GR00T insert": (0.8785, 0.8931, 1.4027, 1.0835, (0.2877, 0.3805)),
    "GR00T wipe": (0.7757, 0.7801, 0.9709, 0.8996, (0.0393, 0.1198)),
    "GR00T USB": (0.7787, 0.7592, 0.9950, 0.9037, (0.0348, 0.1067)),
    "GR00T cabinet": (0.6370, 0.6425, 1.1156, 0.8748, (0.1232, 0.3845)),
    "pi05 insert": (0.8263, 0.8250, 0.8734, 0.8783, (-0.0157, 0.0035)),
    "pi05 wipe": (0.7212, 0.7277, 0.7359, 0.7499, (-0.0127, -0.0024)),
    "pi05 USB": (0.6094, 0.6164, 0.6735, 0.6606, (-0.00005, 0.04273)),
    "pi05 cabinet": (0.5428, 0.5352, 0.5500, 0.5380, (-0.0020, 0.0104)),
}
s7 = {}
for k, (cn, cw, nn, nw, ci) in S7.items():
    dn = nn - cn
    dw = nw - cw
    s7[k] = {
        "noisy_MAE_red_pct_recomputed": round(100 * (nn - nw) / nn, 2),
        "clean_MAE_WiTac_minus_NW_deg": round(cw - cn, 4),
        "dMAE_NW": round(dn, 4),
        "dMAE_W": round(dw, 4),
        "dMAE_diff_NW_minus_W": round(dn - dw, 4),
        "reported_CI_diff": ci,
        "diff_inside_CI": ci[0] <= dn - dw <= ci[1],
        "CI_excludes_zero": not (ci[0] <= 0 <= ci[1]),
    }
res["tableS7_noise"] = s7

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("wrote", OUT)


# ---- 검출력: 두 비율 차이의 최소 검출 가능 크기 (정규 근사, 양측 0.05, 검정력 0.8, 기준 비율 0.5 가정) ----
def mde_pp(n_per_group, p_base=0.5, z_a=1.959964, z_b=0.841621):
    lo, hi = 0.0, 0.99
    for _ in range(200):
        mid = (lo + hi) / 2
        p2 = min(0.999, p_base + mid)
        se = math.sqrt(p_base * (1 - p_base) / n_per_group + p2 * (1 - p2) / n_per_group)
        if mid / se >= z_a + z_b:
            hi = mid
        else:
            lo = mid
    return round(100 * hi, 1)


res["power_mde_pp_80pct"] = {
    "per_task_n20_vs_20": mde_pp(20),
    "pooled_n80_vs_80": mde_pp(80),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("power added")
