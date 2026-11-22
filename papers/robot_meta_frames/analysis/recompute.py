"""Robot Meta-Frames (MetaWAM, arXiv:2611.14365 v1) 논문 표 수치를 다시 계산하고 검정과 내부 일치를 본다.

입력: 논문 본문과 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
  - 표 1, 5, 6, 7, 8과 본문 수치: paper/paper.pdf (arXiv v1)
  - 표 3의 기준선 값 검사: 원자료 Table 4 of arXiv:2603.22078 (Do World Action Models Generalize Better than VLAs?)
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 2(RoboCasa), 표 8(LIBERO)의 대조군 시도 수는 논문에 없다. 두 방법 모두 같은 수라고 가정한다(표 2는 3600회, 표 8은 6000회).
  대조군 값은 반올림된 값이므로 차이에도 ±0.05 pp 정도의 오차가 있다.
- 표 2는 시도가 독립이라고 본다. 과제와 시드에 묶인 분산은 논문에 없어 반영하지 못했다. 따라서 p 값은 낙관적이다.
- 표 3의 Total은 일곱 교란 범주의 단순 평균이 아니다(아래 점검). 가중치와 범주별 시도 수가 없어 신뢰구간은 못 낸다.
- 실로봇 표 4의 분모는 과제당 20회다(논문 캡션). 과제별 값에서 성공 수를 역산해 일치를 확인했다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def mde_pp(n, p=0.5, alpha=0.05, power=0.8):
    """두 독립 표본(크기 n씩)에서 80% 검정력으로 검출할 수 있는 최소 차이 (pp)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return 100 * (za + zb) * math.sqrt(2 * p * (1 - p) / n)


# ---------------------------------------------------------------------------
# 1. 표 1: VDM-prior 호환성 (Cosmos-Predict2 동결 디노이저, 상대 재구성 오차, 낮을수록 좋음)
# ---------------------------------------------------------------------------
SIGMAS = [0.01, 0.05, 0.2, 1.0, 5.0]
T1A = {  # 단일 신호 (표 1a)
    "Spatial Duplication": [0.0141, 0.0672, 0.2335, 0.6459, 0.8953],
    "VAE Encoding": [0.0147, 0.0699, 0.2388, 0.6379, 0.8926],
    "DCT Encoding": [0.0133, 0.0633, 0.2199, 0.5922, 0.8646],
}
T1B = {  # 다중 신호 (표 1b)
    "Sep. Spatial Dup.": [0.0142, 0.0677, 0.2336, 0.6178, 0.8677],
    "Meta-Quadrants": [0.0147, 0.0705, 0.2471, 0.6730, 0.9291],
    "Meta-Compact": [0.0132, 0.0622, 0.2160, 0.6234, 0.9392],
    "Unified Meta-Frame": [0.0126, 0.0598, 0.2061, 0.5613, 0.8159],
}
t1 = {}
for name, tab in (("a", T1A), ("b", T1B)):
    best = []
    for j, s in enumerate(SIGMAS):
        col = {k: v[j] for k, v in tab.items()}
        best.append(min(col, key=col.get))
    t1[name] = best
RES["t1_best_per_sigma"] = {"a": t1["a"], "b": t1["b"]}
RES["t1_unified_lowest_in_b_all_sigma"] = all(b == "Unified Meta-Frame" for b in t1["b"])
RES["t1_dct_lowest_in_a_sigma_le_5"] = all(b == "DCT Encoding" for b in t1["a"])

# ---------------------------------------------------------------------------
# 2. 표 6, 7: Wan 백본 (A.2). 본문 "DCT가 일관되게 낮다"의 점검.
# ---------------------------------------------------------------------------
WAN_SIG = [0.2, 0.4, 0.6, 0.8, 0.9]
W21A = {  # Table 6(a), Wan2.1-I2V-14B, 단일 신호 (PDF 쪽 13에서 굵은 글씨로 확인)
    "Spatial Duplication": [0.8858, 0.9883, 1.0092, 1.0434, 1.0951],
    "VAE Encoding": [0.7534, 0.8993, 0.9891, 1.0478, 1.0989],
    "DCT Encoding": [0.7701, 0.8258, 0.9053, 0.9943, 1.0902],
}
W21B = {  # Table 6(b)
    "Sep. Spatial Dup.": [0.9029, 0.9944, 1.0194, 1.0519, 1.1004],
    "Meta-Quadrants": [0.9182, 1.0005, 1.0271, 1.0580, 1.1047],
    "Meta-Compact": [0.8467, 0.9990, 1.0226, 1.0540, 1.1024],
    "Unified Meta-Frame": [0.7144, 0.8710, 0.9505, 1.0226, 1.0988],
}
W22A = {  # Table 7(a), Wan2.2-TI2V-5B (PDF 쪽 14)
    "Spatial Duplication": [0.9777, 1.0706, 1.1058, 1.1566, 1.2187],
    "VAE Encoding": [0.7537, 1.0241, 1.0659, 1.1025, 1.1445],
    "DCT Encoding": [0.7059, 0.8662, 0.9572, 1.0419, 1.0992],
}
W22B = {  # Table 7(b)
    "Sep. Spatial Dup.": [0.9942, 1.0768, 1.1204, 1.1748, 1.2603],
    "Meta-Quadrants": [1.0030, 1.0814, 1.1188, 1.1683, 1.2482],
    "Meta-Compact": [1.0461, 1.1625, 1.2384, 1.3167, 1.4663],
    "Unified Meta-Frame": [0.7647, 0.9213, 1.0011, 1.0683, 1.1307],
}


def wan_check(tab_a, tab_b):
    dct_vs_vae = [tab_a["DCT Encoding"][j] < tab_a["VAE Encoding"][j] for j in range(5)]
    dct_vs_sp = [tab_a["DCT Encoding"][j] < tab_a["Spatial Duplication"][j] for j in range(5)]
    uni_vs_all = [all(tab_b["Unified Meta-Frame"][j] < v[j] for k, v in tab_b.items() if k != "Unified Meta-Frame")
                  for j in range(5)]
    return {"dct_lower_than_vae_per_sigma": dct_vs_vae,
            "dct_lower_than_spatial_per_sigma": dct_vs_sp,
            "unified_lowest_per_sigma": uni_vs_all}


RES["wan21_check"] = wan_check(W21A, W21B)
RES["wan22_check"] = wan_check(W22A, W22B)

# ---------------------------------------------------------------------------
# 3. 표 2: RoboCasa 평균 성공률 (24 과제 × 50회 × 3 시드 = 3600회, 두 방법 모두 3600회 가정)
# ---------------------------------------------------------------------------
N_ROBOCASA = 24 * 50 * 3
mw_rc = round(0.689 * N_ROBOCASA)
cp_rc = round(0.671 * N_ROBOCASA)
z, p = two_prop_z(mw_rc, N_ROBOCASA, cp_rc, N_ROBOCASA)
RES["t2_robocasa"] = {
    "n_per_method_assumed": N_ROBOCASA,
    "metawam_k": mw_rc, "cosmos_k": cp_rc,
    "diff_pp": round(100 * (0.689 - 0.671), 2),
    "wilson_metawam": [round(100 * x, 2) for x in wilson(mw_rc, N_ROBOCASA)],
    "wilson_cosmos": [round(100 * x, 2) for x in wilson(cp_rc, N_ROBOCASA)],
    "z": round(z, 3), "p_two_sided_independent": round(p, 4),
    "mde_pp_80pct": round(mde_pp(N_ROBOCASA, p=0.68), 2),
    "note": "시도가 독립이라는 가정. 시드 3개와 과제 24개에 묶인 분산은 반영하지 않았다.",
}

# ---------------------------------------------------------------------------
# 4. 표 3: LIBERO-Plus. 행 값을 원자료(arXiv:2603.22078 Table 4)와 대조하고 Total을 점검한다.
# ---------------------------------------------------------------------------
COLS = ["Original", "Camera", "Robot", "Lang", "Light", "BG", "Noise", "Layout", "Total"]
# 본 논문 표 3 (그대로 옮김)
PAPER_T3 = {
    "π0": [94.2, 13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9, 53.6],
    "π0-FAST": [85.5, 65.1, 21.6, 61.0, 73.2, 73.2, 74.4, 68.8, 61.6],
    "OpenVLA-OFT": [97.6, 55.6, 21.7, 81.0, 92.7, 91.0, 78.6, 68.7, 67.9],
    "UniVLA": [95.2, 1.8, 46.2, 69.6, 69.0, 81.0, 21.2, 31.9, 42.9],
    "RIPT-VLA": [97.5, 55.2, 31.2, 77.6, 88.4, 91.6, 73.5, 74.2, 68.4],
    "X-VLA": [98.1, 23.4, 89.7, 75.7, 88.2, 96.0, 62.7, 71.8, 71.4],
    "HoloBrain0-GD": [96.7, 65.5, 58.2, 78.7, 88.1, 90.3, 66.9, 79.5, 74.0],
    "ABot-M0": [98.6, 60.4, 67.9, 86.4, 96.2, 91.6, 86.4, 82.6, 80.5],
    "VLA-JEPA": [97.2, 64.2, 67.7, 88.1, 91.8, 93.4, 65.8, 83.9, 77.9],
    "GE-Act": [94.4, 60.7, 77.0, 77.4, 95.8, 86.0, 90.9, 80.2, 80.3],
    "Cosmos Policy": [98.5, 75.8, 63.3, 81.7, 96.5, 88.9, 92.7, 82.2, 82.2],
    "Fast-WAM": [97.6, 16.4, 44.5, 68.9, 78.2, 53.7, 37.7, 60.7, 51.5],
    "MetaWAM": [98.1, 82.9, 59.9, 89.5, 97.8, 72.6, 94.8, 86.7, 83.4],
}
# 원자료 arXiv:2603.22078 Table 4 (PDF 텍스트에서 옮김). 본 논문 표 3에 없는 두 행 포함.
SRC_T3 = {
    "π0 (rerun)": [91.3, 61.0, 40.8, 63.5, 89.3, 84.1, 80.1, 76.4, 69.4],
    "π0.5": [96.9, 75.4, 77.5, 85.6, 96.9, 94.6, 89.7, 85.7, 85.7],
    "π0": [94.2, 13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9, 53.6],
    "π0-FAST": [85.5, 65.1, 21.6, 61.0, 73.2, 73.2, 74.4, 68.8, 61.6],
    "Openvla-OFT_m": [97.6, 55.6, 21.7, 81.0, 92.7, 91.0, 78.6, 68.7, 67.9],
    "UniVLA": [95.2, 1.8, 46.2, 69.6, 69.0, 81.0, 21.2, 31.9, 42.9],
    "RIPT-VLA": [97.5, 55.2, 31.2, 77.6, 88.4, 91.6, 73.5, 74.2, 68.4],
    "X-VLA": [98.1, 23.4, 89.7, 75.7, 88.2, 96.0, 62.7, 71.8, 71.4],
    "HoloBrain0-GD": [96.7, 65.5, 58.2, 78.7, 88.1, 90.3, 66.9, 79.5, 74.0],
    "ABot-M0": [98.6, 60.4, 67.9, 86.4, 96.2, 91.6, 86.4, 82.6, 80.5],
    "VLA-JEPA": [97.2, 64.2, 67.7, 88.1, 91.8, 93.4, 65.8, 83.9, 77.9],
    "GE-Act": [94.4, 60.7, 77.0, 77.4, 95.8, 86.0, 90.9, 80.2, 80.3],
    "Cosmos Policy": [98.5, 75.8, 63.3, 81.7, 96.5, 88.9, 92.7, 82.2, 82.2],
    "Fast-WAM": [97.6, 16.4, 44.5, 68.9, 78.2, 53.7, 37.7, 60.7, 51.5],
}
ALIAS = {"OpenVLA-OFT": "Openvla-OFT_m"}  # 본 논문 표기 대 원자료 표기
mismatch = {}
compared = []
for name, row in PAPER_T3.items():
    src_name = ALIAS.get(name, name)
    if src_name in SRC_T3 and name != "MetaWAM":
        compared.append(name)
        diffs = [round(a - b, 2) for a, b in zip(row, SRC_T3[src_name]) if abs(a - b) > 1e-9]
        if diffs:
            mismatch[name] = diffs
RES["t3_baseline_rows_compared"] = len(compared)
RES["t3_row_mismatch_vs_source"] = mismatch
RES["t3_rows_in_source_not_in_paper_table"] = [k for k in SRC_T3 if k not in PAPER_T3 and k not in ALIAS.values()]

t3_rows = {}
for name, row in {**PAPER_T3, "π0 (rerun)": SRC_T3["π0 (rerun)"], "π0.5": SRC_T3["π0.5"]}.items():
    cats = row[1:8]
    t3_rows[name] = {"total_reported": row[8], "unweighted_mean_7": round(float(np.mean(cats)), 2)}
RES["t3_total_vs_unweighted_mean"] = t3_rows
RES["t3_margin_vs_cosmos_reported_pp"] = round(83.4 - 82.2, 2)
RES["t3_margin_vs_cosmos_text_pp"] = round(83.44 - 82.2, 2)
RES["t3_margin_vs_cosmos_unweighted_pp"] = round(t3_rows["MetaWAM"]["unweighted_mean_7"] -
                                                 t3_rows["Cosmos Policy"]["unweighted_mean_7"], 2)
RES["t3_pi05_total_vs_metawam_pp"] = round(85.7 - 83.4, 2)

# ---------------------------------------------------------------------------
# 5. 표 4: 실로봇 (과제당 20회, 3과제). 과제별 성공 수 역산과 합계 검정.
# ---------------------------------------------------------------------------
N_TASK = 20
T4 = {  # (ID 과제 3개 %, OOD 과제 3개 %), 캡션과 같은 순서: Sweep, Place Cup, Build Block
    "π0": ([45, 50, 55], [35, 35, 40], 50.0, 36.7),
    "π0.5": ([70, 70, 75], [45, 45, 50], 71.7, 46.7),
    "Cosmos Policy": ([75, 80, 80], [60, 60, 70], 78.3, 63.3),
    "MetaWAM": ([80, 85, 90], [65, 65, 75], 85.0, 68.3),
}
t4 = {}
for name, (idv, oodv, sr_id, sr_ood) in T4.items():
    kid = [round(x * N_TASK / 100) for x in idv]
    kood = [round(x * N_TASK / 100) for x in oodv]
    t4[name] = {
        "id_counts": kid, "id_total": sum(kid), "id_mean_check": round(100 * sum(kid) / 60, 2), "id_reported": sr_id,
        "ood_counts": kood, "ood_total": sum(kood), "ood_mean_check": round(100 * sum(kood) / 60, 2),
        "ood_reported": sr_ood,
    }
RES["t4_counts"] = t4
mw_id, cp_id = t4["MetaWAM"]["id_total"], t4["Cosmos Policy"]["id_total"]
mw_ood, cp_ood = t4["MetaWAM"]["ood_total"], t4["Cosmos Policy"]["ood_total"]
_, p_id_f = stats.fisher_exact([[mw_id, 60 - mw_id], [cp_id, 60 - cp_id]])
_, p_ood_f = stats.fisher_exact([[mw_ood, 60 - mw_ood], [cp_ood, 60 - cp_ood]])
_, p_pool_f = stats.fisher_exact([[mw_id + mw_ood, 120 - mw_id - mw_ood], [cp_id + cp_ood, 120 - cp_id - cp_ood]])
RES["t4_fisher"] = {
    "id_metawam_vs_cosmos": {"k": [mw_id, cp_id], "n": 60, "p_fisher_two_sided": round(p_id_f, 3)},
    "ood_metawam_vs_cosmos": {"k": [mw_ood, cp_ood], "n": 60, "p_fisher_two_sided": round(p_ood_f, 3)},
    "pooled_metawam_vs_cosmos": {"k": [mw_id + mw_ood, cp_id + cp_ood], "n": 120,
                                 "p_fisher_two_sided": round(p_pool_f, 3)},
    "mde_pp_80pct_n60": round(mde_pp(60, p=0.75), 1),
    "id_wilson_metawam_pct": [round(100 * x, 1) for x in wilson(mw_id, 60)],
    "id_wilson_cosmos_pct": [round(100 * x, 1) for x in wilson(cp_id, 60)],
    "id_pi05_vs_metawam_fisher_p": round(stats.fisher_exact([[43, 17], [mw_id, 60 - mw_id]])[1], 3),
}

# ---------------------------------------------------------------------------
# 6. 표 8: LIBERO 원 벤치마크 (4 스위트 × 500회 × 3 시드 = 6000회)
# ---------------------------------------------------------------------------
T8 = {
    "MetaWAM": [99.0, 99.8, 97.4, 96.0, 98.1],
    "Cosmos Policy": [98.1, 100.0, 98.2, 97.6, 98.5],
}
RES["t8_suite_mean_check"] = {k: round(float(np.mean(v[:4])), 3) for k, v in T8.items()}
N_LIBERO = 6000
mw8, cp8 = 0.981, 0.985
z8, p8 = two_prop_z(round(mw8 * N_LIBERO), N_LIBERO, round(cp8 * N_LIBERO), N_LIBERO)
RES["t8_libero_avg"] = {"diff_pp": round(100 * (mw8 - cp8), 2), "n_assumed": N_LIBERO,
                        "z": round(z8, 3), "p_two_sided_independent": round(p8, 3),
                        "mde_pp_80pct": round(mde_pp(N_LIBERO, p=0.98), 2)}

# ---------------------------------------------------------------------------
# 7. 표 5: 인코딩 절제. 지연, 프레임 수에 따른 어텐션 비율, 순위 일치.
# ---------------------------------------------------------------------------
lat_sep, lat_uni = 0.479, 0.291
RES["t5_latency_reduction_pct"] = round(100 * (1 - lat_uni / lat_sep), 1)
RES["t5_attention_reduction_pct_11_to_8"] = round(100 * (1 - (8 / 11) ** 2), 1)
RES["t5_attention_reduction_note"] = "지연을 T^2로 가정(토큰 수에 비례하는 전체 어텐션, 프레임당 토큰 동일)"
T5 = {  # SR, gain (↑), rel L2 (↓)
    "Separate": (67.1, 0.6361, 0.3657),
    "Meta-Quadrants": (65.9, 0.6032, 0.3988),
    "Meta-Compact": (68.1, 0.6563, 0.3345),
    "Unified": (68.9, 0.7146, 0.2928),
}
names = list(T5)
sr_rank = sorted(names, key=lambda k: -T5[k][0])
gain_rank = sorted(names, key=lambda k: -T5[k][1])
l2_rank = sorted(names, key=lambda k: T5[k][2])
RES["t5_rank_sr"] = sr_rank
RES["t5_rank_gain_same_as_sr"] = gain_rank == sr_rank
RES["t5_rank_l2_same_as_sr"] = l2_rank == sr_rank
RES["t5_n_layouts"] = len(names)

# ---------------------------------------------------------------------------
# 8. 본문 수치 일치: 앱스트랙, 실로봇 평균, 어텐션 1.40x 등은 값 자체가 있으므로 여기서는 계산하지 않는다.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    for k, v in RES.items():
        print(k, ":", v)
