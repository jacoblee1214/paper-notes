"""Being-H0 (arXiv:2507.15597) 논문 표 수치의 재계산, 검정, 본문 대조.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (수 초, numpy·scipy)
입력은 모두 논문 본문·표에 적힌 값이다(PDF 6.2–6.6절, 표 1–7). 원자료(샘플 단위)는 공개되지 않았다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


# ---------------------------------------------------------------- 공통 함수
def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    """Newcombe 점수 구간 (두 비율의 차이)."""
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    """두 독립 비율의 차이를 잡는 한 집단당 표본 수(정규 근사, 양측)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


# ---------------------------------------------------------------- A. 실로봇 성공률 (표 7)
TASKS = ["Pick-Place-Toy Seen", "Pick-Place-Toy Unseen", "Pick-Place-Toy Clutter",
         "Close-Toolbox", "Close-Lid", "Pour-Cup", "Unfold-Clothes"]
N_TRIAL = 20  # 본문 6.1.3: 과제마다 20회
RATES = {  # 표 7 값 (성공률). 20회이므로 정수로 환산한다.
    "GR00T N1.5": [0.75, 0.40, 0.50, 0.80, 0.50, 0.90, 0.60],
    "InternVL3": [0.55, 0.55, 0.50, 0.50, 0.25, 0.55, 0.45],
    "Being-H0": [0.75, 0.65, 0.60, 0.85, 0.60, 1.00, 0.75],
}
COUNTS = {m: [round(r * N_TRIAL) for r in v] for m, v in RATES.items()}
RES["A_table7_counts"] = COUNTS
RES["A_totals"] = {m: [sum(c), 7 * N_TRIAL] for m, c in COUNTS.items()}

rows = []
for comp_name in ["InternVL3", "GR00T N1.5"]:
    pv = []
    per = []
    for t, task in enumerate(TASKS):
        k1, k2 = COUNTS["Being-H0"][t], COUNTS[comp_name][t]
        _, p_f = stats.fisher_exact([[k1, N_TRIAL - k1], [k2, N_TRIAL - k2]])
        d, lo, hi = newcombe_diff(k1, N_TRIAL, k2, N_TRIAL)
        pv.append(p_f)
        per.append({"task": task, "being_h0": f"{k1}/20", "other": f"{k2}/20",
                    "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                    "fisher_p": round(p_f, 4)})
    adj = holm(pv)
    for r, a in zip(per, adj):
        r["holm_p"] = round(a, 4)
    # 합산 (과제를 독립 시행으로 취급한 근사: 과제 간 묶음 효과는 무시)
    k1, k2 = sum(COUNTS["Being-H0"]), sum(COUNTS[comp_name])
    N = 7 * N_TRIAL
    _, p_pool = stats.fisher_exact([[k1, N - k1], [k2, N - k2]])
    d, lo, hi = newcombe_diff(k1, N, k2, N)
    rows.append({"vs": comp_name, "per_task": per,
                 "pooled": {"being_h0": f"{k1}/{N}", "other": f"{k2}/{N}", "diff_pp": round(100 * d, 1),
                            "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)], "fisher_p": float(f"{p_pool:.2e}")}})
RES["A_real_robot_tests"] = rows

# 본문 "Close-Lid 25%에서 기준선 0%, Being-H0 15%" (그림 13, 값은 그림에서만 읽힘): 이항 확인만
RES["A_power"] = {
    "n_per_arm_for_0.25_gap_at_0.5_vs_0.75": n_per_arm(0.5, 0.75),
    "n_per_arm_for_0.10_gap_at_0.6_vs_0.7": n_per_arm(0.6, 0.7),
    "n_per_arm_for_0.15_gap_at_0.55_vs_0.70": n_per_arm(0.55, 0.70),
    "note": "정규 근사, 양측 alpha 0.05, 검정력 0.8. 과제당 20회로는 0.25 차이도 확실히 잡기 어렵다.",
}

# 시행 수 20회일 때 Wilson 95% 구간의 폭 (성공률 0.5 근처)
RES["A_wilson_20_at_half"] = [round(x, 3) for x in wilson(10, 20)]

# ---------------------------------------------------------------- B. UniHand 표 1 합계 검산
# (데이터셋, #Inst(M), #Seq(K), #Frames(M), #Hours)  표 1 행
UNIHAND = [
    ("ARCTIC", 17.9, 0.3, 0.245, 2.3),
    ("FPHA", 0.798, 1.2, 0.105, 1.0),
    ("HoloAssist", 8.0, 2.2, 17.1, 166),
    ("H2O", 3.7, 0.9, 0.115, 1.1),
    ("HOI4D", 21.2, 3.0, 0.825, 7.6),
    ("HOT3D", 8.7, 2.8, 0.420, 3.9),
    ("OAKINK2", 18.5, 2.8, 0.695, 6.5),
    ("TACO", 11.5, 2.2, 0.340, 3.2),
    ("DexYCB", 3.6, 5.6, 0.410, 3.8),
    ("Taste-Rob", 1.9, 85, 14, 130),
    ("EgoDex", 70.6, 338, 89.6, 829.4),
]
tot_inst = sum(r[1] for r in UNIHAND)
tot_seq = sum(r[2] for r in UNIHAND)
tot_frames = sum(r[3] for r in UNIHAND)
tot_hours = sum(r[4] for r in UNIHAND)
RES["B_unihand"] = {
    "sum_inst_M": round(tot_inst, 2), "table_inst_M": 166.5,
    "sum_seq_K": round(tot_seq, 1), "table_seq_K": 444.1,
    "sum_frames_M": round(tot_frames, 1), "table_frames_M": 130,
    "sum_hours": round(tot_hours, 1), "table_hours": 1155,
    "frame_rate_check_fps": {r[0]: round(r[3] * 1e6 / (r[4] * 3600), 1) for r in UNIHAND if r[4] > 0},
    "egodex_share_of_inst_pct": round(100 * 70.6 / tot_inst, 1),
    "egodex_share_of_hours_pct": round(100 * 829.4 / tot_hours, 1),
    "note": "프레임 합계는 표의 130M과 약 5% 어긋난다. 시간 합계와 시퀀스 합계는 맞는다.",
}

# ---------------------------------------------------------------- C. 생성 표 3, 6 (모델 규모 단조성)
# 표 3 시각 기반 생성: (지표, 1B, 8B, 14B) head/tail
T3_VG = {
    "MPJPE head": (9.71, 7.20, 6.87), "MPJPE tail": (17.21, 9.02, 8.11),
    "MWTE head": (8.25, 5.69, 5.19), "MWTE tail": (12.04, 8.11, 7.41),
    "PA-MPJPE head": (1.50, 1.09, 1.03), "PA-MPJPE tail": (1.55, 1.32, 1.20),
    "M2T R@3 head": (12.1, 15.9, 17.2), "M2T R@3 tail": (15.3, 18.7, 20.5),
    "FID head": (12.2, 11.5, 10.3), "FID tail": (13.1, 13.4, 11.8),
}
T3_CTX = {
    "MPJPE head": (8.73, 6.67, 6.21), "MPJPE tail": (11.34, 7.98, 7.33),
    "MWTE head": (7.88, 5.03, 4.89), "MWTE tail": (10.67, 6.93, 6.52),
    "PA-MPJPE head": (1.17, 0.90, 0.92), "PA-MPJPE tail": (1.38, 1.03, 1.04),
    "M2T R@3 head": (15.4, 19.7, 20.1), "M2T R@3 tail": (20.6, 21.4, 23.5),
    "FID head": (14.7, 10.1, 9.8), "FID tail": (15.6, 11.7, 10.1),
}


def monotone_report(tab):
    out = {}
    for k, (b1, b8, b14) in tab.items():
        lower_better = not k.startswith("M2T")
        if lower_better:
            ok = b1 >= b8 >= b14
        else:
            ok = b1 <= b8 <= b14
        out[k] = {"1B_8B_14B": [b1, b8, b14], "monotone_in_scale": ok}
    return out


RES["C_table3_scale_monotone_vg"] = monotone_report(T3_VG)
RES["C_table3_scale_monotone_ctx"] = monotone_report(T3_CTX)

# 표 6: 데이터·토크나이저 절제 (Being-H0-8B 기준)
BASE = {"MPJPE head": 7.20, "MPJPE tail": 9.02, "MWTE head": 5.69, "MWTE tail": 8.11,
        "PA head": 1.09, "PA tail": 1.32, "M2T head": 15.9, "M2T tail": 18.7, "FID head": 11.5, "FID tail": 13.4}
T6 = {
    "MANO-D114 + 4-Groups": {"MPJPE head": 8.31, "MPJPE tail": 10.35, "MWTE head": 6.52, "MWTE tail": 9.11,
                             "PA head": 1.14, "PA tail": 1.35, "M2T head": 13.1, "M2T tail": 14.7, "FID head": 13.7, "FID tail": 15.6},
    "MANO-D162 + 4-Groups": {"MPJPE head": 7.98, "MPJPE tail": 9.71, "MWTE head": 5.58, "MWTE tail": 8.98,
                             "PA head": 1.09, "PA tail": 1.38, "M2T head": 15.4, "M2T tail": 17.1, "FID head": 11.9, "FID tail": 14.3},
    "MANO-D114 + Part-Level": {"MPJPE head": 7.74, "MPJPE tail": 9.92, "MWTE head": 6.11, "MWTE tail": 8.83,
                               "PA head": 1.16, "PA tail": 1.41, "M2T head": 12.3, "M2T tail": 16.1, "FID head": 13.1, "FID tail": 12.3},
    "w/o Translate": {"MPJPE head": 7.22, "MPJPE tail": 9.11, "MWTE head": 5.51, "MWTE tail": 8.12,
                      "PA head": 1.27, "PA tail": 1.46, "M2T head": 13.2, "M2T tail": 11.6, "FID head": 13.1, "FID tail": 15.7},
    "w/o Context": {"MPJPE head": 8.01, "MPJPE tail": 10.97, "MWTE head": 6.34, "MWTE tail": 9.03,
                    "PA head": 1.11, "PA tail": 1.52, "M2T head": 13.7, "M2T tail": 15.2, "FID head": 11.1, "FID tail": 14.9},
    "w/o Balance": {"MPJPE head": 8.54, "MPJPE tail": 12.13, "MWTE head": 7.74, "MWTE tail": 10.04,
                    "PA head": 1.24, "PA tail": 1.57, "M2T head": 11.3, "M2T tail": 10.3, "FID head": 15.8, "FID tail": 16.3},
}
# 방향 점검: 값이 기준보다 나빠진(지표별 방향 고려) 항목 수와, "모든 지표가 나빠진다"는 주장 검사
worse_counts = {}
for name, vals in T6.items():
    worse = []
    better = []
    for k, v in vals.items():
        b = BASE[k]
        is_m2t = k.startswith("M2T")
        if (v < b and is_m2t) or (v > b and not is_m2t):
            worse.append(k)
        elif v != b:
            better.append(k)
    worse_counts[name] = {"worse_than_base": len(worse), "better_or_equal_than_base": len(better),
                          "better_items": better}
RES["C_table6_vs_base"] = worse_counts
RES["C_table6_claims"] = {
    "w/o Context: '모든 지표가 떨어진다' (본문 6.4.2)": "FID head 11.1 < 기준 11.5 (더 낫다). 전 지표 하락은 아님.",
    "MANO-D162가 '일관되게 우수' (6.4.1)": "MANO-D114 + Part-Level은 FID tail 12.3 < 13.4 (기준보다 낫다). 전 항목 우위는 아님.",
}

# ---------------------------------------------------------------- D. 장기 생성 표 4 (소프트 모드, GT 혼합)
T4 = {  # (short head, short tail, long head, long tail) for MPJPE, visual-grounded
    "VG 1B": (8.97, 9.96, 9.12, 11.24), "VG 8B": (7.55, 8.45, 8.21, 9.98), "VG 14B": (7.43, 8.39, 7.98, 9.72),
    "CTX 1B": (8.44, 9.52, 9.01, 10.98), "CTX 8B": (7.67, 8.20, 8.23, 9.67), "CTX 14B": (7.39, 8.51, 8.01, 9.45),
}
deg = {k: {"head_increase": round(v[2] - v[0], 2), "tail_increase": round(v[3] - v[1], 2)} for k, v in T4.items()}
RES["D_table4_mpjpe_degradation"] = {
    "values": T4, "increase_short_to_long_cm": deg,
    "note": "소프트 모드는 매 블록에서 GT와 평균을 낸 뒤 재양자화한다(코드 alpha 0.5). 표 3(블록 모드, 1초)과 직접 비교할 수 없다.",
}

# ---------------------------------------------------------------- E. 표 2 (유효율, translation)
T2 = {"1B": (64.8, 12.5, 14.3), "8B": (99.8, 18.4, 19.7), "14B": (100.0, 19.0, 22.1)}
RES["E_table2"] = {
    "gt_T2M_head_tail": [33.5, 42.7],
    "best_model_share_of_gt_pct": {"head": round(100 * 19.0 / 33.5, 1), "tail": round(100 * 22.1 / 42.7, 1)},
    "monotone_T2M": all(T2[a][1] <= T2[b][1] and T2[a][2] <= T2[b][2] for a, b in [("1B", "8B"), ("8B", "14B")]),
    "note": "유효율은 시행 수가 보고되지 않아 신뢰구간을 계산할 수 없다.",
}

# ---------------------------------------------------------------- F. 표 5 단위 환산 (cm -> mm)
RES["F_table5_mm"] = {
    "default_D162_part_MPJPE_mm": round(0.573 * 10, 1), "default_D162_part_PAMPJPE_mm": round(0.129 * 10, 1),
    "note": "본문 '밀리미터 수준'은 MPJPE 약 5.7 mm, PA-MPJPE 약 1.3 mm로 맞다. 다만 같은 표에서 D114 part-level MPJPE 5.2 mm가 더 낮다.",
}

# ---------------------------------------------------------------- G. 주장 검사 (실로봇 문구)
RES["G_claim_checks"] = {
    "highest_in_all_tasks": "Being-H0가 모든 과제에서 가장 높다(6.6) 대 Seen에서 GR00T와 0.75로 동점.",
    "far_less_post_training_data": "6.6은 GR00T가 '훨씬 적은 후처리 데이터'로 졌다고 하지만 6.1.3은 모든 모델이 '동일한 후처리 데이터'를 받았다고 한다.",
}

with OUT.open("w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

if __name__ == "__main__":
    print("wrote", OUT)
    for r in rows:
        print(r["vs"], "pooled", r["pooled"])
        for p in r["per_task"]:
            print("  ", p["task"], p["being_h0"], p["other"], "p=", p["fisher_p"], "holm=", p["holm_p"])
    print("UniHand frame sum (M):", RES["B_unihand"]["sum_frames_M"], "table:", RES["B_unihand"]["table_frames_M"])
    print("power:", RES["A_power"])
