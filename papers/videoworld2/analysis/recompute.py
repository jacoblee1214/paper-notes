"""VideoWorld 2 (arXiv:2602.10102 v1) 논문 표 수치를 옮겨 적고, 재계산과 일관성 검사를 한다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json 과 표준 출력
입력은 논문 본문과 표에서 옮긴 값뿐이다(원자료 없음). 네트워크와 학습 없이 수 초에 끝난다.

가정(논문에 명시되지 않은 값):
- Video-CraftBench 시험 사례 수 n_case = 150 (테스트 영상 약 150개, 본문 4.1). 롤아웃은 사례당 3회(4.2).
  - 시나리오 A: 사례 150개를 독립 시행으로 본다(보수적, 같은 사례의 반복을 묶음).
  - 시나리오 B: 롤아웃 450회를 독립 시행으로 본다(상한, 과대 추정 가능).
- CALVIN 평가 사슬 수는 논문에 없다. Avg. Len.의 재계산에는 사슬 수가 필요 없다(단계별 비율의 합).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"

N_A = 150   # 시나리오 A: 사례 단위
N_B = 450   # 시나리오 B: 롤아웃 단위 (150 x 3)

# ---- Table 1: Video-CraftBench, 종이접기 순차 성공률 (7단계, %) ----
T1 = {
    "Cosmos AR 4B (Craft-text)": [68.4, 56.7, 11.5, 3.3, 0.0, 0.0, 0.0],
    "Cosmos DiT 2B (Craft-text)": [73.4, 63.3, 20.0, 6.7, 0.0, 0.0, 0.0],
    "Hunyuan-13B (Craft-text)": [76.9, 68.1, 27.5, 5.8, 0.0, 0.0, 0.0],
    "Wan 2.2 14B (Craft-text)": [81.2, 75.0, 30.4, 10.6, 0.0, 0.0, 0.0],
    "Moto (Craft)": [19.1, 11.7, 3.3, 0.0, 0.0, 0.0, 0.0],
    "AdaWorld (Craft)": [43.6, 39.8, 27.4, 10.8, 0.0, 0.0, 0.0],
    "VideoWorld (Craft)": [70.3, 66.7, 42.5, 21.3, 6.7, 0.0, 0.0],
    "VideoWorld 2 (Craft)": [97.2, 95.3, 90.0, 83.3, 81.4, 74.6, 68.8],
    "iVideoGPT (OpenX&Craft)": [23.1, 18.7, 13.3, 3.7, 0.0, 0.0, 0.0],
    "Moto (OpenX&Craft)": [43.1, 35.3, 30.7, 25.5, 18.3, 9.7, 0.0],
    "AdaWorld (OpenX&Craft)": [49.5, 41.6, 34.8, 30.7, 22.3, 19.8, 13.0],
    "CoLA (OpenX&Craft)": [83.5, 74.4, 69.1, 64.8, 52.3, 49.8, 40.2],
    "VideoWorld (OpenX&Craft)": [91.7, 75.0, 68.2, 63.1, 51.7, 48.2, 31.9],
    "VideoWorld 2 (OpenX&Craft)": [98.2, 96.4, 90.1, 86.7, 83.3, 81.7, 72.3],
}
# 블록 성공률 (Human, Tower, Horse) 과 시각 품질 (SSIM, LPIPS)
T1_BLOCK = {
    "Wan 2.2 14B (Craft-text)": (39.7, 42.6, 34.1, 0.719, 0.237),
    "Hunyuan-13B (Craft-text)": (30.9, 38.4, 31.5, 0.703, 0.255),
    "VideoWorld (Craft)": (23.8, 33.9, 27.8, 0.680, 0.351),
    "VideoWorld 2 (Craft)": (70.0, 81.5, 80.9, 0.770, 0.205),
    "VideoWorld (OpenX&Craft)": (47.3, 52.7, 49.8, 0.601, 0.389),
    "VideoWorld 2 (OpenX&Craft)": (74.0, 83.0, 85.8, 0.774, 0.193),
}

# ---- Table 2: CALVIN 연속 5과제 순차 성공률 (%) 과 Avg. Len. 보고값 ----
T2 = {
    "1 Transformer Oracle (full, 22k)": ([80.9, 55.6, 44.5, 31.3, 24.6], 2.36),
    "2 Transformer Oracle (10%)": ([50.5, 35.4, 20.1, 5.20, 0.0], 1.11),
    "3 LAPA latent pretrain (10%)": ([74.4, 45.8, 25.2, 15.3, 2.30], 1.49),
    "4 VideoWorld 2 latent pretrain (10%)": ([75.8, 47.9, 31.8, 20.4, 9.70], 1.87),
    "5 Video pretrain OpenX (oracle)": ([85.9, 60.4, 46.0, 30.7, 23.0], 2.46),
    "6 LAPA OpenX": ([84.0, 58.8, 46.2, 35.4, 27.0], 2.51),
    "7 VideoWorld 2 OpenX": ([88.5, 64.6, 55.8, 47.5, 30.9], 2.88),
}

# ---- Table 3: 절제 (Video-CraftBench 종이접기 / 블록 성공률 %, LPIPS) ----
T3A = {  # (VDM, Stop-Grad, Ctrl-Net): paper, block, LPIPS
    "row1 no VDM": (0.0, 28.5, 0.312),
    "row2 VDM": (30.3, 45.2, 0.297),
    "row3 VDM+stopgrad": (47.3, 54.7, 0.275),
    "row4 VDM+ctrlnet": (51.1, 52.0, 0.213),
    "row5 VDM+stopgrad+ctrlnet (default)": (68.8, 77.5, 0.205),
}
T3C = {  # (projection, interplay): paper, block
    "MLP cross": (52.0, 61.3),
    "+self cross": (52.3, 61.8),
    "MLP causal cross": (69.8, 78.6),
    "+self causal cross (default?)": (72.3, 80.9),
}
T3B_DEFAULT_NOTE = "Table 3a row5, 3b N=4, 3d 1000, 3e T=93, 3f full: 68.8 / 77.5"

# ---- 본문과 부록에서 옮긴 값 ----
TASK_HOURS = {"paper_folding": 5.2, "block": 1.8}
PAPER_FOLDING_SHARE_REPORTED = 55.3  # 부록 B 본문 "55.3% (5.2 hours)"
TOTAL_HOURS_TEXT = 7.0               # 본문 4.1 "~7 hours"


def wilson(k_rate_pct, n):
    p = k_rate_pct / 100.0
    z = 1.959963984540054
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half) * 100, 6), round(min(1.0, centre + half) * 100, 6)


def two_prop(p1_pct, p2_pct, n):
    """두 독립 비율의 차이, 합동 z 검정 양측 p, 그리고 Fisher 양측 p (같은 n)."""
    p1, p2 = p1_pct / 100, p2_pct / 100
    k1, k2 = round(p1 * n), round(p2 * n)
    pooled = (k1 + k2) / (2 * n)
    se = math.sqrt(max(pooled * (1 - pooled) * 2 / n, 1e-12))
    z = (p1 - p2) / se
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))
    _, p_f = stats.fisher_exact([[k1, n - k1], [k2, n - k2]])
    return {"diff_pts": round((p1 - p2) * 100, 1), "p_z": float(p_z), "p_fisher": float(p_f)}


def main():
    res = {"assumptions": {"n_case": N_A, "n_rollout": N_B, "note": "논문에 시행 수 없음. 가정 A/B 병기"}}

    # V1: Table 2 Avg. Len. 과 단계 합 비교
    v1 = {}
    for name, (steps, reported) in T2.items():
        s = round(sum(steps) / 100, 4)
        v1[name] = {"sum_of_step_rates": s, "reported": reported, "gap": round(reported - s, 4)}
    res["V1_calvin_avglen"] = v1

    # V2: Table 1 단조성 (단계가 진행될수록 성공률은 늘지 않아야 함)
    res["V2_monotone_violations"] = [k for k, v in T1.items()
                                     if any(v[i + 1] > v[i] + 1e-9 for i in range(len(v) - 1))]

    # V3: Table 3c 마지막 행과 기본 설정 (3a row5) 의 불일치
    res["V3_table3c_default_mismatch"] = {
        "table3c_last_row": T3C["+self causal cross (default?)"],
        "table3a_row5_default": T3A["row5 VDM+stopgrad+ctrlnet (default)"][:2],
        "note": T3B_DEFAULT_NOTE,
    }

    # V4: 본문 "row 3 vs row 2 약 20%", "rows 4·5 LPIPS 약 0.9", "row 5 vs row 3 약 20%"
    r = T3A
    res["V4_ablation_text_checks"] = {
        "row3_minus_row2_paper_pts": round(r["row3 VDM+stopgrad"][0] - r["row2 VDM"][0], 1),
        "row3_minus_row2_block_pts": round(r["row3 VDM+stopgrad"][1] - r["row2 VDM"][1], 1),
        "row5_minus_row3_paper_pts": round(r["row5 VDM+stopgrad+ctrlnet (default)"][0] - r["row3 VDM+stopgrad"][0], 1),
        "lpips_row3_to_row5": round(r["row5 VDM+stopgrad+ctrlnet (default)"][2] - r["row3 VDM+stopgrad"][2], 3),
        "lpips_row2_to_row4": round(r["row2 VDM"][2] - r["row4 VDM+ctrlnet"][2], 3),
        "text_claim_lpips": "~0.9 (본문 5.5)",
    }

    # V5: 본문 "block 최대 약 38%" 대 Table 1 사전학습 모델 블록 최대
    pre = [v for k, v in T1_BLOCK.items() if "Craft-text" in k]
    res["V5_pretrained_block_max"] = {"text_claim": 38.0, "table_max": max(max(v[:3]) for v in pre)}

    # V6: 본문 "step 4 (column 7) ≤10.6%" 의 열 번호 (Wan 2.2 14B 4단계 값)
    res["V6_wan_step4"] = {"table_value_step4": T1["Wan 2.2 14B (Craft-text)"][3],
                           "text_says": "column 7 (step 4)"}

    # V7: 부록 B "paper folding 55.3% (5.2 h)" 와 시간 합
    share_calc = round(100 * TASK_HOURS["paper_folding"] / (TASK_HOURS["paper_folding"] + TASK_HOURS["block"]), 1)
    res["V7_duration_share"] = {"reported_pct": PAPER_FOLDING_SHARE_REPORTED,
                                "calc_from_hours_pct": share_calc,
                                "total_hours_calc": TASK_HOURS["paper_folding"] + TASK_HOURS["block"],
                                "total_hours_text": TOTAL_HOURS_TEXT}

    # V8: 초록 "up to 70% improvement": 단계별 절대 차이 (Craft 단독, VideoWorld 2 대 최고 기준선)
    v2 = T1["VideoWorld 2 (Craft)"]
    base_best = [max(T1[k][i] for k in T1 if "(Craft" in k and "VideoWorld 2" not in k) for i in range(7)]
    diffs = [round(v2[i] - base_best[i], 1) for i in range(7)]
    res["V8_abstract_70pct"] = {"best_baseline_per_step": base_best, "abs_diff_pts_per_step": diffs,
                                "max_abs_diff_pts": max(diffs)}

    # C-checks: Wilson CI 와 두 비율 검정 (단계별, 두 시나리오)
    cmp_pairs = [("VideoWorld 2 (Craft)", "VideoWorld (Craft)"),
                 ("VideoWorld 2 (Craft)", "Wan 2.2 14B (Craft-text)"),
                 ("VideoWorld 2 (OpenX&Craft)", "CoLA (OpenX&Craft)"),
                 ("VideoWorld 2 (OpenX&Craft)", "VideoWorld (OpenX&Craft)")]
    ci = {}
    for a, b in cmp_pairs:
        rows = {}
        for i in range(7):
            for label, n in (("A_case150", N_A), ("B_rollout450", N_B)):
                pa, pb = T1[a][i], T1[b][i]
                rows[f"step{i+1}_{label}"] = {
                    "ci_a": [round(x, 1) for x in wilson(pa, n)],
                    "ci_b": [round(x, 1) for x in wilson(pb, n)],
                    **two_prop(pa, pb, n),
                }
        ci[f"{a} vs {b}"] = rows
    res["C_stepwise_tests"] = ci

    # 블록 과제 (Human, Tower, Horse). 블록 시행 수는 논문에 없으므로 n_case 가정만 사용
    blk = {}
    for name, v in T1_BLOCK.items():
        blk[name] = {"human_ci_A": [round(x, 1) for x in wilson(v[0], N_A)],
                     "tower_ci_A": [round(x, 1) for x in wilson(v[1], N_A)],
                     "horse_ci_A": [round(x, 1) for x in wilson(v[2], N_A)]}
    res["C_block_wilson_A"] = blk
    res["C_block_vs_wan_tower"] = two_prop(81.5, 42.6, N_A)
    res["C_block_vs_wan_tower_B"] = two_prop(81.5, 42.6, N_B)

    # Table 2 는 CALVIN 시도 수가 없어 Avg. Len. 격차만 비교. 행 4 와 행 2 의 차이
    res["C_calvin_vs_lapa_pretrain_gap"] = {
        "row4_minus_row3_reported": round(T2["4 VideoWorld 2 latent pretrain (10%)"][1] - T2["3 LAPA latent pretrain (10%)"][1], 2),
        "row4_minus_row3_from_sums": round(sum(T2["4 VideoWorld 2 latent pretrain (10%)"][0]) / 100
                                           - sum(T2["3 LAPA latent pretrain (10%)"][0]) / 100, 3),
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 요약 출력 ----
    print("[V1] CALVIN Avg. Len. 보고값 대 단계 합")
    for k, v in v1.items():
        print(f"  {k:40s} sum={v['sum_of_step_rates']:.3f} reported={v['reported']:.2f} gap={v['gap']:+.3f}")
    print("[V2] 단조성 위반:", res["V2_monotone_violations"] or "없음")
    print("[V3]", res["V3_table3c_default_mismatch"]["table3c_last_row"], "대", res["V3_table3c_default_mismatch"]["table3a_row5_default"])
    print("[V4]", res["V4_ablation_text_checks"])
    print("[V5]", res["V5_pretrained_block_max"])
    print("[V7]", res["V7_duration_share"])
    print("[V8] 단계별 절대 차이:", diffs, "최대", max(diffs))
    print("[C] VideoWorld 2 vs VideoWorld (Craft), 단계별 p (z / Fisher), 시나리오 A(150) | B(450)")
    key = "VideoWorld 2 (Craft) vs VideoWorld (Craft)"
    for i in range(7):
        a = ci[key][f"step{i+1}_A_case150"]
        b = ci[key][f"step{i+1}_B_rollout450"]
        print(f"  step{i+1}: diff {a['diff_pts']:+.1f} | A p={a['p_fisher']:.2g} | B p={b['p_fisher']:.2g}  CI_A(v2)={a['ci_a']}")
    print("[C] 블록 Tower 81.5 대 Wan 42.6:", res["C_block_vs_wan_tower"], res["C_block_vs_wan_tower_B"]["p_fisher"])
    print("[C] CALVIN 행 4 대 행 3 차이:", res["C_calvin_vs_lapa_pretrain_gap"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
