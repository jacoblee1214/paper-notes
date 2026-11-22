"""MimicPlay (arXiv:2302.12422 v2, CoRL 2023) 논문 표 수치를 다시 계산하고 검정한다.

입력: 논문 본문과 표의 값을 아래에 그대로 옮겨 적었다(출처: paper/paper.txt의 표 1–4, 부록 C).
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 실로봇 표(표 1–3)의 과제당 시도 수는 논문에 없다. 셀 값이 모두 0.1 단위라 과제당 10회로 가정한다(V1).
  이 가정이 틀리면 모든 p와 구간이 달라진다. 확인하려면 원저자 데이터가 필요하다.
- 표 4(시뮬)의 ± 는 5개 시드에 대한 표준편차(SD)로 가정한다(본문은 SD인지 SE인지 밝히지 않음).
- 표 4 캡션은 "success rates %"라고 하지만 값은 0–1 소수다. 값을 그대로 쓴다.
- 표 4는 시드 5개 평균이다. 시드별 시도 100회 중 성공 수는 주어지지 않아 Fisher 검정은 시드를 무시한 근사다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_REAL = 10  # V1: 과제당 시도 수 가정
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def comp(name, k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return {
        "comparison": name, "k1": k1, "n1": n1, "k2": k2, "n2": n2,
        "diff_pp": round(100 * d, 1),
        "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
        "fisher_p": round(fisher(k1, n1, k2, n2), 4),
    }


# ---------- 표 1 (주방, 실로봇 성공률 0–1). 열: 과제1, 과제2, 과제3, ALL(논문 값) ----------
T1 = {  # (subgoal 20 demos, subgoal 40, long 20, long 40)
    "GC-BC (BC-RNN)": ([0.1, 0.0, 0.1, 0.07], [0.1, 0.2, 0.2, 0.17], [0.0, 0.0, 0.0, 0.00], [0.0, 0.0, 0.1, 0.03]),
    "GC-BC (BC-trans)": ([0.2, 0.0, 0.0, 0.07], [0.3, 0.7, 0.6, 0.53], [0.0, 0.0, 0.0, 0.00], [0.0, 0.0, 0.1, 0.03]),
    "C-BeT": ([0.5, 0.6, 0.0, 0.37], [0.4, 1.0, 0.0, 0.47], [0.0, 0.0, 0.0, 0.00], [0.0, 0.0, 0.0, 0.00]),
    "LMP": ([0.3, 0.1, 0.2, 0.20], [0.6, 0.3, 0.2, 0.37], [0.1, 0.0, 0.1, 0.07], [0.3, 0.1, 0.0, 0.13]),
    "R3M-BC": ([0.9, 0.0, 0.0, 0.30], [0.5, 0.4, 0.0, 0.30], [0.0, 0.0, 0.0, 0.00], [0.5, 0.0, 0.0, 0.17]),
    "Ours (0% human)": ([1.0, 0.5, 0.3, 0.60], [1.0, 0.5, 0.5, 0.67], [0.3, 0.1, 0.3, 0.23], [0.4, 0.3, 0.5, 0.40]),
    "Ours": ([1.0, 0.8, 0.7, 0.83], [1.0, 0.9, 0.8, 0.90], [0.7, 0.3, 0.4, 0.47], [0.7, 0.6, 0.8, 0.70]),
}
T1_COLS = ["subgoal_20", "subgoal_40", "long_20", "long_40"]

# ---------- 표 2 (스터디 데스크, 20 demos). 학습 과제 4개 + 미학습 3개 ----------
T2 = {  # trained(4), trained ALL(paper), unseen(3), unseen ALL(paper)
    "GC-BC (BC-trans)": ([0.0, 0.0, 0.0, 0.0], 0.00, [0.0, 0.0, 0.0], 0.00),
    "LMP": ([0.0, 0.0, 0.0, 0.0], 0.00, [0.0, 0.0, 0.0], 0.00),
    "Ours (0% human)": ([0.2, 0.3, 0.1, 0.2], 0.20, [0.2, 0.1, 0.0], 0.10),
    "Ours (50% human)": ([0.3, 0.4, 0.1, 0.4], 0.30, [0.4, 0.3, 0.1], 0.27),
    "Ours (w/o KL)": ([0.3, 0.7, 0.3, 0.2], 0.38, [0.4, 0.2, 0.0], 0.20),
    "Ours (w/o GMM)": ([0.4, 0.2, 0.2, 0.3], 0.28, [0.2, 0.0, 0.0], 0.07),
    "Ours": ([0.6, 0.7, 0.4, 0.5], 0.55, [0.7, 0.5, 0.2], 0.47),
}

# ---------- 표 3 (다중 과제). 열: Flower, Whiteboard, Sandwich, Cloth, ALL(논문) ----------
T3 = {
    "LMP-single": ([0.1, 0.0, 0.1, 0.3], 0.13),
    "LMP": ([0.0, 0.0, 0.0, 0.2], 0.05),
    "R3M-single": ([0.2, 0.1, 0.3, 0.4], 0.25),
    "R3M": ([0.1, 0.1, 0.2, 0.2], 0.15),
    "Ours-single": ([0.5, 0.5, 0.6, 0.7], 0.58),
    "Ours": ([0.4, 0.2, 0.8, 0.8], 0.55),
}

# ---------- 표 4 (시뮬 LIBERO, 5 시드 x 100 시도). (평균, SD) ----------
T4_SUB = {
    "GC-BC (BC-RNN)": [(0.96, 0.02), (0.00, 0.00), (0.00, 0.00), (0.00, 0.00), (0.00, 0.00)],
    "GC-BC (BC-trans)": [(0.95, 0.03), (0.01, 0.02), (0.00, 0.00), (0.01, 0.02), (0.00, 0.00)],
    "Ours (0% human)": [(0.92, 0.06), (0.70, 0.05), (0.80, 0.07), (0.74, 0.11), (0.78, 0.04)],
}
T4_LONG = {
    "GC-BC (BC-RNN)": [(0.00, 0.00)] * 5,
    "GC-BC (BC-trans)": [(0.00, 0.00)] * 5,
    "Ours (0% human)": [(0.58, 0.04), (0.29, 0.10), (0.62, 0.11), (0.59, 0.05), (0.67, 0.09)],
}


def main():
    # V1 일관성: 논문 ALL이 과제 값의 평균과 맞는가 (반올림 0.005 이내)
    incons = []
    for m, cols in T1.items():
        for c, vals in zip(T1_COLS, cols):
            mean = sum(vals[:3]) / 3
            if abs(mean - vals[3]) > 0.006:
                incons.append({"row": m, "col": c, "computed": round(mean, 3), "paper": vals[3]})
    for m, (tr, ta, un, ua) in T2.items():
        if abs(sum(tr) / 4 - ta) > 0.006:
            incons.append({"row": m, "col": "trained_ALL", "computed": round(sum(tr) / 4, 3), "paper": ta})
        if abs(sum(un) / 3 - ua) > 0.006:
            incons.append({"row": m, "col": "unseen_ALL", "computed": round(sum(un) / 3, 3), "paper": ua})
    for m, (vals, all) in T3.items():
        if abs(sum(vals) / 4 - all) > 0.006:
            incons.append({"row": m, "col": "T3_ALL", "computed": round(sum(vals) / 4, 3), "paper": all})
    RES["consistency_mismatches"] = incons

    # 실로봇 이항 비교 (V1: 과제당 10회 가정, ALL은 과제 수 x 10)
    comps = []
    # 표 1 장기 과제 ALL (3과제 x 10 = 30)
    o_long20 = round(sum(T1["Ours"][2][:3]) * N_REAL)
    z_long20 = round(sum(T1["Ours (0% human)"][2][:3]) * N_REAL)
    o_long40 = round(sum(T1["Ours"][3][:3]) * N_REAL)
    z_long40 = round(sum(T1["Ours (0% human)"][3][:3]) * N_REAL)
    comps.append(comp("표1 장기 20 demos: Ours vs Ours(0% human)", o_long20, 30, z_long20, 30))
    comps.append(comp("표1 장기 40 demos: Ours vs Ours(0% human)", o_long40, 30, z_long40, 30))
    comps.append(comp("표1 장기 20 demos: Ours vs LMP", o_long20, 30, round(sum(T1["LMP"][2][:3]) * N_REAL), 30))
    comps.append(comp("표1 장기 20 demos: Ours vs C-BeT", o_long20, 30, round(sum(T1["C-BeT"][2][:3]) * N_REAL), 30))
    comps.append(comp("표1 장기 20 demos: Ours vs R3M-BC", o_long20, 30, round(sum(T1["R3M-BC"][2][:3]) * N_REAL), 30))
    o_sub20 = round(sum(T1["Ours"][0][:3]) * N_REAL)
    rbc_sub20 = round(sum(T1["R3M-BC"][0][:3]) * N_REAL)
    comps.append(comp("표1 첫 부분과제 20 demos: Ours vs R3M-BC", o_sub20, 30, rbc_sub20, 30))
    # 표 2 (4과제 x 10 = 40; 미학습 3 x 10 = 30)
    comps.append(comp("표1 장기 40 demos: Ours vs R3M-BC", round(sum(T1["Ours"][3][:3]) * N_REAL), 30,
                      round(sum(T1["R3M-BC"][3][:3]) * N_REAL), 30))
    comps.append(comp("표1 장기 40 demos: Ours vs LMP", round(sum(T1["Ours"][3][:3]) * N_REAL), 30,
                      round(sum(T1["LMP"][3][:3]) * N_REAL), 30))
    comps.append(comp("표2 학습과제 ALL: Ours vs Ours(50% human)", round(0.55 * 40), 40, round(0.30 * 40), 40))
    comps.append(comp("표2 학습과제 ALL: Ours vs Ours(0% human)", round(0.55 * 40), 40, round(0.20 * 40), 40))
    comps.append(comp("표2 학습과제 ALL: Ours vs Ours(w/o KL)", round(0.55 * 40), 40, round(0.375 * 40), 40))
    comps.append(comp("표2 학습과제 ALL: Ours vs Ours(w/o GMM)", round(0.55 * 40), 40, round(0.275 * 40), 40))
    comps.append(comp("표2 미학습 ALL: Ours vs Ours(0% human)", round(0.4667 * 30), 30, round(0.10 * 30), 30))
    comps.append(comp("표2 미학습 ALL: Ours vs Ours(w/o GMM)", round(0.4667 * 30), 30, round(0.0667 * 30), 30))
    comps.append(comp("표2 미학습 ALL: Ours vs Ours(w/o KL)", round(0.4667 * 30), 30, round(0.20 * 30), 30))
    comps.append(comp("표2 미학습 ALL: Ours vs Ours(50% human)", round(0.4667 * 30), 30, round(0.2667 * 30), 30))
    comps.append(comp("표3 다중과제: Ours 대 Ours-single 감소(ALL 0.58 -> 0.55)", round(0.55 * 40), 40, round(0.575 * 40), 40))
    RES["real_binomial"] = comps

    # 표 3 단일 대 다중 과제 감소 (논문 값 그대로)
    RES["table3_drop_pp"] = {
        "LMP": round(100 * (0.13 - 0.05), 1),
        "R3M": round(100 * (0.25 - 0.15), 1),
        "Ours": round(100 * (0.58 - 0.55), 1),
    }

    # 표 4 시뮬: 과제 평균 (SD는 과제별 SD의 제곱합 평균의 제곱근, 과제 독립 가정 근사)
    sim = {}
    for label, table in (("subgoal", T4_SUB), ("long", T4_LONG)):
        for m, rows in table.items():
            means = np.array([r[0] for r in rows])
            sds = np.array([r[1] for r in rows])
            mean_all = means.mean()
            se_all = math.sqrt(np.sum(sds ** 2 / 5)) / 5  # 5 시드, 과제 독립 가정 근사
            sim[f"{label}|{m}"] = {"mean_of_tasks": round(mean_all, 3),
                                   "approx_se": round(se_all, 3),
                                   "approx_95ci": [round(mean_all - 1.96 * se_all, 3), round(mean_all + 1.96 * se_all, 3)]}
    # Ours vs BC-trans, 장기 과제별 Welch (5 시드)
    welch = []
    for i in range(5):
        m1, s1 = T4_LONG["Ours (0% human)"][i]
        m2, s2 = T4_LONG["GC-BC (BC-trans)"][i]
        if s1 == 0 and s2 == 0:
            welch.append({"task": i + 1, "p": None, "note": "0 vs 0 (분산 0)"})
            continue
        r = stats.ttest_ind_from_stats(m1, s1, 5, m2, s2, 5, equal_var=False)
        welch.append({"task": i + 1, "t": round(float(r.statistic), 2), "p": float(r.pvalue)})
    RES["sim_table4"] = sim
    RES["sim_welch_long_ours_vs_bctrans"] = welch

    # 표 4 시드 SD 해석: 5 시드 Ours 장기 평균 대 BC-RNN 장기 0 이므로 검정 무의미
    # 표 1 Subgoal 전체 대 장기 간격 (Ours)
    RES["gap_long_vs_subgoal_ours_pp"] = {
        "kitchen_20": round(100 * (sum(T1["Ours"][0][:3]) / 3 - sum(T1["Ours"][2][:3]) / 3), 1),
        "kitchen_40": round(100 * (sum(T1["Ours"][1][:3]) / 3 - sum(T1["Ours"][3][:3]) / 3), 1),
    }

    # 텍스트 주장 확인 (논문 문장 -> 계산)
    RES["text_claims"] = {
        "ours_vs_0human_long_all_pp": round(100 * (0.70 - 0.40), 1),  # 본문 "more than 23%"
        "ours_vs_0human_trained_all_pp": round(100 * (0.55 - 0.20), 1),
        "ours_vs_best_end2end_long_20_pp": round(100 * (0.40 - 0.17), 1),  # 0% human vs R3M-BC(0.17), 표1 장기 40
        "conclusion_over_50_pp_vs_R3M_BC_long": round(100 * (0.70 - 0.17), 1),
        "unseen_ours_vs_best_baseline_pp": round(100 * (0.47 - 0.00), 1),  # 본문 "more than 35%"
        "withoutKL_gap_pp": round(100 * (0.55 - 0.38), 1),  # 본문 "17% lower"
        "KL_overlap_claim": "23% (t-SNE 면적, 그림 7) - 그림 값만 있고 수치 표는 없음, 재계산 불가",
    }

    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
    print("불일치(반올림 초과):", incons if incons else "없음")
    for c in comps:
        print(f"{c['comparison']}: {c['k1']}/{c['n1']} vs {c['k2']}/{c['n2']} diff {c['diff_pp']} pp {c['newcombe95_pp']} p={c['fisher_p']}")
    print("표3 감소(pp):", RES["table3_drop_pp"])
    for w in welch:
        print("Welch 장기 과제", w)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
