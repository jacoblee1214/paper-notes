"""논문 표 값의 재계산과 일관성 검사 (CPU, numpy·scipy만 사용). 결과는 recompute_results.json.

실행: python3 analysis/recompute.py
입력: 논문 표 값(아래 상수, 출처 표 번호 표기), 공식 노트북 저장 출력(repo/code_as_policies/*.ipynb, 읽기만).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "repo" / "code_as_policies"
OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


# ---- 표 I: RoboCodeGen, 문제 37개 (논문 표 I) ----
T1 = {"GPT-3 6.7B": (3, 5), "GPT-3 175B": (68, 84), "cushman": (54, 57), "code-davinci-002": (81, 95)}  # (Flat, Hier)
N1 = 37
res["T1_k_over_37"] = {}
for m, (f, h) in T1.items():
    kf = [k for k in range(N1 + 1) if round(100 * k / N1) == f]
    kh = [k for k in range(N1 + 1) if round(100 * k / N1) == h]
    res["T1_k_over_37"][m] = {"flat_k_candidates": kf, "hier_k_candidates": kh}

# 공식 노트북 저장 출력의 성공 수 (k/37). 셀 번호는 노트북 cell index.
NB = {
    "text-curie-001 (6.7B)": {"hier_hier": (2, 30), "hier_flat": (1, 31), "flat_hier": (2, 32), "flat_flat": (1, 33)},
    "text-davinci-002 (175B)": {"hier_hier": (31, 25), "hier_flat": (28, 26), "flat_hier": (26, 27), "flat_flat": (25, 28)},
    "code-cushman-001": {"hier_hier": (21, 20), "hier_flat": (21, 21), "flat_hier": (20, 22), "flat_flat": (22, 23)},
    "code-davinci-002": {"hier_hier": (36, 15), "hier_flat": (32, 16), "flat_hier": (26, 17), "flat_flat": (30, 18)},
}
# 표 I의 Hier 열 = hier_hier, Flat 열 = flat_flat 으로 대응시킨다.
res["T1_notebook_vs_paper"] = {}
for m, cfg in NB.items():
    key = {"text-curie-001 (6.7B)": "GPT-3 6.7B", "text-davinci-002 (175B)": "GPT-3 175B",
           "code-cushman-001": "cushman", "code-davinci-002": "code-davinci-002"}[m]
    hk, fk = cfg["hier_hier"][0], cfg["flat_flat"][0]
    res["T1_notebook_vs_paper"][m] = {
        "notebook_hier_hier_k": hk, "notebook_hier_hier_pct": round(100 * hk / N1, 1), "paper_hier": T1[key][1],
        "notebook_flat_flat_k": fk, "notebook_flat_flat_pct": round(100 * fk / N1, 1), "paper_flat": T1[key][0],
        "notebook_flat_cg_hier_prompt_k": cfg["flat_hier"][0],
        "notebook_hier_cg_flat_prompt_k": cfg["hier_flat"][0],
    }

# 비짝지은 Fisher 정확검정 (짝지은 자료가 없어 참고용). 항목: (성공, 실패) 쌍
def fisher_pair(k1, k2, n1, n2):
    tab = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(tab)[1])

def k_of(pct, n):
    return [x for x in range(n + 1) if round(100 * x / n) == pct][0]


res["T1_fisher_unpaired_hier_vs_flat"] = {
    m: round(fisher_pair(k_of(T1[m][1], N1), k_of(T1[m][0], N1), N1, N1), 4) for m in T1
}

# ---- 표 II / V: HumanEval, 문제 164개 ----
N2 = 164
res["T2_k_over_164"] = {
    "greedy_flat_45.7": round(100 * 75 / N2, 2), "greedy_hier_53.0": round(100 * 87 / N2, 2),
    "T5_flatprompt_greedy_50.6": round(100 * 83 / N2, 2),
}
# 본문 "164문제 중 6.5%가 계층 생성으로 이어짐" 검사: 정수 개수 후보
cands = [k for k in range(N2 + 1) if round(100 * k / N2, 1) == 6.5]
res["HE_6.5pct_integer_candidates"] = cands
# 44%와 56%가 정수 비율이 되는 N (두 Flat 변형의 성공률)
res["HE_44_56_integer_N_below_164"] = [n for n in range(1, N2 + 1)
                                      if round(0.44 * n, 6).is_integer() and round(0.56 * n, 6).is_integer()]
res["HE_44_56_smallest_N_share_pct"] = round(100 * 25 / N2, 1)

# ---- 표 III / VI: 과제당 50 시행 ----
# 표 VI 각 행: (CLIPort oracle, CLIPort no-oracle, NL planner, CaP). None = N/A
rows_seen = {  # SA SI: 1-5 LH, 6-8 SG
    "1 pick-place": (88, 44, 98, 100), "2 stack all": (98, 4, 94, 94), "3 corner/side": (96, 8, 46, 92),
    "4 bowls": (100, 22, 94, 100), "5 matching colors": (12, 14, 100, 100),
    "6 direction": (100, 80, None, 72), "7 distance": (92, 54, None, 98), "8 nth": (100, 38, None, 98)}
rows_ua_si = {
    "1": (12, 10, 98, 100), "2": (96, 8, 96, 100), "3": (0, 0, 58, 100), "4": (46, 0, 88, 96), "5": (30, 26, 100, 92),
    "6": (0, 0, None, 60), "7": (0, 0, None, 100), "8": (0, 0, None, 60)}
rows_ua_ui = {
    "1 different corners": (0, 0, 60, 98), "2 mismatched colors": (0, 0, 92, 60), "3 stack on corner": (0, 0, 40, 82),
    "4 magnitude": (0, 0, None, 38), "5 corner distance": (4, 0, None, 58), "6 line": (0, 0, None, 90)}
LH = {"SA_SI": list(rows_seen)[:5], "UA_SI": list(rows_ua_si)[:5], "UA_UI": list(rows_ua_ui)[:3]}
SG = {"SA_SI": list(rows_seen)[5:], "UA_SI": list(rows_ua_si)[5:], "UA_UI": list(rows_ua_ui)[3:]}
TAB6 = {"SA_SI": rows_seen, "UA_SI": rows_ua_si, "UA_UI": rows_ua_ui}
names = ["CLIPort_oracle", "CLIPort_no_oracle", "NL_planner", "CaP"]
res["T6_recomputed_means"] = {}
res["T6_vs_T3"] = {}
T3_paper = {("SA_SI", "LH"): (78.80, 86.40, 97.20), ("SA_SI", "SG"): (97.33, None, 89.30),
            ("UA_SI", "LH"): (36.80, 88.00, 97.60), ("UA_SI", "SG"): (0.00, None, 73.33),
            ("UA_UI", "LH"): (0.00, 64.00, 80.00), ("UA_UI", "SG"): (0.01, None, 62.00)}
for fam, tab in TAB6.items():
    for kind, sel in [("LH", LH[fam]), ("SG", SG[fam])]:
        vals = {}
        for j, nm in enumerate(names):
            v = [tab[r][j] for r in sel if tab[r][j] is not None]
            if v:
                vals[nm] = round(float(np.mean(v)), 2)
        res["T6_recomputed_means"][f"{fam}_{kind}"] = vals
        ref = T3_paper[(fam, kind)]
        res["T6_vs_T3"][f"{fam}_{kind}"] = {
            "table3_paper": {n: r for n, r in zip(["CLIPort_oracle", "NL_planner", "CaP"], [ref[0], ref[1], ref[2]])},
            "recomputed_from_table6": {n: vals.get(n) for n in ["CLIPort_oracle", "NL_planner", "CaP"]},
        }
# 전체 합계 (표 VI Total 행과 대조)
tot = {}
for fam, tab in TAB6.items():
    pass
allrows_seen = list(rows_seen.values()); allua = list(rows_ua_si.values()); alluu = list(rows_ua_ui.values())
for nm, j in zip(names, range(4)):
    tot[nm] = {
        "SA_SI_total": round(float(np.mean([r[j] for r in allrows_seen if r[j] is not None])), 2),
        "UA_SI_total": round(float(np.mean([r[j] for r in allua if r[j] is not None])), 2),
        "UA_UI_total": round(float(np.mean([r[j] for r in alluu if r[j] is not None])), 2),
    }
res["T6_totals_recomputed"] = tot
res["T6_totals_paper"] = {
    "SA_SI": {"CLIPort_oracle": 85.8, "CLIPort_no_oracle": 33.0, "NL_planner": 86.4, "CaP": 94.3},
    "UA_SI": {"CLIPort_oracle": 23.0, "CLIPort_no_oracle": 5.5, "NL_planner": 88.0, "CaP": 88.5},
    "UA_UI": {"CLIPort_oracle": 0.7, "CLIPort_no_oracle": 0.0, "NL_planner": 64.0, "CaP": 71.0},
}

# 과제 단위 표준오차(과제 평균의 t 구간)와 풀링 Wilson 구간 (CaP 기준)
def task_ci(vals):
    v = np.array(vals, float) / 100
    m = v.mean()
    se = v.std(ddof=1) / math.sqrt(len(v))
    t = stats.t.ppf(0.975, len(v) - 1)
    return [round(100 * (m - t * se), 1), round(100 * (m + t * se), 1)]

res["CaP_family_ci"] = {}
for fam, tab in TAB6.items():
    for kind, sel in [("LH", LH[fam]), ("SG", SG[fam])]:
        v = [tab[r][3] for r in sel]
        k = int(round(sum(v) / 100 * 50))
        n = 50 * len(v)
        res["CaP_family_ci"][f"{fam}_{kind}"] = {"tasks": len(v), "trials": n, "successes": k,
                                                  "wilson95_pooled": wilson(k, n), "task_t95": task_ci(v)}

# 검정 (표 VI 과제별 시행 수 50, 과제를 묶은 합계로 Fisher 정확검정; 과제 군집을 무시하므로 참고용)
def fam_counts(fam, kind, j):
    sel = LH[fam] if kind == "LH" else SG[fam]
    vals = [TAB6[fam][r][j] for r in sel if TAB6[fam][r][j] is not None]
    return int(round(sum(vals) / 100 * 50)), 50 * len(vals)

tests = []
pairs = [("SA_SI", "LH", 3, 2), ("SA_SI", "LH", 3, 0), ("UA_SI", "LH", 3, 0), ("UA_UI", "LH", 3, 2),
         ("UA_UI", "LH", 3, 0), ("SA_SI", "SG", 3, 0), ("UA_UI", "SG", 3, 0)]
for fam, kind, a, b in pairs:
    ka, na = fam_counts(fam, kind, a)
    kb, nb = fam_counts(fam, kind, b)
    p = float(stats.fisher_exact([[ka, na - ka], [kb, nb - kb]])[1])
    tests.append({"family": f"{fam} {kind}", "CaP": f"{ka}/{na}", "baseline": f"{kb}/{nb}",
                  "baseline_name": names[b], "fisher_p": round(p, 6)})
res["T6_fisher_tests"] = tests

# ---- 표 IV: 공간·기하 추론, 노트북 출력으로 n 확인 ----
n_obj, n_pos = 28, 23
T4 = {"vanilla": (39, 30), "cot": (68, 48), "lmp": (96, 100)}
res["T4"] = {}
for m, (o, p_) in T4.items():
    ko = round(o * n_obj / 100); kp = round(p_ * n_pos / 100)
    pooled = 100 * (ko + kp) / (n_obj + n_pos)
    res["T4"][m] = {"obj_k_of_28": ko, "pos_k_of_23": kp,
                    "unweighted_mean_of_rows": round((o + p_) / 2, 2),
                    "pooled_pct_51_items": round(pooled, 2)}
res["T4_notebook_check"] = {"vanilla_obj": "0.39285714 = 11/28", "cot_obj": "0.67857143 = 19/28",
                            "lmp_obj": "0.96428571 = 27/28", "vanilla_pos": "0.30434783 = 7/23",
                            "cot_pos": "0.47826087 = 11/23", "lmp_pos": "1.0 = 23/23"}
res["T4_fisher_lmp_vs_cot_pooled_51"] = {
    "lmp_vs_cot": round(float(stats.fisher_exact([[50, 1], [30, 21]])[1]), 8),
    "lmp_vs_vanilla": round(float(stats.fisher_exact([[50, 1], [18, 33]])[1]), 8),
    "note": "표 IV 행 단위 대신 노트북의 28+23 항목을 합쳐 계산. 과제 내 군집을 무시한 참고값",
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
