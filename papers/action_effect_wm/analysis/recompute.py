"""논문 표 값으로 신뢰구간, 검정, 일관성 검사를 다시 계산한다 (CPU, 수 초).

입력: 논문 Table 1, 2, 3과 본문 수치를 직접 옮긴 값 (paper/paper.txt의 해당 줄 참조).
출력: analysis/recompute_results.json
가정은 analysis/03_analysis.md §0에 적었다. 원자료(정책별 성공률, 에피소드별 결과)는 논문에 없다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
res = {}

# ---- 1. Pearson 상관: 셀 수 n 가정(4 체크포인트 x 2 과제 = 8), Fisher z 근사 CI ----
PEARSON = {  # (환경, 백본, 방법): r   (Table 1, Table 2)
    ("UR5e", "DreamDojo", "base"): 0.451, ("UR5e", "DreamDojo", "AFFECT"): 0.686,
    ("UR5e", "Cosmos3", "base"): 0.091, ("UR5e", "Cosmos3", "AFFECT"): 0.334,
    ("DROID", "DreamDojo", "base"): 0.468, ("DROID", "DreamDojo", "AFFECT"): 0.790,
    ("DROID", "Cosmos3", "base"): 0.425, ("DROID", "Cosmos3", "AFFECT"): 0.583,
    ("ALOHA-2", "DreamDojo", "base"): 0.325, ("ALOHA-2", "DreamDojo", "AFFECT"): 0.490,
    ("ALOHA-2", "Cosmos3", "base"): 0.298, ("ALOHA-2", "Cosmos3", "AFFECT"): 0.694,
}
N_CELLS = 8


def fisher_ci(r, n, alpha=0.05):
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    q = stats.norm.ppf(1 - alpha / 2)
    lo, hi = math.tanh(z - q * se), math.tanh(z + q * se)
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r * r)
    p = 2 * stats.t.sf(abs(t), n - 2)
    return lo, hi, p


pear = {}
for k, r in PEARSON.items():
    lo, hi, p = fisher_ci(r, N_CELLS)
    pear["|".join(k)] = {"r": r, "ci95_lo": round(lo, 3), "ci95_hi": round(hi, 3), "p_two_sided_n8": round(p, 3)}
res["pearson_n8"] = pear

# 6 조합 모두 AFFECT가 base보다 높다. 독립 가정의 부호검정(참고용, 조합들은 독립이 아님)
gains = [(k[0], k[1], PEARSON[(k[0], k[1], "AFFECT")] - PEARSON[(k[0], k[1], "base")]) for k in PEARSON if k[2] == "base"]
res["pearson_gains"] = [{"env": e, "backbone": b, "delta": round(d, 3)} for e, b, d in gains]
res["pearson_sign_test_all_6_improve_if_indep_p"] = 0.5 ** 6

# 독립 표본 Fisher z 차이 검정 (참고: 같은 실로봇 성공률을 공유하므로 실제는 의존 상관)
def indep_diff_p(r1, r2, n1=N_CELLS, n2=N_CELLS):
    z = (math.atanh(r2) - math.atanh(r1)) / math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return 2 * stats.norm.sf(abs(z))

res["droid_dreamdojo_indep_diff_p"] = round(indep_diff_p(0.468, 0.790), 3)
res["aloha_cosmos3_indep_diff_p"] = round(indep_diff_p(0.298, 0.694), 3)

# 표본 크기 민감도: n이 6, 8, 12일 때 r=0.790의 CI 하한
res["droid_affect_ci_by_n"] = {str(n): [round(x, 3) for x in fisher_ci(0.790, n)[:2]] for n in (6, 8, 12)}

# ---- 2. FVD 상대 변화 (논문 값 그대로) ----
FVD = [  # 이름, base, AFFECT
    ("UR5e DreamDojo short", 85, 73), ("UR5e DreamDojo long", 108, 93),
    ("UR5e Cosmos3 short", 470, 463), ("UR5e Cosmos3 long", 562, 540),
    ("DROID DreamDojo", 807, 622), ("DROID Cosmos3", 910, 744),
    ("ALOHA-2 DreamDojo", 183, 188), ("ALOHA-2 Cosmos3", 333, 257),
]
res["fvd_rel_change_pct"] = {n: round(100 * (b - a) / a, 1) for n, a, b in [(x[0], x[1], x[2]) for x in FVD]}

# ---- 3. 계획 성공률 (Figure 3): 비율 검정과 Wilson CI ----
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1)


def two_prop(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    _, p_fisher = stats.fisher_exact(table)
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    return round(p_fisher, 3), round(2 * stats.norm.sf(abs(z)), 3)


# UR5e: 5 과제 x 12 시도 = 60 (46.7 = 28/60, 60.0 = 36/60, 68.3 = 41/60 로 맞아떨어짐)
ur = {"Uniform": 28, "DreamDojo": 36, "DreamDojo+AFFECT": 41}
res["ur5e_planning"] = {k: {"k": v, "n": 60, "pct": round(100 * v / 60, 1), "wilson95": wilson(v, 60)} for k, v in ur.items()}
res["ur5e_planning_tests"] = {
    "DreamDojo_vs_AFFECT_fisher_p_onesided_or_two": two_prop(36, 60, 41, 60),
    "Uniform_vs_AFFECT_fisher_p": two_prop(28, 60, 41, 60),
}

# ALOHA-2: 평균 80.6 / 72.2 는 36 시도(3 과제 x 12)로 맞아떨어짐 (29/36, 26/36). 분모는 역산 추정.
al = {"DreamDojo+AFFECT": 29, "DreamDojo": 26}
res["aloha2_planning_inferred_n36"] = {k: {"k": v, "n": 36, "pct": round(100 * v / 36, 2), "wilson95": wilson(v, 36)} for k, v in al.items()}
res["aloha2_planning_tests"] = {"fisher_p": two_prop(26, 36, 29, 36)}

# 본문 반올림 점검: 80.6 - 72.2 = 8.4 로 적었으나 3/36 = 8.33
res["aloha2_gain_pp_exact"] = round(100 * (29 / 36 - 26 / 36), 2)
res["ur5e_gain_pp_exact"] = round(100 * (41 / 60 - 36 / 60), 2)

# ---- 4. Table 3 ablation (단일 시드 가정, 분산 정보 없음) ----
res["ablation_delta_vs_flowonly_fvd"] = {
    "AFFECT_vs_base_10k": 256 - 284,
    "placebo_vs_base": 460 - 284,
    "uncond_vs_base": 451 - 284,
    "AFFECT_vs_base_20pct": 393 - 405,
}

# ---- 5. 본문 대 표 일관성 검사 ----
checks = {
    "abstract_23pct_fvd_droid": round(100 * (807 - 622) / 807, 1),        # 22.9 -> "up to 23%"
    "text_18pct_fvd_droid_cosmos3": round(100 * (910 - 744) / 910, 1),     # 18.2 -> "18%"
    "section_4_1_ur5e_planning_gain_pp": 8.3,                               # 68.3 - 60.0
    "section_4_2_aloha_gain_text_pp": 8.4,                                  # 자체 계산 8.33
    "intro_six_combos_improve_pearson": all(g["delta"] > 0 for g in res["pearson_gains"]),
    "table2_aloha_dreamdojo_fvd_worse": 188 > 183,
    "ablation_table3_base_matches_table2": False,  # Table 3 base 23.43/0.07/284 는 Table 2 의 어떤 행과도 다르다
}
res["consistency"] = checks

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False)[:4000])
