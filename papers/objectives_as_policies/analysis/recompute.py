"""논문 표 값(Table II–IV)을 시도 수로 되돌려 합계, 비율, 신뢰구간, 검정을 다시 계산한다.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
입력은 논문 본문과 표에서 옮긴 값이다(paper/paper.txt). 네트워크와 모델 호출은 없다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return [round(float(lo), 4), round(float(hi), 4)]


# ---- 1. Table II: 합계와 비율 검사 ------------------------------------------
# 열: 피드백 라운드 0..5. 행: 과제. 값은 누적 성공 수(논문 Table II 그대로).
t2 = {
    "without": {
        "Pick-and-place": [6, 9, 10, 10, 10, 10],
        "Cup Reorientation": [4, 7, 8, 8, 9, 9],
        "Direct Push": [4, 7, 8, 9, 9, 9],
        "Push with Tool": [6, 8, 10, 10, 10, 10],
    },
    "with": {
        "Pick-and-place": [9, 10, 10, 10, 10, 10],
        "Cup Reorientation": [4, 6, 7, 7, 8, 9],
        "Direct Push": [7, 10, 10, 10, 10, 10],
        "Push with Tool": [6, 8, 9, 10, 10, 10],
    },
}
t2_reported_total = {
    "without": [20, 31, 36, 37, 38, 38],
    "with": [26, 34, 36, 37, 38, 39],
}
t2_check = {}
for mode, rows in t2.items():
    mat = np.array(list(rows.values()))
    sums = mat.sum(axis=0).tolist()
    t2_check[mode] = {
        "computed_totals": sums,
        "reported_totals": t2_reported_total[mode],
        "match": sums == t2_reported_total[mode],
        "monotone_nondecreasing": bool(np.all(np.diff(mat, axis=1) >= 0)),
    }
RES["table2_totals_check"] = t2_check

# ---- 2. Table III: 기준선과 절제 ------------------------------------------
t3 = {
    "L2R": [2, 3, 4, 2, 11],
    "ReKep": [9, 6, 0, 2, 17],
    "CoRAL (adapted)": [7, 6, 10, 8, 31],
    "OpenVLA-OFT": [0, 0, 0, 0, 0],
    "pi0.5": [10, 0, 0, 0, 10],
    "w/o library, w/o SR": [1, 1, 3, 3, 8],
    "w/o library, with SR": [5, 3, 3, 3, 14],
    "with library, w/o SR": [6, 4, 4, 6, 20],
    "with library, with SR (0 rounds)": [9, 4, 7, 6, 26],
    "OaP (2 rounds, with SR)": [10, 7, 10, 9, 36],
}
t3_check = {}
for name, v in t3.items():
    t3_check[name] = {"sum_of_tasks": sum(v[:4]), "reported": v[4], "match": sum(v[:4]) == v[4]}
RES["table3_totals_check"] = t3_check
# Table II 0라운드와 Table III 절제 행이 같은 값인지 확인
RES["table2_r0_vs_table3_ablation"] = {
    "without_SR_r0_equals_library_wo_SR": t2["without"] and [t2["without"][k][0] for k in t2["without"]] == t3["with library, w/o SR"][:4],
    "with_SR_r0_equals_library_with_SR": [t2["with"][k][0] for k in t2["with"]] == t3["with library, with SR (0 rounds)"][:4],
}

# ---- 3. Table IV: 실로봇 ------------------------------------------------
t4 = {
    "Pick-and-place": {"small": [4, 5], "large": [4, 5]},
    "Cup Reorientation": {"small": [4, 5], "large": [3, 5]},
    "Direct Push": {"small": [4, 5], "large": [4, 5]},
    "Push with Tool": {"small": [5, 5], "large": [4, 5]},
}
small_sum = sum(v["small"][0] for v in t4.values())
large_sum = sum(v["large"][0] for v in t4.values())
per_task_overall = {k: v["small"][0] + v["large"][0] for k, v in t4.items()}
RES["table4_check"] = {
    "small_total": [small_sum, 20],
    "large_total": [large_sum, 20],
    "per_task_overall_of_10": per_task_overall,
    "reported_per_task": {"Pick-and-place": 8, "Cup Reorientation": 7, "Direct Push": 8, "Push with Tool": 9},
    "overall": [small_sum + large_sum, 40],
}

# ---- 4. 핵심 비율과 신뢰구간 --------------------------------------------
headline = {
    "OaP 2 rounds (sim, 36/40)": (36, 40),
    "OaP 0 rounds with SR (sim, 26/40)": (26, 40),
    "OaP 0 rounds without SR (sim, 20/40)": (20, 40),
    "OaP 1 round with SR (sim, 34/40)": (34, 40),
    "OaP 5 rounds with SR (sim, 39/40)": (39, 40),
    "Real world (32/40)": (32, 40),
    "Real world small perturb (17/20)": (17, 20),
    "Real world large perturb (15/20)": (15, 20),
    "CoRAL adapted (31/40)": (31, 40),
    "ReKep (17/40)": (17, 40),
    "L2R (11/40)": (11, 40),
    "pi0.5 (10/40)": (10, 40),
    "w/o library w/o SR (8/40)": (8, 40),
    "with library w/o SR (20/40)": (20, 40),
}
RES["intervals"] = {
    name: {"k": k, "n": n, "p": round(k / n, 4), "wilson95": wilson(k, n), "clopper_pearson95": clopper(k, n)}
    for name, (k, n) in headline.items()
}

# ---- 5. 검정 (Fisher 정확 검정, 양측과 단측) ---------------------------
# 주의: 같은 과제 안의 시도는 짝이 아니므로 시도를 독립으로 본 근사다. 과제별 짝 정보가 없어서
# McNemar를 쓸 수 없다. 여러 비교는 Holm으로 보정한다.
def fisher(a, b, label):
    (k1, n1), (k2, n2) = a, b
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    _, p2 = stats.fisher_exact(table, alternative="two-sided")
    _, p1 = stats.fisher_exact(table, alternative="greater")
    diff = k1 / n1 - k2 / n2
    return {"label": label, "table": table, "diff_pp": round(100 * diff, 1), "p_two_sided": round(float(p2), 4),
            "p_one_sided_greater": round(float(p1), 4)}


tests = [
    fisher((26, 40), (20, 40), "Sim 0-round: with SR vs without SR"),
    fisher((20, 40), (8, 40), "Sim 0-round, no SR: with library vs without library"),
    fisher((26, 40), (14, 40), "Sim 0-round, with SR: with library vs without library"),
    fisher((36, 40), (31, 40), "OaP 2 rounds vs CoRAL (adapted)"),
    fisher((34, 40), (31, 40), "OaP 1 round vs CoRAL (adapted)"),
    fisher((36, 40), (17, 40), "OaP 2 rounds vs ReKep"),
    fisher((36, 40), (11, 40), "OaP 2 rounds vs L2R"),
    fisher((36, 40), (10, 40), "OaP 2 rounds vs pi0.5"),
    fisher((36, 40), (0, 40), "OaP 2 rounds vs OpenVLA-OFT"),
    fisher((32, 40), (36, 40), "Real world 32/40 vs sim 2-round 36/40 (not independent: programs chosen from sim successes)"),
    fisher((17, 20), (15, 20), "Real small vs large perturbation"),
    fisher((39, 40), (38, 40), "Sim 5 rounds: with SR vs without SR"),
]
# Holm 보정은 OaP 2라운드 대 기준선 다섯 개(ReKep, L2R, CoRAL, pi0.5, OpenVLA-OFT)에만 적용한다.
base_idx = [3, 5, 6, 7, 8]
raw = np.array([tests[i]["p_two_sided"] for i in base_idx])
order = np.argsort(raw)
m = len(raw)
holm = np.empty(m)
running = 0.0
for rank, idx in enumerate(order):
    adj = min(1.0, (m - rank) * raw[idx])
    running = max(running, adj)
    holm[idx] = running
for j, i in enumerate(base_idx):
    tests[i]["p_holm_5_baselines"] = round(float(holm[j]), 4)
RES["tests"] = tests

# ---- 6. 검정력: 40 시도에서 차이를 검출할 최소 차이(근사, 양측 5%, 80%) ----
def mde_pp(p0, n=40, z_a=1.96, z_b=0.8416):
    # 정규 근사: 두 비율이 n개씩일 때 차이 d = (z_a + z_b) * sqrt(2 p(1-p)/n)
    return round(100 * (z_a + z_b) * np.sqrt(2 * p0 * (1 - p0) / n), 1)


RES["power_mde_pp_n40"] = {str(p): mde_pp(p) for p in (0.2, 0.5, 0.65, 0.8, 0.9)}

# ---- 7. 시간 일관성: 250 갱신 x 40 제어 x 2 ms = 20 s -------------------------
RES["time_budget_check"] = {
    "250 updates x 40 controls x 2 ms = seconds": 250 * 40 * 0.002,
    "reported commanded motion limit (s)": 20,
}

# ---- 8. 연속 평가 지표: 1라운드 대 0라운드(같은 40시도) ---------------------
RES["increment_per_round_with_SR"] = [26, 34, 36, 37, 38, 39]

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
