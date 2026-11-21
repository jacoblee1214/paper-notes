"""3DTouch (arXiv:2611.23961 v1) 보고 수치 재계산: 평균, Wilson CI, Newcombe 차이 CI, Fisher 정확 검정.
입력은 논문 Table 1, 2, 3의 값만 사용한다 (성공 횟수 = 비율 x 20/100). 실행: python3 analysis/recompute.py
"""
import json, math, os
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
Z = 1.959964

def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h

def newcombe_diff(k1, n1, k2, n2):
    """Newcombe (1998) method 10: CI for p1 - p2 from Wilson intervals."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi

def pct(k, n):
    return round(100 * k / n, 1)

TASKS = ["Funnel", "Plug", "Button", "Sponge", "Pipette"]
N = 20  # trials per task (논문 4.1)

# Table 1 (%), 열 순서: Funnel, Plug, Button, Sponge, Pipette; 보고된 Avg
T1 = {
    "pi0": ([0, 0, 45, 60, 10], 23.0),
    "pi0.5": ([15, 50, 75, 60, 20], 44.0),
    "SimpleDP3+tactile": ([10, 0, 10, 20, 10], 10.0),
    "PointACT": ([0, 20, 15, 20, 10], 13.0),
    "pi0+tactile": ([0, 15, 45, 70, 25], 31.0),
    "pi0.5+tactile": ([20, 55, 30, 70, 40], 43.0),
    "Tactile-VLA": ([10, 25, 45, 85, 60], 45.0),
    "3DTouch": ([35, 60, 80, 90, 60], 65.0),
}
# Table 3 (%), 열: Funnel, Plug, Button, Sponge, Pipette; 보고된 Avg (괄호 변화량 제외)
T3 = {
    "3DTouch (Full)": ([35, 60, 80, 90, 60], 65.0),
    "w/o geom, Global tactile": ([15, 60, 55, 80, 55], 53.0),
    "w/o geom, Local tactile": ([15, 40, 25, 90, 40], 42.0),
    "Point cloud only": ([10, 50, 50, 80, 40], 46.0),
    "Tactile only (Global)": ([10, 50, 30, 60, 25], 35.0),
    "Geometric relations only": ([5, 45, 50, 70, 35], 41.0),
    "RGB only (=pi0)": ([0, 0, 45, 60, 10], 23.0),
}
# Table 2 (%), 열: Funnel Trans, Rot, Both; Plug Trans, Rot, Both
T2 = {
    "pi0.5+tactile": {"unperturbed": (20, 55), "pert": [10, 0, 0, 55, 50, 30]},
    "3DTouch": {"unperturbed": (35, 60), "pert": [30, 30, 30, 55, 60, 50]},
}

def k_of(pct_val):
    k = round(pct_val * N / 100)
    assert abs(k * 100 / N - pct_val) < 1e-9, pct_val
    return k

out = {"checks": {}, "pooled": {}, "per_task_tests": {}, "ablation": {}, "table2": {}}

# V1: 평균 재계산 대 보고 평균
for name, (vals, avg) in list(T1.items()) + [("T3:" + k, v) for k, v in T3.items()]:
    re = sum(vals) / 5
    out["checks"][name] = {"recomputed_avg": round(re, 2), "reported_avg": avg, "match": abs(re - avg) < 0.05}

# 풀링 (5 task x 20 = 100 trials)
pooled = {}
for name, (vals, _) in T1.items():
    k = sum(k_of(v) for v in vals)
    lo, hi = wilson(k, 100)
    pooled[name] = {"successes": k, "n": 100, "pct": k, "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}
out["pooled"] = pooled

def cmp(a, b, src_a, src_b):
    ka, kb = pooled[a]["successes"], pooled[b]["successes"]
    d, lo, hi = newcombe_diff(ka, 100, kb, 100)
    _, p = fisher_exact([[ka, 100 - ka], [kb, 100 - kb]])
    return {"diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)], "fisher_p": round(p, 4)}

out["pooled_comparisons"] = {
    "3DTouch_vs_Tactile-VLA": cmp("3DTouch", "Tactile-VLA", "", ""),
    "3DTouch_vs_pi0.5+tactile": cmp("3DTouch", "pi0.5+tactile", "", ""),
    "3DTouch_vs_pi0": cmp("3DTouch", "pi0", "", ""),
    "3DTouch_vs_pi0.5": cmp("3DTouch", "pi0.5", "", ""),
}

# 과제별 Fisher (3DTouch 대 최강 기준선 두 개)
for task_i, task in enumerate(TASKS):
    a = k_of(T1["3DTouch"][0][task_i])
    row = {"3DTouch": a}
    res = {}
    for base in ["Tactile-VLA", "pi0.5+tactile", "pi0.5"]:
        b = k_of(T1[base][0][task_i])
        _, p = fisher_exact([[a, N - a], [b, N - b]])
        d, lo, hi = newcombe_diff(a, N, b, N)
        res[base] = {"k_vs_k": [a, b], "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)], "fisher_p": round(p, 3)}
    out["per_task_tests"][task] = res


# Holm 보정: 과제별 검정 15개 (3 기준선 x 5 과제)
pairs = [(t, b) for t in TASKS for b in ["Tactile-VLA", "pi0.5+tactile", "pi0.5"]]
pv = sorted([(out["per_task_tests"][t][b]["fisher_p"], (t, b)) for t, b in pairs])
m = len(pv); run = 0.0
for i, (p, key) in enumerate(pv):
    adj = min(1.0, max(run, (m - i) * p))
    run = adj
    out["per_task_tests"][key[0]][key[1]]["holm_p"] = round(adj, 3)

# 절제 (Table 3): 풀링 기준 Full 대 각 변형
abl = {}
full_k = sum(k_of(v) for v in T3["3DTouch (Full)"][0])
for name, (vals, avg) in T3.items():
    k = sum(k_of(v) for v in vals)
    lo, hi = wilson(k, 100)
    _, p = fisher_exact([[full_k, 100 - full_k], [k, 100 - k]])
    abl[name] = {"successes": k, "pct": k, "wilson95": [round(100 * lo, 1), round(100 * hi, 1)], "fisher_vs_full_p": round(p, 4)}
out["ablation"] = abl

# 표 2: 강건성. 같은 행의 무섭동 시험과 섭동 시험은 서로 다른 시행이므로 독립 두 표본으로 본다.
for m, d in T2.items():
    u_k = sum(k_of(v) for v in d["unperturbed"])  # 2 tasks x 20
    p_k = sum(k_of(v) for v in d["pert"])          # 6 settings x 20 = 120
    u_n, p_n = 40, 120
    drop = 100 * (u_k / u_n) - 100 * (p_k / p_n)
    # 차이의 분산 (독립 가정, Wald)
    pu, pp = u_k / u_n, p_k / p_n
    se = 100 * math.sqrt(pu * (1 - pu) / u_n + pp * (1 - pp) / p_n)
    out["table2"][m] = {
        "unperturbed_pct_pooled": round(100 * pu, 1),
        "perturbed_pct_pooled": round(100 * pp, 1),
        "drop_pp": round(drop, 1),
        "drop_wald95_pp": [round(drop - 1.96 * se, 1), round(drop + 1.96 * se, 1)],
        "reported_drop_pp": {"3DTouch": 5.0, "pi0.5+tactile": 13.3}[m],
        "per_setting_k": d["pert"],
    }

# Table 2 차이의 차이 (3DTouch 드롭 - 기준선 드롭)
a = out["table2"]["3DTouch"]; b = out["table2"]["pi0.5+tactile"]
dd = b["drop_pp"] - a["drop_pp"]
se_a = (a["drop_wald95_pp"][1] - a["drop_wald95_pp"][0]) / (2 * 1.96)
se_b = (b["drop_wald95_pp"][1] - b["drop_wald95_pp"][0]) / (2 * 1.96)
se_dd = math.sqrt(se_a ** 2 + se_b ** 2)
out["table2"]["difference_in_drop_pp"] = {"baseline_minus_3DTouch": round(dd, 1), "wald95": [round(dd - 1.96 * se_dd, 1), round(dd + 1.96 * se_dd, 1)]}

# 과제 평균 (Table 2) 확인: 3DTouch 무섭동 Funnel 35 Plug 60 -> 47.5; 섭동 평균 42.5 -> 드롭 5.0 확인
out["checks"]["table2_text_5pp"] = {"3DTouch_drop": out["table2"]["3DTouch"]["drop_pp"], "reported": 5.0}
out["checks"]["table2_text_13.3pp"] = {"baseline_drop": out["table2"]["pi0.5+tactile"]["drop_pp"], "reported": 13.3}

# 텍스트 주장 검산
claims = {
    "42pp over pi0 (65-23)": 65 - 23,
    "20pp over Tactile-VLA (65-45)": 65 - 45,
    "11pp shared grounding (53-42)": 53 - 42,
    "12pp geometry (65-53)": 65 - 53,
    "7pp PC+tactile over PC (53-46)": 53 - 46,
    "18pp PC+tactile over tactile only (53-35)": 53 - 35,
    "Funnel +25 vs Tactile-VLA": 35 - 10,
    "Plug +35 vs Tactile-VLA": 60 - 25,
    "Button +35 vs Tactile-VLA": 80 - 45,
    "Sponge +5 vs Tactile-VLA": 90 - 85,
    "Funnel +15 vs best task baseline (20)": 35 - 20,
    "Funnel +20 geometry (35-15)": 35 - 15,
    "Button +25 geometry (80-55)": 80 - 55,
}
out["text_claims_arithmetic"] = claims


# 주요 대비 (Table 3 합산, 독립 가정)
def ab(name_a, name_b):
    ka = sum(k_of(v) for v in T3[name_a][0]); kb = sum(k_of(v) for v in T3[name_b][0])
    _, p = fisher_exact([[ka, 100 - ka], [kb, 100 - kb]])
    d, lo, hi = newcombe_diff(ka, 100, kb, 100)
    return {"diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)], "fisher_p": round(p, 4)}
out["ablation_contrasts"] = {
    "Global vs Local (no geom)": ab("w/o geom, Global tactile", "w/o geom, Local tactile"),
    "Full vs Global no geom (geom)": ab("3DTouch (Full)", "w/o geom, Global tactile"),
    "PC+tactile vs PC only": ab("w/o geom, Global tactile", "Point cloud only"),
    "PC+tactile vs tactile only": ab("w/o geom, Global tactile", "Tactile only (Global)"),
}
# 표 1 안의 두 값 대비: pi0.5 대 pi0.5+촉각, Button Press 15/20 대 6/20 (표 1 수치가 보고 값과 같은지 확인용)
out["pi05_vs_pi05tactile_button"] = {"k": [k_of(75.0), k_of(30.0)], "fisher_p": round(fisher_exact([[15, 5], [6, 14]])[1], 4)}

with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ["pooled_comparisons", "table2", "ablation"]}, ensure_ascii=False, indent=1))
print("V1 mismatches:", [k for k, v in out["checks"].items() if isinstance(v, dict) and v.get("match") is False])
