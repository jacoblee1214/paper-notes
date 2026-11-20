"""UVA 논문 보고 수치 재계산 (표준 라이브러리만 사용).

- 이항 비율의 Wilson 95% CI
- 두 비율의 Fisher 정확검정 (양측)
- 보고 수치가 표본 크기와 정수로 맞는지 점검
결과는 recompute_results.json으로 저장한다.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _hyper_logpmf(k, K, N, n):
    # 초기하분포 로그확률: N개 중 K개 성공, n개 뽑을 때 k개 성공
    return (math.lgamma(K + 1) - math.lgamma(k + 1) - math.lgamma(K - k + 1)
            + math.lgamma(N - K + 1) - math.lgamma(n - k + 1) - math.lgamma(N - K - n + k + 1)
            - (math.lgamma(N + 1) - math.lgamma(n + 1) - math.lgamma(N - n + 1)))


def fisher_two_sided(k1, n1, k2, n2):
    """2x2 표 [[k1, n1-k1],[k2, n2-k2]]의 양측 Fisher 정확검정 p값."""
    N = n1 + n2
    K = k1 + k2
    lo, hi = max(0, K - n2), min(K, n1)
    p_obs = math.exp(_hyper_logpmf(k1, K, N, n1))
    total = 0.0
    for k in range(lo, hi + 1):
        p = math.exp(_hyper_logpmf(k, K, N, n1))
        if p <= p_obs * (1 + 1e-7):
            total += p
    return min(1.0, total)


def two_prop_z(p1, n1, p2, n2):
    """두 비율의 정규근사 z검정 (양측 p값). 표본 크기 가정이 필요할 때만 쓴다."""
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))


out = {}

# 1) 실제 로봇 (Table II): 과제당 20회, 단일 과제 Cup 20회
rw = {
    "single_cup": {"DP-UMI": (19, 20), "UVA": (17, 20)},
    "multi_cup": {"DP-UMI": (10, 20), "UVA": (13, 20)},
    "multi_towel": {"DP-UMI": (14, 20), "UVA": (14, 20)},
    "multi_mouse": {"DP-UMI": (8, 20), "UVA": (16, 20)},
    "multi_pooled_60": {"DP-UMI": (32, 60), "UVA": (43, 60)},
}
out["table2_real_world"] = {}
for task, d in rw.items():
    (ka, na), (kb, nb) = d["DP-UMI"], d["UVA"]
    out["table2_real_world"][task] = {
        "DP-UMI": {"k": ka, "n": na, "rate": round(ka / na, 3), "wilson95": [round(x, 3) for x in wilson(ka, na)]},
        "UVA": {"k": kb, "n": nb, "rate": round(kb / nb, 3), "wilson95": [round(x, 3) for x in wilson(kb, nb)]},
        "diff_pp_UVA_minus_DP": round(100 * (kb / nb - ka / na), 1),
        "fisher_p_two_sided": round(fisher_two_sided(kb, nb, ka, na), 4),
    }

# 2) 블록 밀기 (Table V): 조건당 10회 x 4조건 = 40회
bp = {"DP-C": (15, 40), "DP-C+UVA_forward": (24, 40), "DP-C+GT_sim": (30, 40)}
out["table5_block_pushing"] = {}
for name, (k, n) in bp.items():
    out["table5_block_pushing"][name] = {"k": k, "n": n, "rate": round(k / n, 3),
                                         "wilson95": [round(x, 3) for x in wilson(k, n)]}
out["table5_block_pushing"]["fisher_UVA_vs_DPC_p"] = round(fisher_two_sided(24, 40, 15, 40), 4)
out["table5_block_pushing"]["fisher_GT_vs_UVA_p"] = round(fisher_two_sided(30, 40, 24, 40), 4)
out["table5_block_pushing"]["per_cell_check"] = {
    "DP-C cells (k/10)": [2, 5, 6, 2], "DP-C mean": round(15 / 40, 4),
    "UVA cells (k/10)": [8, 7, 5, 4], "UVA mean": round(24 / 40, 4),
    "GT cells (k/10)": [8, 8, 7, 7], "GT mean": round(30 / 40, 4),
}

# 3) Libero10 (Table I): 500 평가 에피소드 가정 (본문 50회 x 10과제). 기준 모델 n은 논문에 없어 가정.
out["table1_libero10_assumed_n500"] = {}
for name, p in [("DP-T", 0.58), ("DP-C", 0.53), ("OpenVLA", 0.54), ("pi0", 0.85),
                ("pi0-FAST", 0.60), ("UVA-action", 0.86)]:
    out["table1_libero10_assumed_n500"][f"UVA(0.90) vs {name}({p})"] = {
        "diff_pp": round(100 * (0.90 - p), 1),
        "z_test_p": round(two_prop_z(0.90, 500, p, 500), 4),
    }

# 4) Table VIII: 인간 영상 추가 효과 (500 test, 10과제 x 50회)
out["table8_human_data"] = {
    "500test UVA 450/500 vs +human 455/500": {
        "fisher_p": round(fisher_two_sided(455, 500, 450, 500), 4),
        "diff_pp": 1.0,
    },
    "30test UVA 28/30 vs +human 29/30": {"fisher_p": round(fisher_two_sided(29, 30, 28, 30), 4)},
}

# 5) 정수 점검: 표본 크기 n에서 보고된 비율이 가능한 값인지
def feasible(rate, n, tol=0.005):
    k = round(rate * n)
    return abs(k / n - rate) <= tol + 1e-9, k


checks = {}
for label, rate, n in [
    ("Table I PushT DP-C 0.91 (n=50)", 0.91, 50),
    ("Table I PushT OpenVLA 0.35 (n=50)", 0.35, 50),
    ("Table I PushT UVA-action 0.45 (n=50)", 0.45, 50),
    ("Table I Tool DP-C 0.95 (n=50)", 0.95, 50),
    ("Table I PushT UVA 0.98 (n=50)", 0.98, 50),
    ("Table I Libero UVA 0.90 (n=500)", 0.90, 500),
    ("Table II Cup DP-UMI 0.95 (n=20)", 0.95, 20),
    ("Table III GoalColor UVA 0.64 (n=50)", 0.64, 50),
    ("Table III BgColor UVA 0.35 (n=50)", 0.35, 50),
    ("Table III BgObject UVA 0.31 (n=50)", 0.31, 50),
    ("Table VIII 30test UVA 0.93 (n=30)", 0.93, 30),
    ("Table VIII 30test +human 0.97 (n=30)", 0.97, 30),
]:
    ok, k = feasible(rate, n)
    checks[label] = {"integer_count_fits": ok, "nearest_k": k}
out["integer_feasibility"] = checks

# 6) Table VII 분해 산술 검증 (ms)
vae, tr, tr_fa, a16, a100, vid16, vid100 = 40, 40, 30, 15, 93, 100, 625
out["table7_arithmetic"] = {
    "UVA 16 steps (40+40+15)": vae + tr + a16,
    "UVA 16 steps FA (40+30+15)": vae + tr_fa + a16,
    "UVA 100 steps (40+40+93)": vae + tr + a100,
    "UVA 100 steps FA (40+30+93)": vae + tr_fa + a100,
    "Policy+Planner 16 (95+100)": vae + tr + a16 + vid16,
    "Policy+Planner 100 (173+625)": vae + tr + a100 + vid100 + 0,
    "reported": {"UVA16": 95, "UVA16FA": 85, "UVA100": 173, "UVA100FA": 163, "P+P16": 195, "P+P100": 798},
    "attention share of UVA16 time": round(40 / 95, 3),
}

with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
