"""TacPilot 논문 표 수치 재계산과 검정 (수 초, numpy·scipy).

입력: 논문 Table 1 (성공 횟수), Table 2 (A, B, C). 값은 논문 PDF(arXiv:2611.23804v1)에서 옮겼다.
출력: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().with_name("recompute_results.json")
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


# Table 1 (성공/시도)
T1 = {
    "ACT": {"4x8": (7, 16), "5x10": (8, 14), "all": (15, 30)},
    "DTR": {"4x8": (13, 16), "5x10": (8, 14), "all": (21, 30)},
    "TacPilot": {"4x8": (13, 16), "5x10": (13, 14), "all": (26, 30)},
}

# 표 1 내부 일관성: 크기별 합 = 전체, 실패 합 = 시도 - 성공
cons = {}
for m, d in T1.items():
    cons[m] = {
        "sum_sizes_equals_overall": bool(d["4x8"][0] + d["5x10"][0] == d["all"][0]
        and d["4x8"][1] + d["5x10"][1] == d["all"][1]),
    }
fail_counts = {"ACT": (9, 6), "DTR": (6, 3), "TacPilot": (2, 2)}  # (safety, timeout)
for m in T1:
    cons[m]["failures_match"] = bool(sum(fail_counts[m]) == T1[m]["all"][1] - T1[m]["all"][0])
res["table1_consistency"] = cons

pairs = [("TacPilot", "DTR"), ("TacPilot", "ACT"), ("DTR", "ACT")]
rows = []
for a, b in pairs:
    for key in ["4x8", "5x10", "all"]:
        ka, na = T1[a][key]
        kb, nb = T1[b][key]
        table = [[ka, na - ka], [kb, nb - kb]]
        _, p_f = stats.fisher_exact(table, alternative="two-sided")
        lo, hi = newcombe(ka, na, kb, nb)
        # 쌍체 설계일 때 가능한 최소 p: 불일치 쌍 수 = |차이|, 한쪽 방향만 있을 때 (McNemar 정확검정)
        d_ab = abs(ka - kb)
        p_pair_min = min(1.0, 2 * 0.5 ** d_ab) if d_ab > 0 else 1.0
        rows.append({
            "pair": f"{a} vs {b}", "size": key,
            "rate_a": round(ka / na, 4), "rate_b": round(kb / nb, 4),
            "diff_pp": round(100 * (ka / na - kb / nb), 1),
            "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
            "fisher_p": round(p_f, 4),
            "paired_min_p_mcnemar": round(p_pair_min, 5),
        })
res["table1_tests"] = rows
res["table1_wilson95"] = {m: {k: [round(x, 3) for x in wilson(*T1[m][k])] for k in T1[m]} for m in T1}

# 검출력: 26/30(0.867) 대 21/30(0.70)을 80%로 잡으려면 군당 n (정규근사, 양측 0.05)
p1, p2 = 26 / 30, 21 / 30
pbar = (p1 + p2) / 2
za, zb = 1.959964, 0.841621
n_need = ((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)) ** 2
res["power_n_per_arm_for_observed_effect"] = math.ceil(n_need)

# Table 2 (B): 전체 MSE가 네 단계 MSE의 평균인지 확인 (단위 ×10^3)
phases = {
    "G/G": [0.016246, 0.300583, 2.898689, 0.575885, 0.947851],
    "A/G": [0.060888, 0.324025, 2.485711, 0.575885, 0.861627],
    "G/A": [0.139274, 0.300583, 3.171612, 0.537579, 1.037262],
    "A/A": [0.183916, 0.324025, 2.758634, 0.537579, 0.951039],
}
T2B = {}
for k, v in phases.items():
    T2B[k] = {"mean_of_phases": round(float(np.mean(v[:4])), 6), "reported": v[4],
              "match": bool(abs(np.mean(v[:4]) - v[4]) < 2e-6)}
res["table2B_overall_check"] = T2B
res["table2B_A_over_G_gripper_gate_reduction_pct"] = round(100 * (0.04193 - 0.00841) / 0.04193, 1)
res["table2B_AG_vs_GG_overall_pct"] = round(100 * (0.947851 - 0.861627) / 0.947851, 1)

# Table 2 (C): 입력 제거 조건의 전체 MSE 평균 검산과 텍스트 주장 (7.70×, 3.07%, 18.2%)
T2C = {
    "Full": [0.016246, 0.300583, 2.898689, 0.575885],
    "No rich": [0.019789, 0.269088, 4.363805, 24.530964],
    "No events": [0.009989, 0.355350, 2.990882, 0.551741],
    "Neither": [0.012198, 0.472941, 5.537737, 23.928948],
}
T2C_reported = {"Full": 0.947851, "No rich": 7.295912, "No events": 0.976991, "Neither": 7.487956}
res["table2C_overall_check"] = {k: {"mean_of_phases": round(float(np.mean(v)), 6),
                                    "reported": T2C_reported[k],
                                    "match": bool(abs(np.mean(v) - T2C_reported[k]) < 2e-5)} for k, v in T2C.items()}
res["table2C_no_rich_ratio"] = round(7.295912 / 0.947851, 2)
res["table2C_no_events_overall_pct"] = round(100 * (0.976991 - 0.947851) / 0.947851, 2)
res["table2C_no_events_grasp_pct"] = round(100 * (0.355350 - 0.300583) / 0.300583, 1)

# Table 2 (A): 계산량
A = {
    "Periodic/Full": {"requests": 14841, "calls": 14841, "regions": 474912, "enc_s": 30.60, "proc_s": 79.12},
    "Event/Full": {"requests": 12753, "calls": 12753, "regions": 408096, "enc_s": 26.16, "proc_s": 76.86},
    "Periodic/Local": {"requests": 14841, "calls": 9290, "regions": 206975, "enc_s": 19.82, "proc_s": 68.00},
    "Event/Local": {"requests": 12753, "calls": 9290, "regions": 206975, "enc_s": 19.17, "proc_s": 69.73},
}
base = A["Periodic/Full"]
checks = {}
for k, v in A.items():
    checks[k] = {
        "regions_reduction_pct": round(100 * (1 - v["regions"] / base["regions"]), 2),
        "enc_reduction_pct": round(100 * (1 - v["enc_s"] / base["enc_s"]), 2),
        "regions_per_call": round(v["regions"] / v["calls"], 2),
    }
res["table2A_costs"] = checks
res["table2A_event_full_regions_claim_14.1pct"] = checks["Event/Full"]["regions_reduction_pct"]
res["table2A_event_local_enc_s_claim_37.3pct"] = round(100 * (1 - 19.17 / 30.60), 2)

# 시간 단위: 호출당 ms 검산
res["table2A_ms_per_call_check"] = {k: round(1000 * v["enc_s"] / v["calls"], 3) for k, v in A.items()}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
