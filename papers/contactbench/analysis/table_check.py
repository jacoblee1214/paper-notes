"""ContactBench 1.0 Table III 재계산: 표 일치 점검, Wilson 구간, Fisher, CMH, Holm, Newcombe, 40 N 역산.

입력은 논문 Table III(v1과 v2 동일, 2026-11-10 확인)와 Fig. 11(v2) 단계 퍼널의 '마지막 단계' 값이다.
원자료(에피소드별 결과)는 공개되지 않았으므로 모든 검정은 집계 표에서만 한다.
실행: python3 analysis/table_check.py → analysis/table_check_results.json (1초 미만)
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, fisher_exact, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"
N = 30  # 방법·과제당 에피소드 수
Z = norm.ppf(0.975)

TASKS = ["thread", "tighten", "plug", "door", "utensil", "card"]
TASK_NAME = {
    "thread": "Thread nut onto screw",
    "tighten": "Tighten screw with Allen key",
    "plug": "Plug in charger",
    "door": "Turn doorknob to open door",
    "utensil": "Retrieve single utensil from bin",
    "card": "Pick up thin card",
}
HUMAN_S = {"thread": 17.5, "tighten": 19.7, "plug": 7.3, "door": 7.6, "utensil": 11.8, "card": 7.5}
HUMAN_B2_S = {"thread": 19.6, "tighten": 21.0, "plug": 8.4, "door": 8.8, "utensil": 12.9, "card": 8.6}

# 표 Table III: (성공 k/30, 진행률, >40 N 퍼센트, 시간 배수). 시간 배수가 없으면 None.
POLICY = {
    "DP": {"thread": (8, 0.39, 33, 1.72), "tighten": (0, 0.58, 10, None), "plug": (2, 0.32, 80, 1.06),
           "door": (30, 1.00, 10, 0.92), "utensil": (0, 0.50, 97, None), "card": (14, 0.74, 57, 1.11)},
    "RDP": {"thread": (0, 0.24, 27, None), "tighten": (0, 0.36, 0, None), "plug": (4, 0.34, 83, 1.15),
            "door": (29, 0.98, 63, 0.91)},
    "PI05": {"thread": (14, 0.66, 27, 1.22), "tighten": (17, 0.82, 7, 1.50), "plug": (6, 0.44, 40, 0.89),
             "door": (30, 1.00, 7, 0.87), "utensil": (16, 0.70, 47, 2.05), "card": (2, 0.28, 60, 0.93)},
    "FTP1": {"thread": (14, 0.63, 83, 1.32), "tighten": (3, 0.45, 3, 0.84), "plug": (4, 0.35, 40, 0.95),
             "door": (29, 0.98, 43, 0.89)},
}
POLICY_NAME = {"DP": "Diffusion Policy", "RDP": "Reactive Diffusion Policy", "PI05": "pi0.5",
               "FTP1": "Foundation Tactile Policy (FTP-1)"}
VISION = {"DP", "PI05"}
# Fig. 11(v2) 퍼널의 마지막 단계('task terminated correctly', 'door opened', 'card grasped') 퍼센트
FUNNEL_LAST = {
    "DP": {"thread": 27, "tighten": 0, "plug": 7, "door": 100, "utensil": 0, "card": 47},
    "RDP": {"thread": 0, "tighten": 0, "plug": 13, "door": 97},
    "PI05": {"thread": 47, "tighten": 57, "plug": 20, "door": 100, "utensil": 53, "card": 7},
    "FTP1": {"thread": 47, "tighten": 10, "plug": 13, "door": 97},
}
# Table II의 기술 수(skills 목록 길이): 가중치 민감도에 쓴다
SKILLS = {"thread": 2, "tighten": 2, "plug": 1, "door": 1, "utensil": 2, "card": 1}


def wilson(k, n=N):
    p = k / n
    den = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def newcombe_diff(k1, n1, k2, n2):
    """두 비율 차이(k1/n1 - k2/n2)의 Newcombe(1998) 점수 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return float(d), float(lo), float(hi)


def fisher_p(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(fisher_exact(table, alternative="two-sided")[1])


def cmh(tables):
    """Mantel–Haenszel 층화 검정(연속성 보정). tables: [[a, b], [c, d]] 목록, a/c = 첫째/둘째 방법의 성공.
    반환: 보정 카이제곱, p, 공통 오즈비(MH)."""
    num_dev, var_sum, or_num, or_den = 0.0, 0.0, 0.0, 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        r1, r2, c1, c2 = a + b, c + d, a + c, b + d
        num_dev += a - r1 * c1 / n
        var_sum += r1 * r2 * c1 * c2 / (n**2 * (n - 1))
        or_num += a * d / n
        or_den += b * c / n
    stat = (abs(num_dev) - 0.5) ** 2 / var_sum
    # 층의 b*c가 모두 0이면 MH 오즈비가 정의되지 않는다(None)
    or_mh = float(or_num / or_den) if or_den > 0 else None
    return {"chi2": float(stat), "p": float(chi2.sf(stat, 1)), "or_mh": or_mh}


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return [float(x) for x in adj]


def cell(pol, task):
    return POLICY[pol].get(task)


def check_table():
    """표의 퍼센트와 퍼널 값이 k/30에서 반올림으로 나오는지 확인하고, 40 N 칸의 k를 역산한다."""
    issues, pct40_counts = [], {}
    for pol, rows in POLICY.items():
        for task, (k, prog, pct, _) in rows.items():
            # 40 N 퍼센트를 가장 가까운 k/30으로 역산
            kk = int(round(pct * N / 100))
            if round(100 * kk / N) != pct:
                issues.append((pol, task, f"40N {pct}% 불일치"))
            pct40_counts[f"{pol}/{task}"] = kk
    funnel_ok = {}
    for pol, rows in FUNNEL_LAST.items():
        for task, pct in rows.items():
            k = POLICY[pol][task][0]
            funnel_ok[f"{pol}/{task}"] = (round(100 * k / N) == pct)
    return issues, pct40_counts, funnel_ok


def main():
    res = {"meta": {"n_per_cell": N, "source": "arXiv:2610.24512 Table III (v1과 v2 동일), Fig. 11(v2)"}}

    issues, pct40, funnel_ok = check_table()
    res["consistency"] = {
        "table_percent_issues": issues,
        "funnel_last_stage_matches_SR": all(funnel_ok.values()),
        "funnel_mismatches": [k for k, v in funnel_ok.items() if not v],
        "time_human_III_vs_B2_diff_s": {t: round(HUMAN_B2_S[t] - HUMAN_S[t], 2) for t in TASKS},
    }

    # 1) 셀별 성공률과 Wilson 95% 구간
    cells = {}
    for pol in POLICY:
        for task in POLICY[pol]:
            k = POLICY[pol][task][0]
            lo, hi = wilson(k)
            cells[f"{pol}/{task}"] = {"k": k, "p": round(k / N, 4), "wilson95": [round(lo, 4), round(hi, 4)]}
    res["cells_wilson"] = cells

    # 2) >40 N 칸: 역산한 개수와 Wilson 구간
    res["force40_counts"] = {key: {"k": v, "wilson95": [round(x, 4) for x in wilson(v)]} for key, v in pct40.items()}

    # 3) 기본 비교 DP vs pi0.5 (6과제, 쌍별 Fisher, Holm)
    base_tasks = [t for t in TASKS if t in POLICY["DP"] and t in POLICY["PI05"]]
    p_base = [fisher_p(POLICY["DP"][t][0], N, POLICY["PI05"][t][0], N) for t in base_tasks]
    holm_base = holm(p_base)
    res["base_dp_vs_pi05"] = [
        {"task": t, "dp": POLICY["DP"][t][0], "pi05": POLICY["PI05"][t][0],
         "diff_pp": round(100 * (POLICY["PI05"][t][0] - POLICY["DP"][t][0]) / N, 1),
         "fisher_p": round(p, 5), "holm_p": round(h, 4)}
        for t, p, h in zip(base_tasks, p_base, holm_base)
    ]

    # 4) 촉각 효과: 시각 기반 → 촉각 기반, 4과제 (CMH 층화, 쌍별 Fisher, Holm)
    def tactile_block(vis, tac):
        tasks = [t for t in TASKS if t in POLICY[vis] and t in POLICY[tac]]
        pairs = [fisher_p(POLICY[vis][t][0], N, POLICY[tac][t][0], N) for t in tasks]
        holm_p = holm(pairs)
        tables = [((POLICY[vis][t][0], N - POLICY[vis][t][0]), (POLICY[tac][t][0], N - POLICY[tac][t][0]))
                  for t in tasks]
        mh = cmh(tables)
        pooled_v = sum(POLICY[vis][t][0] for t in tasks)
        pooled_t = sum(POLICY[tac][t][0] for t in tasks)
        n_tot = N * len(tasks)
        d, lo, hi = newcombe_diff(pooled_t, n_tot, pooled_v, n_tot)
        # 'tighten' 과제를 뺀 민감도 분석
        loo = {}
        for drop in tasks:
            keep = [t for t in tasks if t != drop]
            loo[drop] = cmh([((POLICY[vis][t][0], N - POLICY[vis][t][0]), (POLICY[tac][t][0], N - POLICY[tac][t][0]))
                             for t in keep])["p"]
        return {
            "vision": vis, "tactile": tac, "tasks": tasks,
            "per_task": [{"task": t, "vis_k": POLICY[vis][t][0], "tac_k": POLICY[tac][t][0],
                          "diff_tac_minus_vis_pp": round(100 * (POLICY[tac][t][0] - POLICY[vis][t][0]) / N, 1),
                          "fisher_p": round(p, 4), "holm_p": round(h, 4)}
                         for t, p, h in zip(tasks, pairs, holm_p)],
            "cmh": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in mh.items()},
            "pooled": {"vis_k": pooled_v, "tac_k": pooled_t, "n_per_arm": n_tot,
                       "diff_tac_minus_vis": round(d, 4), "newcombe95": [round(lo, 4), round(hi, 4)],
                       "fisher_p_unstratified": round(fisher_p(pooled_t, n_tot, pooled_v, n_tot), 5)},
            "leave_one_task_out_cmh_p": {k: round(v, 5) for k, v in loo.items()},
        }

    res["tactile_dp_to_rdp"] = tactile_block("DP", "RDP")
    res["tactile_pi05_to_ftp1"] = tactile_block("PI05", "FTP1")

    # 5) 평균 성공률 (같은 과제 집합에서 비교)
    res["mean_sr"] = {
        "DP_6tasks": round(sum(POLICY["DP"][t][0] for t in TASKS if t in POLICY["DP"]) / (N * 6), 4),
        "PI05_6tasks": round(sum(POLICY["PI05"][t][0] for t in TASKS) / (N * 6), 4),
        "DP_4tasks": round(sum(POLICY["DP"][t][0] for t in ["thread", "tighten", "plug", "door"]) / (N * 4), 4),
        "RDP_4tasks": round(sum(POLICY["RDP"][t][0] for t in ["thread", "tighten", "plug", "door"]) / (N * 4), 4),
        "PI05_4tasks": round(sum(POLICY["PI05"][t][0] for t in ["thread", "tighten", "plug", "door"]) / (N * 4), 4),
        "FTP1_4tasks": round(sum(POLICY["FTP1"][t][0] for t in ["thread", "tighten", "plug", "door"]) / (N * 4), 4),
    }

    # 6) 검출력의 해석: 30 대 30에서 80% 검정력의 최소 검출 차이(정규 근사, 양측 0.05)
    def mde(p_bar, n=N):
        za, zb = Z, norm.ppf(0.8)
        return float((za + zb) * np.sqrt(2 * p_bar * (1 - p_bar) / n))

    res["mde_80pct_n30"] = {"p_bar_0.5": round(mde(0.5), 3), "p_bar_0.2": round(mde(0.2), 3)}

    # 7) 텍스트 주장 점검
    res["text_claims"] = {
        "best_overall_56.7pct_is_17_of_30": POLICY["PI05"]["tighten"][0] == 17,
        "pi05_plug_20pct_is_6_of_30": POLICY["PI05"]["plug"][0] == 6,
        "text_80pct_and_53.3pct_are_24_and_16_of_30": (round(24 / 30, 3) == 0.8, round(16 / 30, 3) == 0.533),
    }

    # 8) Table IV(공개 표 V) 게이트 폭: 60 mm 대 55 mm 차이와 기대 차이(약 5 mm)
    grip = {"DP": (58.9, 0.12, 30, 59.0, 0.11, 10), "RDP": (59.0, 0.13, 30, 59.0, 0.11, 10),
            "PI05": (58.8, 0.13, 30, 58.7, 0.11, 10), "FTP1": (59.2, 0.14, 30, 58.9, 0.14, 10)}
    gw = {}
    for pol, (m1, s1, n1, m2, s2, n2) in grip.items():
        se = np.sqrt(s1**2 / n1 + s2**2 / n2)
        gw[pol] = {"diff_mm_55_minus_60": round(m2 - m1, 2), "welch_t": round((m2 - m1) / se, 2),
                   "expected_diff_mm_if_adjusted": 5.0}
    res["gripper_width"] = gw

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("wrote", OUT)
    print("consistency:", json.dumps(res["consistency"], ensure_ascii=False)[:300])
    for key in ["base_dp_vs_pi05", ]:
        for r in res[key]:
            print(r)
    for key in ["tactile_dp_to_rdp", "tactile_pi05_to_ftp1"]:
        b = res[key]
        print(key, "CMH p=", b["cmh"]["p"], "OR_MH=", b["cmh"]["or_mh"], "pooled", b["pooled"])
        print("  per-task", [(r["task"], r["fisher_p"], r["holm_p"]) for r in b["per_task"]])
        print("  leave-one-out", b["leave_one_task_out_cmh_p"])
    print("mean_sr", res["mean_sr"], "mde", res["mde_80pct_n30"])


if __name__ == "__main__":
    main()
