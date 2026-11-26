"""TopoRetarget(arXiv:2606.16272 v2) 표 1, 표 2와 부록 표 3-6 수치를 다시 계산한다.

읽는 것: paper/paper.txt (pdftotext -layout 결과). 표 값을 정규식으로 읽어 옮겨 적은 값과 대조한다.
쓰는 것: analysis/recompute_results.json
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)
"""
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
TXT = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8")
OUT = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k0, n0):
    """두 비율 차이(p1 - p0)의 Newcombe 95% 구간(Wilson 기반)."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = p1 - p0
    lo = d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return lo, hi


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [None] * len(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def half_up(x, nd=1):
    return math.floor(x * 10 ** nd + 0.5) / 10 ** nd


# ---------------------------------------------------------------- 표 2 (성공률)
METHODS = ["TopoRetarget", "OmniRetarget", "Mink", "DexPilot", "GeoRT"]
T2 = {  # (성공 수, 보고된 % 문자열, 위치오차 cm, 회전오차 deg)
    "Ho-cap": {
        "TopoRetarget": (27, "84.4", 0.87, 5.76), "OmniRetarget": (18, "56.2", 1.07, 7.87),
        "Mink": (24, "75.0", 0.91, 5.55), "DexPilot": (24, "75.0", 0.92, 4.99), "GeoRT": (24, "75.0", 0.90, 6.07)},
    "Pen-Spin": {
        "TopoRetarget": (28, "87.5", 0.98, 9.25), "OmniRetarget": (15, "46.9", 1.45, 14.26),
        "Mink": (7, "21.9", 1.61, 15.25), "DexPilot": (13, "40.6", 1.29, 17.66), "GeoRT": (10, "31.2", 1.19, 18.62)},
}
N = 32

# 원문 텍스트와 대조
for m in re.finditer(r"(\d+\.\d)% \((\d+)/32\)", TXT):
    pass
found = re.findall(r"(\d+\.\d)% \((\d+)/32\)", TXT)
txt_pairs = [(p, int(k)) for p, k in found]
mine = [(T2[d][m][1], T2[d][m][0]) for d in T2 for m in METHODS if m == "TopoRetarget" or True]
OUT["table2_text_check"] = {"found_in_text": len(txt_pairs), "expected": len(mine),
                            "all_found": all(x in txt_pairs for x in mine)}
# 방법 사이 칸에는 '(k/32)'가 첫 열(우리 방법)에만 있으므로, 나머지는 줄 단위로 대조한다
for dataset, key in (("Ho-cap", "Ho-cap"), ("Pen-Spin", "MoCap Pen-Spin")):
    pass
rows = {}
lines = [l for l in TXT.splitlines() if l.startswith("Success rate")]
parsed = []
for l in lines:
    vals = re.findall(r"(\d+\.\d)%(?: \((\d+)/32\))?", l)
    parsed.append(vals)
OUT["table2_success_lines_parsed"] = [[v[0] for v in p] for p in parsed]

res = {}
for dataset in T2:
    res[dataset] = {}
    for m in METHODS:
        k, rep, pe, re_ = T2[dataset][m]
        p = k / N
        lo, hi = wilson(k, N)
        res[dataset][m] = {
            "k": k, "n": N, "rate_exact": 100 * p, "reported": rep,
            "round_half_even": f"{100 * p:.1f}", "round_half_up": f"{half_up(100 * p):.1f}",
            "wilson95": [100 * lo, 100 * hi], "pos_err_cm": pe, "rot_err_deg": re_,
        }
        res[dataset][m]["matches_reported_half_even"] = f"{100 * p:.1f}" == rep
        res[dataset][m]["matches_reported_half_up"] = f"{half_up(100 * p):.1f}" == rep
OUT["table2"] = res

# Topo 대 기준선 검정
tests = []
for dataset in T2:
    k1 = T2[dataset]["TopoRetarget"][0]
    for m in METHODS[1:]:
        k0 = T2[dataset][m][0]
        odds, p = fisher_exact([[k1, N - k1], [k0, N - k0]])
        lo, hi = newcombe(k1, N, k0, N)
        tests.append({"dataset": dataset, "baseline": m, "k_topo": k1, "k_base": k0,
                      "diff_pp": 100 * (k1 - k0) / N, "newcombe95_pp": [100 * lo, 100 * hi],
                      "fisher_p": p, "odds_ratio": odds})
adj = holm([t["fisher_p"] for t in tests])
for t, a in zip(tests, adj):
    t["holm_p_8tests"] = a
OUT["table2_tests"] = tests
# 데이터셋별 Holm(기준선 4개)
for dataset in T2:
    idx = [i for i, t in enumerate(tests) if t["dataset"] == dataset]
    a = holm([tests[i]["fisher_p"] for i in idx])
    for i, v in zip(idx, a):
        tests[i]["holm_p_4tests_in_dataset"] = v

# 두 데이터셋을 합친 64개 클립(독립이라고 가정하지 않고 참고용)
pooled = {}
for m in METHODS:
    k = sum(T2[d][m][0] for d in T2)
    pooled[m] = {"k": k, "n": 2 * N, "rate": 100 * k / (2 * N), "wilson95": [100 * x for x in wilson(k, 2 * N)]}
OUT["table2_pooled64"] = pooled
OUT["table2_pooled64_tests"] = []
for m in METHODS[1:]:
    k1, k0 = pooled["TopoRetarget"]["k"], pooled[m]["k"]
    odds, p = fisher_exact([[k1, 64 - k1], [k0, 64 - k0]])
    OUT["table2_pooled64_tests"].append({"baseline": m, "k_topo": k1, "k_base": k0, "fisher_p": p})

# 40.6 pp
OUT["claim_40_6pp"] = {
    "pen_topo_minus_each": {m: 100 * (T2["Pen-Spin"]["TopoRetarget"][0] - T2["Pen-Spin"][m][0]) / N for m in METHODS[1:]},
    "hocap_topo_minus_each": {m: 100 * (T2["Ho-cap"]["TopoRetarget"][0] - T2["Ho-cap"][m][0]) / N for m in METHODS[1:]},
    "exact_pen_gap_vs_best": 100 * (28 - 15) / 32, "gap_from_rounded_values": 87.5 - 46.9,
}

# 검출 가능한 최소 차이: 기준선 k0에 대해 Fisher p<0.05가 되는 Topo의 최소 성공 수
mde = {}
for k0 in (7, 10, 13, 15, 18, 24):
    need = None
    for k1 in range(k0, N + 1):
        if fisher_exact([[k1, N - k1], [k0, N - k0]])[1] < 0.05:
            need = k1
            break
    mde[str(k0)] = {"baseline_k": k0, "topo_min_k_for_p_lt_005": need,
                    "needed_gap_pp": None if need is None else 100 * (need - k0) / N}
OUT["min_detectable_gap_n32"] = mde

# 오차 열 순위
rank = {}
for dataset in T2:
    for j, nm in ((2, "pos_err_cm"), (3, "rot_err_deg")):
        order = sorted(METHODS, key=lambda m: T2[dataset][m][j])
        rank[f"{dataset}:{nm}"] = {"order_low_to_high": order, "topo_rank": order.index("TopoRetarget") + 1,
                                   "topo_vs_best_pct": 100 * (T2[dataset]["TopoRetarget"][j] / T2[dataset][order[0]][j] - 1)}
OUT["table2_error_ranks"] = rank

# ---------------------------------------------------------------- 표 1
T1 = {
    "Contact precision (mm)": [7.71, 14.15, 14.12, 14.13, 26.77],
    "Contact alignment (deg)": [15.67, 30.80, 37.36, 33.71, 25.74],
    "Max penetration (mm)": [1.07, 1.15, 20.12, 11.87, 22.22],
    "Frames pen > 2 mm (%)": [0.00, 0.00, 84.00, 88.00, 96.00],
    "Solve time (ms/frame)": [4.70, 40.96, 4.37, 1.74, 1.17],
}
t1_found = {}
for key, pat in (("Contact precision (mm)", r"Contact precision \(mm\)\s+↓\s+(.*)"),
                 ("Contact alignment (deg)", r"Contact alignment \(◦ \)\s+↓\s+(.*)"),
                 ("Max penetration (mm)", r"Max hand-object penetration \(mm\)\s+↓\s+(.*)"),
                 ("Frames pen > 2 mm (%)", r"% frames with penetration > 2 mm\s+↓\s+(.*)"),
                 ("Solve time (ms/frame)", r"Solve time \(ms / frame\)\s+↓\s+(.*)")):
    m = re.search(pat, TXT)
    vals = [float(x) for x in m.group(1).split()] if m else None
    t1_found[key] = vals
OUT["table1_text_parse"] = {k: {"parsed": t1_found[k], "equals_transcribed": t1_found[k] == T1[k]} for k in T1}
t1 = {}
for key, v in T1.items():
    base = v[1:]
    avg = float(np.mean(base))
    t1[key] = {
        "values": dict(zip(METHODS, v)), "baseline_mean": avg, "topo_vs_baseline_mean_reduction_pct": 100 * (avg - v[0]) / avg,
        "topo_vs_best_baseline_reduction_pct": (100 * (min(base) - v[0]) / min(base)) if min(base) else None,
        "best_baseline": METHODS[1:][int(np.argmin(base))], "topo_rank": sorted(v).index(v[0]) + 1,
        "baseline_median": float(np.median(base)),
    }
OUT["table1"] = t1
OUT["project_page_claims"] = {
    "precision_reduction_55pct_vs_baseline_avg": t1["Contact precision (mm)"]["topo_vs_baseline_mean_reduction_pct"],
    "penetration_reduction_92pct_vs_baseline_avg": t1["Max penetration (mm)"]["topo_vs_baseline_mean_reduction_pct"],
    "alignment_reduction_vs_baseline_avg": t1["Contact alignment (deg)"]["topo_vs_baseline_mean_reduction_pct"],
}
OUT["solve_time"] = {
    "topo_hz": 1000 / 4.70, "omni_hz": 1000 / 40.96, "omni_over_topo": 40.96 / 4.70,
    "topo_over_geort": 4.70 / 1.17, "topo_over_dexpilot": 4.70 / 1.74, "topo_over_mink": 4.70 / 4.37,
    "budget_ms_at_20hz": 50.0, "budget_ms_at_50hz": 20.0, "budget_ms_at_100hz": 10.0,
    "readme_pen_spin_029": {"frames": 231, "claimed_wall_s": 30, "implied_ms_per_frame_upper": 1000 * 30 / 231},
}

# 침투 비율의 정수 가능성: k/n*100이 소수 둘째 자리까지 같은 값이 되는 n
pct = [0.00, 84.00, 88.00, 96.00]
feas = []
for n in range(1, 301):
    ok = True
    for pv in pct:
        k = round(pv / 100 * n)
        if abs(round(100 * k / n, 2) - pv) > 1e-9:
            ok = False
            break
    if ok:
        feas.append(n)
OUT["penetration_rate_feasible_n_le_300"] = feas
n25 = {m: round(v / 100 * 25) for m, v in zip(METHODS, T1["Frames pen > 2 mm (%)"])}
OUT["penetration_counts_if_n25"] = n25
OUT["penetration_zero_wilson_n25"] = {"upper95_pct": 100 * wilson(0, 25)[1]}
mink_k, topo_k = 21, 0
OUT["penetration_fisher_topo_vs_mink_n25"] = fisher_exact([[0, 25], [21, 4]])[1]
# 최대 침투와 비율의 정합: max <= 2 mm 이면 비율 0, max > 2 mm 이면 비율 > 0
OUT["penetration_max_vs_rate_consistent"] = {
    m: ((mx <= 2.0) == (r == 0.0)) for m, mx, r in zip(METHODS, T1["Max penetration (mm)"], T1["Frames pen > 2 mm (%)"])}

# 프로젝트 페이지의 실로봇 펜 돌리기 5/5 (논문 본문에는 시행 수 없음)
OUT["project_page_real_5of5_wilson95_pct"] = [100 * x for x in wilson(5, 5)]

# ---------------------------------------------------------------- 부록 표 3-6 산술
Nv = 21 + 50
OUT["params"] = {
    "Nv": Nv, "lambda_IM": 500, "lambda_IM_over_Nv": 500 / Nv,
    "samples_per_ppo_iter": 4096 * 40, "stated_samples_per_iter": 163840,
    "control_hz": 1 / (0.01 * 5), "episode_steps": 20 * 20, "minibatch_size": 4096 * 40 / 32,
    "tau_mm": 1.0, "b_mm": 30.0, "slack_max_mm": 30.0 - 1.0,
    "topo_max_pen_minus_tau_mm": 1.07 - 1.0,
    "kappa_weight_at_cm": {str(d): math.exp(-30 * d / 100) for d in (1, 2, 3, 5, 10, 20)},
    "kappa_inverse_cm": 100 / 30,
}
psi = lambda e, s: math.exp(-(e / s) ** 2)
OUT["reward_kernel"] = {
    "object_sigma_0.04": {str(e): psi(e, 0.04) for e in (0.0087, 0.0098, 0.0145, 0.0161, 0.05)},
    "link_sigma_0.025": {str(e): psi(e, 0.025) for e in (0.005, 0.01, 0.02, 0.05)},
    "joint_sigma_0.1_normalized": {str(e): psi(e, 0.1) for e in (0.02, 0.05, 0.1, 0.2)},
    "max_total_reward_per_step_without_smooth": 8.0 + 1.0 + 1.0,
    "reward_weight_share_object": 8.0 / 10.0,
}
OUT["dataset"] = {
    "pen_spin_mean_s": 12.4, "ref_hz": 20, "frames_at_20hz": 12.4 * 20, "frames_at_50hz": 12.4 * 50,
    "episode_20s_covers_mean_clip": 20 > 12.4, "contactpose_grasps_used": 25, "contactpose_total_grasps": 28,
}

# 표 3 값이 원문에 있는지
for name, pat in (("lambda_IM", r"λIM\s+500"), ("kappa", r"κ\s+30"), ("tau", r"τ\s+0\.001 m"), ("b", r"b\s+0\.030 m"),
                  ("ws", r"ws\s+1\.0 × 105"), ("lambda_bone", r"λbone\s+0\.1"), ("lambda_smooth", r"λsmooth\s+2\.5")):
    OUT.setdefault("table3_text_found", {})[name] = bool(re.search(pat, TXT))

# 버전 차이는 별도 파일(paper_v1.txt)이 있을 때만
v1 = ROOT / "paper" / "paper_v1.txt"
if v1.exists():
    import difflib
    a = v1.read_text(encoding="utf-8").splitlines()
    b = TXT.splitlines()
    diff = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    substantive = [l for l in diff if re.search(r"[A-Za-z가-힣]{4,}", l) and "arXiv:2606.16272v" not in l]
    OUT["v1_v2_diff"] = {"changed_lines": len(diff), "substantive_lines": substantive}

(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(OUT, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: OUT[k] for k in ("table1_text_parse", "project_page_claims", "claim_40_6pp", "penetration_rate_feasible_n_le_300",
                                       "penetration_counts_if_n25", "params")}, indent=1, ensure_ascii=False)[:3500])
