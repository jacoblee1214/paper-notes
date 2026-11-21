"""TMA (arXiv:2611.18976 v1) 논문 표·본문 수치를 다시 계산한다. 재현 실험은 하지 않는다.

입력: 논문 Table I(시뮬레이션 신호 ablation), Fig. 6(실로봇 S/G 막대 주석, 페이지 7을 확대해 읽음),
      본문 Sec. V-A/B의 시도 수. 값은 아래 상수에 그대로 옮겼다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py  (numpy, scipy; 수 초)

가정(논문에 없는 것은 표시한다)
- Table I의 "±"는 3개 시드에 대한 표준편차(SD)로 가정한다. SE이면 검정은 더 강해진다. 논문은 밝히지 않는다.
- 시뮬레이션 1셀 = 3 seeds x 4096 episodes = 12288 에피소드. 이항 CI는 이 n으로 계산하고,
  시드 간 변동(±SD)은 별도로 다룬다.
- 실로봇 분모는 방법당 4 knob x 10 trial = 40이다. 실패 원인은 Fig. 7 캡션을 따른다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


# ---------- 1. 실로봇 (Fig. 6, 4 knob x 10 trial) ----------
# (S, G) per knob, 논문 Fig. 6 주석 읽기
REAL = {
    "TMA":      [(8, 9), (6, 8), (6, 7), (7, 8)],
    "RMA":      [(0, 1), (0, 0), (0, 0), (0, 0)],
    "Reactive": [(3, 7), (1, 6), (0, 4), (2, 6)],
}
N_PER_KNOB = 10
real_tot = {m: (sum(s for s, _ in v), sum(g for _, g in v)) for m, v in REAL.items()}
n_tot = 4 * N_PER_KNOB
RES["real_totals"] = {m: {"S": s, "G": g, "n": n_tot} for m, (s, g) in real_tot.items()}
# 자기 일관성: S <= G, 칸당 G <= 10
consistency = all(s <= g <= N_PER_KNOB for v in REAL.values() for s, g in v)
RES["real_consistency_S_le_G_le_10"] = consistency

comps = []
pairs = [("TMA", "RMA"), ("TMA", "Reactive")]
for a, b in pairs:
    for metric, idx in (("S", 0), ("G", 1)):
        k1, k2 = real_tot[a][idx], real_tot[b][idx]
        d, lo, hi = newcombe_diff(k1, n_tot, k2, n_tot)
        comps.append({
            "comparison": f"{a} vs {b} ({metric})",
            "k1": k1, "k2": k2, "n": n_tot,
            "diff_pp": round(100 * d, 1),
            "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
            "fisher_p": float(f"{fisher(k1, n_tot, k2, n_tot):.2e}"),
        })
RES["real_comparisons"] = comps

# 막대 주석의 칸별 S/G 일치 확인: TMA 칸별 S-G 차 분포
RES["real_TMA_grasp_to_door_conversion"] = round(real_tot["TMA"][0] / real_tot["TMA"][1], 3)
RES["real_Reactive_conversion_S_over_G"] = round(real_tot["Reactive"][0] / real_tot["Reactive"][1], 3)

# 칸별 TMA 대 Reactive S 차이 (Fisher, 칸당 n=10)
per_knob = []
for i, knob in enumerate(["Knob 1", "Knob 2", "Knob 3", "Knob 4"]):
    st, rt = REAL["TMA"][i][0], REAL["Reactive"][i][0]
    per_knob.append({"knob": knob, "TMA_S": st, "Reactive_S": rt,
                     "fisher_p": round(fisher(st, 10, rt, 10), 4)})
RES["real_per_knob_S_fisher"] = per_knob

# ---------- 2. 시뮬레이션 신호 ablation (Table I, 3 seeds) ----------
SIM = {
    "TMA (cone, 16-d)":   {"G": (91.3, 1.0), "S": (80.9, 1.4)},
    "Binary contact (4-d)": {"G": (88.4, 1.2), "S": (73.6, 1.4)},
    "Normal load (4-d)":  {"G": (86.5, 1.1), "S": (74.2, 1.5)},
    "Shear only (8-d)":   {"G": (72.8, 1.6), "S": (65.1, 1.7)},
}
N_SIM_PER_SEED, SEEDS = 4096, 3
N_SIM = N_SIM_PER_SEED * SEEDS  # 12288
sim_ci = {}
for name, d in SIM.items():
    row = {}
    for metric in ("G", "S"):
        m, sd = d[metric]
        p = m / 100
        lo, hi = wilson(round(p * N_SIM), N_SIM)
        # 시드 평균의 t 신뢰구간 (n=3, df=2, SD 가정)
        t_crit = stats.t.ppf(0.975, df=SEEDS - 1)
        half = t_crit * sd / math.sqrt(SEEDS)
        row[metric] = {
            "mean": m, "sd_seed": sd,
            "binom_wilson95_pct": [round(100 * lo, 2), round(100 * hi, 2)],
            "seed_t95_pct": [round(m - half, 2), round(m + half, 2)],
        }
    sim_ci[name] = row
RES["sim_ci"] = sim_ci

base = "TMA (cone, 16-d)"
sim_tests = []
for other in ["Binary contact (4-d)", "Normal load (4-d)", "Shear only (8-d)"]:
    for metric in ("G", "S"):
        m1, s1 = SIM[base][metric]
        m2, s2 = SIM[other][metric]
        r = stats.ttest_ind_from_stats(m1, s1, SEEDS, m2, s2, SEEDS, equal_var=False)
        # SE 가정일 때: SD = SE*sqrt(3)
        r_se = stats.ttest_ind_from_stats(m1, s1 * math.sqrt(3), SEEDS, m2, s2 * math.sqrt(3), SEEDS, equal_var=False)
        sim_tests.append({
            "comparison": f"{base} vs {other} ({metric})",
            "diff_pp": round(m1 - m2, 1),
            "welch_t_sd": round(float(r.statistic), 2),
            "welch_p_sd": round(float(r.pvalue), 4),
            "welch_p_if_se": round(float(r_se.pvalue), 4),
        })
RES["sim_tests"] = sim_tests

# 논문 본문 대조: Fig. 5 TMA-TConv 80%, privileged 85%; Table I TMA S = 80.9 (반올림 일치 여부)
RES["fig5_vs_tableI_TMA_S"] = {"fig5_TMA_TConv_pct": 80, "tableI_TMA_S_pct": 80.9, "diff_pp": round(80.9 - 80, 1)}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2))

# ---------- 요약 출력 ----------
print("real totals:", RES["real_totals"])
print("consistency S<=G<=10:", consistency)
for c in comps:
    print(c["comparison"], c["k1"], "vs", c["k2"], "diff", c["diff_pp"], "CI", c["newcombe95_pp"], "p", c["fisher_p"])
print("per-knob S Fisher:", per_knob)
for k, v in sim_ci.items():
    print(k, "G", v["G"]["mean"], v["G"]["binom_wilson95_pct"], v["G"]["seed_t95_pct"], "| S", v["S"]["mean"], v["S"]["binom_wilson95_pct"], v["S"]["seed_t95_pct"])
for t in sim_tests:
    print(t)
