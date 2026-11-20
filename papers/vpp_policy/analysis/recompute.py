"""VPP (arXiv:2412.14803 v2) 보고 수치 재계산과 검정.

실행: python3 analysis/recompute.py   (수 초, numpy·scipy)
출력: analysis/recompute_results.json 과 표준출력.

입력은 논문 v2 PDF에서 읽은 값이다(표 번호 주석 참조). 가정은 결과 JSON의 "assumptions"에 적었다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
RES = {"assumptions": {}, "checks": {}}

# ---------------------------------------------------------------- 표 1 (CALVIN ABC→D, v2)
# 열: 과제 1..5의 누적 성공률. 방법별 행.
T1 = {
    "RT-1": [0.533, 0.222, 0.094, 0.038, 0.013],
    "Diffusion Policy": [0.402, 0.123, 0.026, 0.008, 0.000],
    "RoboFlamingo": [0.824, 0.619, 0.466, 0.331, 0.235],
    "UniPi": [0.560, 0.160, 0.080, 0.080, 0.040],
    "MDT": [0.631, 0.429, 0.247, 0.151, 0.091],
    "Susie": [0.870, 0.690, 0.490, 0.380, 0.260],
    "GR-1": [0.854, 0.712, 0.596, 0.497, 0.401],
    "Vidman": [0.915, 0.764, 0.682, 0.592, 0.467],
    "RoboUniview": [0.942, 0.842, 0.734, 0.622, 0.507],
    "VPP": [0.965, 0.909, 0.866, 0.820, 0.769],
}
T1_REPORTED = {"RT-1": 0.90, "Diffusion Policy": 0.56, "RoboFlamingo": 2.47, "UniPi": 0.92, "MDT": 1.55,
               "Susie": 2.69, "GR-1": 3.06, "Vidman": 3.42, "RoboUniview": 3.65, "VPP": 4.33}
N_CHAINS = 1000  # official code: policy_conf/calvin_evaluate.yaml num_sequences: 1000 (논문 본문에는 없음)
RES["assumptions"]["calvin_chains"] = N_CHAINS


def avg_len_se(p, n=N_CHAINS):
    """평균 완료 과제 수의 표준오차. 누적 지시자 I_j = 1[K>=j]는 중첩되어 Cov = min(p_j,p_k) - p_j p_k."""
    p = np.asarray(p)
    cov = np.minimum.outer(p, p) - np.outer(p, p)
    return math.sqrt(cov.sum() / n)


calvin = {}
for m, p in T1.items():
    calvin[m] = {"avg_len_recomputed": round(sum(p), 3), "avg_len_reported": T1_REPORTED[m],
                 "se_chains": round(avg_len_se(p), 4)}
RES["calvin_table1"] = calvin

v = T1["VPP"]
gains = {}
for base, claimed in [("GR-1", "41.5% (intro, project page)"), ("RoboUniview", "18.6% (v2 abstract)"),
                      ("Vidman", "-")]:
    b = sum(T1[base])
    gains[base] = {"relative_gain_pct": round(100 * (sum(v) / b - 1), 1), "claimed": claimed}
gains["3.35 (v2 text, Sec 5.1)"] = {"relative_gain_pct": round(100 * (sum(v) / 3.35 - 1), 1),
                                    "claimed": "text: 'from 3.35 to 4.33'; 3.35 is not a row in v2 Table 1"}
RES["calvin_gains"] = gains

# VPP 대 RoboUniview: 독립 평가 두 개로 가정한 z 검정 (시드 변동은 포함하지 않음)
d = sum(v) - sum(T1["RoboUniview"])
se_d = math.hypot(avg_len_se(v), avg_len_se(T1["RoboUniview"]))
RES["calvin_vpp_vs_roboUniview"] = {"diff": round(d, 3), "se_chain_only": round(se_d, 4),
                                    "z": round(d / se_d, 1),
                                    "note": "채점 표집 오차만 반영. 학습 시드 변동은 논문에 없어 반영 불가"}

# 표 12 절제: 시간적 어텐션 제거(4.18), 2단계 노이즈 제거(4.19), 단일 시점(3.58)
T12 = {"VPP": [0.965, 0.909, 0.866, 0.820, 0.769],
       "Ablation1 (no temporal attn)": [0.949, 0.900, 0.839, 0.780, 0.714],
       "Ablation2 (2-step denoise)": [0.951, 0.904, 0.840, 0.777, 0.718],
       "Single-view": [0.909, 0.815, 0.713, 0.620, 0.518]}
abl = {}
for k, p in T12.items():
    abl[k] = {"avg_len": round(sum(p), 3), "se_chain_only": round(avg_len_se(p), 4)}
abl["z_VPP_vs_Ablation1"] = round((sum(T12["VPP"]) - sum(T12["Ablation1 (no temporal attn)"])) /
                                  math.hypot(avg_len_se(T12["VPP"]), avg_len_se(T12["Ablation1 (no temporal attn)"])), 2)
RES["calvin_table12"] = abl

# 10% 데이터: 0.878 0.746 0.632 0.540 0.453 → 3.25. 논문 주장: 전체 데이터를 쓴 관련 방법을 넘는다.
p10 = [0.878, 0.746, 0.632, 0.540, 0.453]
RES["calvin_10pct"] = {"avg_len": round(sum(p10), 3),
                       "beats_full_data_methods": [m for m in T1 if sum(T1[m]) < sum(p10) and m != "VPP"],
                       "does_not_beat": [m for m in T1 if sum(T1[m]) > sum(p10)]}

# ---------------------------------------------------------------- 표 2 (MetaWorld 50 tasks)
# 열: RT-1, DP, Susie, GR-1, VPP. 행: 난이도 구간(수: 28, 11, 11)
T2 = {"RT-1": (0.605, 0.042, 0.015), "Diffusion Policy": (0.442, 0.062, 0.095),
      "Susie": (0.560, 0.196, 0.255), "GR-1": (0.725, 0.327, 0.451), "VPP": (0.818, 0.493, 0.526)}
T2_REP = {"RT-1": 0.346, "Diffusion Policy": 0.279, "Susie": 0.410, "GR-1": 0.574, "VPP": 0.682}
w = np.array([28, 11, 11])
mw = {}
for m, t in T2.items():
    mw[m] = {"recomputed": round(float(np.dot(w, t) / w.sum()), 4), "reported": T2_REP[m]}
RES["metaworld_table2"] = mw
RES["metaworld_gain"] = {"absolute_pp": round(100 * (T2_REP["VPP"] - T2_REP["GR-1"]), 1),
                         "relative_pct": round(100 * (T2_REP["VPP"] / T2_REP["GR-1"] - 1), 1),
                         "claimed": "10.8% (text: 'by 10.8% in average success rate'; 절대 %p로 읽힘)"}

# ---------------------------------------------------------------- 실로봇 (표 6 Panda, 표 7 Dexterous hand)
# 각 범주의 시도 수와 성공률. 성공 수 = 성공률 × 시도 수가 정수인지 확인한다.
PANDA_SEEN_N = [50, 50, 20, 20, 20]
PANDA_UNSEEN_N = [25, 25, 10, 10, 10]
PANDA_SEEN = {"Diffusion Policy": [.36, .40, .65, .40, .45], "Susie": [.56, .42, .90, .55, .60],
              "GR-1": [.52, .38, .80, .50, .60], "VPP": [.90, .86, .85, .75, .85]}
PANDA_UNSEEN = {"Diffusion Policy": [.24, .12, .50, .20, .40], "Susie": [.40, .44, .60, .50, .50],
                "GR-1": [.32, .32, .60, .50, .40], "VPP": [.80, .72, .80, .70, .60]}
PANDA_REP = {"seen": {"Diffusion Policy": .425, "Susie": .563, "GR-1": .519, "VPP": .856},
             "unseen": {"Diffusion Policy": .250, "Susie": .463, "GR-1": .388, "VPP": .737}}

DEX_SEEN_N = [100, 100, 25, 25, 25, 25, 25, 25, 25]
DEX_UNSEEN_N = [50, 50, 20, 20, 20, 20, 20, 20, 20]
DEX_SEEN = {"Diffusion Policy": [.38, .35, 0, .28, 0, .040, .68, 0, .40],
            "Susie": [.61, .55, 0, .44, .08, 0, .96, 0, .64],
            "GR-1": [.48, .40, 0, .16, 0, 0, .64, 0, .48],
            "VPP": [.83, .79, .64, .80, .64, .48, .96, .52, .72]}
DEX_UNSEEN = {"Diffusion Policy": [.12, .08, 0, .12, 0, 0, .44, 0, .28],
              "Susie": [.42, .32, 0, .32, 0, 0, .76, 0, .44],
              "GR-1": [.26, .20, 0, .12, 0, 0, .40, 0, .24],
              "VPP": [.75, .68, .40, .76, .56, .32, .88, .20, .56]}
DEX_TOOL_N = [10, 10, 10, 10]
DEX_TOOL = {"Diffusion Policy": [0, .2, 0, 0], "Susie": [.4, .2, .1, 0],
            "GR-1": [.3, .1, .2, 0], "VPP": [.9, .6, .8, .4]}
DEX_REP = {"seen": {"Diffusion Policy": .287, "Susie": .450, "GR-1": .319, "VPP": .749},
           "unseen": {"Diffusion Policy": .110, "Susie": .328, "GR-1": .159, "VPP": .605},
           "tool": {"Diffusion Policy": .05, "Susie": .23, "GR-1": .15, "VPP": .68}}


def pooled(rates, ns):
    """범주별 성공률과 시도 수로 합산 성공 수와 전체 시도 수를 구한다."""
    k = [round(r * n) for r, n in zip(rates, ns)]
    bad = [(r, n, r * n) for r, n in zip(rates, ns) if abs(r * n - round(r * n)) > 1e-6]
    return sum(k), sum(ns), bad


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 3), round(c + h, 3)


real = {}
for setting, data, ns, rep in [("Panda seen", PANDA_SEEN, PANDA_SEEN_N, PANDA_REP["seen"]),
                               ("Panda unseen", PANDA_UNSEEN, PANDA_UNSEEN_N, PANDA_REP["unseen"]),
                               ("Dex seen", DEX_SEEN, DEX_SEEN_N, DEX_REP["seen"]),
                               ("Dex unseen", DEX_UNSEEN, DEX_UNSEEN_N, DEX_REP["unseen"]),
                               ("Dex tool", DEX_TOOL, DEX_TOOL_N, DEX_REP["tool"])]:
    rows = {}
    for m, rates in data.items():
        k, n, bad = pooled(rates, ns)
        rows[m] = {"successes": k, "trials": n, "rate_recomputed": round(k / n, 4),
                   "rate_reported": rep[m], "non_integer_counts": bad,
                   "wilson95": wilson(k, n)}
    real[setting] = rows

    # VPP 대 가장 강한 기준선(같은 설정에서 보고 값이 가장 높은 것)
    best = max((m for m in rows if m != "VPP"), key=lambda m: rows[m]["successes"] / rows[m]["trials"])
    kv, nv = rows["VPP"]["successes"], rows["VPP"]["trials"]
    kb, nb = rows[best]["successes"], rows[best]["trials"]
    _, pval = stats.fisher_exact([[kv, nv - kv], [kb, nb - kb]])
    real[setting]["_vs_best"] = {"baseline": best,
                                 "abs_diff_pp": round(100 * (kv / nv - kb / nb), 1),
                                 "rel_gain_pct": round(100 * ((kv / nv) / (kb / nb) - 1), 1),
                                 "fisher_p": float(f"{pval:.2e}")}
RES["real_world"] = real

# 표 5와 표 7 사이의 불일치 점검 (표 5는 표 6·7의 범주 값에서 가져왔다고 적혀 있음)
RES["checks"]["dex_unseen_susie"] = {"table7_reported": 0.328,
                                    "recomputed": real["Dex unseen"]["Susie"]["rate_recomputed"],
                                    "table5_reported": 0.28}
RES["checks"]["dex_tool_susie"] = {"table7_reported": 0.23,
                                  "recomputed_equal_weight": round(float(np.mean(DEX_TOOL["Susie"])), 3),
                                  "table5_reported": 0.23}

# 31.6% (초록, v2) 재현 시도: 여러 분모를 시험하고 가장 가까운 값을 적는다
cands = {}
vpp_real = [real[s]["VPP"]["rate_recomputed"] for s in real]
for s in real:
    b = real[s]["_vs_best"]["baseline"]
    cands[f"{s} vs best"] = real[s]["_vs_best"]["rel_gain_pct"]
pooled_vpp = np.mean([real[s]["VPP"]["rate_recomputed"] for s in real])
pooled_best = np.mean([real[s][real[s]["_vs_best"]["baseline"]]["rate_recomputed"] for s in real])
cands["mean over 5 settings, VPP vs per-setting best (rel)"] = round(100 * (pooled_vpp / pooled_best - 1), 1)
cands["mean over 5 settings, abs diff (pp)"] = round(100 * (pooled_vpp - pooled_best), 1)
RES["checks"]["claim_31_6pct"] = {"claimed": 31.6, "candidates": cands,
                                 "verdict": "표 5–7 값에서 31.6%에 해당하는 계산을 찾지 못함"}

# ---------------------------------------------------------------- 출력
json.dump(RES, OUT.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("CALVIN avg len (recomputed vs reported, SE over 1000 chains):")
for m, r in calvin.items():
    print(f"  {m:18s} {r['avg_len_recomputed']:.3f}  rep {r['avg_len_reported']:.2f}  se {r['se_chains']:.4f}")
print("gains:", json.dumps(gains, ensure_ascii=False))
print("VPP vs RoboUniview:", RES["calvin_vpp_vs_roboUniview"])
print("Table12:", {k: v for k, v in abl.items() if not k.startswith("z")}, abl["z_VPP_vs_Ablation1"])
print("10%:", RES["calvin_10pct"])
print("MetaWorld:", mw, RES["metaworld_gain"])
for s, rows in real.items():
    print(s, {m: (r["rate_recomputed"], r["rate_reported"]) for m, r in rows.items() if not m.startswith("_")},
          rows["_vs_best"])
print("checks:", json.dumps(RES["checks"], ensure_ascii=False))
