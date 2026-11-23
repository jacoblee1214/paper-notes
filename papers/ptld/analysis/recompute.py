"""PTLD (arXiv:2603.04531 v3) 보고 수치 재계산. 실행: python3 analysis/recompute.py  (numpy, scipy, 수 초)

원자료(시행별 값)는 공개되어 있지 않다. 그래서 다음 세 가지 입력으로만 계산한다.
  (a) 표의 평균±편차 (Table 1–4, 논문 6–8쪽)
  (b) 그림 6 막대 그래프를 400 dpi로 렌더링해 막대 높이를 픽셀로 읽은 값 (그림 판독 값, 오차 약 ±1 눈금 단위)
  (c) 시행 수 n = 10 (그림 6 캡션, Table 4 캡션). 표 1–3의 n은 논문에 없어 계산마다 가정을 밝힌다.
결과는 analysis/recompute_results.json 에 저장한다.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------------------------------------------------------------- V1: Table 4의 "57%"
tbl4 = {"proprio": {"ngoals": (2.1, 2.14), "ttf": (10.99, 10.15)},
        "tactile": {"ngoals": (3.3, 1.55), "ttf": (13.42, 10.18)}}
r_goals = tbl4["tactile"]["ngoals"][0] / tbl4["proprio"]["ngoals"][0] - 1
r_ttf = tbl4["tactile"]["ttf"][0] / tbl4["proprio"]["ttf"][0] - 1
rate_p = tbl4["proprio"]["ngoals"][0] / tbl4["proprio"]["ttf"][0]
rate_t = tbl4["tactile"]["ngoals"][0] / tbl4["tactile"]["ttf"][0]
res["V1_ratio"] = {"goals_gain_pct": round(100 * r_goals, 2), "ttf_gain_pct": round(100 * r_ttf, 2),
                   "goals_per_s_proprio": round(rate_p, 4), "goals_per_s_tactile": round(rate_t, 4),
                   "goal_rate_gain_pct": round(100 * (rate_t / rate_p - 1), 2),
                   "decomposition_check": round((1 + r_ttf) * (rate_t / rate_p), 4)}


# ---------------------------------------------------------------- V2: 정수 시행의 평균±편차 정합성
def feasible(mean, sd, n, vmax=15):
    """0..vmax 정수 n개로 평균(소수 1자리)과 편차(소수 2자리)가 반올림 일치하는 (합, 제곱합) 쌍을 센다.
    동적 계획으로 도달 가능한 (합, 제곱합)을 만든 뒤 ddof 0/1 각각 검사한다."""
    states = {(0, 0)}
    for _ in range(n):
        states = {(s + v, q + v * v) for (s, q) in states for v in range(vmax + 1)}
    hits = {0: [], 1: []}
    for s, q in states:
        if round(s / n, 1) != mean:
            continue
        for ddof in (0, 1):
            var = (q - s * s / n) / (n - ddof)
            if var >= 0 and round(math.sqrt(var), 2) == sd:
                hits[ddof].append((s, q))
    return {"n": n, "pairs_ddof0": len(hits[0]), "pairs_ddof1": len(hits[1]),
            "example_ddof0": hits[0][:1], "example_ddof1": hits[1][:1]}


for name, (m, sdv) in (("proprio", tbl4["proprio"]["ngoals"]), ("tactile", tbl4["tactile"]["ngoals"])):
    res[f"V2_ngoals_{name}"] = {"mean": m, "sd": sdv, "by_n": [feasible(m, sdv, n) for n in (10, 20, 30)],
                                "parity_note": "정수 x에서 x^2 와 x 는 홀짝이 같으므로 합과 제곱합의 홀짝도 같아야 한다"}


# ---------------------------------------------------------------- V3: Table 4 Welch 검정 (n=10, ± 를 SD / SEM으로 가정)
def welch(m1, s1, m2, s2, n1=10, n2=10):
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    t = (m2 - m1) / se
    dfv = se ** 4 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), dfv)
    return round(t, 3), round(dfv, 1), round(p, 4)


def n_per_group(d, power=0.8, alpha=0.05):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return math.ceil(2 * ((za + zb) / d) ** 2)


v3 = {}
for key in ("ngoals", "ttf"):
    m1, s1 = tbl4["proprio"][key]
    m2, s2 = tbl4["tactile"][key]
    t, dfv, p = welch(m1, s1, m2, s2)
    # SEM 가정: SD = 값 * sqrt(10)
    t2, df2, p2 = welch(m1, s1 * math.sqrt(10), m2, s2 * math.sqrt(10))
    pooled = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
    d = (m2 - m1) / pooled
    v3[key] = {"welch_sd_assumed": {"t": t, "df": dfv, "p": p}, "welch_sem_assumed": {"t": t2, "df": df2, "p": p2},
               "cohen_d": round(d, 3), "n_per_group_80pct_power": n_per_group(d)}
# 정규 가정 몬테카를로: 평균 비율의 90% 구간 (SD 가정, 음수는 0으로 절단하지 않음)
rng = np.random.default_rng(0)
a = rng.normal(2.1, 2.14 / math.sqrt(10), 200000)
b = rng.normal(3.3, 1.55 / math.sqrt(10), 200000)
ok = a > 0.2
ratio = b[ok] / a[ok]
v3["ratio_goals_mc_95ci_sd_assumed"] = [round(float(np.percentile(ratio, 2.5)), 2), round(float(np.percentile(ratio, 97.5)), 2)]
res["V3_table4_tests"] = v3

# ---------------------------------------------------------------- V4: 그림 6 판독 값과 "182%"
fig = json.loads((Path(__file__).resolve().parent / "fig6_readout.json").read_text())
cw, ccw = fig["rotation_cw"], fig["rotation_ccw"]
cands = []
for panel, d in (("CW", cw), ("CCW", ccw)):
    for base in ("RMA", "AAC", "PD", "TA"):
        cands.append((f"{panel} 회전, 대 {base}", d["PTLD"] / d[base] - 1))
mean_ptld = (cw["PTLD"] + ccw["PTLD"]) / 2
for base in ("RMA", "AAC", "PD", "TA"):
    mb = (cw[base] + ccw[base]) / 2
    cands.append((f"CW·CCW 평균 회전, 대 {base}", mean_ptld / mb - 1))
base_all_cw = np.mean([cw[b] for b in ("RMA", "AAC", "PD", "TA")])
base_all_ccw = np.mean([ccw[b] for b in ("RMA", "AAC", "PD", "TA")])
cands.append(("CW 회전, 대 기준선 4개 평균", cw["PTLD"] / base_all_cw - 1))
cands.append(("CCW 회전, 대 기준선 4개 평균", ccw["PTLD"] / base_all_ccw - 1))
cands.append(("CW+CCW 합, 대 기준선 4개 평균 합", (cw["PTLD"] + ccw["PTLD"]) / (base_all_cw + base_all_ccw) - 1))
best_cw = max(cw[b] for b in ("RMA", "AAC", "PD", "TA"))
best_ccw = max(ccw[b] for b in ("RMA", "AAC", "PD", "TA"))
cands.append(("CW 회전, 대 최고 기준선", cw["PTLD"] / best_cw - 1))
cands.append(("CCW 회전, 대 최고 기준선", ccw["PTLD"] / best_ccw - 1))
cands.append(("CW+CCW 합, 대 최고 기준선 합", (cw["PTLD"] + ccw["PTLD"]) / (best_cw + best_ccw) - 1))
res["V4_182_candidates_pct"] = {k: round(100 * v, 1) for k, v in cands}
res["V4_near_182"] = [k for k, v in cands if abs(100 * v - 182) <= 6]


# 평균 총 회전 / 평균 낙하 시간 (평균의 비, 시행별 속도의 평균이 아님)
rate = {}
for panel, rk, tk in (("CW", "rotation_cw", "ttf_cw"), ("CCW", "rotation_ccw", "ttf_ccw")):
    rate[panel] = {n: round(fig[rk][n] / fig[tk][n], 2) for n in ("RMA", "AAC", "PD", "TA", "PTLD")}
res["V4b_rotation_per_second_ratio_of_means"] = rate
res["V4b_ttf_over_cap"] = {"PTLD_CW_TTF_over_120": round(fig["ttf_cw"]["PTLD"] / 120, 3),
                           "PTLD_CCW_TTF_over_120": round(fig["ttf_ccw"]["PTLD"] / 120, 3)}
res["V4c_pd_vs_aac"] = {"CW rotation PD/AAC": round(fig["rotation_cw"]["PD"] / fig["rotation_cw"]["AAC"], 2),
                        "CCW rotation PD/AAC": round(fig["rotation_ccw"]["PD"] / fig["rotation_ccw"]["AAC"], 2),
                        "CW rotation PTLD/PD": round(fig["rotation_cw"]["PTLD"] / fig["rotation_cw"]["PD"], 1),
                        "CCW rotation PTLD/PD": round(fig["rotation_ccw"]["PTLD"] / fig["rotation_ccw"]["PD"], 1)}

# ---------------------------------------------------------------- V5: 그림 6 PTLD 대 AAC 유의성 (오차 막대 정의가 없어 세 가지 해석)
errs = fig["errorbar_halfwidth"]
v5 = {}
for tag in ("rotation_cw", "rotation_ccw", "ttf_cw", "ttf_ccw"):
    mP, hP = fig[tag]["PTLD"], errs[tag]["PTLD"]
    mA, hA = fig[tag]["AAC"], errs[tag]["AAC"]
    row = {}
    # (i) 막대 반폭 = SD, (ii) SEM, (iii) 95% CI (t_9)
    t9 = stats.t.ppf(0.975, 9)
    for name, sP, sA in (("SD", hP / math.sqrt(10), hA / math.sqrt(10)), ("SEM", hP, hA), ("CI95", hP / t9, hA / t9)):
        z = (mP - mA) / math.sqrt(sP ** 2 + sA ** 2)
        row[name] = {"z": round(z, 2), "p_approx": round(2 * stats.norm.sf(abs(z)), 4)}
    v5[tag] = {"PTLD": mP, "AAC": mA, "half_width_PTLD": hP, "half_width_AAC": hA, **row}
res["V5_fig6_vs_AAC"] = v5

# ---------------------------------------------------------------- V6: 시뮬 Table 1
t1 = {"Oracle": (159.4, 0.89, 31.86), "RMA_prop": (139.0, 0.79, 28.53), "RMA_pose": (153.1, 0.86, 31.09),
      "AAC_prop": (141.9, 0.80, 27.83), "AAC_pose": (168.6, 0.88, 26.5)}
v1_extra = {"AAC_pose_nodistill_v1_v2": (161.48, 0.85, 29.74)}
rel = lambda a, b: round(100 * (a / b - 1), 1)
res["V6_table1"] = {
    "RMA pose vs prop (RotR, TTF, RotP) %": [rel(t1["RMA_pose"][i], t1["RMA_prop"][i]) for i in range(3)],
    "AAC pose vs prop %": [rel(t1["AAC_pose"][i], t1["AAC_prop"][i]) for i in range(3)],
    "AAC vs RMA (prop) %": [rel(t1["AAC_prop"][i], t1["RMA_prop"][i]) for i in range(3)],
    "AAC vs RMA (pose) %": [rel(t1["AAC_pose"][i], t1["RMA_pose"][i]) for i in range(3)],
    "AAC pose vs Oracle %": [rel(t1["AAC_pose"][i], t1["Oracle"][i]) for i in range(3)],
    "v1·v2 전용 행: AAC(no distillation)+Pose 대비 AAC+Pose (RotR, TTF, RotP) %": [rel(t1["AAC_pose"][i], v1_extra["AAC_pose_nodistill_v1_v2"][i]) for i in range(3)],
    "note": "RotP는 ↓(낮을수록 좋음). RMA에서 Pose를 더하면 RotP는 28.53 → 31.09로 올라간다(표의 화살표 기준 악화).",
}

# ---------------------------------------------------------------- V7: Table 2 (회전 오차) - n 가정
t2 = {"abs_prop": (0.43, 0.11), "abs_tac": (0.21, 0.03), "rel_prop": (0.95, 0.09), "rel_tac": (0.26, 0.05)}
v7 = {"abs_reduction_pct": round(100 * (1 - 0.21 / 0.43), 1), "rel_reduction_pct": round(100 * (1 - 0.26 / 0.95), 1),
      "ratio_abs": round(0.43 / 0.21, 2), "ratio_rel": round(0.95 / 0.26, 2)}
for n in (5, 10, 30):
    v7[f"welch_abs_n{n}_sd"] = welch(*t2["abs_prop"], *t2["abs_tac"], n1=n, n2=n)
    v7[f"welch_rel_n{n}_sd"] = welch(*t2["rel_prop"], *t2["rel_tac"], n1=n, n2=n)
# 상대 자세 오차가 절대보다 작으면 안 되는 것은 아님: 촉각 모델에서 두 값 차이
v7["tactile_rel_minus_abs"] = round(0.26 - 0.21, 2)
res["V7_table2"] = v7

# ---------------------------------------------------------------- V8: Table 3 (정육면체) - 교사 대비
t3 = {"AAC": {"RotR": (17.3, 8.8), "TTF": (0.33, 0.15), "RotP": (5.5, 3.6)},
      "Proprio": {"RotR": (0.5, 0.5), "TTF": (0.11, 0.03), "RotP": (1.3, 0.9)},
      "Tactile": {"RotR": (4.5, 2.3), "TTF": (0.34, 0.21), "RotP": (50.8, 28.1)}}
v8 = {}
for k in ("RotR", "TTF", "RotP"):
    for other in ("AAC", "Proprio"):
        t, dfv, p = welch(*t3[other][k], *t3["Tactile"][k])
        v8[f"Tactile vs {other} {k}"] = {"ratio": round(t3["Tactile"][k][0] / t3[other][k][0], 2), "t": t, "df": dfv, "p_sd_n10": p}
res["V8_table3_cube"] = v8

# ---------------------------------------------------------------- V9: 단위·산술 점검
res["V9_arith"] = {
    "0.25 rad in deg": round(math.degrees(0.25), 2),
    "20 deg in rad": round(math.radians(20), 3), "30 deg in rad": round(math.radians(30), 3),
    "DAgger 데이터 분량 (30 궤적 x 2분 x 3회) 분": 30 * 2 * 3,
    "30 inference steps at 20 Hz s": 30 / 20, "100 steps at 20 Hz s": 100 / 20,
    "4.5 s at 20 Hz steps": 4.5 * 20, "4.5 s / 30 steps = s per step": 4.5 / 30,
    "30-step history at 20 Hz s": 30 / 20, "0.5 s history at 100 Hz samples": 0.5 * 100,
    "368 taxel / 18 pad": round(368 / 18, 2),
    "그림 6 캡션의 124 s, 220 s vs 그림 6 막대의 최대 에피소드 120 s": "124 > 120, 220 > 120",
    "Fig.6 PTLD CW TTF 판독 114.5 s vs 최대 에피소드 120 s 비율": round(114.5 / 120, 3),
    "teacher AAC 대비 Tactile 정육면체 RotR 비": round(4.5 / 17.3, 3),
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
