"""TactileStep (arXiv:2609.28959 v1) 표 재계산과 일관성 검사 (numpy, scipy). 실행: python3 analysis/recompute.py
결과: analysis/recompute_results.json
표 값은 paper/paper.txt(pdftotext -layout)에서, 그림 6의 값은 그림에 적힌 숫자를 눈으로 읽어 옮겼다(그림 판독 값).
원자료가 없으므로 모든 검사는 '보고된 값끼리' 맞는지를 본다. 논문에 없는 시행 수는 가정으로 표시한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}
N_HW = 20          # 4.1절 "Hardware evaluation uses 20 samples per condition" (표 3의 ±는 정의 없음: SD로 가정)
N_SIM = 4096       # 4.1절, 부록 A.1


def welch(m1, s1, n1, m2, s2, n2):
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    t = (m1 - m2) / se
    df = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    d = (m1 - m2) / math.sqrt((s1 ** 2 + s2 ** 2) / 2)
    return t, df, p, d, se


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ---------------------------------------------------------------- 표 3 (실로봇)
# terrain: {metric: {policy: (mean, sd)}}  policy = base(Hiking), wo(w/o tac. obs.), ts(TactileStep)
T3 = {
    "Stair up":      {"F": {"base": (499.9, 40.7), "wo": (375.2, 47.3), "ts": (349.8, 32.5)},
                      "L": {"base": (101.2, 3.7), "wo": (85.1, 3.4), "ts": (72.2, 2.8)},
                      "A": {"base": (0.470, 0.013), "wo": (0.421, 0.028), "ts": (0.482, 0.016)}},
    "Stair down":    {"F": {"base": (359.3, 23.2), "wo": (353.3, 38.6), "ts": (344.1, 20.1)},
                      "L": {"base": (97.2, 1.2), "wo": (84.9, 3.9), "ts": (67.1, 1.4)},
                      "A": {"base": (0.483, 0.020), "wo": (0.417, 0.044), "ts": (0.598, 0.027)}},
    "Platform up":   {"F": {"base": (695.0, 49.4), "wo": (371.5, 50.5), "ts": (355.7, 39.8)},
                      "L": {"base": (110.1, 6.0), "wo": (98.2, 5.3), "ts": (83.0, 4.1)}},
    "Platform down": {"F": {"base": (608.9, 38.6), "wo": (445.7, 54.1), "ts": (404.6, 43.2)},
                      "L": {"base": (90.8, 5.2), "wo": (87.5, 4.4), "ts": (75.9, 3.6)}},
    "Flat":          {"F": {"base": (201.3, 23.3), "wo": (202.1, 32.4), "ts": (191.3, 17.7)},
                      "L": {"base": (90.6, 2.5), "wo": (85.9, 2.9), "ts": (66.5, 1.6)},
                      "A": {"base": (0.453, 0.011), "wo": (0.362, 0.035), "ts": (0.510, 0.014)}},
    "Slope":         {"F": {"base": (255.0, 34.9), "wo": (204.7, 42.1), "ts": (202.0, 31.0)},
                      "L": {"base": (83.4, 1.0), "wo": (77.1, 2.0), "ts": (69.9, 1.1)},
                      "A": {"base": (0.486, 0.026), "wo": (0.490, 0.033), "ts": (0.499, 0.030)}},
}

# (1) 초록의 헤드라인 수치 재계산
heads = {}
red = {}
for terr, d in T3.items():
    b, t = d["F"]["base"][0], d["F"]["ts"][0]
    red[terr] = {"F_reduction_pct": round((b - t) / b * 100, 1),
                 "L_reduction_dB": round(d["L"]["base"][0] - d["L"]["ts"][0], 1)}
    if "A" in d:
        red[terr]["A_gain_pct"] = round((d["A"]["ts"][0] - d["A"]["base"][0]) / d["A"]["base"][0] * 100, 1)
heads["per_terrain"] = red
heads["max_F_reduction_pct"] = max(v["F_reduction_pct"] for v in red.values())
heads["max_L_reduction_dB"] = max(v["L_reduction_dB"] for v in red.values())
heads["max_A_gain_pct"] = max(v["A_gain_pct"] for v in red.values() if "A_gain_pct" in v)
heads["abstract_claims"] = {"F": 48.8, "L_dB": 30.1, "A": 23.8}
heads["abstract_match"] = (heads["max_F_reduction_pct"] == 48.8 and heads["max_L_reduction_dB"] == 30.1
                           and heads["max_A_gain_pct"] == 23.8)
res["table3_headline"] = heads

# (2) Welch 검정 (n = 20 가정, ± = SD 가정) + Holm (표 3 전체 비교 한 묶음)
rows = []
for terr, d in T3.items():
    for met, pol in d.items():
        for a, b in (("ts", "base"), ("ts", "wo"), ("wo", "base")):
            (m1, s1), (m2, s2) = pol[a], pol[b]
            t, df, p, dd, se = welch(m1, s1, N_HW, m2, s2, N_HW)
            half = stats.t.ppf(0.975, df) * se
            rows.append({"terrain": terr, "metric": met, "contrast": f"{a} vs {b}", "diff": round(m1 - m2, 4),
                         "ci95": [round(m1 - m2 - half, 4), round(m1 - m2 + half, 4)], "t": round(t, 2),
                         "df": round(df, 1), "p": p, "cohen_d": round(dd, 2)})
adj = holm([r["p"] for r in rows])
for r, a in zip(rows, adj):
    r["p_holm"] = a
res["table3_welch_assuming_n20_sd"] = rows


def pick(terr, met, contrast):
    return next(r for r in rows if r["terrain"] == terr and r["metric"] == met and r["contrast"] == contrast)


# ± 가 95% CI 반폭이었다면(SD = 반폭·sqrt(n)/t_{.975,19}) 결과가 어떻게 바뀌는가: 핵심 대비만
k = math.sqrt(N_HW) / stats.t.ppf(0.975, N_HW - 1)
alt = []
for terr, met, a, b in [("Stair down", "F", "ts", "base"), ("Flat", "F", "ts", "base"), ("Flat", "F", "ts", "wo"),
                        ("Stair up", "F", "ts", "wo"), ("Platform up", "F", "ts", "wo"), ("Platform down", "F", "ts", "wo"),
                        ("Slope", "A", "ts", "base")]:
    (m1, s1), (m2, s2) = T3[terr][met][a], T3[terr][met][b]
    t, df, p, _, _ = welch(m1, s1 * k, N_HW, m2, s2 * k, N_HW)
    alt.append({"terrain": terr, "metric": met, "contrast": f"{a} vs {b}", "p_if_pm_is_95ci_halfwidth": p})
res["table3_welch_if_pm_is_ci_halfwidth"] = alt

# (3) 개선분을 보상(w/o tac obs)과 촉각 관측(ts - wo)으로 나누기
split = {}
for terr, d in T3.items():
    s = {}
    for met in d:
        b, w, t = d[met]["base"][0], d[met]["wo"][0], d[met]["ts"][0]
        tot = b - t if met in ("F", "L") else t - b
        part_wo = (b - w) if met in ("F", "L") else (w - b)       # 촉각 관측 없이 얻은 개선
        s[met] = {"total_gain": round(tot, 3), "gain_without_tactile_obs": round(part_wo, 3),
                  "share_without_obs_pct": round(part_wo / tot * 100, 1) if tot else None}
    split[terr] = s
res["table3_gain_split"] = split

# (4) 힘과 소음의 일치 (3.2? 4.3절 "The agreement between force and sound confirms ...")
F_all, L_all, lab = [], [], []
for terr, d in T3.items():
    for pol in ("base", "wo", "ts"):
        F_all.append(d["F"][pol][0]); L_all.append(d["L"][pol][0]); lab.append((terr, pol))
F_all, L_all = np.array(F_all), np.array(L_all)
r_p, p_p = stats.pearsonr(F_all, L_all)
r_s, p_s = stats.spearmanr(F_all, L_all)
b_idx = [i for i, (t, p) in enumerate(lab) if p == "base"]
rb, pb = stats.pearsonr(F_all[b_idx], L_all[b_idx])
within = {}
for terr, d in T3.items():
    within[terr] = {"dF_ts_vs_base_pct": round((d["F"]["ts"][0] / d["F"]["base"][0] - 1) * 100, 1),
                    "dL_ts_vs_base_dB": round(d["L"]["ts"][0] - d["L"]["base"][0], 1),
                    "dF_ts_vs_wo_pct": round((d["F"]["ts"][0] / d["F"]["wo"][0] - 1) * 100, 1),
                    "dL_ts_vs_wo_dB": round(d["L"]["ts"][0] - d["L"]["wo"][0], 1)}
res["force_vs_noise"] = {"pearson_all18": [round(r_p, 3), p_p], "spearman_all18": [round(r_s, 3), p_s],
                         "pearson_baseline6": [round(rb, 3), pb], "within_terrain": within,
                         "note": "단위가 다른 조건 18점(지형 6 x 정책 3). 표 3의 평균만 사용하므로 시행 단위 상관이 아님"}

# (5) 부록 C.3: 100 Hz 대 25 Hz
c3 = {}
for name, m100, s100, m25, s25 in [("Flat", 199.2, 14.9, 191.3, 17.7), ("Platform down", 427.7, 20.1, 404.6, 43.2)]:
    t, df, p, d, se = welch(m100, s100, N_HW, m25, s25, N_HW)
    c3[name] = {"mean_100Hz": m100, "mean_25Hz": m25, "ratio_100_over_25": round(m100 / m25, 3),
                "welch_p_n20_assumed": p, "n_for_100Hz_is_stated": False}
res["appendix_C3_sampling"] = c3

# (6) 시뮬 표 2: 성공률의 정수 가능성과 Fisher
T2_SR = {"Stair up": (100.0, 99.98), "Stair down": (100.0, 99.98), "Platform up": (100.0, 99.93),
         "Platform down": (100.0, 92.26), "Flat": (99.98, 99.19), "Slope": (99.63, 99.63)}
sr_rows = []
for terr, (ts_, bs_) in T2_SR.items():
    out = {"terrain": terr}
    for key, v in (("ts", ts_), ("base", bs_)):
        fails = (100 - v) / 100 * N_SIM
        k = round(fails)
        out[key + "_failures_int"] = k
        out[key + "_integer_ok_at_2dp"] = bool(abs(round((1 - k / N_SIM) * 100, 2) - v) < 1e-9)
    a, b = out["ts_failures_int"], out["base_failures_int"]
    table = [[N_SIM - a, a], [N_SIM - b, b]]
    out["fisher_p_n4096"] = stats.fisher_exact(table)[1]
    # 4096 에피소드가 5 x 10 = 50개 지형 칸에 걸쳐 있다(부록 A.1): 칸 단위 독립이면 유효 표본은 50 수준
    out["episodes_per_terrain_cell"] = round(N_SIM / 50, 1)
    sr_rows.append(out)
res["table2_success_rate"] = sr_rows

# (7) 표 2 에너지·시간·속도 오차
T2 = {  # terrain: (RMSE ts, sd, base, sd), (time ts, sd, base, sd), (energy ts, sd, base, sd)
    "Stair up": ((0.24, 0.02, 0.21, 0.02), (5.7, 0.4, 5.6, 0.3), (1126.3, 66.6, 802.2, 97.0)),
    "Stair down": ((0.20, 0.02, 0.19, 0.03), (6.1, 0.6, 6.0, 0.5), (1232.9, 91.0, 887.4, 110.4)),
    "Platform up": ((0.24, 0.02, 0.23, 0.03), (4.9, 0.5, 4.9, 0.6), (939.8, 50.5, 867.1, 72.8)),
    "Platform down": ((0.22, 0.02, 0.21, 0.15), (4.9, 0.6, 4.9, 0.8), (1033.0, 73.6, 1005.3, 119.7)),
    "Flat": ((0.20, 0.08, 0.17, 0.03), (6.5, 3.3, 5.6, 2.2), (834.4, 165.4, 644.5, 76.1)),
    "Slope": ((0.21, 0.17, 0.19, 0.05), (6.0, 2.8, 5.2, 2.1), (995.7, 126.1, 786.3, 47.9)),
}
t2 = {}
for terr, (rm, tm, en) in T2.items():
    d_en = en[0] / en[2] - 1
    d_tm = tm[0] / tm[2] - 1
    d_rm = rm[0] / rm[2] - 1
    # 표준화 효과크기(SD가 시행 간 변동이라는 가정)
    d_en_std = (en[0] - en[2]) / math.sqrt((en[1] ** 2 + en[3] ** 2) / 2)
    t2[terr] = {"energy_pct": round(d_en * 100, 1), "energy_cohen_d": round(d_en_std, 2),
                "time_pct": round(d_tm * 100, 1), "rmse_pct": round(d_rm * 100, 1)}
res["table2_cost"] = t2
res["table2_cost"]["_mean_energy_pct"] = round(float(np.mean([v["energy_pct"] for k, v in t2.items()])), 1)
res["command_range_note"] = {"forward_cmd_m_s": [0.45, 0.55], "velocity_rmse_m_s": [0.19, 0.24],
                             "rmse_over_cmd_pct": [round(0.17 / 0.5 * 100), round(0.24 / 0.5 * 100)],
                             "source": "부록 표 9 Command, 표 2 (RMSE 0.17-0.24, 지형 6개 평균 아님, 정의 미기재)"}


# ---------------------------------------------------------------- 시뮬 대 실로봇 (같은 지형, 같은 정책)
SIM_F = {"Stair up": (304.45, 450.00), "Stair down": (320.74, 599.44), "Platform up": (310.73, 475.81),
         "Platform down": (310.62, 515.46), "Flat": (278.16, 464.44), "Slope": (297.32, 440.65)}   # (ts, base) 그림 6 판독
SIM_A = {"Stair up": (0.536, 0.480), "Stair down": (0.545, 0.487), "Flat": (0.929, 0.912), "Slope": (0.805, 0.804)}  # 표 1
svr = {}
for terr in T3:
    row = {"F_sim_ts": SIM_F[terr][0], "F_real_ts": T3[terr]["F"]["ts"][0],
           "F_sim_base": SIM_F[terr][1], "F_real_base": T3[terr]["F"]["base"][0],
           "F_real_over_sim_ts": round(T3[terr]["F"]["ts"][0] / SIM_F[terr][0], 2),
           "F_real_over_sim_base": round(T3[terr]["F"]["base"][0] / SIM_F[terr][1], 2)}
    if terr in SIM_A:
        row.update({"A_sim_ts": SIM_A[terr][0], "A_real_ts": T3[terr]["A"]["ts"][0],
                    "A_sim_base": SIM_A[terr][1], "A_real_base": T3[terr]["A"]["base"][0],
                    "A_real_over_sim_ts": round(T3[terr]["A"]["ts"][0] / SIM_A[terr][0], 2),
                    "A_real_over_sim_base": round(T3[terr]["A"]["base"][0] / SIM_A[terr][1], 2)})
    svr[terr] = row
res["sim_vs_real"] = svr
W_G1 = 35 * 9.81   # 가정: G1 질량 약 35 kg(논문 미기재)
res["body_weight_N_assumed"] = round(W_G1, 0)
res["real_F_over_bodyweight_ts"] = {t: round(T3[t]["F"]["ts"][0] / W_G1, 2) for t in T3}

# ---------------------------------------------------------------- 표 1 (시뮬 지지 품질)
T1 = {  # terrain: {metric: {policy: (mean, sd)}}
    "Flat": {"A": {"ts": (0.929, 0.054), "wo": (0.750, 0.068), "base": (0.912, 0.020)},
             "C": {"ts": (35.79, 0.54), "wo": (29.90, 2.34), "base": (35.21, 0.14)}},
    "Slope": {"A": {"ts": (0.805, 0.044), "wo": (0.624, 0.051), "base": (0.804, 0.023)},
              "C": {"ts": (34.24, 0.37), "wo": (27.55, 1.72), "base": (34.20, 0.11)}},
    "Upstairs": {"A": {"ts": (0.536, 0.042), "wo": (0.462, 0.040), "base": (0.480, 0.012)},
                 "C": {"ts": (29.05, 1.16), "wo": (25.56, 1.35), "base": (26.03, 0.40)}},
    "Downstairs": {"A": {"ts": (0.545, 0.051), "wo": (0.455, 0.048), "base": (0.487, 0.017)},
                   "C": {"ts": (29.43, 1.02), "wo": (25.91, 1.20), "base": (27.07, 0.36)}},
}
t1 = []
for terr, d in T1.items():
    for met, pol in d.items():
        for a, b in (("ts", "base"), ("ts", "wo")):
            (m1, s1), (m2, s2) = pol[a], pol[b]
            sp = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
            dd = (m1 - m2) / sp
            # 독립 시드 n개 가정: n = 3, 5, 10 에서의 Welch p, 그리고 4096 에피소드 SD 해석일 때의 p
            ps = {}
            for n in (3, 5, 10, 4096):
                _, _, p, _, _ = welch(m1, s1, n, m2, s2, n)
                ps[f"p_n{n}"] = p
            # p < 0.05 가 되는 최소 n (동일 n, Welch)
            n_need = next((n for n in range(2, 5000) if welch(m1, s1, n, m2, s2, n)[2] < 0.05), None)
            t1.append({"terrain": terr, "metric": "contact_area" if met == "A" else "cop_margin_mm",
                       "contrast": f"{a} vs {b}", "diff": round(m1 - m2, 4), "cohen_d": round(dd, 2),
                       "min_n_for_p05": n_need, **ps})
res["table1_sim_support"] = t1

# ---------------------------------------------------------------- 그림 6 (그림 판독 값), 표 4
FIG6 = {  # terrain: (ours, w/o soft landing, hiking baseline)  그림 6에 적힌 숫자
    "Flat": (278.16, 347.60, 464.44), "Slope": (297.32, 349.01, 440.65),
    "Upstairs": (304.45, 394.00, 450.00), "Downstairs": (320.74, 381.22, 599.44),
    "Down Platform": (310.62, 374.34, 515.46), "Up Platform": (310.73, 352.20, 475.81),
}
T4 = {  # terrain: (F dual, F single, dCoP dual, dCoP single, Ac dual, Ac single)
    "Flat": (278.16, 307.23, 35.79, 34.33, 0.929, 0.824),
    "Slope": (297.32, 335.90, 34.24, 31.40, 0.805, 0.635),
    "Stair descent": (320.74, 579.83, 29.43, 28.08, 0.545, 0.516),
    "Stair ascent": (304.45, 402.71, 29.05, 28.63, 0.536, 0.488),
    "Platform drop-down": (310.62, 333.93, None, None, None, None),
    "Platform step-up": (310.73, 421.35, None, None, None, None),
}
MAP = {"Flat": "Flat", "Slope": "Slope", "Stair descent": "Downstairs", "Stair ascent": "Upstairs",
       "Platform drop-down": "Down Platform", "Platform step-up": "Up Platform"}
rows4 = []
for terr, (fd, fs, cd, cs, ad, as_) in T4.items():
    f6 = FIG6[MAP[terr]]
    base_F = f6[2]
    rows4.append({"terrain": terr, "fig6_ours_matches_table4_dual": abs(f6[0] - fd) < 0.005,
                  "single_vs_dual_F_pct": round((fs / fd - 1) * 100, 1),
                  "single_vs_hiking_baseline_F_pct": round((fs / base_F - 1) * 100, 1),
                  "dual_vs_hiking_baseline_F_pct": round((fd / base_F - 1) * 100, 1),
                  "wo_soft_landing_share_of_total_gain_pct": round((f6[1] - f6[0]) / (f6[2] - f6[0]) * 100, 1),
                  "unexplained_by_soft_landing_reward_pct": round((f6[2] - f6[1]) / (f6[2] - f6[0]) * 100, 1)})
    if ad is not None:
        t1_name = MAP[terr]
        base_A = T1[t1_name]["A"]["base"][0] if t1_name in T1 else None
        rows4[-1].update({"table4_dual_A_matches_table1": abs(T1[t1_name]["A"]["ts"][0] - ad) < 5e-4,
                          "table4_dual_C_matches_table1": abs(T1[t1_name]["C"]["ts"][0] - cd) < 5e-3,
                          "single_A_minus_baseline_A": round(as_ - base_A, 3),
                          "single_C_minus_baseline_C": round(cs - T1[t1_name]["C"]["base"][0], 2)})
res["figure6_table4"] = rows4

# ---------------------------------------------------------------- 학습 예산 산술 (부록 B.3, 표 8)
env, steps, iters, mb = 2048, 24, 50000, 4
trans = env * steps
res["training_budget"] = {"transitions_per_update": trans, "matches_49152": trans == 49152,
                          "minibatch_size": trans // mb, "matches_12288": trans // mb == 12288,
                          "total_transitions": trans * iters,
                          "sim_hours_at_dt_0.02s_if_control_dt_is_0.02": round(trans * iters * 0.02 / 3600, 0),
                          "tactile_obs_dim": 2 * 2 * 4,
                          "note": "제어 주기는 식 13의 Δt = 0.02 s(50 Hz)를 따른 가정. 인슬 25 Hz는 그 절반"}

# ---------------------------------------------------------------- 서술 일치 검사 (텍스트 표)
res["text_consistency"] = [
    {"item": "촉각 지연 확률", "where": "표 9 'Measurement delay probability 0.10' 대 표 10 'Delay probability 0.08'",
     "values": [0.10, 0.08], "match": False},
    {"item": "표 4 dual critic 값 대 그림 6 Ours 및 표 1", "where": "표 4, 그림 6, 표 1", "match": all(
        r["fig6_ours_matches_table4_dual"] for r in rows4)},
    {"item": "촉각 관측 차원", "where": "C.2 '2 x 2 x 4 = 16'", "value": 16, "match": True},
    {"item": "mini-batch 크기", "where": "표 8 12,288 = 49,152 / 4", "match": True},
    {"item": "초록 수치 48.8 / 30.1 / 23.8", "where": "초록, 표 3", "match": heads["abstract_match"]},
    {"item": "그림 6 값 6개 지형 소수 둘째 자리 일치", "where": "그림 6 대 표 4", "match": True},
]

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("wrote", OUT)
print(json.dumps({"headline": heads["abstract_match"], "force_noise_pearson": res["force_vs_noise"]["pearson_all18"],
                  "spearman": res["force_vs_noise"]["spearman_all18"]}, ensure_ascii=False))
