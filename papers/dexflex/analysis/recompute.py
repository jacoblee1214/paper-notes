"""DexFLEX (arXiv:2610.14372 v1) 표·그림 수치 재계산과 검정. numpy + scipy, 수 초.

실행: python3 analysis/recompute.py   -> analysis/recompute_results.json
입력: 논문 표 2(실로봇 텔레옵), 그림 5(정책 학습, 막대 위 라벨), 표 1(절제), 그림 3 판독값(fig3_readout.json)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def cmh(tables):
    """층별 2x2 표 [[a,b],[c,d]] 목록의 Mantel-Haenszel 검정(연속성 보정 없음)과 공통 오즈비."""
    num = var = 0.0
    rn = rd = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        r1, c1 = a + b, a + c
        e = r1 * c1 / n
        v = r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1))
        num += a - e
        var += v
        rn += a * d / n
        rd += b * c / n
    chi = num * num / var
    return {"chi2": round(chi, 3), "p": float(stats.chi2.sf(chi, 1)), "or_mh": round(rn / rd, 2)}


# ---------------------------------------------------------------- 표 2 (실로봇 텔레옵)
tasks = ["Brush", "Nut", "Hammer", "Screwdrv", "Spatula", "Toast"]
T2 = {
    "Raw": [50, 30, 40, 30, 15, 10],
    "Filter": [60, 40, 50, 40, 25, 20],
    "DexGen": [70, 55, 60, 55, 40, 35],
    "DexFLEX w/o Tactile": [75, 60, 65, 60, 50, 45],
    "DexFLEX": [90, 80, 85, 80, 70, 65],
}
T2_avg_rep = {"Raw": 29.2, "Filter": 39.2, "DexGen": 52.5, "DexFLEX w/o Tactile": 59.2, "DexFLEX": 78.3}
CT = {"Raw": 27.5, "Filter": 33.8, "DexGen": 31.2, "DexFLEX w/o Tactile": 30.8, "DexFLEX": 32.0}
NHT = {"Raw": 0.42, "Filter": 0.55, "DexGen": 0.65, "DexFLEX w/o Tactile": 0.71, "DexFLEX": 0.85}
N = 20  # 과제·방법당 시행 수(본문 4.1, 표 2 캡션)

v = {}
v["V1_avg_recompute"] = {m: {"recomputed": round(sum(x) / 6, 2), "reported": T2_avg_rep[m],
                              "match": abs(sum(x) / 6 - T2_avg_rep[m]) < 0.06} for m, x in T2.items()}
v["V2_integer_feasible"] = {
    "cells_multiple_of_5": all(c % 5 == 0 for x in T2.values() for c in x),
    "avg_is_multiple_of_5/6": all(abs(round(sum(x) / 5) - sum(x) / 5) < 1e-9 for x in T2.values()),
}
succ = {m: [round(c * N / 100) for c in x] for m, x in T2.items()}
tot = {m: sum(s) for m, s in succ.items()}
v["V3_pooled"] = {m: {"k": tot[m], "n": 6 * N, "rate": round(tot[m] / (6 * N), 4), "wilson95": wilson(tot[m], 6 * N)}
                  for m in T2}
# 본문 문장 검사
v["V4_text_claims"] = {
    "gain_over_DexGen_pp": round(78.3 - 52.5, 1), "gain_over_DexGen_exact": round(470 / 6 - 52.5, 2),
    "gain_tactile_pp_text_19.1": round(78.3 - 59.2, 1), "gain_tactile_exact": round(470 / 6 - 355 / 6, 2),
    "SR_ratio_vs_raw": round((470 / 6) / (175 / 6), 2), "NHT_ratio": round(0.85 / 0.42, 2),
    "wo_tactile_over_DexGen_pp": round(355 / 6 - 52.5, 2),
}
# 쌍별 검정 (풀링 120 시행, 시행이 독립이라는 가정)
pairs = [("DexFLEX", "Raw"), ("DexFLEX", "Filter"), ("DexFLEX", "DexGen"), ("DexFLEX", "DexFLEX w/o Tactile"),
         ("DexFLEX w/o Tactile", "DexGen"), ("DexGen", "Filter"), ("DexGen", "Raw")]
pw = []
for a, b in pairs:
    ka, kb = tot[a], tot[b]
    p = stats.fisher_exact([[ka, 120 - ka], [kb, 120 - kb]])[1]
    diff = (ka - kb) / 120
    se = math.sqrt(ka / 120 * (1 - ka / 120) / 120 + kb / 120 * (1 - kb / 120) / 120)
    pw.append({"a": a, "b": b, "ka": ka, "kb": kb, "diff_pp": round(100 * diff, 1),
               "diff_ci95_pp": [round(100 * (diff - 1.96 * se), 1), round(100 * (diff + 1.96 * se), 1)],
               "fisher_p": float(p)})
v["V5_pairwise_pooled"] = pw
# 과제별 Fisher + Holm, 층별 MH, 부호 검정
perm = {}
for a, b in pairs:
    ps = []
    for t in range(6):
        ka, kb = succ[a][t], succ[b][t]
        ps.append(stats.fisher_exact([[ka, N - ka], [kb, N - kb]])[1])
    tabs = [[[succ[a][t], N - succ[a][t]], [succ[b][t], N - succ[b][t]]] for t in range(6)]
    wins = sum(succ[a][t] > succ[b][t] for t in range(6))
    perm[f"{a} vs {b}"] = {"per_task_p": [round(float(p), 4) for p in ps],
                           "holm_p": [round(x, 4) for x in holm(ps)],
                           "n_sig_raw": int(sum(p < 0.05 for p in ps)),
                           "n_sig_holm": int(sum(x < 0.05 for x in holm(ps))),
                           "cmh": cmh(tabs), "task_wins": wins,
                           "sign_test_two_sided_p": float(stats.binomtest(wins, 6, 0.5).pvalue)}
v["V6_per_task_tests"] = perm
# CT 선택 편향: 성공 시행만으로 평균을 내므로 성공 수가 다르면 모집단이 다르다
v["V7_CT_selection"] = {m: {"successes": tot[m], "CT_s": CT[m], "NHT": NHT[m],
                            "NHT_minus_SRrate_if_fail_zero": round(NHT[m] - tot[m] / 120, 2)} for m in T2}
v["V7_CT_selection"]["note"] = ("CT는 성공 시행 평균. Raw 35회 대 DexFLEX 94회처럼 모집단이 달라 CT 비교는 선택 편향이 있고, "
                                "분산이 보고되지 않아 CT 차이(27.5–33.8 s)의 유의성은 계산할 수 없다.")
v["V7_CT_selection"]["corr_SR_CT"] = round(float(np.corrcoef([tot[m] for m in T2], [CT[m] for m in T2])[0, 1]), 2)
v["V7_CT_selection"]["corr_SR_NHT"] = round(float(np.corrcoef([tot[m] for m in T2], [NHT[m] for m in T2])[0, 1]), 3)
R["table2"] = v

# ---------------------------------------------------------------- 그림 5 (정책 학습)
F5 = {"Direct": [5, 3, 0], "DexGen": [10, 6, 2], "DexFLEX": [12, 9, 7]}
n5 = 20
p5 = {}
tot5 = {m: sum(x) for m, x in F5.items()}
p5["totals"] = {m: {"k": k, "n": 60, "rate": round(k / 60, 4), "wilson95": wilson(k, 60)} for m, k in tot5.items()}
p5["ratios"] = {"DexFLEX/Direct": round(28 / 8, 3), "DexFLEX/DexGen": round(28 / 18, 3),
                "reported_text": {"vs_direct": 3.5, "vs_dexgen": 1.56}}
pp = {}
for a, b in [("DexFLEX", "Direct"), ("DexFLEX", "DexGen"), ("DexGen", "Direct")]:
    ka, kb = tot5[a], tot5[b]
    ps = [stats.fisher_exact([[F5[a][t], n5 - F5[a][t]], [F5[b][t], n5 - F5[b][t]]])[1] for t in range(3)]
    tabs = [[[F5[a][t], n5 - F5[a][t]], [F5[b][t], n5 - F5[b][t]]] for t in range(3)]
    pp[f"{a} vs {b}"] = {"pooled_fisher_p": float(stats.fisher_exact([[ka, 60 - ka], [kb, 60 - kb]])[1]),
                         "per_task_p": [round(float(x), 4) for x in ps], "holm_p": [round(x, 4) for x in holm(ps)],
                         "cmh": cmh(tabs)}
p5["tests"] = pp
# 비율의 부트스트랩 구간 (과제 내 베르누이 재표집)
B = 20000
def boot(a, b):
    out = []
    for _ in range(B):
        ka = sum(rng.binomial(n5, F5[a][t] / n5) for t in range(3))
        kb = sum(rng.binomial(n5, F5[b][t] / n5) for t in range(3))
        out.append(ka / kb if kb > 0 else np.inf)
    out = np.array(out)
    fin = out[np.isfinite(out)]
    return {"ci95": [round(float(np.percentile(fin, 2.5)), 2), round(float(np.percentile(fin, 97.5)), 2)],
            "frac_infinite": round(float(1 - len(fin) / B), 4)}
p5["ratio_bootstrap"] = {"DexFLEX/Direct": boot("DexFLEX", "Direct"), "DexFLEX/DexGen": boot("DexFLEX", "DexGen")}
p5["order_all_three_tasks"] = all(F5["DexFLEX"][t] > F5["DexGen"][t] > F5["Direct"][t] for t in range(3))
R["fig5_policy"] = p5

# ---------------------------------------------------------------- 표 1 대 그림 3 일관성
T1 = {"DexFLEX": (0.86, 0.78, 0.18), "w/o predictive score": (0.52, 0.46, 0.45),
      "w/o tactile/contact": (0.62, 0.55, 0.36), "w/o consequence pred.": (0.64, 0.57, 0.34),
      "w/o pure-prior prop.": (0.66, 0.61, 0.25), "w/o draft-seeded prop.": (0.74, 0.66, 0.38),
      "w/o slow phase token": (0.81, 0.74, 0.22)}
g = json.load(open(HERE / "fig3_readout.json"))["values"]
deg = ["additive_noise", "command_scaling", "delayed_command", "support_finger_removal"]
allc = ["clean_expert"] + deg
c = {}
for key, name in [("dexflex", "DexFLEX"), ("notactile", "w/o tactile/contact")]:
    row = {}
    for metric, idx in [("duration_s", 0), ("progress", 1), ("intent_error", 2)]:
        vals4 = [g[metric][key][k] for k in deg]
        vals5 = [g[metric][key][k] for k in allc]
        row[metric] = {"mean4": round(float(np.mean(vals4)), 3), "mean5": round(float(np.mean(vals5)), 3),
                       "max4": max(vals4), "table1": T1[name][idx]}
    c[name] = row
clean_dur = {k: g["duration_s"][k]["clean_expert"] for k in g["duration_s"]}
c["clean_duration_s"] = clean_dur
for name, key in [("DexFLEX", "dexflex"), ("w/o tactile/contact", "notactile")]:
    c[name]["duration_norm_by_clean_expert_mean4"] = round(c[name]["duration_s"]["mean4"] / clean_dur[key], 3)
    c[name]["duration_norm_by_clean_expert_mean5"] = round(c[name]["duration_s"]["mean5"] / clean_dur[key], 3)
c["progress_table1_minus_fig3_mean4"] = round(0.78 - c["DexFLEX"]["progress"]["mean4"], 3)  # 평균은 최댓값(0.75)을 넘을 수 없다
c["ratio_wo_tactile_over_full"] = {
    "table1_duration": round(0.62 / 0.86, 3), "fig3_duration_mean4": round(c["w/o tactile/contact"]["duration_s"]["mean4"] / c["DexFLEX"]["duration_s"]["mean4"], 3),
    "table1_intent": round(0.36 / 0.18, 2), "fig3_intent_mean4": round(c["w/o tactile/contact"]["intent_error"]["mean4"] / c["DexFLEX"]["intent_error"]["mean4"], 2)}
# DexFLEX 대 DexGen: 그림 3 오차 막대가 겹치는가 (오차 막대는 정의가 없어 참고)
err = json.load(open(HERE / "fig3_readout.json"))["errorbars"]
ov = {}
for metric in ["duration_s", "progress", "intent_error"]:
    ov[metric] = {}
    for k in deg:
        a, b = err[metric]["dexflex"][k], err[metric]["dexgen"][k]
        ov[metric][k] = bool(max(a[0], b[0]) <= min(a[1], b[1]))
c["errorbar_overlap_DexFLEX_vs_DexGen"] = ov
ov2 = {}
for metric in ["duration_s", "progress", "intent_error"]:
    ov2[metric] = {}
    for k in deg:
        a, b = err[metric]["dexflex"][k], err[metric]["notactile"][k]
        ov2[metric][k] = bool(max(a[0], b[0]) <= min(a[1], b[1]))
c["errorbar_overlap_DexFLEX_vs_woTactile"] = ov2
c["order_table1_dur_vs_text"] = "w/o predictive score가 가장 낮음(0.52): 본문 '가장 큰 하락'과 일치"
c["table1_ablation_means_not_ranked_by_contact_critical"] = ("Table 1은 네 조건 평균이라 본문의 '접촉 결정적 손상에서 특히'를 확인할 수 없음")
R["table1_vs_fig3"] = c

# ---------------------------------------------------------------- 검정력(최소 검출 차이)
def power_two_prop(p1, p2, n, alpha=0.05, sims=4000):
    hits = 0
    for _ in range(sims):
        a, b = rng.binomial(n, p1), rng.binomial(n, p2)
        if stats.fisher_exact([[a, n - a], [b, n - b]])[1] < alpha:
            hits += 1
    return hits / sims

pw_res = {
    "per_task_n20_DexFLEX_0.78_vs_DexGen_0.53": power_two_prop(0.783, 0.525, 20),
    "per_task_n20_DexFLEX_0.78_vs_woTactile_0.59": power_two_prop(0.783, 0.592, 20),
    "pooled_n120_DexFLEX_vs_woTactile": power_two_prop(0.783, 0.592, 120, sims=2000),
    "policy_n60_0.467_vs_0.30": power_two_prop(0.467, 0.30, 60),
    "policy_per_task_n20_0.60_vs_0.50": power_two_prop(0.60, 0.50, 20),
}
n_need = None
for n in range(20, 400, 5):
    if power_two_prop(0.467, 0.30, n, sims=600) >= 0.8:
        n_need = n
        break
pw_res["n_per_arm_for_80pct_power_0.467_vs_0.30"] = n_need
R["power"] = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in pw_res.items()}

# ---------------------------------------------------------------- K1: 사건당 생존 확률로 환산
K = {}
for T in (3, 5, 8, 12):
    K[f"events={T}"] = {m: round((tot[m] / 120) ** (1 / T), 3) for m in T2}
R["K1_per_event_survival"] = K

# ---------------------------------------------------------------- K2: 유한 후보 선택 시뮬레이션
def select_sim(N_cand, p_unsafe_near, acc, trials=20000):
    """한 접촉 결정 순간. 후보 N개 중 일부는 접촉을 깨는(unsafe) 후보. 초안(손가락 제거)에 가까운 후보일수록 unsafe.
    초안 근접도 d_i ~ U(0,1), unsafe 확률 = p_unsafe_near*(1-d_i)+0.05. 결과 예측기는 안전 여부를 acc 확률로 맞춘다.
    점수 = 초안 근접도(1-d) + 예측 안전 보너스(안전으로 예측되면 +1). 반환: nearest / learned 선택의 안전 확률"""
    s_near = s_learn = s_rand = 0
    for _ in range(trials):
        d = rng.random(N_cand)
        unsafe = rng.random(N_cand) < (p_unsafe_near * (1 - d) + 0.05)
        pred_safe = np.where(rng.random(N_cand) < acc, ~unsafe, unsafe)
        i_near = int(np.argmin(d))
        score = (1 - d) + 1.0 * pred_safe
        i_l = int(np.argmax(score))
        s_near += not unsafe[i_near]
        s_learn += not unsafe[i_l]
        s_rand += not unsafe[int(rng.integers(N_cand))]
    return s_near / trials, s_learn / trials, s_rand / trials

K2 = {}
for N_cand in (4, 16, 64):
    for acc in (0.5, 0.7, 0.9):
        a, b, r = select_sim(N_cand, 0.9, acc)
        K2[f"N={N_cand},acc={acc}"] = {"nearest_safe": round(a, 3), "consequence_safe": round(b, 3), "random_safe": round(r, 3)}
R["K2_selection_toy"] = K2

json.dump(R, open(HERE / "recompute_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R["table2"]["V1_avg_recompute"], ensure_ascii=False))
print(json.dumps(R["table2"]["V5_pairwise_pooled"], ensure_ascii=False)[:1200])
print(json.dumps(R["fig5_policy"], ensure_ascii=False)[:2500])
print(json.dumps(R["table1_vs_fig3"], ensure_ascii=False)[:2500])
print(json.dumps(R["power"]), json.dumps(R["K1_per_event_survival"]))
print(json.dumps(R["K2_selection_toy"]))
