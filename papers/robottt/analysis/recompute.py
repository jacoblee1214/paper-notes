"""RoboTTT (arXiv:2607.15275 v1) 보고 수치 재계산과 일관성 검사.

입력은 논문 표 1–3의 성공 횟수(본문 값)와, 그림 7·8·10·12·A.1의 벡터 막대·선에서 읽은 값(그림 판독 값, 오차 약 ±0.005)이다.
출력: analysis/recompute_results.json 과 표준 출력 요약. numpy, scipy만 쓴다(1초 미만).
실행: python3 analysis/recompute.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def clopper_upper(k, n, a=0.05):
    return 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


# ---------------------------------------------------------------- V1. 표 1 (완전 성공 횟수)
T1 = {  # 방법: (Pup Go Car /20, Circuit /20, Gear Bot /10)
    "RoboTTT": (9, 13, 2),
    "GR00T N1.7": (3, 3, 0),
    "GR00T N1.7 Hist.": (0, 8, 0),
    "GDN": (3, 8, 0),
}
N1 = (20, 20, 10)
tasks = ["Pup Go Car", "Circuit", "Gear Bot"]
v1 = {"pooled": {}, "per_task_fisher": {}, "pooled_fisher_vs_RoboTTT": {}}
for m, ks in T1.items():
    k, n = sum(ks), sum(N1)
    v1["pooled"][m] = {"k": k, "n": n, "rate": round(k / n, 3), "wilson95": wilson(k, n)}
for m in ["GR00T N1.7", "GR00T N1.7 Hist.", "GDN"]:
    v1["pooled_fisher_vs_RoboTTT"][m] = round(fisher(24, 50, sum(T1[m]), 50), 5)
    v1["per_task_fisher"][m] = {t: round(fisher(T1["RoboTTT"][i], N1[i], T1[m][i], N1[i]), 4) for i, t in enumerate(tasks)}
# Holm (9개 과제별 비교)
ps = sorted((v, (m, t)) for m, d in v1["per_task_fisher"].items() for t, v in d.items())
holm = {}
mtests = len(ps)
run = 0.0
for i, (p, key) in enumerate(ps):
    run = max(run, min(1.0, p * (mtests - i)))
    holm[f"{key[0]} / {key[1]}"] = round(run, 4)
v1["per_task_holm_adjusted"] = holm
v1["gear_bot_baseline_upper95_if_0_of_10"] = round(clopper_upper(0, 10), 3)
v1["gear_bot_RoboTTT_wilson"] = wilson(2, 10)
res["V1_table1"] = v1

# ---------------------------------------------------------------- V2. 표 2 (일회 모방), 표 3 (교란)
res["V2_table2"] = {
    "RoboTTT_wilson": wilson(6, 10), "GDN_clopper_upper": round(clopper_upper(0, 10), 3),
    "fisher_two_sided": round(fisher(6, 10, 0, 10), 5),
}
T3 = {"RoboTTT": (15, 18), "GR00T N1.7": (10, 11), "GR00T N1.7 Hist.": (3, 5), "GDN": (13, 18)}
v3 = {"pooled_rate": {}, "fisher_vs_RoboTTT": {}}
for m, (a, b) in T3.items():
    v3["pooled_rate"][m] = {"k": a + b, "n": 40, "rate": round((a + b) / 40, 3), "wilson95": wilson(a + b, 40)}
for m in ["GR00T N1.7", "GR00T N1.7 Hist.", "GDN"]:
    a, b = T3[m]
    v3["fisher_vs_RoboTTT"][m] = {
        "roof": round(fisher(15, 20, a, 20), 4), "tire": round(fisher(18, 20, b, 20), 4),
        "pooled": round(fisher(33, 40, a + b, 40), 4)}
res["V3_table3"] = v3

# ---------------------------------------------------------------- V4. 본문 백분율 (상대 향상) 재계산
def rel(a, b):
    return round(100 * (a / b - 1), 1)


# 그림 7: 벡터 막대 높이를 본문 앵커(RoboTTT 평균 79%, GR00T 42%, GDN 56%, Hist Pup 39.5%)로 보정한 값
F7 = {  # [Average, Pup Go Car, Circuit, Gear Bot]
    "RoboTTT": [0.789, 0.886, 0.775, 0.705],
    "GR00T N1.7": [0.421, 0.566, 0.230, 0.467],
    "GR00T N1.7 Hist.": [0.489, 0.392, 0.539, 0.536],
    "GDN": [0.558, 0.564, 0.526, 0.586],
}
res["F7_read_values"] = F7
res["V4_F7_mean_of_tasks"] = {m: round(float(np.mean(v[1:])), 3) for m, v in F7.items()}
res["V4_text_vs_recomputed"] = {
    "RoboTTT vs GR00T (text 87%)": rel(79, 42), "RoboTTT vs GR00T (F7 read)": rel(F7["RoboTTT"][0], F7["GR00T N1.7"][0]),
    "RoboTTT vs GDN (text 41%)": rel(79, 56),
    "8K vs 1K (text abstract 62%, intro 63%)": rel(71.5, 43.9),
    "8K vs short ctx (text 57%)": rel(71.5, 45.6),
    "Pup Hist vs GR00T (text 39.5 vs 57)": rel(39.5, 57),
    "tire/roof robustness 83% vs 53% (RoboTTT 33/40, GR00T 21/40)": [round(33 / 40 * 100, 1), round(21 / 40 * 100, 1)],
}

# ---------------------------------------------------------------- V5. 점수가 시행 수에서 가능한 범위인지
SMAX = {"Pup Go Car": 0.9, "Circuit": 0.75, "Gear Bot": 0.9}  # 루브릭에서 완전 성공 바로 아래 최대 점수(부록 B.1)
feas = {}
for m, ks in T1.items():
    for i, t in enumerate(tasks):
        k, n = ks[i], N1[i]
        lo = k / n
        hi = (k + (n - k) * SMAX[t]) / n
        feas[f"{m} / {t}"] = {"k": k, "n": n, "score_read": F7[m][i + 1], "lo": round(lo, 3), "hi": round(hi, 3),
                              "inside": bool(lo - 0.006 <= F7[m][i + 1] <= hi + 0.006)}
res["V5_score_feasible_given_table1"] = feas
# Pup Go Car RoboTTT: 비완전 11회의 평균 점수와 바퀴 삽입(0.75) 단계 이상 도달 최소 횟수
mean_non = (0.886 * 20 - 9) / 11
x_low = (0.9 * 11 - (0.886 * 20 - 9 - 0.1)) / (0.9 - 0.6)  # 점수 0.6 이하인 시행 수의 상한 (0.1 반올림 여유)
res["V5_pup_RoboTTT_nonfull_mean"] = round(mean_non, 3)
res["V5_pup_RoboTTT_min_trials_reaching_0p75_of_11"] = int(11 - math.floor(x_low))

# ---------------------------------------------------------------- V6. 점수 차의 유의성 (시행별 점수 없음 -> 상계 SD로 보수적 z)
def z_bound(m1, n1, m2, n2, sd=None):
    s1 = math.sqrt(m1 * (1 - m1)) if sd is None else sd
    s2 = math.sqrt(m2 * (1 - m2)) if sd is None else sd
    return (m1 - m2) / math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)


zt = {}
for m in ["GR00T N1.7", "GR00T N1.7 Hist.", "GDN"]:
    for i, t in enumerate(tasks):
        n = N1[i]
        zt[f"RoboTTT vs {m} / {t}"] = {
            "diff": round(F7["RoboTTT"][i + 1] - F7[m][i + 1], 3),
            "z_upperbound_sd": round(z_bound(F7["RoboTTT"][i + 1], n, F7[m][i + 1], n), 2),
            "z_sd0.3": round(z_bound(F7["RoboTTT"][i + 1], n, F7[m][i + 1], n, 0.3), 2),
        }
res["V6_score_diff_z"] = zt

# ---------------------------------------------------------------- V7. 그림 8 (사전학습 문맥 길이 스케일링)
Ls = [128, 256, 512, 1024, 2048, 4096, 8192]
F8 = {
    "RoboTTT": [0.306, 0.288, 0.392, 0.438, 0.543, 0.604, 0.716],
    "GDN": [0.011, 0.177, 0.401, 0.344, 0.425, 0.354, 0.442],
    "single_step_ref": 0.375, "short_ctx_ref": 0.456,
}
res["F8_read_values"] = F8
x = np.log2(Ls)
fit = {}
for name, sl in [("128-8K", slice(0, 7)), ("1K-8K", slice(3, 7))]:
    r = stats.linregress(x[sl], F8["RoboTTT"][sl])
    fit[name] = {"slope_per_doubling": round(r.slope, 4), "r2": round(r.rvalue ** 2, 3), "p": round(r.pvalue, 4)}
rg = stats.linregress(x, F8["GDN"])
fit["GDN_128-8K"] = {"slope_per_doubling": round(rg.slope, 4), "r2": round(rg.rvalue ** 2, 3), "p": round(rg.pvalue, 3)}
rg2 = stats.linregress(x[2:], F8["GDN"][2:])
fit["GDN_512-8K"] = {"slope_per_doubling": round(rg2.slope, 4), "r2": round(rg2.rvalue ** 2, 3), "p": round(rg2.pvalue, 3)}
res["V7_fit"] = fit
res["V7_increments_RoboTTT"] = [round(F8["RoboTTT"][i + 1] - F8["RoboTTT"][i], 3) for i in range(6)]
res["V7_RoboTTT_minus_short_ctx"] = {str(L): round(v - F8["short_ctx_ref"], 3) for L, v in zip(Ls, F8["RoboTTT"])}
res["V7_RoboTTT_minus_GDN"] = {str(L): round(a - b, 3) for L, a, b in zip(Ls, F8["RoboTTT"], F8["GDN"])}
# 평균 점수의 표준오차 (세 과제 20/20/10회, 시행당 SD 가정)
se = {}
for sd in (0.25, 0.30, 0.40):
    se_avg = math.sqrt((sd ** 2 / 20) * 2 + sd ** 2 / 10) / 3
    se[str(sd)] = {"se_avg": round(se_avg, 4), "se_diff": round(se_avg * math.sqrt(2), 4),
                   "z_1K_to_8K": round((F8["RoboTTT"][6] - F8["RoboTTT"][3]) / (se_avg * math.sqrt(2)), 2),
                   "z_1K_vs_short": round((F8["RoboTTT"][3] - 0.456) / (se_avg * math.sqrt(2)), 2),
                   "z_adjacent_mean": round(float(np.mean([abs(d) for d in res["V7_increments_RoboTTT"][3:]])) / (se_avg * math.sqrt(2)), 2)}
res["V7_se_assumed_sd"] = se
# 그림 8 기준선과 그림 7·10을 합친 값의 비교 (사전 DAgger 값 = 그림 10 "Before DAgger")
F10 = {  # 막대 높이 / 0.886(= 그림 7 Pup RoboTTT 보정) 비율에서 읽음
    "RoboTTT": {"before": 0.650, "human_only": 0.720, "distill": 0.886},
    "GDN": {"before": 0.437, "human_only": 0.506, "distill": 0.563},
    "GR00T N1.7": {"before": 0.524, "human_only": 0.566, "full_traj": 0.566},
    "GR00T N1.7 Hist.": {"before": 0.385, "human_only": 0.392},
}
res["F10_read_values"] = F10
pre = {}
for m, ref, col in [("RoboTTT", F8["RoboTTT"][6], "RoboTTT"), ("GR00T N1.7", F8["single_step_ref"], "GR00T N1.7"),
                    ("GR00T N1.7 Hist.", F8["short_ctx_ref"], "GR00T N1.7 Hist."), ("GDN", F8["GDN"][6], "GDN")]:
    c, g = F7[col][2], F7[col][3]
    est = (F10[col]["before"] + c + g) / 3
    pre[m] = {"F8_value": ref, "mean(F10_before, F7_circuit, F7_gear)": round(est, 3), "gap": round(ref - est, 3),
              "implied_pup_from_F8": round(3 * ref - c - g, 3), "F10_before_pup": F10[col]["before"]}
res["V7_F8_vs_F7_F10_consistency"] = pre
# DAgger 상대 향상
dg = {}
for m, d in F10.items():
    dg[m] = {k: rel(v, d["before"]) for k, v in d.items() if k != "before"}
res["V8_dagger_relative_gain_pct"] = dg
res["V8_text_check"] = {
    "human_only_avg4 (text 9)": round(float(np.mean([dg[m]["human_only"] for m in dg])), 1),
    "human_only_seq2 (text 13)": round(float(np.mean([dg["RoboTTT"]["human_only"], dg["GDN"]["human_only"]])), 1),
    "distill_avg2 (text 33)": round(float(np.mean([dg["RoboTTT"]["distill"], dg["GDN"]["distill"]])), 1),
    "distill_RoboTTT (text 36)": dg["RoboTTT"]["distill"], "distill_GDN (text 29)": dg["GDN"]["distill"],
}
res["V8_absolute_gain"] = {m: {k: round(v - d["before"], 3) for k, v in d.items() if k != "before"} for m, d in F10.items()}
# 그림 7의 Pup Go Car 값 == 그림 10의 사후 값인지
res["V8_F7_pup_equals_F10_after"] = {
    "RoboTTT F7 .886 vs F10 distill": F10["RoboTTT"]["distill"], "GDN F7 .564 vs distill": F10["GDN"]["distill"],
    "GR00T F7 .566 vs human_only": F10["GR00T N1.7"]["human_only"], "Hist F7 .392 vs human_only": F10["GR00T N1.7 Hist."]["human_only"]}

# ---------------------------------------------------------------- V9. 그림 12 (절제)
F12 = {"No Seq. Action Forcing": 0.01, "TTT Linear": 0.477, "State Tokens": 0.447, "+ Action Tokens": 0.551,
       "+ Register Tokens (full)": 0.650, "GR00T N1.7 w/ Register Tokens": 0.434}
res["F12_read_values_scaled_to_full_0.65"] = F12
res["V9_text_check"] = {
    "action_vs_state (text 23)": rel(F12["+ Action Tokens"], F12["State Tokens"]),
    "register_vs_action (text 18)": rel(F12["+ Register Tokens (full)"], F12["+ Action Tokens"]),
    "linear_vs_MLP deficit % (text 27)": round(100 * (1 - F12["TTT Linear"] / F12["+ Register Tokens (full)"]), 1),
    "full F12 vs F10 before (RoboTTT Pup)": [F12["+ Register Tokens (full)"], F10["RoboTTT"]["before"]],
    "GR00T+reg vs GR00T F10 before": [F12["GR00T N1.7 w/ Register Tokens"], F10["GR00T N1.7"]["before"]],
}

# ---------------------------------------------------------------- V10. 시간·규모 산술
hz = 30
res["V10_time"] = {
    "128 steps s": round(128 / hz, 1), "1K steps s": round(1024 / hz, 1), "8K steps s": round(8192 / hz, 1), "8K min": round(8192 / hz / 60, 2),
    "Pup Go Car 2min steps": 2 * 60 * hz, "Circuit 1min steps": 60 * hz, "Gear Bot 5min steps": 5 * 60 * hz,
    "Gear Bot episode / 8K": round(5 * 60 * hz / 8192, 2), "post-train 1K / Gear Bot episode": round(1024 / (5 * 60 * hz), 2),
    "Circuit human video 1min + robot 1min steps": 2 * 60 * hz,
}
# 사전학습 배치: 문맥 길이별 한 스텝의 타임스텝 수와 총 시퀀스 수
bs = {}
for L in Ls:
    g = 64 if L <= 4096 else 16
    bs[str(L)] = {"global_batch": g, "timesteps_per_update": g * L, "sequences_total_30k": g * 30000,
                  "timesteps_total_30k_M": round(g * L * 30000 / 1e6, 1)}
res["V10_pretrain_batches"] = bs
# 그림 A.1 (벡터 막대 높이 -> 궤적 수)
A1_pt = [5.33, 21.20, 44.03, 70.51, 29.14]  # 막대 높이(pt)
pt_per_100k = 2 * 53.12 * 0.741618  # 눈금 50K = 53.12 (국소) * 0.7416
cnt = [h / pt_per_100k * 100_000 for h in A1_pt]
tot = sum(cnt)
res["V10_figA1"] = {"bins": ["<512", "512-1K", "1K-2K", "2K-4K", ">4K"], "share_pct": [round(100 * c / tot, 1) for c in cnt],
                    "count_est": [int(round(c, -2)) for c in cnt], "total_est": int(round(tot, -3)),
                    "sum_of_labels": round(3.1 + 12.5 + 25.9 + 41.4 + 17.1, 1),
                    "epochs_at_64x30k_of_total": round(64 * 30000 / tot, 2), "epochs_at_16x30k_of_total": round(16 * 30000 / tot, 2)}
# 파라미터
res["V10_params"] = {
    "DiT 538M + 16 x 10M": 538 + 16 * 10, "stated total 690M -> per TTT layer M": round((690 - 538) / 16, 2),
    "public GR00T N1.7 DiT inner dim (32 heads x 48)": 32 * 48,
    "fast MLP d x d hidden params (M)": round(2 * 1536 * 1536 / 1e6, 2), "fast MLP hidden d/2 (M)": round(2 * 1536 * 768 / 1e6, 2),
    "q,k,v proj d x d (M)": round(3 * 1536 * 1536 / 1e6, 2),
}
# 배치·GPU 산술
res["V10_gpu"] = {"pretrain GPUs": 16, "per-device batch<=4K": 4, "global<=4K": 4 * 16, "per-device >4K": 1, "global >4K": 1 * 16,
                  "post-train GPUs": 8, "post-train global batch (1 per device)": 8}

# ---------------------------------------------------------------- V11. 상태 크기 대 KV 캐시 (가정을 명시한 비용 모델)
d, layers, tok_ctx = 1536, 16, 17  # 시간 축으로 흐르는 토큰: 레지스터 16 + 상태 1 (+ 행동 토큰 H)
cost = {}
for H in (16, 40):
    per_step_tokens = 16 + 1 + H
    kv_bytes = lambda T, tpt: T * tpt * layers * 2 * d * 2  # K,V, bf16
    fast_bytes_hidden_d = layers * (2 * d * d) * 2
    fast_bytes_hidden_half = layers * (2 * d * (d // 2)) * 2
    cost[f"H={H}"] = {
        "tokens_per_timestep": per_step_tokens,
        "kv_cache_GB_8K_all_tokens": round(kv_bytes(8192, per_step_tokens) / 1e9, 1),
        "kv_cache_GB_8K_per_chunk_stride": round(kv_bytes(8192 // H, per_step_tokens) / 1e9, 2),
        "fast_weights_MB_hidden_d": round(fast_bytes_hidden_d / 1e6, 1),
        "fast_weights_MB_hidden_d_half": round(fast_bytes_hidden_half / 1e6, 1),
        "updates_over_8K_if_one_per_chunk": 8192 // H,
        "fast_update_GFLOP_per_chunk_6PT": round(6 * (2 * d * d) * per_step_tokens * layers / 1e9, 1),
    }
res["V11_cost_model_assumptions"] = "d=1536(공개 N1.7 설정), 16층, bf16, 고속 MLP 은닉=d 또는 d/2(논문 미기재, 가정), 토큰=레지스터16+상태1+행동H"
res["V11_cost_model"] = cost

# ---------------------------------------------------------------- V12. 학습 데이터 대비 DAgger 데이터
ep_minutes = {"Pup Go Car": (8, 2), "Circuit": (6, 1), "Gear Bot": (5, 5)}
res["V12_demo_counts_est"] = {k: int(h * 60 / m) for k, (h, m) in ep_minutes.items()}
res["V12_dagger_share_of_pup_demos"] = round(100 / (8 * 60 / 2 + 100), 3)

# ---------------------------------------------------------------- V13. DAgger 이전 값으로 평균을 다시 만들면
pre_avg = {m: round(float((F10[m]["before"] + F7[m][2] + F7[m][3]) / 3), 3) for m in F10}
res["V13_pre_dagger_average_read"] = pre_avg
res["V13_relative_RoboTTT_vs_pre"] = {m: rel(pre_avg["RoboTTT"], pre_avg[m]) for m in pre_avg if m != "RoboTTT"}
res["V13_relative_RoboTTT_post_vs_baseline_post"] = {m: rel(F7["RoboTTT"][0], F7[m][0]) for m in F7 if m != "RoboTTT"}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("wrote", OUT)
print(json.dumps({k: res[k] for k in ["V1_table1", "V2_table2", "V3_table3"]}, ensure_ascii=False, indent=1))
