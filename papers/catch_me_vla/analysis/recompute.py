"""논문 표·그림 수치를 다시 계산하고 일관성과 통계 검정을 확인한다 (CPU, 수 초).

입력: 논문 본문·부록 표의 값을 그대로 옮겨 적었다 (arXiv v1, 표 1·2·3·5).
      그림 4·5의 막대는 PDF에서 막대 끝 위치를 격자선(눈금) 기준으로 읽은 값이다(픽셀 측정, 오차 약 ±0.5 %p).
출력: analysis/recompute_results.json
실행: python3 analysis/recompute.py   (numpy, scipy)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
res = {}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ------------------------------------------------------------------ V1 표 1 (성공률 %)
cols = ["LIBERO Goal", "LIBERO Object", "Pick 1D", "Pick 2D", "Drop Ball", "Catch Block*", "1D Speed*", "Ball Color*", "Ball Speed*"]
table1 = {
    "OpenVLA": [78, 88.5, 5, 60, 0, 0, 0, 0, 0],
    "FiS-VLA": [41.5, 53.5, 0, 67.5, 80, 0, 0, 60, 0],
    "GR00T": [92, 97.5, 47.5, 20, 15, 60, 0, 2.5, 0],
    "VLA-Feedback": [92, 95.5, 80, 75, 100, 67.5, 50, 97.5, 57.5],
}
N = [200, 200, 40, 40, 40, 40, 40, 40, 40]  # LIBERO: 과제당 20회 x 10과제(가정), 동적: 과제당 40회 (본문 4.1)
res["table1_columns"] = cols
res["table1_n_assumed"] = N

# 정수 실현 가능성: 각 값이 N회 중 정수 성공 수로 나타나는가
feas = {}
for m, vals in table1.items():
    feas[m] = []
    for v, n in zip(vals, N):
        k = v * n / 100
        feas[m].append(abs(k - round(k)) < 1e-9)
res["table1_integer_feasible"] = feas
res["table1_all_feasible"] = all(all(x) for x in feas.values())
# 정수 가능성의 분모 검사: LIBERO가 N=100이면 소수 .5가 불가능함을 확인
res["libero_N100_feasible_for_x.5"] = False

# 평균 (초록: 27.5 -> 85.0)
dyn_seen = slice(2, 5)
dyn_unseen = slice(5, 9)
avg = {}
for m, vals in table1.items():
    avg[m] = {
        "dyn3_mean": round(float(np.mean(vals[dyn_seen])), 2),
        "unseen4_mean": round(float(np.mean(vals[dyn_unseen])), 2),
        "dyn7_mean": round(float(np.mean(vals[2:])), 2),
        "libero2_mean": round(float(np.mean(vals[:2])), 2),
    }
res["table1_means"] = avg

# GR00T 대 VLA-Feedback, 열별 Fisher (비쌍체, 같은 시행 수 가정), Holm(9열)
g, f = table1["GR00T"], table1["VLA-Feedback"]
rows = []
ps = []
for c, n, a, b in zip(cols, N, g, f):
    ka, kb = round(a * n / 100), round(b * n / 100)
    p = fisher(ka, n, kb, n)
    ps.append(p)
    rows.append({"col": c, "n": n, "gr00t": f"{ka}/{n}", "vlafb": f"{kb}/{n}", "diff_pp": round(b - a, 1),
                 "fisher_p": p, "wilson_gr00t": wilson(ka, n), "wilson_vlafb": wilson(kb, n)})
adj = holm(ps)
for r, a in zip(rows, adj):
    r["holm_p"] = a
res["gr00t_vs_vlafb"] = rows

# 동적 7과제를 합친 풀링 (과제 간 이질성 때문에 참고용)
kg = sum(round(v * 40 / 100) for v in g[2:])
kf = sum(round(v * 40 / 100) for v in f[2:])
res["pooled_dyn7"] = {"gr00t": f"{kg}/280", "vlafb": f"{kf}/280", "fisher_p": fisher(kg, 280, kf, 280)}

# VLA-Feedback 대 최강 기준선 (과제별)
best = []
for j, c in enumerate(cols):
    vals = {m: table1[m][j] for m in ("OpenVLA", "FiS-VLA", "GR00T")}
    mb = max(vals, key=vals.get)
    n = N[j]
    p = fisher(round(f[j] * n / 100), n, round(vals[mb] * n / 100), n)
    best.append({"col": c, "best_baseline": mb, "value": vals[mb], "vlafb": f[j], "fisher_p": p})
res["vlafb_vs_best_baseline"] = best

# ------------------------------------------------------------------ V2 실로봇 (그림 4, 막대 읽기)
real = {  # 격자 눈금 기준 픽셀 읽기: 그림 4 오른쪽 막대
    "GR00T": {"bread": 80.0, "lemon": 45.0, "can": 35.0, "avg_bar": 51.0},
    "VLA-Feedback": {"bread": 90.0, "lemon": 65.0, "can": 65.0, "avg_bar": 73.0},
}
n_real = 20
rr = {}
for m, d in real.items():
    ks = {t: round(d[t] * n_real / 100) for t in ("bread", "lemon", "can")}
    mean3 = float(np.mean([d[t] for t in ("bread", "lemon", "can")]))
    rr[m] = {"counts": ks, "mean_of_three": round(mean3, 2), "reported_average": d["avg_bar"],
             "gap_pp": round(d["avg_bar"] - mean3, 2)}
res["real_bars"] = rr
# 20회 x 3과제의 평균이 51이 되려면 합이 153 (5 %p 단위 합은 5의 배수라 불가)
res["real_gr00t_avg51_requires_sum"] = 51 * 3
res["real_gr00t_sum_options"] = {"150": round(150 / 3, 2), "155": round(155 / 3, 2)}

real_tests = []
for t in ("bread", "lemon", "can"):
    k1, k2 = rr["GR00T"]["counts"][t], rr["VLA-Feedback"]["counts"][t]
    real_tests.append({"task": t, "gr00t": f"{k1}/20", "vlafb": f"{k2}/20", "fisher_p": fisher(k1, 20, k2, 20),
                       "wilson_gr00t": wilson(k1, 20), "wilson_vlafb": wilson(k2, 20)})
res["real_per_task"] = real_tests
kg_dyn = rr["GR00T"]["counts"]["lemon"] + rr["GR00T"]["counts"]["can"]
kf_dyn = rr["VLA-Feedback"]["counts"]["lemon"] + rr["VLA-Feedback"]["counts"]["can"]
kg_all = kg_dyn + rr["GR00T"]["counts"]["bread"]
kf_all = kf_dyn + rr["VLA-Feedback"]["counts"]["bread"]
res["real_pooled"] = {
    "dynamic_2tasks": {"gr00t": f"{kg_dyn}/40", "vlafb": f"{kf_dyn}/40", "fisher_p": fisher(kg_dyn, 40, kf_dyn, 40)},
    "all_3tasks": {"gr00t": f"{kg_all}/60", "vlafb": f"{kf_all}/60", "fisher_p": fisher(kg_all, 60, kf_all, 60),
                   "gr00t_pct": round(100 * kg_all / 60, 2), "vlafb_pct": round(100 * kf_all / 60, 2)},
}

# ------------------------------------------------------------------ V3 표 3 (속도) 와 표 1 대조
speed = {"Original": (0.050, 15, 100), "+10%": (0.055, 5, 92.5), "+20%": (0.060, 5, 85),
         "+30%": (0.065, 0, 57.5), "+40%": (0.070, 0, 5)}
t3 = []
for k, (v, a, b) in speed.items():
    ka, kb = round(a * 40 / 100), round(b * 40 / 100)
    t3.append({"row": k, "speed_mps": v, "gr00t": f"{ka}/40", "vlafb": f"{kb}/40", "fisher_p": fisher(ka, 40, kb, 40),
               "wilson_vlafb": wilson(kb, 40)})
res["table3"] = t3
res["table3_ratio_of_0.066"] = round(0.066 / 0.05, 3)
res["table3_vs_table1"] = {
    "Original_equals_Drop_Ball": (15, 100) == (g[4], f[4]),
    "plus30_equals_Ball_Speed": (0, 57.5) == (g[8], f[8]),
    "note": "Table 1 Drop Ball Speed는 0.066 m/s(+32%), Table 3 +30%는 0.065 m/s인데 두 줄의 값이 같다",
}

# ------------------------------------------------------------------ V4 표 2 반응 지연과 표 5 합계
dt_ms = 100.0  # 10 Hz
t2 = {"OpenVLA": (160, 1), "GR00T": (80, 16), "FiS-VLA": (40, 1), "VLA-Feedback": (2, 1)}
t2_out = {}
for m, (tu, r) in t2.items():
    t2_out[m] = {"T_update_ms": tu, "R": r, "formula": f"{tu} + {r/2}dt", "at_dt100_ms": tu + r / 2 * dt_ms}
res["table2_at_10Hz"] = t2_out
res["table2_ratio_to_vlafb"] = {m: round(v["at_dt100_ms"] / t2_out["VLA-Feedback"]["at_dt100_ms"], 2) for m, v in t2_out.items()}
# 표 5
sim = [0.03, 0.06, 1.95]
hw = [62.0, 4.0, 0.06, 1.95, 2.0, 1.0]
res["table5"] = {"sim_sum": round(sum(sim), 2), "sim_reported": 2.04, "hw_sum": round(sum(hw), 2), "hw_reported": 71.01,
                 "hw_non_inference_ms": round(sum(hw) - 1.95, 2),
                 "hw_reaction_with_sensing_ms": round(sum(hw) + dt_ms / 2, 2),
                 "formula_reaction_ms": 2 + dt_ms / 2}
# 추론 주파수: 1/(16*0.1 s)
res["planner_hz"] = 1 / (16 * dt_ms / 1000)
# 낡은 정도(stale) 변위: 속도 x (T_update + R/2 dt)
vel = {"Pick 1D (0.04 m/s)": 0.04, "Drop Ball (0.05 m/s)": 0.05}
stale = {}
for task, v in vel.items():
    stale[task] = {m: round(100 * v * x["at_dt100_ms"] / 1000, 2) for m, x in t2_out.items()}  # cm
    stale[task]["GR00T_chunk_end_cm"] = round(100 * v * (80 + 16 * dt_ms) / 1000, 2)
res["stale_displacement_cm"] = stale
res["code_cup_xy_tol_cm"] = 3.0

# ------------------------------------------------------------------ V5 그림 5 막대 (픽셀 읽기)
fig5a = {"Delta-a (direct residual)": (63, 92), "No Obs": (53, 91), "Transformer->MLP": (77, 91), "VLA-Feedback": (85, 94)}
fig5b = {"1:2": (62, 90), "1:4": (67, 90), "1:8": (67, 92), "1:16": (85, 94)}
res["fig5a_dynamic_static"] = fig5a
res["fig5b_dynamic_static"] = fig5b
dyn_g, sta_g = avg["GR00T"]["dyn3_mean"], avg["GR00T"]["libero2_mean"]
res["fig5_checks"] = {
    "full_dynamic_equals_table1_mean": [85, avg["VLA-Feedback"]["dyn3_mean"]],
    "full_static_vs_libero2_mean": [94, avg["VLA-Feedback"]["libero2_mean"]],
    "gr00t_dyn3_mean": dyn_g, "gr00t_static_mean": sta_g,
    "no_obs_gain_over_gr00t_pp": round(53 - dyn_g, 1),
    "full_gain_over_gr00t_pp": round(85 - dyn_g, 1),
    "no_obs_share_of_gain": round((53 - dyn_g) / (85 - dyn_g), 3),
    "obs_gain_over_no_obs_pp": 85 - 53,
    "freq_1_2_gain_over_gr00t_pp": round(62 - dyn_g, 1),
    "freq_1_2_share_of_gain": round((62 - dyn_g) / (85 - dyn_g), 3),
}

# ------------------------------------------------------------------ V6 학습 가능 파라미터 (부록 A.2)
def decoder_layer_params(d, ff):
    sa = 4 * d * d + 4 * d          # in_proj(3d*d+3d) + out_proj(d*d+d)
    ca = 4 * d * d + 4 * d
    ffn = d * ff + ff + ff * d + d
    ln = 3 * 2 * d
    return sa + ca + ffn + ln


d, ff, act_dim = 256, 512, 7
resnet18_wo_fc = 11_176_512  # conv+bn 합 (fc 제외)
count = {}
for in_dim in (1024, 1536):
    trainable = (in_dim * d + d) + (512 * d + d) + (d * act_dim + act_dim) + decoder_layer_params(d, ff) + 1
    count[str(in_dim)] = {"trainable": trainable, "trainable_M": round(trainable / 1e6, 3),
                          "total_M": round((trainable + resnet18_wo_fc) / 1e6, 2)}
res["param_count"] = count
res["param_reported"] = {"trainable_M": 1.19, "total_M": 12.4, "resnet_M": 11.2}
# ResNet18 224x224 -> 7x7 (49 위치) -> 8x8 평균 풀링(64 토큰): 해상도가 올라가는 방향
res["resnet_tokens"] = {"native_224": 49, "after_adaptive_pool_8x8": 64, "sim_input_128_resized_to": 224}

# ------------------------------------------------------------------ V7 시행 수와 최소 검출 차이 (근사, 양측 alpha 0.05, 검정력 0.8)
def mde(n, p0=0.5):
    # 정규 근사로 두 비율 검정의 최소 검출 차이 (p0 기준 대칭 가정)
    za, zb = 1.96, 0.8416
    d = (za + zb) * math.sqrt(2 * p0 * (1 - p0) / n)
    return round(100 * d, 1)

res["mde_pp_at_p50"] = {"n20": mde(20), "n40": mde(40), "n200": mde(200)}
# 필요 시행 수: 두 비율 p1 대 p2
def need_n(p1, p2):
    za, zb = 1.96, 0.8416
    pb = (p1 + p2) / 2
    return math.ceil(((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2)

res["need_n_per_arm"] = {"45_vs_65": need_n(0.45, 0.65), "35_vs_65": need_n(0.35, 0.65), "80_vs_90": need_n(0.80, 0.90)}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
for k in ("table1_all_feasible", "table1_means", "real_bars", "real_pooled", "table2_at_10Hz", "table5", "fig5_checks", "param_count", "mde_pp_at_p50", "need_n_per_arm", "pooled_dyn7"):
    print(k, json.dumps(res[k], ensure_ascii=False))
print("per-column:")
for r in res["gr00t_vs_vlafb"]:
    print(r["col"], r["gr00t"], r["vlafb"], round(r["fisher_p"], 5), round(r["holm_p"], 5))
print("vs best:")
for r in res["vlafb_vs_best_baseline"]:
    print(r["col"], r["best_baseline"], r["value"], r["vlafb"], round(r["fisher_p"], 5))
print("real tests:", json.dumps(res["real_per_task"]))
print("table3:", json.dumps(res["table3"]))
print("stale:", json.dumps(res["stale_displacement_cm"]))
