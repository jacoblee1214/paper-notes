"""IMLE-VLA 표 재계산과 지연 산술 (CPU, numpy·scipy, 수 초). 실행: python3 analysis/table_check.py

- T1: Table II (LIBERO, suite당 500회 = 50회 × 10과제, 전체 2000회). 퍼센트를 정수 횟수로 바꾸고 평균·Wilson 95% 구간,
      suite별 Fisher 정확검정, Holm 보정, 전체 Fisher. π0.5 행은 논문이 [14]에서 인용한 수치라 같은 실행이 아님.
- T2: Table II 내부 정합성: suite 4개 평균이 표의 Avg와 맞는지, H=30 행도 같은지.
- T3: 최소 검출 차이(80% 검정력, 양측 5%) 근사: 2000회, 500회, 20회.
- T4: Table V (실물, 과제당 20회): 과제별 Fisher + Holm, 합산 Fisher, Cochran–Mantel–Haenszel(과제 층화).
- T5: 지연 산술: Table I(fps), Table III(처리량), Table IV(VLA-WC, 패스 수), Table V(VLA-WC 배수). 패스당 시간, 패스 수 역산.
- T6: 지연 분해 (가정 명시): 한 청크 시간 = P(접두부, VLM) + n·a(행동 전문가 1회). IMLE는 n=1.
      같은 엔진 가정으로 P, a를 역산하고, 엔진 가정을 바꿨을 때 속도비의 범위를 본다.

주의: 논문 π0.5 행과 IMLE 행은 서로 다른 실행(다른 시드, 다른 체크포인트 선택)일 수 있다. 검정은 독립 두 표본 근사이다.
결과: analysis/table_check_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"


def count(pct, n):
    k = round(pct / 100 * n)
    assert abs(100 * k / n - pct) < 0.051, (pct, n, k)
    return k


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    ps = list(ps)
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * ps[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def mdd(p, n, alpha=0.05, power=0.8):
    # 두 독립 비율의 최소 검출 차이 (정규 근사, 같은 p 가정)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * math.sqrt(2 * p * (1 - p) / n))


def mh_cmh(tables):
    # Cochran–Mantel–Haenszel 카이제곱(연속성 보정), 공통 승산비
    num_a = num_e = num_v = 0.0
    or_num = or_den = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n == 0:
            continue
        r1, c1 = a + b, a + c
        num_a += a
        num_e += r1 * c1 / n
        num_v += r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1)) if n > 1 else 0
        or_num += a * d / n
        or_den += b * c / n
    chi2 = (abs(num_a - num_e) - 0.5) ** 2 / num_v
    return float(chi2), float(1 - stats.chi2.cdf(chi2, 1)), or_num / or_den


out = {}

# ---------- T1 Table II: LIBERO ----------
suites = ["Spatial", "Object", "Goal", "Long"]
imle10 = [98.0, 99.8, 98.2, 96.0]
pi05_10 = [97.2, 99.0, 97.8, 96.0]
imle30 = [97.2, 99.0, 96.2, 95.8]
pi05_30 = [95.0, 98.0, 96.2, 95.2]
cogvla = [99.0, 99.0, 97.0, 95.0]
N = 500
t1 = {}
for label, a, b in [("IMLE H10 vs pi0.5 H10 (cited)", imle10, pi05_10),
                     ("IMLE H30 vs pi0.5 H30 (cited)", imle30, pi05_30)]:
    ka = [count(x, N) for x in a]
    kb = [count(x, N) for x in b]
    ps = [fisher(ka[i], N, kb[i], N) for i in range(4)]
    adj = holm(ps)
    K1, K2 = sum(ka), sum(kb)
    rows = []
    for i, s in enumerate(suites):
        rows.append({"suite": s, "imle_k": ka[i], "pi05_k": kb[i], "imle_pct": a[i], "pi05_pct": b[i],
                     "diff_pp": round(a[i] - b[i], 2), "fisher_p": ps[i], "holm_p": adj[i],
                     "imle_wilson": [round(100 * v, 2) for v in wilson(ka[i], N)],
                     "pi05_wilson": [round(100 * v, 2) for v in wilson(kb[i], N)]})
    pooled = fisher(K1, 4 * N, K2, 4 * N)
    t1[label] = {"rows": rows, "pooled_imle_k": K1, "pooled_pi05_k": K2, "pooled_n": 4 * N,
                 "pooled_imle_pct": round(100 * K1 / (4 * N), 2), "pooled_pi05_pct": round(100 * K2 / (4 * N), 2),
                 "pooled_fisher_p": pooled,
                 "pooled_diff_pp": round(100 * (K1 - K2) / (4 * N), 2),
                 "pooled_imle_wilson": [round(100 * v, 2) for v in wilson(K1, 4 * N)]}
# 최강 기준선(CogVLA 97.5%)과 전체 비교
k_imle = sum(count(x, N) for x in imle10)
k_cog = sum(count(x, N) for x in cogvla)
t1["IMLE H10 vs CogVLA (cited)"] = {"imle_k": k_imle, "cogvla_k": k_cog, "n": 4 * N,
                                    "fisher_p": fisher(k_imle, 4 * N, k_cog, 4 * N),
                                    "diff_pp": round(100 * (k_imle - k_cog) / (4 * N), 2)}
out["T1_table2"] = t1

# ---------- T2: internal consistency of Table II average ----------
t2 = {}
for label, arr, avg in [("IMLE H10", imle10, 98.0), ("pi0.5 H10", pi05_10, 97.5),
                        ("IMLE H30", imle30, 97.1), ("pi0.5 H30", pi05_30, 96.1),
                        ("CogVLA", cogvla, 97.5)]:
    m = sum(arr) / 4
    t2[label] = {"mean_of_suites": round(m, 3), "table_avg": avg, "ok": abs(m - avg) < 0.051}
out["T2_avg_check"] = t2

# ---------- T3: minimum detectable difference ----------
t3 = {}
for n in (20, 500, 2000):
    for p in (0.85, 0.975):
        t3[f"n={n},p={p}"] = round(100 * mdd(p, n), 2)
out["T3_mdd_pp"] = t3

# ---------- T4: Table V (real robot, 20 trials per task) ----------
tasks = ["Pineapple in bowl", "Swap pineapple & cube", "Pineapple in cabinet", "Pineapple on moving plate"]
ours = [19, 18, 15, 16]
pi05 = [15, 15, 12, 12]
n_r = 20
ps = [fisher(ours[i], n_r, pi05[i], n_r) for i in range(4)]
adj = holm(ps)
rows = []
for i, t in enumerate(tasks):
    rows.append({"task": t, "imle": f"{ours[i]}/{n_r}", "pi05": f"{pi05[i]}/{n_r}",
                 "fisher_p": round(ps[i], 4), "holm_p": round(adj[i], 4),
                 "imle_wilson": [round(100 * v, 1) for v in wilson(ours[i], n_r)],
                 "pi05_wilson": [round(100 * v, 1) for v in wilson(pi05[i], n_r)]})
tables = [(ours[i], n_r - ours[i], pi05[i], n_r - pi05[i]) for i in range(4)]
chi2, pcmh, orm = mh_cmh([(a, b, c, d) for a, b, c, d in tables])
pool_o, pool_p = sum(ours), sum(pi05)
out["T4_table5"] = {"rows": rows, "pooled": f"{pool_o}/80 vs {pool_p}/80",
                    "pooled_fisher_p": round(fisher(pool_o, 80, pool_p, 80), 4),
                    "cmh_chi2": round(chi2, 3), "cmh_p": round(pcmh, 4), "cmh_or": round(orm, 3),
                    "mdd_pp_n20_p0.7": round(100 * mdd(0.7, 20), 1)}

# ---------- T5: latency arithmetic ----------
fps = {"pi0.5 JAX (Table I)": 15, "PyTorch+compile (Table I)": 20, "Triton pi0.5 (Table I)": 25, "IMLE-VLA (Table I)": 55}
ms = {k: 1000 / v for k, v in fps.items()}
t5 = {"ms_per_pass": {k: round(v, 2) for k, v in ms.items()},
      "ratio_vs_jax": {k: round(55 / 15 if "IMLE" in k else 1, 3) for k in fps},
      "imle_vs_compile": round(55 / 20, 3), "imle_vs_triton_pi05": round(55 / 25, 3),
      "imle_vs_jax": round(55 / 15, 3)}
# Table IV: 패스 수와 VLA-WC로 패스당 시간 역산
t4 = {"pi05_H10": (15.4, 1.03), "ours_H10": (15.5, 0.28), "ours_H30": (5.4, 0.10)}
per = {k: round(1000 * wc / passes, 2) for k, (passes, wc) in t4.items()}
t5["table4_ms_per_pass"] = per
t5["table4_ratio_wc"] = {"H10_vs_pi05": round(1.03 / 0.28, 3), "H30_vs_pi05": round(1.03 / 0.10, 3),
                         "paper_says": [3.6, 10.5]}
t5["table4_decomp_H30"] = {"pass_ratio": round(15.4 / 5.4, 3),
                           "per_pass_ratio": round(per["pi05_H10"] / per["ours_H30"], 3)}
# Table III: 처리량 = fps × H
thr = {"pi05 H10": 15 * 10, "ours H10": 55 * 10, "ours H30": 55 * 30}
t5["table3_throughput"] = thr
t5["table3_ratio"] = {"ours_H10": round(thr["ours H10"] / thr["pi05 H10"], 3),
                      "ours_H30": round(thr["ours H30"] / thr["pi05 H10"], 3)}
# Table V: 배수
vla = [(18.56, 123.0), (31.1, 178.3), (99.4, 389.4), (16.7, 89.5)]
t5["table5_wc_ratio"] = [round(b / a, 2) for a, b in vla]
t5["table5_implied_passes"] = {"pi05_at_67ms": [round(b / 0.0667) for _, b in vla],
                               "imle_at_18ms": [round(a / 0.0182) for a, _ in vla]}
# DROID 1 에피소드 길이 대조: 15 Hz 제어 기준 행동 수 = 패스 수 × H
t5["table5_actions_implied_at_15Hz_s"] = {
    "pi05_H8_seconds": [round(b / 0.0667 * 8 / 15, 0) for _, b in vla],
    "imle_H12_seconds": [round(a / 0.0182 * 12 / 15, 0) for a, _ in vla]}
out["T5_latency"] = t5

# ---------- T6: decomposition under explicit assumptions ----------
# 가정 A: 같은 엔진(Dexmal Triton 계열)에서 P + 10a = 40 ms(25 Hz), IMLE 엔진의 P + a = 18.2 ms(55 Hz)
#        P: 접두부(VLM)가 두 방법에서 같다, a: 행동 전문가 1회가 같다.
t_flow, t_imle = 40.0, 1000 / 55
a = (t_flow - t_imle) / 9
P = t_flow - 10 * a
out["T6_decomp"] = {"assumption": "same engine efficiency, same prefix cost P, same expert pass cost a",
                    "a_ms": round(a, 2), "P_ms": round(P, 2),
                    "head_share_of_imle_pct": round(100 * a / t_imle, 1),
                    "ceiling_if_head_free_x": round(t_imle / P, 3),
                    "algorithmic_speedup_same_engine_x": round(t_flow / t_imle, 3)}
# 민감도: 접두부 P를 바꿀 때 (총 40 ms 고정 기준 flow, IMLE = P + a)
sens = []
for Pv in (5, 10, 15.8, 20, 25, 30):
    av = (40 - Pv) / 10
    sens.append({"P_ms": Pv, "a_ms": round(av, 2), "imle_ms": round(Pv + av, 2),
                 "speedup_x": round(40 / (Pv + av), 2)})
out["T6_sensitivity"] = sens
# JAX 기준선: 같은 가정에서 JAX 66.7 ms, IMLE 18.2 ms 사이의 차이가 엔진에 얼마나 의존하는지
# 3.67x = (JAX 66.7 ms -> Triton pi0.5 40 ms: 엔진 요인 1.67x) × (Triton 10단계 40 ms -> IMLE 18.2 ms: 헤드 요인 2.2x)
# 로그 기준 비중: 엔진 요인과 헤드 요인
_eng = math.log(1000 / 15 / 40)
_head = math.log(40 / t_imle)
out["T6_factor_split"] = {"engine_factor_x": round((1000 / 15) / 40, 3),
                          "head_factor_x": round(40 / t_imle, 3),
                          "engine_log_share_pct": round(100 * _eng / (_eng + _head), 1),
                          "head_log_share_pct": round(100 * _head / (_eng + _head), 1)}

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
print(json.dumps({k: out[k] for k in ["T1_table2"]}, ensure_ascii=False, indent=1, default=float)[:3000])
print("T2", out["T2_avg_check"])
print("T3", out["T3_mdd_pp"])
print("T4", json.dumps(out["T4_table5"], ensure_ascii=False, default=float))
print("T5", json.dumps(out["T5_latency"], ensure_ascii=False, default=float))
print("T6", json.dumps(out["T6_decomp"], ensure_ascii=False))
print("T6s", out["T6_sensitivity"])
print("T6f", out["T6_factor_split"])
