"""BitVLA (arXiv:2506.07530 v2) 표와 본문 수치를 다시 계산하고 일관성을 검사한다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (numpy, scipy, 수 초)

검사 항목(V1–V12):
  V1  표 I 행별 네 스위트 평균 대 보고 평균
  V2  표 I의 LIBERO 성공률이 스위트당 500회(0.2 %p 단위) 정수 비율인지
  V3  표 I 메모리 열 배율(= GB / 1.4)과 파라미터 수 x 2 byte 관계
  V4  표 II 행별 평균, 메모리 배율, "1/3 미만" 주장
  V5  본문 차이 수치(7.6, 5.2, 1.1, 11x, 29.8 %)
  V6  LIBERO 스위트별 Wilson 구간과 BitVLA 대 OpenVLA-OFT, 대 π0, 사전학습 유무의 검정
  V7  표 III 평균과 본문의 1.5 %p, 42.4 대 50.8, 0.8 GB 대 0.1 GB
  V8  그림 4, 5 막대 값(그림 판독)으로 평균과 시행 수 정수 가능성, 그림 4 평균 막대와의 차이
  V9  실로봇 과제별 Fisher 검정과 검출 한계(시행 수 20 가정)
  V10 그림 6 지연 대 처리량(= 청크 길이 25 / 지연)과 배율
  V11 부록 표 IV, V의 스텝 x 배치 대비 데이터 규모와 GPU 시간
  V12 메모리 1.4 GB의 부품별 합(공개 설정과 GGUF 파일 크기 메타데이터 기준)

가정과 출처는 각 항목 옆에 적었다. 그림 판독 값은 "그림 판독"으로 표시한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h) * 100, (c + h) * 100


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def newcombe(k1, n1, k2, n2, z=1.959964):
    l1, u1 = (x / 100 for x in wilson(k1, n1, z))
    l2, u2 = (x / 100 for x in wilson(k2, n2, z))
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo * 100, hi * 100


# ---------------------------------------------------------------- 표 I
TABLE1 = [
    # 이름, 크기(B), 메모리(GB), 배율, Spatial, Object, Goal, Long, 보고 평균
    ("OpenVLA", 7.5, 15.1, 10.79, 84.7, 88.4, 79.2, 53.7, 76.5),
    ("CoT-VLA", 8.0, 16.2, 11.57, 87.5, 91.6, 87.6, 69.0, 81.1),
    ("UniVLA", 8.5, 17.0, 12.14, 96.5, 96.8, 95.6, 92.0, 95.2),
    ("UnifiedVLA", 8.5, 17.0, 12.14, 95.4, 98.8, 93.6, 94.0, 95.5),
    ("OpenVLA-OFT", 7.7, 15.4, 11.00, 97.6, 98.4, 97.9, 94.5, 97.1),
    ("SpatialVLA", 4.2, 8.5, 6.07, 88.2, 89.9, 78.6, 55.5, 78.1),
    ("NORA-Long", 3.8, 7.5, 5.36, 92.2, 95.4, 89.4, 74.6, 87.9),
    ("4D-VLA", 4.1, 8.3, 5.93, 88.9, 95.2, 90.9, 79.1, 88.6),
    ("SmolVLA", 2.3, 4.6, 3.29, 93.0, 94.0, 91.0, 77.0, 88.8),
    ("GROOT-N1", 2.2, 4.4, 3.14, 94.4, 97.6, 93.0, 90.6, 93.9),
    ("pi0", 3.5, 7.0, 5.00, 96.8, 98.8, 95.8, 85.2, 94.2),
    ("BitVLA w/o pre-training", 3.0, 1.4, 1.00, 97.4, 99.6, 94.4, 87.6, 94.8),
    ("BitVLA", 3.0, 1.4, 1.00, 96.6, 99.0, 95.4, 92.8, 96.0),
]

def feasible(v, n):
    """보고 값(소수 첫째 자리 반올림)이 n회 시행의 정수 성공 수 k에서 나올 수 있는가"""
    return any(abs(round(k / n * 100 + 1e-9, 1) - v) < 1e-9 for k in range(n + 1))


v1, v2, v3 = [], [], []
for name, size, mem, ratio, s, o, g, l, avg in TABLE1:
    m = (s + o + g + l) / 4
    v1.append({"row": name, "mean4": round(m, 3), "reported": avg, "diff": round(avg - m, 3),
               "mismatch_gt_0.06": abs(avg - m) > 0.06})
    cells = {"Spatial": s, "Object": o, "Goal": g, "Long": l}
    feas = {k: {"n500_ok": feasible(v, 500), "n1500_ok": feasible(v, 1500)} for k, v in cells.items()}
    v2.append({"row": name, "feasible": feas,
               "n500_all": all(f["n500_ok"] for f in feas.values()),
               "n1500_all": all(f["n1500_ok"] for f in feas.values())})
    v3.append({"row": name, "mem_over_1.4": round(mem / 1.4, 2), "reported_ratio": ratio,
               "ratio_ok": abs(mem / 1.4 - ratio) < 0.006, "size_x2": round(size * 2, 2), "mem_gb": mem})
R["V1_table1_row_means"] = v1
R["V2_table1_integer_feasibility"] = v2
R["V3_table1_memory_ratio"] = v3

# ---------------------------------------------------------------- 표 II
TABLE2 = [
    ("INT8 OpenVLA", 7.4, 5.29, 86.4, 85.2, 77.2, 58.8, 76.9),
    ("INT8 OpenVLA-OFT", 7.7, 5.50, 98.8, 98.0, 96.6, 94.0, 96.7),
    ("INT4 OpenVLA", 4.4, 3.14, 83.0, 84.0, 72.0, 51.6, 72.7),
    ("INT4 OpenVLA-OFT", 4.7, 3.36, 98.2, 98.2, 97.2, 93.8, 96.9),
    ("BitVLA", 1.4, 1.00, 96.6, 99.0, 95.4, 92.8, 96.0),
]
v4 = []
for name, mem, ratio, s, o, g, l, avg in TABLE2:
    m = (s + o + g + l) / 4
    v4.append({"row": name, "mean4": round(m, 3), "reported": avg, "diff": round(avg - m, 3),
               "mismatch_gt_0.06": abs(avg - m) > 0.06, "mem_ratio_calc": round(mem / 1.4, 2), "mem_ratio_reported": ratio})
R["V4_table2"] = {"rows": v4, "bitvla_mem_over_int4_oft": round(1.4 / 4.7, 4), "one_third": round(1 / 3, 4)}

# ---------------------------------------------------------------- 본문 수치
R["V5_text_claims"] = {
    "long_vs_pi0_pp": round(92.8 - 85.2, 2),          # 본문 7.6
    "pretrain_gain_long_pp": round(92.8 - 87.6, 2),   # 본문 5.2
    "avg_gap_vs_oft_pp": round(97.1 - 96.0, 2),       # 본문 1.1
    "mem_ratio_reported_gb": round(15.4 / 1.4, 2),    # 11.0
    "mem_ratio_with_gguf_1.44gb": round(15.4 / 1.4402, 2),
    "v1_mem_share_vs_int4_oft": round(1.4 / 4.7 * 100, 1),  # v1 초록 29.8 %
    "v1_long_vs_pi0_pp": round(87.6 - 85.2, 2),       # v1 본문 2.4
    "v1_oft_pretrain_long_gain": round(94.5 - 86.5, 2),
}

# ---------------------------------------------------------------- V6 LIBERO 검정 (스위트당 500회 가정)
N = 500
rows = {r[0]: r[4:8] for r in TABLE1}
rows["OpenVLA-OFT INT4"] = [98.2, 98.2, 97.2, 93.8]  # 표 II
rows["OpenVLA-OFT INT8"] = [98.8, 98.0, 96.6, 94.0]
rows["OFT v1 사전학습 없음"] = [94.3, 95.2, 91.7, 86.5]  # v1 표 1의 행(v2에서 삭제)
names = ["Spatial", "Object", "Goal", "Long"]


def cnt(p, n=N):
    return int(round(p / 100 * n))


v6 = {"assumption": "스위트당 500회(50회 x 10과제, 공식 평가 스크립트 기본값). OpenVLA-OFT 행은 비정수 값이 있어 가장 가까운 정수로 반올림",
      "wilson_95": {}, "pairs": []}
for model in ["BitVLA", "BitVLA w/o pre-training", "OpenVLA-OFT", "pi0"]:
    v6["wilson_95"][model] = {nm: [round(x, 1) for x in wilson(cnt(v), N)] for nm, v in zip(names, rows[model])}
    k = sum(cnt(v) for v in rows[model])
    v6["wilson_95"][model]["Avg(2000)"] = [round(x, 1) for x in wilson(k, 4 * N)]


def pair(a, b, label, na=N, nb=N):
    out = {"pair": label, "suites": {}}
    for nm, va, vb in zip(names, rows[a], rows[b]):
        ka, kb = cnt(va, na), cnt(vb, nb)
        out["suites"][nm] = {"a": va, "b": vb, "diff_pp": round(va - vb, 2), "fisher_p": round(fisher(ka, na, kb, nb), 4),
                             "newcombe95_pp": [round(x, 1) for x in newcombe(ka, na, kb, nb)]}
    ka = sum(cnt(v, na) for v in rows[a])
    kb = sum(cnt(v, nb) for v in rows[b])
    out["avg"] = {"diff_pp": round(ka / (4 * na) * 100 - kb / (4 * nb) * 100, 2),
                  "fisher_p": round(fisher(ka, 4 * na, kb, 4 * nb), 4),
                  "newcombe95_pp": [round(x, 1) for x in newcombe(ka, 4 * na, kb, 4 * nb)]}
    return out


v6["pairs"].append(pair("BitVLA", "OpenVLA-OFT", "BitVLA 대 OpenVLA-OFT (500회)"))
# OFT를 3시드 평균(1500회)로 보는 민감도
oft_alt = pair("BitVLA", "OpenVLA-OFT", "BitVLA(500) 대 OFT(1500, 3시드 가정)", nb=1500)
v6["pairs"].append(oft_alt)
v6["pairs"].append(pair("BitVLA", "pi0", "BitVLA 대 π0"))
v6["pairs"].append(pair("BitVLA", "OpenVLA-OFT INT4", "BitVLA 대 OpenVLA-OFT INT4 (표 II)"))
v6["pairs"].append(pair("BitVLA", "BitVLA w/o pre-training", "BitVLA(사전학습) 대 BitVLA w/o pre-training"))
v6["pairs"].append(pair("BitVLA w/o pre-training", "OpenVLA-OFT", "BitVLA w/o pre-training 대 OpenVLA-OFT"))
v6["pairs"].append(pair("BitVLA w/o pre-training", "OFT v1 사전학습 없음", "BitVLA w/o pre-training 대 OFT 사전학습 없음 (v1 표 1, v2에는 없음)"))
# 동등성(비열등) 여유: 평균 격차 1.1 %p의 95% 구간 상한
ka, kb = sum(cnt(v) for v in rows["BitVLA"]), sum(cnt(v) for v in rows["OpenVLA-OFT"])
v6["noninferiority"] = {"avg_diff_pp": round((ka - kb) / 2000 * 100, 2),
                        "newcombe95_pp": [round(x, 1) for x in newcombe(ka, 2000, kb, 2000)],
                        "note": "스위트당 500회 가정 시 평균 격차의 95% 구간(2000회 대 2000회)이 0을 걸칠 듯 말 듯하다. 반올림 때문에 보고 평균 1.1과 약간 다르다"}
# 최소 검출 차이(MDE): 스위트당 500회, 80% 검정력, 양측 5%, p ≈ 0.95 근방
def mde(p0, n, power=0.8, alpha=0.05):  # 정규 근사. p0가 작은 n에서는 하한 부정확
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    for d in np.arange(0.002, 0.6, 0.001):
        p1 = min(p0 + d, 0.9999)
        se = math.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
        if d / se >= za + zb:
            return float(d * 100)
    return None


v6["mde_pp"] = {"p0=0.95, n=500": mde(0.95, 500), "p0=0.95, n=2000": mde(0.95, 2000),
                "p0=0.90, n=500": mde(0.90, 500), "p0=0.85, n=500": mde(0.85, 500)}
R["V6_libero_tests"] = v6

# ---------------------------------------------------------------- V7 표 III
T3 = {
    "BF16 teacher": [37.4, 70.6, 45.0, 43.6, 68.6, 53.0],
    "1.58b, 10B, Laux": [35.4, 69.3, 43.7, 41.5, 67.6, 51.5],
    "1.58b, 5B, Laux": [33.3, 69.1, 43.3, 41.4, 66.4, 50.8],
    "1.58b, 5B, no Laux": [32.4, 52.9, 38.8, 30.7, 57.5, 42.4],
}
v7 = {"rows": {}}
for k, v in T3.items():
    m = sum(v[:5]) / 5
    v7["rows"][k] = {"mean5": round(m, 3), "reported": v[5], "diff": round(v[5] - m, 3)}
v7["drop_teacher_to_10B_pp"] = round(53.0 - 51.5, 2)
v7["laux_gain_5B_pp"] = round(50.8 - 42.4, 2)
v7["tokens_10B_vs_5B_pp"] = round(51.5 - 50.8, 2)
v7["vit_mem_ratio"] = {"bf16_gb": 0.8, "w158_gb": 0.1, "ratio": 8.0}
v7["seedbench_drop_without_laux_pp"] = round(69.1 - 52.9, 2)
R["V7_table3"] = v7

# ---------------------------------------------------------------- V8 실로봇 (그림 4, 5 막대 값)
# 순서: [과제1, 과제2, 과제3]. 그림 4 = 수박 잡기, 빵 담기, 종 세우기. 그림 5 = 스펀지 잡기, 수박 담기, 종 세우기(식탁보)
FIG4 = {"BitVLA w/o PT": [0, 0, 22.5], "BitVLA": [85, 80, 50], "pi0": [75, 70, 45], "OpenVLA-OFT": [75, 95, 65]}
FIG5 = {"BitVLA w/o PT": [5, 15, 0], "BitVLA": [75, 60, 40], "pi0": [50, 35, 40], "OpenVLA-OFT": [55, 77.5, 55]}
# 그림 판독 값: 평균 막대(그림 4, 그림 5의 오른쪽 패널)를 300 dpi로 렌더링해 눈금(0, 20, ..., 80)에서 읽음
FIG4_AVG_READ = {"BitVLA w/o PT": 1.3, "BitVLA": 71.0, "pi0": 53.0, "OpenVLA-OFT": 77.7}
FIG5_AVG_READ = {"BitVLA w/o PT": 6.5, "BitVLA": 58.0, "pi0": 42.0, "OpenVLA-OFT": 62.5}
n_real = 20  # 4 x 5 격자(부록 IV-C), 격자점당 1회로 가정
v8 = {"assumption": "4x5 격자 20점, 점당 1회(시행 수 n = 20 가정). 부분 점수(0.5)가 있는 과제는 점수 합 / 20",
      "fig4": {}, "fig5": {}}
for fig, data, avg_read in (("fig4", FIG4, FIG4_AVG_READ), ("fig5", FIG5, FIG5_AVG_READ)):
    for m, vals in data.items():
        mean = sum(vals) / 3
        pts = [v / 100 * n_real for v in vals]
        v8[fig][m] = {"values": vals, "mean_of_labels": round(mean, 2), "avg_bar_read(그림 판독)": avg_read[m],
                      "bar_minus_label_mean": round(avg_read[m] - mean, 2),
                      "points_of_20": pts, "half_point_grid": all(abs(p * 2 - round(p * 2)) < 1e-9 for p in pts)}
R["V8_real_world_figures"] = v8

# ---------------------------------------------------------------- V9 실로봇 검정 (n=20 가정)
def k_of(v):  # 점수 합(0.5 단위)을 가장 가까운 정수로 반올림해 Fisher에 넣는다(근사)
    return int(math.floor(v / 100 * n_real + 0.5))


tasks4 = ["수박 잡기", "빵 담기", "종 세우기"]
tasks5 = ["스펀지 잡기(OOD)", "수박 바구니(OOD)", "종 세우기+식탁보(OOD)"]
v9 = {"assumption": "n = 20/셀. 0.5 단위 부분 점수는 반올림한 정수 성공으로 근사(근사 오차가 p의 자릿수는 바꾸지 않음)",
      "cells": [], "pooled": []}
for fig, data, tasks in (("fig4", FIG4, tasks4), ("fig5", FIG5, tasks5)):
    for ti, t in enumerate(tasks):
        for other in ("pi0", "OpenVLA-OFT", "BitVLA w/o PT"):
            ka, kb = k_of(data["BitVLA"][ti]), k_of(data[other][ti])
            v9["cells"].append({"fig": fig, "task": t, "pair": f"BitVLA 대 {other}",
                                "a": data["BitVLA"][ti], "b": data[other][ti], "fisher_p": round(fisher(ka, n_real, kb, n_real), 4)})
    for other in ("pi0", "OpenVLA-OFT"):
        ka = sum(data["BitVLA"]) / 100 * n_real
        kb = sum(data[other]) / 100 * n_real
        ka_i, kb_i = int(round(ka)), int(round(kb))
        v9["pooled"].append({"fig": fig, "pair": f"BitVLA 대 {other}", "a_pct": round(sum(data["BitVLA"]) / 3, 1),
                             "b_pct": round(sum(data[other]) / 3, 1),
                             "fisher_p_60": round(fisher(ka_i, 3 * n_real, kb_i, 3 * n_real), 4),
                             "newcombe95_pp": [round(x, 1) for x in newcombe(ka_i, 60, kb_i, 60)]})
v9["mde_pp_n20"] = {"p0=0.5": mde(0.5, 20), "p0=0.75": mde(0.75, 20), "p0=0.5, n=60": mde(0.5, 60)}
# 정확 검출 한계: n=20에서 80% 검정력으로 0.75 대 x를 구분하는 최소 격차를 Fisher 기준으로 계산
def fisher_power(p1, p2, n, sims=4000, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        a, b = rng.binomial(n, p1), rng.binomial(n, p2)
        hits += fisher(a, n, b, n) < 0.05
    return hits / sims


v9["fisher_power_n20"] = {f"{p1:.2f} 대 {p2:.2f}": round(fisher_power(p1, p2, 20), 3)
                          for p1, p2 in ((0.85, 0.75), (0.85, 0.50), (0.85, 0.40), (0.75, 0.35))}
R["V9_real_world_tests"] = v9

# ---------------------------------------------------------------- V10 지연 대 처리량
LAT = {"BitVLA": (73, 341.1), "OpenVLA-OFT+": (321, 77.9), "pi0": (86, 291.6), "RDT-1B": (297, 84.1), "Diffusion Policy": (90, 267.4)}
K = 25
v10 = {"chunk_K": K, "rows": {}}
for m, (lat, thr) in LAT.items():
    v10["rows"][m] = {"latency_ms": lat, "throughput_hz": thr, "K_over_latency": round(K / (lat / 1000), 1),
                      "implied_latency_ms": round(K / thr * 1000, 1),
                      "rel_diff_pct": round((K / (lat / 1000) - thr) / thr * 100, 2)}
v10["speedup_latency"] = round(321 / 73, 3)
v10["speedup_throughput"] = round(341.1 / 77.9, 3)
v10["readme_robotcpp"] = {"eager_ms": 377.8, "compile_ms": 209.2, "robotcpp_3090_ms": 39,
                          "x_vs_eager": round(377.8 / 39, 2), "x_vs_compile": round(209.2 / 39, 2),
                          "hz_4090_17ms": round(1000 / 17, 1)}
R["V10_latency_throughput"] = v10

# ---------------------------------------------------------------- V11 부록 하이퍼파라미터 대비 규모
S1_steps, S2_steps, S3_steps, bs = 25_000, 40_000, 20_000, 256
v11 = {
    "stage1_samples_M": S1_steps * bs / 1e6, "stage1_epochs_over_558k": round(S1_steps * bs / 558_000, 2),
    "stage2_samples_M": S2_steps * bs / 1e6, "stage2_epochs_over_10M": round(S2_steps * bs / 10e6, 2),
    "stage3_samples_M": S3_steps * bs / 1e6, "stage3_tokens_B_at_seq2048": round(S3_steps * bs * 2048 / 1e9, 2),
    "stage3_5B_tokens_steps": 10_000, "vlm_total_steps": S1_steps + S2_steps + S3_steps,
    "vlm_gpu_hours": 7 * 24 * 8, "vlm_v1_gpu_hours_a100": 14 * 24 * 8,
    "pretrain_samples_seen_M": 200_000 * 2048 / 1e6, "pretrain_gpu_hours_h800": 14 * 24 * 16,
    "pretrain_sec_per_step": round(14 * 86400 / 200_000, 2),
    "pretrain_samples_per_sec_per_gpu": round(2048 / (14 * 86400 / 200_000) / 16, 1),
    "libero_ft_batch": 2 * 8 * 4,  # 공식 스크립트: batch 2 x 누적 8 x GPU 4
    "libero_ft_samples_10k_steps_M": 10_000 * 64 / 1e6,
    "libero_long_100k_steps_samples_M": 100_000 * 64 / 1e6,
    "libero_demos_per_suite": 500,
}
# 체크포인트 config의 norm_stats(27개 OXE 데이터셋)에서 읽은 궤적 수와 전이 수(data/lxsy_bitvla_bf16_config.json)
v11["oxe_norm_stats"] = {"datasets": 27, "trajectories": 957_161, "transitions": 56_532_302}
v11["pretrain_epochs_over_transitions"] = round(200_000 * 2048 / 56_532_302, 2)
v11["pretrain_passes_over_trajectories"] = round(200_000 * 2048 / 957_161, 0)
R["V11_hparam_scale"] = v11

# ---------------------------------------------------------------- V12 메모리 부품별 합
d, inter, L, nh, nkv, hd, vocab = 2560, 6912, 30, 20, 5, 128, 128268  # 공개 config(lxsy/bitvla-bf16)의 text_config
llm_lin = L * (d * nh * hd + 2 * d * nkv * hd + nh * hd * d + 3 * d * inter)
emb = vocab * d
vh, vi, vL, vp, vimg = 1152, 4304, 26, 14, 224  # vision_config
vit_lin = vL * (4 * vh * vh + 2 * vh * vi)
vit_other = 3 * vp * vp * vh + (vimg // vp) ** 2 * vh + vL * (4 * vh + 2 * vh) + vh * 4 + vL * (vh + vi + 3 * vh)  # 패치 임베딩, 위치 임베딩, 편향·LN 근사
conn = vh * d + d + d * d + d
total_hf_pretrained = 3_147_925_664      # HF 메타데이터(safetensors total, BF16)
total_hf_finetuned = 2_819_549_344       # HF 메타데이터(미세조정 체크포인트 safetensors total)
gguf = {"action": 118_237_952, "llm": 1_187_325_312, "proprio": 13_169_536, "vision": 121_491_936}
gguf_total = sum(gguf.values())
bf = 2 / 1e9
v12 = {
    "params": {"llm_linear": llm_lin, "embedding_one": emb, "vit_linear": vit_lin, "connector": conn,
               "sum_with_one_embedding": llm_lin + emb + vit_lin + conn + vit_other,
               "sum_with_two_embeddings": llm_lin + 2 * emb + vit_lin + conn + vit_other,
               "hf_pretrained_total": total_hf_pretrained, "hf_finetuned_total": total_hf_finetuned},
    "gb_if_ternary_1.58bit": round(llm_lin * 1.58 / 8 / 1e9 + emb * bf + vit_lin * 1.58 / 8 / 1e9 + (vit_other + conn) * bf, 3),
    "gb_if_2bit_packing": round(llm_lin * 2 / 8 / 1e9 + emb * bf + vit_lin * 2 / 8 / 1e9 + (vit_other + conn) * bf, 3),
    "gb_llm_linear_2bit": round(llm_lin * 2 / 8 / 1e9, 3), "gb_embedding_bf16": round(emb * bf, 3),
    "gb_vit_linear_2bit": round(vit_lin * 2 / 8 / 1e9, 3),
    "gguf_bytes": gguf, "gguf_total_bytes": gguf_total, "gguf_total_gb": round(gguf_total / 1e9, 4),
    "gguf_total_gib": round(gguf_total / 2 ** 30, 3),
    "gguf_llm_minus_2bit_linear_gb": round((gguf["llm"] - llm_lin * 2 / 8) / 1e9, 3),
    "hf_bf16_gb_pretrained": round(total_hf_pretrained * bf, 2), "hf_bf16_gb_finetuned": round(total_hf_finetuned * bf, 2),
    "hf_safetensors_file_bytes": 6_295_953_280,
    "ratio_oft_15.4_over_gguf": round(15.4e9 / gguf_total, 2),
    "ratio_oft_15.4_over_1.4": round(15.4 / 1.4, 2),
    "heads_gb": round((gguf["action"] + gguf["proprio"]) / 1e9, 3),
    "note": "임베딩 한 벌(bf16) + 선형층 2비트 + 비양자화 부분이 GGUF 합과 맞는다. lm_head는 미세조정 체크포인트와 GGUF에 없다(HF 메타데이터 2.82B)",
}
v12["gb_2bit_plus_heads"] = round(v12["gb_if_2bit_packing"] + v12["heads_gb"], 3)
v12["gb_ternary_ideal_plus_heads"] = round(v12["gb_if_ternary_1.58bit"] + v12["heads_gb"], 3)
v12["ratio_oft_over_ternary_ideal_plus_heads"] = round(15.4 / v12["gb_ternary_ideal_plus_heads"], 2)
v12["unaccounted_params_pretrained_vs_two_emb"] = total_hf_pretrained - v12["params"]["sum_with_two_embeddings"]
v12["unaccounted_params_finetuned_vs_one_emb"] = total_hf_finetuned - v12["params"]["sum_with_one_embedding"]
R["V12_memory_budget"] = v12

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")

# ---- 요약 출력
print("V1 불일치 행:", [x["row"] + f" ({x['reported']} 대 {x['mean4']})" for x in v1 if x["mismatch_gt_0.06"]])
print("V2 500회 정수 불가 행:", [x["row"] for x in v2 if not x["n500_all"]])
print("V3 배율 불일치:", [x["row"] for x in v3 if not x["ratio_ok"]])
print("V4 표 II 불일치:", [(x["row"], x["reported"], x["mean4"]) for x in v4 if x["mismatch_gt_0.06"]])
print("V6 BitVLA 대 OFT 평균:", v6["pairs"][0]["avg"], "Long:", v6["pairs"][0]["suites"]["Long"])
print("V6 BitVLA 대 π0 Long:", v6["pairs"][2]["suites"]["Long"])
print("V6 사전학습 평균:", v6["pairs"][3]["avg"], v6["pairs"][3]["suites"]["Long"])
print("V8 그림 4 평균 막대 - 라벨 평균:", {m: v["bar_minus_label_mean"] for m, v in v8["fig4"].items()})
print("V8 그림 5:", {m: v["bar_minus_label_mean"] for m, v in v8["fig5"].items()})
print("V9 pooled:", v9["pooled"])
print("V10:", {m: v["rel_diff_pct"] for m, v in v10["rows"].items()})
print("V12 GGUF 합:", v12["gguf_total_gb"], "GB; 1.58bit 추정", v12["gb_if_ternary_1.58bit"], "; 2bit 추정", v12["gb_if_2bit_packing"])
print("V12 미설명 파라미터:", v12["unaccounted_params_pretrained_vs_two_emb"], v12["unaccounted_params_finetuned_vs_one_emb"])
