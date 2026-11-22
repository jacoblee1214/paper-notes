"""논문 표 값을 다시 계산하고 검정한다 (arXiv:2502.19645 v2).

실행: python3 analysis/recompute.py   → analysis/recompute_results.json 과 요약 출력
입력: 논문 본문의 표 값(아래 상수, 원문 PDF의 표 I, II, III, X–XVI에서 옮김). 외부 데이터를 받지 않는다.

확인하는 것
  V1  표 I의 네 과제 평균이 보고된 평균과 맞는가 (반올림 범위 안인가)
  V2  표 I의 SR이 '시도 500회'의 정확한 비율로 나타날 수 있는가 (3시드 평균 여부 점검)
  V3  표 II, III의 처리량 = K / 지연 이 맞는가, 본문의 배율(26×, 43×, 4×, 17%)이 맞는가
  V4  LIBERO 평균 SR의 이항 95% 구간과 OFT 대 베이스라인 차이의 검정 (시도 수 가정 명시)
  V5  ALOHA 점수표(표 X–XIII)의 합계와 과제 평균, 'up to 15%' 주장
  V6  BridgeData 표 XVI의 과제 쌍 차이와 검정
  V7  표 XIV, XV의 차이 (단일 정책, FiLM, 사전학습 제거)
  V8  v1 대 v2 표 II의 확산 정책 처리량 차이
  V9  파라미터 수 (표 IV, V의 합계) 와 공식 설정 구조로부터의 추정
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

# ---------------------------------------------------------------- 표 I (v2): LIBERO SR (%)
# 열: Spatial, Object, Goal, Long, Average(보고값). None = 보고되지 않음
TABLE1 = {
    # 블록 1: 제3자 시점 영상 + 언어, 필터링된 학습 데이터
    "block1": {
        "Diffusion Policy (scratch)": ([78.3, 92.5, 68.3, 50.5], 72.4),
        "Octo (fine-tuned)": ([78.9, 85.7, 84.6, 51.1], 75.1),
        "DiT Policy (fine-tuned)": ([84.2, 96.3, 85.4, 63.8], 82.4),
        "OpenVLA (fine-tuned)": ([84.7, 88.4, 79.2, 53.7], 76.5),
        "OpenVLA + PD&AC": ([91.3, 92.7, 90.5, 86.5], 90.2),
        "OpenVLA + PD&AC, Cont-Diffusion": ([96.9, 98.1, 95.5, 91.1], 95.4),
        "OpenVLA-OFT (제3자 영상만)": ([96.2, 98.3, 96.2, 90.7], 95.3),
    },
    # 블록 2: 제3자 + 손목 카메라 + 자세, 필터링 없는 원본 데이터
    "block2": {
        "MDT (scratch, 언어 2%)": ([78.5, 87.5, 73.5, 64.8], 76.1),
        "MDT (언어 100%)": ([95.2, 97.8, None, 83.0], None),
        "Seer (scratch)": ([None, None, None, 78.7], None),
        "Seer (LIBERO-90 사전학습)": ([None, None, None, 87.7], None),
        "OpenVLA-OFT (원본 데이터)": ([95.2, 94.2, 95.2, 93.2], 94.5),
    },
    # 블록 3: 제3자 + 손목 + 자세, 필터링 데이터 (논문 헤드라인 비교)
    "block3": {
        "pi0+FAST (fine-tuned)": ([96.4, 96.8, 88.6, 60.2], 85.5),
        "pi0 (fine-tuned)": ([96.8, 98.8, 95.8, 85.2], 94.2),
        "OpenVLA-OFT": ([97.6, 98.4, 97.9, 94.5], 97.1),
    },
}
SUITES = ["Spatial", "Object", "Goal", "Long"]

# 표 XIV: 단일 정책 (네 스위트 합동 학습) 과 FiLM
TABLE14 = {
    "OFT (스위트별 정책, 표 I)": ([97.6, 98.4, 97.9, 94.5], 97.1),
    "OFT (단일 정책, 4개 합동)": ([97.7, 98.0, 96.1, 95.3], 96.8),
    "OFT+ (단일 정책, FiLM)": ([97.8, 98.2, 98.2, 93.8], 97.0),
}
# 표 XV: 사전학습 OpenVLA 표현 제거
TABLE15 = {
    "OpenVLA-OFT": ([97.6, 98.4, 97.9, 94.5], 97.1),
    "OpenVLA-OFT (scratch)": ([94.3, 95.2, 91.7, 86.5], 91.9),
}

# ---------------------------------------------------------------- 표 II (v2): LIBERO 효율
# (이름, 처리량 Hz, 지연 s, 지연 소수 자리수, K(청크 길이), 긴 과제 SR 또는 None)
TABLE2 = [
    ("OpenVLA", 4.2, 0.2396, 4, 1, 53.7),
    ("+ PD", 15.9, 0.0629, 4, 1, None),
    ("+ PD&AC", 108.8, 0.0735, 4, 8, 86.5),
    ("+ PD&AC, Cont-L1", 109.7, 0.0729, 4, 8, 90.7),
    ("+ PD&AC, Cont-Diffusion T=50", 4.2, 1.9070, 4, 8, 91.1),
    ("+ PD&AC, Cont-Diffusion T=10", 19.3, 0.4145, 4, 8, 91.0),
    ("+ PD&AC, Cont-Diffusion T=5", 35.1, 0.2279, 4, 8, 90.0),
    ("+ PD&AC, Cont-Diffusion T=2", 80.3, 0.0996, 4, 8, 85.7),
    ("+ PD&AC, Cont-Diffusion T=1", 109.4, 0.0731, 4, 8, 0.0),
    ("+ 추가 입력 (손목, 자세)", 71.4, 0.1120, 4, 8, 94.5),
]
# v1 표 II (확산 행만 다름)
TABLE2_V1_DIFF = {"+ PD&AC, Cont-Diffusion T=50": (10.1, 0.792)}

# ---------------------------------------------------------------- 표 III (v2): ALOHA 효율
# (이름, 처리량 Hz, 지연 s, 지연 소수 자리수, K)
TABLE3 = [
    ("OpenVLA (K=1)", 1.8, 0.543, 3, 1),
    ("OpenVLA-OFT+", 77.9, 0.321, 3, 25),
    ("RDT-1B", 84.1, 0.297, 3, 25),
    ("Diffusion Policy", 267.4, 0.090, 3, 24),
    ("pi0 (JAX)", 291.6, 0.086, 3, 25),
    ("ACT", 432.8, 0.058, 3, 25),
]

# ---------------------------------------------------------------- 표 X–XIII (v2): ALOHA 점수 (평균, 점)
METHODS = ["ACT", "Diffusion Policy", "RDT-1B", "pi0", "OpenVLA-OFT+"]
ALOHA_TASKS = {
    # 과제: (시도 수, 방법별 총점 (100점 만점))
    "fold shorts": (10, [100, 100, 100, 100, 100]),
    "fold shirt": (10, [94, 100, 96, 100, 100]),
    "scoop X into bowl": (12, [71.25, 79.17, 73.33, 93.33, 100.0]),
    "put X into pot": (24, [23.75, 30.83, 44.17, 42.08, 51.25]),
}
# 표 XII, XIII의 OpenVLA-OFT (FiLM 없음) 행
ALOHA_NOFILM = {"scoop X into bowl": 35.00, "put X into pot": 31.67}
# 표 XII 세부 점 (합계 검사용)
SCOOP_ROWS = {
    "ACT": [9.17, 9.17, 18.33, 15.00, 15.00, 5.00, -0.42, 0.00],
    "OpenVLA-OFT+": [10.00, 10.00, 20.00, 20.00, 20.00, 20.00, 0.00, 0.00],
    "OpenVLA-OFT (no FiLM)": [10.00, 9.17, 6.67, 5.00, 3.33, 1.67, -0.83, 0.00],
}
SCOOP_TOTAL = {"ACT": 71.25, "OpenVLA-OFT+": 100.0, "OpenVLA-OFT (no FiLM)": 35.00}
# 표 XIII 세부 점
POT_ROWS = {
    "ACT": [10.00, 6.67, 2.08, 1.67, 1.67, 1.67],
    "OpenVLA-OFT+": [10.00, 15.83, 6.25, 8.33, 5.83, 5.00],
    "OpenVLA-OFT (no FiLM)": [10.00, 6.67, 3.33, 5.00, 3.33, 3.33],
}
POT_TOTAL = {"ACT": 23.75, "OpenVLA-OFT+": 51.25, "OpenVLA-OFT (no FiLM)": 31.67}

# ---------------------------------------------------------------- 표 XVI (v2): BridgeData V2 (과제별 점수, 10회 평균)
BRIDGE = [
    # (범주, 과제, OpenVLA, OFT)
    ("Visual gen", "Put Eggplant into Pot", 60, 90),
    ("Motion gen", "Lift Eggplant", 70, 60),
    ("Physical gen", "Flip Pot Upright", 90, 80),
    ("Semantic gen", "Lift White Tape", 0, 20),
    ("Language", "Put {Blue Cup, Pink Cup} on Plate", 85, 90),
    ("Language", "Lift {Cheese, Red Chili Pepper}", 90, 75),
]

# ---------------------------------------------------------------- 설정 (파라미터 수, 표 IV/V)
PARAM_LIBERO = {"lora": 111e6, "head": 151e6, "proprio": 17e6, "total_reported": 279e6}
PARAM_ALOHA = {"lora": 111e6, "head": 269e6, "proprio": 17e6, "film": 456e6, "total_reported": 853e6}


def wilson(p, n, z=1.96):
    """윌슨 구간 (비율 p, 시도 수 n)."""
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def two_prop_z(p1, n1, p2, n2):
    """두 비율의 차이 검정 (풀링 z 검정, 양측). p는 0–1."""
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def main():
    res = {}

    # ---- V1: 표 I 평균 검증
    v1 = []
    for blk, rows in TABLE1.items():
        for name, (vals, avg) in rows.items():
            if avg is None or any(v is None for v in vals):
                continue
            mean = float(np.mean(vals))
            v1.append({"block": blk, "method": name, "mean_of_suites": round(mean, 3), "reported": avg,
                       "ok_within_rounding": abs(mean - avg) <= 0.05 + 1e-9})
    res["V1_table1_average_check"] = v1
    res["V1_all_ok"] = all(r["ok_within_rounding"] for r in v1)

    # ---- V2: 시도 500회 정확성 점검 (스위트별 SR에 대해 n 가정별 정수 여부)
    def fits(sr, n):
        """sr(%)이 n회 시도의 정확한 비율로 나올 수 있는가 (반올림 0.05%p 범위)."""
        k = sr / 100 * n
        return abs(k - round(k)) <= 0.0005 * n + 1e-9

    v2 = {}
    for n in (500, 1500):
        bad = []
        for blk, rows in TABLE1.items():
            for name, (vals, _) in rows.items():
                for s, v in zip(SUITES, vals):
                    if v is None:
                        continue
                    if not fits(v, n):
                        bad.append({"method": name, "suite": s, "sr": v, "n": n,
                                    "k_expected": round(v / 100 * n, 2)})
        v2[f"n={n}"] = {"cells_not_fitting": len(bad), "examples": bad[:8]}
    res["V2_trial_count_granularity"] = v2

    # ---- V3: 처리량 = K / 지연 (보고 반올림 범위 포함)
    v3 = {"table2": [], "table3": []}
    for name, thr, lat, d, K, _ in TABLE2:
        implied = K / lat
        lo = K / (lat + 0.5 * 10 ** (-d))
        hi = K / (lat - 0.5 * 10 ** (-d))
        v3["table2"].append({"row": name, "throughput_reported": thr, "K_over_latency": round(implied, 2),
                             "ok": lo - 0.05 <= thr <= hi + 0.05})
    for name, thr, lat, d, K in TABLE3:
        implied = K / lat
        lo = K / (lat + 0.5 * 10 ** (-d))
        hi = K / (lat - 0.5 * 10 ** (-d))
        v3["table3"].append({"row": name, "throughput_reported": thr, "K_over_latency": round(implied, 2),
                             "ok": lo - 0.05 <= thr <= hi + 0.05})
    v3["all_ok"] = all(r["ok"] for r in v3["table2"] + v3["table3"])
    # 본문 배율
    base_thr, base_lat = 4.2, 0.2396
    v3["ratios"] = {
        "PD&AC_vs_base_throughput": round(108.8 / base_thr, 2),          # 본문 26×
        "Cont-L1_vs_base_throughput": round(109.7 / base_thr, 2),       # 본문 26×
        "PD_vs_base_throughput": round(15.9 / base_thr, 2),              # 본문 4×
        "PD_vs_base_latency_speedup": round(base_lat / 0.0629, 2),       # 본문 4×
        "AC_latency_increase_pct": round((0.0735 / 0.0629 - 1) * 100, 1),  # 본문 17%
        "OFTplus_vs_base_aloha_throughput": round(77.9 / 1.8, 2),       # 본문 43×
        "OFTplus_vs_RDT_throughput_pct": round((84.1 - 77.9) / 84.1 * 100, 1),
        "OFT_latency_vs_base_speedup": round(base_lat / 0.0729, 2),
        "diff_T50_v1_vs_v2_throughput_ratio": round(4.2 / 10.1, 2),     # v1 10.1 Hz 대 v2 4.2 Hz
    }
    res["V3_efficiency"] = v3

    # ---- V4: LIBERO 차이 검정 (시도 수 가정)
    oft = TABLE1["block3"]["OpenVLA-OFT"]
    pi0 = TABLE1["block3"]["pi0 (fine-tuned)"]
    ovla = TABLE1["block1"]["OpenVLA (fine-tuned)"]
    pdac = TABLE1["block1"]["OpenVLA + PD&AC"]
    oft1 = TABLE1["block1"]["OpenVLA-OFT (제3자 영상만)"]
    diff_cont = TABLE1["block1"]["OpenVLA + PD&AC, Cont-Diffusion"]
    v4 = {"assumption": "스위트당 500 시도 (논문 표 I 주석). 베이스라인도 500으로 가정 (원 논문 확인 불가)"}

    def avg_counts(p_avg, n_per_suite):
        n = 4 * n_per_suite
        return p_avg / 100, n

    for n_per in (500, 1500):
        tag = f"n_per_suite={n_per}"
        p_oft, n_oft = avg_counts(97.1, n_per)
        out = {}
        for name, row in [("pi0", pi0), ("OpenVLA-FT", ovla), ("OpenVLA+PD&AC", pdac)]:
            p_b, n_b = avg_counts(row[1], n_per)
            z, pval = two_prop_z(p_oft, n_oft, p_b, n_b)
            out[name] = {"oft_minus_base_pp": round((p_oft - p_b) * 100, 2), "z": round(z, 2),
                         "p_two_sided": float(f"{pval:.3g}")}
        out["OFT_wilson95_pct"] = [round(x * 100, 2) for x in wilson(p_oft, n_oft)]
        v4[tag] = out
    # 스위트별 OFT 대 π0 (블록 3, 스위트당 시도 500)
    per = []
    for s, a, b in zip(SUITES, oft[0], pi0[0]):
        z, pval = two_prop_z(a / 100, 500, b / 100, 500)
        per.append({"suite": s, "oft": a, "pi0": b, "diff_pp": round(a - b, 1), "p_two_sided": float(f"{pval:.3g}")})
    v4["per_suite_oft_vs_pi0_n500"] = per
    # 제3자 영상만 OFT(95.3) 대 OpenVLA FT(76.5)
    p1, n1 = 0.953, 2000
    p2, n2 = 0.765, 2000
    z, pval = two_prop_z(p1, n1, p2, n2)
    v4["oft_third_person_vs_openvla_ft_pp"] = round((p1 - p2) * 100, 1)
    v4["oft_third_person_vs_openvla_ft_p"] = float(f"{pval:.3g}")
    # 연속 대 이산 (PD&AC 90.2 대 Cont-L1 95.3): 블록 1
    v4["cont_L1_minus_discrete_pp"] = round(oft1[1] - pdac[1], 1)   # 본문 5%
    v4["PDAC_minus_base_pp"] = round(pdac[1] - ovla[1], 1)          # 본문 14%
    v4["L1_minus_diffusion_pp"] = round(oft1[1] - diff_cont[1], 1)  # 본문 comparable
    z, pval = two_prop_z(0.953, 2000, 0.954, 2000)
    v4["L1_vs_diffusion_p_n500perSuite"] = float(f"{pval:.3g}")
    # 원본 데이터 OFT (94.5) 대 필터링 OFT (97.1)
    v4["filtered_minus_unfiltered_OFT_pp"] = round(97.1 - 94.5, 1)
    # 헤드라인 주장: OFT 대 MDT 100% (블록 2) 등 참고값
    res["V4_libero_tests"] = v4

    # ---- V5: ALOHA 점수
    v5 = {"tasks": {}}
    totals = {m: [] for m in METHODS}
    for task, (n, scores) in ALOHA_TASKS.items():
        v5["tasks"][task] = {"trials": n, "scores": dict(zip(METHODS, scores))}
        for m, s in zip(METHODS, scores):
            totals[m].append(s)
    avg = {m: round(float(np.mean(v)), 2) for m, v in totals.items()}
    v5["unweighted_task_mean"] = avg
    v5["oft_plus_minus"] = {m: round(avg["OpenVLA-OFT+"] - avg[m], 2) for m in METHODS if m != "OpenVLA-OFT+"}
    v5["per_task_oft_plus_minus_pi0"] = {
        t: round(ALOHA_TASKS[t][1][4] - ALOHA_TASKS[t][1][3], 2) for t in ALOHA_TASKS
    }
    # 합계 검사
    sums = {
        "scoop": all(abs(sum(SCOOP_ROWS[k]) - SCOOP_TOTAL[k]) < 0.02 for k in SCOOP_ROWS),
        "pot": all(abs(sum(POT_ROWS[k]) - POT_TOTAL[k]) < 0.02 for k in POT_ROWS),
    }
    v5["row_sum_checks"] = sums
    v5["no_film_scoop_pot"] = {"scoop": ALOHA_NOFILM["scoop X into bowl"], "pot": ALOHA_NOFILM["put X into pot"]}
    # 시도 수 (점수 기반이라 이항 구간을 쓸 수 없다)
    v5["note"] = "점수는 부분 점수의 평균이라 시도별 이진 결과가 없다. 구간 추정 불가. 과제당 시도 10–24회."
    res["V5_aloha"] = v5

    # ---- V6: BridgeData (표 XVI) 과제 쌍 차이
    ov = np.array([b[2] for b in BRIDGE], dtype=float)
    oft_b = np.array([b[3] for b in BRIDGE], dtype=float)
    d = oft_b - ov
    t_stat, t_p = stats.ttest_rel(oft_b, ov)
    res["V6_bridge"] = {
        "mean_openvla": round(ov.mean(), 2), "mean_oft": round(oft_b.mean(), 2),
        "mean_diff_pp": round(d.mean(), 2),
        "wins_losses_ties": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())],
        "paired_t_p": round(float(t_p), 3),
        "note": "과제당 10회 평균 점수 6개. 논문 본문 '평균 우위' 주장은 과제 수준에서 확인되지 않는다.",
    }

    # ---- V7: 표 XIV, XV
    def diffs_row(tab, a, b):
        return {s: round(x - y, 1) for s, x, y in zip(SUITES + ["Average"], tab[a][0] + [tab[a][1]],
                                                       tab[b][0] + [tab[b][1]])}
    res["V7_table14_15"] = {
        "t14_oft_minus_single": round(TABLE14["OFT (단일 정책, 4개 합동)"][1] - TABLE14["OFT (스위트별 정책, 표 I)"][1], 1),
        "t14_oftplus_minus_oft_single": round(TABLE14["OFT+ (단일 정책, FiLM)"][1] - TABLE14["OFT (단일 정책, 4개 합동)"][1], 1),
        "t15_oft_minus_scratch_avg": round(TABLE15["OpenVLA-OFT"][1] - TABLE15["OpenVLA-OFT (scratch)"][1], 1),
        "t15_per_suite_diff": diffs_row(TABLE15, "OpenVLA-OFT", "OpenVLA-OFT (scratch)"),
    }
    # 표 XV 스위트별 검정 (n=500 가정)
    per15 = []
    for s, a, b in zip(SUITES, TABLE15["OpenVLA-OFT"][0], TABLE15["OpenVLA-OFT (scratch)"][0]):
        z, pval = two_prop_z(a / 100, 500, b / 100, 500)
        per15.append({"suite": s, "diff": round(a - b, 1), "p": float(f"{pval:.3g}")})
    res["V7_table14_15"]["t15_per_suite_test_n500"] = per15

    # ---- V8: v1 대 v2 확산 행
    res["V8_v1_v2_diffusion"] = {
        "v1": {"throughput_hz": 10.1, "latency_s": 0.792, "K_over_latency": round(8 / 0.792, 2)},
        "v2": {"throughput_hz": 4.2, "latency_s": 1.9070, "K_over_latency": round(8 / 1.9070, 2)},
        "note": "두 판본 모두 내부적으로는 K/지연과 맞지만 같은 행의 값이 4.8배 다르다. 본문 문장은 v1에서 '2배 높다', v2에서 '같다'로 바뀌었다.",
    }

    # ---- V9: 파라미터 수
    libero_total = sum(PARAM_LIBERO[k] for k in ("lora", "head", "proprio"))
    aloha_total = sum(PARAM_ALOHA[k] for k in ("lora", "head", "proprio", "film"))
    # 구조로부터 계산: 행동 헤드 (MLPResNet, 블록 2개, 은닉 4096) 와 FiLM (ViT 블록마다 scale, shift)
    def head_params(in_dim, out_dim=7, hidden=4096, blocks=2):
        ln = 2 * in_dim
        fc1 = in_dim * hidden + hidden
        res_blocks = blocks * (2 * hidden + hidden * hidden + hidden)
        ln2 = 2 * hidden
        fc2 = hidden * out_dim + out_dim
        return ln + fc1 + res_blocks + ln2 + fc2

    head_libero = head_params(4096 * 7, 7)
    head_aloha = head_params(4096 * 14, 14)
    film_aloha = 27 * 2 * (4096 * 1152 + 1152) + 24 * 2 * (4096 * 1024 + 1024)  # SigLIP 27블록, DINOv2 24블록
    # LoRA(all-linear, r=32): 7B LLM 32층 + DINOv2-L 24층 + SigLIP so400m 27층 (+ 투영기 제외 근사)
    def lora(i, o, r=32):
        return r * (i + o)
    llm_layer = 4 * lora(4096, 4096) + 2 * lora(4096, 11008) + lora(11008, 4096)
    dino_layer = lora(1024, 3072) + lora(1024, 1024) + 2 * lora(1024, 4096)
    siglip_layer = lora(1152, 3456) + lora(1152, 1152) + 2 * lora(1152, 4304)
    lora_est = 32 * llm_layer + 24 * dino_layer + 27 * siglip_layer
    res["V9_params"] = {
        "reported_libero_total_M": PARAM_LIBERO["total_reported"] / 1e6,
        "sum_of_reported_parts_libero_M": libero_total / 1e6,
        "head_libero_computed_M": round(head_libero / 1e6, 2),
        "head_aloha_computed_M": round(head_aloha / 1e6, 2),
        "film_aloha_computed_M": round(film_aloha / 1e6, 2),
        "reported_aloha_total_M": PARAM_ALOHA["total_reported"] / 1e6,
        "sum_of_reported_parts_aloha_M": aloha_total / 1e6,
        "lora_all_linear_estimate_M": round(lora_est / 1e6, 1),
        "lora_reported_M": 111,
        "note": "LoRA 111M은 구조 추정 약 108M과 약 3% 차이. 투영기 크기와 PEFT의 대상 모듈 정의에 따라 달라진다 (추정).",
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 요약 출력
    print("V1 all ok:", res["V1_all_ok"])
    print("V2:", {k: v["cells_not_fitting"] for k, v in v2.items()})
    print("V3 all ok:", v3["all_ok"], "| ratios:", v3["ratios"])
    print("V4 n500:", {k: v for k, v in v4["n_per_suite=500"].items()})
    print("V4 n1500:", {k: v for k, v in v4["n_per_suite=1500"].items() if k != "OFT_wilson95_pct"})
    print("V4 per suite:", v4["per_suite_oft_vs_pi0_n500"])
    print("V5 avg:", avg, "| diffs:", v5["oft_plus_minus"], "| sums:", sums)
    print("V6:", res["V6_bridge"])
    print("V7:", res["V7_table14_15"]["t14_oft_minus_single"], res["V7_table14_15"]["t15_per_suite_test_n500"])
    print("V9:", {k: v for k, v in res["V9_params"].items() if k != "note"})


if __name__ == "__main__":
    main()
