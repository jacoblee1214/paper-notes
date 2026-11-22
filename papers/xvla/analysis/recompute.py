"""X-VLA (arXiv:2510.10274 v1) 표 재계산과 일관성 검사 (numpy, scipy). 실행: python3 analysis/recompute.py
결과: analysis/recompute_results.json
논문 표 값은 paper/paper.txt(pdftotext -layout)에서 옮겼다. 원자료가 없으므로 모든 검사는 '보고된 값끼리' 맞는지를 본다.
시뮬레이션 실행, 학습, 가중치 다운로드는 하지 않는다. CPU에서 1초 안에 끝난다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.96):
    """Wilson 95% 구간 (비율 단위)."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- 표 14 CALVIN (ABC→D, 연속 완료 과제 수)
# P(완료 >= i) for i = 1..5, 논문 표 14 값 (%)
calvin_rates = [97.1, 92.6, 88.5, 84.4, 78.8]
calvin_avg_len = sum(r / 100 for r in calvin_rates)
res["calvin_avg_len_from_table14"] = round(calvin_avg_len, 4)
res["calvin_reported"] = {"table2": 4.43, "table5_single_domain": 4.42, "table5_multi_domain_calvin": 4.32}
# 표 값을 0.1%(1/1000) 단위 반올림으로 보면 합의 오차는 최대 5 * 0.0005 = 0.0025
res["calvin_table14_max_rounding_err"] = 0.0025
res["calvin_table2_vs_table14_gap"] = round(4.43 - calvin_avg_len, 4)

# ---------------------------------------------------------------- 표 16 RoboTwin-2.0 (50 과제, Easy / Hard)
robo = [
    (97, 56), (78, 18), (79, 26), (42, 9), (96, 69), (100, 61), (94, 59), (99, 66), (27, 30), (100, 38),
    (34, 15), (99, 75), (50, 44), (52, 29), (94, 57), (58, 35), (85, 73),
    (85, 57), (27, 25), (30, 27), (62, 21), (54, 17), (75, 39), (82, 17), (98, 47), (58, 18), (100, 85),
    (98, 60), (98, 28), (98, 34), (72, 27), (19, 3), (50, 0), (39, 13),
    (78, 33), (80, 9), (70, 51), (70, 13), (0, 1), (78, 82), (78, 52), (60, 44), (99, 100), (99, 99),
    (22, 15), (87, 55), (80, 42), (83, 10), (52, 13), (40, 13),
]
easy = np.array([a for a, _ in robo], dtype=float)
hard = np.array([b for _, b in robo], dtype=float)
res["robotwin_n_tasks"] = int(len(robo))
res["robotwin_easy_mean"] = round(float(easy.mean()), 2)
res["robotwin_hard_mean"] = round(float(hard.mean()), 2)
res["robotwin_reported"] = {"easy": 70.0, "hard": 39.0}

# ---------------------------------------------------------------- 표 12 Simpler (VM / VA / WidowX)
vm = [98.3, 97.1, 69.5, 56.5]
va = [85.5, 79.8, 61.9, 75.7]
wx = [100, 91.7, 95.8, 95.8]
res["simpler_vm_mean"] = round(float(np.mean(vm)), 2)
res["simpler_va_mean"] = round(float(np.mean(va)), 2)
res["simpler_widowx_mean"] = round(float(np.mean(wx)), 2)
res["simpler_reported"] = {"vm": 80.4, "va": 75.7, "widowx": 95.8}
# WidowX 과제당 24회 가정: 성공 수 정수 확인 (100, 91.7, 95.8, 95.8)
wx_counts = [round(v / 100 * 24) for v in wx]
res["simpler_widowx_counts_if_24_trials"] = wx_counts
res["simpler_widowx_pooled_rate_96_trials"] = round(sum(wx_counts) / 96 * 100, 2)
# 표 4 WidowX 96.0 은 24회 x 4과제 = 96회에서 정수 성공 수로 만들 수 없음
res["simpler_widowx_96_0_attainable_96_trials"] = any(abs(k / 96 * 100 - 96.0) < 0.05 for k in range(97))

# ---------------------------------------------------------------- 표 13 LIBERO (과제당 50회 x 10과제 = 500회 가정)
libero_t2 = {"Spatial": 98.2, "Object": 98.6, "Goal": 97.8, "Long": 97.6}
libero_counts = {k: round(v / 100 * 500) for k, v in libero_t2.items()}
res["libero_t2_counts_if_500"] = libero_counts
res["libero_t2_avg"] = round(float(np.mean(list(libero_t2.values()))), 3)
# 표 3 (PEFT, 우리 LoRA)과 표 6 (50 데모, 같은 설정이라고 표에 적힘)
t3 = {"Spatial": 95.4, "Object": 96.6, "Goal": 96.0, "Long": 84.2}
t6 = {"Spatial": 96.6, "Object": 95.4, "Goal": 95.0, "Long": 84.2}
res["libero_table3_avg"] = round(float(np.mean(list(t3.values()))), 3)
res["libero_table6_avg"] = round(float(np.mean(list(t6.values()))), 3)
res["libero_table3_vs_table6_diff"] = {k: round(t3[k] - t6[k], 1) for k in t3}
res["libero_pi0_table3_avg"] = round(float(np.mean([96.8, 98.8, 95.8, 85.2])), 3)  # 표 3 pi0 열
res["libero_pi0_text_claim"] = 94.2

# ---------------------------------------------------------------- 표 2 "최고 기존 SOTA" 행 검산
rows = {
    "MemoryVLA": {"vm": 77.7, "va": 72.7, "wx": 71.9, "sp": 98.4, "ob": 98.4, "go": 96.4, "lo": 93.4, "lavg": 96.7, "calvin": None, "e": None, "h": None, "vlab": None, "pdms": None},
    "FPC-VLA": {"vm": 78.0, "va": 65.8, "wx": 64.6, "sp": 86.2, "ob": 87.0, "go": 92.0, "lo": None, "lavg": 86.9, "calvin": None, "e": None, "h": None, "vlab": None, "pdms": None},
    "pi0": {"vm": 58.8, "va": 56.8, "wx": 27.8, "sp": 96.8, "ob": 98.8, "go": 95.8, "lo": 85.2, "lavg": 94.1, "calvin": None, "e": 46.4, "h": 16.4, "vlab": 37.8, "pdms": None},
    "OpenVLA-OFT": {"vm": 63.0, "va": 54.3, "wx": 31.3, "sp": 97.6, "ob": 98.4, "go": 97.9, "lo": 94.5, "lavg": 97.1, "calvin": None, "e": None, "h": None, "vlab": None, "pdms": None},
    "FLOWER": {"vm": None, "va": None, "wx": 40.0, "sp": 97.1, "ob": 96.7, "go": 95.6, "lo": 95.7, "lavg": 93.5, "calvin": 4.53, "e": None, "h": None, "vlab": None, "pdms": None},
    "GR00T-N1": {"vm": 45.0, "va": 48.4, "wx": None, "sp": 94.4, "ob": 97.6, "go": 93.0, "lo": 90.6, "lavg": 93.9, "calvin": None, "e": None, "h": None, "vlab": 39.7, "pdms": None},
    "UniVLA": {"vm": None, "va": None, "wx": 69.8, "sp": 95.4, "ob": 98.8, "go": 93.6, "lo": 94.0, "lavg": 95.4, "calvin": 4.41, "e": None, "h": None, "vlab": None, "pdms": 81.7},
    "SmolVLA(2B)": {"vm": None, "va": None, "wx": None, "sp": 93.0, "ob": 94.0, "go": 91.0, "lo": 77.0, "lavg": 88.8, "calvin": None, "e": None, "h": None, "vlab": None, "pdms": None},
}
max_claim = {"vm": 78.0, "va": 72.7, "wx": 71.9, "sp": 98.4, "ob": 98.8, "go": 97.9, "lo": 94.5, "lavg": 97.1, "calvin": 4.53, "e": 46.4, "h": 16.4, "vlab": 39.7, "pdms": 81.7}
# 표에 보이는 행만으로 최댓값을 다시 구함 (목록에 없는 방법은 빠져 있을 수 있음)
recomputed_max = {}
for col in max_claim:
    vals = [r[col] for r in rows.values() if r.get(col) is not None]
    recomputed_max[col] = max(vals) if vals else None
res["table2_max_row_reported"] = max_claim
res["table2_max_row_recomputed_from_listed_rows"] = recomputed_max
# X-VLA 열별 비교 (표 2 X-VLA 행)
xvla = {"vm": 80.4, "va": 75.7, "wx": 95.8, "sp": 98.2, "ob": 98.6, "go": 97.8, "lo": 97.6, "lavg": 98.1, "calvin": 4.43, "e": 70.0, "h": 39.0, "vlab": 51.1, "pdms": 87.3}
res["xvla_beats_max_row"] = {k: (xvla[k] > max_claim[k]) if k not in ("calvin",) else (xvla[k] > max_claim[k] if max_claim[k] else None) for k in xvla}
res["xvla_below_max_cells"] = [k for k in xvla if xvla[k] < max_claim[k]]

# ---------------------------------------------------------------- 표 15 VLABench 평균
res["vlabench_avg_from_subscores"] = round(float(np.mean([67.8, 25.1, 48.2, 63.1])), 2)

# ---------------------------------------------------------------- 표 11 NAVSIM PDMS 재계산
# NAVSIM 표준 가중: PDMS = NC * DAC * (5*EP + 5*TTC + 2*C) / 12 (각 지표는 평균)
def pdms_from_means(nc, dac, ep, ttc, c):
    return nc * dac * (5 * ep + 5 * ttc + 2 * c) / 12

navsim = {
    "X-VLA": (97.5, 96.5, 82.2, 92.9, 100.0, 87.3),
    "Transfuser": (97.7, 92.8, 79.2, 92.8, 100.0, 84.0),
    "UniAD": (97.8, 91.9, 78.8, 92.9, 100.0, 83.4),
    "UniVLA": (96.9, 91.1, 76.8, 91.7, 96.7, 81.7),
}
navsim_out = {}
for name, (nc, dac, ep, ttc, c, pd) in navsim.items():
    calc = 100 * pdms_from_means(nc / 100, dac / 100, ep / 100, ttc / 100, c / 100)
    navsim_out[name] = {"reported_pdms": pd, "pdms_from_mean_components": round(calc, 2), "gap": round(pd - calc, 2)}
res["navsim_pdms_check"] = navsim_out

# ---------------------------------------------------------------- 표 1 누적 효과 (표 1 Acc 열 차분)
ablation = [
    ("Custom LR", 4.1, 39.6, 35.5), ("Heterogeneous PT", 39.6, 25.0, -14.6),
    ("Intension abstraction", 25.0, 50.0, 25.0), ("Replace DiT", 50.0, 47.9, -2.1),
    ("Encoding pipeline", 47.9, 64.6, 16.7), ("Soft-prompt", 64.6, 73.8, 9.2),
    ("Scaling up", 73.8, 89.6, 15.8), ("Two-step adaptation", 89.6, 95.8, 6.2),
]
res["table1_delta_mismatch"] = [name for name, prev, cur, d in ablation if abs((cur - prev) - d) > 0.05]
res["table1_final_vs_table2_widowx"] = [95.8, 95.8]

# ---------------------------------------------------------------- Soft-Fold 처리량 (본문, 부록 F, 그림 11)
episodes = 1200
per_hour_low, per_hour_high = 20, 25
hours_implied_low = episodes / per_hour_high
hours_implied_high = episodes / per_hour_low
res["softfold_hours_implied_by_collection_rate"] = [hours_implied_low, hours_implied_high]
res["softfold_figure11_time_cost_h"] = 24
res["softfold_hf_card_episodes"] = 1500
res["softfold_min_per_episode"] = 1.5
res["softfold_max_episodes_per_hour_at_1p5min"] = round(60 / 1.5, 1)

# ---------------------------------------------------------------- 부록 G 표 8 데이터 가중치와 에피소드 수
sources = {"AGIBOT": (141, 0.4), "Droid-Left": (45, 0.15), "Droid-Left(2nd row, likely Right)": (45, 0.15),
           "RoboMind-Franka": (19, 0.1), "RoboMind-Dual-Franka": (2, 0.03), "RoboMind-UR": (25, 0.1),
           "RoboMind-Agilex": (11, 0.07)}
ep_total = sum(v[0] for v in sources.values())
res["table8_episode_total_k"] = ep_total
res["table8_weight_sum"] = round(sum(v[1] for v in sources.values()), 3)
res["table8_agibot_episode_share"] = round(141 / ep_total, 4)
res["table8_text_claim_k"] = 290

# ---------------------------------------------------------------- 그림 5 R² (그림 안 글자, 재계산 불가)
res["fig5_r2_printed"] = -0.925
res["fig5_r2_note"] = "음수 R²는 그 점들에 대해 맞춘 직선이 평균보다 나쁘다는 뜻이다. 점 값은 그림 이미지에만 있어 재계산하지 못했다."

# ---------------------------------------------------------------- 신뢰구간 (가정: LIBERO 과제당 50회, Simpler WidowX 24회)
lib_ci = {}
for suite, pct in libero_t2.items():
    k = libero_counts[suite]
    lo, hi = wilson(k, 500)
    lib_ci[suite] = {"k": k, "n": 500, "rate": round(pct, 1), "wilson95": [round(lo * 100, 2), round(hi * 100, 2)]}
res["libero_wilson_if_500_trials"] = lib_ci
wx_pool_k = sum(wx_counts)
lo, hi = wilson(wx_pool_k, 96)
res["simpler_widowx_wilson_if_96_trials"] = {"k": wx_pool_k, "n": 96, "wilson95": [round(lo * 100, 2), round(hi * 100, 2)]}
# 표 3 PEFT Libero avg 대 pi0 (정확한 사후 검정 아님, 시드 수 미기재)
t_stat, p_val = stats.ttest_ind([95.4, 96.6, 96.0, 84.2], [96.8, 98.8, 95.8, 85.2], equal_var=False)
res["peft_vs_pi0_welch_over_4_suites_indicative"] = {"t": round(float(t_stat), 3), "p": round(float(p_val), 3),
                                                    "note": "4개 스위트 평균 비교일 뿐이며 독립 표본 가정이 맞지 않는다. 참고용."}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
