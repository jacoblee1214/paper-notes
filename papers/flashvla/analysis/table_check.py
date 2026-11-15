"""FlashVLA 본문 표 수치 재계산, 일관성 검사, 근사 검정 (numpy, scipy).

실행: python3 analysis/table_check.py  → analysis/table_check_results.json
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "table_check_results.json"
R = {}


def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(p1, n1, p2, n2, z=1.96):
    l1, u1 = wilson(p1, n1, z)
    l2, u2 = wilson(p2, n2, z)
    d = p2 - p1
    return d, d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


# 1. Table 1 (LIBERO, d=1): 평균과 속도비 재계산
t1 = {
    "pi05": [98.8, 98.2, 98.0, 92.4, 156.0, 8.4, 53.8],
    "vlash_d1": [98.8, 99.2, 96.7, 94.4, 153.9, 7.2, 46.8],
    "vlash_d4": [92.5, 96.9, 93.3, 89.6, 176.7, 5.8, 32.8],
    "streamingvla_d1": [96.6, 96.6, 95.4, 91.0, None, None, 31.6],
    "flashvla_d1": [98.8, 99.6, 97.6, 95.4, 158.0, 3.5, 22.1],
}
base = t1["pi05"][6]
R["table1"] = {}
for k, v in t1.items():
    avg = float(np.mean(v[:4]))
    row = {"avg_recomputed": round(avg, 3), "speedup_recomputed": round(base / v[6], 3)}
    if v[5] is not None:
        row["episode_time_check_steps_x_tps_s"] = round(v[4] * v[6] / 1000, 2)
        row["episode_time_reported_s"] = v[5]
    R["table1"][k] = row
# 시간: 에피소드 시간 = 스텝 x Time/Step (보고값과 대조)
R["table1"]["flashvla_d1"]["avg_gain_vs_pi05"] = round(R["table1"]["flashvla_d1"]["avg_recomputed"] - 96.9, 2)

# 2. Table 2 속도비와 Hz
t2 = {"4090_2v": (45.8, 26.7), "4090_3v": (55.4, 36.8), "5090_2v": (37.0, 20.3), "5090_3v": (44.8, 27.1)}
R["table2"] = {k: {"ratio_vs_pi05": round(a / b, 3), "flashvla_hz": round(1000 / b, 1)} for k, (a, b) in t2.items()}

# 3. Fig 1b 분해: 텍스트의 "75%", "20x", "3.5x" 와 막대 라벨(128.9, 97.4, 4.9, 36.6) 대조
pi_tot, pi_act, fv_tot, fv_act = 128.9, 97.4, 36.6, 4.9
R["fig1b"] = {
    "action_share_pi05": round(pi_act / pi_tot, 3),
    "vlm_pi05_ms": round(pi_tot - pi_act, 1),
    "vlm_flashvla_ms": round(fv_tot - fv_act, 1),
    "overall_ratio": round(pi_tot / fv_tot, 3),
    "action_ratio": round(pi_act / fv_act, 2),
    "per_pass_pi05_if_10_passes_ms": round(pi_act / 10, 2),
    "implied_per_pass_gain_if_1_pass_flashvla": round((pi_act / 10) / fv_act, 2),
}

# 4. Table 4 과 Table 3 의 가중 평균 일관성: 50 과제를 짧/중/긴 (n_s, n_m, n_l)으로 나눈 정수 해 탐색
g = {  # (clean, random) per group, from Table 4
    "pi05": {"s": (93.5, 94.2), "m": (81.9, 80.4), "l": (54.2, 51.8)},
    "flash": {"s": (93.2, 93.3), "m": (86.0, 84.8), "l": (90.8, 88.4)},
}
feasible = []
for ns, nm in itertools.product(range(51), repeat=2):
    nl = 50 - ns - nm
    if nl < 0:
        continue
    ok = True
    for key, target in [("pi05", (86.1, 85.8)), ("flash", (90.8, 90.2))]:
        for idx in (0, 1):
            tot = ns * g[key]["s"][idx] + nm * g[key]["m"][idx] + nl * g[key]["l"][idx]
            if abs(tot / 50 - target[idx]) > 0.061:
                ok = False
    if ok:
        feasible.append((ns, nm, nl))
R["table4_consistency"] = {
    "feasible_group_sizes_n_s_n_m_n_l": feasible[:30],
    "count": len(feasible),
    "note": "Table 3의 Clean/Random 평균(86.1, 85.8, 90.8, 90.2)과 Table 4 그룹 평균의 가중합이 0.061 이내인 정수 해 (반올림 0.05 + 여유)",
}

# 5. LIBERO 검정: 2000 에피소드/실행 가정 (suite당 500 x 4). 실행 간 상관은 무시한 근사
n_run = 2000
R["libero_test_one_run"] = {}
for lab, p0, p1 in [("flash_d1_vs_pi05", 0.969, 0.978), ("flash_d0_vs_pi05", 0.969, 0.979)]:
    d, lo, hi = newcombe_diff(p0, n_run, p1, n_run)
    z = (p1 - p0) / math.sqrt(p0 * (1 - p0) / n_run + p1 * (1 - p1) / n_run)
    R["libero_test_one_run"][lab] = {
        "diff_pp": round(d * 100, 2), "newcombe95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
        "two_prop_z": round(z, 2), "p_two_sided": round(2 * (1 - stats.norm.cdf(abs(z))), 4),
    }
# 실행 5회(seed) 평균을 독립으로 보는 상한 근사 (n=10000)
d, lo, hi = newcombe_diff(0.969, 10000, 0.978, 10000)
R["libero_test_five_runs_upper_bound"] = {"diff_pp": round(d * 100, 2), "newcombe95_pp": [round(lo * 100, 2), round(hi * 100, 2)]}

# 6. RoboTwin 2.0: 과제당 시행 수 미보고. 100/과제 x 50 = 5000 가정
n_rt = 5000
d, lo, hi = newcombe_diff(0.860, n_rt, 0.905, n_rt)
R["robotwin_test_assumed_n5000"] = {"diff_pp": round(d * 100, 2), "newcombe95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
                                     "note": "시행 수 미보고. 과제 군집을 무시한 근사"}

# 7. 실로봇: 3과제 x 15 롤아웃 = 45. 점수(0,1,2)/2를 이진 성공 근사로 씀. 과제별 n=15
R["real_world_approx"] = {}
for lab, a, b in [("flash_vs_pi05_sync", 0.800, 0.844), ("flash_vs_naive", 0.756, 0.844), ("flash_vs_rtc", 0.800, 0.844)]:
    d, lo, hi = newcombe_diff(a, 45, b, 45)
    R["real_world_approx"][lab] = {"diff_pp": round(d * 100, 1), "newcombe95_pp": [round(lo * 100, 1), round(hi * 100, 1)]}
R["real_world_approx"]["per_task_wilson_n15_at_0.8"] = [round(v, 3) for v in wilson(0.8, 15)]

# 8. Fig 4 TTR-TTFA 차이로 본 실행 구간 길이 (E[dt_react] = dt_infer + 0.5 dt_exec 에서 dt_infer = TTFA 가정)
R["fig4_implied_exec_window_ms"] = {
    "pi05": round(2 * (130.0 - 80.0), 1),
    "faster": round(2 * (112.1 - 62.1), 1),
    "flashvla": round(2 * (70.4 - 37.1), 1),
    "note": "TTFA를 dt_infer로 본 추정. 세 방법의 실행 구간이 같지 않을 수 있음",
}
R["fig4_ratios"] = {"ttfa_vs_faster": round(62.1 / 37.1, 2), "ttr_vs_faster": round(112.1 / 70.4, 2)}

# 9. 실로봇 완료 시간 비율 (Fig 5e 그래프 판독 근사, 3과제)
approx_time = {"pick_place": (14.5, 12.0), "whiteboard": (27.0, 21.5), "table_clean": (54.0, 43.5)}
approx_rtc = {"pick_place": (13.0, 12.0), "whiteboard": (26.0, 21.5), "table_clean": (47.0, 43.5)}
R["fig5e_read_speedup_by_horizon"] = {k: round(a / b, 2) for k, (a, b) in approx_time.items()}
R["fig5e_read_speedup_vs_rtc"] = {k: round(a / b, 2) for k, (a, b) in approx_rtc.items()}
R["fig5d_read_score_avg"] = {"pi05": round((90 + 80 + 70) / 3, 1), "rtc": round((87 + 83 + 70) / 3, 1), "naive": round((83 + 77 + 67) / 3, 1), "flashvla": round((93 + 87 + 73) / 3, 1)}
R["fig5e_note"] = "그래프 판독 근사치(±0.5 s). 본문은 '긴 과제일수록 속도 향상이 크다'고 쓰지만 판독값은 과제 사이에서 1.2–1.26으로 뚜렷한 증가 추세가 없음. 본문 평균 1.3x는 판독 범위의 윗부분"

# 10. 실로봇 A4000 지연: 67.3 ms vs 30 Hz 주기
R["a4000"] = {"period_30hz_ms": round(1000 / 30, 2), "invocation_ms": 67.3, "invocations_per_s": round(1000 / 67.3, 1),
              "periods_covered": round(67.3 / (1000 / 30), 2)}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=2)[:4000])
