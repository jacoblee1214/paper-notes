"""논문 표 수치를 다시 계산하고 일관성과 통계 검정을 확인한다 (CPU, 수 초).

입력: 논문 본문과 부록 표의 값을 그대로 옮겨 적었다 (paper/paper.txt 기준, arXiv v1).
출력: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
res = {}

# ---------------------------------------------------------------- 표 1 (RoboCasa, 평균 성공률 %)
robocasa = {  # (방법, 백본, d) -> 값
    ("Naive", "GR00T", 1): 68.0, ("Naive", "GR00T", 2): 69.6, ("Naive", "GR00T", 3): 68.6,
    ("RTC", "GR00T", 1): 69.6, ("RTC", "GR00T", 2): 70.5, ("RTC", "GR00T", 3): 70.0,
    ("CBF", "GR00T", 1): 68.5, ("CBF", "GR00T", 2): 69.8, ("CBF", "GR00T", 3): 68.0,
    ("FreeBar", "GR00T", 1): 72.8, ("FreeBar", "GR00T", 2): 72.3, ("FreeBar", "GR00T", 3): 71.4,
    ("Naive", "pi0", 1): 33.8, ("Naive", "pi0", 2): 32.7, ("Naive", "pi0", 3): 31.2,
    ("RTC", "pi0", 1): 31.3, ("RTC", "pi0", 2): 32.7, ("RTC", "pi0", 3): 31.7,
    ("CBF", "pi0", 1): 31.3, ("CBF", "pi0", 2): 33.3, ("CBF", "pi0", 3): 34.0,
    ("FreeBar", "pi0", 1): 36.0, ("FreeBar", "pi0", 2): 36.5, ("FreeBar", "pi0", 3): 36.3,
}
gains = {}
for bb in ["GR00T", "pi0"]:
    for d in [1, 2, 3]:
        gains[f"FreeBar-Naive {bb} d={d}"] = round(robocasa[("FreeBar", bb, d)] - robocasa[("Naive", bb, d)], 2)
        gains[f"FreeBar-RTC {bb} d={d}"] = round(robocasa[("FreeBar", bb, d)] - robocasa[("RTC", bb, d)], 2)
fb_naive = [v for k, v in gains.items() if k.startswith("FreeBar-Naive")]
res["robocasa_gain_vs_naive"] = gains
res["robocasa_mean_gain_vs_naive_pp"] = round(float(np.mean(fb_naive)), 2)

# ---------------------------------------------------------------- 표 1 오른쪽, 부록 표 A5–A8 (민감도)
sweeps = {
    "RoboCasa RTC (A5)": ([68.0, 66.6, 70.0, 69.6, 68.6, 52.8], (65.93, 6.55), ["0.1", "0.5", "5", "10", "30", "50"]),
    "RoboCasa FreeBar (A5)": ([70.6, 71.4, 70.4, 70.6, 67.6, 67.2], (69.63, 1.77), ["0.1", "1", "5", "10", "30", "50"]),
    "Conveyor pi0 RTC (A6)": ([53.33, 56.67, 67.67, 53.33, 46.67], (55.53, 7.70), ["0.1", "0.5", "1", "3", "8"]),
    "Conveyor pi0 FreeBar (A6)": ([66.70, 73.30, 76.67, 83.33, 73.33], (74.67, 6.04), ["0.1", "1", "3", "5", "10"]),
    "Conveyor pi0.5 RTC (A7)": ([53.33, 63.33, 73.33, 56.67, 50.00], (59.33, 9.25), ["0.1", "0.5", "1", "3", "8"]),
    "Conveyor pi0.5 FreeBar (A7)": ([70.00, 73.30, 80.00, 80.00, 76.67], (75.99, 4.35), ["0.1", "1", "3", "5", "10"]),
    "Whack pi0 RTC (A8)": ([5.01, 6.95, 10.08, 8.95, 8.04], (7.80, 1.92), ["0.1", "0.5", "1", "3", "8"]),
    "Whack pi0 FreeBar (A8)": ([10.06, 12.03, 9.95, 11.02, 8.97], (10.40, 1.14), ["0.1", "1", "3", "5", "10"]),
}
sens = {}
for name, (vals, rep, grid) in sweeps.items():
    v = np.array(vals)
    sens[name] = {
        "n_configs": len(v),
        "mean": round(float(v.mean()), 2),
        "std_ddof0": round(float(v.std(ddof=0)), 2),
        "std_ddof1": round(float(v.std(ddof=1)), 2),
        "reported_mean_std": rep,
        "grid_used": grid,
    }
# RTC RoboCasa에서 β=50 한 점을 뺀 결과 (분산이 이 한 점에서 나오는지 확인)
rtc_wo50 = np.array(sweeps["RoboCasa RTC (A5)"][0][:5])
fb_rc = np.array(sweeps["RoboCasa FreeBar (A5)"][0])
sens["RoboCasa RTC without beta=50"] = {"mean": round(float(rtc_wo50.mean()), 2), "std_ddof1": round(float(rtc_wo50.std(ddof=1)), 2)}
# 분산비 F 검정 (구성 간 독립을 가정한 대략 검사, 격자가 다르고 같은 설정을 공유하므로 참고용)
f_ratio = (sweeps["RoboCasa RTC (A5)"][1][1] ** 2) / (sweeps["RoboCasa FreeBar (A5)"][1][1] ** 2)
sens["RoboCasa var ratio RTC/FreeBar (reported std)"] = round(f_ratio, 2)
res["hyperparam_sensitivity"] = sens

# ---------------------------------------------------------------- 표 3 (실로봇, 30회 시행)
def k30(pct):
    """백분율을 30회 중 성공 횟수로 바꾼다. 정수가 아니면 None."""
    x = pct * 30 / 100
    return int(round(x)) if abs(x - round(x)) < 0.02 else None

table3 = {
    "pi0": {
        "Naive": [56.67, 70.00, 66.67, 63.33], "RTC": [73.33, 83.33, 80.00, 76.67], "Ours": [90.00, 86.67, 86.67, 76.67],
    },
    "pi0.5": {
        "Naive": [63.33, 66.67, 76.67, 63.33], "RTC": [76.67, 80.00, 83.33, 80.00], "Ours": [93.33, 90.00, 90.00, 83.33],
    },
}
horizons = [6, 8, 10, 12]
conv = {}
for bb, rows in table3.items():
    counts = {m: [k30(x) for x in vals] for m, vals in rows.items()}
    conv[bb] = {"counts_out_of_30": counts}
    for other in ["RTC", "Naive"]:
        per_h = []
        for i, h in enumerate(horizons):
            a, b = counts["Ours"][i], counts[other][i]
            _, p = stats.fisher_exact([[a, 30 - a], [b, 30 - b]])
            per_h.append({"h": h, "ours": a, other.lower(): b, "diff_pp": round((a - b) / 30 * 100, 1), "fisher_p": round(float(p), 4)})
        # 4개 h를 합친 2x2 (같은 설정 반복이 아니라 서로 다른 실행이라고 가정)
        A = sum(counts["Ours"]); B = sum(counts[other])
        _, p_pool = stats.fisher_exact([[A, 120 - A], [B, 120 - B]])
        conv[bb][f"Ours_vs_{other}"] = {
            "per_h": per_h,
            "pooled": {"ours": A, other.lower(): B, "n_each": 120, "diff_pp": round((A - B) / 120 * 100, 1), "fisher_p": round(float(p_pool), 4)},
        }
    # 공칭 평균과 격차
    conv[bb]["mean_pct"] = {m: round(float(np.mean(v)), 2) for m, v in rows.items()}
    conv[bb]["mean_gain_vs_naive_pp"] = round(float(np.mean(np.array(rows["Ours"]) - np.array(rows["Naive"]))), 2)
    conv[bb]["mean_gain_vs_RTC_pp"] = round(float(np.mean(np.array(rows["Ours"]) - np.array(rows["RTC"]))), 2)
res["conveyor_table3"] = conv

# 표 3 이외 값 중 30회 격자에 맞지 않는 값 (A6, A7의 일부, 표 3의 whack은 평균)
off_grid = []
for name, (vals, _, _) in sweeps.items():
    if "Conveyor" in name:
        for v in vals:
            if k30(v) is None:
                off_grid.append((name, v))
res["conveyor_values_not_k_over_30"] = off_grid

# ---------------------------------------------------------------- 속도 (그림 4 라벨 값)
lat_sim = {"naive": 51.7, "rtc": 102.5, "cbf": 57.0, "freebar": 55.7}
res["latency_sim_ms"] = {
    "freebar_over_naive_pct": round((lat_sim["freebar"] / lat_sim["naive"] - 1) * 100, 1),
    "rtc_over_naive_pct": round((lat_sim["rtc"] / lat_sim["naive"] - 1) * 100, 1),
    "freebar_reduction_vs_rtc_pct": round((1 - lat_sim["freebar"] / lat_sim["rtc"]) * 100, 1),
}
lat_real = {"pi0": {"rtc": 133, "freebar": 86}, "pi0.5": {"rtc": 150, "freebar": 93}}
res["latency_real_ms"] = {k: round((1 - v["freebar"] / v["rtc"]) * 100, 1) for k, v in lat_real.items()}

# ---------------------------------------------------------------- 휘틀 (Whack-a-Mole, 평균 hit만 있음)
whack_mean_gain = round((10.40 / 7.80 - 1) * 100, 1)
whack_std_drop = round((1 - 1.14 / 1.92) * 100, 1)
res["whack"] = {"hit_rate_increase_pct_from_means": whack_mean_gain, "std_decrease_pct": whack_std_drop}
# 표 3 whack (h=6..12)의 평균 (논문은 'total'이라 하지만 값이 정수가 아님)
wn = [2.03, 5.02, 7.05, 5.98]; wf = [9.61, 13.04, 12.58, 10.62]; wr = [10.05, 9.98, 10.02, 8.01]
res["whack_table3"] = {"naive_mean": round(float(np.mean(wn)), 2), "rtc_mean": round(float(np.mean(wr)), 2),
                       "freebar_mean": round(float(np.mean(wf)), 2), "freebar_lt_rtc_h": [h for h, a, b in zip(horizons, wf, wr) if a < b]}

# ---------------------------------------------------------------- Kinetix (그림 3 본문 수치)
def two_prop(p1, p2, n1, n2):
    p1, p2 = p1 / 100, p2 / 100
    pp = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return {"diff_pp": round((p1 - p2) * 100, 1), "z": round(z, 2), "p": round(2 * (1 - stats.norm.cdf(abs(z))), 4)}

def wilson(p, n):
    p = p / 100; z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round((c - h) * 100, 1), round((c + h) * 100, 1)]

kin = {
    "d4_freebar_vs_rtc": {"freebar": 68.3, "rtc": 61.4},
    "h7_freebar_vs_rtc": {"freebar": 81.0, "rtc": 79.0},
    "h7_freebar_vs_naive": {"freebar": 81.0, "naive": 71.0},
    "d4_freebar_vs_naive": {"freebar": 68.3, "naive": 49.0},
}
kres = {}
for key, pair in kin.items():
    keys = list(pair)
    a, b = pair[keys[0]], pair[keys[1]]
    kres[key] = {
        "n100_per_config": two_prop(a, b, 100, 100),
        "n1200_if_12_tasks_x100": two_prop(a, b, 1200, 1200),
    }
kres["wilson_95_freebar_d4_n100"] = wilson(68.3, 100)
kres["wilson_95_freebar_d4_n1200"] = wilson(68.3, 1200)
res["kinetix_tests"] = kres

# ---------------------------------------------------------------- RoboCasa 검정 (시도 수 가정: 11 과제 x 50 에피소드 = 550, 코드 기본값 기준)
rc = {}
for bb in ["GR00T", "pi0"]:
    for d in [1, 2, 3]:
        rc[f"FreeBar vs RTC {bb} d={d} (n=550)"] = two_prop(robocasa[("FreeBar", bb, d)], robocasa[("RTC", bb, d)], 550, 550)
        rc[f"FreeBar vs Naive {bb} d={d} (n=550)"] = two_prop(robocasa[("FreeBar", bb, d)], robocasa[("Naive", bb, d)], 550, 550)
res["robocasa_tests_assumed_n"] = rc

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
