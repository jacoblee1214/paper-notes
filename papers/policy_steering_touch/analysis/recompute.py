"""논문 표·그림 수치를 다시 계산하고 검정한다 (수 초, numpy·scipy만 사용).

입력은 두 종류다.
  (a) 본문·부록 표에 적힌 정확한 값: Table 1(n=40), Table 10–13, C.2 본문 수치
  (b) 그림에서 읽은 값: Fig. 2 (시도 20회, 막대 높이는 5% 단위로 읽음, 세 과제와 평균 패널이 서로 맞는지 검사)

실행: python3 analysis/recompute.py  → analysis/recompute_results.json 과 표준 출력
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def fisher(a, n1, b, n2):
    """2x2 Fisher 양측 검정. a/n1 대 b/n2."""
    _, p = stats.fisher_exact([[a, n1 - a], [b, n2 - b]], alternative="two-sided")
    return float(p)


# ---------- (a) Table 1: 보상 정확도, n = 40 시행 (표의 % 값에서 정수 개수로 환산) ----------
t1 = {
    "visual_GT":   {"Wiping": 80.0, "Insertion": 70.0, "Pipette": 100.0, "Avg": 83.3},
    "visual_Pred": {"Wiping": 82.5, "Insertion": 72.5, "Pipette": 100.0, "Avg": 85.0},
    "tactile_GT":  {"Wiping": 85.0, "Insertion": 70.0, "Pipette": 80.0,  "Avg": 78.3},
    "tactile_Pred": {"Wiping": 90.0, "Insertion": 77.5, "Pipette": 77.5, "Avg": 81.7},
}
N1 = 40
counts = {k: {t: round(v[t] * N1 / 100) for t in ("Wiping", "Insertion", "Pipette")} for k, v in t1.items()}
R["table1_counts"] = counts
# 표의 ± 값이 n=40 Wald SE와 맞는지 확인 (인쇄된 SE: 6.3, 7.2, 0.0 등)
printed_se = {"Wiping": 6.3, "Insertion": 7.2, "Pipette": 0.0}
check_se = {}
for k, v in t1.items():
    for t in ("Wiping", "Insertion", "Pipette"):
        p = v[t] / 100
        se = 100 * math.sqrt(p * (1 - p) / N1)
        check_se[f"{k}/{t}"] = round(se, 2)
R["table1_se_check_n40"] = check_se
# 평균 열이 세 과제의 산술 평균과 맞는지
R["table1_avg_check"] = {k: round(sum(v[t] for t in ("Wiping", "Insertion", "Pipette")) / 3, 2) for k, v in t1.items()}
R["table1_avg_printed"] = {k: v["Avg"] for k, v in t1.items()}
# GT 대 Pred (같은 과제, 두 독립 집합으로 보고 Fisher 검정)
gt_pred = {}
for mod in ("visual", "tactile"):
    for t in ("Wiping", "Insertion", "Pipette"):
        a = counts[f"{mod}_GT"][t]
        b = counts[f"{mod}_Pred"][t]
        gt_pred[f"{mod}/{t}"] = {"GT": f"{a}/40", "Pred": f"{b}/40", "p_fisher": round(fisher(a, 40, b, 40), 3)}
R["table1_gt_vs_pred"] = gt_pred
# 하한 확인: 정확도 70% 이상이라는 주장은 점추정 기준이다. 개별 셀의 Wilson 하한
R["table1_wilson_lower_min"] = {f"{k}/{t}": wilson(counts[k][t], 40)[0] for k in counts for t in ("Wiping", "Insertion", "Pipette")}

# ---------- (b) Fig. 2: 그림에서 읽은 값 (n = 20). 순서: (Overall, Visual, Contact) ----------
COND = ["Base", "VL8", "VL16", "TS", "TG", "Naive", "Ours"]
READ = {
    "Wiping":    {"Base": (20, 50, 30), "VL8": (20, 55, 30), "VL16": (25, 80, 35), "TS": (40, 60, 40), "TG": (50, 60, 60), "Naive": (50, 70, 50), "Ours": (80, 90, 90)},
    "Pipette":   {"Base": (15, 45, 30), "VL8": (20, 55, 40), "VL16": (25, 80, 35), "TS": (20, 40, 65), "TG": (25, 50, 75), "Naive": (35, 60, 70), "Ours": (70, 80, 75)},
    "Insertion": {"Base": (30, 55, 40), "VL8": (35, 65, 40), "VL16": (45, 70, 45), "TS": (40, 55, 55), "TG": (45, 55, 75), "Naive": (50, 80, 60), "Ours": (70, 75, 80)},
}
# 그림의 "Average Across Tasks" 패널을 눈으로 읽은 값 (세 과제 평균과 비교하는 용)
READ_AVG_PANEL = {
    "Base": (22, 50, 33), "VL8": (25, 58, 37), "VL16": (32, 77, 38), "TS": (33, 52, 53),
    "TG": (40, 55, 70), "Naive": (45, 70, 60), "Ours": (73, 82, 82),
}
# 1) 막대 값은 시도 수 20의 배수 5%인지 (정수 개수 확인)
R["fig2_all_multiples_of_5"] = all(v % 5 == 0 for t in READ.values() for c in t.values() for v in c)
# 2) 전체 성공 = 시각 성공 AND 접촉 성공이므로 Overall ≤ min(Visual, Contact)
viol = []
for task, t in READ.items():
    for c, (o, v, k) in t.items():
        if o > min(v, k):
            viol.append((task, c, o, v, k))
R["fig2_overall_le_min_violations"] = viol
# 3) 세 과제 평균과 평균 패널의 차이 (읽기 오차 허용 ±3)
avg_diff = {}
for c in COND:
    m = [round(sum(READ[t][c][i] for t in READ) / 3, 1) for i in range(3)]
    avg_diff[c] = {"mean_of_tasks": m, "panel": READ_AVG_PANEL[c], "max_abs_diff": round(max(abs(m[i] - READ_AVG_PANEL[c][i]) for i in range(3)), 1)}
R["fig2_avg_panel_check"] = avg_diff
# 4) 오차 막대: 인쇄 캡션은 "standard errors"인데, 막대 반폭이 1.96·SE(Wald)와 맞는지
bar_check = {}
for c in ("Base", "Ours"):
    for i, metric in enumerate(("Overall", "Visual", "Contact")):
        for task in READ:
            p = READ[task][c][i] / 100
            se = math.sqrt(p * (1 - p) / 20) * 100
            bar_check[f"{task}/{c}/{metric}"] = {"SE_pp": round(se, 1), "95pct_halfwidth_pp": round(1.96 * se, 1)}
R["fig2_errorbar_halfwidth_example"] = {k: bar_check[k] for k in list(bar_check)[:6]}
# 5) 주장 검사
claims = {}
ours_avg_overall = READ_AVG_PANEL["Ours"][0]
base_avg_overall = READ_AVG_PANEL["Base"][0]
naive_avg_overall = READ_AVG_PANEL["Naive"][0]
unimodal = ["VL8", "VL16", "TS", "TG"]
claims["C_51pp_avg_overall_gain_over_base_pp"] = ours_avg_overall - base_avg_overall
claims["C_51pp_relative_gain_ratio"] = round(ours_avg_overall / base_avg_overall, 2)
per_task = {}
for task in READ:
    o = READ[task]["Ours"][0]
    best_uni = max(READ[task][c][0] for c in unimodal)
    best_uni_c = max(unimodal, key=lambda c: READ[task][c][0])
    per_task[task] = {"ours_overall": o, "best_unimodal": best_uni, "best_unimodal_cond": best_uni_c,
                      "gap_pp": o - best_uni, "naive": READ[task]["Naive"][0], "gap_vs_naive_pp": o - READ[task]["Naive"][0],
                      "gap_vs_base_pp": o - READ[task]["Base"][0]}
claims["C_per_task_overall_gaps"] = per_task
claims["C_33pp_vs_best_unimodal_avg_pp"] = ours_avg_overall - max(READ_AVG_PANEL[c][0] for c in unimodal)
claims["C_20pp_vs_naive_avg_pp"] = ours_avg_overall - naive_avg_overall
# 6) 전체 60 시행 합계 대비 검정 (세 과제 합)
pool = {c: sum(round(READ[t][c][0] * 20 / 100) for t in READ) for c in COND}
R["fig2_pooled_overall_counts_of_60"] = pool
pv = {}
for c in ("TG", "Naive", "TS", "VL16", "VL8", "Base"):
    pv[f"Ours_vs_{c}"] = {"ours": f"{pool['Ours']}/60", "other": f"{pool[c]}/60", "p_fisher": round(fisher(pool["Ours"], 60, pool[c], 60), 5)}
R["fig2_pooled_overall_tests"] = pv
# 과제별 Ours 대 Naive, Ours 대 최고 단일 모달
pt = {}
for task in READ:
    a = round(READ[task]["Ours"][0] * 20 / 100)
    b = round(READ[task]["Naive"][0] * 20 / 100)
    c_ = round(max(READ[task][c][0] for c in unimodal) * 20 / 100)
    pt[task] = {"Ours_vs_Naive": f"{a}/20 vs {b}/20, p={fisher(a, 20, b, 20):.3f}",
                "Ours_vs_best_unimodal": f"{a}/20 vs {c_}/20, p={fisher(a, 20, c_, 20):.3f}"}
R["fig2_per_task_tests"] = pt
R["claims_fig2"] = claims

# 시도 수 20에서 33 pp 차이를 80% 검정력으로 보려면 필요한 군당 시도 수 (두 비율 정규 근사)
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)
R["power_n_per_arm_33pp_at_0.73_vs_0.40"] = n_per_arm(0.73, 0.40)
R["power_n_per_arm_20pp_at_0.70_vs_0.50"] = n_per_arm(0.70, 0.50)

# ---------- (c) C.2 본문 수치 (base 시각 정책, 시도 20회 가정: 30% = 6/20 등) ----------
c2 = {
    "visual_base_to_ours": {"Wiping": (6, 18), "Pipette": (6, 15), "Insertion": (8, 16)},
    "vt_base_to_ours": {"Wiping": (6, 14), "Pipette": (14, 16), "Insertion": (12, 14)},
}
c2_tests = {}
for pol, d in c2.items():
    for task, (a, b) in d.items():
        c2_tests[f"{pol}/{task}"] = {"base": f"{a}/20", "ours": f"{b}/20", "p_fisher": round(fisher(a, 20, b, 20), 5)}
R["c2_tests"] = c2_tests
# 같은 시각 base 정책의 기준 성공률: 본문 Fig. 2(읽은 값) 대 C.2 본문
R["base_visual_disagreement"] = {"Fig2_base_overall": {"Wiping": 20, "Pipette": 15, "Insertion": 30},
                                 "C2_text_base_overall": {"Wiping": 30, "Pipette": 30, "Insertion": 40}}
R["ours_disagreement"] = {"Fig2_ours_overall": {"Wiping": 80, "Pipette": 70, "Insertion": 70},
                          "C2_text_ours_overall": {"Wiping": 90, "Pipette": 75, "Insertion": 80},
                          "Fig2_ours_overall_counts": {"Wiping": 16, "Pipette": 14, "Insertion": 14},
                          "C2_ours_overall_counts": {"Wiping": 18, "Pipette": 15, "Insertion": 16}}
R["c2_avg_gain_pp"] = round((90 + 75 + 80) / 3 - (30 + 30 + 40) / 3, 1)

# ---------- (d) Table 13: 추론 시간 (ms, 평균 ± SD) ----------
t13 = {
    "Ours": (69, 121, 216, 65, 471, 3, 12), "Naive": (69, 121, 216, 24, 429, 1, 13),
    "VL8_only": (69, 11, 216, None, 295, 0, 11), "VL16_only": (69, 121, 216, None, 405, 0, 12),
    "TS_only": (69, 11, None, 24, 103, 0, 4), "TG_only": (None, None, None, 94, 94, 0, 4),
}
sums = {}
for k, (s, w, v, tv, total, _, _sd) in t13.items():
    parts = [x for x in (s, w, v, tv) if x is not None]
    sums[k] = {"sum_of_parts": sum(parts), "reported_total": total}
R["table13_sums"] = sums
diff = 471 - 429
R["table13_ours_minus_naive_ms"] = diff
R["table13_ours_minus_naive_sd_ms_indep"] = round(math.sqrt(12 ** 2 + 13 ** 2), 1)
R["table13_text_claim_s_per_8actions"] = 0.05
R["table13_text_vs_table_ratio"] = round(0.05 / (diff / 1000), 2)

# ---------- (e) Tables 10–12: 세계 모델 예측 (30 영상, 신뢰구간 없음) ----------
# 형식: metric -> (RGB-only 또는 Tactile-only, ViTaL). lower_better 플래그
tabs = {
    "wiping_front": {"FID": (9.0932, 8.1015, True), "LPIPS": (0.0995, 0.0986, True), "MAE": (0.0162, 0.0161, True), "MSE": (0.0023, 0.0023, True), "PSNR": (26.9366, 26.9702, False), "SSIM": (0.9293, 0.9300, False)},
    "wiping_wrist": {"FID": (24.9114, 22.1616, True), "LPIPS": (0.1741, 0.1719, True), "MAE": (0.0202, 0.0201, True), "MSE": (0.0029, 0.0030, True), "PSNR": (26.0685, 25.8676, False), "SSIM": (0.9058, 0.9045, False)},
    "insertion_front": {"FID": (8.1047, 8.0094, True), "LPIPS": (0.1576, 0.1566, True), "MAE": (0.0122, 0.0121, True), "MSE": (0.0012, 0.0012, True), "PSNR": (29.5526, 29.6457, False), "SSIM": (0.9367, 0.9372, False)},
    "insertion_wrist": {"FID": (9.3695, 8.9580, True), "LPIPS": (0.1395, 0.1382, True), "MAE": (0.0195, 0.0191, True), "MSE": (0.0023, 0.0022, True), "PSNR": (26.8440, 27.1264, False), "SSIM": (0.9129, 0.9156, False)},
    "pipette_front": {"FID": (12.4612, 11.5609, True), "LPIPS": (0.1700, 0.1699, True), "MAE": (0.0162, 0.0163, True), "MSE": (0.0017, 0.0019, True), "PSNR": (27.8010, 27.5515, False), "SSIM": (0.9245, 0.9222, False)},
    "pipette_wrist": {"FID": (11.3101, 10.5830, True), "LPIPS": (0.1481, 0.1480, True), "MAE": (0.0173, 0.0173, True), "MSE": (0.0017, 0.0018, True), "PSNR": (27.9782, 27.7871, False), "SSIM": (0.9273, 0.9266, False)},
}
tac = {
    "wiping": {"FlowAE": (6.6576, 6.4938, True), "FlowEPE": (0.1396, 0.1364, True), "GradCos": (0.5907, 0.5829, False), "HeightMAE": (0.2380, 0.2352, True), "HeightRMSE": (0.2965, 0.2959, True)},
    "insertion": {"FlowAE": (8.8939, 7.5189, True), "FlowEPE": (0.2209, 0.1742, True), "GradCos": (0.4977, 0.5014, False), "HeightMAE": (0.6896, 0.6507, True), "HeightRMSE": (0.8166, 0.7656, True)},
    "pipette": {"FlowAE": (2.2372, 1.9514, True), "FlowEPE": (0.0451, 0.0381, True), "GradCos": (0.5391, 0.5415, False), "HeightMAE": (0.6648, 0.6061, True), "HeightRMSE": (0.7642, 0.7082, True)},
}
rgb_win, rgb_tot, rgb_lpips_fid_win = 0, 0, 0
rgb_worse = []
for cam, d in tabs.items():
    for m, (base, ours, low) in d.items():
        rgb_tot += 1
        better = ours < base if low else ours > base
        rgb_win += better
        if not better:
            rgb_worse.append(f"{cam}/{m}")
    if all(d[m][1] < d[m][0] for m in ("FID", "LPIPS")):
        rgb_lpips_fid_win += 1
tac_win, tac_tot, tac_worse = 0, 0, []
for task, d in tac.items():
    for m, (base, ours, low) in d.items():
        tac_tot += 1
        better = ours < base if low else ours > base
        tac_win += better
        if not better:
            tac_worse.append(f"{task}/{m}")
R["tables10_12_rgb_metrics_ours_better"] = f"{rgb_win}/{rgb_tot}"
R["tables10_12_rgb_worse_cells"] = rgb_worse
R["tables10_12_FID_LPIPS_both_better_in_cams"] = f"{rgb_lpips_fid_win}/6 cameras (both FID and LPIPS lower)"
R["tables10_12_tactile_metrics_ours_better"] = f"{tac_win}/{tac_tot}"
R["tables10_12_tactile_worse_cells"] = tac_worse

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
if __name__ == "__main__":
    for k, v in R.items():
        print(k, "=>", json.dumps(v, ensure_ascii=False, default=str)[:400])
