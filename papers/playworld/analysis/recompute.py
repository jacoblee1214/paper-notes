"""PlayWorld (arXiv:2603.09030 v3) 표·그림 수치 재계산.

검사 항목(V1-V10):
  V1  표 3 합계 (533 클립, 출처별·범주별)
  V2  표 1: 행별 평균, 사람 시연 대비 개선, 곡선 학습 대 30h
  V3  식 7 정규화와 그림 10 판독값의 일치
  V4  표 4 (PSNR, MSE): 행 순서와 표 1 순서 비교
  V5  그림 7 (정책 평가 상관): 상관 신뢰구간, 상대 개선, 판독 재계산
  V6  그림 6 (실패 모드 원그래프): 합계, 성공률, 정책별 Fisher, 편향, 분포 거리
  V7  그림 9 (RL 미세조정): 시행 수 20 가정 Fisher·Wilson, "65%"의 정의
  V8  그림 11 (객체 일반화) 표: 단조성과 상대 변화
  V9  부록 A 곡선 스케줄 산술
  V10 요약 수치 ("40%", "5배")

그림 판독 값은 "(그림 판독)"으로 표시한다. 판독 오차는 눈금 한 칸의 10% 안팎(0.02-0.03)이다.
실행: python3 analysis/recompute.py  → analysis/recompute_results.json
"""
import json
import math
from itertools import combinations
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
    return [round(c - h, 4), round(c + h, 4)]


# ------------------------------------------------------------------ V1 표 3
src = {"Human-collected": 97, "pi0": 85, "pi0.5": 122, "Diffusion Policies": 229}
cat = {"Success": 98, "Missed Grasp": 88, "Slide": 76, "Slip": 89, "Deformation": 90, "Collision": 92}
R["V1"] = {"sum_by_source": sum(src.values()), "sum_by_category": sum(cat.values()),
           "source_share": {k: round(v / sum(src.values()), 3) for k, v in src.items()},
           "note": "본문 '500+ clips'와 합계 533은 일치"}

# ------------------------------------------------------------------ V2 표 1
cats = ["Success", "Missed Grasp", "Slide", "Slip", "Deformation", "Collision"]
T1 = {  # (LPIPS, SSIM) 범주 순서
    "Human Demo (6h)": [(0.084, 0.867), (0.080, 0.875), (0.090, 0.850), (0.090, 0.865), (0.108, 0.820), (0.086, 0.852)],
    "Human Play (6h)": [(0.086, 0.861), (0.071, 0.869), (0.089, 0.864), (0.088, 0.867), (0.100, 0.831), (0.080, 0.873)],
    "Robot Play (6h)": [(0.082, 0.870), (0.066, 0.883), (0.077, 0.865), (0.078, 0.871), (0.099, 0.831), (0.074, 0.888)],
    "Robot Play (30h)": [(0.071, 0.873), (0.064, 0.887), (0.073, 0.876), (0.072, 0.879), (0.094, 0.833), (0.076, 0.883)],
    "Robot Play (Curriculum)": [(0.070, 0.880), (0.066, 0.890), (0.071, 0.890), (0.070, 0.884), (0.093, 0.836), (0.072, 0.893)],
}
w = np.array([cat[c] for c in cats], float)
rows = {}
for name, v in T1.items():
    a = np.array(v)
    rows[name] = {
        "lpips_mean6": round(a[:, 0].mean(), 4), "ssim_mean6": round(a[:, 1].mean(), 4),
        "lpips_mean5_fail": round(a[1:, 0].mean(), 4), "ssim_mean5_fail": round(a[1:, 1].mean(), 4),
        "lpips_weighted": round(float((a[:, 0] * w).sum() / w.sum()), 4),
    }
base = np.array(T1["Human Demo (6h)"])
for name, v in T1.items():
    a = np.array(v)
    rows[name]["lpips_rel_change_vs_demo_pct"] = [round(100 * (a[i, 0] - base[i, 0]) / base[i, 0], 1) for i in range(6)]
    rows[name]["ssim_diff_vs_demo"] = [round(a[i, 1] - base[i, 1], 3) for i in range(6)]
R["V2"] = rows
# 곡선 대 30h 개선
c30 = np.array(T1["Robot Play (30h)"]); cur = np.array(T1["Robot Play (Curriculum)"])
R["V2"]["curriculum_vs_30h"] = {
    "lpips_diff": [round(cur[i, 0] - c30[i, 0], 3) for i in range(6)],
    "ssim_diff": [round(cur[i, 1] - c30[i, 1], 3) for i in range(6)],
    "lpips_ties_or_worse_count": int(sum(cur[i, 0] >= c30[i, 0] for i in range(6))),
    "note": "표 1의 곡선 행은 데이터 규모가 30h인지 표에 적혀 있지 않다 (본문은 same training data)",
}
# Success 열: 6h 로봇 놀이 대 사람 시연 차이 (LPIPS 0.084 vs 0.082)
R["V2"]["success_col_lpips_range_6h"] = [min(T1[k][0][0] for k in list(T1)[:3]), max(T1[k][0][0] for k in list(T1)[:3])]
# 사람 시연 대비 LPIPS 개선이 가장 큰 범주
best = {}
for name in ["Robot Play (6h)", "Robot Play (30h)", "Robot Play (Curriculum)"]:
    a = np.array(T1[name])
    best[name] = {"category": cats[int(np.argmin((a[:, 0] - base[:, 0]) / base[:, 0]))],
                  "rel_pct": round(100 * float(np.min((a[:, 0] - base[:, 0]) / base[:, 0])), 1)}
R["V2"]["largest_lpips_gain_vs_demo"] = best

# ------------------------------------------------------------------ V3 식 7과 그림 10
smin, smax = 0.072, 0.10


def norm_pos(s):  # 그림 10 축(높을수록 좋음)과 같은 방향: 1 - (s - smin)/(smax - smin)
    return 1 - (s - smin) / (smax - smin)


fig10 = {  # (그림 판독) 시간(h) -> 정규화 점수
    "Human Demo": {0.15: .13, 0.3: .25, 0.6: .38, 1.5: .40, 3: .365, 6: .37},
    "Human Play": {0.15: .25, 0.3: .31, 0.6: .28, 1.5: .41, 3: .40, 6: .51},
    "PlayWorld": {0.15: .25, 0.3: .34, 0.6: .54, 1.5: .58, 3: .58, 6: .74, 15: .82, 30: .89},
}
tab = {"Human Demo": "Human Demo (6h)", "Human Play": "Human Play (6h)"}
v3 = {}
for label, row in [("Human Demo 6h", "Human Demo (6h)"), ("Human Play 6h", "Human Play (6h)"),
                   ("Robot Play 6h", "Robot Play (6h)"), ("Robot Play 30h", "Robot Play (30h)"),
                   ("Robot Play Curriculum", "Robot Play (Curriculum)")]:
    s6 = rows[row]["lpips_mean6"]; s5 = rows[row]["lpips_mean5_fail"]
    v3[label] = {"norm_from_table1_mean6": round(norm_pos(s6), 3), "norm_from_table1_mean5_fail": round(norm_pos(s5), 3)}
v3["fig10_read"] = {"Human Demo 6h": 0.37, "Human Play 6h": 0.51, "Robot Play 6h": 0.74, "Robot Play 30h": 0.89}
v3["eq7_as_written_sign"] = {"example_s=0.0751": round(-(0.0751 - smin) / (smax - smin), 3),
                              "note": "식 7의 부호대로면 값은 [-1, 0] 범위, 그림 10 축은 0-1 (1에서 뺀 형태와 일치)"}
v3["fig10_mean6_matches_within"] = max(abs(v3[k]["norm_from_table1_mean6"] - v3["fig10_read"][k]) for k in v3["fig10_read"])
v3["demo_plateau_hours"] = "0.6h부터 0.37-0.40 (그림 판독)"
v3["playworld_plateau_1p5_to_3h"] = [fig10["PlayWorld"][1.5], fig10["PlayWorld"][3]]
R["V3"] = v3

# ------------------------------------------------------------------ V4 표 4
psnr = {"Human Demo": [24.78, 24.37, 23.98, 23.66, 22.58, 24.67],
        "Human Play": [23.56, 24.41, 23.90, 24.01, 22.93, 24.91],
        "Robot Play": [24.95, 24.83, 25.02, 24.84, 23.22, 24.61],
        "Robot Play (Curriculum)": [25.01, 24.82, 25.14, 24.78, 23.55, 24.95]}
mse = {"Human Demo": [279.80, 280.87, 341.36, 324.57, 428.55, 354.46],
       "Human Play": [284.01, 276.54, 302.10, 332.65, 390.98, 304.12],
       "Robot Play": [274.81, 259.14, 263.17, 259.10, 366.55, 297.34],
       "Robot Play (Curriculum)": [265.13, 264.19, 250.46, 243.51, 350.12, 295.53]}
v4 = {"psnr_mean6": {k: round(float(np.mean(v)), 2) for k, v in psnr.items()},
      "mse_mean6": {k: round(float(np.mean(v)), 1) for k, v in mse.items()},
      "robot_play_psnr_wins_vs_demo": sum(psnr["Robot Play"][i] > psnr["Human Demo"][i] for i in range(6)),
      "robot_play_mse_wins_vs_demo": sum(mse["Robot Play"][i] < mse["Human Demo"][i] for i in range(6)),
      "curriculum_psnr_wins_vs_robot_play": sum(psnr["Robot Play (Curriculum)"][i] > psnr["Robot Play"][i] for i in range(6)),
      "curriculum_mse_wins_vs_robot_play": sum(mse["Robot Play (Curriculum)"][i] < mse["Robot Play"][i] for i in range(6))}
# 순서 불일치: PSNR 순위와 MSE 순위가 다른 (열, 행쌍)
disagree = []
names = list(psnr)
for j, c in enumerate(cats):
    for a, b in combinations(names, 2):
        if (psnr[a][j] > psnr[b][j]) != (mse[a][j] < mse[b][j]):
            disagree.append(f"{c}: {a} vs {b}")
v4["psnr_mse_order_disagree"] = disagree
# Success열 PSNR: 사람 놀이가 가장 낮음 (23.56) 반면 표 1 Success LPIPS는 0.086로 비슷
R["V4"] = v4

# ------------------------------------------------------------------ V5 그림 7
rep = {"Human Demo": (0.6636, 0.275), "Human Play": (0.6619, 0.297), "PlayWorld": (0.8766, 0.171)}
n = 18


def fisher_z_ci(r, n):
    z = np.arctanh(r); se = 1 / math.sqrt(n - 3)
    return [round(float(np.tanh(z - 1.959964 * se)), 3), round(float(np.tanh(z + 1.959964 * se)), 3)]


v5 = {"n_policies": n, "reported": rep, "ci95_fisher_z": {k: fisher_z_ci(v[0], n) for k, v in rep.items()}}
for base_name in ["Human Demo", "Human Play"]:
    z = (np.arctanh(rep["PlayWorld"][0]) - np.arctanh(rep[base_name][0])) / math.sqrt(2 / (n - 3))
    v5[f"independent_z_test_PlayWorld_vs_{base_name}"] = {"z": round(float(z), 2), "p_two_sided": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 3),
                                                            "note": "같은 정책 18개를 쓰는 종속 상관이라 독립 가정은 보수적이고, 예측 간 상관이 없어 Steiger 검정은 못 냄"}
v5["relative_change_vs_demo"] = {"rmse_pct": round(100 * (0.171 - 0.275) / 0.275, 1), "r_pct": round(100 * (0.8766 - 0.6636) / 0.6636, 1)}
v5["relative_change_vs_play"] = {"rmse_pct": round(100 * (0.171 - 0.297) / 0.297, 1), "r_pct": round(100 * (0.8766 - 0.6619) / 0.6619, 1)}
# 그림 판독 (실제 SR, 세계모델 SR, 과제 색 1/2/3). PlayWorld 패널 18점은 모두 보이는 점.
PW = [(0.0, .08, 3), (.05, .12, 1), (.05, 0.0, 2), (.15, .17, 1), (.15, .20, 2), (.30, .20, 3), (.30, .14, 1),
      (.45, .22, 1), (.45, 0.0, 2), (.60, .24, 1), (.50, .44, 2), (.65, .42, 2), (.55, .64, 3), (.65, .62, 3),
      (.80, .70, 1), (.80, .84, 1), (.85, .90, 3), (1.0, .94, 3)]
HD = [(0, .30), (.05, .70), (.05, .08), (.15, .02), (.15, .90), (.30, .60), (.30, .36), (.30, 0), (.45, .50), (.45, .08),
      (.50, .78), (.55, .88), (.60, .54), (.65, .90), (.65, .92), (.80, .68), (.85, .95), (1.0, 1.0)]
HP = [(0, 0), (.05, .09), (.05, .50), (.15, 0), (.15, .03), (.30, 0), (.30, .03), (.30, .74), (.50, .36), (.45, 0), (.45, .06),
      (.55, .68), (.65, .60), (.60, .42), (.65, .02), (.80, .66), (.80, .32), (.85, .90), (1.0, .66)]


def rr(pts):
    a = np.array([[p[0], p[1]] for p in pts])
    return {"n_points_read": len(a), "r": round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 4),
            "rmse": round(float(np.sqrt(((a[:, 0] - a[:, 1]) ** 2).mean())), 3),
            "mean_bias_pred_minus_real": round(float((a[:, 1] - a[:, 0]).mean()), 3)}


v5["figure_read"] = {"PlayWorld": rr(PW), "Human Demo": rr(HD), "Human Play": rr(HP)}
# 과제별 (PlayWorld 패널)
within = {}
for t in (1, 2, 3):
    a = np.array([[p[0], p[1]] for p in PW if p[2] == t])
    within[f"task{t}"] = {"n": len(a), "r": round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3),
                          "rmse": round(float(np.sqrt(((a[:, 0] - a[:, 1]) ** 2).mean())), 3)}
v5["figure_read_PlayWorld_within_task"] = within
v5["figure_read_point_count_note"] = "PlayWorld 패널 보이는 점 18개(과제별 7/5/6). 사람 놀이 패널은 19개로 보임(겹침). 본문은 정책 18개"
R["V5"] = v5

# ------------------------------------------------------------------ V6 그림 6
# 정책 6개: 실제(20회), PlayWorld(50), Baseline(50) 범주별 표기 값
labels = ["Success", "Missed Grasp", "Slide", "Slip", "Premature Release", "Failed Placement", "Irregular Fold", "No Attempt"]
def vec(**kw):
    return [kw.get(k, 0) for k in ["succ", "miss", "slide", "slip", "prem", "fail", "fold", "none"]]
real = [vec(miss=2, fold=3, prem=15), vec(succ=3, none=1, fail=16), vec(succ=6, fail=4, miss=3, slip=7),
        vec(succ=10, fail=8, miss=2), vec(succ=13, fold=1, miss=5, slip=1), vec(succ=20)]
pw = [vec(succ=4, miss=2, fold=9, prem=35), vec(succ=10, none=3, fail=29, miss=8), vec(succ=7, fail=7, miss=6, slip=30),
      vec(succ=22, none=3, fail=5, miss=20), vec(succ=31, slide=6, miss=10, slip=1), vec(succ=47, fold=3)]
bl = [vec(succ=10, miss=8, prem=32), vec(succ=45, fail=5), vec(succ=18, miss=3, fail=29),
      vec(succ=39, none=3, fail=1, miss=7), vec(succ=46, fold=1, slide=1, miss=2), vec(succ=50)]
v6 = {"label_sums": {"Real": [sum(x) for x in real], "PlayWorld": [sum(x) for x in pw], "Baseline": [sum(x) for x in bl]}}
v6["success_rate"] = {"Real": [x[0] / 20 for x in real], "PlayWorld": [x[0] / 50 for x in pw], "Baseline": [x[0] / 50 for x in bl]}
sr_r = np.array(v6["success_rate"]["Real"]); sr_p = np.array(v6["success_rate"]["PlayWorld"]); sr_b = np.array(v6["success_rate"]["Baseline"])
v6["mean_bias_pred_minus_real"] = {"PlayWorld": round(float((sr_p - sr_r).mean()), 3), "Baseline": round(float((sr_b - sr_r).mean()), 3)}
v6["rmse_vs_real"] = {"PlayWorld": round(float(np.sqrt(((sr_p - sr_r) ** 2).mean())), 3), "Baseline": round(float(np.sqrt(((sr_b - sr_r) ** 2).mean())), 3)}
fp, fb = [], []
for i in range(6):
    kr, kp, kb = real[i][0], pw[i][0], bl[i][0]
    fp.append(round(float(stats.fisher_exact([[kr, 20 - kr], [kp, 50 - kp]])[1]), 4))
    fb.append(round(float(stats.fisher_exact([[kr, 20 - kr], [kb, 50 - kb]])[1]), 4))
v6["fisher_p_real_vs_PlayWorld"] = fp
v6["fisher_p_real_vs_Baseline"] = fb
v6["n_policies_differ_p05"] = {"PlayWorld": int(sum(p < 0.05 for p in fp)), "Baseline": int(sum(p < 0.05 for p in fb))}


def tvd(a, b):
    a = np.array(a, float) / sum(a); b = np.array(b, float) / sum(b)
    return 0.5 * float(np.abs(a - b).sum())


v6["failure_mode_tvd_vs_real"] = {"PlayWorld": [round(tvd(real[i], pw[i]), 3) for i in range(6)],
                                  "Baseline": [round(tvd(real[i], bl[i]), 3) for i in range(6)]}
v6["failure_mode_tvd_mean"] = {k: round(float(np.mean(v)), 3) for k, v in v6["failure_mode_tvd_vs_real"].items()}
v6["spearman_real_vs"] = {"PlayWorld": round(float(stats.spearmanr(sr_r, sr_p)[0]), 3), "Baseline": round(float(stats.spearmanr(sr_r, sr_b)[0]), 3)}
v6["pearson_real_vs_six"] = {"PlayWorld": round(float(np.corrcoef(sr_r, sr_p)[0, 1]), 3), "Baseline": round(float(np.corrcoef(sr_r, sr_b)[0, 1]), 3)}
v6["fig7_demo_panel_point_at_real0"] = {"read_pred": 0.30, "pie_pred": sr_b[0], "note": "실제 SR 0.0 정책: 그림 7 사람 시연 패널 판독 0.30, 그림 6 Baseline 원 10/50 = 0.20"}
v6["PlayWorld_pie5_sum_note"] = "다섯째 원 표기 합 48 (31+6+10+1), 나머지 원은 50"
R["V6"] = v6

# ------------------------------------------------------------------ V7 그림 9 (시행 수 20 가정)
f9 = {"carrot": {"steps": [0, 200, 400, 800, 1200], "baseline": [.30, .30, .40, .45, .25], "playworld": [.30, .50, .65, .75, .75]},
      "towel": {"steps": [0, 800, 1200, 2000, 4000], "baseline": [.10, .25, .40, .35, .25], "playworld": [.10, .40, .50, .75, .50]}}
v7 = {"trials_per_checkpoint_assumed": 20, "source": "그림 9 판독 + 부록 D 'With each checkpoint, we perform 20 hardware trials'"}
for t, d in f9.items():
    out = []
    for i, s in enumerate(d["steps"]):
        kp = round(d["playworld"][i] * 20); kb = round(d["baseline"][i] * 20)
        out.append({"step": s, "playworld": f"{kp}/20", "baseline": f"{kb}/20",
                    "fisher_p_pw_vs_baseline": round(float(stats.fisher_exact([[kp, 20 - kp], [kb, 20 - kb]])[1]), 3),
                    "wilson_pw": wilson(kp, 20)})
    base0 = round(d["baseline"][0] * 20)
    best_i = int(np.argmax(d["playworld"]))
    kp = round(d["playworld"][best_i] * 20)
    out_best = {"best_step": d["steps"][best_i], "best": f"{kp}/20",
                "fisher_p_best_vs_initial": round(float(stats.fisher_exact([[kp, 20 - kp], [base0, 20 - base0]])[1]), 4),
                "abs_gain_pp": round(100 * (kp - base0) / 20, 1)}
    kl = round(d["playworld"][-1] * 20)
    out_best["last_checkpoint"] = f"{kl}/20"
    out_best["last_abs_gain_pp"] = round(100 * (kl - base0) / 20, 1)
    out_best["last_fisher_p_vs_initial"] = round(float(stats.fisher_exact([[kl, 20 - kl], [base0, 20 - base0]])[1]), 3)
    v7[t] = {"per_checkpoint": out, "summary": out_best}
v7["abstract_65_percent"] = "수건: 초기 2/20 대 최고 체크포인트(2000스텝) 15/20 → 절대 +65%p(상대 7.5배). 당근: +45%p. 두 값 모두 '최고 체크포인트' 기준"
v7["max_over_checkpoints_selection"] = "체크포인트 4-5개 중 최댓값을 보고. 같은 20회 시행으로 고르고 보고하면 선택 편향 가능(별도 평가 시행 언급 없음)"
R["V7"] = v7

# ------------------------------------------------------------------ V8 그림 11 표
f11 = {"Fruit": {"lpips": [0.0991, 0.0794, 0.0761, 0.0705], "ssim": [0.8333, 0.8612, 0.8697, 0.8757]},
       "Blocks": {"lpips": [0.0978, 0.0713, 0.0684, 0.0640], "ssim": [0.8501, 0.8810, 0.8853, 0.8948]}}
v8 = {}
for k, d in f11.items():
    l = d["lpips"]; s = d["ssim"]
    v8[k] = {"lpips_monotone": all(l[i] > l[i + 1] for i in range(3)), "ssim_monotone": all(s[i] < s[i + 1] for i in range(3)),
             "lpips_rel_change_0_to_100_pct": round(100 * (l[-1] - l[0]) / l[0], 1),
             "share_of_gain_in_first_step_pct": round(100 * (l[0] - l[1]) / (l[0] - l[-1]), 1)}
v8["note"] = "0%→33% 구간에 LPIPS 이득의 대부분(share_of_gain_in_first_step_pct). 33→100%는 완만. 시드·시행 수 표기 없음"
v8["compare_with_table1_30h"] = "표 11의 100% 행(0.0705, 0.0640)은 표 1의 30h LPIPS 평균(0.075)과 다른 시험 집합이므로 직접 비교 불가"
R["V8"] = v8

# ------------------------------------------------------------------ V9 곡선 스케줄 (부록 A)
p0 = np.array([.5, .3, .1, .05, .05]); p1 = np.array([.1, .2, .2, .25, .25])
R["V9"] = {"p0_sum": float(p0.sum()), "p1_sum": float(p1.sum()),
           "rank5_mass_ratio": round(float(p1[-1] / p0[-1]), 1), "rank1_mass_ratio": round(float(p1[0] / p0[0]), 2),
           "ranks_4_5_mass_start_end": [round(float(p0[3:].sum()), 2), round(float(p1[3:].sum()), 2)],
           "unspecified": ["갱신 간격 5000스텝은 있으나 총 스텝 수와 보간 방식(선형 가정)은 표 2에 없음", "거리 임곗값 δ_r 값 없음", "표 1 곡선 행의 데이터 규모 표기 없음"]}

# ------------------------------------------------------------------ V10 요약 수치
R["V10"] = {"abstract_up_to_40pct": "본문에 '40%'가 나오는 곳은 초록뿐(grep). 그림 7의 RMSE 감소 37.8%(대 사람 시연), 42.4%(대 사람 놀이), r 증가 32%와 맞는 정의는 어느 것도 명시되지 않음",
            "five_x_claim": "5배 이상 규모 주장: 사람 시연 곡선이 약 0.6-1.5h에서 평탄(그림 판독 0.38-0.40), PlayWorld는 30h까지(0.89). 사람 데이터는 6h까지만 수집(그림 10 x축)",
            "human_hours_max": 6, "robot_hours_max": 30,
            "benchmark_sources_share_pi05": round(122 / 533, 3)}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
for k in ["V1", "V3", "V5", "V6", "V7"]:
    print(k, json.dumps(R[k], ensure_ascii=False)[:1500])
