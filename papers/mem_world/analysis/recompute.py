"""Mem-World (arXiv:2606.18960 v3) 표·그림 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 표 1, 표 2의 값과 그림 4, 그림 5에서 읽은 값을 아래에 옮겨 적었다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 그림 4와 그림 5의 모든 값이 0.05의 배수라서 셀당 시행 수 n = 20으로 가정한다(논문은 평가 시행 수를 쓰지 않는다).
- 그림 4의 점 위치는 그림 판독 값이다. 두 개의 Ctrl-World 점(π0 wipe table, π0.5 pick&place)은
  Mem-World 점 뒤에 가려져 보이지 않아, 그림에 적힌 회귀식으로 역산해 위치를 정한다(V7).
- 표 1, 표 2에는 분산이 없다. 쌍 표준편차는 민감도 분석의 가정이다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}

# ---------------------------------------------------------------- V1 표 1
T1 = {
    "third": {
        "Cosmos": dict(psnr=22.80, ssim=0.819, lpips=0.089, obj=0.579),
        "Ctrl": dict(psnr=23.17, ssim=0.828, lpips=0.076, obj=0.573),
        "Mem": dict(psnr=25.30, ssim=0.878, lpips=0.054, obj=0.619),
    },
    "wrist": {
        "Ctrl": dict(psnr=17.34, ssim=0.623, lpips=0.281, obj=0.476),
        "Mem": dict(psnr=19.21, ssim=0.691, lpips=0.236, obj=0.524),
    },
}
T2 = {
    "third": {
        "Short": dict(psnr=21.25, ssim=0.802, lpips=0.082, obj=0.526),
        "Stride": dict(psnr=22.58, ssim=0.814, lpips=0.079, obj=0.544),
        "WVMem": dict(psnr=24.78, ssim=0.869, lpips=0.062, obj=0.597),
    },
    "wrist": {
        "Short": dict(psnr=15.04, ssim=0.526, lpips=0.353, obj=0.401),
        "Stride": dict(psnr=17.06, ssim=0.614, lpips=0.295, obj=0.463),
        "WVMem": dict(psnr=18.97, ssim=0.680, lpips=0.248, obj=0.502),
    },
}


def gain(a, b):
    """b가 a보다 얼마나 나은가(큰 쪽이 좋은 지표는 b-a, LPIPS는 a-b), 절대와 상대(%)."""
    out = {}
    for k in a:
        d = b[k] - a[k]
        if k == "lpips":
            d = -d
        out[k] = dict(abs=round(d, 3), rel_pct=round(100 * d / a[k], 1))
    return out


R["V1_table1_gains"] = {
    "third_Mem_vs_Ctrl": gain(T1["third"]["Ctrl"], T1["third"]["Mem"]),
    "third_Mem_vs_Cosmos": gain(T1["third"]["Cosmos"], T1["third"]["Mem"]),
    "wrist_Mem_vs_Ctrl": gain(T1["wrist"]["Ctrl"], T1["wrist"]["Mem"]),
}
R["V2_table2_gains"] = {
    "third_WVMem_vs_Stride": gain(T2["third"]["Stride"], T2["third"]["WVMem"]),
    "third_WVMem_vs_Short": gain(T2["third"]["Short"], T2["third"]["WVMem"]),
    "wrist_WVMem_vs_Stride": gain(T2["wrist"]["Stride"], T2["wrist"]["WVMem"]),
    "wrist_WVMem_vs_Short": gain(T2["wrist"]["Short"], T2["wrist"]["WVMem"]),
}
# 표 1과 표 2 사이: 같은 방향·크기인가
R["V2b_table1_vs_table2"] = {
    "third_psnr_gain_T1_vs_T2": [R["V1_table1_gains"]["third_Mem_vs_Ctrl"]["psnr"]["abs"],
                                 R["V2_table2_gains"]["third_WVMem_vs_Stride"]["psnr"]["abs"]],
    "wrist_psnr_gain_T1_vs_T2": [R["V1_table1_gains"]["wrist_Mem_vs_Ctrl"]["psnr"]["abs"],
                                 R["V2_table2_gains"]["wrist_WVMem_vs_Stride"]["psnr"]["abs"]],
    "Mem_minus_WVMem_third_psnr": round(T1["third"]["Mem"]["psnr"] - T2["third"]["WVMem"]["psnr"], 2),
    "Mem_minus_WVMem_wrist_psnr": round(T1["wrist"]["Mem"]["psnr"] - T2["wrist"]["WVMem"]["psnr"], 2),
    "Ctrl_minus_Stride_third_psnr": round(T1["third"]["Ctrl"]["psnr"] - T2["third"]["Stride"]["psnr"], 2),
    "Ctrl_minus_Stride_wrist_psnr": round(T1["wrist"]["Ctrl"]["psnr"] - T2["wrist"]["Stride"]["psnr"], 2),
}

# ---------------------------------------------------------------- V3 쌍 SD 민감도
N_TRAJ = 34
sens = {}
for name, d in [("third_psnr_Mem_vs_Ctrl", 25.30 - 23.17), ("wrist_psnr_Mem_vs_Ctrl", 19.21 - 17.34),
                ("third_psnr_WVMem_vs_Stride", 24.78 - 22.58), ("wrist_psnr_WVMem_vs_Stride", 18.97 - 17.06)]:
    tcrit = stats.t.ppf(0.975, N_TRAJ - 1)
    s_star = d * math.sqrt(N_TRAJ) / tcrit  # 이 쌍 SD보다 크면 p >= 0.05
    sens[name] = dict(mean_diff_dB=round(d, 2), paired_sd_for_p05_dB=round(s_star, 2))
R["V3_psnr_paired_sd_sensitivity_n34"] = sens

# ---------------------------------------------------------------- V4–V7 그림 4
TASKS = ["pick&place", "wipe table", "pick 1 to 2", "pick 2 to 1", "pick 2 to 2"]
LONG = [2, 3, 4]
real = {"pi0": [.70, .65, .40, .40, .20], "pi05": [.90, .80, .60, .70, .45]}
mem = {"pi0": [.55, .55, .25, .25, .10], "pi05": [.70, .70, .40, .45, .25]}
ctrl_vis = {  # 그림에서 보이는 점
    ("pi0", 0): .60, ("pi0", 2): .10, ("pi0", 3): .10, ("pi0", 4): .05,
    ("pi05", 1): .65, ("pi05", 2): .15, ("pi05", 3): .20, ("pi05", 4): .15,
}
KEYS = [("pi0", i) for i in range(5)] + [("pi05", i) for i in range(5)]
X = np.array([real[p][i] for p, i in KEYS])
YM = np.array([mem[p][i] for p, i in KEYS])

lr_m = stats.linregress(X, YM)
R["V4_fig4_memworld_refit"] = dict(
    r=round(lr_m.rvalue, 4), slope=round(lr_m.slope, 3), intercept=round(lr_m.intercept, 3),
    p=float(lr_m.pvalue), reported=dict(r=0.9704, slope=0.934, intercept=-0.122),
    note="그림 판독 점 10개로 그림에 적힌 r, 기울기, 절편이 재현됨")

# 가려진 두 점(π0 wipe, π0.5 pick&place)을 그림의 회귀식에서 역산
best = None
for a in np.arange(0, 1.0001, .05):
    for b in np.arange(0, 1.0001, .05):
        yy = np.array([ctrl_vis.get(k, a if k == ("pi0", 1) else b) for k in KEYS])
        s = stats.linregress(X, yy)
        err = abs(s.slope - 1.046) + abs(s.intercept + .282) + abs(s.rvalue - .8476)
        if best is None or err < best[0]:
            best = (err, a, b, s)
err, a, b, s = best
ctrl_full = {**ctrl_vis, ("pi0", 1): round(a, 2), ("pi05", 0): round(b, 2)}
YC = np.array([ctrl_full[k] for k in KEYS])
R["V5_fig4_ctrlworld_hidden_points"] = dict(
    hidden_pi0_wipe=round(a, 2), hidden_pi05_pickplace=round(b, 2),
    refit=dict(r=round(s.rvalue, 4), slope=round(s.slope, 3), intercept=round(s.intercept, 3)),
    reported=dict(r=0.8476, slope=1.046, intercept=-0.282),
    same_as_memworld=bool(round(a, 2) == mem["pi0"][1] and round(b, 2) == mem["pi05"][0]),
    note="가려진 두 점은 Mem-World 점과 같은 값에서 회귀식이 정확히 재현됨(두 모델이 같은 성공률)")

n = len(X)
r_m, r_c = lr_m.rvalue, s.rvalue


def fisher_ci(r, n, z=1.959964):
    zr = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return math.tanh(zr - z * se), math.tanh(zr + z * se)


def pval_r(r, n):
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))


# 두 상관 차이(종속, 공통 x): Meng-Rosenthal-Rubin
r_mc = float(np.corrcoef(YM, YC)[0, 1])


def meng(r1, r2, r12, n):
    z1, z2 = math.atanh(r1), math.atanh(r2)
    rm2 = (r1 ** 2 + r2 ** 2) / 2
    f = min((1 - r12) / (2 * (1 - rm2)), 1)
    h = (1 - f * rm2) / (1 - rm2)
    z = (z1 - z2) * math.sqrt((n - 3) / (2 * (1 - r12) * h))
    return z, float(2 * stats.norm.sf(abs(z)))


zm, pm = meng(r_m, r_c, r_mc, n)
R["V6_fig4_inference"] = dict(
    n_points=n,
    p_memworld_n10=pval_r(r_m, n), p_ctrlworld_n10=pval_r(r_c, n),
    fisher95_memworld=[round(v, 3) for v in fisher_ci(r_m, n)],
    fisher95_ctrlworld=[round(v, 3) for v in fisher_ci(r_c, n)],
    rel_gain_pct=round(100 * (0.9704 / 0.8476 - 1), 1),
    abs_gain=round(0.9704 - 0.8476, 4),
    r_between_worldmodels=round(r_mc, 3),
    meng_z=round(zm, 2), meng_p=round(pm, 3),
    note="n = 10은 5과제 × 2정책. 같은 과제의 두 정책은 독립이 아니므로 p는 낙관적")

# 과제 평균(n = 5)과 정책별(n = 5)
def task_mean(v):
    return np.array([(v[("pi0", i)] + v[("pi05", i)]) / 2 for i in range(5)])


rt = task_mean({k: real[k[0]][k[1]] for k in KEYS})
mt = task_mean({k: mem[k[0]][k[1]] for k in KEYS})
ct = task_mean(ctrl_full)
R["V6b_task_level_n5"] = dict(
    r_mem=round(float(np.corrcoef(rt, mt)[0, 1]), 4), r_ctrl=round(float(np.corrcoef(rt, ct)[0, 1]), 4),
    p_mem=pval_r(float(np.corrcoef(rt, mt)[0, 1]), 5), p_ctrl=pval_r(float(np.corrcoef(rt, ct)[0, 1]), 5))
per_pol = {}
for p in ("pi0", "pi05"):
    xr = np.array(real[p])
    ym = np.array(mem[p])
    yc = np.array([ctrl_full[(p, i)] for i in range(5)])
    per_pol[p] = dict(r_mem=round(float(np.corrcoef(xr, ym)[0, 1]), 3), r_ctrl=round(float(np.corrcoef(xr, yc)[0, 1]), 3))
R["V6c_per_policy_n5"] = per_pol

# 순위 상관, 편차
sp_m, sp_c = stats.spearmanr(X, YM)[0], stats.spearmanr(X, YC)[0]
kt_m, kt_c = stats.kendalltau(X, YM)[0], stats.kendalltau(X, YC)[0]
gap_m, gap_c = X - YM, X - YC
R["V7_rank_and_gap"] = dict(
    spearman=dict(mem=round(float(sp_m), 3), ctrl=round(float(sp_c), 3)),
    kendall=dict(mem=round(float(kt_m), 3), ctrl=round(float(kt_c), 3)),
    mean_gap_real_minus_wm=dict(mem=round(float(gap_m.mean()), 3), ctrl=round(float(gap_c.mean()), 3)),
    sd_gap=dict(mem=round(float(gap_m.std(ddof=1)), 3), ctrl=round(float(gap_c.std(ddof=1)), 3)),
    mae=dict(mem=round(float(np.abs(gap_m).mean()), 3), ctrl=round(float(np.abs(gap_c).mean()), 3)),
    gap_by_task_mem=[round(float(g), 3) for g in task_mean({k: gap_m[i] for i, k in enumerate(KEYS)})],
    gap_by_task_ctrl=[round(float(g), 3) for g in task_mean({k: gap_c[i] for i, k in enumerate(KEYS)})],
    wm_over_real_count=dict(mem=int((YM > X).sum()), ctrl=int((YC > X).sum())),
)
# 단문 이동의 영향: 장기 과제 6점 vs 단기 4점
short_idx = [i for i, k in enumerate(KEYS) if k[1] in (0, 1)]
long_idx = [i for i, k in enumerate(KEYS) if k[1] in LONG]
R["V7b_short_vs_long"] = dict(
    mean_gap_short=dict(mem=round(float(gap_m[short_idx].mean()), 3), ctrl=round(float(gap_c[short_idx].mean()), 3)),
    mean_gap_long=dict(mem=round(float(gap_m[long_idx].mean()), 3), ctrl=round(float(gap_c[long_idx].mean()), 3)),
    r_short4=dict(mem=round(float(np.corrcoef(X[short_idx], YM[short_idx])[0, 1]), 3),
                  ctrl=round(float(np.corrcoef(X[short_idx], YC[short_idx])[0, 1]), 3)),
    r_long6=dict(mem=round(float(np.corrcoef(X[long_idx], YM[long_idx])[0, 1]), 3),
                 ctrl=round(float(np.corrcoef(X[long_idx], YC[long_idx])[0, 1]), 3)),
    n_cells_equal=int(np.sum(np.isclose(YM, YC))))

# 같은 과제 안에서 두 정책(π0, π0.5)의 순서를 맞혔는가
def order_agree(v):
    return int(sum((v[("pi05", i)] > v[("pi0", i)]) == (real["pi05"][i] > real["pi0"][i]) and v[("pi05", i)] != v[("pi0", i)]
               for i in range(5)))


R["V7c_within_task_policy_order"] = dict(
    pairs=5, memworld_agree=order_agree({k: mem[k[0]][k[1]] for k in KEYS}), ctrlworld_agree=order_agree(ctrl_full),
    real_pi05_gt_pi0_all=bool(all(real["pi05"][i] > real["pi0"][i] for i in range(5))))

# 과제 단위 부트스트랩(과제 5개를 복원추출, 두 정책 점을 함께 가져감)
rng = np.random.default_rng(0)
diffs, rms, rcs = [], [], []
for _ in range(20000):
    t = rng.integers(0, 5, 5)
    ids = [i + 5 * pi for i in t for pi in (0, 1)]
    xx, ym, yc = X[ids], YM[ids], YC[ids]
    if np.std(xx) == 0 or np.std(ym) == 0 or np.std(yc) == 0:
        continue
    a1, a2 = np.corrcoef(xx, ym)[0, 1], np.corrcoef(xx, yc)[0, 1]
    rms.append(a1)
    rcs.append(a2)
    diffs.append(a1 - a2)
diffs = np.array(diffs)
R["V8_cluster_bootstrap_tasks"] = dict(
    n_valid=len(diffs), diff_mean=round(float(diffs.mean()), 3),
    diff_ci95=[round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)],
    p_diff_le_0=round(float((diffs <= 0).mean()), 3),
    r_mem_ci95=[round(float(np.percentile(rms, 2.5)), 3), round(float(np.percentile(rms, 97.5)), 3)],
    r_ctrl_ci95=[round(float(np.percentile(rcs, 2.5)), 3), round(float(np.percentile(rcs, 97.5)), 3)])

# 과제 하나씩 빼기
loo = []
for t in range(5):
    ids = [i + 5 * pi for i in range(5) if i != t for pi in (0, 1)]
    loo.append(dict(left_out=TASKS[t], r_mem=round(float(np.corrcoef(X[ids], YM[ids])[0, 1]), 3),
                    r_ctrl=round(float(np.corrcoef(X[ids], YC[ids])[0, 1]), 3)))
R["V8b_leave_one_task_out"] = loo

# 잡음만으로 생기는 차이: 두 세계모델이 같은 "참" 관계를 갖고 셀당 20회씩 측정할 때
rng = np.random.default_rng(1)
N_CELL = 20
true_real = X.copy()
res = []
for gap in (0.0, 0.15):
    true_wm = np.clip(true_real - gap, 0.02, 0.98)
    dd, rr = [], []
    for _ in range(20000):
        xr = rng.binomial(N_CELL, true_real) / N_CELL
        a1 = rng.binomial(N_CELL, true_wm) / N_CELL
        a2 = rng.binomial(N_CELL, true_wm) / N_CELL
        if xr.std() == 0 or a1.std() == 0 or a2.std() == 0:
            continue
        r1, r2 = np.corrcoef(xr, a1)[0, 1], np.corrcoef(xr, a2)[0, 1]
        rr.append(r1)
        dd.append(r1 - r2)
    dd = np.array(dd)
    res.append(dict(true_gap=gap, mean_r_if_perfect_truth=round(float(np.mean(rr)), 3),
                    sd_of_r_diff=round(float(dd.std()), 3),
                    p_diff_ge_observed_0123=round(float((dd >= 0.1228).mean()), 3),
                    p_abs_diff_ge_observed=round(float((np.abs(dd) >= 0.1228).mean()), 3),
                    p_r_ge_0p9704=round(float((np.array(rr) >= 0.9704).mean()), 4),
                    p_r_ge_0p8476=round(float((np.array(rr) >= 0.8476).mean()), 3)))
R["V9_noise_only_null_n20"] = dict(
    setting="두 세계모델의 참 성공률이 같고(실로봇 − gap), 모든 셀을 n = 20 이항 측정으로 가정, 20,000회, 시드 1",
    results=res)

# ---------------------------------------------------------------- V10 그림 5
base = np.array([.60, .70, .45])
ft = np.array([.75, .85, .55])
names = ["pick 1 to 2", "pick 2 to 1", "pick 2 to 2"]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 3), round(c + h, 3)


f5 = {"avg_base_exact": round(float(base.mean()), 4), "avg_ft_exact": round(float(ft.mean()), 4),
      "avg_diff_exact_pp": round(float((ft - base).mean() * 100), 1),
      "reported_avg": [0.58, 0.72], "reported_diff_pp": 14,
      "per_task": []}
for nm, b, f in zip(names, base, ft):
    kb, kf = round(b * 20), round(f * 20)
    f5["per_task"].append(dict(task=nm, base=f"{kb}/20", ft=f"{kf}/20", diff_pp=round((f - b) * 100),
                               fisher_p=round(fisher(kf, 20, kb, 20), 3),
                               wilson_base=wilson(kb, 20), wilson_ft=wilson(kf, 20)))
kb, kf = int(round(base.sum() * 20)), int(round(ft.sum() * 20))
f5["pooled"] = dict(base=f"{kb}/60", ft=f"{kf}/60", fisher_p=round(fisher(kf, 60, kb, 60), 3),
                    wilson_base=wilson(kb, 60), wilson_ft=wilson(kf, 60))
# 검출 가능한 최소 차이(두 비율 z 근사, 검정력 80%, 양측 5%), 기준 0.58
for n_ in (20, 60):
    p1 = 0.5833
    for d in np.arange(0.05, 0.6, 0.005):
        p2 = p1 + d
        pb = (p1 + p2) / 2
        se0 = math.sqrt(2 * pb * (1 - pb) / n_)
        se1 = math.sqrt(p1 * (1 - p1) / n_ + p2 * (1 - p2) / n_)
        z = (d - 1.959964 * se0) / se1
        if stats.norm.cdf(z) >= 0.8:
            f5[f"mde_pp_n{n_}_power80"] = round(100 * d, 1)
            break
# 20–30궤적 / 200궤적
f5["kept_fraction_if_all_successes_kept"] = [20 / 200, 30 / 200]
f5["fig4_pi05_long_horizon_worldmodel_success"] = [mem["pi05"][i] for i in LONG]
f5["fig4_pi05_long_horizon_real_success"] = [real["pi05"][i] for i in LONG]
f5["fig4_real_equals_fig5_base"] = bool(np.allclose([real["pi05"][i] for i in LONG], base))
R["V10_fig5_policy_improvement"] = f5

# ---------------------------------------------------------------- V11 식 2
lam, Hfrac = 0.1, 0.3
rows = []
for frac in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0):  # (T − t) / Tmax
    decay = lam + (1 - lam) * 2 ** (-frac / Hfrac)
    rows.append(dict(age_over_Tmax=frac, temporal_factor=round(decay, 3)))
R["V11_eq2"] = dict(
    temporal_factor_vs_age=rows,
    limit_age_large=lam,
    manip_flag_factor=dict(m0=round(math.log(math.e + 0), 3), m1=round(math.log(math.e + 1), 3),
                           ratio=round(math.log(math.e + 1) / math.log(math.e), 3)),
    depth_factor=[dict(d=d, v=round(1 / (1 + d), 3)) for d in (0.2, 0.5, 1.0, 2.0)],
    note="식 2에서 task 플래그는 최대 1.313배, 시간 감쇠는 0.19–1배, 깊이 항은 d = 0.2–2에서 0.83–0.33배")

# ---------------------------------------------------------------- V12 규모 산술
R["V12_scale_arithmetic"] = dict(
    droid_traj=95599, sampled_traj=11000, sampled_fraction_pct=round(100 * 11000 / 95599, 1),
    train_gpu_hours_h100=8 * 48, train_batch=32,
    note="2일 = 48시간 가정(논문은 'approximately 2 days')",
    rollout_len_s=20, steps=20, secs_per_step=1.0,
    real_episodes_posttrain=5 * 50,
)

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1))
print(json.dumps(R, ensure_ascii=False, indent=1))
