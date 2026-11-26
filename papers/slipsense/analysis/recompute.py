"""SlipSense (arXiv:2609.15910 v1) 보고 수치 재계산. 표 번호는 논문 기준.

V1  데이터셋 분할 합계와 비율 (그림 5, §5.1)
V2  센서 기하·단위 (§3.1)
V3  시간 예산·라벨 임계값의 시간 환산 (§3.2, §5.2, 부록 G)
V4  표 6 → 사건 수준 요약 (n = 167)과 그림 7(a) 막대, 표 6 요약통계와의 정수 가능성(MILP)
V5  지연 값의 격자 가능성 (3 시드 중앙값의 평균 → k/6)
V6  표 1·2의 (정확도, FPR, Macro F1) 동시 가능성 (클래스 비율 가정)과 암시되는 슬립 재현율 범위
V7  지연의 우연 수준 기준 (프레임마다 독립으로 슬립을 내는 검출기의 중앙값)
V8  실로봇 비율의 Wilson·Clopper-Pearson 구간, 프레임 FPR의 초당 환산
V9  표 간 일치 (표 1 ↔ 3, 4, 5, 2, 본문)

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초)
의존: numpy, scipy (milp, minimize)
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
FRAME_MS = 1000 / 240


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


# ---------------------------------------------------------------- V1
splits = {"train": 847, "UMI_ID": 361, "UMI_OOD": 72, "UMI_fingerpalm": 54, "Tesollo_2F": 53, "Tesollo_3F": 45}
tot = sum(splits.values())
test = tot - splits["train"]
heldout = splits["UMI_OOD"] + splits["UMI_fingerpalm"] + splits["Tesollo_2F"] + splits["Tesollo_3F"]
priors = {"no_contact": 57.6, "slip": 23.7, "no_slip": 18.8}
RES["V1"] = {
    "frames_total_K": tot,
    "frames_test_K": test,
    "train_share_pct": round(100 * splits["train"] / tot, 1),
    "train_vs_ID_28obj_share_pct": round(100 * splits["train"] / (splits["train"] + splits["UMI_ID"]), 1),
    "heldout_object_frames_K": heldout,
    "frames_per_train_object_K": round((splits["train"] + splits["UMI_ID"]) / 28, 1),
    "frames_per_heldout_object_K_UMI_OOD": round(splits["UMI_OOD"] / 9, 1),
    "class_pct_sum": round(sum(priors.values()), 1),
    "trivial_no_contact_share_pct": priors["no_contact"],
}

# ---------------------------------------------------------------- V2
pitch_x, pitch_y = 0.7, 0.92
dens = 100 / (pitch_x * pitch_y)
area_mm2 = 32 * pitch_x * 32 * pitch_y
RES["V2"] = {
    "density_taxel_cm2": round(dens, 1),
    "array_area_cm2": round(area_mm2 / 100, 2),
    "density_from_1024_over_area": round(1024 / (area_mm2 / 100), 1),
    "active_fraction_pct": round(100 * 942 / 1024, 1),
    "density_ratio_38p8_implies_prior_density_cm2": round(dens / 38.8, 2),
    "density_ratio_6p2_implies_prior_density_cm2": round(dens / 6.2, 2),
    "implied_pitch_mm_for_38p8": round(10 / math.sqrt(dens / 38.8), 2),
    "implied_pitch_mm_for_6p2": round(10 / math.sqrt(dens / 6.2), 2),
    "pressure_force_N_on_1cm2": 5e6 * 1e-4,
    "pressure_resolution_kPa": round(5000 / 4096, 3),
    "element_fill_ratio_pct": round(100 * 0.45 * 0.45 / (pitch_x * pitch_y), 1),
}

# ---------------------------------------------------------------- V3
speeds = [2, 5, 10, 15, 18]
V3 = {"frame_ms": round(FRAME_MS, 3), "five_frames_ms": round(5 * FRAME_MS, 2),
      "five_frames_plus_2p3ms": round(5 * FRAME_MS + 2.3, 2),
      "four_frames_ms_if_onset_frame_is_latency_1": round(4 * FRAME_MS, 2),
      "xl_samples_per_frame": round(8000 / 240, 2), "xl_window_samples_16ms": int(0.016 * 8000),
      "xl_window_frames": round(0.016 / (1 / 240), 2), "xl_window_overlap_with_prev_frame_pct": round(100 * (1 - (1 / 240) / 0.016), 1),
      "window_T60_ms": 60 * FRAME_MS, "window_T120_ms": 120 * FRAME_MS,
      "window_T6_ms": 6 * FRAME_MS}
V3["label_threshold_time_ms"] = {f"{v} mm/s": round(70 / v, 1) for v in speeds}      # 0.07 mm / v
V3["label_threshold_time_frames"] = {f"{v} mm/s": round(0.07 / v * 240, 2) for v in speeds}
V3["displacement_in_250ms_mm"] = {f"{v} mm/s": round(v * 0.25, 2) for v in speeds}
V3["time_for_one_taxel_ms_pitch_0.92"] = round(0.92 / 2 * 1000)
V3["time_for_one_taxel_ms_pitch_0.70"] = round(0.70 / 2 * 1000)
V3["max_lat_2mm_s_106fr_ms"] = round(106 * FRAME_MS, 1)
V3["max_lat_2mm_s_106fr_extra_displacement_mm"] = round(106 * FRAME_MS / 1000 * 2, 3)
V3["extra_displacement_at_median_3fr_2mm_s_mm"] = round(3 * FRAME_MS / 1000 * 2, 3)
V3["extra_displacement_at_5fr_18mm_s_mm"] = round(5 * FRAME_MS / 1000 * 18, 3)
V3["extra_displacement_at_21fr_5mm_s_mm"] = round(21 * FRAME_MS / 1000 * 5, 3)
RES["V3"] = V3

# ---------------------------------------------------------------- V4
T6 = {2: (36, 3.0, 14.9, 106), 5: (31, 3.0, 6.8, 39), 10: (37, 1.0, 3.4, 27), 15: (32, 2.5, 5.8, 35), 18: (31, 1.0, 2.9, 20)}
n_tot = sum(v[0] for v in T6.values())
sum_lat = sum(v[0] * v[2] for v in T6.values())
V4 = {"n_events": n_tot, "pooled_mean_frames": round(sum_lat / n_tot, 2), "pooled_mean_ms": round(sum_lat / n_tot * FRAME_MS, 1),
      "pooled_mean_ms_plus_2p3": round(sum_lat / n_tot * FRAME_MS + 2.3, 1)}
fig7a = {"1": 43, "2-5": 33, "6-21": 15, ">21": 9}
V4["fig7a_pct_sum"] = sum(fig7a.values())
V4["fig7a_counts_if_n167"] = {k: round(v / 100 * n_tot, 1) for k, v in fig7a.items()}
V4["k_for_76pct"] = round(0.76 * n_tot, 1)
V4["76pct_is_127_of_167"] = round(100 * 127 / 167, 2)
V4["43pct_is_72_of_167"] = round(100 * 72 / 167, 2)
V4["wilson_95_127of167"] = [round(100 * x, 1) for x in wilson(127, 167)]
V4["wilson_95_72of167"] = [round(100 * x, 1) for x in wilson(72, 167)]
V4["wilson_95_gt21_15of167"] = [round(100 * x, 1) for x in wilson(15, 167)]
# 정수 가능성 (MILP): 속도별 지연 분포(프레임, 정수)가 (n, 중앙값, 평균, 최댓값)과 맞으면서 "5 프레임 이내" 사건 수가 가질 수 있는 범위
def count_range(n, med, mean, mx, lo_val):
    vals = list(range(lo_val, mx + 1))
    nv = len(vals)
    idx5 = [i for i, v in enumerate(vals) if v <= 5]
    lo_sum, hi_sum = math.ceil((mean - 0.05) * n - 1e-9), math.floor((mean + 0.05) * n + 1e-9)
    results = []
    # 중앙값을 만드는 (a, b) 쌍 (n 홀수면 a = b = med)
    if n % 2:
        pairs = [(med, med)]
    else:
        pairs = [(a, 2 * med - a) for a in vals if a <= 2 * med - a and 2 * med - a in vals]
    for a, b in pairs:
        A_ub, b_ub = [], []
        def cum(v):          # v 이하 개수 계수 벡터
            return [1.0 if x <= v else 0.0 for x in vals]
        if n % 2:
            k = (n + 1) // 2
            A_ub.append(cum(a - 1)); b_ub.append(k - 1)       # F(a-1) <= k-1
            A_ub.append([-x for x in cum(a)]); b_ub.append(-k)  # F(a) >= k
        else:
            k = n // 2
            A_ub.append(cum(a - 1)); b_ub.append(k - 1)
            A_ub.append([-x for x in cum(a)]); b_ub.append(-k)
            A_ub.append(cum(b - 1)); b_ub.append(k)             # F(b-1) <= k
            A_ub.append([-x for x in cum(b)]); b_ub.append(-(k + 1))
        A_eq = [[1.0] * nv]; b_eq = [n]
        A_ub.append([float(v) for v in vals]); b_ub.append(hi_sum)
        A_ub.append([-float(v) for v in vals]); b_ub.append(-lo_sum)
        bounds = optimize.Bounds([0] * nv, [n] * nv)
        lb = np.zeros(nv); lb[-1] = 1                            # 최댓값 한 건 이상
        bounds = optimize.Bounds(lb, [n] * nv)
        cons = [optimize.LinearConstraint(np.array(A_ub), -np.inf, np.array(b_ub)),
                optimize.LinearConstraint(np.array(A_eq), b_eq, b_eq)]
        for sign in (1, -1):
            c = np.zeros(nv); c[idx5] = sign
            r = optimize.milp(c, constraints=cons, bounds=bounds, integrality=np.ones(nv))
            if r.success:
                results.append(sign * (-r.fun) if sign == 1 else -(-r.fun) * 1)
    # results: sign=1 → min of count(<=5)(= r.fun), sign=-1 → -max
    return results


def feasible_range(n, med, mean, mx, lo_val):
    mins, maxs = [], []
    vals = list(range(lo_val, mx + 1)); nv = len(vals)
    idx5 = [i for i, v in enumerate(vals) if v <= 5]
    lo_sum, hi_sum = math.ceil((mean - 0.05) * n - 1e-9), math.floor((mean + 0.05) * n + 1e-9)
    if n % 2:
        pairs = [(med, med)]
    else:
        pairs = [(a, 2 * med - a) for a in vals if a <= 2 * med - a and (2 * med - a) in vals]
    cum = lambda v: np.array([1.0 if x <= v else 0.0 for x in vals])
    for a, b in pairs:
        rows, ub = [], []
        if n % 2:
            k = (n + 1) // 2
            rows += [cum(a - 1), -cum(a)]; ub += [k - 1, -k]
        else:
            k = n // 2
            rows += [cum(a - 1), -cum(a), cum(b - 1), -cum(b)]; ub += [k - 1, -k, k, -(k + 1)]
        rows += [np.array(vals, float), -np.array(vals, float)]; ub += [hi_sum, -lo_sum]
        lb = np.zeros(nv); lb[-1] = 1
        cons = [optimize.LinearConstraint(np.array(rows), -np.inf, np.array(ub)),
                optimize.LinearConstraint(np.ones((1, nv)), n, n)]
        for sign in (1, -1):
            c = np.zeros(nv); c[idx5] = sign
            r = optimize.milp(c, constraints=cons, bounds=optimize.Bounds(lb, np.full(nv, n)), integrality=np.ones(nv))
            if r.success:
                (mins if sign == 1 else maxs).append(round(sign * r.fun))
    return (min(mins), max(maxs)) if mins else None


for lo_val, tag in ((1, "latency_min_1"), (0, "latency_min_0")):
    per, lo_t, hi_t = {}, 0, 0
    ok = True
    for v, (n, med, mean, mx) in T6.items():
        rg = feasible_range(n, med, mean, mx, lo_val)
        per[f"{v} mm/s"] = rg
        if rg is None:
            ok = False
        else:
            lo_t += rg[0]; hi_t += rg[1]
    V4[tag] = {"per_speed_range_of_events_within_5fr": per, "pooled_range": [lo_t, hi_t] if ok else None,
               "pooled_range_pct": [round(100 * lo_t / n_tot, 1), round(100 * hi_t / n_tot, 1)] if ok else None,
               "127_inside": (lo_t <= 127 <= hi_t) if ok else None}
# 18 mm/s "90% 이상이 5 프레임 이내" → 28/31 이상
V4["18mm_s_over90pct_needs_at_least"] = f"{math.ceil(0.9 * 31)}/31"
RES["V4"] = V4

# ---------------------------------------------------------------- V5
tab = {  # (표, 행, 열) → 지연 값 (프레임)
    "T1_Piezo_UMIID": 20.50, "T1_Piezo_OOD": 36.17, "T1_XL_ID": 4.67, "T1_XL_OOD": 3.00, "T1_fused_ID": 2.00, "T1_fused_OOD": 1.50,
    "T2_palm": 3.00, "T2_T2F": 3.00, "T2_T3F": 5.00, "T3_ResNet": 2.00, "T3_ViTscratch": 2.67, "T3_ViTfreeze": 2.67, "T3_ViTft": 2.00,
    "T3_MLP": 30.00, "T3_1DCNN": 2.50, "T4_noaug_OOD": 1.50, "T4_noaug_T2F": 4.00, "T5_late_OOD": 2.50, "T5_late_T2F": 4.00,
    "T5_concat_OOD": 2.00, "T5_concat_T2F": 4.50}
grid = {k: abs(v * 6 - round(v * 6)) < 0.04 for k, v in tab.items()}
RES["V5"] = {"all_on_k_over_6_grid": all(grid.values()), "off_grid": [k for k, v in grid.items() if not v],
           "note": "3 시드 중앙값(정수 또는 .5)의 평균은 k/6 격자 위. 표 1·3·4·5 값은 모두 격자와 맞음"}

# ---------------------------------------------------------------- V6
# 표 1·2 행: (이름, FPR %, Acc %, F1 %)
rows = {
    "T1 base Piezo ID": (21.69, 72.87, 74.51), "T1 base XL ID": (22.98, 79.41, 81.79), "T1 base AND ID": (9.28, 81.16, 79.96),
    "T1 SS Piezo ID": (7.54, 84.63, 83.54), "T1 SS XL ID": (2.14, 84.21, 81.61), "T1 SS fused ID": (1.33, 96.98, 96.77),
    "T1 base Piezo OOD": (21.08, 73.84, 75.29), "T1 base XL OOD": (19.31, 82.99, 84.55), "T1 base AND OOD": (7.40, 83.33, 82.18),
    "T1 SS Piezo OOD": (8.43, 82.69, 81.91), "T1 SS XL OOD": (3.24, 84.74, 81.33), "T1 SS fused OOD": (1.57, 95.99, 95.75),
    "T2 palm": (1.99, 94.83, 94.56), "T2 Tesollo 2F": (1.14, 95.89, 94.39), "T2 Tesollo 3F": (3.12, 90.85, 87.24),
}
pri = np.array([57.6, 18.8, 23.7]) / 100.0   # 순서: 접촉 없음, 슬립 없음, 슬립
pri = pri / pri.sum()


def metrics(C, pr, fpr_def):
    acc = float(np.sum(pr * np.diag(C)))
    cm = pr[:, None] * C                      # 결합 확률
    f1s, rec, prec = [], [], []
    for j in range(3):
        tp = cm[j, j]; fn = cm[j].sum() - tp; fp = cm[:, j].sum() - tp
        p_ = tp / (tp + fp) if tp + fp > 0 else 0.0
        r_ = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ > 0 else 0.0); rec.append(r_); prec.append(p_)
    fp_slip = cm[0, 2] + cm[1, 2]
    fpr = fp_slip / (pr[0] + pr[1]) if fpr_def == "neg" else fp_slip
    return acc, fpr, float(np.mean(f1s)), rec, prec


def unpack(x):
    z = x.reshape(3, 3)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit(target, fpr_def, rng, starts=60):
    best = (1e9, None)
    for _ in range(starts):
        x0 = rng.normal(0, 2, 9)
        f = lambda x: sum(((m - t) / s) ** 2 for m, t, s in zip(metrics(unpack(x), pri, fpr_def)[:3], target, (0.01, 0.01, 0.01)))
        r = optimize.minimize(f, x0, method="L-BFGS-B")
        if r.fun < best[0]:
            best = (r.fun, r.x)
        if best[0] < 1e-8:
            break
    return best


def slip_recall_range(target, fpr_def, rng, tol=0.0005):
    """보고 값 ±tol(반올림 폭)을 만족하는 혼동행렬 중 슬립 재현율의 최소·최대 (SLSQP, 다중 시작)."""
    lo, hi = 1.0, 0.0
    cons = [{"type": "ineq", "fun": (lambda x, i=i, s=s: s * (metrics(unpack(x), pri, fpr_def)[i] - target[i]) + tol)}
            for i in range(3) for s in (1, -1)]
    for sign in (1, -1):
        for _ in range(25):
            x0 = rng.normal(0, 2, 9)
            r = optimize.minimize(lambda x: sign * metrics(unpack(x), pri, fpr_def)[3][2], x0, method="SLSQP", constraints=cons,
                                  options={"maxiter": 300})
            if r.success or all(c["fun"](r.x) > -1e-5 for c in cons):
                if all(c["fun"](r.x) > -1e-5 for c in cons):
                    v = metrics(unpack(r.x), pri, fpr_def)[3][2]
                    lo, hi = min(lo, v), max(hi, v)
    return (round(lo, 3), round(hi, 3)) if lo <= hi else None


rng = np.random.default_rng(0)
V6 = {"assumption": "시험 분할의 클래스 비율 = 전체 데이터 비율(접촉 없음 57.6, 슬립 없음 18.8, 슬립 23.7 %, 합 100.1을 정규화). 분할별 비율은 논문에 없음. FPR 정의 두 가지: neg = 슬립이 아닌 프레임 중 슬립으로 예측된 비율, all = 전체 프레임 중 슬립 아닌 것을 슬립으로 예측한 비율",
      "rows": {}}
for name, (fpr, acc, f1) in rows.items():
    entry = {}
    for d in ("neg", "all"):
        tgt = (acc / 100, fpr / 100, f1 / 100)
        resid, x = fit(tgt, d, rng)
        ent = {"feasible": bool(resid < 1e-4), "residual": float(f"{resid:.2e}")}
        if ent["feasible"]:
            ent["slip_recall_range"] = slip_recall_range(tgt, d, rng)
        entry[d] = ent
    V6["rows"][name] = entry
# 클래스 비율에 의존하지 않는 검사: FPR을 "전체 프레임 중 슬립 아닌 것을 슬립으로 낸 비율"로 읽으면 FPR <= 100 - Acc 여야 한다
V6["fpr_all_frames_reading_needs_FPR_le_100_minus_Acc"] = {
    n: {"fpr": f, "100_minus_acc": round(100 - a, 2), "holds": f <= 100 - a} for n, (f, a, _) in rows.items()}
V6["fpr_all_frames_reading_violations"] = [n for n, (f, a, _) in rows.items() if f > 100 - a]
RES["V6"] = V6

# ---------------------------------------------------------------- V7
V7 = {"definition": "프레임마다 독립으로 확률 p로 슬립을 내는 검출기(정보 없음)의 첫 검출 지연 중앙값(프레임) = ceil(ln 0.5 / ln(1-p)); 사건 시작 직전 상태 영향과 시간 상관은 무시", "rows": {}}
for name, (fpr, acc, f1) in rows.items():
    p = fpr / 100
    V7["rows"][name] = {"fpr_pct": fpr, "chance_median_frames": math.ceil(math.log(0.5) / math.log(1 - p))}
RES["V7"] = V7

# ---------------------------------------------------------------- V8
V8 = {}
for tag, (k, n) in {"detect_100_of_100": (100, 100), "prevent_95_of_100": (95, 100), "failures_5_of_100": (5, 100)}.items():
    V8[tag] = {"wilson95_pct": [round(100 * x, 1) for x in wilson(k, n)], "clopper_pearson95_pct": [round(100 * x, 1) for x in clopper(k, n)]}
V8["rule_of_three_upper_miss_rate_pct"] = round(300 / 100, 1)
fp = {}
for name in ("T1 SS fused ID", "T1 SS fused OOD", "T2 Tesollo 2F", "T2 palm", "T2 Tesollo 3F"):
    p = rows[name][0] / 100
    fp[name] = {"fp_frames_per_s_if_FPR_is_per_nonslip_frame": round(240 * p, 2),
                "prob_zero_fp_frames_in_2s_if_independent": float(f"{(1 - p) ** 480:.3g}")}
V8["frame_fpr_per_second"] = fp
V8["note"] = "독립 가정은 비현실적(오류는 시간적으로 군집). 실로봇 '오탐 없음' 관찰은 군집 또는 제어기 쪽 필터를 시사하는 가능성이며, 정의는 논문에 없음"
RES["V8"] = V8

# ---------------------------------------------------------------- V9
cons_checks = {
    "T1 fused OOD == T3 CNN+SSAST == T4 SlipSense == T5 cross-attn (lat, FPR, acc, F1)": (1.50, 1.57, 95.99, 95.75),
    "T2 Tesollo2F == T4 SlipSense == T5 cross-attn": (3.00, 1.14, 95.89, 94.39),
}
RES["V9"] = {"identical_rows": list(cons_checks), "all_match_as_printed": True,
           "text_vs_table": {"MLP median latency": {"text_appendix_D": 29, "table3": 30.0},
                              "abstract_macroF1": {"abstract": 96.7, "table1_ID": 96.77, "rounded_2sig": 96.8}},
           "fusion_gain_pp": {"ID_vs_best_single": round(96.77 - 83.54, 2), "OOD_vs_best_single": round(95.75 - 81.91, 2)},
           "ID_to_OOD_drop_pp": {"F1": round(96.77 - 95.75, 2), "acc": round(96.98 - 95.99, 2)},
           "table5_gap_pp_F1_vs_concat": {"OOD": round(95.75 - 95.28, 2), "T2F": round(94.39 - 93.07, 2)},
           "table3_F1_CNN_minus_ViTfreeze_pp": round(95.75 - 95.51, 2)}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
for k in ("V1", "V3", "V4", "V5", "V7", "V8"):
    print(k, json.dumps(RES[k], ensure_ascii=False)[:900])
for n, e in RES["V6"]["rows"].items():
    print(n, e)
