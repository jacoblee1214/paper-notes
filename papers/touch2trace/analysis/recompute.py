"""Touch2Trace(arXiv:2609.15921 v1) 보고 수치 재계산. 결과는 recompute_results.json.

실행: python3 analysis/recompute.py   (numpy, scipy 필요, 수 초)

논문 표(표 1, 2, 3, 5, 6, 9, 10)의 값을 아래 TABLES에 옮겨 놓고 다음을 확인한다.
  V1  표 사이 같은 조건의 값이 일치하는지
  V2  파생 비율(2.75배, 7배, 37배, 6.2배 등)
  V3  성공률(SR) 퍼센트가 어떤 시행 수에서 정수 횟수로 나올 수 있는지 (n = 30 가정이 맞는지)
  V4  평균 거리(Dist)가 SR@10, SR@20과 모순되지 않는지 (가능한 범위)
  V5  SR의 Wilson 95% 구간 (n = 30 가정)
  V6  SR 비교의 Fisher 정확 검정 (n = 30 가정)과 시행 수 민감도
  V7  거리 평균의 Welch 검정 (표의 ±를 시드 평균 3개의 표준편차로 읽음)
  V8  하드웨어와 파라미터 수 산술
  V9  제어 주파수와 프레임 수에 따른 시간 창
  V10 Table 7, 8, 10 산술
  V11 부록 D 비교의 시행 수 제약
  K1  "Smooth" 지표의 샘플링 주기 의존 (합성 신호)
  K2  3 시드 설계의 검출력 (몬테카를로)

가정은 문장 안에 밝힌다. 논문에 없는 시행 수는 "n = 30 가정"(3 시드 × 10 롤아웃, 5.1절)이다.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "recompute_results.json"
RES = {}


def rnd(x):  # 반올림(0.5는 올림)
    return int(math.floor(x + 0.5))


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# (이름, Dist 평균, Dist ± , SR@10 %, SR@20 %, Smooth, 출처)
TABLES = {
    "T1": [
        ("TF-GMM Proprio 60Hz", 0.2, 0.2, 0, 0, 0.011),
        ("TF-GMM Proprio+Tactile 60Hz", 20.1, 4.6, 93, 57, 0.004),
        ("DP Proprio 60Hz", 0.5, 0.4, 0, 0, 0.002),
        ("DP Proprio+Tactile 60Hz", 1.3, 0.4, 0, 0, 0.007),
        ("DP Proprio 30Hz", 0.1, 0.2, 0, 0, 0.002),
        ("DP Proprio+Tactile 30Hz", 15.5, 2.7, 100, 0, 0.028),
    ],
    "T2": [
        ("USB-0 ring (train)", 25.5, 7.2, 96, 73, 0.026),
        ("USB-1 straight", 17.8, 8.1, 73, 38, 0.008),
        ("USB-2 straight", 10.4, 4.9, 47, 13, 0.005),
        ("Ethernet straight", 20.1, 4.6, 93, 57, 0.004),
        ("Ethernet ring", 24.7, 7.2, 97, 73, 0.005),
    ],
    "T3": [
        ("Baseline frozen 10.1min", 20.1, 4.6, 93, 57, 0.004),
        ("Finetuned", 4.7, 1.6, 0, 0, 0.005),
        ("Random-init", 0.2, 0.1, 0, 0, 0.009),
        ("3.5 min", 6.6, 3.5, 17, 7, 0.002),
        ("6.7 min", 8.1, 1.5, 37, 0, 0.004),
    ],
    "T5": [
        ("Proprio only", 0.2, 0.2, 0, 0, None),
        ("Compact tactile 16-d", 7.2, 3.1, 30, 0, None),
        ("Dense tactile frozen ViT-MAE", 20.1, 4.6, 93, 57, None),
    ],
    "T9": [
        ("Default 60Hz/15/32x32", 20.1, 4.6, 93, 57, 0.004),
        ("15 Hz", 0.5, 0.2, 0, 0, 0.002),
        ("30 Hz", 5.7, 2.2, 23, 0, 0.004),
        ("frame 5", 5.3, 3.5, 11, 0, 0.150),
        ("frame 10", 8.0, 6.0, 28, 17, 0.008),
        ("frame 20", 14.5, 10.1, 50, 37, 0.003),
        ("8x8", 6.2, 5.7, 17, 7, 0.007),
        ("16x16", 11.2, 10.3, 41, 24, 0.012),
    ],
}


# ---------------------------------------------------------------- V1
def v1():
    base = (20.1, 4.6, 93, 57, 0.004)
    same = []
    for t, rows in TABLES.items():
        for r in rows:
            if r[0] in ("TF-GMM Proprio+Tactile 60Hz", "Ethernet straight", "Baseline frozen 10.1min",
                        "Dense tactile frozen ViT-MAE", "Default 60Hz/15/32x32"):
                same.append((t, r[0], r[1:3] + r[3:5] + (r[5],)))
    ok = all(tuple(s[2][:4]) == base[:4] for s in same)
    p_proprio = [r for t, rows in TABLES.items() for r in rows if r[0] in ("TF-GMM Proprio 60Hz", "Proprio only")]
    RES["V1"] = {"default_row_in_5_tables_identical": ok, "n_tables": len(same),
                 "proprio_rows_identical": all(r[1:5] == (0.2, 0.2, 0, 0) for r in p_proprio),
                 "table2_vs_table8_dist_match": True,
                 "ethernet_straight_smooth_T2_vs_T1": (0.004, 0.004)}


# ---------------------------------------------------------------- V2
def v2():
    d = {
        "smooth_ratio_tf_proprio_vs_tactile": 0.011 / 0.004,           # 논문 2.75배
        "smooth_ratio_dp30_vs_tf60": 0.028 / 0.004,                     # 논문 7배
        "smooth_ratio_frame5_vs_default": 0.150 / 0.004,                # 논문 37배
        "taxel_density_tacv5_per_cm2": 1 / (0.07 * 0.092),              # 0.7 x 0.92 mm
        "taxel_density_flexitac_per_cm2": 1 / (0.2 * 0.2),              # 2 mm pitch (pitch를 정사각으로 가정)
        "density_ratio": (1 / (0.07 * 0.092)) / (1 / (0.2 * 0.2)),
        "dist_ratio_tactile_proprio": 20.1 / 0.2,
        "fixed_dist_cap_note": "Fig. 10: 30 s 평균 약 0.67 cm/s",
        "params_dp_over_tf": 74.4 / 4.1,
        "mem_dp_over_tf": 602 / 37,
        "latency_headroom": (1000 / 3.4) / 60,
        "latency_hz": 1000 / 3.4,
        "dp_latency_hz": 1000 / 32,
        "encoder_plus_backbone_ms": 2.0 + 1.4,
        "sense_to_act_sum_ms": 4.2 + 4.3 + 3.3,
        "ratio_dp_latency_over_cycle60": 32 / (1000 / 60),
    }
    RES["V2"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


# ---------------------------------------------------------------- V3
def feasible_pooled(p10, p20, nmax=30, nmin=15):
    out = []
    for n in range(nmin, nmax + 1):
        ks10 = [k for k in range(n + 1) if rnd(100 * k / n) == p10]
        ks20 = [k for k in range(n + 1) if rnd(100 * k / n) == p20]
        if any(a >= b for a in ks10 for b in ks20):
            out.append(n)
    return out


def feasible_seed_mean(p10, p20):
    """3 시드 각각 n_i ∈ [6, 10]회, SR을 시드별 비율의 평균으로 보고했다고 가정. 최소 제외 횟수와 예."""
    best = None
    for ns in itertools.combinations_with_replacement(range(6, 11), 3):
        loss = 30 - sum(ns)
        if best is not None and loss >= best[0]:
            continue
        found = None
        ranges = [range(n + 1) for n in ns]
        for k10 in itertools.product(*ranges):
            if rnd(100 * sum(k / n for k, n in zip(k10, ns)) / 3) != p10:
                continue
            for k20 in itertools.product(*[range(k + 1) for k in k10]):
                if rnd(100 * sum(k / n for k, n in zip(k20, ns)) / 3) == p20:
                    found = (k10, k20)
                    break
            if found:
                break
        if found:
            best = (loss, ns, found)
    return best


def v3():
    rows = []
    for t, tab in TABLES.items():
        for r in tab:
            name, p10, p20 = r[0], r[3], r[4]
            ns = feasible_pooled(p10, p20)
            pooled30 = 30 in ns
            minexc_pooled = 30 - max(ns) if ns else None
            sm = feasible_seed_mean(p10, p20)
            rows.append({"table": t, "row": name, "SR10": p10, "SR20": p20,
                         "feasible_at_n30_pooled": pooled30,
                         "min_excluded_pooled": minexc_pooled,
                         "feasible_n_pooled(15-30)": ns,
                         "min_excluded_seed_mean": None if sm is None else sm[0],
                         "seed_mean_example": None if sm is None else {"n_per_seed": sm[1], "k10": sm[2][0], "k20": sm[2][1]}})
    # 중복 행(기본 설정) 제거
    seen, uniq = set(), []
    for r in rows:
        key = (r["SR10"], r["SR20"], r["row"] if r["row"].startswith(("Default", "Baseline", "TF-GMM Proprio+", "Ethernet straight", "Dense")) is False else "default")
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    infeasible = [r for r in uniq if not r["feasible_at_n30_pooled"]]
    RES["V3"] = {"rows": uniq,
                 "infeasible_at_n30": [(r["table"], r["row"], r["SR10"], r["SR20"], r["min_excluded_pooled"]) for r in infeasible],
                 "sum_min_excluded_pooled_over_infeasible_rows": sum(r["min_excluded_pooled"] or 0 for r in infeasible),
                 "sum_min_excluded_seed_mean_over_infeasible_rows": sum(r["min_excluded_seed_mean"] or 0 for r in infeasible),
                 "text_claim_excluded_total": 6}


# ---------------------------------------------------------------- V4
def v4():
    rows = []
    for t, tab in TABLES.items():
        for r in tab:
            name, dist, sd, p10, p20 = r[:5]
            f10, f20 = p10 / 100, p20 / 100
            lo = 10 * (f10 - f20) + 20 * f20          # 10-20 구간 하한 10, 20 이상 하한 20
            hi = 20 * (f10 - f20) + 10 * (1 - f10) if f20 == 0 else None  # SR@20=0이면 상한 있음
            ok = dist >= lo - 1e-9 and (hi is None or dist <= hi + 1e-9)
            rows.append({"table": t, "row": name, "dist": dist, "lower": round(lo, 2),
                         "upper": None if hi is None else round(hi, 2), "consistent": ok})
    RES["V4"] = {"rows": rows, "all_consistent": all(r["consistent"] for r in rows)}


# ---------------------------------------------------------------- V5
def v5():
    out = {}
    for name, k, n in [("TF-GMM tactile SR@10 28/30", 28, 30), ("TF-GMM tactile SR@20 17/30", 17, 30),
                       ("Proprio SR@10 0/30", 0, 30), ("DP30 SR@10 30/30", 30, 30),
                       ("Compact SR@10 9/30", 9, 30), ("USB-2 SR@10 14/30", 14, 30),
                       ("frame15 vs 5: frame5 SR@10 3/30", 3, 30), ("30Hz SR@10 7/30", 7, 30),
                       ("8x8 SR@10 5/30", 5, 30), ("16x16 SR@10 12/30", 12, 30), ("frame20 SR@10 15/30", 15, 30)]:
        lo, hi = wilson(k, n)
        out[name] = [round(100 * k / n, 1), round(lo, 1), round(hi, 1)]
    RES["V5"] = out


# ---------------------------------------------------------------- V6
def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def v6():
    comps = [
        ("SR@10 TF tactile 28/30 vs proprio 0/30", 28, 0), ("SR@10 TF tactile 28 vs compact 9", 28, 9),
        ("SR@20 TF tactile 17 vs compact 0", 17, 0),
        ("SR@10 TF 60Hz 28 vs 30Hz 7", 28, 7), ("SR@10 60Hz 28 vs 15Hz 0", 28, 0),
        ("SR@10 frame15 28 vs frame20 15", 28, 15), ("SR@10 frame15 28 vs frame10 8", 28, 8),
        ("SR@10 frame15 28 vs frame5 3", 28, 3),
        ("SR@10 32x32 28 vs 16x16 12", 28, 12), ("SR@10 32x32 28 vs 8x8 5", 28, 5),
        ("SR@10 frozen 28 vs finetuned 0", 28, 0), ("SR@10 10.1min 28 vs 6.7min 11", 28, 11),
        ("SR@10 6.7min 11 vs 3.5min 5", 11, 5),
        ("SR@10 Ethernet straight 28 vs USB-1 22", 28, 22), ("SR@10 Ethernet straight 28 vs USB-2 14", 28, 14),
        ("SR@10 Ethernet ring 29 vs straight 28", 29, 28), ("SR@20 Ethernet ring 22 vs straight 17", 22, 17),
        ("SR@20 USB-0 ring 22 vs Ethernet ring 22", 22, 22),
        ("SR@10 DP30 30 vs TF-GMM60 28", 30, 28),
        ("SR@20 TF 17 vs DP30 0", 17, 0),
    ]
    out = {}
    for name, k1, k2 in comps:
        out[name] = float(f"{fisher(k1, 30, k2, 30):.3g}")
    # 정지 조건 확인: 시행 수 민감도 (n = 24 ~ 30에서 가장 약한 비교의 p)
    sens = {}
    for name, (k1f, k2f) in {"frame15 vs frame20 SR@10": (0.933, 0.50), "32x32 vs 16x16 SR@10": (0.933, 0.41),
                              "Ethernet ring vs straight SR@20": (0.733, 0.567)}.items():
        sens[name] = {n: float(f"{fisher(rnd(k1f * n), n, rnd(k2f * n), n):.3g}") for n in (24, 27, 30)}
    # Holm 보정 (표 9 중 기본 대비 6개)
    pv = [out[k] for k in ["SR@10 TF 60Hz 28 vs 30Hz 7", "SR@10 60Hz 28 vs 15Hz 0", "SR@10 frame15 28 vs frame20 15",
                           "SR@10 frame15 28 vs frame10 8", "SR@10 frame15 28 vs frame5 3",
                           "SR@10 32x32 28 vs 16x16 12", "SR@10 32x32 28 vs 8x8 5"]]
    order = np.argsort(pv)
    holm = [None] * len(pv)
    m = len(pv)
    prev = 0
    for rank, idx in enumerate(order):
        val = min(1, (m - rank) * pv[idx])
        prev = max(prev, val)
        holm[idx] = prev
    out["holm_table9_default_vs_7_cells(SR@10)"] = [float(f"{x:.3g}") for x in holm]
    RES["V6"] = {"fisher": out, "n_sensitivity": sens}


# ---------------------------------------------------------------- V7
def welch(m1, s1, m2, s2, n=3, scale=1.0):
    r = stats.ttest_ind_from_stats(m1, s1 * scale, n, m2, s2 * scale, n, equal_var=False)
    return float(r.pvalue)


def v7():
    pairs = [
        ("Tactile vs proprio (T1)", (20.1, 4.6), (0.2, 0.2)),
        ("Dense vs compact (T5)", (20.1, 4.6), (7.2, 3.1)),
        ("60 Hz vs 30 Hz (T9)", (20.1, 4.6), (5.7, 2.2)),
        ("frame15 vs frame20 (T9)", (20.1, 4.6), (14.5, 10.1)),
        ("frame15 vs frame10 (T9)", (20.1, 4.6), (8.0, 6.0)),
        ("frame15 vs frame5 (T9)", (20.1, 4.6), (5.3, 3.5)),
        ("32x32 vs 16x16 (T9)", (20.1, 4.6), (11.2, 10.3)),
        ("32x32 vs 8x8 (T9)", (20.1, 4.6), (6.2, 5.7)),
        ("frozen vs finetuned (T3)", (20.1, 4.6), (4.7, 1.6)),
        ("10.1 min vs 6.7 min (T3)", (20.1, 4.6), (8.1, 1.5)),
        ("6.7 min vs 3.5 min (T3)", (8.1, 1.5), (6.6, 3.5)),
        ("Ethernet straight vs USB-1 (T2)", (20.1, 4.6), (17.8, 8.1)),
        ("Ethernet straight vs USB-2 (T2)", (20.1, 4.6), (10.4, 4.9)),
        ("Ethernet ring vs straight (T2)", (24.7, 7.2), (20.1, 4.6)),
        ("USB-0 ring vs Ethernet ring (T2)", (25.5, 7.2), (24.7, 7.2)),
        ("TF-GMM60 vs DP30 tactile (T1)", (20.1, 4.6), (15.5, 2.7)),
    ]
    out = {}
    for name, a, b in pairs:
        out[name] = {"p_sd_as_sample": round(welch(*a, *b), 4),
                     "p_sd_as_population(x sqrt(3/2))": round(welch(*a, *b, scale=math.sqrt(1.5)), 4)}
    RES["V7"] = out


# ---------------------------------------------------------------- V8
def v8():
    def vit_params(d, layers, mlp=4, patch=16, tokens=64, in_ch=1):
        per = 4 * d * d + 4 * d + 2 * mlp * d * d + (mlp + 1) * d + 4 * d  # attn qkv+proj, mlp, 2 LN
        return layers * per + patch * in_ch * d + d + (tokens + 1) * d + d + 2 * d
    enc = vit_params(128, 8)
    d = 256
    per = 4 * d * d + 4 * d + 2 * 4 * d * d + 5 * d + 4 * d
    gpt = 3 * per + 264 * d + d + 15 * d + 2 * d
    head = d * (5 + 5 * 8 + 5 * 8) + (5 + 5 * 8 + 5 * 8)  # 가중치 5, 평균 5x8, 표준편차 5x8
    RES["V8"] = {"encoder_params_est": enc, "paper_encoder_params": 1.6e6,
                 "policy_backbone_params_est": gpt + head, "paper_policy_params": 2.5e6,
                 "sum_est": enc + gpt + head, "paper_total": 4.1e6,
                 "note": "MLP 비율 4, 편향 포함, 디코더 제외(사전학습 뒤 버림)로 추정. 논문에 층 세부가 없어 근사",
                 "steps_5355_over_3x30s_at_60Hz": 5355 / (3 * 30 * 60),
                 "mean_period_ms_to_hz": 1000 / 16.65, "drift_ms_per_cycle": 16.65 - 1000 / 60,
                 "drift_pct": 100 * abs(16.65 - 1000 / 60) / (1000 / 60),
                 "taxel_array": 32 * 32, "tactile_frames_in_window": 2 * 15}


# ---------------------------------------------------------------- V9
def v9():
    out = {}
    for hz in (60, 30, 15):
        out[f"{hz}Hz frame15"] = 15 * 1000 / hz
    for f in (5, 10, 15, 20):
        out[f"60Hz frame{f}"] = f * 1000 / 60
    out["fixed_250ms_frames_at_30Hz"] = 0.25 * 30
    out["fixed_250ms_frames_at_15Hz"] = 0.25 * 15
    out["dp_obs_horizon2_ms_at_60Hz"] = 2 * 1000 / 60
    out["dp_obs_horizon2_ms_at_30Hz"] = 2 * 1000 / 30
    out["dp_chunk_8_exec_ms_at_30Hz"] = 8 * 1000 / 30
    out["dp_chunk_8_exec_ms_at_60Hz"] = 8 * 1000 / 60
    RES["V9"] = {k: round(v, 2) for k, v in out.items()}


# ---------------------------------------------------------------- V10
def v10():
    t7 = {"tactile": {"thumb": (11352, 53.7), "index": (43923, 100.1), "middle": (9958, 38.0)},
          "proprio": {"thumb": (1203, 6.9), "index": (8395, 11.2), "middle": (2734, 14.7)}}
    t8 = {"USB-0 ring": (25.5, 59138, 0.026), "USB-1 straight": (17.8, 49583, 0.008), "USB-2 straight": (10.4, 44344, 0.005),
          "Ethernet straight": (20.1, 39780, 0.004), "Ethernet ring": (24.7, 31988, 0.005)}
    isj = [v[1] for v in t8.values()]
    dist = [v[0] for v in t8.values()]
    sm = [v[2] for v in t8.values()]
    r_isj_dist = float(np.corrcoef(isj, dist)[0, 1])
    r_isj_smooth = float(np.corrcoef(isj, sm)[0, 1])
    RES["V10"] = {
        "t7_index_speed_ratio": 100.1 / 11.2, "t7_index_isj_ratio": 43923 / 8395,
        "t7_jerk_per_speed_tactile": 43923 / 100.1, "t7_jerk_per_speed_proprio": 8395 / 11.2,
        "t7_trial_isj_sum_tactile": sum(v[0] for v in t7["tactile"].values()),
        "t8_ethernet_straight_isj_seed_avg": 39780,
        "t7_trial_over_t8_seed_avg": sum(v[0] for v in t7["tactile"].values()) / 39780,
        "t8_isj_range": (min(isj), max(isj)),
        "t8_corr_isj_vs_dist": r_isj_dist, "t8_corr_isj_vs_smooth": r_isj_smooth,
        "t8_usb0_over_ethernet_ring_isj": 59138 / 31988, "t2_usb0_over_ethernet_ring_smooth": 0.026 / 0.005,
        "t10_smoothness_ratio": 0.000196 / 0.000234, "t10_pred_over_gt_pct_smoother": 100 * (1 - 0.000196 / 0.000234),
        "online_smooth_over_offline_pred": 0.004 / 0.000196, "online_smooth_over_offline_gt": 0.004 / 0.000234,
        "demo_minutes": {"16_demos_total": 14.0, "mean_per_demo_s": 14 * 60 / 16,
                         "12_train_mean_s": 10.1 * 60 / 12, "val_4_minutes_if_14_total": 14 - 10.1},
        "subset_minutes_frac": [3.5 / 10.1, 6.7 / 10.1],
        "approx_demos_in_subsets": [3.5 / 10.1 * 12, 6.7 / 10.1 * 12],
        "appD_4demos_35.4min_per_demo_min": 35.4 / 4,
    }


# ---------------------------------------------------------------- V11
def v11():
    out = {}
    for name, p10, p20 in [("Vision only", 67, 33), ("Vision+dense tactile", 100, 50)]:
        ns = [n for n in range(1, 31) if any(rnd(100 * k / n) == p10 for k in range(n + 1))
              and any(rnd(100 * k / n) == p20 for k in range(n + 1))]
        out[name] = {"SR10": p10, "SR20": p20, "feasible_n_1_30": ns}
    both = sorted(set(out["Vision only"]["feasible_n_1_30"]) & set(out["Vision+dense tactile"]["feasible_n_1_30"]))
    out["common_feasible_n"] = both
    lo, hi = wilson(4, 6)
    out["wilson_4of6"] = [round(lo, 1), round(hi, 1)]
    lo, hi = wilson(6, 6)
    out["wilson_6of6"] = [round(lo, 1), round(hi, 1)]
    out["fisher_6_of_6_vs_4_of_6"] = round(fisher(6, 6, 4, 6), 4)
    out["fisher_9_of_9_vs_6_of_9"] = round(fisher(9, 9, 6, 9), 4)
    RES["V11"] = out


# ---------------------------------------------------------------- K1
def k1():
    """같은 연속 신호를 60 Hz와 30 Hz로 표본화했을 때 mean||a_t - a_{t-1}||(^2)가 얼마나 달라지는지."""
    rng = np.random.default_rng(0)
    out = {}
    for label, power in (("norm", 1), ("squared_norm", 2)):
        vals = {}
        for hz in (60, 30, 15):
            acc = []
            for _ in range(200):
                f = rng.uniform(1.0, 3.0)  # 핀치-컬 진동 주파수(Hz), 합성
                amp = rng.uniform(0.2, 0.5, size=8)
                ph = rng.uniform(0, 2 * np.pi, size=8)
                t = np.arange(0, 30, 1 / hz)
                a = amp[None, :] * np.sin(2 * np.pi * f * t[:, None] + ph[None, :])
                d = np.linalg.norm(np.diff(a, axis=0), axis=1) ** power
                acc.append(d.mean())
            vals[hz] = float(np.mean(acc))
        out[label] = {"mean_by_hz": vals, "ratio_30_over_60": vals[30] / vals[60], "ratio_15_over_60": vals[15] / vals[60]}
    RES["K1"] = out
    RES["K1"]["paper_ratio_dp30_over_tf60"] = 0.028 / 0.004
    RES["K1"]["adjusted_ratio_if_squared"] = 0.028 / 0.004 / out["squared_norm"]["ratio_30_over_60"]
    RES["K1"]["adjusted_ratio_if_norm"] = 0.028 / 0.004 / out["norm"]["ratio_30_over_60"]


# ---------------------------------------------------------------- K2
def k2():
    """3 시드 설계 (표 9 SD 수준)에서 평균 거리 차이를 Welch p<0.05로 잡을 확률."""
    rng = np.random.default_rng(1)
    cases = {"frame15 vs frame20 (diff 5.6, sd 4.6 / 10.1)": (20.1, 4.6, 14.5, 10.1),
             "32x32 vs 16x16 (diff 8.9, sd 4.6 / 10.3)": (20.1, 4.6, 11.2, 10.3),
             "ring vs straight (diff 4.6, sd 7.2 / 4.6)": (24.7, 7.2, 20.1, 4.6),
             "60Hz vs 30Hz (diff 14.4, sd 4.6 / 2.2)": (20.1, 4.6, 5.7, 2.2)}
    out = {}
    for name, (m1, s1, m2, s2) in cases.items():
        hits = 0
        N = 4000
        for _ in range(N):
            a = rng.normal(m1, s1, 3)
            b = rng.normal(m2, s2, 3)
            if stats.ttest_ind(a, b, equal_var=False).pvalue < 0.05:
                hits += 1
        out[name] = hits / N
    # 필요한 차이: 같은 SD 가정에서 Welch(3 대 3) 유의에 필요한 평균 차이
    need = {}
    for s in (2.2, 4.6, 7.2, 10.1):
        t_crit = stats.t.ppf(0.975, 4)
        need[f"sd={s}"] = round(t_crit * s * math.sqrt(2 / 3), 1)
    out["needed_diff_cm_for_p05_equal_sd_n3"] = need
    # SR 쪽: n = 30 대 30에서 80% 검출력에 필요한 차이 (기준 93%)
    from itertools import product
    def power(p1, p2, n=30, N=3000):
        hits = 0
        for _ in range(N):
            k1_ = rng.binomial(n, p1)
            k2_ = rng.binomial(n, p2)
            if stats.fisher_exact([[k1_, n - k1_], [k2_, n - k2_]])[1] < 0.05:
                hits += 1
        return hits / N
    out["sr_power_n30"] = {f"{p1:.2f} vs {p2:.2f}": power(p1, p2, N=1500) for p1, p2 in [(0.93, 0.80), (0.93, 0.70), (0.93, 0.50), (0.57, 0.38), (0.97, 0.93)]}
    RES["K2"] = out


def main():
    for f in (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, k1, k2):
        f()
    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({k: RES[k] for k in ("V3",)}, ensure_ascii=False, indent=1)[:3500])


if __name__ == "__main__":
    main()
