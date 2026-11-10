"""TactileReflex 논문 표(Table III, V, VI)와 Phase 1 임계값 주장을 재계산한다.

실행: python3 analysis/table_check.py  → analysis/table_check_results.json
사용: scipy (Fisher 정확검정, Welch t, Beta 분포). 논문의 표 수치만 쓰며 원시 데이터는 없다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("table_check_results.json")
ALPHA = 0.05


def wilson(k, n, z=1.959963984540054):
    """Wilson 점수 구간 (95%)."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher(k1, n1, k2, n2):
    """두 비율의 양측 Fisher 정확검정 p."""
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(pvals):
    """Holm 단계적 보정. 입력 순서대로 보정 p를 돌려준다."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def welch(m1, s1, n1, m2, s2, n2):
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return float(t), float(p)


def exact_power(n, p1, p0, alpha=ALPHA):
    """두 팔 n명씩, 참 성공률 p1 대 p0일 때 양측 Fisher 검정의 검정력(정확 열거)."""
    power = 0.0
    for k1 in range(n + 1):
        w1 = stats.binom.pmf(k1, n, p1)
        for k0 in range(n + 1):
            if fisher(k1, n, k0, n) < alpha:
                power += w1 * stats.binom.pmf(k0, n, p0)
    return power


results = {}

# 민감도: 한 시행의 결과만 뒤집었을 때 p가 어떻게 바뀌는지 (N=5의 취약성)
flip = {
    "T1_C_1of5_to_2of5_vs_D": fisher(5, 5, 2, 5),
    "T1_A_0of5_to_1of5_vs_D": fisher(5, 5, 1, 5),
    "T2_90ml_ON_4of5_to_3of5": fisher(0, 5, 3, 5),
    "T2_90ml_ON_4of5_to_5of5": fisher(0, 5, 5, 5),
}
results["one_trial_flip_sensitivity"] = flip

# ---------------------------------------------------------------- T1 Table V 절제 실험
# 성공 횟수 / 시행 수 (N=5, 구성당). 논문 Table V 및 Sec. V-B.
abl = {"A": (0, 5), "B": (0, 5), "C": (1, 5), "D": (5, 5)}
pooled_abc = (sum(abl[c][0] for c in "ABC"), sum(abl[c][1] for c in "ABC"))  # (1, 15)
pv = {c: fisher(*abl["D"], *abl[c]) for c in "ABC"}
pv_pool = fisher(*abl["D"], *pooled_abc)
holm_abl = dict(zip("ABC", holm([pv[c] for c in "ABC"])))
results["T1_ablation"] = {
    "config": {c: {"k": abl[c][0], "n": abl[c][1], "wilson95": wilson(*abl[c])} for c in abl},
    "pooled_ABC": {"k": pooled_abc[0], "n": pooled_abc[1], "wilson95": wilson(*pooled_abc)},
    "fisher_D_vs_each_p": pv,
    "holm_D_vs_each_p": holm_abl,
    "fisher_D_vs_pooledABC_p": pv_pool,
    "note": "논문 Sec. V-B는 D 대 A/B/C 합산(1/15)을 p<0.001로 보고. 쌍별 최소 p(D 대 C)는 보정 전 0.048로, 한 구성 기준으로는 약하다.",
}

# ---------------------------------------------------------------- T2 Table VI 붓기
pour = {"45ml": {"off": (0, 5), "on": (5, 5)}, "90ml": {"off": (0, 5), "on": (4, 5)}}
p_all = fisher(0, 10, 9, 10)
p_vol = {v: fisher(*pour[v]["off"], *pour[v]["on"]) for v in pour}
holm_vol = dict(zip(pour, holm([p_vol[v] for v in pour])))
results["T2_pouring_success"] = {
    "pooled_off_0_10_vs_on_9_10_p": p_all,
    "per_volume_p": p_vol,
    "per_volume_holm_p": holm_vol,
    "wilson95": {v: {"off": wilson(*pour[v]["off"]), "on": wilson(*pour[v]["on"])} for v in pour},
    "wilson95_pooled": {"off": wilson(0, 10), "on": wilson(9, 10)},
    "note": "두 부피를 합친 10회 대 10회 검정은 부피 효과를 가린다. 부피별 검정이 더 정직하다.",
}

# ---------------------------------------------------------------- T3 Table VI 슬립 비율 (평균 ± SD, n=5)
slip = {"45ml": {"off": (42, 4, 5), "on": (35, 4, 5)}, "90ml": {"off": (63, 1, 5), "on": (42, 5, 5)}}
tt = {}
for v in slip:
    t, p = welch(*slip[v]["off"], *slip[v]["on"])
    sp = math.sqrt((slip[v]["off"][1] ** 2 + slip[v]["on"][1] ** 2) / 2)
    tt[v] = {"welch_t": t, "p": p, "cohen_d": (slip[v]["off"][0] - slip[v]["on"][0]) / sp}
holm_tt = dict(zip(tt, holm([tt[v]["p"] for v in tt])))
for v in tt:
    tt[v]["holm_p"] = holm_tt[v]
# Table VI Sy peak (2.5±0.5 → 1.8±0.4, 45 ml) — 같은 방식
t_sy, p_sy = welch(2.5, 0.5, 5, 1.8, 0.4, 5)
results["T3_slip_fraction_welch"] = {"per_volume": tt, "peak_Sy_45ml": {"welch_t": t_sy, "p": p_sy}}

# ---------------------------------------------------------------- T4 Table V 의 Fn 피크 대 Flim
# 논문 Flim = 45 (연한 컵). 보호 채널이 Fn을 Flim 아래로 "잡는다"는 주장과 비교.
fn_peaks = {"A": (63.0, 12.6, 5), "B": (70.4, 17.2, 5), "C": (70.5, 6.9, 5), "D": (67.3, 6.6, 5)}
fn_test = {}
for c, (m, s, n) in fn_peaks.items():
    t = (m - 45.0) / (s / math.sqrt(n))
    fn_test[c] = {"mean": m, "sd": s, "n": n, "t_vs_45": t, "p_two_sided": float(2 * stats.t.sf(abs(t), n - 1))}
results["T4_fn_peak_vs_flim"] = {
    "per_config": fn_test,
    "note": "Fn 피크 신호(raw 또는 EMA)는 논문에 명시되지 않았다. 피크가 Flim보다 크게 넘는 것은 보호 채널이 사후에만 작동한다는 뜻이다.",
}

# ---------------------------------------------------------------- T5 Table III 신호 분리비 (SNR)
snr = {
    "soft": {"noise_p95": 0.112, "slip_min": 0.429, "ratio": 0.429 / 0.112},
    "hard": {"noise_p95": 0.083, "slip_min": 0.204, "ratio": 0.204 / 0.083},
}
results["T5_snr_table3"] = snr

# ---------------------------------------------------------------- T6 θs 여유와 최소 슬립
theta_s = 0.20
slip_min_obs = 0.2006
results["T6_theta_s_margin"] = {
    "theta_s": theta_s,
    "min_observed_slip": slip_min_obs,
    "absolute_margin": slip_min_obs - theta_s,
    "relative_margin_pct": 100 * (slip_min_obs - theta_s) / theta_s,
    "note": "임계값은 관측된 최소 슬립에서 0.0006 아래다. 같은 데이터로 고른 값이므로 여유가 아니라 경계다. "
            "논문은 Exp. 3(같은 밀기 실험)을 교차검증이라 부르지만, 최솟값 0.2006도 그 실험에서 나왔다.",
}

# ---------------------------------------------------------------- T7 P99.9 추정의 불확실성 (정적 프레임 6,477개)
# 논문 Sec. V-A: 정적 보유 프레임 6,477개. 표본 P99.9의 초과 확률은 Beta(m, n-m+1) 분포를 따른다(순서통계량 이론).
n_static = 6477
q_rows = {}
for m in (6, 7):
    dist = stats.beta(m, n_static - m + 1)
    q_rows[f"m={m}"] = {"mean_exceed_prob": m / (n_static + 1), "upper95_exceed_prob": float(dist.ppf(0.975))}
results["T7_p999_uncertainty"] = {
    "n_static_frames": n_static,
    "rows": q_rows,
    "note": "P99.9 임계값의 초과 확률 기댓값은 약 0.1%, 95% 상한은 약 0.2%다. 논문은 Sy의 P99.9 값을 보고하지 않고 '< 0.20'이라고만 썼다.",
}

# ---------------------------------------------------------------- T8 검정력 (정확 열거, 양측 Fisher, alpha 0.05)
pow_rows = {}
for label, (p1, p0) in {"0.9_vs_0.1": (0.9, 0.1), "0.8_vs_0.2": (0.8, 0.2), "0.6_vs_0.2": (0.6, 0.2)}.items():
    pow_rows[label] = {n: round(exact_power(n, p1, p0), 3) for n in (5, 10, 15, 20, 30, 40)}
results["T8_power_fisher"] = {"per_arm_n_to_power": pow_rows,
                              "note": "논문 N=5 구성은 참 성공률 차이가 0.8 이상이 아니면 검정력이 낮다."}

# ---------------------------------------------------------------- T9 Table V ∆e 해석 점검
results["T9_effort_range"] = {
    "config_D_delta_e": {"mean": 667, "sd": 1155, "n": 5},
    "note": "∆e는 시험 중 최대 - 최소 effort(범위)다. 범위 667은 '초기 수준으로 회복'을 보이지 않는다. 종료 effort 값이 필요하다. "
            "A의 3800±0은 1200→5000(emax) 범위와 맞고, 붓기 ON의 3700은 1200→약 4900과 맞다.",
}

OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))

# ------------------------------------------------------------- 요약 출력
print("T1 D vs A/B/C p:", {c: round(pv[c], 4) for c in pv}, "Holm:", {c: round(holm_abl[c], 4) for c in holm_abl})
print("T1 D vs pooled(1/15) p:", round(pv_pool, 5))
print("T2 pooled 0/10 vs 9/10 p:", round(p_all, 5), "per-volume:", {v: round(p_vol[v], 4) for v in p_vol},
      "Holm:", {v: round(holm_vol[v], 4) for v in holm_vol})
print("T3 slip% Welch:", {v: (round(tt[v]["welch_t"], 2), round(tt[v]["p"], 4), round(tt[v]["holm_p"], 4)) for v in tt})
print("T3 peak Sy 45ml Welch:", round(t_sy, 2), round(p_sy, 4))
print("T4 D Fn peak vs Flim 45: t=", round(fn_test["D"]["t_vs_45"], 2), "p=", round(fn_test["D"]["p_two_sided"], 6))
print("T5 SNR soft/hard:", round(snr["soft"]["ratio"], 2), round(snr["hard"]["ratio"], 2))
print("T6 margin:", round(results["T6_theta_s_margin"]["absolute_margin"], 4), "=",
      round(results["T6_theta_s_margin"]["relative_margin_pct"], 2), "%")
print("T7 P99.9 exceed mean/upper95:", {k: (round(v["mean_exceed_prob"], 5), round(v["upper95_exceed_prob"], 5)) for k, v in q_rows.items()})
print("T8 power:", pow_rows)
print("saved:", OUT)
