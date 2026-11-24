"""MEM 보고 수치 재계산 (V 항목). 단일 프로세스, CPU, 수 초.

논문에는 표가 없고 결과가 막대 그림(그림 3, 6, 7, 8, 9, 10)에만 있다. 막대 높이와 오차 막대 끝은
analysis/extract_figures.py 가 PDF 벡터에서 읽은 값(analysis/fig_readings.json)이다. 좌표 정밀도는 높지만 원자료가 아니므로
'그림 판독 값'으로 부른다. 오차 막대는 본문(IV-A)이 '평균 ± 표준오차'라고 쓴 것을 따른다. 시행 수는 논문에 거의 없어서
막대가 정수 비율이 되는 값에서 가정한다(문장마다 가정이라고 밝힘).
실행: python3 -I analysis/extract_figures.py; python3 -I analysis/recompute.py  → analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

from scipy import stats

HERE = Path(__file__).parent
R = json.load(open(HERE / "fig_readings.json"))
OUT = HERE / "recompute_results.json"

COL = {  # 채움색 → 이름
    "fig6": {(0.78, 0.486, 0.408): "no_memory", (0.612, 0.745, 0.439): "only_video", (0.502, 0.682, 0.792): "naive_text_video",
             (0.718, 0.784, 0.788): "only_text", (1.0, 0.824, 0.235): "mem"},
    "fig8": {(0.78, 0.486, 0.408): "no_memory", (0.718, 0.784, 0.788): "pool", (0.612, 0.745, 0.439): "proprio",
             (1.0, 0.824, 0.235): "mem"},
    "fig9": {(0.612, 0.745, 0.439): "post_train_only", (1.0, 0.824, 0.235): "mem"},
    "fig10": {(0.612, 0.745, 0.439): "pi06", (1.0, 0.824, 0.235): "mem"},
    "fig7": {(0.612, 0.745, 0.439): "no_memory", (1.0, 0.824, 0.235): "mem"},
}
PANEL_NAMES = {
    "fig6": ["recipe", "clean_kitchen", "average"],
    "fig8": ["swap_3_mugs", "find_object", "unpack_groceries", "scoop_coffee", "grilled_cheese", "window_cleaning", "average"],
    "fig9": ["swap_3_mugs", "find_object", "unpack_groceries", "scoop_coffee", "grilled_cheese", "window_cleaning", "average"],
    "fig10": ["table_bussing", "shirt_folding", "clean_up_counter", "make_bed", "dishes_to_sink", "batch_folding", "box_building", "average"],
}


def table(fig):
    """{패널: {조건: (평균, 표준오차)}} (비율 0–1)"""
    out = {}
    panels = sorted({tuple(b["panel_x"]) for b in R[fig]["bars"]})
    for pname, px in zip(PANEL_NAMES[fig], panels):
        d = {}
        for b in R[fig]["bars"]:
            if tuple(b["panel_x"]) != px:
                continue
            key = COL[fig][tuple(b["fill"])]
            lo, hi = b["err_lo_hi"]
            d[key] = (b["value"] / 100, (hi - lo) / 200)
        out[pname] = d
    return out


def zp(m1, s1, m2, s2):
    """두 평균의 차이에 대한 정규 근사 양측 p (표준오차만 사용, 시행 수 가정 불필요)"""
    z = (m1 - m2) / math.sqrt(s1 * s1 + s2 * s2)
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def welch_p(m1, s1, n1, m2, s2, n2):
    se2 = s1 * s1 + s2 * s2
    df = se2 ** 2 / (s1 ** 4 / (n1 - 1) + s2 ** 4 / (n2 - 1))
    t = (m1 - m2) / math.sqrt(se2)
    return t, df, 2 * (1 - stats.t.cdf(abs(t), df))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def granularity(values, nmax=200, tol=0.0005):
    """모든 값(0–1)이 k/N 꼴(정수 k)이 되는 가장 작은 N들. 좌표 정밀도 때문에 허용 오차는 0.05%p."""
    ok = []
    for N in range(2, nmax + 1):
        if all(abs(v * N - round(v * N)) < tol * N * 1.0 + 1e-9 and abs(v - round(v * N) / N) < tol for v in values):
            ok.append(N)
    return ok


res = {}
T = {f: table(f) for f in ("fig6", "fig8", "fig9", "fig10")}
for pn in ("fig7_top", "fig7_bottom"):
    pass

# ---------------- V1: 그림 3 지연 ----------------
m3 = [m for m in R["fig3"]["marks"] if m["x"] < 300]
frames = [1, 2, 4, 6, 8, 12, 16]
wo = [m["sec"] for m in m3 if m["series"] == "without"]
wi = [m["sec"] for m in m3 if m["series"] == "with"]
import numpy as np
fw = np.polyfit(frames, wi, 1)
fo2 = np.polyfit(frames, wo, 2)
res["V1_fig3_latency"] = {
    "frames": frames, "without_s": wo, "with_s": wi,
    "ratio_without_over_with": [round(a / b, 2) for a, b in zip(wo, wi)],
    "first_frames_over_300ms_without": next(f for f, v in zip(frames, wo) if v > 0.3),
    "with_linear_fit_ms_per_frame": round(fw[0] * 1000, 2), "with_linear_fit_intercept_ms": round(fw[1] * 1000, 1),
    "with_linear_r2": round(float(1 - np.sum((np.array(wi) - np.polyval(fw, frames)) ** 2) / np.sum((np.array(wi) - np.mean(wi)) ** 2)), 4),
    "without_quad_coef": [float(x) for x in fo2],
    "with_extrapolated_19frames_s": round(float(np.polyval(fw, 19)), 3), "with_extrapolated_18frames_s": round(float(np.polyval(fw, 18)), 3),
    "without_extrapolated_19frames_s": round(float(np.polyval(fo2, 19)), 2),
    "with_at_16_over_at_1": round(wi[-1] / wi[0], 2), "without_at_16_over_at_1": round(wo[-1] / wo[0], 1),
}

# ---------------- V2: 그림 7 인-컨텍스트 적응 (성공률, Fisher) ----------------
f7 = {}
for key, pn in (("pick_up_chopstick", "fig7_top"), ("open_fridge", "fig7_bottom")):
    d = {COL["fig7"][tuple(b["fill"])]: b for b in R[pn]["bars"]}
    row = {}
    for cond, b in d.items():
        p = b["value"] / 100
        lo, hi = b["err_lo_hi"]
        se = (hi - lo) / 200
        cands = []
        for n in range(4, 121):
            k = round(p * n)
            if abs(k / n - p) < 0.0006:
                se_s = math.sqrt(p * (1 - p) / (n - 1)) if 0 < p < 1 else 0
                se_p = math.sqrt(p * (1 - p) / n) if 0 < p < 1 else 0
                cands.append({"k": k, "n": n, "se_sample": round(se_s, 4), "se_pop": round(se_p, 4)})
        row[cond] = {"p": p, "se_read": round(se, 4), "cands_n_le_120": cands[:8],
                     "best_by_se": min(cands, key=lambda c: abs(c["se_sample"] - se)) if cands else None}
    f7[key] = row
# 가정한 시행 수 (오차 막대와 정수 비율에서): 문 열기 21회씩, 젓가락은 읽은 값에서 따로 정함 (아래 pick 결과 참조)
res["V2_fig7_raw"] = f7

# ---------------- V3: 그림 6 ----------------
v3 = {}
for panel, d in T["fig6"].items():
    m, s = d["mem"]
    row = {}
    for k, (m2, s2) in d.items():
        if k == "mem":
            continue
        z, p = zp(m, s, m2, s2)
        row[k] = {"diff_pp": round((m - m2) * 100, 1), "z": round(z, 2), "p_z": float(f"{p:.2g}")}
    v3[panel] = row
res["V3_fig6_mem_vs_ablations"] = v3
# 그림 6 점수 평균이 두 패널 평균과 같은가 (가중치 역산)
w = {}
for k in T["fig6"]["average"]:
    a = T["fig6"]["recipe"][k][0]
    c = T["fig6"]["clean_kitchen"][k][0]
    av = T["fig6"]["average"][k][0]
    w[k] = {"recipe": round(a * 100, 2), "clean": round(c * 100, 2), "avg_read": round(av * 100, 2),
            "unweighted_mean": round((a + c) / 2 * 100, 2), "implied_weight_clean": round((a - av) / (a - c), 3) if a != c else None}
res["V4_fig6_average_weight"] = w
ws = [x["implied_weight_clean"] for x in w.values()]
res["V4_fig6_average_weight_summary"] = {"min": round(min(ws), 3), "max": round(max(ws), 3),
                                          "unweighted_would_be": 0.5, "rollout_weighted_5recipes_x10_vs_10": round(10 / 60, 3),
                                          "note": "평균 막대의 가중치는 본문에 없음"}
# 막대 굵기 격자: 클린 키친 값이 1/80 격자인가
cl = [v[0] for v in T["fig6"]["clean_kitchen"].values()]
res["V4_clean_kitchen_granularity_N"] = granularity(cl)[:6]
# 표준오차로부터 롤아웃당 SD와 상한 대비
def sd_ratio(m, se, n):
    sd = se * math.sqrt(n)
    sdmax = math.sqrt(m * (1 - m) * n / (n - 1))
    return round(sd, 3), round(sdmax, 3), round(sd / sdmax, 2)
v5 = {}
for panel, n_list in (("recipe", (5, 10, 50)), ("clean_kitchen", (10,))):
    for k, (m, se) in T["fig6"][panel].items():
        v5[f"{panel}:{k}"] = {f"n={n}": sd_ratio(m, se, n) for n in n_list}
res["V5_fig6_se_vs_max_sd"] = v5

# ---------------- V6: 그림 8 ----------------
v6 = {}
for panel, d in T["fig8"].items():
    m, s = d["mem"]
    row = {}
    for k, (m2, s2) in d.items():
        if k == "mem":
            continue
        z, p = zp(m, s, m2, s2)
        row[k] = {"mem": round(m * 100, 1), "other": round(m2 * 100, 1), "diff_pp": round((m - m2) * 100, 1), "z": round(z, 2), "p_z": float(f"{p:.2g}")}
    v6[panel] = row
res["V6_fig8_mem_vs_baselines"] = v6
# 그림 8 격자: 패널별 정수 N
v7 = {}
for panel, d in T["fig8"].items():
    if panel == "average":
        continue
    vals = [v[0] for v in d.values()]
    v7[panel] = granularity(vals)[:6]
res["V7_fig8_granularity_N"] = v7
for panel, d in T["fig9"].items():
    if panel == "average":
        continue
    vals = [v[0] for v in d.values()]
    res["V7_fig8_granularity_N"][panel + "(fig9)"] = granularity(vals)[:6]
# 그림 8의 MEM 막대 = 그림 9의 MEM 막대?
v8 = {}
for panel in T["fig8"]:
    a, b = T["fig8"][panel]["mem"], T["fig9"][panel]["mem"]
    v8[panel] = {"fig8": round(a[0] * 100, 2), "fig9": round(b[0] * 100, 2), "diff_pp": round((a[0] - b[0]) * 100, 2),
                 "se_fig8": round(a[1] * 100, 2), "se_fig9": round(b[1] * 100, 2)}
res["V8_fig8_vs_fig9_mem_bars"] = v8
# 평균 막대와 여섯 과제 단순 평균
v9 = {}
for fig in ("fig8", "fig9"):
    for k in T[fig]["average"]:
        tasks = [p for p in T[fig] if p != "average"]
        mean6 = sum(T[fig][p][k][0] for p in tasks) / len(tasks)
        v9[f"{fig}:{k}"] = {"avg_bar": round(T[fig]["average"][k][0] * 100, 2), "mean_of_6": round(mean6 * 100, 2),
                            "diff_pp": round((T[fig]["average"][k][0] - mean6) * 100, 2)}
tasks10 = [p for p in T["fig10"] if p != "average"]
for k in T["fig10"]["average"]:
    mean7 = sum(T["fig10"][p][k][0] for p in tasks10) / len(tasks10)
    v9[f"fig10:{k}"] = {"avg_bar": round(T["fig10"]["average"][k][0] * 100, 2), "mean_of_7": round(mean7 * 100, 2),
                        "diff_pp": round((T["fig10"]["average"][k][0] - mean7) * 100, 2)}
for fig in ("fig8", "fig9", "fig10"):
    tk = [p for p in T[fig] if p != "average"]
    for k in T[fig]["average"]:
        prop = math.sqrt(sum(T[fig][p][k][1] ** 2 for p in tk)) / len(tk)
        v9[f"{fig}:{k}"]["se_avg_bar_pp"] = round(T[fig]["average"][k][1] * 100, 2)
        v9[f"{fig}:{k}"]["se_propagated_unweighted_pp"] = round(prop * 100, 2)
res["V9_average_vs_task_mean"] = v9

# ---------------- V10: 그림 9 (사전 학습 효과) ----------------
v10 = {}
for panel, d in T["fig9"].items():
    m, s = d["mem"]
    m2, s2 = d["post_train_only"]
    z, p = zp(m, s, m2, s2)
    v10[panel] = {"mem": round(m * 100, 1), "post_train_only": round(m2 * 100, 1), "diff_pp": round((m - m2) * 100, 1), "z": round(z, 2), "p_z": float(f"{p:.2g}")}
res["V10_fig9_pretraining"] = v10
# 사후 학습 전용 대 풀 메모리 (본문의 비교 주장)
v11 = {}
for panel in T["fig9"]:
    a, b = T["fig9"][panel]["post_train_only"], T["fig8"][panel]["pool"]
    z, p = zp(a[0], a[1], b[0], b[1])
    v11[panel] = {"post_train_only": round(a[0] * 100, 1), "pool": round(b[0] * 100, 1), "diff_pp": round((a[0] - b[0]) * 100, 1), "z": round(z, 2), "p_z": float(f"{p:.2g}")}
res["V11_posttrain_only_vs_pool"] = v11

# ---------------- V12: 그림 10 (메모리가 필요 없는 과제에서 π0.6과 같은가) ----------------
v12 = {}
for panel, d in T["fig10"].items():
    m, s = d["mem"]
    m2, s2 = d["pi06"]
    se = math.sqrt(s * s + s2 * s2)
    diff = m - m2
    z = diff / se if se > 0 else float("nan")
    p = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else None
    v12[panel] = {"mem": round(m * 100, 1), "pi06": round(m2 * 100, 1), "diff_pp": round(diff * 100, 1),
                  "ci95_pp": [round((diff - 1.96 * se) * 100, 1), round((diff + 1.96 * se) * 100, 1)] if se > 0 else None,
                  "p_z": float(f"{p:.2g}") if p is not None else None}
res["V12_fig10_mem_vs_pi06"] = v12
# 비열등 마진 ±10%p 기준으로 CI가 마진 안인지
res["V12_within_10pp"] = {k: (v["ci95_pp"] is None or (v["ci95_pp"][0] > -10 and v["ci95_pp"][1] < 10)) for k, v in v12.items()}
# 7개 과제 부호
diffs = [v12[p]["diff_pp"] for p in tasks10]
res["V12_sign_counts"] = {"positive": sum(d > 0.05 for d in diffs), "negative": sum(d < -0.05 for d in diffs), "zero": sum(abs(d) <= 0.05 for d in diffs)}

# ---------------- V13: 그림 10 격자 ----------------
res["V13_fig10_granularity_N"] = {p: granularity([v[0] for v in d.values()])[:6] for p, d in T["fig10"].items() if p != "average"}

# ---------------- V14: 본문 수치와 막대 ----------------
f = res["V2_fig7_raw"]
res["V14_text_vs_bars"] = {
    "text_plus_11_chopstick": round(f["pick_up_chopstick"]["mem"]["p"] * 100 - f["pick_up_chopstick"]["no_memory"]["p"] * 100, 2),
    "text_plus_62_fridge": round(f["open_fridge"]["mem"]["p"] * 100 - f["open_fridge"]["no_memory"]["p"] * 100, 2),
    "text_25pct_find_object_bar_no_memory": round(T["fig8"]["find_object"]["no_memory"][0] * 100, 2),
    "text_50pct_scoop_coffee_bar_no_memory": round(T["fig8"]["scoop_coffee"]["no_memory"][0] * 100, 2),
}
# 25%·50% 대 막대: 정수 비율 후보에서 이항 검정
fo = T["fig8"]["find_object"]["no_memory"][0]
res["V14_find_object_binom"] = {"p_bar": round(fo * 100, 2), "k_n": "5/16 (n=16 가정, 오차 막대와 표본 SD에서)",
                                "binom_p_two_sided_vs_0.25": round(float(stats.binomtest(5, 16, 0.25).pvalue), 3)}
sc = T["fig8"]["scoop_coffee"]["no_memory"][0]
res["V14_scoop_binom"] = {"p_bar": round(sc * 100, 2), "k_n": "8/15", "binom_p_two_sided_vs_0.5": round(float(stats.binomtest(8, 15, 0.5).pvalue), 3)}

# ---------------- V15: 단순 산술 (본문 수치) ----------------
res["V15_arithmetic"] = {
    "pretrain_horizon_s": 5 * 1, "posttrain_54s_over_18_frames": 54 / 18, "posttrain_54s_over_17_intervals": round(54 / 17, 2),
    "tokens_per_image_448_patch14": (448 // 14) ** 2,
    "naive_attention_cost_ratio_K6": 6 * 6, "factorized_over_naive_K6": round((6 * (448 // 14) ** 4 + (448 // 14) ** 2 * 36 / 4) / (6 * 6 * (448 // 14) ** 4), 4),
}
# 어텐션 비용 비율 (논문 식: O(n²K²) 대 O(Kn²+nK²), 시간 어텐션은 4개 층마다)
n = (448 // 14) ** 2
cost = {}
for K in (6, 12, 19):
    naive = (n * K) ** 2
    fact = K * n * n + n * K * K / 4
    cost[K] = {"naive_over_single_frame": naive / (n * n), "factorized_over_single_frame": round(fact / (n * n), 3),
               "temporal_share_of_factorized": round((n * K * K / 4) / fact, 4)}
res["V15_attention_cost"] = {"n_tokens": n, "per_K": cost}

# ---------------- V16: 막대 판독 가능한 시행 수 가정 목록 ----------------
res["V16_inferred_trials"] = {
    "find_object": "16 (이진 성공, 값 5/16·31/64 등 모두 /16 안, 표본 SD 표준오차와 일치)",
    "unpack_groceries": "15 (이진, 5/15 ...)", "window_cleaning": "17 (이진)",
    "scoop_coffee": "15 (이진)", "swap_3_mugs": "20회 x 3점 (또는 이에 해당하는 N=60)",
    "open_fridge": "21 (이진)",
    "note": "모두 막대 값의 정수성에서 낸 가정이며 논문에는 시행 수가 없음(10회는 IV-A 문장)",
}
# ---------------- V17: 막대별 최소 격자 N, 표본 SD 상한 대비 ----------------
def min_N(v, nmax=100, tol=0.0003):
    for N in range(1, nmax + 1):
        if abs(v - round(v * N) / N) < tol:
            return N
    return None


v17 = {}
for fig in ("fig8", "fig9", "fig10"):
    for panel, d in T[fig].items():
        if panel == "average":
            continue
        v17[f"{fig}:{panel}"] = {k: min_N(m) for k, (m, se) in d.items()}
res["V17_min_grid_N_per_bar"] = v17
# n = N 로 보았을 때(점수 1점 과제) SD/상한 비
v18 = {}
for fig, panel, n_by in (("fig8", "unpack_groceries", 15), ("fig8", "scoop_coffee", 15), ("fig8", "window_cleaning", 17), ("fig8", "swap_3_mugs", 20), ("fig8", "find_object", 16)):
    v18[f"{fig}:{panel}(n={n_by})"] = {k: sd_ratio(m, se, n_by)[2] for k, (m, se) in T[fig][panel].items()}
res["V18_sd_over_max_sd"] = v18

# ---------------- V19: 그림 7 Fisher (시행 수 가정 시나리오) ----------------
def fx(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


res["V19_fig7_fisher"] = {
    "open_fridge_n21": {"no_memory": "7/21", "mem": "20/21", "p": float(f"{fx(7, 21, 20, 21):.2g}"),
                        "wilson_no_memory": [round(x, 3) for x in wilson(7, 21)], "wilson_mem": [round(x, 3) for x in wilson(20, 21)]},
    "chopstick_scenarios": {
        "27/32_vs_19/20": float(f"{fx(27, 32, 19, 20):.2g}"),
        "54/64_vs_76/80": float(f"{fx(54, 64, 76, 80):.2g}"),
        "70/83_vs_95/100": float(f"{fx(70, 83, 95, 100):.2g}")},
    "chopstick_note": "무기억 84.37%는 27/32, 54/64, 70/83 등과 맞고 기억 95.0%는 19/20, 76/80, 95/100과 맞아 시행 수가 하나로 정해지지 않음",
}
# ---------------- V20: 이진 성공 가정 시 오차 막대에서 시행 수 n = 1 + m(1-m)/SE^2 ----------------
v20 = {}
for fig in ("fig6", "fig8", "fig9", "fig10"):
    for panel, d in T[fig].items():
        for k, (m, se) in d.items():
            if se <= 0 or m <= 0 or m >= 1:
                continue
            n_se = 1 + m * (1 - m) / (se * se)
            n_r = round(n_se)
            kk = m * n_r
            v20[f"{fig}:{panel}:{k}"] = {"n_from_se": round(n_se, 2), "k_at_round_n": round(kk, 2),
                                         "binary_consistent": bool(abs(n_se - n_r) < 0.15 and abs(kk - round(kk)) < 0.03)}
res["V20_binary_n_from_se"] = v20
res["V20_summary"] = {
    "n_binary_consistent": sum(v["binary_consistent"] for v in v20.values()),
    "n_total": len(v20),
    "consistent_keys": [k for k, v in v20.items() if v["binary_consistent"]],
    "consistent_n": {k: round(v["n_from_se"]) for k, v in v20.items() if v["binary_consistent"]},
}
# ---------------- V21: 이진으로 확인된 칸의 Fisher (시행 수는 V20에서 복원한 가정값) ----------------
v21 = {}
for panel, n in (("unpack_groceries", 15), ("scoop_coffee", 15), ("window_cleaning", 17)):
    ks = {k: round(T["fig8"][panel][k][0] * n) for k in T["fig8"][panel]}
    ks["post_train_only"] = round(T["fig9"][panel]["post_train_only"][0] * n)
    row = {"n": n, "k": ks}
    for k in ("no_memory", "pool", "proprio", "post_train_only"):
        row[f"mem_vs_{k}_p"] = float(f"{fx(ks['mem'], n, ks[k], n):.2g}")
        wl = wilson(ks[k], n)
        row[f"wilson_{k}"] = [round(wl[0], 2), round(wl[1], 2)]
    row["wilson_mem"] = [round(x, 2) for x in wilson(ks["mem"], n)]
    v21[panel] = row
bk = {k: round(T["fig10"]["batch_folding"][k][0] * 45) for k in ("pi06", "mem")}
v21["batch_folding"] = {"n": 45, "k": bk, "fisher_p": float(f"{fx(bk['mem'], 45, bk['pi06'], 45):.2g}")}
bx = {k: round(T["fig10"]["box_building"][k][0] * 10) for k in ("pi06", "mem")}
v21["box_building"] = {"n": 10, "k": bx, "fisher_p": float(f"{fx(bx['mem'], 10, bx['pi06'], 10):.2g}")}
# 최소 검출 가능한 하락: 이 시행 수에서 90% 기준 Fisher p<0.05가 되려면 필요한 하락폭
def min_detectable_drop(n, base):
    for d in range(1, n + 1):
        k2 = round(base * n) - d
        if k2 < 0:
            return None
        if fx(round(base * n), n, k2, n) < 0.05:
            return round(d / n * 100, 1)
v21["min_detectable_drop_pp_from_90pct"] = {"n=10": min_detectable_drop(10, 0.9), "n=45": min_detectable_drop(45, 0.9)}
res["V21_binary_cells"] = v21
# ---------------- V22: 그림 6 Welch 민감도 (레시피 n=5 또는 n=10, 부엌 청소 n=10) ----------------
v22 = {}
for panel, n_ in (("recipe", 5), ("recipe_n10", 10), ("clean_kitchen", 10)):
    d = T["fig6"]["recipe" if panel.startswith("recipe") else panel]
    m, s_ = d["mem"]
    row = {}
    for k, (m2, s2) in d.items():
        if k == "mem":
            continue
        t, df, p = welch_p(m, s_, n_, m2, s2, n_)
        row[k] = {"t": round(t, 2), "df": round(df, 1), "p": float(f"{p:.2g}")}
    v22[f"{panel}(n={n_})"] = row
res["V22_fig6_welch_sensitivity"] = v22
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print("wrote", OUT)
