"""MuSe(arXiv:2606.30988 v4) 보고 수치 재계산과 일관성 검사.

입력은 논문 표·본문·그림에서 옮긴 값뿐이다(코드 미공개, 원자료 없음).
  - 표 1, 3, 4(실로봇 성공 점수/시행 수), 그림 3 막대 라벨, 그림 4 막대 라벨, 표 2, 표 5, 6.3–6.5절 설정값
  - 점수는 부분 점수(0.5)를 포함하므로 정수 성공 횟수가 아니다. 검정은 점수를 내림/올림한 정수로
    바꾼 두 경우의 p값 범위로 보고한다(시행 수 n은 논문 6.5절 기재값, 시행 간 독립 가정).

결과: analysis/recompute_results.json
실행: python3 analysis/recompute.py    (numpy, scipy)
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
R = {}

# ---------------------------------------------------------------- 논문 값
N = {"wipe": 15, "peg": 15, "pp": 10}
# 표 3 (전진 전이, 부록 6.2)
FWD = {
    "No F/T": {"wipe": 5, "peg": 9, "pp": 7.5},
    "No Pretrain": {"wipe": 8, "peg": 6, "pp": 5},
    "Diffusion Policy": {"wipe": 10, "peg": 10, "pp": 6},
    "MuSe": {"wipe": 11.5, "peg": 13, "pp": 7.5},
}
# 표 1 (보상 제어 끔)
TAB1 = {
    "No F/T": {"wipe": 5, "peg": 9},
    "No Pretrain": {"wipe": 8, "peg": 6},
    "MuSe w/o Compliance": {"wipe": 8, "peg": 11},
    "MuSe": {"wipe": 11.5, "peg": 13},
}
# 표 4 (후진 전이)
BWD = {
    "Pretraining Only": {"wipe": 8.5, "peg": 8, "pp": 5.5},
    "No ER": {"wipe": 0.5, "peg": 2, "pp": 5.5},
    "Diffusion Policy": {"wipe": 6.5, "peg": 5, "pp": 6},
    "MuSe": {"wipe": 12.5, "peg": 10, "pp": 6.5},
}
# 본문 4.2, 4.3절 서술값
TEXT_FWD = {"MuSe": {"wipe": 11.5, "peg": 13, "pp": 7.5}, "No F/T": {"wipe": 5, "peg": 9, "pp": 7.5},
            "No Pretrain": {"wipe": 8, "peg": 6, "pp": 5}}
TEXT_BWD = {"MuSe": {"wipe": 12.5, "peg": 10, "pp": 6.5}, "Pretraining Only": {"wipe": 8.5, "peg": 8, "pp": 5.5},
            "No ER": {"wipe": 0.5, "peg": 2}}
# 그림 3 막대 라벨(%)
FIG3_BWD = {"pp": {"Pretraining Only": 55, "No ER": 55, "MuSe": 65},
            "wipe": {"Pretraining Only": 57, "No ER": 3, "MuSe": 83},
            "peg": {"Pretraining Only": 53, "No ER": 13, "MuSe": 67}}
FIG3_FWD = {"wipe": {"No Pretrain": 53, "No F/T": 33, "MuSe": 77},
            "peg": {"No Pretrain": 40, "No F/T": 60, "MuSe": 93},
            "pp": {"No Pretrain": 50, "No F/T": 75, "MuSe": 75}}
FIG3_FWD_V1 = {"peg": {"MuSe": 87}}  # v1 그림 3의 값(v2 이후 93)
# 그림 1 라벨
FIG1 = {"bwd_muse": 72.5, "bwd_pre": 55.0, "bwd_delta": 17.5, "fwd_muse": 80.0, "fwd_noft": 54.0}
# 그림 4 막대 라벨 (그림 판독 값), 본문 값
FIG4 = {"MuSe": 8.42, "No VP": 9.90, "No ER": 12.38, "No Early": 12.06, "No Late": 11.52,
        "Linear": 11.80, "Additive": 11.80}
TEXT4 = {"MuSe": 8.424, "late-only(=No Early)": 12.061, "Linear": 11.518, "Additive": 11.800,
         "No VP": 9.90, "No ER": 12.376}
TAB2 = {"Diffusion Policy": 18.27, "MuSe": 8.42}
DP_FT_ERR = 6.04

# 표 5
PRE = [278, 94, 140, 99, 50, 49, 43, 22, 50, 43, 50, 18, 20, 20, 20, 18, 18, 22, 17, 100, 100]
FT = [60, 20, 160, 144, 50]


def rnd(x, k=3):
    return round(float(x), k)


# ---------------------------------------------------------------- V1: 표 3 <-> 표 1 <-> 본문
v1 = []
for m, d in TEXT_FWD.items():
    for t, v in d.items():
        v1.append({"method": m, "task": t, "text": v, "table3": FWD[m][t], "ok": v == FWD[m][t]})
for m in ("No F/T", "No Pretrain", "MuSe"):
    for t in ("wipe", "peg"):
        v1.append({"method": m, "task": t, "table1": TAB1[m][t], "table3": FWD[m][t], "ok": TAB1[m][t] == FWD[m][t]})
R["V1_fwd_table_text_consistency"] = {"checked": len(v1), "mismatch": [x for x in v1 if not x["ok"]]}

# V2: 표 4 <-> 본문
v2 = []
for m, d in TEXT_BWD.items():
    for t, v in d.items():
        v2.append({"method": m, "task": t, "text": v, "table4": BWD[m][t], "ok": v == BWD[m][t]})
R["V2_bwd_table_text_consistency"] = {"checked": len(v2), "mismatch": [x for x in v2 if not x["ok"]]}

# ---------------------------------------------------------------- V3: 그림 3 라벨 <-> 점수/시행 수
v3 = []
for tag, fig, tab in (("bwd", FIG3_BWD, BWD), ("fwd", FIG3_FWD, FWD)):
    for t, d in fig.items():
        for m, pct in d.items():
            calc = 100 * tab[m][t] / N[t]
            v3.append({"set": tag, "task": t, "method": m, "fig_label_pct": pct, "calc_pct": rnd(calc, 2),
                       "ok": round(calc) == pct})
R["V3_fig3_labels"] = {"checked": len(v3), "mismatch": [x for x in v3 if not x["ok"]],
                       "v1_muse_peg_label": 87, "v1_ok": round(100 * 13 / 15) == 87,
                       "k_if_93": {"score_needed_of_15": rnd(0.93 * 15, 2), "calc_pct_for_14": rnd(100 * 14 / 15, 2)}}

# ---------------------------------------------------------------- V4: 그림 1 합산
def pooled(tab, m):
    return sum(tab[m].values()) / sum(N.values()) * 100  # N 합 40


bw = {m: rnd(pooled(BWD, m), 2) for m in BWD}
fw = {m: rnd(pooled(FWD, m), 2) for m in FWD}
R["V4_fig1_pooled"] = {
    "backward_pooled_pct": bw, "forward_pooled_pct": fw,
    "fig1": FIG1,
    "bwd_delta_pp": rnd(bw["MuSe"] - bw["Pretraining Only"], 2),
    "bwd_relative_gain_pct": rnd(100 * (bw["MuSe"] / bw["Pretraining Only"] - 1), 1),
    "fwd_noft_calc": fw["No F/T"],
    "fwd_muse_if_peg_93": rnd((11.5 + 14 + 7.5) / 40 * 100, 2),
    "ok_bwd": bw["MuSe"] == FIG1["bwd_muse"] and bw["Pretraining Only"] == FIG1["bwd_pre"],
    "ok_fwd": fw["MuSe"] == FIG1["fwd_muse"] and round(fw["No F/T"]) == FIG1["fwd_noft"],
    "note": "그림 1 합산(80%, 72.5%)은 표 3·4의 점수 합/40과 일치. 그림 3의 peg 93%는 14/15에 해당해 합산 80%와는 맞지 않음",
    "mean_of_task_pct_bwd_muse": rnd(np.mean([100 * BWD['MuSe'][t] / N[t] for t in N]), 2),
}

# ---------------------------------------------------------------- V5: 표 5 합계
R["V5_table5"] = {
    "pretrain_rows": len(PRE), "pretrain_sum": sum(PRE), "reported_pretrain_total": 1271,
    "ok_pretrain": sum(PRE) == 1271,
    "finetune_rows": len(FT), "finetune_sum": sum(FT), "reported_finetune_total": 434,
    "ok_finetune": sum(FT) == 434,
    "pretrain_share_of_uniform_mix_pct": rnd(100 * 1271 / (1271 + 434), 1),
    "finetune_share_pct": rnd(100 * 434 / (1271 + 434), 1),
    "largest_pretrain_task_eps": max(PRE),
    "wipe_related_eps": {"pretrain_erase_the_drawing": 278, "finetune_vase_erase": 160},
    "peg_related_eps": {"pretrain_rect+white+green+small_purple": 50 + 49 + 43 + 50, "finetune_peg": 144},
}

# ---------------------------------------------------------------- V6: 평가 시행 구성(6.5절)
comp = {
    "wipe_bwd 10+5": 10 + 5, "peg_bwd 7+3+5": 7 + 3 + 5, "pp_bwd 4+2+4": 4 + 2 + 4,
    "wipe_fwd 4+11": 4 + 11, "wipe_fwd eraser 3+12": 3 + 12, "pp_fwd 3+7": 3 + 7, "peg_fwd": 15,
}
R["V6_trial_composition"] = {"sums": comp, "expected": {"wipe": 15, "peg": 15, "pp": 10},
                             "ok": all(v in (15, 10) for v in comp.values())}

# ---------------------------------------------------------------- V7: 그림 4 <-> 본문
v7 = {
    "fig4_vs_text": {
        "MuSe": {"fig": FIG4["MuSe"], "text": TEXT4["MuSe"], "ok": round(TEXT4["MuSe"], 2) == FIG4["MuSe"]},
        "No VP": {"fig": FIG4["No VP"], "text": TEXT4["No VP"], "ok": TEXT4["No VP"] == FIG4["No VP"]},
        "No ER": {"fig": FIG4["No ER"], "text": TEXT4["No ER"], "ok": round(TEXT4["No ER"], 2) == FIG4["No ER"]},
        "late-only = No Early": {"fig": FIG4["No Early"], "text": TEXT4["late-only(=No Early)"],
                                 "ok": round(TEXT4["late-only(=No Early)"], 2) == FIG4["No Early"]},
        "Linear": {"fig": FIG4["Linear"], "text": TEXT4["Linear"], "ok": round(TEXT4["Linear"], 2) == FIG4["Linear"],
                   "note": "본문 11.518은 그림에서 No Late(11.52)와 일치"},
        "Additive": {"fig": FIG4["Additive"], "text": TEXT4["Additive"], "ok": round(TEXT4["Additive"], 2) == FIG4["Additive"]},
    },
    "relative_to_muse_pct": {k: rnd(100 * (v / FIG4["MuSe"] - 1), 1) for k, v in FIG4.items() if k != "MuSe"},
    "text_says_no_er_highest": TEXT4["No ER"] == max(FIG4.values()) or round(TEXT4["No ER"], 2) == max(FIG4.values()),
    "text_has_no_late_value": False,
    "dp_vs_muse_ratio": rnd(TAB2["Diffusion Policy"] / TAB2["MuSe"], 3),
    "dp_pretrain_over_finetune_ratio": rnd(TAB2["Diffusion Policy"] / DP_FT_ERR, 3),
    "dp_vs_no_er_ratio": rnd(TAB2["Diffusion Policy"] / FIG4["No ER"], 3),
}
R["V7_fig4_table2"] = v7

# ---------------------------------------------------------------- V8: 차원·토큰 산술(부록 6.3, 3.5절)
dims = {
    "tokens_per_frame_256/16=16 -> 16*16": 16 * 16,
    "video_tokens_4x256": 4 * 256,
    "ft_per_action_step_6x6": 6 * 6,
    "ft_hist_4x36": 4 * 36,
    "ft_future_samples_per_finger_16x6": 16 * 6,
    "ft_head_dim_2_fingers_x_36": 2 * 36,
    "action_steps_per_frame_20/5": 20 // 5,
    "ft_samples_per_action_step_120/20": 120 // 20,
    "horizon_s_16_steps_at_20Hz": 16 / 20,
    "per_gpu_batch_40/8": 40 / 8,
    "ft_force_limit_note": "종료 조건 20 N, 6 Nm (6.4절); L2 오차 8.42는 N과 Nm을 섞은 6D 노름이라 단위 없음",
}
R["V8_arithmetic"] = dims

# ---------------------------------------------------------------- V9–V10: 검정
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def newcombe(k1, n1, k2, n2):
    """두 비율 차(p1 - p2)의 Newcombe-Wilson 95% 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def variants(c):
    return sorted({math.floor(c), math.ceil(c)})


def fisher_range(c1, c2, n):
    ps = []
    for a, b in product(variants(c1), variants(c2)):
        ps.append(stats.fisher_exact([[a, n - a], [b, n - b]])[1])
    return min(ps), max(ps)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [None] * len(ps)
    run = 0.0
    m = len(ps)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


tests = []
for setname, tab in (("fwd", FWD), ("bwd", BWD)):
    for base in tab:
        if base == "MuSe":
            continue
        for t in ("wipe", "peg", "pp"):
            lo, hi = fisher_range(tab["MuSe"][t], tab[base][t], N[t])
            tests.append({"set": setname, "task": t, "vs": base, "muse": tab["MuSe"][t], "base": tab[base][t],
                          "n": N[t], "p_min": lo, "p_max": hi,
                          "muse_ci": [rnd(x, 3) for x in wilson(tab["MuSe"][t], N[t])],
                          "base_ci": [rnd(x, 3) for x in wilson(tab[base][t], N[t])]})
holm_p = holm([x["p_max"] for x in tests])
for x, a in zip(tests, holm_p):
    x["holm_p_on_p_max"] = rnd(a, 4)
    x["p_min"] = rnd(x["p_min"], 4)
    x["p_max"] = rnd(x["p_max"], 4)
R["V9_per_task_fisher"] = {"n_tests": len(tests), "holm_family": "전진 9 + 후진 9 = 18 비교, p_max 기준 Holm",
                           "tests": tests}

# 표 1 (보상 제어 끔) 비교
t1 = []
for t in ("wipe", "peg"):
    for a, b in (("MuSe", "MuSe w/o Compliance"), ("MuSe w/o Compliance", "No F/T"),
                 ("MuSe w/o Compliance", "No Pretrain")):
        lo, hi = fisher_range(TAB1[a][t], TAB1[b][t], 15)
        t1.append({"task": t, "a": a, "b": b, "a_score": TAB1[a][t], "b_score": TAB1[b][t],
                   "p_min": rnd(lo, 4), "p_max": rnd(hi, 4)})
R["V10_table1_compliance_tests"] = t1

# 합산(3과제, n = 40) 비교
pool = []
for setname, tab in (("fwd", FWD), ("bwd", BWD)):
    for base in tab:
        if base == "MuSe":
            continue
        a = sum(tab["MuSe"].values())
        b = sum(tab[base].values())
        lo, hi = fisher_range(a, b, 40)
        pool.append({"set": setname, "vs": base, "muse": a, "base": b, "n": 40,
                     "pct": [rnd(100 * a / 40, 1), rnd(100 * b / 40, 1)],
                     "p_min": rnd(lo, 4), "p_max": rnd(hi, 4),
                     "muse_ci": [rnd(x, 3) for x in wilson(a, 40)], "base_ci": [rnd(x, 3) for x in wilson(b, 40)],
                     "diff_pp": rnd(100 * (a - b) / 40, 1),
                     "diff_ci_pp": [rnd(100 * x, 1) for x in newcombe(a, 40, b, 40)]})
R["V11_pooled_fisher_n40"] = {"note": "과제 풀링은 과제 간 이질성과 같은 과제 내 상관을 무시한 근사",
                              "tests": pool}


# Cochran–Mantel–Haenszel(과제 층화, 점수를 정수로 내림/올림한 두 경우)
def cmh(tables):
    num = 0.0
    var = 0.0
    for a, n1, c, n2 in tables:  # a: 그룹1 성공, c: 그룹2 성공
        n = n1 + n2
        m1 = a + c
        e = n1 * m1 / n
        v = n1 * n2 * m1 * (n - m1) / (n * n * (n - 1))
        num += a - e
        var += v
    chi = (abs(num) - 0.5) ** 2 / var
    return stats.chi2.sf(chi, 1)


cm = []
for setname, tab in (("fwd", FWD), ("bwd", BWD)):
    for base in tab:
        if base == "MuSe":
            continue
        ps = []
        for pick in ("floor", "ceil"):
            f = math.floor if pick == "floor" else math.ceil
            tabs = [(f(tab["MuSe"][t]), N[t], f(tab[base][t]), N[t]) for t in ("wipe", "peg", "pp")]
            ps.append(cmh(tabs))
        cm.append({"set": setname, "vs": base, "p_floor": rnd(ps[0], 4), "p_ceil": rnd(ps[1], 4)})
R["V12_cmh_task_stratified"] = {"tests": cm}

# ---------------------------------------------------------------- V13: 검정력(전진/후진 평균 차이를 과제당 n으로)
def n_needed(p1, p2, alpha=0.05, power=0.8):
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


R["V13_power"] = {
    "wipe_fwd_muse_vs_nopretrain_(76.7% vs 53.3%)": n_needed(11.5 / 15, 8 / 15),
    "peg_fwd_muse_vs_dp_(86.7% vs 66.7%)": n_needed(13 / 15, 10 / 15),
    "pp_bwd_muse_vs_pretrain_(65% vs 55%)": n_needed(0.65, 0.55),
    "peg_bwd_muse_vs_pretrain_(66.7% vs 53.3%)": n_needed(10 / 15, 8 / 15),
    "wipe_bwd_muse_vs_pretrain_(83.3% vs 56.7%)": n_needed(12.5 / 15, 8.5 / 15),
    "note": "시행당 두 표본 비율 비교, 양측 α=0.05, 검정력 0.8, 정규근사. 각 값은 한 방법당 필요한 시행 수",
}

# ---------------------------------------------------------------- V14: 교란 점검용 계산
# 표 1에서 보상 제어를 끈 MuSe와 No Pretrain/No F/T의 차이 = 사전학습(+F/T 입력) 효과, 보상 제어 효과 = MuSe - w/o
R["V14_compliance_decomposition"] = {
    "wipe": {"full_minus_wo_compliance": 11.5 - 8, "wo_compliance_minus_no_ft": 8 - 5,
             "wo_compliance_minus_no_pretrain": 8 - 8},
    "peg": {"full_minus_wo_compliance": 13 - 11, "wo_compliance_minus_no_ft": 11 - 9,
            "wo_compliance_minus_no_pretrain": 11 - 6},
    "share_of_full_gain_over_no_ft_from_compliance_pct": {
        "wipe": rnd(100 * (11.5 - 8) / (11.5 - 5), 1), "peg": rnd(100 * (13 - 11) / (13 - 9), 1)},
    "note": "보상 제어를 켜고 끄는 비교는 전진 과제 두 개에만 있고 후진 과제에는 없음",
}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1))
print("wrote", OUT)
print("V3 mismatch:", R["V3_fig3_labels"]["mismatch"])
print("V4:", R["V4_fig1_pooled"]["backward_pooled_pct"], R["V4_fig1_pooled"]["forward_pooled_pct"])
print("V5:", R["V5_table5"]["pretrain_sum"], R["V5_table5"]["finetune_sum"])
for x in tests:
    print(x["set"], x["task"], x["vs"], x["muse"], x["base"], x["p_min"], x["p_max"], x["holm_p_on_p_max"])
for x in pool:
    print("pool", x["set"], x["vs"], x["muse"], x["base"], x["p_min"], x["p_max"])
for x in cm:
    print("cmh", x)
print(R["V13_power"])
