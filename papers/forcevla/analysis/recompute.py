"""논문 표·그림 수치를 다시 계산하고 검정한다 (수 초, 학습·추론 없음).

입력: 논문 본문 표 2, 3, 5의 보고 수치와 그림 5 막대 라벨(원문 PDF 7쪽을 확대해 읽음).
시행 수(n)는 본문과 보고 백분율이 맞아떨어지는 값으로 정한다. 본문에 없는 n은 "가정"으로 표시한다.
출력: analysis/recompute_results.json
"""
import json
import math
import pathlib

from scipy.stats import fisher_exact

OUT = pathlib.Path(__file__).with_name("recompute_results.json")
Z = 1.959964


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(fisher_exact(table, alternative="two-sided")[1])


res = {}

# ---------- 표 2: 일반화 5개 설정 ----------
# 설정별 n: Object Gen 1 = 25 (48, 32, 80% 는 4% 단위), Object Gen 2 = 20, Height = 9 (66.67, 77.78, 88.89, 44.44 = k/9),
# Occlusion = 20, Unstable = 20 (모두 5% 단위). 4 설정 n은 본문에 없고, 백분율이 k/n과 맞는 최소 n으로 정했다(가정).
settings = ["Object Gen.1", "Object Gen.2", "Height Gen.", "Visual Occlusion", "Unstable Socket"]
n_set = [25, 20, 9, 20, 20]
t2_reported = {  # 보고된 백분율 (%)
    "pi0-base w/o F": [48.00, 10.00, 66.67, 60.00, 10.00],
    "pi0-base w/ F": [32.00, 10.00, 77.78, 30.00, 10.00],
    "pi0-fast w/o F": [80.00, 35.00, 88.89, 50.00, 10.00],
    "pi0-fast w/ F": [32.00, 5.00, 44.44, 50.00, 30.00],
    "ForceVLA": [80.00, 40.00, 88.89, 90.00, 20.00],
}
t2_avg_reported = {"pi0-base w/o F": 38.93, "pi0-base w/ F": 31.96, "pi0-fast w/o F": 52.78,
                   "pi0-fast w/ F": 32.29, "ForceVLA": 63.78}

t2 = {}
for m, pcts in t2_reported.items():
    ks, rt_err = [], []
    for pct, n in zip(pcts, n_set):
        k = round(pct / 100 * n)
        ks.append(k)
        rt_err.append(abs(100 * k / n - pct))
    unweighted = sum(pcts) / len(pcts)
    t2[m] = {
        "k": ks,
        "max_roundtrip_err_pp": round(max(rt_err), 3),
        "unweighted_mean_recomputed": round(unweighted, 2),
        "unweighted_mean_reported": t2_avg_reported[m],
        "pooled_k": sum(ks),
        "pooled_n": sum(n_set),
        "pooled_rate_pct": round(100 * sum(ks) / sum(n_set), 2),
        "pooled_wilson95_pct": [round(100 * x, 1) for x in wilson(sum(ks), sum(n_set))],
    }
res["table2"] = t2
res["table2_trials_per_setting"] = dict(zip(settings, n_set))

# 풀링 Fisher 검정 (독립 가정), ForceVLA 대 기준선 4개, Holm 보정
base_names = ["pi0-base w/o F", "pi0-base w/ F", "pi0-fast w/o F", "pi0-fast w/ F"]
fv_k, fv_n = t2["ForceVLA"]["pooled_k"], t2["ForceVLA"]["pooled_n"]
raw_p = []
pooled_tests = {}
for b in base_names:
    p = fisher(fv_k, fv_n, t2[b]["pooled_k"], t2[b]["pooled_n"])
    raw_p.append(p)
for b, p, pa in zip(base_names, raw_p, holm(raw_p)):
    pooled_tests[b] = {"fisher_p": round(p, 6), "holm_p": round(pa, 6)}
res["table2_pooled_fisher_vs_ForceVLA"] = pooled_tests

# 설정별 Fisher (ForceVLA 대 pi0-base w/ F), 설정 5개 Holm
per = []
for i, s in enumerate(settings):
    per.append(fisher(t2["ForceVLA"]["k"][i], n_set[i], t2["pi0-base w/ F"]["k"][i], n_set[i]))
res["table2_per_setting_fisher_vs_base_wF"] = {s: {"p": round(p, 4), "holm": round(q, 4)} for s, p, q in zip(settings, per, holm(per))}

# ---------- 그림 5 막대 라벨 (원문 7쪽) ----------
# 막대 순서: Insert USB, Insert Plug, Pump Bottle, Wipe Board-1, Wipe Board-2, Peel Cucumber, Average(보고값)
fig5 = {
    "pi0-fast w/o F": [0.0, 25.0, 25.0, 91.0, 45.0, 0.0, 31.0],
    "pi0-fast w/ F": [0.0, 30.0, 25.0, 20.0, 10.0, 0.0, 14.2],
    "pi0-base w/o F": [5.0, 45.0, 33.0, 45.0, 9.0, 87.0, 37.3],
    "pi0-base w/ F": [5.0, 60.0, 83.0, 0.0, 0.0, 93.0, 40.2],
    "ForceVLA": [25.0, 80.0, 67.0, 64.0, 27.0, 100.0, 60.5],
}
fig5_cols = ["Insert USB", "Insert Plug", "Pump Bottle", "Wipe Board-1", "Wipe Board-2", "Peel Cucumber"]
f5 = {}
for m, v in fig5.items():
    six = sum(v[:6]) / 6  # 그림의 Average 열과 같은 6개 막대 평균
    five_task = (v[0] + v[1] + v[2] + (v[3] + v[4]) / 2 + v[5]) / 5  # 와이프 보드 두 막대를 한 과제로 평균
    f5[m] = {
        "reported_average": v[6],
        "mean_of_6_bars": round(six, 2),
        "mean_of_5_tasks_wipe_avg": round(five_task, 2),
    }
res["figure5_average_check"] = f5
res["figure5_gap_pp"] = {
    "ForceVLA_minus_base_w/o_F_6bar": round(60.5 - 37.3, 1),
    "ForceVLA_minus_base_w/_F_6bar": round(60.5 - 40.2, 1),
    "ForceVLA_minus_base_w/_F_5task": round(f5["ForceVLA"]["mean_of_5_tasks_wipe_avg"] - f5["pi0-base w/ F"]["mean_of_5_tasks_wipe_avg"], 1),
    "ForceVLA_minus_base_w/o_F_5task": round(f5["ForceVLA"]["mean_of_5_tasks_wipe_avg"] - f5["pi0-base w/o F"]["mean_of_5_tasks_wipe_avg"], 1),
}

# 막대 값이 본문 시행 수로 나누어 떨어지는지 확인 (USB 20, Plug 20, Pump 20, Wipe 10, Peel 15 가정)
trial_n = {"Insert USB": 20, "Insert Plug": 20, "Pump Bottle": 20, "Wipe Board-1": 10, "Wipe Board-2": 10, "Peel Cucumber": 15}
div = {}
for m, v in fig5.items():
    bad = []
    for col, val in zip(fig5_cols, v[:6]):
        n = trial_n[col]
        k = round(val / 100 * n)
        if abs(100 * k / n - val) > 0.05:
            bad.append({"col": col, "value": val, "nearest_k_over_n": f"{k}/{n}"})
    div[m] = bad
res["figure5_not_multiple_of_stated_trials"] = div

# ---------- 표 5: 다중 과제 (4개 과제, 시행 수 가정: Pump 20, Plug 20, USB 20, Wipe 10) ----------
t5_n = {"Pump Bottle": 20, "Insert Plug": 20, "Insert USB": 20, "Wipe Board": 10}
t5_pct = {
    "pi0-fast w/o F": [0, 0, 0, 0],
    "pi0-fast w/ F": [0, 0, 0, 0],
    "pi0-base w/o F": [20, 0, 0, 0],
    "pi0-base w/ F": [50, 100, 10, 10],
    "ForceVLA": [80, 100, 10, 80],
}
t5 = {}
for m, pcts in t5_pct.items():
    ks = [round(p / 100 * n) for p, n in zip(pcts, t5_n.values())]
    t5[m] = {"k": ks, "avg_recomputed_pct": round(sum(pcts) / 4, 2), "pooled_k": sum(ks)}
tot_n = sum(t5_n.values())
t5_fv_vs_wF = fisher(t5["ForceVLA"]["pooled_k"], tot_n, t5["pi0-base w/ F"]["pooled_k"], tot_n)
t5_fv_vs_woF = fisher(t5["ForceVLA"]["pooled_k"], tot_n, t5["pi0-base w/o F"]["pooled_k"], tot_n)
res["table5"] = t5
res["table5_fisher"] = {"ForceVLA_vs_base_wF_p": round(t5_fv_vs_wF, 4), "ForceVLA_vs_base_woF_p": round(t5_fv_vs_woF, 6),
                        "pooled_n": tot_n}

# ---------- 표 3: 어블레이션 (n=20 가정, 과제 미지정) ----------
t3_n = 20
t3 = {"baseline[10]": 45, "linear before VLM": 55, "MoE before VLM": 0, "concat after VLM": 60, "ForceVLA": 80}
t3_k = {m: round(p / 100 * t3_n) for m, p in t3.items()}
res["table3"] = {
    "k_of_20": t3_k,
    "ForceVLA_vs_baseline_p": round(fisher(t3_k["ForceVLA"], 20, t3_k["baseline[10]"], 20), 4),
    "ForceVLA_vs_concat_after_VLM_p": round(fisher(t3_k["ForceVLA"], 20, t3_k["concat after VLM"], 20), 4),
    "concat_after_VLM_vs_baseline_p": round(fisher(t3_k["concat after VLM"], 20, t3_k["baseline[10]"], 20), 4),
}

# ---------- 표 1: 큐컴버 (표준편차·시행 정보 없음) ----------
res["table1_note"] = "평균 길이 차이(14.12 대 13.17 대 10.27 cm)에 대한 검정은 표준편차와 시행별 값이 없어 불가능"

# ---------- 주요 수치 차이 ----------
res["headline_pp"] = {"ForceVLA_minus_base_w/o_F": round(60.5 - 37.3, 1),
                      "ForceVLA_minus_base_w/_F": round(60.5 - 40.2, 1),
                      "relative_gain_vs_base_w/o_F_pct": round(100 * (60.5 - 37.3) / 37.3, 1)}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
print(json.dumps(res, ensure_ascii=False, indent=2))
