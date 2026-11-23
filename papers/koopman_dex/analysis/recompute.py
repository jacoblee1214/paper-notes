"""논문 표 수치 재계산 (arXiv:2602.07413 v4). 실행: python3 analysis/recompute.py
표 3-6, 9, 10, 2 와 Fig. 3 막대 판독값(그림 판독 값, 오차 약 1%p)을 사용한다."""
import json, math, itertools
from math import comb
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
RES = {}
TASKS = ["Bucket", "Laptop", "Toilet", "Door", "Tool", "Reorient", "Reloc"]

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

# ---------- 표 3-6 (Flow, DynaMo 순서; 7개 과제) ----------
T3 = {"Diff(2,4,8)": ([90.7,78.0,69.0,0.0,88.3,80.7,89.2],[75.3,78.7,68.0,0.0,65.3,32.7,50.0],(70.8,52.7)),
      "Diff(2,8,16)": ([92.7,60.0,67.6,0.0,92.7,81.3,86.7],[76.0,82.7,76.7,0.0,78.7,3.3,55.3],(68.9,53.2))}
T4 = {"ACT10": ([80.7,90.0,74.5,0.0,52.0,80.0,82.0],[72.7,78.7,82.7,0.0,88.7,28.0,72.0],(65.6,60.4)),
      "ACT20": ([77.3,86.7,69.0,23.2,79.2,85.0,89.4],[83.3,92.0,78.7,41.3,96.7,25.3,72.0],(73.4,69.9)),
      "ACT40": ([74.7,83.3,68.3,44.2,67.2,78.8,84.6],[74.7,76.0,82.0,44.7,92.0,26.0,81.3],(71.6,68.1))}
T5 = {"KODex V1": ([10.0,6.7,3.4,3.3,36.7,73.3,26.7],[80.0,33.3,16.7,50.0,20.0,70.0,23.3],(22.9,41.9)),
      "KODex V2": ([13.3,10.0,10.3,10.0,30.0,76.7,23.3],[80.0,33.3,6.7,53.3,33.3,73.3,46.7],(24.8,46.7)),
      "KODex V3": ([20.0,23.3,3.4,13.3,30.0,63.3,33.3],[86.7,13.3,3.3,53.3,50.0,76.7,20.0],(26.7,43.3))}
T6 = {"KOROL [64]": ([50.7,20.7,33.8,10.7,32.0,32.0,8.0],[56.7,72.0,42.8,11.33,13.3,52.0,15.33],(26.8,37.6)),
      "KOROL [2in,128]": ([46.7,19.3,27.6,9.3,32.7,36.7,8.0],[81.3,71.3,43.4,9.33,4.7,46.0,15.33],(25.8,44.9)),
      "KOROL [512,256,128]": ([35.3,16.9,16.6,14.0,13.0,34.0,6.7],[78.7,71.3,53.1,11.33,5.33,37.33,17.33],(19.5,46.1))}
ALL = {}
for d in (T3, T4, T5, T6): ALL.update(d)

avg_rows = {}
for k, (fl, dy, rep) in ALL.items():
    f = np.array(fl); y = np.array(dy)
    avg_rows[k] = {"flow_mean": round(f.mean(), 2), "flow_reported": rep[0], "flow_sd0": round(f.std(), 1), "flow_sd1": round(f.std(ddof=1), 1),
                   "dyn_mean": round(y.mean(), 2), "dyn_reported": rep[1], "dyn_sd0": round(y.std(), 1), "dyn_sd1": round(y.std(ddof=1), 1),
                   "flow_diff": round(f.mean() - rep[0], 2), "dyn_diff": round(y.mean() - rep[1], 2)}
RES["V1_avg_columns"] = avg_rows


# ---------- 5-seed 행의 "Average ± SD": 과제 간 SD 만으로는 설명되지 않는지 (풀링 SD 근사) ----------
CELLSD = {  # (flow SDs, dynamo SDs) 표 3, 4
 "Diff(2,4,8)": ([3.9,4.5,3.1,0,2.9,2.5,6.0],[5.0,6.5,8.8,0,8.6,3.8,2.1]),
 "Diff(2,8,16)": ([2.5,19.1,4.6,0,1.3,4.5,2.9],[3.3,4.9,4.7,0,6.2,2.9,3.4]),
 "ACT10": ([5.7,3.7,6.4,0,18.3,7.7,9.8],[8.5,2.7,3.9,0,8.4,8.7,3.8]),
 "ACT20": ([5.5,4.1,8.1,9.3,11.6,5.8,7.8],[7.0,3.4,3.4,5.1,4.1,4.5,3.0]),
 "ACT40": ([6.9,5.8,4.5,8.0,13.9,5.2,3.9],[2.7,6.5,5.4,13.9,1.8,8.0,1.8])}
pooled = {}
for k, (fsd, dsd) in CELLSD.items():
    fl, dy, rep = ALL[k]
    pf = math.sqrt(np.var(fl) + np.mean(np.square(fsd))); pd_ = math.sqrt(np.var(dy) + np.mean(np.square(dsd)))
    pooled[k] = {"flow_pooled_approx": round(pf, 1), "flow_reported_sd": {"Diff(2,4,8)":30.5,"Diff(2,8,16)":30.6,"ACT10":30.3,"ACT20":25.7,"ACT40":18.7}[k],
                 "dyn_pooled_approx": round(pd_, 1), "dyn_reported_sd": {"Diff(2,4,8)":25.5,"Diff(2,8,16)":25.2,"ACT10":27.0,"ACT20":24.7,"ACT40":22.7}[k]}
RES["V1b_pooled_sd"] = pooled

# ---------- 정수 실현 가능성 ----------
def feasible(v, n):
    """보고된 소수 1자리 값 v(%)가 n회 시행의 정수 성공 수로 나올 수 있는가"""
    return any(round(100 * k / n, 1) == round(v, 1) for k in range(n + 1))

def classify(v, ns):
    return [n for n in ns if feasible(v, n)]

seed_cells = []
for k, (fl, dy, _) in {**T3, **T4}.items():
    for feat, arr in (("Flow", fl), ("DynaMo", dy)):
        for t, v in zip(TASKS, arr):
            seed_cells.append((k, feat, t, v))
res = {}
for k, feat, t, v in seed_cells:
    ok150 = feasible(v, 150); ok145 = feasible(v, 145); ok120 = feasible(v, 120); ok = ok150 or ok145
    res.setdefault(f"{t}/{feat}", []).append((k, v, ok150, ok145, ok120))
bad = [(key, k, v) for key, lst in res.items() for (k, v, a, b, c) in lst if not (a or b)]
RES["V2_integer_feasibility_5seed"] = {
    "cells": len(seed_cells), "infeasible_for_150_or_145": [{"cell": a, "row": b, "value": c} for a, b, c in bad],
    "toilet_flow_feasible_150": [(k, v, feasible(v, 150)) for k, f, t, v in seed_cells if t == "Toilet" and f == "Flow"],
    "toilet_flow_feasible_145": [(k, v, feasible(v, 145)) for k, f, t, v in seed_cells if t == "Toilet" and f == "Flow"],
}
# 분석적 KODex: 단일 학습, n=30 또는 29
kod = []
for k, (fl, dy, _) in {**T5}.items():
    for feat, arr in (("Flow", fl), ("DynaMo", dy)):
        for t, v in zip(TASKS, arr):
            kod.append({"row": k, "feat": feat, "task": t, "v": v, "n30": feasible(v, 30), "n29": feasible(v, 29)})
RES["V3_kodex_denominator"] = {"not_n30": [c for c in kod if not c["n30"]], "n29_only": [c for c in kod if c["n29"] and not c["n30"]],
                              "toilet_flow_all_n29": all(c["n29"] for c in kod if c["task"] == "Toilet" and c["feat"] == "Flow")}

# ---------- Fig. 3 막대 판독값 (그림 판독 값) ----------
fig3 = {  # Flow, DynaMo 평균 패널과 과제별 (판독, 오차 약 1%p)
 "Flow": {"KUBM": [72.0,74.9,67.0,45.7,83.7,88.3,82.9], "UVA": [78.7,69.5,82.0,89.6,94.2,82.0,89.6],
          "ACT20": [77.3,86.7,69.0,23.2,79.2,85.0,89.4], "Diff(2,8,16)": [92.7,60.0,67.6,0.0,92.5,81.2,86.6]},
 "DynaMo": {"KUBM": [75.5,56.5,75.5,54.4,88.6,92.0,68.8], "UVA": [78.9,69.6,81.9,89.9,94.5,81.9,89.9],
            "ACT20": [83.3,92.0,78.7,41.3,96.7,25.3,72.0], "Diff(2,8,16)": [76.0,82.7,76.7,0.0,78.7,3.3,55.3]}}
f3 = {}
for feat, d in fig3.items():
    f3[feat] = {m: round(float(np.mean(v)), 1) for m, v in d.items()}
    for m in ("UVA", "ACT20", "Diff(2,8,16)"):
        diff = np.array(d["KUBM"]) - np.array(d[m])
        f3[feat][f"KUBM_minus_{m}_per_task"] = [round(float(x), 1) for x in diff]
        f3[feat][f"KUBM_minus_{m}_wins"] = int((diff > 0).sum())
        f3[feat][f"KUBM_minus_{m}_wilcoxon_p"] = round(float(stats.wilcoxon(diff).pvalue), 3) if np.any(diff != 0) else None
RES["V4_fig3_readoff"] = f3
# 과제 평균의 표준편차(과제 간)와 5-seed 풀링 이항 SE 비교
RES["V4_binomial_se_150"] = {"p=0.73": round(100 * math.sqrt(.73 * .27 / 150), 1), "p=0.5": round(100 * math.sqrt(.25 / 150), 1)}


# ---------- 그림 4 (벡터 도형 좌표에서 읽은 막대 높이, 오차 0.1%p 이내) ----------
FIG4 = {"L-L": [26.7,12.3,42.6,0.0,57.8,56.7,2.3], "NL-L+S": [64.5,35.6,20.7,12.3,30.0,43.4,21.2],
        "NL-S MLP": [58.9,47.8,31.1,3.4,34.5,61.1,32.3], "NL-S LSTM": [62.2,60.0,27.6,22.3,66.7,65.6,48.9],
        "NL-S Transformer": [62.2,54.5,32.2,23.4,50.0,68.9,47.8], "K-UBM": [72.4,75.3,67.3,45.8,84.0,88.6,83.4]}
k5 = np.array(fig3["Flow"]["KUBM"]); k3 = np.array(FIG4["K-UBM"])
RES["V5_fig4_vs_fig3"] = {"fig4_means": {k: round(float(np.mean(v)), 1) for k, v in FIG4.items()},
    "kubm_fig4_3seeds": FIG4["K-UBM"], "kubm_fig3_5seeds_pixel_read": fig3["Flow"]["KUBM"],
    "abs_diff_per_task": [round(float(x), 1) for x in np.abs(k3 - k5)], "max_abs_diff": round(float(np.abs(k3 - k5).max()), 1),
    "kubm_minus_best_other_per_task": [round(float(FIG4["K-UBM"][i] - max(v[i] for k, v in FIG4.items() if k != "K-UBM")), 1) for i in range(7)]}

# ---------- 표 9: 반응성 ----------
SR = {"Diffusion": [23,22,13,0,12,1,16], "ACT": [24,23,11,10,9,0,10], "K-UBM": [15,17,7,15,23,17,27]}
RT = [22,27,17,30,29,30,30]
t9 = {}
for m, v in SR.items():
    rates = np.array(v) / 30
    t9[m] = {"total": sum(v), "of": 210, "rate": round(sum(v) / 210, 3), "wilson95": wilson(sum(v), 210),
             "task_mean": round(rates.mean(), 3), "task_sd0": round(rates.std(), 2), "task_sd1": round(rates.std(ddof=1), 2)}
t9["RT"] = {"total": sum(RT), "rate": round(sum(RT) / 210, 3), "task_sd0": round((np.array(RT) / 30).std(), 2), "task_sd1": round((np.array(RT) / 30).std(ddof=1), 2)}
for m in ("Diffusion", "ACT"):
    k1, k2 = sum(SR["K-UBM"]), sum(SR[m])
    t9[f"fisher_KUBM_vs_{m}"] = round(float(stats.fisher_exact([[k1, 210 - k1], [k2, 210 - k2]])[1]), 4)
    per = []
    for i, t in enumerate(TASKS):
        a, b = SR["K-UBM"][i], SR[m][i]
        per.append({"task": t, "kubm": a, "other": b, "p": round(float(stats.fisher_exact([[a, 30 - a], [b, 30 - b]])[1]), 3)})
    t9[f"per_task_{m}"] = per
# K-UBM 트리거 성공 대비 과제 성공
t9["success_given_trigger_upper"] = [round(s / r, 2) for s, r in zip(SR["K-UBM"], RT)]
# 비교용: 섭동 없는 K-UBM 성공률(Fig.3 Flow 판독) 대비
nom = np.array(fig3["Flow"]["KUBM"]) / 100 * 30
t9["kubm_nominal_flow_expected_count_readoff"] = [round(float(x), 1) for x in nom]
t9["kubm_perturbed_over_nominal"] = [round(s / n, 2) for s, n in zip(SR["K-UBM"], nom)]
RES["V5_table9"] = t9

# ---------- 표 10: 실로봇 ----------
SUC = {"Diffusion": [70, 100, 40, 80], "ACT": [100, 90, 80, 100], "K-UBM": [90, 70, 100, 80]}
t10 = {}
for m, v in SUC.items():
    k = sum(v) // 10
    t10[m] = {"mean_of_tasks": sum(v) / 4, "k_of_40_assumed_10_per_task": k, "wilson95": wilson(k, 40), "task_sd0": round(float(np.std(v)), 1), "task_sd1": round(float(np.std(v, ddof=1)), 1)}
def fish(a, b): return round(float(stats.fisher_exact([[a, 40 - a], [b, 40 - b]])[1]), 3)
t10["fisher_KUBM_vs_ACT"] = fish(34, 37); t10["fisher_KUBM_vs_Diff"] = fish(34, 29); t10["fisher_ACT_vs_Diff"] = fish(37, 29)
t10["per_task_KUBM_vs_ACT_p"] = [round(float(stats.fisher_exact([[a // 10, 10 - a // 10], [b // 10, 10 - b // 10]])[1]), 3) for a, b in zip(SUC["K-UBM"], SUC["ACT"])]
jerk = {"arm": {"Diffusion": [27.33,44.00,77.57,29.88], "ACT": [27.33,22.15,40.19,24.00], "K-UBM": [5.55,7.42,4.30,2.65]},
        "hand": {"Diffusion": [12.03,8.57,16.51,19.55], "ACT": [13.56,5.43,12.68,34.59], "K-UBM": [3.84,2.88,3.21,6.65]}}
rep_avg = {"arm": {"Diffusion": (44.69, 23.11), "ACT": (28.42, 8.14), "K-UBM": (4.98, 2.01)},
           "hand": {"Diffusion": (14.16, 4.84), "ACT": (16.56, 12.56), "K-UBM": (4.14, 1.71)}}
jr = {}
for part, d in jerk.items():
    jr[part] = {m: {"mean": round(float(np.mean(v)), 2), "reported_mean": rep_avg[part][m][0], "sd1": round(float(np.std(v, ddof=1)), 2),
                    "sd0": round(float(np.std(v)), 2), "reported_sd": rep_avg[part][m][1]} for m, v in d.items()}
    jr[part]["ratio_Diff_over_KUBM"] = round(float(np.mean(d["Diffusion"]) / np.mean(d["K-UBM"])), 2)
    jr[part]["ratio_ACT_over_KUBM"] = round(float(np.mean(d["ACT"]) / np.mean(d["K-UBM"])), 2)
    jr[part]["kubm_lower_in_tasks_vs_ACT"] = int(sum(a < b for a, b in zip(d["K-UBM"], d["ACT"])))
t10["jerk"] = jr
t10["inference_ms"] = {"table1_sim_bucket_flow": {"KUBM": 0.42, "ACT": 0.52, "Diff": 29.95}, "fig5_real": {"KUBM": 0.017, "ACT": 0.058, "Diff": 26.4},
                       "ratio_sim_over_real": {"KUBM": round(0.42 / 0.017, 1), "ACT": round(0.52 / 0.058, 1), "Diff": round(29.95 / 26.4, 2)}}
RES["V6_table10"] = t10

# ---------- 표 2: 리프팅 차원 ----------
def dims(nh, no):
    base = 2 * (nh + no); pair = comb(nh, 2)
    v1 = base + no + pair + nh; v2 = base + pair + nh; v3 = v2 + 2 * nh + 2 * no
    return v1, v2, v3
rep2 = {(28,128): (846,718,1030), (26,256): (1171,915,1479), (24,256): (1116,860,1420), (30,256): (1293,1037,1609)}
RES["V7_table2_dims"] = {f"nh={a},no={b}": {"computed": dims(a, b), "reported": rep2[(a, b)], "match": dims(a, b) == rep2[(a, b)]} for a, b in rep2}

# ---------- K-UBM 모델 크기 (공식 코드 구조 기준) ----------
def kubm_size(orig, hid=(128, 256), lift=256):
    sizes = (orig,) + hid + (lift,)
    mlp = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))
    dz = orig + lift
    return {"orig_dim": orig, "dz": dz, "mlp_params": mlp, "K_params": dz * dz, "total": mlp + dz * dz}
RES["V8_model_size"] = {"dexart_bucket_28+128": kubm_size(156), "adroit_relocate_30+256": kubm_size(286),
                      "real_23+128": kubm_size(23 + 128)}
RES["V8_flops_per_step_K_matvec"] = {"dexart": 2 * 412 ** 2, "note": "한 스텝 롤아웃은 dz x dz 행렬-벡터 곱"}

# ---------- 학습/시행 규모 ----------
RES["V9_counts"] = {"sim_test_trials_per_cell_5seeds": 150, "sim_cells": 7 * 2 * 9, "real_trials_total": 3 * 4 * 10, "reactivity_instances": 210,
                  "real_demos_per_task": 20, "real_tasks": 4}
# ---------- 스펙트럼 반경/식별 초기화: 첫 스텝 K=I 이면 z_{t+1}=z_t ----------
RES["V10_note"] = "K=I 초기화에서 K^l z = z 이므로 초기 손실은 정지 예측 손실과 같다 (식 L_K-coherence)."

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps(RES, ensure_ascii=False, indent=1, default=str)[:6000])
