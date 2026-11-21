"""Octo (arXiv:2405.12213 v2) 표 재계산과 일관성 검사 (numpy, scipy). 실행: python3 analysis/recompute.py
결과: analysis/recompute_results.json
논문 표 값은 paper/paper.txt(pdftotext -layout)에서 옮겼다. 원자료(시도별 기록)가 없으므로
검사는 '보고된 성공률과 시도 수'로만 한다. 정책 간 검정은 모두 두 집단 Fisher 정확검정(양측)이다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


# ---------------------------------------------------------------- Table I (finetuning, 6 domains, 20 trials each)
DOM = ["Berkeley Insertion*", "Stanford Coffee", "CMU Baking", "Berkeley Pick-Up†", "Berkeley Coke", "Berkeley Bimanual†"]
T1 = {
    "Scratch": [10, 45, 25, 0, 20, 20],
    "VC-1": [5, 0, 30, 0, 10, 50],
    "Octo": [70, 75, 50, 60, 100, 80],
}
REPORTED_AVG = {"Scratch": 20, "VC-1": 15, "Octo": 72}
N_T1 = 20  # 표 I 캡션: 도메인마다 20회. 단, 부록 F는 Berkeley Bimanual을 10회로 적음 (V3)
t1 = {"avg_check": {}, "per_domain": {}}
for m, v in T1.items():
    t1["avg_check"][m] = {"mean_computed": round(float(np.mean(v)), 2), "reported": REPORTED_AVG[m]}
t1["octo_minus_best_baseline_pp_avg"] = round(np.mean(T1["Octo"]) - max(np.mean(T1["Scratch"]), np.mean(T1["VC-1"])), 2)
for i, d in enumerate(DOM):
    k_o = round(T1["Octo"][i] / 100 * N_T1)
    best = max(("Scratch", "VC-1"), key=lambda m: T1[m][i])
    k_b = round(T1[best][i] / 100 * N_T1)
    tab = [[k_o, N_T1 - k_o], [k_b, N_T1 - k_b]]
    p = stats.fisher_exact(tab, alternative="two-sided")[1]
    t1["per_domain"][d] = {
        "octo": f"{k_o}/{N_T1}", "octo_wilson95": wilson(k_o, N_T1),
        "best_baseline": best, "baseline": f"{k_b}/{N_T1}", "fisher_p_two_sided": round(float(p), 4),
    }
# 쌍체 부호 검정 느낌의 요약: Octo가 최고 기준선보다 높은 도메인 수
wins = sum(1 for i in range(6) if T1["Octo"][i] > max(T1["Scratch"][i], T1["VC-1"][i]))
t1["octo_beats_best_baseline_in_domains"] = f"{wins}/6"
# 6개 도메인 평균 차이의 대략 신뢰구간 (도메인을 단위로 본 t, 참고용)
diff = np.array(T1["Octo"]) - np.array([max(a, b) for a, b in zip(T1["Scratch"], T1["VC-1"])])
tt = stats.ttest_1samp(diff, 0)
t1["paired_t_over_domains_vs_best_baseline"] = {"mean_diff_pp": round(float(diff.mean()), 2), "t": round(float(tt.statistic), 2), "p": round(float(tt.pvalue), 4), "n_domains": 6}
res["table_I"] = t1

# ---------------------------------------------------------------- Table VI (= Table II, WidowX ablation, 4 tasks x 10 trials)
TASKS = ["carrot", "eggplant", "bread", "spoon-glove"]
T6 = {
    "Octo-small": [80, 90, 70, 90],
    "RT-X mix": [80, 80, 40, 40],
    "Bridge only": [20, 70, 60, 20],
    "Discretized": [0, 20, 10, 40],
    "MSE head": [70, 30, 0, 40],
    "ResNet+Transformer": [80, 60, 100, 40],
}
REP_T6 = {"Octo-small": 83, "RT-X mix": 60, "Bridge only": 43, "Discretized": 18, "MSE head": 35, "ResNet+Transformer": 70}
N_PER_TASK = 10
t6 = {"rows": {}}
for m, v in T6.items():
    kk = sum(x * N_PER_TASK // 100 for x in v)
    t6["rows"][m] = {"mean_computed": round(float(np.mean(v)), 2), "reported_avg": REP_T6[m],
                     "successes_of_40": kk, "wilson95": wilson(kk, 40)}
base_k = t6["rows"]["Octo-small"]["successes_of_40"]
t6["fisher_vs_octo"] = {}
for m in T6:
    if m == "Octo-small":
        continue
    k = t6["rows"][m]["successes_of_40"]
    p = stats.fisher_exact([[base_k, 40 - base_k], [k, 40 - k]])[1]
    t6["fisher_vs_octo"][m] = round(float(p), 4)
res["table_VI_II"] = t6

# ---------------------------------------------------------------- Table VII (generalization, WidowX, 10 trials per task)
T7 = {"In-distribution": [80, 90], "Novel objects": [70, 90], "Novel environment": [20, 60], "Novel skill": [10, 0]}
REP_T7 = {"In-distribution": 85, "Novel objects": 80, "Novel environment": 40, "Novel skill": 5}
t7 = {}
for g, v in T7.items():
    ks = [x // 10 for x in v]
    t7[g] = {"mean_computed": float(np.mean(v)), "reported": REP_T7[g], "successes_of_20": sum(ks), "wilson95": wilson(sum(ks), 20)}
t7["note"] = "과제당 10회. 집계 20회(두 과제)."
t7["fisher_in_dist_vs_novel_skill_p"] = float(stats.fisher_exact([[17, 3], [1, 19]])[1])
res["table_VII"] = t7

# ---------------------------------------------------------------- Appendix C / Table III mixture weights (v2)
W = [17.0, 17.0, 17.0, 9.1, 6.0, 5.9, 3.6, 3.3, 3.0, 2.9, 2.8, 2.4, 2.3, 1.5, 1.2, 1.2, 1.0, 0.9, 0.05, 0.6, 0.3, 0.3, 0.2, 0.1, 0.1]
res["table_III"] = {"n_datasets": len(W), "sum_pct_reported_values": round(sum(W), 2), "note": "UCSD Kitchen '<0.1%'는 0.05로 넣음. 반올림 오차 최대 ±1.25%p"}

# ---------------------------------------------------------------- Compute and size consistency (paper's own numbers)
steps, batch, hours = 300_000, 2048, 14
samples = steps * batch
res["compute"] = {
    "samples_total_M": round(samples / 1e6, 1),
    "samples_per_sec_all_chips": round(samples / (hours * 3600)),
    "samples_per_sec_per_chip_v4_128": round(samples / (hours * 3600) / 128, 1),
    "note": "논문 부록 D: 300k 스텝, 배치 2048, TPU v4-128, 14시간. 공개 config 배치는 512 (V4)",
}
# 파라미터 대략치: 트랜스포머 블록만 12·d²·L (MLP 4d, 어텐션 4d²) — 토크나이저, 헤드, 임베딩 제외
res["params_rough"] = {
    "octo_small_block_only_M": round(12 * 384 ** 2 * 12 / 1e6, 1), "octo_small_reported_M": 27,
    "octo_base_block_only_M": round(12 * 768 ** 2 * 12 / 1e6, 1), "octo_base_reported_M": 93,
    "note": "블록만 세면 Small 21.2M, Base 84.9M. 차이는 토크나이저와 헤드 등(추정)",
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps(res, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:6000])
