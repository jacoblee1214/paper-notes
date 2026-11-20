"""WorldEval 논문 표 수치 재계산 (CPU, numpy/scipy). 실행: python3 analysis/recompute.py
출력: analysis/recompute_results.json
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# Table 1: 3 tasks (Place Cup, Strike Block, Handover Block), 값은 논문 표 1 그대로
t1 = {
    "MMRV_real2sim": [0.332, 0.266, 0.187], "MMRV_worldeval": [0.000, 0.133, 0.000],
    "r_real2sim": [0.328, 0.442, 0.465], "r_worldeval": [0.958, 0.887, 0.980],
    "avg_MMRV_real2sim_reported": 0.261, "avg_MMRV_worldeval_reported": 0.044,
    "avg_r_real2sim_reported": 0.411, "avg_r_worldeval_reported": 0.942,
}
res["table1_avg_check"] = {
    "MMRV_real2sim_mean": round(float(np.mean(t1["MMRV_real2sim"])), 4),
    "MMRV_worldeval_mean": round(float(np.mean(t1["MMRV_worldeval"])), 4),
    "r_real2sim_mean": round(float(np.mean(t1["r_real2sim"])), 4),
    "r_worldeval_mean": round(float(np.mean(t1["r_worldeval"])), 4),
}

# 정책 4개(논문 표 1 캡션·본문 기준)일 때 Pearson r의 양측 p < 0.05 임계값
n = 4
df = n - 2
tcrit = stats.t.ppf(0.975, df)
r_crit = tcrit / np.sqrt(tcrit**2 + df)
res["pearson_n4"] = {"df": df, "r_crit_p05_two_sided": round(float(r_crit), 4),
                    "sig_flags_worldeval": [bool(abs(r) > r_crit) for r in t1["r_worldeval"]],
                    "r_worldeval": t1["r_worldeval"]}

# Fisher z 95% CI for each r with n=4 (se = 1/sqrt(n-3))
ci = []
for r in t1["r_worldeval"]:
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); zc = stats.norm.ppf(0.975)
    ci.append([round(float(np.tanh(z - zc * se)), 3), round(float(np.tanh(z + zc * se)), 3)])
res["pearson_n4_fisher_ci95"] = ci

# Table 2: Policy2Vec vs. VQVAE vs. one-hot (MMRV)
p2v, vq, oh = 0.192, 0.292, 0.416
res["table2_mmrv_diffs"] = {"VQVAE_minus_P2V": round(vq - p2v, 3),
                           "onehot_minus_P2V": round(oh - p2v, 3),
                           "text_claims": {"VQVAE": 0.1, "onehot": 2.24}}

# Table 3: 성공률 k/40 정수 확인, Wilson 95% CI, 체크포인트 간 Fisher 정확검정
N = 40
t3 = {
    "pi0_taskB": {"5k": 0.075, "30k": 0.150, "final60k": 0.600},
    "pi0_taskA": {"5k": 0.725, "30k": 0.775, "final60k": 0.925},
    "pi0_taskC": {"5k": 0.300, "30k": 0.450, "final60k": 0.750},
    "dexvla_taskA_final": 0.125, "dexvla_taskB_final": 0.250, "dexvla_taskC_final": 0.450,
}
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(float(c - h), 3), round(float(c + h), 3)

def as_count(v):
    k = round(v * N)
    assert abs(k / N - v) < 1e-9, v
    return k

integer_ok = {}
for key, val in t3.items():
    if isinstance(val, dict):
        integer_ok[key] = {s: as_count(v) for s, v in val.items()}
    else:
        integer_ok[key] = as_count(val)
res["table3_counts_k_of_40"] = integer_ok
res["table3_wilson95_pi0_final_taskB"] = wilson(24, N)
res["table3_wilson95_pi0_5k_taskB"] = wilson(3, N)
# 체크포인트 5k vs 최종 (Task B): 3/40 vs 24/40
tab = [[3, 37], [24, 16]]
res["table3_fisher_pi0_taskB_5k_vs_final"] = {"p_two_sided": float(stats.fisher_exact(tab)[1])}
# 체크포인트 5k vs 30k (Task A): 29/40 vs 31/40
res["table3_fisher_pi0_taskA_5k_vs_30k"] = {"p_two_sided": float(stats.fisher_exact([[29, 11], [31, 9]])[1])}
# 같은 체크포인트(final)에서 π0 vs DexVLA, Task B: 24/40 vs 10/40
res["table3_fisher_pi0_vs_dexvla_final_taskB"] = {"p_two_sided": float(stats.fisher_exact([[24, 16], [10, 30]])[1])}

# 실로봇 표본 크기: 40 rollouts x 5 tasks x 4 policies = 800, 논문은 "over 1,000 trials"라고 씀
res["trial_count_naive"] = {"main_table_40x5x4": 40 * 5 * 4, "paper_claim": ">1000"}

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))
