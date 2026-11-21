"""pi0 (arXiv:2410.24164 v4) 논문 수치의 재계산과 일관성 검사.

논문의 결과 그림(Fig. 7, 9, 11, 13)은 막대와 선만 있고 값 표가 없어서, 아래 판독값은 300 dpi 렌더링을
눈으로 읽은 근사치입니다(±0.01 안팎). 판독값은 이 파일에 상수로 적어 두었고, 출처 그림을 주석에 적었습니다.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
필요: numpy, scipy (CPU, 1초 안팎)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
results = {}

# ---------------------------------------------------------------------------
# 1. Fig. 7 (out-of-box, 10 trials/task) 판독값과 객체 단위 Fisher 검정
#    점수의 분모: 과제별 객체(또는 항목) 수 x 10 trials. 객체가 독립이라고 가정한 상한(낙관적) 검정이다.
# ---------------------------------------------------------------------------
fig7 = {
    # task: (points per trial, {method: read score})
    "Shirt folding (Bi-ARX)": (1, {"pi0": 1.00, "parity": 0.90, "small": 0.50}),
    "Bussing easy (UR5e)": (7, {"pi0": 0.97, "parity": 0.81, "small": 0.45}),
    "Bussing hard (Bi-Trossen)": (12, {"pi0": 0.875, "parity": 0.50, "small": 0.335}),
    "Grocery bagging (UR5e)": (7, {"pi0": 0.785, "parity": 0.345, "small": 0.275}),
    "Toast (UR5e)": (4, {"pi0": 0.75, "parity": 0.40}),
}
N_TRIALS = 10


def fisher_pi0_vs(pts, read_pi0, read_other):
    n = pts * N_TRIALS
    k1 = int(round(read_pi0 * n))
    k2 = int(round(read_other * n))
    table = [[k1, n - k1], [k2, n - k2]]
    _, p = stats.fisher_exact(table)
    return {"n_each": n, "k_pi0": k1, "k_other": k2, "diff_pp": round(100 * (k1 - k2) / n, 1), "fisher_p": float(p)}


fig7_tests = {}
for task, (pts, reads) in fig7.items():
    row = {}
    for other in [m for m in reads if m != "pi0"]:
        row[other] = fisher_pi0_vs(pts, reads["pi0"], reads[other])
    if pts == 1:
        # 이진 판정(10 evals). 같은 방식으로 검정
        row["note"] = "shirt folding is binary per eval (10 evals); object count = 10"
    fig7_tests[task] = row
results["fig7_object_level_fisher"] = fig7_tests

# Wilson 95% 구간 (π0 의 과제별 판독 점수를 객체 단위로 본 경우)
def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)

fig7_ci = {}
for task, (pts, reads) in fig7.items():
    n = pts * N_TRIALS
    k = int(round(reads["pi0"] * n))
    fig7_ci[task] = {"k": k, "n": n, "point": round(k / n, 3), "wilson95": [round(x, 3) for x in wilson(k, n)]}
results["fig7_pi0_wilson_object_level"] = fig7_ci

# 10 trial 평균의 정밀도 한계: 이진 과제(셔츠)에서 10번 시도의 95% 구간 폭
results["shirt_folding_10_evals_wilson"] = {f"k={k}": [round(x, 3) for x in wilson(k, 10)] for k in (9, 10)}

# ---------------------------------------------------------------------------
# 2. Fig. 13 (post-training, 10 trials) 판독값: scratch 대 fine-tuned
# ---------------------------------------------------------------------------
fig13 = {
    # task: (points per trial, fine-tuned, scratch, out-of-box)
    "Laundry folding": (4, 0.82, 0.22, 0.30),
    "Table bussing": (12, 0.88, 0.71, 0.48),
    "Mobile laundry": (4, 0.93, 0.32, 0.55),
    "Dryer unloading": (5, 0.72, 0.38, 0.16),
    "Box building": (5, 0.64, 0.32, 0.005),
    "To-go box": (5, 0.70, 0.46, 0.005),
    "Packing eggs": (7, 0.845, 0.885, 0.005),
}
fig13_out = {}
for task, (pts, ft, sc, oob) in fig13.items():
    n = pts * N_TRIALS
    k_ft, k_sc = int(round(ft * n)), int(round(sc * n))
    table = [[k_ft, n - k_ft], [k_sc, n - k_sc]]
    _, p = stats.fisher_exact(table)
    fig13_out[task] = {
        "ft": ft, "scratch": sc, "out_of_box": oob, "n_points": n,
        "ft_minus_scratch_pp": round(100 * (ft - sc), 1), "fisher_p_object_level": round(float(p), 3),
        "max_score_ft_ge_0_5": ft > 0.5,
    }
results["fig13_points"] = fig13_out
results["fig13_all_ft_above_half"] = all(v[1] > 0.5 for v in fig13.values())
results["fig13_scratch_beats_ft"] = [t for t, v in fig13.items() if v[2] > v[1]]

# ---------------------------------------------------------------------------
# 3. Fig. 11 판독값과 텍스트 대조 (Tupperware 5시간, Paper towel 추세)
# ---------------------------------------------------------------------------
fig11 = {
    "Tupperware in microwave": {
        "hours": [1, 5, 10],
        "pi0": [0.87, 0.57, 0.76],
        "pi0_scratch": [0.67, 0.75, 0.77],
        "DP": [0.65, 0.62, 0.67],
        "ACT": [0.53, 0.63, 0.73],
    },
    "Paper towel replacement": {
        "hours": [1, 5, 10],
        "pi0": [0.82, 0.80, 0.70],
        "pi0_scratch": [0.37, 0.57, 0.57],
        "DP": [0.23, 0.33, 0.30],
    },
    "Average across all tasks": {
        "hours": [1, 5, 10],
        "pi0": [0.57, 0.86, 0.87],
        "pi0_scratch": [0.38, 0.78, 0.76],
        "DP": [0.20, 0.32, 0.38],
    },
}
tup = fig11["Tupperware in microwave"]
results["fig11_tupperware_5h_pi0_below_all_baselines"] = all(
    tup["pi0"][1] < tup[k][1] for k in ("pi0_scratch", "DP", "ACT"))
results["fig11_tupperware_10h_pi0_minus_scratch"] = round(tup["pi0"][2] - tup["pi0_scratch"][2], 2)
results["fig11_paper_towel_pi0_change_1h_to_10h"] = round(
    fig11["Paper towel replacement"]["pi0"][2] - fig11["Paper towel replacement"]["pi0"][0], 2)
results["fig11_points"] = fig11

# ---------------------------------------------------------------------------
# 4. Table I (추론 시간, RTX 4090) 합계 검사
# ---------------------------------------------------------------------------
t1 = {"image_encoders": 14, "observation_forward": 32, "flow_x10": 27, "network_latency": 13}
onboard = t1["image_encoders"] + t1["observation_forward"] + t1["flow_x10"]
results["table1"] = {
    "onboard_sum_ms": onboard, "paper_onboard_ms": 73, "match_onboard": onboard == 73,
    "offboard_sum_ms": onboard + t1["network_latency"], "paper_offboard_ms": 86,
    "match_offboard": onboard + t1["network_latency"] == 86,
    "per_flow_step_ms": round(t1["flow_x10"] / 10, 2),
    "chunk_budget_50hz_ms": round(25 / 50 * 1000), "chunk_budget_20hz_ms": round(16 / 20 * 1000),
    "onboard_fraction_of_50hz_chunk": round(onboard / 500, 3),
}

# ---------------------------------------------------------------------------
# 5. 데이터 규모: 903M 스텝 대 10,000시간, 9.1% 비율
# ---------------------------------------------------------------------------
own_steps = 903e6
results["data_scale"] = {
    "single_arm_plus_dual_arm": 106e6 + 797e6,
    "own_steps_equal_903M": (106e6 + 797e6) == own_steps,
    "hours_if_all_50hz": round(own_steps / 50 / 3600),
    "hours_if_all_20hz": round(own_steps / 20 / 3600),
    "implied_mean_hz_for_10000h": round(own_steps / (10000 * 3600), 1),
    "total_steps_if_open_source_is_9_1pct": round(own_steps / (1 - 0.091) / 1e6),
    "open_source_steps_implied_M": round(own_steps / (1 - 0.091) * 0.091 / 1e6),
}

# ---------------------------------------------------------------------------
# 6. 파라미터 수 산술 (Gemma 설정, 코드 gemma.py의 값)
# ---------------------------------------------------------------------------
def gemma_params(width, depth, mlp, heads, kv, head_dim, vocab=0):
    attn = width * heads * head_dim + 2 * width * kv * head_dim + heads * head_dim * width
    mlp_p = 3 * width * mlp  # GeGLU: gate, up, down
    return depth * (attn + mlp_p) + vocab * width

expert = gemma_params(1024, 18, 4096, 8, 1, 256)
gemma2b = gemma_params(2048, 18, 16384, 8, 1, 256, vocab=256_000)
results["param_count"] = {
    "action_expert_M_layers_only": round(expert / 1e6, 1),
    "code_comment_says_M": 311,
    "gemma_2b_B_with_embedding": round(gemma2b / 1e9, 2),
    "paligemma_B_approx_gemma2b_plus_siglip400m": round((gemma2b + 0.4e9) / 1e9, 2),
    "pi0_total_B_approx": round((gemma2b + 0.4e9 + expert) / 1e9, 2),
    "paper_total_B": 3.3,
}

# ---------------------------------------------------------------------------
# 7. 시간 표본 변환: 논문 식 (Appendix B)과 코드(pi0.py sample/time)의 대응 확인
#    코드: t = 0.999 x + 0.001, x ~ Beta(1.5, 1), t = 1 - tau
#    논문: u = (s - tau)/s ~ Beta(1.5, 1), s = 0.999, tau = 1 - t
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
x = rng.beta(1.5, 1.0, 200_000)
t_code = 0.999 * x + 0.001
tau = 1 - t_code
u = (0.999 - tau) / 0.999
ks = stats.kstest(u, "beta", args=(1.5, 1.0))
results["timestep_mapping_ks"] = {"statistic": round(float(ks.statistic), 5), "p": round(float(ks.pvalue), 3),
                                  "note": "code t=1-tau reproduces paper's Beta((s-tau)/s;1.5,1) if p is not small"}
results["timestep_range_check"] = {"tau_min": round(float(tau.min()), 4), "tau_max": round(float(tau.max()), 4)}

# Euler 스텝: 10 steps, dt = 0.1 (paper) ; code dt = -1/10 in t convention
results["euler_steps"] = {"paper_delta": 0.1, "code_dt": -0.1, "steps": 10}

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
