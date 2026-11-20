"""WorldArena (arXiv:2602.08971 v2) 표 재계산과 일관성 검사 (numpy, scipy).
실행: python3 analysis/recompute.py   결과: analysis/recompute_results.json

논문 표 값은 paper/paper.pdf를 pdftotext -layout으로 읽어 옮겼다 (Table 2, 3, 4, 5, 6, Figure 1(a) 막대 라벨).
원자료(영상별 점수, 사람 평가 점수)는 논문에 없으므로 모든 검사는 '보고된 값끼리' 맞는지를 본다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

MODELS = [
    "GigaWorld-0", "Genie Envisioner", "TesserAct", "RoboMaster", "Vidar",
    "Cosmos-Predict 2.5 (text)", "Cosmos-Predict 2.5 (action)", "WoW", "CtrlWorld",
    "Wan 2.2", "CogVideoX", "IRASim", "Veo 3.1", "Wan 2.6",
]
GENERAL = {"Wan 2.2", "CogVideoX", "Veo 3.1", "Wan 2.6"}  # 논문 4.1: 일반 목적 video 모델

# ---------------------------------------------------------------- Table 2 (16개 중 9개)
METRICS_16 = [
    "Image Quality", "Aesthetic Quality", "JEPA Similarity",          # Visual quality
    "Dynamic Degree", "Flow Score", "Motion Smoothness",             # Motion quality (표 2)
    "Subject Consist.", "Background Consist.", "Photometric Consist.",  # Content consistency (표 2)
    "Interaction Quality", "Trajectory Acc.",                        # Physics adherence (표 3)
    "Depth Acc.", "Perspectivity",                                   # 3D accuracy (표 3)
    "Instruction Following", "Semantic Alignment", "Action Following",  # Controllability (표 3)
]
T2 = np.array([
    [0.5041, 0.3991, 0.4413, 0.6709, 0.3118, 0.7811, 0.7303, 0.8563, 0.1756],
    [0.2305, 0.3289, 0.3340, 0.6930, 0.0855, 0.6966, 0.7760, 0.9024, 0.2006],
    [0.3322, 0.4590, 0.4579, 0.5150, 0.2447, 0.7579, 0.8250, 0.9238, 0.2491],
    [0.3487, 0.3842, 0.2966, 0.6124, 0.1484, 0.6940, 0.8295, 0.9123, 0.3356],
    [0.4145, 0.4068, 0.5608, 0.2767, 0.1426, 0.7973, 0.7629, 0.8300, 0.2350],
    [0.6668, 0.4501, 0.3126, 0.5911, 0.4302, 0.7882, 0.7488, 0.8511, 0.1383],
    [0.4489, 0.3576, 0.9296, 0.3994, 0.0573, 0.7100, 0.8197, 0.8894, 0.3528],
    [0.4587, 0.3868, 0.7440, 0.4608, 0.2706, 0.7692, 0.8161, 0.9025, 0.2170],
    [0.3522, 0.3893, 0.9185, 0.4257, 0.3449, 0.7377, 0.8411, 0.9057, 0.1729],
    [0.3884, 0.3963, 0.7575, 0.4349, 0.1269, 0.7019, 0.8388, 0.9042, 0.4776],
    [0.3582, 0.3777, 0.9384, 0.3166, 0.2189, 0.7391, 0.8083, 0.8773, 0.3580],
    [0.3489, 0.3623, 0.9330, 0.4139, 0.2083, 0.7052, 0.8312, 0.9068, 0.3522],
    [0.6605, 0.4632, 0.5694, 0.5450, 0.1396, 0.6989, 0.7878, 0.8710, 0.3247],
    [0.6824, 0.4433, 0.7229, 0.7421, 0.4532, 0.8539, 0.7517, 0.8687, 0.1904],
])
# ---------------------------------------------------------------- Table 3 (나머지 7개)
T3 = np.array([
    [0.5368, 0.1552, 0.6316, 0.7596, 0.6156, 0.8591, 0.1134],
    [0.2052, 0.0679, 0.8663, 0.5284, 0.2028, 0.8544, 0.0109],
    [0.5800, 0.1396, 0.7159, 0.7920, 0.6152, 0.8783, 0.0311],
    [0.5364, 0.1158, 0.8335, 0.7588, 0.5772, 0.8761, 0.0352],
    [0.5348, 0.1928, 0.7872, 0.7592, 0.5912, 0.8826, 0.0819],
    [0.3872, 0.0816, 0.7051, 0.7964, 0.2664, 0.7733, 0.1418],
    [0.5500, 0.2945, 0.8862, 0.7644, 0.5840, 0.8879, 0.0133],
    [0.5564, 0.2058, 0.7283, 0.7672, 0.5692, 0.8842, 0.0434],
    [0.6212, 0.4766, 0.9300, 0.7960, 0.7272, 0.8912, 0.0210],
    [0.5184, 0.1627, 0.7768, 0.7660, 0.5376, 0.8877, 0.0512],
    [0.5940, 0.3526, 0.9097, 0.7828, 0.7268, 0.8977, 0.0076],
    [0.5656, 0.3639, 0.9312, 0.7788, 0.6604, 0.8849, 0.0526],
    [0.7872, 0.1231, 0.7421, 0.8276, 0.9328, 0.8607, 0.0852],
    [0.7280, 0.1182, 0.7144, 0.8032, 0.8536, 0.8728, 0.0992],
])
ALL16 = np.hstack([T2, T3])  # 14 x 16, 열 순서는 METRICS_16
assert ALL16.shape == (14, 16)
COL = {m: i for i, m in enumerate(METRICS_16)}

# ---------------------------------------------------------------- Figure 1(a) EWMScore 막대 라벨 (텍스트 층, 내림차순)
# 라벨 순서와 막대 값을 내림차순으로 짝지었다 (figure 텍스트 층 기준, 시각 확인은 하지 않음).
FIG1_EWM = {
    "Wan 2.6": 61.86, "CtrlWorld": 59.70, "Veo 3.1": 58.87, "IRASim": 58.11,
    "CogVideoX": 57.88, "Cosmos-Predict 2.5 (action)": 55.90, "WoW": 54.88,
    "Wan 2.2": 54.54, "GigaWorld-0": 53.39, "TesserAct": 53.23,
    "RoboMaster": 51.84, "Vidar": 51.60, "Cosmos-Predict 2.5 (text)": 50.81,
    "Genie Envisioner": 43.65,
}

# ---------------------------------------------------------------- Table 4, 5 (성공률 %, 각 100회)
T4 = {  # data engine: (Task1 adjust bottle, Task2 click bell)
    "Genie Envisioner": (7, 21), "TesserAct": (1, 35), "RoboMaster": (7, 68),
    "Vidar": (13, 53), "WoW": (45, 71), "Wan 2.2": (15, 41),
}
T4_REAL = (77, 66)
T4_ZERO = (2, 5)
T5 = {  # action planner
    "Genie Envisioner": (10, 20), "TesserAct": (1, 35), "RoboMaster": (8, 20),
    "Vidar": (2, 19), "WoW": (20, 21), "Wan 2.2": (12, 20),
}
T5_PI05 = (77, 66)
N_TRIALS = 100  # 논문 Table 4 캡션: 각 과제 100회

# ================================================================ 1. EWMScore 재계산
mean16 = ALL16.mean(axis=1) * 100
ewm_check = {}
for i, m in enumerate(MODELS):
    if m in FIG1_EWM:
        ewm_check[m] = {"mean16_x100": round(mean16[i], 2), "fig1": FIG1_EWM[m],
                        "diff": round(mean16[i] - FIG1_EWM[m], 2)}
diffs = [v["diff"] for v in ewm_check.values()]
res["ewmscore_recompute"] = {
    "per_model": ewm_check,
    "max_abs_diff": round(max(abs(d) for d in diffs), 2),
    "mean_abs_diff": round(float(np.mean(np.abs(diffs))), 2),
    "pearson_mean16_vs_fig1": round(stats.pearsonr([ewm_check[m]["mean16_x100"] for m in ewm_check],
                                                   [ewm_check[m]["fig1"] for m in ewm_check])[0], 4),
    "note": "Table 2·3 값을 16개 단순 평균 x100 한 값이 Figure 1(a) 막대와 얼마나 맞는지 본다. "
            "정규화 bound(Table 6)는 표 값에 이미 적용됐다고 가정.",
}

# ================================================================ 2. 일반 모델 vs 임베디드 모델 평균 (표 2·3)
gen_idx = [i for i, m in enumerate(MODELS) if m in GENERAL]
emb_idx = [i for i, m in enumerate(MODELS) if m not in GENERAL]
group = {}
for met in METRICS_16:
    j = COL[met]
    g = ALL16[gen_idx, j].mean()
    e = ALL16[emb_idx, j].mean()
    group[met] = {"general_mean": round(float(g), 4), "embodied_mean": round(float(e), 4),
                  "diff_emb_minus_gen": round(float(e - g), 4)}
res["group_means"] = group

# ================================================================ 3. 본문 주장 vs 표 순위 (C 항목)
def rank_of(met, model):
    j = COL[met]
    vals = ALL16[:, j]
    order = np.argsort(-vals)
    pos = [MODELS[k] for k in order].index(model) + 1
    return {"value": float(ALL16[MODELS.index(model), j]), "rank_of_14": pos,
            "top": MODELS[order[0]], "top_value": float(vals[order[0]])}

claims = {
    "C1_TesserAct_trajectory_high (4.2.1)": rank_of("Trajectory Acc.", "TesserAct"),
    "C2_WoW_action_following_strong (4.2.1)": rank_of("Action Following", "WoW"),
    "C3_CogVideoX_visual_quality_excels (4.2.1)": rank_of("Image Quality", "CogVideoX"),
    "C4_CogVideoX_lags_physics (4.2.1, interaction)": rank_of("Interaction Quality", "CogVideoX"),
    "C5_CtrlWorld_trajectory_high (4.2.1)": rank_of("Trajectory Acc.", "CtrlWorld"),
    "C6_Veo31_limited_embodied_gain (4.2.1, interaction)": rank_of("Interaction Quality", "Veo 3.1"),
    "C7_Veo31_Wan26_top_visual (4.2.1, image quality)": rank_of("Image Quality", "Wan 2.6"),
}
res["claim_vs_table_ranks"] = claims

# Action Following 값 범위 vs Figure 11 사례 값 (12.01 / 0.47)
res["action_following_range"] = {"min": float(ALL16[:, COL["Action Following"]].min()),
                                 "max": float(ALL16[:, COL["Action Following"]].max()),
                                 "figure11_example_high_low": [12.01, 0.47]}

# ================================================================ 4. Table 4·5 비율 통계 (n=100)
def wilson(p_pct, n=N_TRIALS, z=1.96):
    p = p_pct / 100
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(center - half, 3), round(center + half, 3)

def two_prop_p(a_pct, b_pct, n=N_TRIALS):
    a, b = round(a_pct * n / 100), round(b_pct * n / 100)
    table = [[a, n - a], [b, n - b]]
    return float(stats.fisher_exact(table)[1])

stat4 = {}
for name, (t1, t2) in T4.items():
    stat4[name] = {
        "task1": t1, "task2": t2,
        "task1_ci95": wilson(t1), "task2_ci95": wilson(t2),
        "task1_p_vs_real_77": round(two_prop_p(t1, T4_REAL[0]), 6),
        "task2_p_vs_real_66": round(two_prop_p(t2, T4_REAL[1]), 6),
        "task2_diff_pp_vs_real": t2 - T4_REAL[1],
    }
res["table4_stats"] = {"real_data_ci95": {"task1": wilson(T4_REAL[0]), "task2": wilson(T4_REAL[1])},
                       "rows": stat4,
                       "note": "Fisher exact, 두 비율 모두 n=100 가정. 시드 반복이 없어 학습 분산은 빠져 있다."}

# TesserAct 행이 Table 4와 Table 5에서 같은지 (복사 오류 가능성)
res["tesseract_identical_t4_t5"] = (T4["TesserAct"] == T5["TesserAct"])

# ================================================================ 5. 평균 성공률과 EWMScore 상관 (Figure 5 재현 시도)
six = list(T4.keys())  # Figure 5 라벨과 같은 6개 모델
ewm6 = np.array([FIG1_EWM[m] for m in six])
avg_t4 = np.array([np.mean(T4[m]) for m in six])
avg_t5 = np.array([np.mean(T5[m]) for m in six])
r4, p4 = stats.pearsonr(ewm6, avg_t4)
r5, p5 = stats.pearsonr(ewm6, avg_t5)
r4_t1, _ = stats.pearsonr(ewm6, np.array([T4[m][0] for m in six]))
r4_t2, _ = stats.pearsonr(ewm6, np.array([T4[m][1] for m in six]))
res["ewm_vs_tasks_n6"] = {
    "models": six,
    "data_engine_avg_r": [round(r4, 3), round(p4, 4)],
    "data_engine_avg_r_paper": 0.600,
    "action_planner_avg_r": [round(r5, 3), round(p5, 4)],
    "action_planner_avg_r_paper": 0.360,
    "data_engine_task1_r": round(r4_t1, 3),
    "data_engine_task2_r": round(r4_t2, 3),
    "critical_r_n6_alpha05": round(float(stats.t.ppf(0.975, 4) / math.sqrt(4 + stats.t.ppf(0.975, 4) ** 2)), 3),
    "note": "논문 r=0.600, 0.360과 같은지 본다. 평균 방식(Task1·Task2 단순 평균)은 논문에 명시되지 않아 가정. "
            "EWMScore 값은 Figure 1(a) 라벨에서 옮긴 값이다.",
}

# ================================================================ 6. 정책 평가자 (n=5 정책, Figure 4)
def fisher_ci(r, n, z=1.96):
    zr = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return round(math.tanh(zr - z * se), 3), round(math.tanh(zr + z * se), 3)

n_pol = 5
for label, r in [("ctrlworld", 0.986), ("cosmos_predict", 0.483)]:
    t = r * math.sqrt(n_pol - 2) / math.sqrt(1 - r * r)
    p = 2 * (1 - stats.t.cdf(abs(t), n_pol - 2))
    res.setdefault("policy_evaluator", {})[label] = {
        "pearson_r_paper": r, "n": n_pol, "p_two_sided": round(float(p), 4),
        "fisher_ci95": fisher_ci(r, n_pol),
    }
zdiff = (math.atanh(0.986) - math.atanh(0.483)) / math.sqrt(2 / (n_pol - 3))
res["policy_evaluator"]["z_diff_independent_approx"] = {
    "z": round(zdiff, 3), "p_two_sided": round(float(2 * (1 - stats.norm.cdf(abs(zdiff)))), 4),
    "note": "두 상관이 같은 시뮬레이터 점수를 공유하므로 독립 가정은 근사다.",
}
# 코드 기본 시뮬레이터 점수 (calculate_policy_pearson_r.py DEFAULT_SIMULATOR_SCORES)
res["policy_evaluator"]["code_default_sim_scores"] = {
    "10data": 28.6, "20data": 34.58, "30data": 37.78, "50data": 43.52, "fulldata": 46.8}

# ================================================================ 7. 사람 평가 상관 (n 미보고)
res["human_eval"] = {
    "pearson_paper": 0.825,
    "n_models_reported": 14,  # 그림 5 왼쪽 산점도의 점 개수를 눈으로 셈 (본문에는 n 없음)
    "p_if_n14": float(2 * (1 - stats.t.cdf(abs(0.825 * math.sqrt(12) / math.sqrt(1 - 0.825 ** 2)), 12))),
    "note": "표 값이 없어 r 자체는 재계산 불가. 그림 5의 점 14개를 근거로 n=14로 두면 p<0.001이다.",
}

# ================================================================ 저장
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "ewm_max_abs_diff": res["ewmscore_recompute"]["max_abs_diff"],
    "ewm_mean_abs_diff": res["ewmscore_recompute"]["mean_abs_diff"],
    "ewm_pearson_fig1": res["ewmscore_recompute"]["pearson_mean16_vs_fig1"],
    "ewm_vs_tasks": {k: res["ewm_vs_tasks_n6"][k] for k in
                     ["data_engine_avg_r", "action_planner_avg_r", "data_engine_task1_r", "data_engine_task2_r", "critical_r_n6_alpha05"]},
    "tesseract_identical": res["tesseract_identical_t4_t5"],
    "policy": {k: res["policy_evaluator"][k] for k in ["ctrlworld", "cosmos_predict"]},
    "z_diff": res["policy_evaluator"]["z_diff_independent_approx"],
    "t2_rows": {k: (v["task2"], v["task2_ci95"], v["task2_p_vs_real_66"]) for k, v in stat4.items()},
}, ensure_ascii=False, indent=2))
