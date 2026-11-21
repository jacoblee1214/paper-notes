"""PhysTwin (arXiv:2503.17973 v1) 표 재계산과 일관성 검사 (numpy, scipy). 실행: python3 analysis/recompute.py
결과: analysis/recompute_results.json

논문 표 값은 paper/paper.txt(pdftotext -layout)에서 옮겼다. 논문은 시나리오별 원자료와 SD를 주지 않으므로
모든 검사는 (a) 보고된 값끼리 맞는지, (b) 시나리오 수 n = 22를 쓴 유의성 역산(SD 상한)만 한다.
시나리오 수와 분할 규칙은 공개 저장소의 정적 읽기(data_config.csv, process_data.py)로 확인했다.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "recompute_results.json"
REPO = HERE.parent / "repo"
res = {}

# ---------------------------------------------------------------- Table 1 (재구성·재시뮬, 미래 예측)
# 열 순서: CD, Track, IoU(%), PSNR, SSIM, LPIPS. 값은 (Spring-Gaus, GS-Dynamics, PhysTwin)
METRICS = ["cd", "track", "iou", "psnr", "ssim", "lpips"]
T1 = {
    "recon": {
        "spring": [0.041, 0.050, 57.6, 23.445, 0.928, 0.102],
        "gs": [0.014, 0.022, 72.1, 26.260, 0.940, 0.052],
        "ours": [0.005, 0.009, 84.4, 28.214, 0.945, 0.034],
    },
    "future": {
        "spring": [0.062, 0.094, 46.4, 22.488, 0.924, 0.113],
        "gs": [0.041, 0.070, 49.8, 22.540, 0.924, 0.097],
        "ours": [0.012, 0.022, 72.5, 25.617, 0.941, 0.055],
    },
}
# 표 2: 처음 보는 상호작용 (GS-Dynamics, PhysTwin)
T2 = {"gs": [0.029, 0.038, 63.4, 25.053, 0.934, 0.067], "ours": [0.013, 0.018, 72.18, 26.199, 0.938, 0.047]}
# 표 3: 희소 최적화 ablation (zero-order only, first-order only, full)
T3 = {
    "recon": {
        "zero": [0.007, 0.012, 80.2, 27.409, 0.943, 0.039],
        "first": [0.008, 0.012, 82.7, 27.913, 0.944, 0.037],
        "full": [0.005, 0.009, 84.4, 28.214, 0.945, 0.034],
    },
    "future": {
        "zero": [0.014, 0.025, 69.2, 25.008, 0.938, 0.061],
        "first": [0.019, 0.034, 65.7, 24.572, 0.936, 0.067],
        "full": [0.012, 0.022, 72.5, 25.617, 0.941, 0.055],
    },
}

# 1. 표 1 full 행 = 표 3 full 행 (두 과제 모두)
match = {}
for task in ["recon", "future"]:
    match[task] = bool(np.allclose(T1[task]["ours"], T3[task]["full"]))
res["check_table1_ours_equals_table3_full"] = match

# 2. 개선량: 낮을수록 좋은 지표(CD, Track, LPIPS)는 감소율(%), IoU는 %p, PSNR은 dB, SSIM은 차이
LOWER_BETTER = {"cd", "track", "lpips"}


def improvement(base, ours, metric):
    if metric in LOWER_BETTER:
        return round(100 * (base - ours) / base, 1)  # 감소율 %
    if metric == "iou":
        return round(ours - base, 2)  # %p
    return round(ours - base, 4)  # PSNR dB, SSIM 절대차


imp = {}
for task in ["recon", "future"]:
    for ref in ["gs", "spring"]:
        imp[f"{task}_vs_{ref}"] = {
            m: improvement(T1[task][ref][i], T1[task]["ours"][i], m) for i, m in enumerate(METRICS)
        }
res["improvement_table1"] = imp
imp2 = {m: improvement(T2["gs"][i], T2["ours"][i], m) for i, m in enumerate(METRICS)}
res["improvement_table2_gs_to_ours"] = imp2

# 3. 미래 예측 대 재구성: 오차가 얼마나 커지는가 (배수)
res["future_over_recon_ratio_ours"] = {
    m: round(T1["future"]["ours"][i] / T1["recon"]["ours"][i], 2) for i, m in enumerate(METRICS)
}

# 4. 표 2(처음 보는 상호작용)와 표 1 미래 예측(같은 PhysTwin)의 비교
res["table2_ours_minus_table1_future_ours"] = {
    m: round(T2["ours"][i] - T1["future"]["ours"][i], 4) for i, m in enumerate(METRICS)
}

# 5. 표 3 ablation: zero-order만 쓸 때 오차 증가율(%) (CD, Track, LPIPS는 낮을수록 좋음)
abl = {}
for task in ["recon", "future"]:
    abl[task] = {}
    for variant in ["zero", "first"]:
        abl[task][variant] = {
            m: round(100 * (T3[task][variant][i] - T3[task]["full"][i]) / T3[task]["full"][i], 1)
            for i, m in enumerate(METRICS)
            if m in LOWER_BETTER
        }
res["ablation_error_increase_pct_vs_full"] = abl

# 6. 시나리오 수 (정적 읽기: 공개 저장소 data_config.csv)
if (REPO / "data_config.csv").exists():
    rows = [r for r in csv.reader(open(REPO / "data_config.csv", encoding="utf-8")) if r]
    n_scen = len(rows)
    n_noshape = sum(1 for r in rows if r[2].strip().lower() == "false")
    n_cloth = sum(1 for r in rows if r[1].strip() == "cloth")
    res["scenarios_from_repo_data_config"] = {
        "n": n_scen,
        "shape_prior_false": n_noshape,
        "cloth_cases": n_cloth,
        "shape_prior_true": n_scen - n_noshape,
    }
else:  # 게시본에는 repo/가 없다. 확인된 값(22, 9, 8)을 그대로 쓴다.
    res["scenarios_from_repo_data_config"] = {"n": 22, "shape_prior_false": 9, "cloth_cases": 8, "shape_prior_true": 13,
                                              "note": "repo/ 없음: 정적 읽기 결과(22행) 사용"}
# 학습·평가 분할: frame_len의 70%를 train, 나머지를 test (process_data.py 169–170행)
res["split_example"] = {"frame_len": 100, "train": int(100 * 0.7), "test": 100 - int(100 * 0.7)}

# 7. 유의성 역산: 논문 SD 미보고. n = 22 쌍 비교라고 가정하고,
#    평균 차이 d가 p < 0.05(양측 쌍체 t)가 되려면 쌍별 차이의 SD가 d * sqrt(n) / t_crit 이하여야 한다.
N = 22
tcrit = stats.t.ppf(0.975, N - 1)
sd_max = {}
for task, base_key in [("recon", "gs"), ("future", "gs")]:
    for i, m in enumerate(METRICS):
        d = abs(T1[task][base_key][i] - T1[task]["ours"][i])
        smax = d * np.sqrt(N) / tcrit
        base_mean = T1[task][base_key][i]
        sd_max[f"{task}_{m}"] = {
            "diff": round(d, 4),
            "sd_max": round(float(smax), 4),
            "sd_max_over_baseline_mean_pct": round(float(100 * smax / base_mean), 1),
        }
res["sd_max_for_p005_vs_gs_dynamics_n22"] = sd_max
res["t_crit_df21"] = round(float(tcrit), 4)

# 표 2 (n = 22 실험, 쌍 비교 가정)
sd2 = {}
for i, m in enumerate(METRICS):
    d = abs(T2["gs"][i] - T2["ours"][i])
    smax = d * np.sqrt(N) / tcrit
    sd2[m] = {"diff": round(d, 4), "sd_max": round(float(smax), 4), "sd_max_over_baseline_mean_pct": round(float(100 * smax / T2["gs"][i]), 1)}
res["sd_max_table2_n22"] = sd2

# 8. 부호 검정: 시나리오별 승패가 보고되지 않으므로, 22쌍 모두에서 이기면 p의 크기만 계산
k_needed = None
for k in range(N + 1):
    p = 2 * stats.binom.sf(k - 1, N, 0.5)  # 양측, k승 이상
    if p < 0.05:
        k_needed = k
        break
res["sign_test_n22"] = {
    "min_wins_for_two_sided_p_lt_0.05": k_needed,
    "p_if_all_22_win_two_sided": float(2 * 0.5 ** N),
}

# 9. 신뢰구간: 원자료가 없어 계산 불가. 표시만 남긴다.
res["confidence_intervals"] = "계산 불가: 시나리오별 값, SD, 실패 사례 수가 논문에 없음"

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
