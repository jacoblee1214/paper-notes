"""DreamDojo (arXiv:2602.06949v1) 보고 수치의 재계산과 일관성 검사.

표준 라이브러리만 사용한다. 결과는 recompute_results.json에 저장한다.
입력은 논문 본문과 표에서 옮긴 값이며, 원자료(샘플 단위 점수)는 공개되지 않았다.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binom_p_two_sided(k, n):
    # 정규 근사 (연속성 보정 없음), H0: p = 0.5
    z = (k - n / 2) / math.sqrt(n / 4)
    return math.erfc(abs(z) / math.sqrt(2))


def pearson_ci(r, n, z=1.96):
    zr = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return math.tanh(zr - z * se), math.tanh(zr + z * se)


res = {}

# ---- 1. 표 4 인간 선호도 ----
t4 = {
    "2B>Cosmos phys": 62.50, "2B>Cosmos act": 63.45,
    "14B>Cosmos phys": 73.50, "14B>Cosmos act": 72.55,
    "14B>2B phys": 72.50, "14B>2B act": 65.53,
}
# 정수 표본 수(분모)로 여섯 값을 모두 표현할 수 있는지 검사
def fits(N):
    for v in t4.values():
        if not any(abs(round(k / N * 100, 2) - v) < 1e-9
                   for k in range(int(v / 100 * N) - 2, int(v / 100 * N) + 3)):
            return False
    return True

res["table4_integer_denominator_found"] = [N for N in range(2, 3001) if fits(N)]
t4_ci = {}
for name, pct in t4.items():
    row = {}
    for n_eff in (50, 600):  # 50쌍(샘플) 또는 50쌍 x 12명 = 600 판정 (가정)
        k = round(pct / 100 * n_eff)
        lo, hi = wilson(pct / 100 * n_eff, n_eff)
        row[f"n={n_eff}"] = {"wilson95": [round(lo * 100, 1), round(hi * 100, 1)],
                             "p_vs_50": round(binom_p_two_sided(pct / 100 * n_eff, n_eff), 4)}
    t4_ci[name] = {"pct": pct, **row}
res["table4_ci_assumed_n"] = t4_ci

# ---- 2. 정책 평가 상관 (Fig 5a) ----
# 그림 5a의 점은 A-F 여섯 개로 읽힘. 점별 좌표는 그림에서 정확히 읽을 수 없음.
res["fig5a_pearson"] = {
    "r": 0.995,
    "n_points_read_from_figure": 6,
    "ci95_if_n6": [round(x, 3) for x in pearson_ci(0.995, 6)],
    "ci95_if_n20_scenes": [round(x, 3) for x in pearson_ci(0.995, 20)],
}

# ---- 3. 모델 기반 계획 (Fig 5b, 막대 값은 그림 라벨) ----
g1 = {"A": 5, "B": 16, "C": 35, "D": 47, "E": 53, "DreamDojo": 70, "uniform": 37}
g2 = {"A": 35, "B": 39, "C": 41, "D": 46, "E": 63, "DreamDojo": 68, "uniform": 55}
res["fig5b"] = {
    "group1_gain_vs_best_ckpt_points": g1["DreamDojo"] - g1["E"],
    "group1_ratio_vs_uniform": round(g1["DreamDojo"] / g1["uniform"], 2),
    "group2_gain_vs_best_ckpt_points": g2["DreamDojo"] - g2["E"],
    "group2_ratio_vs_uniform": round(g2["DreamDojo"] / g2["uniform"], 2),
    "group2_ratio_vs_worst_ckpt": round(g2["DreamDojo"] / g2["A"], 2),
    "note": "본문은 두 번째 그룹도 '거의 2배'라고 씀. 그림의 균등 샘플링(55%) 대비는 1.24배. 10개 장면, 불확실성 없음.",
}

# ---- 4. 표 1 규모 주장 ----
hours_prior_max = 2900          # AgiBot-World 2.9k h (표 1)
skills_prior_max = 87           # AgiBot-World 87 (표 1)
scenes_prior_max = 564          # DROID 564 (표 1)
res["table1"] = {
    "duration_ratio": round(44711 / hours_prior_max, 1),                 # 논문 15x
    "skill_ratio_vs_max_prior_87": round(6015 / skills_prior_max, 1),    # 논문 96x
    "scene_ratio_using_table_1135k": round(1135000 / scenes_prior_max),  # 논문 2000x
    "scene_ratio_using_text_9869": round(9869 / scenes_prior_max, 1),    # 본문 9,869 장면
    "hours_sum_check": 55 + 829 + 43827,                                 # 44,711
    "traj_sum_check_k": 13.9 + 30 + 1135,                                # 1,178.9k -> 1,179k
}

# ---- 5. 표 2, 3, 5, 6, 7 차이 ----
def diff_table(name, a, b):
    return {k: round(b[k] - a[k], 4) for k in a} | {"label": name}

t5_base = {"psnr": 16.199, "ssim": 0.557, "lpips": 0.315}
t5_rel = {"psnr": 16.522, "ssim": 0.576, "lpips": 0.304}
t5_chunk = {"psnr": 17.626, "ssim": 0.620, "lpips": 0.267}
t5_all = {"psnr": 17.630, "ssim": 0.622, "lpips": 0.266}
res["table5_gr1_val"] = {
    "relative": diff_table("base->relative", t5_base, t5_rel),
    "chunked": diff_table("relative->chunked", t5_rel, t5_chunk),
    "temporal": diff_table("chunked->temporal", t5_chunk, t5_all),
}
cf_chunk = {"psnr": 20.783, "ssim": 0.790, "lpips": 0.193}
cf_all = {"psnr": 20.980, "ssim": 0.796, "lpips": 0.189}
res["table5_counterfactual_temporal"] = diff_table("chunked->temporal (cf)", cf_chunk, cf_all)

t2_inlab = {"action_free": 20.797, "latent": 20.913, "gt_retarget": 20.960}
t2_egodex = {"action_free": 19.924, "latent": 20.344, "mano": 20.474}
res["table2_psnr_gaps_db"] = {
    "inlab_latent_minus_actionfree": round(t2_inlab["latent"] - t2_inlab["action_free"], 3),
    "inlab_retarget_minus_latent": round(t2_inlab["gt_retarget"] - t2_inlab["latent"], 3),
    "egodex_latent_minus_actionfree": round(t2_egodex["latent"] - t2_egodex["action_free"], 3),
    "egodex_mano_minus_latent": round(t2_egodex["mano"] - t2_egodex["latent"], 3),
}

t3 = {  # 표 3: 행 순서는 논문과 같음 (Cosmos, In-lab, +EgoDex, +HV)
    "inlab_eval": [20.576, 20.913, 20.972, 21.016],
    "egodex_eval": [19.952, 20.267, 20.334, 20.414],
    "hv_eval": [18.274, 18.621, 18.706, 18.724],
    "cf_eval": [20.472, 20.755, 20.797, 20.852],
}
res["table3_increments_db"] = {
    k: [round(v[i + 1] - v[i], 3) for i in range(3)] for k, v in t3.items()
}
res["table3_monotonic"] = {k: all(v[i + 1] > v[i] for i in range(3)) for k, v in t3.items()}

t7 = {  # 표 7: (w/o pretrain, w/ pretrain) 각 PSNR, SSIM, LPIPS
    "inlab": ((20.304, 0.770, 0.230), (20.733, 0.782, 0.220)),
    "egodex": ((19.119, 0.762, 0.240), (19.313, 0.765, 0.235)),
    "hv": ((17.869, 0.736, 0.259), (18.195, 0.740, 0.254)),
    "cf": ((19.782, 0.758, 0.232), (19.891, 0.746, 0.234)),
}
res["table7_with_minus_without"] = {
    k: {"psnr": round(b[0] - a[0], 3), "ssim": round(b[1] - a[1], 3), "lpips": round(b[2] - a[2], 3)}
    for k, (a, b) in t7.items()
}

# 표 6: 증류
teacher_fps, student_fps = 2.72, 10.81
res["table6"] = {
    "speedup": round(student_fps / teacher_fps, 2),                   # 논문 nearly 4x
    "seconds_for_600_frames_at_10.81fps": round(600 / student_fps, 1),  # 1분이 아님
    "seconds_for_600_frames_at_2.72fps": round(600 / teacher_fps, 1),
    "frames_in_60s_at_10.81fps": round(60 * student_fps),
    "psnr_teacher_minus_student": round(14.086 - 13.146, 3),
}

# ---- 6. 검정력: 같은 100개 샘플에서 차이를 탐지하려면 얼마나 커야 하나 ----
# 샘플별 차이의 SD를 가정 (논문에 없음). 양측 alpha 0.05, power 0.8, n = 100.
n = 100
mde = {}
for sd in (0.5, 1.0, 2.0, 3.0):
    mde[f"sd_diff_{sd}dB"] = round((1.96 + 0.8416) * sd / math.sqrt(n), 3)
res["mde_psnr_db_n100"] = mde
res["mde_note"] = ("샘플 단위 PSNR 차이의 SD가 1 dB이면 탐지 한계는 약 0.28 dB. 표 2·3의 차이(0.02–0.13 dB)를 탐지하려면 "
                   "쌍 차이의 SD가 약 0.1–0.45 dB 이하여야 함. SD는 가정이며 논문에 없음.")

with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print(json.dumps(res, ensure_ascii=False, indent=2))
