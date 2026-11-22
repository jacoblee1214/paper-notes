"""ABC (arXiv:2606.27375) 논문 표·본문 수치 재계산. 결과는 recompute_results.json 에 저장한다.

실행: python3 analysis/recompute.py   (numpy, scipy 필요, 수 초)

검사 항목 (V1-V16)은 analysis/03_analysis.md 의 번호와 같다. 논문에서 옮긴 값은 아래 상수에 출처(표·절)를 달았다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


# ---------------------------------------------------------------- V1 표 1 집계
# 표 1 (3.1절): 과제별 strict / progress (%), 모델 6종
MODELS = ["VLA Pooled adaLN", "VLA FAST+X-Attn", "VLA X-Attn", "DiT DINOv3-xattn", "DiT CLIP-adaln", "DiT CLIP-xattn"]
T1 = {
    "Bottles": {"strict": [67.3, 8.0, 0.0, 73.5, 11.8, 38.9], "progress": [83.0, 44.0, 5.2, 93.1, 53.6, 74.1]},
    "Dishrack": {"strict": [30.0, 2.8, 0.0, 23.2, 28.6, 34.7], "progress": [75.7, 47.2, 25.0, 74.1, 72.3, 81.3]},
    "Mugs": {"strict": [1.0, 0.0, 0.0, 2.0, 0.0, 0.0], "progress": [25.5, 6.5, 5.0, 35.3, 15.9, 21.0]},
}
T1_AGG = {"strict": [32.8, 3.6, 0.0, 32.9, 13.4, 24.5], "progress": [61.4, 32.6, 11.7, 67.5, 47.3, 58.8]}
T1_LAT = [17.24, 19.24, 19.24, 37.4, 27.5, 37.5]
v1 = {}
for m, name in enumerate(MODELS):
    row = {}
    for met in ("strict", "progress"):
        vals = [T1[t][met][m] for t in T1]
        mean = sum(vals) / 3
        row[met] = {"recomputed": round(mean, 2), "reported": T1_AGG[met][m], "match": abs(round(mean, 1) - T1_AGG[met][m]) < 0.051}
    v1[name] = row
R["V1_table1_aggregate"] = v1

# ---------------------------------------------------------------- V2 시도 50회에서 가능한 비율인가
# 3.1절 "Each task is evaluated for 50 trials". strict는 성공/50 이므로 2%의 배수여야 한다.
v2 = {"strict_cells": [], "progress_cells_bottles_dishrack": []}
nonint = 0
for t in T1:
    for m, name in enumerate(MODELS):
        p = T1[t]["strict"][m]
        ok50 = any(round(100 * k / 50, 1) == p for k in range(51))
        feas = [n for n in range(5, 201) if any(round(100 * k / n, 1) == p for k in range(n + 1))]
        small = [n for n in feas if 30 <= n <= 70]
        v2["strict_cells"].append({"task": t, "model": name, "reported": p, "feasible_at_n50": ok50, "feasible_n_in_30_70": small})
        nonint += (not ok50)
v2["strict_infeasible_at_n50"] = nonint
v2["strict_cells_total"] = 18
# progress: 병·접시는 최대 점수 6 (표 3), 50회면 분모 300. 점수 합이 정수여야 한다.
pi = 0
tot = 0
for t, mx in (("Bottles", 6), ("Dishrack", 6)):
    for m, name in enumerate(MODELS):
        p = T1[t]["progress"][m]
        ok = any(round(100 * k / (mx * 50), 1) == p for k in range(mx * 50 + 1))
        v2["progress_cells_bottles_dishrack"].append({"task": t, "model": name, "reported": p, "feasible_at_n50": ok})
        pi += (not ok)
        tot += 1
v2["progress_infeasible_at_n50"] = pi
v2["progress_cells_total"] = tot
R["V2_trial_count_feasibility"] = v2

# ---------------------------------------------------------------- V3 DINOv3 vs CLIP (xattn) 과제별
# 비정수 비율이라 성공 횟수를 round(p*50)로 근사한다(가정). 검정은 참고용.
v3 = []
for t in T1:
    for met in ("strict", "progress"):
        d, c = T1[t][met][3], T1[t][met][5]
        row = {"task": t, "metric": met, "dinov3_xattn": d, "clip_xattn": c, "diff": round(d - c, 1)}
        if met == "strict":
            kd, kc = round(d / 2), round(c / 2)
            row["fisher_p_assumed_n50"] = round(stats.fisher_exact([[kd, 50 - kd], [kc, 50 - kc]])[1], 4)
        v3.append(row)
R["V3_dino_vs_clip_per_task"] = v3

# ---------------------------------------------------------------- V4 오프라인 지표와 실제 성능 상관 (그림 8)
# 값은 그림 8 패널 제목에서 옮김. n = 16 체크포인트.
def p_from_r(r, n):
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return 2 * stats.t.sf(abs(t), n - 2)


def fisher_ci(r, n, z=1.959964):
    zr = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return [round(math.tanh(zr - z * se), 3), round(math.tanh(zr + z * se), 3)]


FIG8 = [
    ("train loss", "strict", -0.84, 4.1e-5, -0.83, 7.9e-5),
    ("val loss", "strict", -0.04, 0.886, 0.00, 1.000),
    ("val action error", "strict", -0.89, 3.8e-6, -0.93, 1.5e-7),
    ("train loss", "progress", -0.78, 3.9e-4, -0.84, 5.0e-5),
    ("val loss", "progress", 0.14, 0.599, 0.09, 0.737),
    ("val action error", "progress", -0.85, 2.9e-5, -0.93, 1.9e-7),
]
v4 = []
for name, met, r, pr, rho, prho in FIG8:
    v4.append({
        "metric": name, "target": met, "n": 16,
        "r": r, "p_reported": pr, "p_recomputed": float(f"{p_from_r(r, 16):.3g}"), "r_ci95_fisher_z": fisher_ci(r, 16),
        "rho": rho, "p_rho_reported": prho, "p_rho_from_t_approx": float(f"{p_from_r(rho, 16):.3g}") if abs(rho) < 1 else None,
    })
R["V4_fig8_offline_vs_real"] = v4
# 표본 의존: 16개 체크포인트는 같은 학습 실행의 다른 스텝을 포함하므로 독립이 아니다. 유효 n을 8(실행 8개 가정)로 줄였을 때
R["V4_fig8_effective_n_sensitivity"] = {
    str(ne): {"val_action_error_strict_r-0.89_p": float(f"{p_from_r(-0.89, ne):.3g}"), "train_loss_strict_r-0.84_p": float(f"{p_from_r(-0.84, ne):.3g}"),
              "val_loss_r-0.04_p": float(f"{p_from_r(-0.04, ne):.3g}")}
    for ne in (16, 12, 8, 6)
}

# ---------------------------------------------------------------- V5 sim-real 상관 (그림 9)
v5 = []
for name, r, pr in (("strict", 0.85, 4.2e-4), ("progress", 0.91, 5.0e-5)):
    v5.append({"metric": name, "n": 12, "r": r, "p_reported": pr, "p_recomputed": float(f"{p_from_r(r, 12):.3g}"), "r_ci95_fisher_z": fisher_ci(r, 12),
               "r2": round(r * r, 3)})
R["V5_fig9_sim_real"] = v5
# 12개 점은 4개씩 3묶음(DiT, VLA, DiT+sim 혼합)이다(4절). 묶음 효과 민감도: 유효 n = 6, 8
R["V5_fig9_effective_n_sensitivity"] = {str(ne): {"strict_r0.85_p": float(f"{p_from_r(0.85, ne):.3g}"), "progress_r0.91_p": float(f"{p_from_r(0.91, ne):.3g}")} for ne in (12, 8, 6, 5)}

# ---------------------------------------------------------------- V6 표 2 파라미터·FLOPs 일관성과 코드 구조 기반 파라미터 추정
H = 1536
blk = 25 * H * H  # self-attn 4H^2 + cross-attn 4H^2 + MLP 8H^2 + adaLN 9H^2
dit_params = 32 * blk + 4 * H * H + 2 * H * H  # + cond_proj(3H->H, H->H) + final adaLN(2H^2) 근사
pool = 3 * (4 * 768 ** 2 + 8 * 768 ** 2)  # 카메라별 attention-pool block
dit_head_est = dit_params + pool + 768 * H
Hv = 512
vla_head = 8 * 18 * Hv * Hv + (Hv + 4096 + Hv) * Hv + Hv * Hv + 2 * Hv * Hv + 8 * 512 * 3 * 0
vla_pool = 2560 * Hv + 4 * Hv * Hv
lm = 34 * (2560 * 2048 + 2 * 2560 * 1024 + 2048 * 2560 + 3 * 2560 * 10240)  # Gemma3 4B 본체(임베딩 제외)
emb = 262144 * 2560
sig = 27 * (4 * 1152 ** 2 + 2 * 1152 * 4304)
v6 = {
    "table2_dit_total_tflops": {"reported": 0.678, "sum": round(0.329 + 0.349, 3)},
    "table2_vla_total_tflops": {"reported": 7.020, "sum": round(6.957 + 0.063, 3)},
    "vla_over_dit_tflops_per_sample": round(7.020 / 0.678, 2),
    "dit_head_params_est_from_code": round(dit_head_est / 1e9, 3),
    "dit_head_params_reported_B": 1.93,
    "dit_total_params_reported_B": round(1.93 + 0.0857, 3),
    "fig21_xL_total_params_B": 1.93,
    "vla_head_params_est_M": round((vla_head + vla_pool) / 1e6, 1),
    "vla_head_params_reported_M": {"table2": 44.7, "sec3.2_text": 42, "project_page": 45},
    "gemma3_lm_nonembed_params_B": round(lm / 1e9, 3),
    "gemma3_embedding_params_B": round(emb / 1e9, 3),
    "siglip_params_M": round(sig / 1e6, 1),
    "vla_backbone_params_est_B": round((lm + emb + sig) / 1e9, 2),
    "vla_backbone_params_reported_B": 4.3,
}
# FLOPs: 훈련 = 6 x params x tokens (곱-덧셈 2 FLOPs, 순전파 x3). 토큰 수는 코드 상수에서 읽었다(정적).
tok_lm = 256  # fixed_seq_len
tok_sig = 3 * 256  # 224/14 = 16 -> 256 patch per image
f_lm = 6 * lm * tok_lm
f_sig = 6 * sig * tok_sig
v6["vla_backbone_tflops_est"] = round((f_lm + f_sig) / 1e12, 2)
v6["vla_backbone_tflops_reported"] = 6.957
v6["vla_head_tflops_est"] = round(6 * 44.7e6 * 30 * 8 / 1e12, 3)  # 30 토큰 x 8 draw
v6["vla_head_tflops_reported"] = 0.063
v6["dit_head_tflops_est"] = round(6 * 1.93e9 * 30 / 1e12, 3)
v6["dit_head_tflops_reported"] = 0.349
v6["dit_backbone_implied_tokens"] = round(0.329e12 / (6 * 85.7e6), 0)  # 3장 x (196 patch + cls + 4 reg) = 603 기대
R["V6_table2_params_flops"] = v6

# ---------------------------------------------------------------- V7 학습 계산량, 에폭, GPU 시간 (3.3절, B.3, 표 2)
steps, bs = 200_000, 9216
frames_130k = 3553 * 3600 * 30
v7 = {
    "dit_batch_arith": 12 * 8 * 96,
    "vla_batch_arith": 8 * 8 * 24 * 6,
    "batch_ratio": [1536, 4608, 9216],
    "samples_seen_200k_bs9216": steps * bs,
    "frames_in_abc130k_at_30fps": frames_130k,
    "epochs_over_frames_abc130k": round(steps * bs / frames_130k, 2),
    "epochs_over_frames_internal_7000h": round(steps * bs / (7000 * 3600 * 30), 2),
    "dit_train_flops_200k_bs9216": float(f"{0.678e12 * steps * bs:.3g}"),
    "vla_train_flops_200k_bs9216": float(f"{7.020e12 * steps * bs:.3g}"),
    "dit_train_flops_100k_bs1536": float(f"{0.678e12 * 100_000 * 1536:.3g}"),
    "vla_train_flops_100k_bs1536": float(f"{7.020e12 * 100_000 * 1536:.3g}"),
    "vla_k8_overhead_pct": round((1.366 / 1.346 - 1) * 100, 2),
    "vla_h200_gpu_hours_200k_assuming_64gpu_1.366s": round(200_000 * 1.366 * 64 / 3600),
}
R["V7_compute_budget"] = v7

# ---------------------------------------------------------------- V8 데이터셋 통계 (2절, 부록 A)
cats = {"Pick-and-Place": (67, 793), "Fine Pick-and-Place": (39, 736), "Folding": (36, 883), "Insertion/Ejection": (19, 441),
        "Tool Use": (16, 321), "Sorting": (8, 205), "Tying/Untying": (10, 175)}
v8 = {
    "tasks_sum": sum(c[0] for c in cats.values()),
    "tasks_reported": 195,
    "hours_sum": sum(c[1] for c in cats.values()),
    "hours_reported": 3553,
    "episodes": 134806,
    "mean_episode_sec": round(3553 * 3600 / 134806, 1),
    "subtask_hours": 1552,
    "subtask_share_pct": round(100 * 1552 / 3553, 1),
    "subtask_share_reported_pct": 44,
    "hours_per_task_by_category": {k: round(v[1] / v[0], 1) for k, v in cats.items()},
    "share_pct": {k: round(100 * v[1] / 3553, 1) for k, v in cats.items()},
    "tshirt_hours": 268.64,
    "tshirt_share_pct": round(100 * 268.64 / 3553, 1),
    "op0_hours_per_episode_s": round(19.5 * 3600 / 1183, 1),
    "op0_mean_episode_reported_s": 59,
}
R["V8_dataset_stats"] = v8

# ---------------------------------------------------------------- V9 부록 H 조건화 실험 (n = 10)
def fe(a, b, n=10):
    return round(stats.fisher_exact([[a, n - a], [b, n - b]])[1], 4)


v9 = {
    "table4_completions": {"all_task_only": 4, "op0_filtered": 2, "op0_cond": 8, "marginalize": 6, "op1_cond": 5},
    "fisher_p": {
        "op0_cond_vs_all_task_only": fe(8, 4),
        "op0_cond_vs_op0_filtered": fe(8, 2),
        "op0_cond_vs_op1_cond": fe(8, 5),
        "marginalize_vs_all_task_only": fe(6, 4),
        "op1_cond_vs_all_task_only": fe(5, 4),
        "prefix1_vs_prefix4": fe(8, 5),
        "h3_reflatten_5of10_vs_0of10": fe(5, 0),
    },
    "wilson_95_completions": {k: wilson(v, 10) for k, v in {"op0_cond": 8, "all_task_only": 4, "marginalize": 6, "op1_cond": 5, "op0_filtered": 2}.items()},
}
# 평균 시간 정합성: 시간초과 시도는 390초로 센다. 완료 시도의 평균 시간을 역산하고 가능한 범위인지 본다.
rows = {"all_task_only": (4, 302), "op0_filtered": (2, 369), "op0_cond": (8, 237), "marginalize": (6, 247), "op1_cond": (5, 277), "prefix4": (5, 309)}
impl = {}
for k, (c, mean) in rows.items():
    to = 10 - c
    avg_done = (mean * 10 - to * 390) / c
    impl[k] = {"completions": c, "mean_time_s": mean, "implied_mean_time_of_completed_s": round(avg_done, 1), "feasible": 0 < avg_done <= 390}
v9["implied_completed_time"] = impl
# H/M/L 합계가 completions와 맞는지
hml = {"all_task_only": (2, 2, 0, 4), "op0_filtered": (1, 0, 1, 2), "op0_cond": (4, 3, 1, 8), "marginalize": (1, 4, 1, 6), "op1_cond": (1, 1, 3, 5), "prefix4": (3, 2, 0, 5), "prefix1": (4, 3, 1, 8)}
v9["hml_sum_matches"] = {k: (v[0] + v[1] + v[2] == v[3]) for k, v in hml.items()}
v9["table4_op0_row_equals_table5_prefix1_row"] = (4.6, 8, 237, (4, 3, 1)) == (4.6, 8, 237, (4, 3, 1))
R["V9_appendix_H"] = v9

# ---------------------------------------------------------------- V10 DAgger 혼합 비율 (5.3절)
v10 = {}
for rd, share in (("round1", 0.30), ("round2", 0.15)):
    # 현재 라운드 데이터 안에서 개입 구간의 자연 비율 share, 혼합에서의 비율 0.10/(0.10+0.10)=0.5
    v10[rd] = {"natural_intervention_share_in_round": share, "mix_share_in_round_portion": 0.5, "oversample_factor": round(0.5 / share, 2)}
v10["round_portion_of_mix"] = 0.20
v10["collapsed_data_hours"] = 10
v10["round_hours"] = [1.0, 1.5]
v10["round_natural_share_of_all_data_pct"] = [round(100 * h / (10 + h), 1) for h in (1.0, 1.5)]
R["V10_dagger_mix"] = v10

# ---------------------------------------------------------------- V11 추론 지연 (표 1 vs 그림 23, 24, 부록 D)
ms_dit = [63.0, 47.5, 41.3, 36.3]
hz_dit = [15.9, 21.0, 24.2, 27.6]
ms_vla = [47.8, 22.6, 17.5]
hz_vla = [20.9, 44.2, 57.2]
R["V11_latency"] = {
    "dit_hz_from_ms": [round(1000 / m, 1) for m in ms_dit], "dit_hz_reported": hz_dit,
    "vla_hz_from_ms": [round(1000 / m, 1) for m in ms_vla], "vla_hz_reported": hz_vla,
    "dit_speedup": round(63.0 / 36.3, 2), "vla_speedup": round(47.8 / 17.5, 2),
    "table1_vs_fig23_dit_ms": {"table1": 37.4, "fig23": 36.3, "diff_pct": round((37.4 / 36.3 - 1) * 100, 1)},
    "table1_vs_fig24_vla_ms": {"table1": 17.24, "fig24": 17.5, "diff_pct": round((17.5 / 17.24 - 1) * 100, 1)},
    "fig19_read_reduction": round(9.75 / 0.14, 1),
    "chunk_exec_budget_ms_15_actions_30hz": round(15 / 30 * 1000),
}

# ---------------------------------------------------------------- V12 프로젝트 페이지 sim 진행률 곡선 (그림 12 원자료, SVG 점에서 역산)
def y2pct(y):
    return round((148 - y) / 126 * 100, 1)


dit_y = [132.4, 116.2, 90.3, 83.8, 78.0, 79.0, 78.5, 67.3, 69.7, 67.7, 68.4, 66.2, 61.6, 66.1, 61.4, 61.2, 59.5, 64.7, 60.5, 56.4]
vla_y = [116.7, 93.1, 73.9, 64.1, 61.2, 62.6, 61.7, 54.0, 55.5, 57.7, 54.2, 53.5, 53.5, 56.3, 54.0, 53.2, 55.0, 53.7, 49.6, 51.1]
dit = [y2pct(y) for y in dit_y]
vla = [y2pct(y) for y in vla_y]
R["V12_fig12_sim_progress_from_project_page"] = {
    "note": "abc.bot SVG 폴리라인에서 y 픽셀을 0-100%로 역산 (y=148 -> 0, y=22 -> 100). 20점, 체크포인트 간격 10k 스텝으로 가정(축 눈금 0-200k).",
    "dit": dit, "vla": vla,
    "final_dit": dit[-1], "final_vla": vla[-1],
    "gain_last_10_points_dit": round(dit[-1] - dit[9], 1), "gain_last_10_points_vla": round(vla[-1] - vla[9], 1),
    "gain_first_5_points_dit": round(dit[4] - dit[0], 1), "gain_first_5_points_vla": round(vla[4] - vla[0], 1),
    "vla_minus_dit_at_each_point": [round(a - b, 1) for a, b in zip(vla, dit)],
}

# ---------------------------------------------------------------- V13 처리량 역산 (공식 README의 it/s 와 표 2 TFLOPs)
its = (2.6, 3.0)  # README: 8 GPU, 배치 90/GPU, DiT, H100/H200
H100_BF16_DENSE_TFLOPS = 989  # 공개 사양(밀집 bf16), 참고용 가정
per_gpu = [0.678 * 90 * i for i in its]
R["V13_throughput_from_readme"] = {
    "readme_it_per_s": its,
    "dit_tflops_per_gpu_achieved": [round(x, 0) for x in per_gpu],
    "mfu_vs_989_dense_bf16_pct": [round(100 * x / H100_BF16_DENSE_TFLOPS, 1) for x in per_gpu],
    "hours_200k_steps_at_readme_speed": [round(200_000 / i / 3600, 1) for i in its],
    "gpu_hours_96gpu_200k_steps": [round(200_000 / i / 3600 * 96) for i in its],
    "note": "배치 90/GPU 기준으로 논문의 96 GPU x 96 = 9216 과 GPU당 배치가 비슷하다는 가정. 실제 학습 시간은 논문에 없다.",
}

# ---------------------------------------------------------------- V16 검정력: 시도 수별 검출 가능한 최소 차이 (Cohen의 h, 양측 alpha = 0.05, 검정력 80%)
def h_effect(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def n_per_group(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return 2 * ((za + zb) / h_effect(p1, p2)) ** 2


def mde(n, base=0.5, alpha=0.05, power=0.8):
    for d in np.arange(0.01, 0.9, 0.01):
        if base + d / 2 > 1 or base - d / 2 < 0:
            break
        za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
        if abs(h_effect(base + d / 2, base - d / 2)) * math.sqrt(n / 2) >= za + zb:
            return round(float(d), 2)


R["V16_power"] = {
    "min_detectable_diff_at_p0.5": {str(n): mde(n) for n in (10, 20, 50, 100, 200)},
    "n_per_group_sim_engine_0.71_vs_0.54": round(n_per_group(0.71, 0.54)),
    "n_per_group_table1_strict_dishrack_34.7_vs_23.2": round(n_per_group(0.347, 0.232)),
    "n_per_group_73.5_vs_67.3": round(n_per_group(0.735, 0.673)),
}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT.name)
print(json.dumps({k: R[k] for k in ("V1_table1_aggregate",)}, ensure_ascii=False)[:600])
print("V2 strict infeasible at n=50:", R["V2_trial_count_feasibility"]["strict_infeasible_at_n50"], "/ 18;",
      "progress infeasible:", R["V2_trial_count_feasibility"]["progress_infeasible_at_n50"], "/", R["V2_trial_count_feasibility"]["progress_cells_total"])
