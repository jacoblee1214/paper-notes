"""CHORD 표·그림·예산 재계산 (CPU, 단일 스레드). 실행: python3 analysis/table_check.py

출처: arXiv:2607.00033v2 Table 1–3, Fig. 5–6, Sec. 4.1·4.6, App. B·D·G,
공식 코드 video_to_data @ 9cad120 설정값, 이 저장소 flashdexretarget/ Table I.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import json
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
out = {}


def welch(m1, s1, m2, s2, n=3):
    t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    return {"t": round(float(t), 2), "p_two_sided": round(float(p), 4)}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


# ---------- C1: 1,831개 평가 (Abstract 82.12%, Fig. 5 왼쪽) ----------
fig5_left = {"HOT3D-1Obj": (1131, 0.772), "OakInk2-1Obj": (147, 0.786), "OakInk2-2Obj": (130, 0.746),
             "ARCTIC": (196, 0.814), "TACO": (77, 0.935), "HOT3D-2Obj": (150, 0.753)}
n_tot = sum(n for n, _ in fig5_left.values())
wmean = sum(n * v for n, v in fig5_left.values()) / n_tot
umean = sum(v for _, v in fig5_left.values()) / len(fig5_left)
out["C1_1831_tasks"] = {
    "n_sum": n_tot,
    "fig5_weighted_mean_completion": round(wmean, 4),
    "fig5_unweighted_mean": round(umean, 4),
    "abstract_claim": 0.8212,
    "claim_x1831": round(0.8212 * 1831, 2),
    "nearest_task_fractions": [f"{k}/1831 = {k / 1831:.4%}" for k in (1503, 1504)],
    "if_3_seeds_x1831": f"4511/5493 = {4511 / 5493:.4%}",
    "hot3d_share_of_eval": round((1131 + 150) / n_tot, 4),
    "eval_share_of_4739": round(1831 / 4739, 4),
    "hot3d_clips_share_of_4739": round(4045 / 4739, 4),
    "note": "Fig. 5의 막대는 완료율 평균(0.781). 82.12%는 '완료율 > 0.7인 과제 비율'로 읽어야 하며, 1,831로는 정수가 안 되고 시드 3개(5,493)면 맞는다",
}

# ---------- C2: 전신 90.77%, Table 3 ----------
fig5_right = {"ARCTIC-rigid": (4, 0.994), "TPV": (5, 0.867), "ARCTIC-artic": (4, 0.914), "TACO-multi": (4, 0.866)}
n_r = sum(n for n, _ in fig5_right.values())
w_r = sum(n * v for n, v in fig5_right.values()) / n_r
t3 = {"rigid": (0.994, 0.460), "articulated": (0.914, 0.000), "multi": (0.866, 0.192)}
out["C2_whole_body"] = {
    "n_sum": n_r,
    "fig5_right_weighted_mean": round(w_r, 4),
    "abstract_claim": 0.9077,
    "table3_overall_chord_recomputed": round(sum(v[0] for v in t3.values()) / 3, 4),
    "table3_overall_chord_reported": 0.925,
    "table3_overall_posonly_recomputed": round(sum(v[1] for v in t3.values()) / 3, 4),
    "table3_overall_posonly_reported": 0.217,
    "fig5_std_vs_table3_std": {"rigid": [0.038, 0.008], "articulated": [0.140, 0.107], "multi": [0.170, 0.129]},
    "note": "90.77%는 손만 12개 + TPV 5개 = 17개 가중평균과 정확히 맞는다. Table 3(n = 12)은 TPV를 뺀 값. 같은 과제인데 Fig. 5와 Table 3의 표준편차가 다르다",
}

# ---------- C3: Table 1 (60개 과제, 시드 3개) ----------
T1 = {  # AUC, SP-SR, MP-SR, JPE, RPE, MPPE: (mean, std)
    "DM": [(0.737, 0.053), (0.690, 0.068), (0.569, 0.097), (13.10, 2.31), (9.92, 1.64), (15.88, 0.70)],
    "MT": [(0.506, 0.006), (0.423, 0.011), (0.323, 0.003), (30.07, 0.61), (9.91, 0.42), (21.24, 6.19)],
    "SP": [(0.804, 0.044), (0.911, 0.039), (0.789, 0.047), (41.35, 2.44), (6.99, 1.56), (14.28, 2.79)],
    "H2S2R": [(0.730, 0.056), (0.686, 0.068), (0.569, 0.085), (11.80, 1.50), (9.78, 1.03), (16.59, 0.51)],
    "CHORD": [(0.918, 0.018), (0.926, 0.022), (0.894, 0.032), (4.34, 0.25), (5.14, 0.54), (5.11, 0.39)],
}
metrics = ["AUC", "SP-SR", "MP-SR", "JPE", "RPE", "MPPE"]
higher = [True, True, True, False, False, False]
c3 = {}
for i, m in enumerate(metrics):
    base = [k for k in T1 if k != "CHORD"]
    best = max(base, key=lambda k: T1[k][i][0]) if higher[i] else min(base, key=lambda k: T1[k][i][0])
    (m1, s1), (m2, s2) = T1["CHORD"][i], T1[best][i]
    c3[m] = {"best_baseline": best, "chord": m1, "baseline": m2, "gap": round(m1 - m2, 3), **welch(m1, s1, m2, s2)}
c3["SP-SR_x180"] = round(0.926 * 180, 2)
c3["note"] = "SP-SR은 CHORD 0.926 vs SPIDER 0.911로 시드 3개 기준 차이가 유의하지 않다. SP-SR x 180이 정수가 아니므로 과제 성공 비율이 아니라 과제별 성공률의 평균으로 보인다"
out["C3_table1"] = c3

# ---------- C4: Table 2 (보상 ablation) ----------
T2 = {"CHORD": T1["CHORD"],
      "Pos. Only": [(0.862, 0.019), (0.856, 0.028), (0.780, 0.042), (6.41, 0.44), (7.81, 0.64), (7.63, 0.06)],
      "No Cont.": [(0.774, 0.036), (0.756, 0.059), (0.612, 0.055), (9.81, 0.73), (10.91, 1.24), (14.38, 2.23)]}
c4 = {}
for v in ["Pos. Only", "No Cont."]:
    c4[v] = {m: {"gap": round(T2["CHORD"][i][0] - T2[v][i][0], 3),
                 **welch(T2["CHORD"][i][0], T2["CHORD"][i][1], T2[v][i][0], T2[v][i][1])}
             for i, m in enumerate(metrics)}
c4["share_of_gain_SP-SR"] = {
    "no_contact_to_position": round(0.856 - 0.756, 3), "position_to_wrench": round(0.926 - 0.856, 3),
    "wrench_share_of_total": round((0.926 - 0.856) / (0.926 - 0.756), 3)}
c4["table2_chord_row_equals_table1"] = True
out["C4_table2"] = c4

# ---------- C5: 실물 (Sec. 4.6) ----------
real = {"bowl handover": (5, 5), "in-hand plate reorientation": (3, 5), "bowl-plate stacking": (5, 5),
        "mixer open-close": (5, 5), "capsule machine pick-place": (5, 5)}
k = sum(a for a, _ in real.values())
n = sum(b for _, b in real.values())
fisher = stats.fisher_exact([[6, 1], [7, 0]])
out["C5_real_world"] = {
    "open_loop_total": f"{k}/{n}", "rate": round(k / n, 3), "wilson95": wilson(k, n),
    "plate_reorientation_wilson95": wilson(3, 5),
    "box_open_vs_closed": "6/7 vs 7/7", "fisher_p": round(float(fisher[1]), 3),
}

# ---------- C6: Fig. 6 상관 ----------
r = 0.798
out["C6_fig6"] = {"r": r, "r2": round(r * r, 3), "claim": "roughly two-thirds of the variance",
                  "fit": "y = 1.112x + 0.057", "x_where_fit_hits_1": round((1 - 0.057) / 1.112, 3)}

# ---------- C7: 예산 산술 (코드 설정 + FDR Table I) ----------
ppo_iter = 4096 * 24                           # num_envs x num_steps_per_env
sched = [2000, 3500, 5000, 6500, 8000, 9500, 11000, 12500, 14000, 15500]
voc = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.0, 0.0]
voc_off_iter = sched[voc.index(0.0) - 1]       # bisect_right: index i는 sched[i-1]부터 적용
flash_th = [19660800, 34406400, 49152000, 63897600, 78643200, 93388800, 108134400, 122880000,
            137625600, 200007680, 250003456]
flash_voc = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.0, 0.0, 0.0]
flash_voc_off = flash_th[flash_voc.index(0.0) - 1]
ppo_full = 20000 * ppo_iter
flash_total = 250003456
fdr = {"XHand": 2847, "Sharpa": 3314}
c7 = {
    "ppo_transitions_per_iter": ppo_iter,
    "ppo_voc_zero_from_iter": voc_off_iter,
    "ppo_voc_zero_transitions": voc_off_iter * ppo_iter,
    "ppo_max_iter_transitions": ppo_full,
    "flash_first9_thresholds_equal_ppo_div10": all(abs(t - s * ppo_iter / 10) < 1 for t, s in zip(flash_th[:9], sched[:9])),
    "flash_extra_stages": "200.0M부터 reset_to_first_frame 0.5 단계, 마지막 250.0M 임계값은 학습 끝과 같음",
    "flash_voc_zero_transitions": flash_voc_off,
    "flash_total_transitions": flash_total,
    "ppo_over_flash_transitions": round(ppo_full / flash_total, 2),
    "flash_throughput_if_2h": round(flash_total / 7200),
    "fdr_gpu_h_per_motion": {k: round(v / 50, 1) for k, v in fdr.items()},
    "fdr_implied_throughput_if_full_ppo": {k: round(ppo_full / (v / 50 * 3600)) for k, v in fdr.items()},
    "fdr_per_motion_over_chord_paper_2h": {k: round(v / 50 / 2, 1) for k, v in fdr.items()},
    "hypothetical_chord_flashsac_50_motions_gpu_h": 100,
    "fdr_flashdexretarget_gpu_h": {"XHand": 29, "Sharpa": 44},
    "ratio_hypothetical_vs_fdr": {"XHand": round(100 / 29, 2), "Sharpa": round(100 / 44, 2)},
    "paper_per_task_hours": {"CHORD": 2, "DexMachina": 10, "ManipTrans": 20, "Spider": 4},
    "note": ("FDR이 돌린 PPO 기본 설정은 2.0B 전이(VOC는 1.23B부터 0), 논문 v2의 FlashSAC 설정은 0.25B(VOC는 0.123B부터 0). "
             "FDR의 모션당 57–66 GPU-h(RTX 3090)는 논문의 2시간(L40S)보다 28–33배 크다. 전이 수 7.9배와 처리량 차이로 설명되지만, "
             "처리량(3090에서 약 9.6k/s)은 이 산술로 역산한 값이라 독립 검증이 아니다. 하드웨어가 달라 GPU-h 비율은 참고값"),
}
out["C7_budget"] = c7

# ---------- C8: 임계값·환경 수·시드 (논문 vs 코드) ----------
out["C8_settings"] = {
    "termination_paper": {"object_pos_m": 0.15, "object_rot_deg": 40},
    "termination_ppo_code": {"wrist_m": 0.2, "object_pos_m": 0.2, "object_rot_rad": 0.7,
                             "object_rot_deg": round(math.degrees(0.7), 1)},
    "termination_flashchord_code": {"wrist_m": 0.15, "object_pos_m": 0.10, "object_rot_rad": 0.50,
                                    "object_rot_deg": round(math.degrees(0.5), 1)},
    "envs_paper_appB": 2048, "envs_table2_caption": 4096, "world_count_code": 4096,
    "seeds_table1": 3, "seeds_appG_dexmachina_spider": 5,
    "friction_mu_code": 0.1, "friction_cone_half_angle_deg": round(math.degrees(math.atan(0.1)), 2),
    "edges_per_contact_code": "8 + normal = 9", "basis_directions": 512,
    "motion_speed_code": 0.5,
}

# ---------- C9: v1 Table 1 (과제 묶음별 비교) ----------
out["C9_v1_table1"] = {
    "DM_suite_AUC": {"DexMachina": "0.232±0.214", "Ours": "0.687±0.358"},
    "MT_suite_SR": {"ManipTrans": 0.428, "Ours": 0.639},
    "SP_suite_SR": {"Spider": "0.333±0.488", "Ours": "0.359±0.482"},
    "Ours-1": {"AUC": 0.211, "AUC_ours": 0.895, "SP-SR": 0.133, "SP-SR_ours": 0.999},
    "Ours-2_SP-SR": {"Spider": 0.533, "Ours": 0.982},
    "note": "v1은 기준선마다 자기 과제 묶음·자기 지표로 비교했고 표준편차가 평균과 비슷하게 컸다. v2는 60개 과제 x 시드 3개, 지표 6개, H2S2R 추가",
}

(ROOT / "analysis" / "table_check_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
for key, v in out.items():
    print(key, json.dumps(v, ensure_ascii=False)[:700])
