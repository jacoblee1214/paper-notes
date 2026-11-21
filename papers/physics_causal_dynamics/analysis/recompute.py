"""PICR 논문(arXiv:2611.19251 v1)의 보고 수치를 다시 계산하고 일관성을 검사한다.

입력은 논문 표에서 옮긴 값뿐이다(출처를 상수 옆에 적었다). 원데이터, 학습, 재현은 하지 않는다.
실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, CPU)
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}

# ---------------------------------------------------------------- Table 5 (3 seeds, mean ± SD, final epoch)
# 열: ADD seen (0.1-2 kg), ADD unseen (2-4 kg)
T5 = {
    "FiLM (PICR)": ((4.93, 0.54), (6.76, 1.28)),
    "Input concat": ((4.45, 0.37), (8.49, 0.69)),
    "Shift only": ((4.59, 0.54), (8.43, 0.80)),
    "Low-rank hypernet": ((4.26, 0.35), (11.75, 2.63)),
}
N_SEEDS = 3
T_CRIT = stats.t.ppf(0.975, df=N_SEEDS - 1)  # 두 자유도 t 분포, 3 시드


def ci95(mean, sd, n=N_SEEDS):
    h = T_CRIT * sd / math.sqrt(n)
    return (round(mean - h, 2), round(mean + h, 2))


R["table5_ci95"] = {k: {"seen": ci95(*v[0]), "unseen": ci95(*v[1])} for k, v in T5.items()}

film_u = T5["FiLM (PICR)"][1]
welch = {}
for k, v in T5.items():
    if k == "FiLM (PICR)":
        continue
    for band, idx in (("seen", 0), ("unseen", 1)):
        t, p = stats.ttest_ind_from_stats(film_u[0] if band == "unseen" else T5["FiLM (PICR)"][0][0],
                                          film_u[1] if band == "unseen" else T5["FiLM (PICR)"][0][1], N_SEEDS,
                                          v[idx][0], v[idx][1], N_SEEDS, equal_var=False)
        welch[f"{k} / {band}"] = {"t": round(float(t), 2), "p_welch": round(float(p), 4),
                                  "increase_pct": round(100 * (v[idx][0] / T5["FiLM (PICR)"][idx][0] - 1), 1)}
R["table5_welch_vs_film"] = welch

# 본문 주장: 대안 8.4-11.8 cm, 25-74% 증가
R["table5_unseen_increase_pct_range"] = (
    round(100 * (8.43 / 6.76 - 1)), round(100 * (11.75 / 6.76 - 1)))  # 기대 25, 74

# ---------------------------------------------------------------- Table 1 (mean over 3 seeds per the caption)
T1_ADD = {  # (seen, unseen)
    "PICR": (4.93, 6.76), "RWM": (7.45, 11.15), "CaDM": (6.06, 11.40),
    "CaDeLaC": (8.75, 21.41), "DeLaN": (6.96, 25.78), "Neural-Fly": (8.05, 22.27),
}
# seed 0 (Table 6와 프로젝트 페이지, 저장소 paper/fig_payload_sweep 기대값과 비교)
T6_ADD_SEED0 = {  # (seen, unseen) 0.1-2 kg / 2-4 kg
    "PICR": (4.58, 6.92), "RWM": (7.09, 11.63), "CaDM": (6.79, 10.90),
    "CaDeLaC": (9.07, 24.75), "DeLaN": (6.82, 25.65), "Neural-Fly": (7.66, 24.78),
}
R["table1_increase_seen_to_unseen_cm"] = {k: round(v[1] - v[0], 2) for k, v in T1_ADD.items()}
# 본문: PICR은 1.8 cm 증가, CaDeLaC/DeLaN/Neural-Fly는 12.7-18.8 cm 증가
R["table1_text_claim_increase_range_others"] = [
    round(min(T1_ADD[k][1] - T1_ADD[k][0] for k in ("CaDeLaC", "DeLaN", "Neural-Fly")), 2),
    round(max(T1_ADD[k][1] - T1_ADD[k][0] for k in ("CaDeLaC", "DeLaN", "Neural-Fly")), 2)]

base_ratio_mean = {k: round(T1_ADD[k][1] / T1_ADD["PICR"][1], 2) for k in T1_ADD if k != "PICR"}
base_ratio_seed0 = {k: round(T6_ADD_SEED0[k][1] / T6_ADD_SEED0["PICR"][1], 2) for k in T6_ADD_SEED0 if k != "PICR"}
R["unseen_ratio_vs_picr_table1_mean"] = base_ratio_mean          # 초록 1.6-3.8x 확인용
R["unseen_ratio_vs_picr_seed0"] = base_ratio_seed0
R["abstract_ratio_range_check"] = {
    "table1_mean": (min(base_ratio_mean.values()), max(base_ratio_mean.values())),
    "seed0": (min(base_ratio_seed0.values()), max(base_ratio_seed0.values())),
    "claimed": (1.6, 3.8)}

# 본문 "several times lower" 주장 확인 (Table 1 RMSE, 3 kg carry, joint 2)
RMSE_T1 = {"PICR": (2.15, 2.31, 1.59), "CaDeLaC": (11.29, 12.49, 2.15), "DeLaN": (9.99, 5.83, 1.67)}
R["rmse_ratio_baseline_over_picr"] = {
    k: {"gravity": round(RMSE_T1[k][0] / RMSE_T1["PICR"][0], 2),
        "inertial": round(RMSE_T1[k][1] / RMSE_T1["PICR"][1], 2),
        "coriolis": round(RMSE_T1[k][2] / RMSE_T1["PICR"][2], 2)} for k in ("CaDeLaC", "DeLaN")}

# Table 1 PICR 행과 Table 5 FiLM 행의 일치, 그리고 seed-0 저장소 값과의 차이
R["picr_table1_vs_table5_match"] = {
    "add_seen": (T1_ADD["PICR"][0], T5["FiLM (PICR)"][0][0]),
    "add_unseen": (T1_ADD["PICR"][1], T5["FiLM (PICR)"][1][0]),
    "gravity": (2.15, 2.15), "inertial": (2.31, 2.31), "coriolis": (1.59, 1.59)}
R["picr_table1_mean_vs_seed0_repo"] = {
    "add_seen": (4.93, 4.58), "add_unseen": (6.76, 6.92), "gain": (0.90, 0.86),
    "gravity": (2.15, 1.84), "inertial": (2.31, 1.63), "coriolis": (1.59, 1.46)}

# ---------------------------------------------------------------- Table 3 (seed 0, 단일 시드)
T3 = {"PICR": (4.58, 6.92, 1.84, 1.63, 1.46), "w/o contrastive": (6.07, 9.78, 2.11, 2.87, 1.15)}
R["table3_change_pct"] = {
    "add_seen": round(100 * (T3["w/o contrastive"][0] / T3["PICR"][0] - 1), 1),
    "add_unseen": round(100 * (T3["w/o contrastive"][1] / T3["PICR"][1] - 1), 1),
    "gravity": round(100 * (T3["w/o contrastive"][2] / T3["PICR"][2] - 1), 1),
    "inertial": round(100 * (T3["w/o contrastive"][3] / T3["PICR"][3] - 1), 1),
    "coriolis": round(100 * (T3["w/o contrastive"][4] / T3["PICR"][4] - 1), 1)}
# 단일 시드 변화량을 Table 5의 FiLM 시드 SD(3 시드)에 비교: 시드 잡음의 몇 배인가 (대략적 척도, 검정 아님)
R["table3_change_in_seed_sd_units"] = {
    "add_seen": round((T3["w/o contrastive"][0] - T3["PICR"][0]) / 0.54, 2),
    "add_unseen": round((T3["w/o contrastive"][1] - T3["PICR"][1]) / 1.28, 2)}

# ---------------------------------------------------------------- Table 4 (단일 시드, 다른 평가 프로토콜)
T4 = {"RWM": (10.58, 13.57), "RWM+contrastive": (12.48, 13.21), "CaDM": (9.76, 10.56),
      "CaDM+contrastive": (11.06, 11.66), "CaDeLaC": (11.33, 22.16), "CaDeLaC+contrastive": (10.48, 17.94),
      "PICR": (7.21, 6.30)}
R["table4_picr_unseen_below_seen"] = T4["PICR"][1] < T4["PICR"][0]
R["table4_cadelac_contrastive_unseen_vs_picr"] = round(T4["CaDeLaC+contrastive"][1] / T4["PICR"][1], 2)  # 본문 약 3배

# ---------------------------------------------------------------- Table 6 (폭을 맞춘 모델, 시드 0; PICR 좁은 모델은 3 시드 평균)
T6 = {"PICR 6.55M": (4.58, 6.92), "PICR width256 (3 seeds)": (4.77, 8.66),
      "CaDeLaC 6.7k": (9.07, 24.75), "CaDeLaC width640": (7.49, 16.76),
      "DeLaN 12k": (6.82, 25.65), "DeLaN width512": (7.22, 23.70)}
R["table6_gaps_narrow_picr"] = {
    "cadelac_w640_seen_minus_picr_w256": round(7.49 - 4.77, 2),     # 본문 2.7
    "cadelac_w640_unseen_minus_picr_w256": round(16.76 - 8.66, 2)}  # 본문 8.1
R["table6_claim_widening_deln"] = {
    "seen_change": round(7.22 - 6.82, 2), "unseen_change": round(23.70 - 25.65, 2)}

# ---------------------------------------------------------------- 파라미터 수 (식으로 확인)
def mlp_params(sizes):
    return sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))


def film_params(width, z=4, n_mod_layers=2):
    # 변조 층 2개마다 γ, β 두 맵(각 R^z -> R^width, 가중치 + 편향)
    return n_mod_layers * 2 * (z * width + width)


def lagrangian_params(width, n_out_list, film=True):
    total = 0
    for n_out in n_out_list:
        trunk = 7 * width + width
        hid = 2 * (width * width + width)  # 두 변조 층의 선형부
        out = width * n_out + n_out
        total += trunk + hid + out + (film_params(width) if film else 0)
    return total


ENC = mlp_params([252, 256, 256]) + 2 * (256 * 4 + 4)  # 인코더 MLP + μ, log σ² 머리
R["params_encoder"] = ENC                                 # 논문 132,616
R["params_picr_lagrangian_1024"] = lagrangian_params(1024, [7, 21, 1])   # 3개 망(A 대각 7, A 하삼각 21, V 1)
R["params_picr_total_1024"] = R["params_picr_lagrangian_1024"] + ENC     # 논문 표 2: 6,545,957
R["params_picr_lagrangian_256"] = lagrangian_params(256, [7, 21, 1])
R["params_picr_total_256"] = R["params_picr_lagrangian_256"] + ENC       # 논문 표 6: 0.56M
R["params_film_map_per_net"] = film_params(1024)                          # 논문 식 (14) 설명: 61,440은 3개 망 합

# ---------------------------------------------------------------- 데이터 수 확인
R["counts"] = {
    "train_plus_val_episodes": 318 + 40,            # 논문 5.1, D.2: 358
    "payload_sweep_windows_seen_plus_unseen": 16071 + 16896,  # 부록 C.1, 표 7: 32,967
    "torque_pairs_kept_fraction": round(1887 / 5616, 3),  # 부록 D.5: 33.6%
    "mass_levels_x_eps": 8 * 50,                    # D.2: 400
}

# ---------------------------------------------------------------- 저장소 기대값과 Table 1 시드 0 대조
R["repo_seed0_expectations_match_paper_table6"] = all(
    abs(a - b) < 1e-9 for a, b in zip(T6_ADD_SEED0["PICR"], (4.58, 6.92)))

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
