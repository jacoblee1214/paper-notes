"""Assembling Two Parts in One Hand (arXiv:2609.10137) 표·그림 수치 재계산.

논문에는 코드가 없어 학습은 재현하지 않는다. 표 1(실로봇 20회 시행), 부록 C(500 에피소드),
그림 5·6의 판독 값, 보상 식, 표 2·3의 산술을 다시 계산하고 결과를 recompute_results.json 에 저장한다.

실행: python3 analysis/recompute.py   (numpy, scipy 필요, 수 초)

그림에서 읽은 값에는 "fig_read" 라고 표시했다. 막대 위 숫자는 논문 값이고 오차 막대 길이는 그림 판독 값(약 ±10)이다.
오차 막대의 종류(SD, SE, 95% CI)가 논문에 없어 세 가지 해석을 모두 계산한다(독립 학습 3회 가정).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


# ---------------------------------------------------------------- V1 표 1
T1 = {  # 과제: (닫힌 루프 정렬, 닫힌 루프 조립, 열린 루프 정렬, 열린 루프 조립), n = 20
    "Bottle": (18, 15, 2, 2),
    "Syringe": (18, 17, 1, 0),
    "Marker": (16, 16, 0, 0),
}
N = 20
t1 = {"per_task": {}, "pooled": {}}
for task, (ca, cs, oa, os_) in T1.items():
    t1["per_task"][task] = {
        "closed_align": [ca, N, round(100 * ca / N, 1), wilson(ca, N)],
        "closed_assembly": [cs, N, round(100 * cs / N, 1), wilson(cs, N)],
        "open_align": [oa, N, round(100 * oa / N, 1), wilson(oa, N)],
        "open_assembly": [os_, N, round(100 * os_ / N, 1), wilson(os_, N)],
        "fisher_assembly_closed_vs_open_p": fisher(cs, N, os_, N),
        "fisher_align_closed_vs_open_p": fisher(ca, N, oa, N),
        "assembly_le_align": cs <= ca and os_ <= oa,
    }
sums = [sum(v[i] for v in T1.values()) for i in range(4)]
t1["pooled"] = {
    "closed_align": [sums[0], 3 * N, round(100 * sums[0] / 60, 1), wilson(sums[0], 60)],
    "closed_assembly": [sums[1], 3 * N, round(100 * sums[1] / 60, 1), wilson(sums[1], 60)],
    "open_align": [sums[2], 3 * N, round(100 * sums[2] / 60, 1), wilson(sums[2], 60)],
    "open_assembly": [sums[3], 3 * N, round(100 * sums[3] / 60, 1), wilson(sums[3], 60)],
    "fisher_assembly_p": fisher(sums[1], 60, sums[3], 60),
    "text_open_loop_alignment_total_claim": 3,
    "open_loop_alignment_total_recount": sums[2],
}
# 최소 검출 가능 차이: 20회에서 한 과제 성공률의 Wilson 반폭
t1["wilson_halfwidth_at_80pct_n20"] = [round((wilson(16, 20)[1] - wilson(16, 20)[0]) / 2, 1)]
R["V1_table1"] = t1

# ---------------------------------------------------------------- V2 프로젝트 페이지 영상 카운터
page = {"Bottle": 14, "Syringe": 19, "Marker": 16}
paper_asm = {k: v[1] for k, v in T1.items()}
paper_aln = {k: v[0] for k, v in T1.items()}
R["V2_video_counters"] = {
    "project_page_counter": page,
    "table1_assembly": paper_asm,
    "table1_align": paper_aln,
    "diff_vs_assembly": {k: page[k] - paper_asm[k] for k in page},
    "diff_vs_align": {k: page[k] - paper_aln[k] for k in page},
    "total_counter": sum(page.values()),
    "total_table1_assembly": sum(paper_asm.values()),
    "wilson_counter": {k: wilson(v, 20) for k, v in page.items()},
    "note": "프로젝트 페이지가 스스로 '논문 표와 다르다'고 적은 값. 영상 카운터의 성공 기준은 알 수 없다.",
}

# ---------------------------------------------------------------- V3 부록 C 시뮬레이션 성공률(500 에피소드)
marker = {  # 변형: (목표 도달 보상, 성공률 %)
    "Ours": (1280.1, 65),
    "Proprioception-only": (338.0, 11),
    "History-MLP": (995.3, 49),
    "w/o finger reward": (629.4, 27),
    "w/o pose reward": (981.8, 40),
}
n500 = 500
v3 = {"rows": {}, "integer_feasible_for_500": {}}
for k, (rw, pct) in marker.items():
    cnt = pct * n500 / 100
    v3["integer_feasible_for_500"][k] = abs(cnt - round(cnt)) < 1e-9
    v3["rows"][k] = {"reward": rw, "success_pct": pct, "count": round(cnt), "wilson": wilson(round(cnt), n500)}
full = round(65 * 5)
v3["fisher_full_vs_variant_p"] = {
    k: fisher(full, n500, round(v[1] * 5), n500) for k, v in marker.items() if k != "Ours"
}
v3["fisher_history_vs_wo_pose_p"] = fisher(round(49 * 5), n500, round(40 * 5), n500)
rw = np.array([v[0] for v in marker.values()])
sp = np.array([v[1] for v in marker.values()], dtype=float)
rho = stats.spearmanr(rw, sp)
slope, icpt, r_val, p_val, _ = stats.linregress(rw, sp)
v3["reward_vs_success"] = {"spearman_rho": float(rho[0]), "pearson_r2": float(r_val ** 2),
                           "slope_pct_per_reward": float(slope), "intercept_pct": float(icpt)}
v3["sim_vs_real_marker"] = {
    "sim_success_pct": 65, "sim_wilson_n500": wilson(325, 500),
    "real_closed_loop": [16, 20, 80.0, wilson(16, 20)],
    "fisher_sim_vs_real_p": fisher(325, 500, 16, 20),
    "sim_inside_real_wilson": wilson(16, 20)[0] <= 65 <= wilson(16, 20)[1],
}
R["V3_marker_success"] = v3

# ---------------------------------------------------------------- V4 그림 5 / 그림 6 판독 값
fig5 = {  # 과제: 변형: (막대 위 숫자, 오차 막대 반폭 fig_read)
    "Bottle": {"Ours": (1429.1, 85), "Proprioception-only": (805.3, 55), "History-MLP": (1088.4, 95),
               "w/o finger reward": (960.9, 90), "w/o pose reward": (1223.5, 110)},
    "Syringe": {"Ours": (921.2, 48), "Proprioception-only": (786.9, 88), "History-MLP": (638.5, 50),
                "w/o finger reward": (630.0, 48), "w/o pose reward": (790.2, 40)},
}
axis_floor = {"Bottle": 550, "Syringe": 500}  # 막대 아래쪽 축 시작(그림 판독, 0이 아님)
v4 = {"rel_to_full_pct": {}, "visual_bar_ratio_vs_value_ratio": {}, "tests": {}}
for task, rows in fig5.items():
    full_v = rows["Ours"][0]
    v4["rel_to_full_pct"][task] = {k: round(100 * v[0] / full_v, 1) for k, v in rows.items()}
    fl = axis_floor[task]
    v4["visual_bar_ratio_vs_value_ratio"][task] = {
        k: {"value_ratio": round(v[0] / full_v, 3), "drawn_height_ratio": round((v[0] - fl) / (full_v - fl), 3)}
        for k, v in rows.items() if k != "Ours"
    }
    v4["tests"][task] = {}
    for k, (m, hw) in rows.items():
        if k == "Ours":
            continue
        out = {}
        for label, sd_of in {"bar_is_SD": lambda h: h, "bar_is_SE": lambda h: h * math.sqrt(3),
                             "bar_is_95CI_t": lambda h: h * math.sqrt(3) / stats.t.ppf(0.975, 2)}.items():
            t, p = stats.ttest_ind_from_stats(full_v, sd_of(rows["Ours"][1]), 3, m, sd_of(hw), 3, equal_var=False)
            out[label] = round(float(p), 4)
        v4["tests"][task][k] = out
# 그림 6 판독 값
fig6 = {
    "gauss": {"Bottle": {1: 1430, 2: 1340, 3: 1252}, "Syringe": {1: 921, 2: 870, 3: 788}},
    "offset": {"Bottle": {0: 1430, 1: 1403, 2: 1315, 3: 1037}, "Syringe": {0: 920, 1: 868, 2: 763, 3: 543}},
}
prop = {"Bottle": 805.3, "Syringe": 786.9, "Marker": 338.0}
full_r = {"Bottle": 1429.1, "Syringe": 921.2, "Marker": 1280.1}
steps = {"Bottle": 300, "Syringe": 150, "Marker": 300}  # 20 s, 10 s, 20 s x 15 Hz
v4["fig6_fig_read"] = {}
for kind in ("gauss", "offset"):
    for task, d in fig6[kind].items():
        mag3 = d[3]
        v4["fig6_fig_read"][f"{task}_{kind}_3cm"] = {
            "value_fig_read": mag3,
            "pct_of_full_policy": round(100 * mag3 / full_r[task], 1),
            "pct_of_step_ceiling": round(100 * mag3 / (10 * steps[task]), 1),
            "vision_benefit_retained": round((mag3 - prop[task]) / (full_r[task] - prop[task]), 2),
        }
for kind, val in (("gauss", 963.7), ("offset", 670.9)):
    v4["fig6_fig_read"][f"Marker_{kind}_3cm"] = {
        "value_paper_text": val, "pct_of_full_policy": round(100 * val / 1280.1, 1),
        "pct_of_step_ceiling": round(100 * val / (10 * steps["Marker"]), 1),
        "vision_benefit_retained": round((val - 338.0) / (1280.1 - 338.0), 2),
    }
R["V4_fig5_fig6"] = v4

# ---------------------------------------------------------------- V5 보상 식 (식 3)
def goal_reward(exy, ez, eth):
    return 5 * (math.exp(-60 * exy - 60 * ez - 5 * eth) + math.exp(-200 * exy - 120 * ez - 30 * eth))


def eth_from_deg(d):
    return 1 - math.cos(math.radians(d))


v5 = {"per_step_reward": {}, "implied_error_mm": {}, "ceiling": {}}
for exy_mm, ez_mm, deg in [(0, 0, 0), (2, 2, 0), (3, 3, 0), (5, 5, 0), (10, 10, 0), (10, 10, 5), (20, 20, 10), (5, 5, 20)]:
    v5["per_step_reward"][f"xy{exy_mm}mm_z{ez_mm}mm_{deg}deg"] = round(
        goal_reward(exy_mm / 1000, ez_mm / 1000, eth_from_deg(deg)), 3)
# 한 스텝당 평균 보상 = 누적 / 스텝 수 (워밍업 7-12 스텝, 조기 종료는 무시)
for task in ("Bottle", "Syringe", "Marker"):
    per_step = full_r[task] / steps[task]
    lo, hi = 0.0, 0.05
    for _ in range(60):  # exy = ez = e, eth = 0 인 오차 e 를 이분법으로 찾는다
        mid = (lo + hi) / 2
        if goal_reward(mid, mid, 0) > per_step:
            lo = mid
        else:
            hi = mid
    v5["implied_error_mm"][task] = {"per_step_mean": round(per_step, 2), "equal_xy_z_error_mm": round(1000 * lo, 2)}
    v5["ceiling"][task] = {"steps": steps[task], "max_cumulative": 10 * steps[task],
                           "reported_pct_of_ceiling": round(100 * full_r[task] / (10 * steps[task]), 1)}
v5["time_in_seconds"] = {"Bottle": 20, "Syringe": 10, "Marker": 20, "control_hz": 15}
R["V5_reward"] = v5

# ---------------------------------------------------------------- V6 산술 검사 (표 2, 식 1-2, 부록 A)
v6 = {}
v6["obs_dim"] = {"formula": "3+3+3+3+22", "sum": 3 + 3 + 3 + 3 + 22, "paper": 34}
v6["samples_per_rollout"] = {"fixed": 256 * 128, "paper_fixed": 32768, "tilt": 1024 * 128, "paper_tilt": 131072}
v6["minibatches_per_epoch"] = {"ppo_fixed": 32768 / 2048, "ppo_tilt": 131072 / 8192,
                               "critic_fixed": 32768 / 512, "critic_tilt": 131072 / 2048}
v6["samples_at_epoch_600"] = {"fixed": 600 * 32768, "tilt": 600 * 131072,
                              "robot_days_at_15hz_fixed": round(600 * 32768 / 15 / 86400, 1)}
v6["gamma_horizon_steps"] = round(1 / (1 - 0.995))
v6["gamma_horizon_seconds_at_15hz"] = round(1 / (1 - 0.995) / 15, 1)
v6["max_joint_speed_rad_s"] = 0.1 * 15
v6["physics_substeps_per_policy_step"] = 120 / 15
v6["gravity_mean"] = (9.51 + 10.11) / 2
v6["noise_1cm_matches_sec44"] = True  # 표 3 object position sigma = 0.01 m, 4.4절 '1 cm'
# 부록 A 초기 상태 교란: c = 2 / (dt^2 N (N+1)), 반암시적 오일러에서 N 스텝 뒤 변위가 정확히 dp 인지
dt, Nn, dp = 1 / 120, 2, 1.0
c = 2 / (dt ** 2 * Nn * (Nn + 1))
a = dp * c  # 가속도 = F/m = dp * c
x, v = 0.0, 0.0
for _ in range(Nn):
    v += a * dt
    x += v * dt
v6["reset_perturbation"] = {"c": round(c, 1), "displacement_after_N_steps_per_unit_dp": round(x, 9)}
# 정책 파라미터 수 (LayerNorm 파라미터 제외, 입력은 34, 출력 22)
def lstm_params(inp, hid):
    return 4 * hid * (inp + hid) + 2 * 4 * hid


lstm = lstm_params(34, 1024) + lstm_params(1024, 1024)
mlp = (1024 * 512 + 512) + (512 * 128 + 128) + (128 * 64 + 64) + (64 * 22 + 22)
hist_in = 8 * (34 + 22)  # 8프레임의 입력-출력 이력: 관측 34 + 행동 22 로 가정
hist = (hist_in * 1024 + 1024) + (1024 * 512 + 512) + (512 * 128 + 128) + (128 * 64 + 64) + (64 * 22 + 22)
v6["policy_params"] = {"lstm_backbone": lstm, "mlp_head": mlp, "recurrent_total": lstm + mlp,
                       "history_mlp_total_assumed_input_448": hist,
                       "ratio": round((lstm + mlp) / hist, 1)}
R["V6_arithmetic"] = v6

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
