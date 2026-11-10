"""AutoDex 논문(arXiv:2606.23689v2) 수치 재계산과 유의성 검정.

실행: python3 analysis/table_check.py  -> analysis/table_check_results.json (1초 미만, numpy/scipy 1스레드)
입력은 논문 본문·Fig. 5 표에 적힌 수치뿐이다. 원자료는 공개되지 않았다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, norm

OUT = Path(__file__).with_name("table_check_results.json")


def wilson(k, n, z=1.959964):
    """Wilson 점수 구간 (95%)."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    """두 비율 차이의 Newcombe(Wilson 결합) 구간, 단위 pp."""
    l1, u1 = [x / 100 for x in wilson(k1, n1)]
    l2, u2 = [x / 100 for x in wilson(k2, n2)]
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return round(100 * lo, 1), round(100 * hi, 1)


def fisher(k1, n1, k2, n2):
    tab = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(fisher_exact(tab)[1])


def holm(pvals):
    """Holm 보정 (입력 순서 유지)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


R = {}

# ---- T1. 전체 성공률과 손별 분해 (본문 3.5절) ----
n_all, k_all = 3593, 1912
R["overall"] = {"n": n_all, "success": k_all, "rate_pct": round(100 * k_all / n_all, 1),
                "wilson95": wilson(k_all, n_all), "fails": n_all - k_all}

n_alg, n_ins = 2226, 1367
# 본문 성공률(39.4%, 75.7%)을 정수 횟수로 되돌린다. 합이 본문 1,912와 맞는지 확인한다.
k_alg = round(0.394 * n_alg)
k_ins = round(0.757 * n_ins)
R["hand_split"] = {
    "allegro": {"n": n_alg, "k_from_rate": k_alg, "rate_check_pct": round(100 * k_alg / n_alg, 2),
                "wilson95": wilson(k_alg, n_alg)},
    "inspire": {"n": n_ins, "k_from_rate": k_ins, "rate_check_pct": round(100 * k_ins / n_ins, 2),
                "wilson95": wilson(k_ins, n_ins)},
    "sum_success_check": k_alg + k_ins, "sum_success_paper": k_all,
    "fisher_p_hand": fisher(k_ins, n_ins, k_alg, n_alg),
    "diff_pp_inspire_minus_allegro": round(100 * (k_ins / n_ins - k_alg / n_alg), 1),
    "newcombe95_pp": newcombe_diff(k_ins, n_ins, k_alg, n_alg),
    "note": "손은 물체 집합·세션·시기와 섞여 있어 손 효과를 분리할 수 없다(교란).",
}

# ---- T2. 데이터베이스 검증 효과 (Fig. 5 우측, 515회, 20 물체) ----
# 두 팔(검증 DB, 모델 선별 DB)에 몇 회씩 갔는지 논문에 없다. 분할을 바꿔 민감도를 본다.
rows = []
for n1 in (100, 150, 200, 257, 258, 300, 400):
    n2 = 515 - n1
    k1 = round(0.76 * n1)
    k2 = round(0.34 * n2)
    rows.append({"n_validated": n1, "n_modelscreened": n2,
                 "k_validated": k1, "k_modelscreened": k2,
                 "rate_v_pct": round(100 * k1 / n1, 1), "rate_m_pct": round(100 * k2 / n2, 1),
                 "wilson_v": wilson(k1, n1), "wilson_m": wilson(k2, n2),
                 "fisher_p": fisher(k1, n1, k2, n2),
                 "newcombe_diff_pp": newcombe_diff(k1, n1, k2, n2)})
R["validation_effect_split_sensitivity"] = rows
R["validation_effect_note"] = ("분할을 어떻게 하든 p는 10^-20 이하다. 문제는 검정력이 아니라 "
                               "독립성이다: 같은 20개 물체·자세에 반복 시험했을 가능성, 라벨 규칙 "
                               "동일성, 기준선 선택 순서가 검정에 들어가지 않았다.")

# Fig. 5 우측 범주별 값(repo rebuttal_fig.py에 들어 있음; 개수는 없음)
cats = ["Wood", "Plastic", "Metal", "Paper", "Table", "Wall", "Clutt.", "Light", "Middle", "Heavy"]
sim = [19.5, 51.7, 30.9, 42.6, 36.8, 13.5, 6.7, 59.1, 18.5, 22.5]
sim_e = [8.3, 12.6, 7.7, 12.4, 5.7, 11.0, 12.6, 9.0, 8.5, 6.7]
real = [67.5, 73.3, 75.7, 90.3, 81.7, 67.6, 86.7, 73.7, 73.8, 83.9]
real_e = [14.5, 11.2, 13.8, 10.4, 7.9, 15.1, 17.2, 9.9, 11.0, 12.9]
# 오차막대가 이항 표준오차인지 확인. 각 축(재질 4, 장면 3, 무게 3)은 같은 시험을 나눈 것이므로
# 한 arm당 n을 257로 두고 축별 범주 크기를 잡는다(재질 ≈ 64, 장면·무게 ≈ 86). 실제 n은 논문에 없다.
groups = {"Wood": 64, "Plastic": 64, "Metal": 64, "Paper": 64, "Table": 86, "Wall": 86, "Clutt.": 86,
          "Light": 86, "Middle": 86, "Heavy": 86}
cat_check = []
for c, p, e in zip(cats, real, real_e):
    n_guess = groups[c]
    se_bin = 100 * math.sqrt((p / 100) * (1 - p / 100) / n_guess)
    cat_check.append({"cat": c, "real_pct": p, "errbar_pp": e, "n_assumed_per_arm": n_guess,
                      "binomial_se_pp": round(se_bin, 1)})
R["fig5_right_category_check"] = {
    "values_source": "repo rebuttal_fig.py (Fig. 5 우측과 같은 수치로 보임: 막대 높이 일치)",
    "rows": cat_check,
    "note": ("오차막대(7.9–17.2 pp)는 범주별 이항 SE(n=64 또는 86 가정, 약 2.3–6 pp)보다 크다. "
             "따라서 물체 간 분산 같은 다른 정의일 가능성이 높다. 정의는 논문에 없다. "
             "세 축은 같은 시험을 나눈 것이라 범주끼리 독립이 아니다. 범주별 n이 없어 CMH와 Holm 보정은 계산할 수 없다."),
}
R["fig5_right_cmh_holm"] = "not computable: per-category counts absent"

# ---- T3. 전체 대 teleop 성공률 (71.7% 대 50.7%) ----
# 두 캠페인의 n이 없다. 민감도로 보인다.
sens = []
for n in (50, 100, 200, 500):
    k_t = round(0.717 * n)
    k_a = round(0.507 * n)
    sens.append({"n_each": n, "k_tele": k_t, "k_autodex": k_a,
                 "fisher_p": fisher(k_t, n, k_a, n),
                 "newcombe_diff_pp_teleop_minus_autodex": newcombe_diff(k_t, n, k_a, n)})
R["teleop_vs_autodex_success_sensitivity"] = sens
R["teleop_vs_autodex_note"] = ("50.7%는 AutoDex 전체 3,593회 53.2%와도 다르다(캠페인 정의 차이). "
                               "손 분해가 없어 조정 비교도 불가능하다.")

# ---- T4. 사람 개입 31회 / 500회 ----
R["interventions"] = {"n": 500, "k": 31, "rate_pct": round(100 * 31 / 500, 2),
                      "wilson95": wilson(31, 500),
                      "human_h": 0.2,
                      "human_s_per_intervention_if_all_time_is_intervention":
                          round(0.2 * 3600 / 31, 1),
                      "note": ("0.2 h = 720 s를 31회 개입에 나누면 회당 약 23 s다. 개입 정의(무엇을 "
                               "세었는지)는 본문에 없다.")}

# ---- T5. 시간 예산 ----
loop_mean = 48.2
n_trials = 500
loop_total_h = loop_mean * n_trials / 3600
wall = 10.3
R["time_budget"] = {
    "loop_mean_s": loop_mean,
    "loop_total_h_500": round(loop_total_h, 2),
    "wall_h_500": wall,
    "non_loop_h": round(wall - loop_total_h, 2),
    "non_loop_share_pct": round(100 * (wall - loop_total_h) / wall, 1),
    "non_loop_per_trial_s": round((wall - loop_total_h) * 3600 / n_trials, 1),
    "loop_component_sum_s": round(24.8 + 11.9 + 7.8 + 3.8, 1),
    "loop_component_shares_pct": [51.5, 24.6, 16.1, 7.8],
    "trials_per_wall_h_autodex": round(n_trials / wall, 1),
    "trials_per_wall_h_teleop_v2": round(n_trials / 28.4, 1),
    "trials_per_human_h_autodex": round(n_trials / 0.2, 0),
    "trials_per_human_h_teleop": round(n_trials / 9.3, 1),
    "wall_ratio_v2": round(28.4 / 10.3, 2),
    "wall_ratio_v1": round(49.4 / 10.3, 2),
    "human_ratio": round(9.3 / 0.2, 1),
    "teleop_s_per_trial_human": round(9.3 * 3600 / n_trials, 1),
    "autodex_s_per_trial_human": round(0.2 * 3600 / n_trials, 2),
    "note": ("루프 밖 시간 3.6 h(35%, 회당 약 26 s)는 본문에 이름이 없다. 리셋 시간이 여기 들어가는지 "
             "논문은 밝히지 않는다. 28.4 h(teleop 벽시계)는 '8시간 근무일' 가정이라 하지만, "
             "9.3 h 인적 시간과 8 h/일로 28.4 h가 재현되지 않는다."),
}

# ---- T6. 카메라 수 ADD-S (Fig. 6 우측): 평균만 있고 분산이 없다 ----
R["camera_add_s"] = {"k": [2, 4, 8, 16], "mean_mm_reported": {"2": 14.3, "16": 0.5},
                     "n_trials": 51, "n_objects": 5,
                     "note": "분산·분위수가 없어 검정 불가. 'k=2 꼬리 실패'는 평균을 끌어올린 것으로만 해석 가능."}

# ---- T7. 리셋 대조군 ----
R["reset_ejection"] = {"reported_pct": 5.3, "n": "not reported",
                       "note": "분모가 없어 구간을 만들 수 없다. 5.3%는 n=57이면 3/57, n=94면 5/94 등 여러 값에 가능하다."}

# ---- T8. 데이터셋 README와 본문 대조 (DATASET.md, 저장소 기준) ----
R["dataset_md_vs_paper"] = {
    "dataset_md_counts": {"selected_100 (Allegro, legacy)": 1656, "corl_selected_100 (Allegro)": 574,
                          "selected_100_inspire (Inspire)": 384},
    "allegro_sum_dataset_md": 1656 + 574,
    "allegro_paper": 2226,
    "inspire_dataset_md": 384,
    "inspire_paper": 1367,
    "total_dataset_md": 1656 + 574 + 384,
    "total_paper": 3593,
    "note": "공개 설명서의 Inspire 수(384)는 본문 1,367과 다르다. 시점과 품질 필터(pose outlier 격리)가 원인일 수 있으나 논문은 설명하지 않는다.",
}

# ---- T9. 라벨 오차 구간 (성공 라벨을 맞다고 가정했을 때 관측 성공률 범위) ----
# 자동 라벨 오분류율 e0(거짓 성공), e1(거짓 실패)이 있으면 관측 성공률 q = e0 + (1-e0-e1) p.
# 관측 53.2%에서 참 성공률 p = (q - e0) / (1 - e0 - e1).
qobs = k_all / n_all
lab = []
for e0, e1 in ((0.0, 0.0), (0.02, 0.02), (0.05, 0.05), (0.10, 0.02), (0.02, 0.10), (0.10, 0.10)):
    p_true = (qobs - e0) / (1 - e0 - e1)
    lab.append({"e0_false_success": e0, "e1_false_fail": e1, "implied_true_rate_pct": round(100 * p_true, 1)})
R["label_error_bounds_on_overall_rate"] = lab

# ---- Holm: 손 분해·데이터셋 비교 등 p값 묶음 (검정 수는 3개) ----
pv = [R["hand_split"]["fisher_p_hand"], R["teleop_vs_autodex_success_sensitivity"][2]["fisher_p"],
      R["validation_effect_split_sensitivity"][3]["fisher_p"]]
R["holm_example_three_tests"] = {"p_raw": [float(x) for x in pv],
                                 "p_holm": [float(x) for x in holm(pv)],
                                 "labels": ["hand (Inspire vs Allegro)", "teleop vs AutoDex (n=200)",
                                            "validated vs model-screened (n=257/258)"]}

OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: R[k] for k in ("overall", "hand_split", "interventions", "time_budget")},
                 indent=1, ensure_ascii=False, default=float)[:3000])
print("written", OUT)
