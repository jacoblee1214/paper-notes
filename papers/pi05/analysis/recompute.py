"""pi0.5 논문의 수치를 CPU로 재계산한다 (재현 아님).

입력: 논문 본문·부록의 텍스트 수치와, 그림 10·12에서 눈으로 읽은 근사값(±2 pt 내외).
출력: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
res = {}

# 1. 평가 설계에서 나오는 값 (본문·부록 B 그대로)
rubric_max = {"dishes_in_sink": 8, "items_in_drawer": 4, "laundry_basket": 3, "make_bed": 5}
trials_per_task = 10
n_tasks = 4
res["design"] = {
    "rubric_max_points": rubric_max,
    "episodes_per_policy_standard": trials_per_task * n_tasks,  # 부록 B: 40
    "locations_for_standard_eval": 12,  # 3 실가정 + 3 kitchen + 3 bedroom 모의 (부록 B 기준)
    "language_following_chance": 1 / 5,  # 물체 5개, 부록 C
}

# 2. 그림 10 (mock homes, 평균 task progress) 근사 판독. (평균 %, 95% 반폭 % 추정)
fig10 = {"pi05": (78, 4), "no_WD": (73, 5.5), "no_CE": (51, 4.5), "no_ME": (54, 4.5), "no_CE_or_ME": (40, 4.5)}
n = 40  # 정책당 40 평가 (4 과제 x 10)
pvals_printed = {"no_WD": 0.385, "no_CE": 0.001, "no_ME": 0.001, "no_CE_or_ME": 0.001}  # "p<0.001"은 상한으로 사용
res["fig10_check"] = {}
for k in ["no_WD", "no_CE", "no_ME", "no_CE_or_ME"]:
    m1, h1 = fig10["pi05"]
    m2, h2 = fig10[k]
    diff = m1 - m2
    # 해석 A: 막대 = 95% CI  -> SD = (반폭/1.96) * sqrt(n)
    # 해석 B: 막대 = 표준오차 -> SD = 반폭 * sqrt(n)
    out = {"diff_pt": diff}
    for label, scale in [("A_bar_is_95CI", 1 / 1.96), ("B_bar_is_SE", 1.0)]:
        se1 = h1 * scale
        se2 = h2 * scale
        se_d = math.sqrt(se1**2 + se2**2)
        t = diff / se_d
        df = (se1**2 + se2**2) ** 2 / ((se1**4) / (n - 1) + (se2**4) / (n - 1))
        p = 2 * stats.t.sf(abs(t), df)
        out[label] = {"welch_t": round(t, 2), "df": round(df, 1), "p_two_sided": round(p, 4)}
    out["printed_p"] = pvals_printed[k]
    res["fig10_check"][k] = out

# 3. 그림 12 (π0 계열 비교, 과제별 근사값, %). 검정 표시가 없다.
fig12 = {
    "dishes_in_sink": {"pi05": 91, "fastflow_nohl": 55, "pi0_300k": 41, "pi0_80k": 19},
    "items_in_drawer": {"pi05": 85, "fastflow_nohl": 71, "pi0_300k": 32, "pi0_80k": 5},
    "laundry_basket": {"pi05": 80, "fastflow_nohl": 73, "pi0_300k": 31, "pi0_80k": 39},
    "make_bed": {"pi05": 58, "fastflow_nohl": 48, "pi0_300k": 20, "pi0_80k": 14},
}
# 과제 평균 (4 과제 동일 가중 가정) 와 π0.5 대비 차이
means = {k: round(sum(v[k] for v in fig12.values()) / 4, 1) for k in ["pi05", "fastflow_nohl", "pi0_300k", "pi0_80k"]}
res["fig12_read"] = {"per_task_read": fig12, "task_mean_equal_weight": means}

# 4. 언어 추종 chance(20%)와 Fig 11 OOD 값 비교 (근사) -- 텍스트 주장 "no WD가 OOD 추종을 크게 낮춘다"
ood_follow = {"pi05": 94, "no_WD": 80, "no_CE": 67, "no_ME": 33, "no_ME_or_CE": 33}
res["fig11_ood_follow_read"] = ood_follow
res["fig11_ood_follow_read"]["chance"] = 20

# 5. 실제 승인 가능한 n: 정책당 40 에피소드에서 rubric 평균의 표준오차 상한 (점수 SD 가정 50%, 최대 SD)
se_max = 0.5 * 100 / math.sqrt(40)
res["sampling_note"] = {
    "se_of_mean_if_per_episode_sd_50pct_of_scale": round(se_max, 2),
    "meaning": "비율 척도 점수의 SD가 50pt이면 n=40 평균의 SE는 약 7.9pt. 그림 막대 폭과 비교하기 위한 상한 감각치",
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res["fig10_check"], ensure_ascii=False, indent=1))
print("fig12 task means:", means)
print("wrote", OUT)
