"""Visuo-Tactile World Models (arXiv:2602.06001 v1) 본문 수치를 다시 맞춰 본다.

입력: 본문에 적힌 값만 사용한다(출처 주석). 그림 4와 그림 6의 막대 값과 시행 수는 본문에 없다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 실로봇 성공률의 분모는 본문의 "five trials per task"(§4.2)와 새 작업 "ten trials"(§4.3)다.
- 짝지은 t 검정의 자유도(df)는 논문에 없다. 보고된 t와 p 기준으로 가능한 df 범위만 역산한다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}

# §4.2 Figure 7(left) 본문 인용: 개선폭(percentage point). 원 막대 값은 그림에만 있다.
PLAN_DELTA_PP = {"push_fruits": 10, "reach_push": 35, "wipe_cloth": 31, "stack_cubes": 11}
PLAN_TRIALS = 5

# §4.3 새 작업 성공률 77% (10회 시행, 본문).
NOVEL_RATE = 0.77
NOVEL_TRIALS = 10


def feasible_pp_from_k_over_n(n):
    """n회 시행의 성공률 차이(절대 %p)로 가능한 값 집합."""
    vals = set()
    for a in range(n + 1):
        for b in range(n + 1):
            vals.add(round(100 * (b - a) / n, 1))
    return sorted(vals)


def rel_feasible(n):
    """n회 시행의 상대 개선(%) 가능한 값 집합 (a>0)."""
    vals = set()
    for a in range(1, n + 1):
        for b in range(n + 1):
            if b > a:
                vals.add(round(100 * (b - a) / a, 1))
    return sorted(vals)


# 1) 실로봇 개선폭이 5회 시행으로 나올 수 있는가
abs_ok = feasible_pp_from_k_over_n(PLAN_TRIALS)
rel_ok = rel_feasible(PLAN_TRIALS)
plan_check = {}
for task, d in PLAN_DELTA_PP.items():
    plan_check[task] = {
        "reported_pp": d,
        "possible_as_absolute_5_trials": d in abs_ok,
        "possible_as_relative_5_trials": any(abs(d - r) < 0.05 for r in rel_ok),
    }
RES["planning_delta_consistency_5_trials"] = plan_check
RES["abs_possible_5_trials"] = abs_ok

# 2) 77%는 10회 시행으로 가능한가
RES["novel_task_77pct_possible_10_trials"] = any(
    abs(NOVEL_RATE - k / NOVEL_TRIALS) < 0.005 for k in range(NOVEL_TRIALS + 1)
)
RES["novel_task_nearest_k_of_10"] = round(NOVEL_RATE * NOVEL_TRIALS, 2)

# 3) 짝지은 t 검정에서 보고된 (t, p) 쌍과 맞는 자유도 범위 역산 (양측)
#    p 표기는 "<" 또는 정확값. 허용 구간 = 본문 표기와 일치하는 df
def df_range(t, p_bound, op):
    ok = []
    for df in range(2, 201):
        p = 2 * stats.t.sf(abs(t), df)
        if op == "<" and p < p_bound:
            ok.append(df)
        if op == "=" and abs(p - p_bound) < 0.005:
            ok.append(df)
    return (min(ok), max(ok)) if ok else None

TESTS = {
    "object_permanence": [
        ("place_fruits", 4.38, 0.001, "<"),
        ("push_fruits", 6.06, 1e-6, "<"),
        ("cube_stacking", 2.40, 0.05, "<"),
    ],
    "causal_compliance": [
        ("place_fruits", 3.66, 0.001, "<"),
        ("push_fruits", 2.28, 0.05, "<"),
        ("wipe_cloth", 2.99, 0.01, "<"),
        ("cube_stacking", 1.75, 0.09, "="),
        ("scribble_marker", -1.22, 0.23, "="),
    ],
}
df_res = {}
for block, rows in TESTS.items():
    df_res[block] = {}
    for name, t, p, op in rows:
        df_res[block][name] = {"t": t, "p_reported": p, "op": op, "df_range_consistent": df_range(t, p, op)}
RES["implied_df"] = df_res

# 4) 상대 감소율 평균 검산 (§4.1 causal compliance, 보고값 43.6, 16.4, 66.1 + 나머지 두 작업)
known = [43.6, 16.4, 66.1]
need_sum = 29 * 5
RES["causal_compliance_mean_check"] = {
    "known_three_mean": round(float(np.mean(known)), 2),
    "sum_needed_for_29pct_over_5_tasks": need_sum,
    "remaining_two_must_sum_to": round(need_sum - sum(known), 1),
    "note": "stack은 '작은 개선', scribble은 '악화'라 두 값 합이 18.9 이상이어야 평균 29%가 된다. 그림 6 막대 값 없이는 확인 불가.",
}

# 5) 객체 영속성 평균 33%: 작업별 범위 18–47%와 평균 33%는 양립 가능(범위 검사만)
RES["object_permanence_range_check"] = {"min": 18, "max": 47, "mean": 33, "feasible_bounds": 18 <= 33 <= 47}

# 6) 데이터셋 datapoint 평균 검산 (부록 B.0.1)
RES["dataset_datapoints_check"] = {
    "train_112k_per_demo": round(112000 / 124, 1),
    "val_17k_per_demo": round(17000 / 26, 1),
    "frames_per_demo_6fps_40s": 40 * 6,
    "note": "1 demo당 datapoint 평균이 학습(약 903)과 검증(약 654)에서 다르다. datapoint 정의가 본문에 없다.",
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")

print("planning 5-trial check:")
for k, v in plan_check.items():
    print(" ", k, v)
print("77% possible with 10 trials:", RES["novel_task_77pct_possible_10_trials"], "(7.7/10)")
print("implied df:")
for block, d in df_res.items():
    for k, v in d.items():
        print(" ", block, k, v["df_range_consistent"])
print("causal mean check:", RES["causal_compliance_mean_check"])
print("datapoints:", RES["dataset_datapoints_check"])
print("written", OUT)
