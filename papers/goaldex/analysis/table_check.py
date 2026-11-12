"""논문 표 재계산과 검정 (scipy, 약 2초).

입력: 논문 Table 1–5의 값을 PDF 렌더링 이미지에서 옮긴 값(아래 DATA). 원문 PDF의 표 페이지 2–8쪽 기준.
출력: analysis/table_check_results.json

검정:
- 실로봇 성공 수: Fisher 정확 검정, Wilson 95% 구간, 객체 간 이질성(카이제곱)
- 시뮬레이션 평균(3 시드): Welch t (n = 3, 표준편차는 모집단 기준 ddof=0으로 가정 후 표본 표준편차로 환산), Holm 보정
- 논문 "±" 표기 규칙(ddof=0)은 Table 5로 확인한다(평균 16.0 ± 1.67).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "table_check_results.json"
N_SEEDS = 3

# (평균, 표준편차) 쌍. 값은 논문 Table 1–4에서 옮김.
TABLE1 = {
    "Proprio only": (1.05, 0.00),
    "Pose: JP+P+GP": (8.64, 10.17),
    "Pose: JP+P+C+GP": (11.23, 9.65),
    "Pose: P+GP": (1.81, 0.65),
    "Pose: P+C+GP": (4.08, 1.46),
    "Image+GP (no proprio)": (0.01, 0.01),
    "Image+JP+GP": (69.20, 5.47),
    "Image-goal (ours)": (77.30, 3.25),
}
TABLE3 = {  # 정렬 전략별 CS, TTS, F1, PFR
    "Separate": {"CS": (56.99, 1.89), "TTS": (4.50, 1.12), "F1": (0.913, 0.004), "PFR": (10.50, 0.61)},
    "EMA": {"CS": (67.95, 6.80), "TTS": (5.65, 0.40), "F1": (0.857, 0.004), "PFR": (10.30, 0.58)},
    "Shared": {"CS": (67.18, 4.43), "TTS": (4.69, 0.42), "F1": (0.936, 0.002), "PFR": (9.32, 0.12)},
    "InfoNCE": {"CS": (63.01, 15.51), "TTS": (4.65, 0.37), "F1": (0.934, 0.001), "PFR": (8.11, 0.54)},
    "Ours (continuous)": {"CS": (77.30, 3.25), "TTS": (4.35, 0.24), "F1": (0.921, 0.000), "PFR": (8.80, 0.15)},
}
TABLE4 = {  # 25 objects
    "Privileged teacher": {"CS": (110.84, 3.98), "TTS": (2.42, 0.37)},
    "Image-goal (1x envs)": {"CS": (10.11, 2.06), "TTS": (6.74, 0.05)},
    "Image-goal (2x envs)": {"CS": (41.42, 0.00), "TTS": (5.30, 0.00)},
}
# 실로봇 Table 5: 객체별 (성공, 객체 낙하, 시간 초과, 예측 실패), 객체당 20회
TABLE5 = {
    "Cube": (18, 0, 0, 2),
    "3D Printed Toy": (16, 1, 2, 1),
    "Tablet Bottle": (13, 2, 1, 4),
    "Rubber Duck": (17, 0, 0, 3),
    "Globe": (16, 1, 1, 2),
}
TABLE5_MEAN_PAPER = {"succ": (16.0, 1.67), "drop": (0.8, 0.75), "timeout": (0.8, 0.75), "pfail": (2.4, 1.02)}
POSE_BASELINE_REAL = 27  # 논문 본문: "strongest pose-based baseline ... succeeds in only 27 of the 100 trials" (표 없음)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def welch_from_paper(m1, s1, m2, s2, n=N_SEEDS):
    # 논문 표준편차 = 모집단(ddof=0) 가정 → 표본 표준편차 = s * sqrt(n/(n-1))
    f = math.sqrt(n / (n - 1))
    t, p = stats.ttest_ind_from_stats(m1, s1 * f, n, m2, s2 * f, n, equal_var=False)
    return float(t), float(p)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    res = {}

    # 0) 논문 표준편차 규칙 확인 (Table 5, 객체별 20회)
    succ = [v[0] for v in TABLE5.values()]
    drop = [v[1] for v in TABLE5.values()]
    tout = [v[2] for v in TABLE5.values()]
    pf = [v[3] for v in TABLE5.values()]
    rule = {}
    for name, arr in (("succ", succ), ("drop", drop), ("timeout", tout), ("pfail", pf)):
        mean = sum(arr) / len(arr)
        sd0 = math.sqrt(sum((x - mean) ** 2 for x in arr) / len(arr))
        sd1 = math.sqrt(sum((x - mean) ** 2 for x in arr) / (len(arr) - 1))
        rule[name] = {"mean": round(mean, 2), "sd_ddof0": round(sd0, 2), "sd_ddof1": round(sd1, 2),
                      "paper": TABLE5_MEAN_PAPER[name]}
    res["table5_sd_convention"] = rule

    # 1) 실로봇: 합계, Wilson 구간, Fisher, 객체 간 카이제곱
    tot_s = sum(succ)
    tot_n = 20 * len(TABLE5)
    lo, hi = wilson(tot_s, tot_n)
    ours_tot = {"successes": tot_s, "n": tot_n, "rate": tot_s / tot_n, "wilson95": [lo, hi]}
    lo2, hi2 = wilson(POSE_BASELINE_REAL, 100)
    fis = stats.fisher_exact([[tot_s, tot_n - tot_s], [POSE_BASELINE_REAL, 100 - POSE_BASELINE_REAL]])
    res["real_world"] = {
        "ours": ours_tot,
        "pose_baseline_27_of_100_wilson95": [lo2, hi2],
        "fisher_ours_vs_pose_baseline_two_sided_p": float(fis[1]),
        "failure_counts_total": {"drop": sum(drop), "timeout": sum(tout), "prediction_failure": sum(pf)},
    }
    table = [[v[0], 20 - v[0]] for v in TABLE5.values()]
    chi2, p_chi, dof, _ = stats.chi2_contingency(table)
    res["real_world"]["object_heterogeneity_chi2"] = {"chi2": float(chi2), "dof": int(dof), "p": float(p_chi)}
    # 객체별 Fisher (대 나머지 80) + Holm
    per = {}
    raw = []
    names = list(TABLE5)
    for nm in names:
        s = TABLE5[nm][0]
        rest_s = tot_s - s
        rest_n = tot_n - 20
        _, p = stats.fisher_exact([[s, 20 - s], [rest_s, rest_n - rest_s]])
        raw.append(p)
    adj = holm(raw)
    for nm, p_raw, p_adj in zip(names, raw, adj):
        s = TABLE5[nm][0]
        l, h = wilson(s, 20)
        per[nm] = {"successes": s, "wilson95": [l, h], "fisher_vs_rest_p": p_raw, "holm_p": p_adj}
    res["real_world"]["per_object"] = per

    # 2) 시뮬레이션 Table 1: 이미지 목표 대 이미지+포즈 목표 (같은 입력, 목표 표현만 다름)
    pairs = {}
    for label, a, b in (
        ("Table1 image-goal vs image+JP+pose-goal (CS)", TABLE1["Image-goal (ours)"], TABLE1["Image+JP+GP"]),
        ("Table1 image-goal vs best pose-only baseline (CS)", TABLE1["Image-goal (ours)"], TABLE1["Pose: JP+P+C+GP"]),
    ):
        t, p = welch_from_paper(a[0], a[1], b[0], b[1])
        pairs[label] = {"diff": a[0] - b[0], "welch_t": t, "p_two_sided": p}
    res["sim_table1"] = pairs

    # 3 시드 검정력: 평균 차이 d를 검출할 확률 (정규 가정, 표본 표준편차 기준, 몬테카를로 40,000회)
    def power_3seed(diff, sd_pop_a, sd_pop_b, reps=40000):
        rng = np.random.default_rng(3)
        fa = math.sqrt(N_SEEDS / (N_SEEDS - 1))
        a = rng.normal(0, sd_pop_a * fa, (reps, N_SEEDS))
        b = rng.normal(diff, sd_pop_b * fa, (reps, N_SEEDS))
        pv = stats.ttest_ind(b, a, axis=1, equal_var=False).pvalue
        return float(np.mean(pv < 0.05))
    res["power_3seed_CS"] = {
        "diff_8.1_vs_image_pose_goal": power_3seed(8.1, 3.25, 5.47),
    }
    lo_, hi_ = 1.0, 40.0
    for _ in range(18):
        mid = 0.5 * (lo_ + hi_)
        if power_3seed(mid, 3.25, 5.47, reps=20000) < 0.8:
            lo_ = mid
        else:
            hi_ = mid
    res["power_3seed_CS"]["MDE_80pct_diff_pooled_sd_like_T1"] = hi_

    # Table 3: ours 대 대안 4개 (CS), Holm
    ours = TABLE3["Ours (continuous)"]
    alts = [k for k in TABLE3 if not k.startswith("Ours")]
    raw = []
    rows = {}
    for k in alts:
        t, p = welch_from_paper(ours["CS"][0], ours["CS"][1], TABLE3[k]["CS"][0], TABLE3[k]["CS"][1])
        raw.append(p)
        rows[k] = {"CS_diff": ours["CS"][0] - TABLE3[k]["CS"][0], "welch_t": t, "p_raw": p}
    for k, p_adj in zip(alts, holm(raw)):
        rows[k]["p_holm"] = p_adj
    res["sim_table3_CS_vs_ours"] = rows

    # Table 4: 2x 환경 대 1x: 0.00 표준편차 표기 확인용
    res["sim_table4_2x_sd_zero_flag"] = TABLE4["Image-goal (2x envs)"]["CS"][1] == 0.0

    # 3) 재계산 대조: 논문 본문 수치
    res["checks"] = {
        "image_goal_row_identical_in_T1_T2_T3": [TABLE1["Image-goal (ours)"], TABLE3["Ours (continuous)"]["CS"]],
        "real_world_mean_80_of_100": tot_s == 80,
        "sim_seeds": N_SEEDS,
        "episodes_per_policy_per_seed": 64 * 2,
    }

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
