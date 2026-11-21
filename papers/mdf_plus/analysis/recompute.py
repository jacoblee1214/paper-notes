"""MDF++ (arXiv:2611.09293 v1) 논문 표 수치 재계산과 검정.

실행: python3 analysis/recompute.py   (수 초, numpy·scipy 필요, 학습·데이터 다운로드 없음)
출력: analysis/recompute_results.json

가정(명시): 시뮬 표 1의 각 셀은 과제별 150회 시도다. 본문의 "150 trials for each method"를
이렇게 읽으면 모든 백분율이 k/150의 정수 k로 떨어진다(V1에서 확인). 실로봇은 과제당 20회다.
표 1의 MDF 계열 두 행은 과제별 최댓값(여러 시드·모드 중 최고)이므로 그 행의 검정은 참고용이다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_SIM = 150  # 과제당 시도 수 (가정)
N_REAL = 20  # 실로봇 과제당 시도 수 (논문 §5.4)

TASKS = ["PegInsert", "GearMesh", "NutThread", "BallSort", "ObjectSearch"]
# 표 1 (논문 Table 1), 과제 순서 TASKS
SIM = {
    "MDF++": [85.3, 78.7, 92.0, 93.3, 85.3],
    "MDF (adapted)": [89.3, 78.7, 92.7, 42.0, 39.3],
    "MDF + selective E2E": [86.7, 54.7, 92.0, 79.3, 44.7],
    "MDF++ (naive grad)": [56.7, 50.0, 90.7, 67.3, 56.0],
    "MDF++ (SIGReg)": [50.7, 32.0, 84.7, 48.0, 75.3],
    "MDF++ (frozen-ViT)": [86.7, 66.0, 89.3, 36.7, 14.7],
    "MDF++ (frozen-Sparsh)": [90.0, 64.7, 81.3, 40.0, 18.7],
    "MDF++ (policy)": [62.7, 64.7, 90.7, 90.0, 86.6],
    "UWM w/ tactile": [88.6, 72.0, 89.3, 53.3, 68.0],
    "DP w/ tactile": [88.7, 81.3, 91.3, 28.0, 54.0],
}
SIM_AVG_REPORTED = {
    "MDF++": 86.9, "MDF (adapted)": 68.4, "MDF + selective E2E": 71.5, "MDF++ (naive grad)": 64.1,
    "MDF++ (SIGReg)": 58.1, "MDF++ (frozen-ViT)": 58.7, "MDF++ (frozen-Sparsh)": 58.9,
    "MDF++ (policy)": 78.9, "UWM w/ tactile": 74.2, "DP w/ tactile": 68.7,
}
# 표 9 (논문 Table 9): 단계별 시도 수 (Stage 1, 4는 표에 열이 없음), 보고된 총점
REAL = {
    ("Bolt", "MDF++"): dict(s0=0, s2=0, s3=4, s5=16, total=1900, avg=95.0),
    ("Bolt", "UWM"): dict(s0=8, s2=0, s3=2, s5=10, total=1150, avg=57.5),
    ("Bolt", "MDF++(frozen-ViT)"): dict(s0=9, s2=3, s3=2, s5=6, total=930, avg=46.5),
    ("Nut", "MDF++"): dict(s0=1, s2=1, s3=10, s5=8, total=1660, avg=83.0),
    ("Nut", "UWM"): dict(s0=6, s2=0, s3=9, s5=5, total=1220, avg=61.0),
    ("Nut", "MDF++(frozen-ViT)"): dict(s0=16, s2=0, s3=1, s5=3, total=380, avg=19.0),
}
# 표 8 규칙 점수 (Stage 2 = 65, Stage 3 = 75, Stage 5 = 100). 출시 5점 보너스는 단계 3에서만 가능
STAGE_SCORE = {"s0": 0, "s2": 65, "s3": 75, "s5": 100}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table)[1])


def main():
    res = {"assumptions": {"sim_trials_per_task": N_SIM, "real_trials_per_policy_task": N_REAL}}

    # V1: 표 1 값이 k/150 정수 배인지, 평균이 보고값과 맞는지
    counts, max_dev, avg_check = {}, 0.0, {}
    for name, vals in SIM.items():
        ks = []
        for v in vals:
            k = v / 100 * N_SIM
            max_dev = max(max_dev, abs(k - round(k)))
            ks.append(int(round(k)))
        counts[name] = ks
        avg_check[name] = dict(recomputed=round(float(np.mean(vals)), 2),
                               reported=SIM_AVG_REPORTED[name],
                               ok=bool(abs(np.mean(vals) - SIM_AVG_REPORTED[name]) < 0.06))
    res["V1_sim_grid"] = dict(max_deviation_from_integer_count=round(max_dev, 3), counts=counts,
                              average_check=avg_check)

    # V2: 평균 차이 (본문 주장)
    base = np.mean(SIM["MDF++"])
    res["V2_claims"] = {
        "MDF++ minus frozen-ViT (pp)": round(base - np.mean(SIM["MDF++ (frozen-ViT)"]), 2),
        "MDF++ minus frozen-Sparsh (pp)": round(base - np.mean(SIM["MDF++ (frozen-Sparsh)"]), 2),
        "MDF++ minus naive grad (pp)": round(base - np.mean(SIM["MDF++ (naive grad)"]), 2),
        "MDF++ minus MDF adapted (pp)": round(base - np.mean(SIM["MDF (adapted)"]), 2),
        "MDF++ minus MDF++(policy) (pp)": round(base - np.mean(SIM["MDF++ (policy)"]), 2),
        "MDF++ minus DP (pp)": round(base - np.mean(SIM["DP w/ tactile"]), 2),
        "MDF++ minus UWM (pp)": round(base - np.mean(SIM["UWM w/ tactile"]), 2),
        "MDF+selective minus MDF adapted (pp)": round(np.mean(SIM["MDF + selective E2E"]) - np.mean(SIM["MDF (adapted)"]), 2),
    }

    # V3: 시뮬 셀별 Wilson 95% 구간과 Fisher 검정 (MDF++ 대 주요 기준선, 과제별 및 합산)
    rivals = ["MDF++ (policy)", "MDF++ (naive grad)", "MDF++ (frozen-ViT)", "MDF++ (frozen-Sparsh)",
              "UWM w/ tactile", "DP w/ tactile", "MDF (adapted)", "MDF++ (SIGReg)"]
    per_task = {}
    for rv in rivals:
        rows = []
        for t, task in enumerate(TASKS):
            k1, k2 = counts["MDF++"][t], counts[rv][t]
            rows.append(dict(task=task, k_mdf=k1, k_rival=k2,
                             diff_pp=round(100 * (k1 - k2) / N_SIM, 1),
                             p_fisher=round(fisher(k1, N_SIM, k2, N_SIM), 4)))
        per_task[rv] = rows
    res["V3_sim_per_task_fisher"] = per_task

    pooled = {}
    n_tot = N_SIM * len(TASKS)
    k_mdf = sum(counts["MDF++"])
    for rv in rivals:
        k_r = sum(counts[rv])
        lo1, hi1 = wilson(k_mdf, n_tot)
        lo2, hi2 = wilson(k_r, n_tot)
        pooled[rv] = dict(k_mdf=k_mdf, k_rival=k_r, n=n_tot,
                          mdf_pct=round(100 * k_mdf / n_tot, 2), rival_pct=round(100 * k_r / n_tot, 2),
                          diff_pp=round(100 * (k_mdf - k_r) / n_tot, 2),
                          p_fisher_pooled=float(f"{fisher(k_mdf, n_tot, k_r, n_tot):.3g}"),
                          mdf_wilson95=[round(lo1, 1), round(hi1, 1)],
                          rival_wilson95=[round(lo2, 1), round(hi2, 1)])
    res["V4_sim_pooled_over_tasks"] = pooled

    # V5: 시뮬 MDF++ 대 MDF (adapted)의 과제 수준 구간 (BallSort, ObjectSearch만 큰 차이)
    res["V5_mdf_wilson_per_task"] = {
        task: dict(k=counts["MDF++"][t], wilson95=[round(x, 1) for x in wilson(counts["MDF++"][t], N_SIM)])
        for t, task in enumerate(TASKS)
    }

    # K1: 표 9 실로봇 점수 재계산 (단계 점수 + 출시 보너스 잔차)
    k1 = {}
    for (task, pol), r in REAL.items():
        n = r["s0"] + r["s2"] + r["s3"] + r["s5"]
        base_score = sum(r[s] * STAGE_SCORE[s] for s in STAGE_SCORE)
        resid = r["total"] - base_score
        k1[f"{task}/{pol}"] = dict(trials=n, base_score=base_score, reported_total=r["total"],
                                   residual=resid, residual_over_5=resid / 5,
                                   reported_avg_check=round(r["total"] / 20, 2), reported_avg=r["avg"],
                                   stage3_trials_max_bonus=r["s3"])
    res["K1_real_score_recompute"] = k1

    # V6: 실로봇 성공률 (Stage 5 개수) 검정
    succ = {(t, p): REAL[(t, p)]["s5"] for (t, p) in REAL}
    real_fisher = {}
    for task in ["Bolt", "Nut"]:
        for rv in ["UWM", "MDF++(frozen-ViT)"]:
            k_m, k_r = succ[(task, "MDF++")], succ[(task, rv)]
            lo1, hi1 = wilson(k_m, N_REAL)
            lo2, hi2 = wilson(k_r, N_REAL)
            real_fisher[f"{task} MDF++ vs {rv}"] = dict(
                k_mdf=k_m, k_rival=k_r,
                diff_pp=round(100 * (k_m - k_r) / N_REAL, 1),
                p_fisher=round(fisher(k_m, N_REAL, k_r, N_REAL), 4),
                mdf_wilson95=[round(lo1, 1), round(hi1, 1)], rival_wilson95=[round(lo2, 1), round(hi2, 1)])
    # 두 과제 합산 (40회씩)
    for rv in ["UWM", "MDF++(frozen-ViT)"]:
        k_m = succ[("Bolt", "MDF++")] + succ[("Nut", "MDF++")]
        k_r = succ[("Bolt", rv)] + succ[("Nut", rv)]
        real_fisher[f"pooled MDF++ vs {rv}"] = dict(
            k_mdf=k_m, k_rival=k_r, n=2 * N_REAL,
            diff_pp=round(100 * (k_m - k_r) / (2 * N_REAL), 1),
            p_fisher=round(fisher(k_m, 2 * N_REAL, k_r, 2 * N_REAL), 4))
    res["V6_real_success_tests"] = real_fisher

    # V7: 실로봇 점수 분포 (단계 점수로 풀어 쓴 20개 점수)에 대한 Mann-Whitney
    def expand(task, pol):
        r = REAL[(task, pol)]
        scores = []
        for s in STAGE_SCORE:
            scores += [STAGE_SCORE[s]] * r[s]
        # 잔차는 출시 보너스로 가정하지 않고 그대로 두면 평균이 맞지 않으므로 평균 보정 대신 단계 점수만 쓴다
        return scores

    mw = {}
    for task in ["Bolt", "Nut"]:
        a = expand(task, "MDF++")
        for rv in ["UWM", "MDF++(frozen-ViT)"]:
            b = expand(task, rv)
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            mw[f"{task} MDF++ vs {rv}"] = dict(U=float(u.statistic), p=round(float(u.pvalue), 4),
                                               note="단계 점수 기준 재구성, 보너스 제외")
    res["V7_real_score_mannwhitney"] = mw

    # V8: 논문 본문 주장의 real-world 격차 (평균 점수, 두 과제 평균)
    mdf_avg = np.mean([95.0, 83.0])
    uwm_avg = np.mean([57.5, 61.0])
    res["V8_claims_real"] = dict(
        mdf_two_task_mean_score=mdf_avg, uwm_two_task_mean_score=uwm_avg,
        gap_pp=round(mdf_avg - uwm_avg, 2),
        bolt_success_gap_vs_frozen_pp=round(100 * (16 - 6) / 20, 1),
        note="초록의 '최대 50%'는 볼트 성공률 80% 대 30%(pp)와 맞는다. 본문의 '29%'는 평균 점수 격차 29.75와 맞지만 단위가 명시되지 않았다.",
    )

    # V9: 실로봇에서 차이를 검출하려면 과제당 몇 회가 필요한지 (정규 근사, 양측 5%, 검정력 80%)
    def n_per_arm(p1, p2, z_a=1.959964, z_b=0.841621):
        pbar = (p1 + p2) / 2
        num = z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        return int(np.ceil((num / (p1 - p2)) ** 2))
    res["V9_real_power"] = {
        "bolt 80 vs 50 (논문 성공률)": n_per_arm(0.80, 0.50),
        "nut 40 vs 25 (논문 성공률)": n_per_arm(0.40, 0.25),
        "bolt 80 vs 30 (프로즌 ViT)": n_per_arm(0.80, 0.30),
        "시뮬 150회, 86.9 vs 78.9 (정책 전용)": n_per_arm(0.869, 0.789),
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
