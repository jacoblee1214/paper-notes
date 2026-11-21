"""Table 5 재계산: 평균, Wilson 구간, 분모 검사, 상관, 주장 검사.

입력은 논문 Table 5(p.11)와 부록 D의 에피소드 수뿐이다. 실행: python3 analysis/recompute.py
출력: analysis/recompute_results.json (수 초, CPU).
"""
import json
import math
import os
from itertools import product

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

# Table 5 전사. 값 순서: pre_iid_train, pre_rep_train, pre_iid_extrap, post_train, post_extrap (%)
SCREW = {
    "EXPERT-ONLY": {2: (6, 46, 0, 100, 98), 4: (6, 54, 8, 100, 73), 8: (12, 32, 2, 100, 88), 16: (16, 28, 0, 100, 45)},
    "FROZEN-LM": {2: (0, 38, 0, 100, 87), 4: (0, 30, 0, 99, 70), 8: (0, 6, 0, 0, 0), 16: (0, 0, 0, 0, 0)},
    "FULL-FT": {2: (0, 40, 0, 100, 73), 4: (0, 32, 0, 67, 15), 8: (0, 6, 0, 33, 0), 16: (0, 10, 0, 46, 0)},
    "LORA": {2: (6, 44, 2, 100, 63), 4: (0, 34, 0, 33, 5), 8: (24, 6, 0, 17, 0), 16: (50, 24, 0, 50, 0)},
    "LORA-LM": {2: (0, 42, 0, 100, 85), 4: (10, 24, 0, 2, 3), 8: (0, 0, 0, 0, 0), 16: (30, 8, 0, 38, 0)},
}
COFFEE = {
    "EXPERT-ONLY": {4: (32, 6, 20, 100, 94), 8: (24, 6, 6, 99, 87), 14: (28, 12, 28, 97, 83), 18: (20, 8, 22, 95, 90)},
    "FROZEN-LM": {4: (46, 18, 40, 98, 85), 8: (60, 42, 38, 94, 89), 14: (72, 52, 44, 91, 62), 18: (78, 64, 56, 78, 43)},
    "FULL-FT": {4: (40, 18, 22, 97, 81), 8: (72, 30, 14, 96, 49), 14: (80, 54, 38, 93, 31), 18: (48, 44, 22, 81, 20)},
    "LORA": {4: (36, 16, 16, 95, 83), 8: (72, 28, 18, 95, 35), 14: (72, 62, 14, 65, 17), 18: (66, 76, 14, 55, 7)},
    "LORA-LM": {4: (82, 18, 40, 95, 91), 8: (74, 40, 26, 97, 63), 14: (74, 58, 22, 94, 33), 18: (88, 72, 14, 82, 34)},
}
TASKS = {"ScrewLightbulb": SCREW, "CoffeePressButton": COFFEE}
COLS = ["pre_iid_train", "pre_rep_train", "pre_iid_extrap", "post_train", "post_extrap"]

# 부록 D의 에피소드 수 가정. 시험 대상 분모 후보와 맞는지 검사한다.
#   pre-RL: 한 placement당 25 (Screw extrap은 20). 학습 placement 2개 합산 50.
#   post-RL: 5 에피소드 x 5회 평가(400k..500k, 25k 간격) = 25/시드/placement. 3시드 평균.


def granularity_check():
    """pre-RL 열(단일 시도)이 정수 개수 k/N을 반올림한 값과 맞는 분모 N을 찾는다.

    기준: 어떤 정수 k(0..N)가 있어 |100k/N - v| <= 0.5 이면 그 N은 가능하다.
    post-RL 열은 3시드 평균이라 이 방법으로 분모를 판정하지 않는다.
    """
    out = {}
    cands = [20, 25, 30, 40, 50, 60, 75, 100, 120, 150]
    for tname, tab in TASKS.items():
        for ci, col in enumerate(COLS[:3]):
            vals = [row[ci] for rec in tab.values() for row in tab_rows(tab, rec)]
            ok = [n for n in cands if all(any(abs(100.0 * k / n - v) <= 0.5 for k in range(n + 1)) for v in vals)]
            out[f"{tname}.{col}"] = ok
    return out


def tab_rows(tab, rec):
    # tab_rows(tab, rec) returns list of tuples for the recipe (rec is a dict of ckpt->row)
    return list(rec.values())


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def flat(tab):
    rows = []
    for rec, ck in tab.items():
        for step, row in ck.items():
            rows.append((rec, step, row))
    return rows


def main():
    res = {}
    res["granularity_candidates"] = granularity_check()

    # 1. 레시피별 평균 (체크포인트 4개 평균)
    means = {}
    for tname, tab in TASKS.items():
        means[tname] = {}
        for rec, ck in tab.items():
            arr = np.array(list(ck.values()), dtype=float)
            means[tname][rec] = {c: round(float(arr[:, i].mean()), 2) for i, c in enumerate(COLS)}
    res["recipe_means"] = means
    # 두 과제 평균
    overall = {}
    for rec in SCREW:
        overall[rec] = {c: round((means["ScrewLightbulb"][rec][c] + means["CoffeePressButton"][rec][c]) / 2, 2) for c in COLS}
    res["recipe_means_both_tasks"] = overall

    # 2. 주장 검사
    claims = {}
    # (a) Expert-only 학습 성공률 95-100% (두 과제)
    claims["C_expert_post_train_range"] = {t: [min(r[3] for r in TASKS[t]["EXPERT-ONLY"].values()), max(r[3] for r in TASKS[t]["EXPERT-ONLY"].values())] for t in TASKS}
    # (b) Screw 8k: Expert 100 / 88, 나머지 학습 성공률 <= 33
    claims["C_screw8k"] = {rec: SCREW[rec][8][3:] for rec in SCREW}
    # (c) "Frozen-LM이 Coffee SFT-time 성공률이 가장 높다" 검사 (i.i.d. 학습, 반복 잡음, 체크포인트별 최대)
    coffee_max = {}
    for col_i, col in [(0, "pre_iid_train"), (1, "pre_rep_train")]:
        best = max(((rec, step, row[col_i]) for rec in COFFEE for step, row in COFFEE[rec].items()), key=lambda x: x[2])
        coffee_max[col] = best
        coffee_max[col + "_recipe_mean"] = {rec: round(float(np.mean([row[col_i] for row in COFFEE[rec].values()])), 2) for rec in COFFEE}
    claims["C_coffee_sft_time_highest"] = coffee_max
    # (d) 부록 F: Expert-only 반복 잡음 6-12% (Coffee) / 전 체크포인트 범위
    claims["F_expert_rep_range"] = {t: [min(r[1] for r in TASKS[t]["EXPERT-ONLY"].values()), max(r[1] for r in TASKS[t]["EXPERT-ONLY"].values())] for t in TASKS}
    claims["F_expert_iid_range"] = {t: [min(r[0] for r in TASKS[t]["EXPERT-ONLY"].values()), max(r[0] for r in TASKS[t]["EXPERT-ONLY"].values())] for t in TASKS}
    # (e) LoRA 18k: 반복 잡음 최고(76), 학습 후 최저 (Coffee 안에서만)
    all_rep = [(t, rec, step, row[1]) for t in TASKS for rec in TASKS[t] for step, row in TASKS[t][rec].items()]
    claims["F_max_rep"] = max(all_rep, key=lambda x: x[3])
    coffee_post_min = min(((rec, step, row[3]) for rec in COFFEE for step, row in COFFEE[rec].items()), key=lambda x: x[2])
    all_post_min = min(((t, rec, step, row[3]) for t in TASKS for rec in TASKS[t] for step, row in TASKS[t][rec].items()), key=lambda x: x[3])
    claims["F_coffee_post_min"] = coffee_post_min
    claims["F_all_post_min"] = all_post_min
    # (f) "LoRA가 가장 덜 조종 가능" - 과제 평균 기준으로 학습/추가 평가 각각
    lm_recipes = ["FULL-FT", "LORA", "LORA-LM"]
    claims["C_worst_recipe_extrap_both"] = sorted(((rec, overall[rec]["post_extrap"]) for rec in overall), key=lambda x: x[1])
    claims["C_worst_recipe_train_both"] = sorted(((rec, overall[rec]["post_train"]) for rec in overall), key=lambda x: x[1])
    claims["C_lm_recipes_vs_frozen"] = {rec: overall[rec] for rec in lm_recipes + ["FROZEN-LM"]}
    res["claims"] = claims

    # 3. Wilson 구간 (pre-RL 단일 시도). 분모는 granularity 결과와 맞춘 가정을 쓴다.
    #    train: 두 placement x 25 = 50, extrap: Coffee 2 x 25 = 50, Screw 2 x 20 = 40
    wil = {}
    for tname, tab in TASKS.items():
        den_train, den_extrap = 50, (50 if tname == "CoffeePressButton" else 40)
        wil[tname] = {}
        for rec, ck in tab.items():
            for step, row in ck.items():
                k_iid = round(row[0] * den_train / 100)
                k_rep = round(row[1] * den_train / 100)
                k_ex = round(row[2] * den_extrap / 100)
                wil[tname][f"{rec}@{step}"] = {
                    "iid_train_k_of_n": [k_iid, den_train, [round(x * 100, 1) for x in wilson(k_iid, den_train)]],
                    "rep_train_k_of_n": [k_rep, den_train, [round(x * 100, 1) for x in wilson(k_rep, den_train)]],
                    "iid_extrap_k_of_n": [k_ex, den_extrap, [round(x * 100, 1) for x in wilson(k_ex, den_extrap)]],
                }
    res["wilson_pre_rl"] = wil

    # 4. 상관: 체크포인트-레시피 20개 (과제마다). Pearson과 Spearman, 양측 p.
    #    점들은 같은 레시피 안에서 독립이 아니므로 p는 참고용이다.
    corr = {}
    for tname, tab in TASKS.items():
        pts = [row for rec in tab for row in tab[rec].values()]
        arr = np.array(pts, dtype=float)
        corr[tname] = {}
        for pre_i, pre_name in [(0, "pre_iid_train"), (1, "pre_rep_train"), (2, "pre_iid_extrap")]:
            for post_i, post_name in [(3, "post_train"), (4, "post_extrap")]:
                x, y = arr[:, pre_i], arr[:, post_i]
                pr, pp = stats.pearsonr(x, y)
                sr, sp = stats.spearmanr(x, y)
                corr[tname][f"{pre_name}~{post_name}"] = {
                    "pearson_r": round(float(pr), 3), "pearson_p": round(float(pp), 4),
                    "spearman_rho": round(float(sr), 3), "spearman_p": round(float(sp), 4),
                    "n": len(x),
                }
    # 두 과제 합산 (40 점)
    allp = np.array([row for t in TASKS for rec in TASKS[t] for row in TASKS[t][rec].values()], dtype=float)
    corr["pooled"] = {}
    for pre_i, pre_name in [(0, "pre_iid_train"), (1, "pre_rep_train"), (2, "pre_iid_extrap")]:
        for post_i, post_name in [(3, "post_train"), (4, "post_extrap")]:
            pr, pp = stats.pearsonr(allp[:, pre_i], allp[:, post_i])
            sr, sp = stats.spearmanr(allp[:, pre_i], allp[:, post_i])
            corr["pooled"][f"{pre_name}~{post_name}"] = {
                "pearson_r": round(float(pr), 3), "pearson_p": round(float(pp), 4),
                "spearman_rho": round(float(sr), 3), "spearman_p": round(float(sp), 4), "n": len(allp)}
    res["correlations"] = corr

    # 4b. 강건성: EXPERT-ONLY를 뺀 4개 레시피 (20점 중 15점)
    robust = {}
    for tname, tab in TASKS.items():
        pts = np.array([row for rec in tab if rec != "EXPERT-ONLY" for row in tab[rec].values()], dtype=float)
        robust[tname] = {}
        for pre_i, pre_name in [(0, "pre_iid_train"), (1, "pre_rep_train")]:
            pr, pp = stats.pearsonr(pts[:, pre_i], pts[:, 3])
            sr, sp = stats.spearmanr(pts[:, pre_i], pts[:, 3])
            robust[tname][f"{pre_name}~post_train_wo_expert"] = {
                "pearson_r": round(float(pr), 3), "pearson_p": round(float(pp), 4),
                "spearman_rho": round(float(sr), 3), "spearman_p": round(float(sp), 4), "n": len(pts)}
    res["correlations_without_expert_only"] = robust

    # 5. 레시피 간 비교 (체크포인트 4개의 쌍체 구조). 레시피 1개 SFT 시드라서 체크포인트는 독립 표본이 아니다.
    #    Expert-only 대 Frozen-LM의 post_extrap 차이: 같은 체크포인트끼리 짝지음 (참고용).
    pairs = {}
    for tname, tab in TASKS.items():
        e = np.array([tab["EXPERT-ONLY"][s][4] for s in sorted(tab["EXPERT-ONLY"])], dtype=float)
        f = np.array([tab["FROZEN-LM"][s][4] for s in sorted(tab["FROZEN-LM"])], dtype=float)
        d = e - f
        pairs[tname] = {"expert_minus_frozen_post_extrap": d.tolist(), "mean_diff": round(float(d.mean()), 2)}
    res["paired_expert_vs_frozen"] = pairs

    with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("written", os.path.join(HERE, "recompute_results.json"))


if __name__ == "__main__":
    main()
