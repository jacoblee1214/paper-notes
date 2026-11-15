"""EgoScale 논문 그림 4, 5, 8과 본문 수치를 다시 계산하고 일관성을 검사한다.

논문에는 표가 없고 수치는 막대 위 라벨(그림 4, 7, 8)에만 있다. 여기서는 라벨을 그대로 입력하고,
(1) 시도 수 n을 2 seeds x 10 trials = 20으로 두고 성공률의 Wilson 95% 구간과 Fisher 정확 검정,
(2) 라벨이 n에 맞는 분수인지(격자 검사), (3) 본문 문장과 그림 값의 일치(절대 차이 vs 상대 차이),
(4) 같은 체크포인트가 두 그림에 나올 때 값이 같은지를 본다.

실행: python3 analysis/table_check.py  → analysis/table_check_results.json (CPU, 수 초)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("table_check_results.json")
TASKS = ["Shirt", "Card", "Tong", "Bottle", "Syringe"]
CKPTS = ["NoPre", "MidOnly", "HumanPre", "HumanPre+Mid"]
N_PRIMARY = 20  # 본문 3.1: 학습 시드 2개 x 평가 10회

# 그림 4(a) Task Completion Score, 그림 4(b) Task Success Rate (막대 라벨, 같은 순서)
SCORE = {
    "NoPre":        [0.40, 0.14, 0.35, 0.18, 0.16, 0.24],
    "MidOnly":      [0.55, 0.54, 0.85, 0.51, 0.23, 0.53],
    "HumanPre":     [0.83, 0.74, 0.79, 0.63, 0.53, 0.71],
    "HumanPre+Mid": [0.90, 0.87, 0.87, 0.82, 0.70, 0.83],
}
SUCC = {
    "NoPre":        [0.05, 0.00, 0.05, 0.00, 0.00, 0.02],
    "MidOnly":      [0.20, 0.50, 0.60, 0.10, 0.00, 0.28],
    "HumanPre":     [0.40, 0.70, 0.45, 0.21, 0.17, 0.38],
    "HumanPre+Mid": [0.50, 0.65, 0.65, 0.59, 0.42, 0.56],
}
# 그림 8 (과제: Card, Tong, Bottle; 막대 순서 Wrist only, Fingertip, Full)
FIG8_SCORE = {"Card": [0.56, 0.17, 0.74], "Tong": [0.24, 0.76, 0.79], "Bottle": [0.26, 0.55, 0.61]}
# 그림 7 (G1). 순서: Midtrain Only, Human Pretrain, Human Pretrain + Midtrain
G1_SCORE = {"PenInBin": [0.45, 0.68, 0.83], "DishInRack": [0.54, 0.58, 0.88]}
G1_SUCC = {"PenInBin": [0.3, 0.3, 0.67], "DishInRack": [0.2, 0.2, 0.5]}
# 그림 5 오른쪽 막대 라벨 (평균 과제 완료 점수, 데이터 규모)
SCALE_SCORE = {"1k": 0.30, "2k": 0.45, "4k": 0.48, "10k": 0.57, "20k": 0.71}


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def counts(p, n):
    """라벨 p를 n회 시도의 성공 수로 바꾼다. 정수가 아니면 가장 가까운 정수와 반올림 오차를 돌려준다."""
    k = round(p * n)
    return k, abs(k / n - p)


def granularity(values, n_list=range(1, 61)):
    """라벨(소수 둘째 자리 반올림)을 모두 만족하는 n이 있는지 찾는다."""
    ok = []
    for n in n_list:
        if all(abs(round(k / n, 2) - v) < 1e-9 for v in values for k in [round(v * n)]):
            ok.append(n)
    return ok


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def main():
    res = {"n_primary": N_PRIMARY, "assumption": "2 seeds x 10 trials = 20 per cell (본문 3.1)"}

    # 1. 성공률 라벨 -> 성공 수, 격자 검사
    cells = []
    for c in CKPTS:
        for t, p in zip(TASKS + ["Average"], SUCC[c]):
            if t == "Average":
                continue
            k, err = counts(p, N_PRIMARY)
            cells.append({"ckpt": c, "task": t, "label": p, "k": k, "n": N_PRIMARY,
                          "wilson95": [round(x, 3) for x in wilson(k, N_PRIMARY)],
                          "label_minus_k_over_n": round(p - k / N_PRIMARY, 3)})
    res["success_cells"] = cells

    # 라벨이 n=20 분수에 정확히 맞는 셀(격자 위반) 목록
    off_grid = []
    for c in CKPTS:
        for t, p in zip(TASKS, SUCC[c][:5]):
            if abs(round(p * N_PRIMARY) - p * N_PRIMARY) > 0.01:
                off_grid.append({"ckpt": c, "task": t, "label": p, "multiple_of_1_over_20": False})
    res["off_grid_n20"] = off_grid
    res["bottle_label_trial_counts_consistent_n_le_60"] = {
        ck: granularity([SUCC[ck][3]]) for ck in CKPTS}
    res["bottle_joint_consistent_n_le_60"] = granularity([SUCC[ck][3] for ck in CKPTS])
    res["syringe_joint_consistent_n_le_60"] = granularity([SUCC[ck][4] for ck in CKPTS])

    # 2. 과제별 Fisher 정확 검정 (두 체크포인트 사이, 독립 가정) + Holm 보정
    pairs = [("HumanPre+Mid", "HumanPre"), ("HumanPre+Mid", "MidOnly"),
             ("HumanPre+Mid", "NoPre"), ("HumanPre", "MidOnly"), ("HumanPre", "NoPre")]
    tests = []
    for a, b in pairs:
        pvals = []
        rows = []
        for i, t in enumerate(TASKS):
            ka, kb = counts(SUCC[a][i], N_PRIMARY)[0], counts(SUCC[b][i], N_PRIMARY)[0]
            _, p = stats.fisher_exact([[ka, N_PRIMARY - ka], [kb, N_PRIMARY - kb]])
            pvals.append(p)
            rows.append({"task": t, "k_a": ka, "k_b": kb, "p_fisher": round(p, 4)})
        adj = holm(pvals)
        for r, pa in zip(rows, adj):
            r["p_holm"] = round(pa, 4)
        ka = sum(counts(SUCC[a][i], N_PRIMARY)[0] for i in range(5))
        kb = sum(counts(SUCC[b][i], N_PRIMARY)[0] for i in range(5))
        N = 5 * N_PRIMARY
        _, pooled = stats.fisher_exact([[ka, N - ka], [kb, N - kb]])
        tests.append({"a": a, "b": b, "per_task": rows, "pooled_k_a": ka, "pooled_k_b": kb,
                      "pooled_n_each": N, "p_pooled_fisher": float("%.3g" % pooled)})
    res["fisher_tests"] = tests

    # 3. 검출력: n=20 per arm, 기준 p=0.5, 두 비율 차이의 최소 검출 크기(정확 Fisher, 모의실험)
    rng = np.random.default_rng(0)
    mde = {}
    for base in [0.2, 0.5]:
        for delta in np.arange(0.05, 0.75, 0.05):
            p2 = min(base + delta, 0.999)
            hits = 0
            reps = 400
            for _ in range(reps):
                a = rng.binomial(N_PRIMARY, base)
                b = rng.binomial(N_PRIMARY, p2)
                if stats.fisher_exact([[a, N_PRIMARY - a], [b, N_PRIMARY - b]])[1] < 0.05:
                    hits += 1
            if hits / reps >= 0.8:
                mde[str(base)] = round(float(delta), 2)
                break
    res["power_mde_80pct_n20_per_arm"] = mde

    # 4. 본문 문장과 그림 값
    avg_succ_hpm = np.mean(SUCC["HumanPre+Mid"][:5])
    avg_succ_no = np.mean(SUCC["NoPre"][:5])
    res["claim_54pct"] = {
        "abstract": "평균 성공률 54% 향상 (no pretraining 대비)",
        "label_avg_success_HPmid": 0.56, "label_avg_success_NoPre": 0.02,
        "absolute_pp_difference": round(0.56 - 0.02, 2),
        "relative_ratio": round(0.56 / 0.02, 1),
        "computed_mean_of_task_labels_HPmid": round(float(avg_succ_hpm), 3),
        "computed_mean_of_task_labels_NoPre": round(float(avg_succ_no), 3),
    }
    avg_score_hpm = 0.83
    avg_score_no = 0.24
    per_task_gain = [SCORE["HumanPre+Mid"][i] - SCORE["NoPre"][i] for i in range(5)]
    res["claim_55pct_task_completion"] = {
        "text": "평균 과제 완료 점수 55% 이상 향상 (3.2절)",
        "absolute_avg_diff": round(avg_score_hpm - avg_score_no, 2),
        "relative_ratio": round(avg_score_hpm / avg_score_no, 2),
        "mean_of_per_task_absolute_gains": round(float(np.mean(per_task_gain)), 3),
        "note": "절대 차이(0.59)로 맞춤. 상대 향상은 약 3.5배. 단위를 밝히지 않음",
    }
    res["g1_claim_over_30pct"] = {
        "text": "G1에서 인간 사전학습 정책이 사전학습 없는 기준 대비 30% 이상 절대 향상 (서론)",
        "pen_in_bin_success_gain_vs_MidOnly_pp": round((0.67 - 0.30) * 100),
        "dish_in_rack_success_gain_vs_MidOnly_pp": round((0.50 - 0.20) * 100),
        "note": "두 번째 과제는 정확히 30 pp이므로 '30% 초과'가 아니라 같음. 두 과제 모두 Human Pretrain(중간학습 없음)은 Midtrain Only와 성공률이 같음(0.3, 0.2)",
        "human_pretrain_only_success": G1_SUCC,
    }
    res["g1_granularity_n20"] = {
        "pen_in_bin_0.67_multiple_of_0.05": abs(round(0.67 * 20) - 0.67 * 20) < 0.01,
        "note": "2 seeds x 10 trials이면 0.67은 나올 수 없음(0.65 또는 0.70)",
    }

    # 5. 그림 8과 그림 4의 같은 체크포인트 비교
    res["fig8_vs_fig4_full_pretrain"] = {
        "Card": {"fig8_full": FIG8_SCORE["Card"][2], "fig4_HumanPre+Mid_or_HumanPre": [SCORE["HumanPre"][1], SCORE["HumanPre+Mid"][1]]},
        "Tong": {"fig8_full": FIG8_SCORE["Tong"][2], "fig4": [SCORE["HumanPre"][2], SCORE["HumanPre+Mid"][2]]},
        "Bottle": {"fig8_full": FIG8_SCORE["Bottle"][2], "fig4_HumanPre": SCORE["HumanPre"][3], "fig4_HumanPre+Mid": SCORE["HumanPre+Mid"][3]},
        "note": "Full(22-DoF) 표현은 그림 4의 Human Pretrain과 같은 체크포인트로 보이나 Bottle은 0.61 vs 0.63 (차이 0.02)",
    }
    res["fig8_wrist_only_text_check"] = {
        "text": "wrist-only는 모든 과제에서 나쁘다(4.6절 또는 3.6절)",
        "Card_wrist_vs_full": [FIG8_SCORE["Card"][0], FIG8_SCORE["Card"][2]],
        "Tong_wrist_vs_full": [FIG8_SCORE["Tong"][0], FIG8_SCORE["Tong"][2]],
        "Bottle_wrist_vs_full": [FIG8_SCORE["Bottle"][0], FIG8_SCORE["Bottle"][2]],
        "note": "Card에서 wrist-only 0.56은 full 0.74보다 낮지만 격차 0.18. 'all tasks poor'는 과장. 반복 횟수가 없음",
    }

    # 6. 시도 수 불일치 (3.1절 vs 부록 B)
    res["trial_count_conflict"] = {
        "sec_3_1": "Task III에서 병(bottle)당 4회, 4개 병, 16회 (표현은 Task III, 실제 병 과제는 Task IV)",
        "appendix_B": "Task IV(병): 4개 구성 각 4회, 총 12회. Task III(Tong): 과일 2종 x 5회 = 10회",
        "effect": "병 과제의 n을 16(시드당), 12(시드당 총), 20(10회 x 2) 중 무엇으로 봐야 하는지 불명. 여기서는 20으로 씀",
    }
    res["bottle_n_sensitivity"] = {
        "n_20": {"HumanPre+Mid_0.59_Wilson": [round(x, 3) for x in wilson(round(0.59 * 20), 20)]},
        "n_24": {"HumanPre+Mid_0.59_Wilson": [round(x, 3) for x in wilson(round(0.59 * 24), 24)]},
        "n_32": {"HumanPre+Mid_0.59_Wilson": [round(x, 3) for x in wilson(round(0.59 * 32), 32)]},
    }

    # 7. 시드 구조: 시드 2개면 시드 간 분산이 이항 구간에 안 들어간다
    res["seed_caveat"] = ("두 시드의 평가 10회를 한 이항 표본으로 합쳤다. 시드 간 변동이 크면 실제 구간은 더 넓다. "
                          "보고된 값만으로 시드 분산을 추정할 수 없다.")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
    for t in tests:
        print(t["a"], "vs", t["b"], "pooled p=", t["p_pooled_fisher"], [r["p_holm"] for r in t["per_task"]])
    print("off-grid cells (n=20):", [(o["ckpt"], o["task"], o["label"]) for o in off_grid])
    print("bottle granularity n<=60 per ckpt:", res["bottle_label_trial_counts_consistent_n_le_60"])
    print("MDE:", mde)


if __name__ == "__main__":
    main()
