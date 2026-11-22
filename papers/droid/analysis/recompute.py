"""DROID 정책 실험(논문 Fig. 8, Fig. 10)의 막대 값을 다시 계산하고 검정한다.

입력: 논문 v2 그림 8과 그림 10의 막대 위 숫자(PDF 9쪽, 10쪽을 렌더링해 읽음).
각 셀은 롤아웃 10회(논문 V-B: "10 rollouts for each task setting and method")이므로
값은 모두 0.1의 배수여야 한다. 이를 확인하고, 평균, 표준오차, 이항 구간, 검정을 계산한다.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, numpy·scipy)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N = 10  # 셀당 롤아웃 수

# 그림 8 (논문 v2 9쪽). 순서: No Co-training, OXE, DROID
ID_TASKS = ["Close Waffle Maker", "Put Chips in Plate", "Put Apple in Pot", "Toasting", "Clean up Desk", "Cook Lentils"]
ID_VALS = {
    "noco":  [0.8, 0.6, 0.3, 0.7, 0.7, 0.1],
    "oxe":   [0.7, 0.4, 0.1, 0.7, 0.6, 0.1],
    "droid": [0.9, 0.7, 0.4, 0.8, 1.0, 0.7],
}
ID_PAPER_AVG = {"noco": 0.53, "oxe": 0.43, "droid": 0.75}

OOD_TASKS = ["Close Waffle Maker (distractors)", "Put Chips in Plate (distractors)", "Put Chips in Plate (novel chips)",
             "Put Apple in Pot (distractors)", "Toasting (novel objects)", "Clean up Desk (distractors)",
             "Cook Lentils (distractors + camera shift)"]
OOD_VALS = {
    "noco":  [0.4, 0.1, 0.2, 0.0, 0.2, 0.0, 0.3],
    "oxe":   [0.5, 0.4, 0.6, 0.1, 0.3, 0.6, 0.0],
    "droid": [0.7, 0.4, 0.3, 0.2, 0.7, 0.8, 0.6],
}
OOD_PAPER_AVG = {"noco": 0.17, "oxe": 0.36, "droid": 0.53}

# 그림 10 (논문 v2 10쪽), OOD 세 과제. 20 Scenes 대 Diverse Scenes (각 7362 궤적)
SD_TASKS = ["Toasting (novel objects)", "Clean up Desk (distractors)", "Cook Lentils (distractors + camera shift)"]
SD_VALS = {"20_scenes": [0.3, 0.7, 0.2], "diverse": [0.6, 0.8, 0.4]}
SD_PAPER_AVG = {"20_scenes": 0.40, "diverse": 0.60}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def to_counts(vals):
    """0.1 배수 확인 후 성공 횟수로 바꾼다."""
    counts = []
    for v in vals:
        k = round(v * N)
        assert abs(k / N - v) < 1e-9, f"value {v} is not a multiple of 0.1"
        counts.append(k)
    return counts


def setting_block(tasks, vals, paper_avg, label, focal="droid"):
    res = {"setting": label, "tasks": tasks, "methods": {}}
    counts = {m: to_counts(v) for m, v in vals.items()}
    k_tasks = len(tasks)
    for m, v in vals.items():
        arr = np.array(v)  # 과제별 성공률
        mean = arr.mean()
        sd = arr.std(ddof=1)
        se = sd / np.sqrt(k_tasks)  # 논문 오차막대와 같은 정의: 과제 간 표준편차 / sqrt(과제 수)로 확인
        pooled_k = sum(counts[m])
        pooled_n = N * k_tasks
        res["methods"][m] = {
            "per_task_counts": counts[m],
            "mean_recomputed": round(float(mean), 4),
            "paper_avg": paper_avg[m],
            "mean_matches_paper": bool(abs(round(float(mean), 2) - paper_avg[m]) < 1e-9),
            "se_task_level": round(float(se), 4),
            "pooled_success": pooled_k,
            "pooled_n": pooled_n,
            "pooled_rate": round(pooled_k / pooled_n, 4),
            "pooled_wilson95": [round(x, 4) for x in wilson(pooled_k, pooled_n)],
        }
    # 쌍 비교: DROID 대 각 기준선 (과제별 쌍, 과제 수 만큼의 차이)
    comps = {}
    if focal in vals:
        for base in vals:
            if base == focal:
                continue
            d = np.array(vals[focal]) - np.array(vals[base])
            t = stats.ttest_1samp(d, 0.0)
            se_d = d.std(ddof=1) / np.sqrt(len(d))
            tcrit = stats.t.ppf(0.975, len(d) - 1)
            # 풀링 Fisher (과제 구분 없이 모든 시행을 합침)
            k1, n1 = sum(counts[focal]), N * k_tasks
            k2, n2 = sum(counts[base]), N * k_tasks
            _, p_fisher_pool = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
            # 과제별 Fisher
            p_task = []
            for a, b in zip(counts[focal], counts[base]):
                p_task.append(stats.fisher_exact([[a, N - a], [b, N - b]])[1])
            comps[f"{focal}_minus_{base}"] = {
                "mean_diff": round(float(d.mean()), 4),
                "per_task_diffs": [round(float(x), 2) for x in d],
                "paired_t_p": float(f"{t.pvalue:.6g}"),
                "paired_ci95": [round(float(d.mean() - tcrit * se_d), 4), round(float(d.mean() + tcrit * se_d), 4)],
                "pooled_fisher_p": float(f"{p_fisher_pool:.6g}"),
                "per_task_fisher_p": [float(f"{x:.6g}") for x in p_task],
                "per_task_fisher_p_holm": [float(f"{x:.6g}") for x in holm(p_task)],
                "tasks_droid_higher": int(sum(x > 0 for x in d)),
                "tasks_tied": int(sum(x == 0 for x in d)),
                "tasks_base_higher": int(sum(x < 0 for x in d)),
            }
    res["comparisons"] = comps
    return res


def power_mde(n_per_group, p=0.5, alpha=0.05, power=0.8):
    """두 비율 차이의 최소 검출 크기(정규 근사, 양측)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(2 * p * (1 - p) / n_per_group))


def main():
    results = {}
    results["fig8_id"] = setting_block(ID_TASKS, ID_VALS, ID_PAPER_AVG, "in-distribution")
    results["fig8_ood"] = setting_block(OOD_TASKS, OOD_VALS, OOD_PAPER_AVG, "out-of-distribution")
    results["fig10_scene_diversity"] = setting_block(SD_TASKS, SD_VALS, SD_PAPER_AVG, "OOD, 20 scenes vs diverse", focal="diverse")

    # 시행 수와 검출력
    results["power"] = {
        "cell_n10_mde_p05": round(power_mde(10), 3),
        "pooled_id_n60_mde_p05": round(power_mde(60), 3),
        "pooled_ood_n70_mde_p05": round(power_mde(70), 3),
        "pooled_sd_n30_mde_p05": round(power_mde(30), 3),
        "note": "두 비율의 최소 검출 차이(alpha 0.05 양측, 검정력 0.8, 정규 근사, p=0.5 가정)",
    }

    # 전체 롤아웃 수 (논문 V-B와 V-C에서 계산)
    results["rollout_count"] = {
        "fig8_cells": (len(ID_TASKS) + len(OOD_TASKS)) * 3,
        "fig8_rollouts": (len(ID_TASKS) + len(OOD_TASKS)) * 3 * N,
        "fig10_rollouts": len(SD_TASKS) * 2 * N,
    }
    results["rollout_count"]["total"] = results["rollout_count"]["fig8_rollouts"] + results["rollout_count"]["fig10_rollouts"]

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약 출력
    for key in ["fig8_id", "fig8_ood", "fig10_scene_diversity"]:
        r = results[key]
        print(f"== {r['setting']}")
        for m, v in r["methods"].items():
            print(f"  {m:12s} mean={v['mean_recomputed']:.3f} paper={v['paper_avg']:.2f} "
                  f"SE(task)={v['se_task_level']:.3f} pooled={v['pooled_success']}/{v['pooled_n']} "
                  f"Wilson95={v['pooled_wilson95']}")
        for c, v in r["comparisons"].items():
            print(f"  {c}: diff={v['mean_diff']:+.3f} paired_t_p={v['paired_t_p']:.3g} "
                  f"pooled_Fisher_p={v['pooled_fisher_p']:.3g} per-task Holm={v['per_task_fisher_p_holm']} "
                  f"(DROID higher in {v['tasks_droid_higher']}/{len(v['per_task_diffs'])})")
    print("power:", results["power"])
    print("rollouts:", results["rollout_count"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
