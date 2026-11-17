"""SimToolReal Table II(실세계 24과제 x 5 rollout) 재계산과 본문 수치 검사. numpy, scipy만 사용.
실행: python3 analysis/recompute.py  → analysis/recompute_results.json
"""
import json, math
from pathlib import Path
import numpy as np
from scipy import stats

ROWS = [  # (카테고리, 도구, 과제, rollout 1..5, 보고된 Avg)  -- paper v3 Table II
    ("Hammer", "Claw", "Swing Down", [100, 100, 100, 100, 100], 100.0),
    ("Hammer", "Claw", "Swing Side", [100, 100, 77.5, 100, 100], 95.5),
    ("Hammer", "Mallet", "Swing Down", [100, 100, 33.3, 100, 88.9], 84.4),
    ("Hammer", "Mallet", "Swing Side", [65.6, 81.3, 100, 78.1, 62.5], 77.5),
    ("Marker", "Sharpie", "Draw Smile", [100, 100, 100, 100, 0.0], 80.0),
    ("Marker", "Sharpie", "Write C", [100, 100, 100, 32.0, 56.0], 77.6),
    ("Marker", "Staples", "Draw Smile", [100, 53.1, 100, 100, 100], 90.6),
    ("Marker", "Staples", "Write C", [31.0, 0.0, 100, 100, 100], 66.2),
    ("Eraser", "Handle", "Wipe Smile", [100] * 5, 100.0),
    ("Eraser", "Handle", "Wipe C", [100] * 5, 100.0),
    ("Eraser", "Flat", "Wipe Smile", [100] * 5, 100.0),
    ("Eraser", "Flat", "Wipe C", [100] * 5, 100.0),
    ("Brush", "Blue", "Sweep Fwd", [47.8, 47.8, 40.0, 60.0, 60.0], 51.1),
    ("Brush", "Blue", "Sweep Right", [100] * 5, 100.0),
    ("Brush", "Red", "Sweep Fwd", [100, 100, 13.5, 100, 100], 82.7),
    ("Brush", "Red", "Sweep Right", [43.2, 62.2, 100, 42.2, 59.5], 61.4),
    ("Spatula", "Spoon", "Serve Plate", [100, 76.5, 100, 48.5, 100], 85.0),
    ("Spatula", "Spoon", "Flip Over", [77.5, 100, 100, 100, 100], 95.5),
    ("Spatula", "Flat", "Serve Plate", [100, 75.0, 78.1, 56.3, 78.1], 77.5),
    ("Spatula", "Flat", "Flip Over", [0.0, 52.2, 0.0, 78.3, 100], 46.1),
    ("Screwdriver", "Long", "Spin Vert", [20.9, 72.1, 53.5, 74.4, 58.1], 55.8),
    ("Screwdriver", "Long", "Spin Horiz", [48.4, 86.7, 83.3, 3.3, 86.7], 61.7),
    ("Screwdriver", "Short", "Spin Vert", [15.4, 61.5, 12.8, 100, 0.0], 37.9),
    ("Screwdriver", "Short", "Spin Horiz", [100, 100, 100, 77.8, 0.0], 75.6),
]


def main():
    out = {}
    R = np.array([r for *_, r, _ in ROWS])
    avg_rep = np.array([a for *_, a in ROWS])
    avg_calc = R.mean(axis=1)
    out["task_avg_max_abs_diff"] = float(np.max(np.abs(avg_calc - avg_rep)))
    flat = R.ravel()
    n = flat.size
    m, sd = flat.mean(), flat.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    out["rollouts_n"] = int(n)
    out["mean_all_rollouts"] = round(m, 2)
    out["sd_rollouts"] = round(sd, 2)
    out["ci95_rollouts_t"] = [round(m - t * sd / math.sqrt(n), 2), round(m + t * sd / math.sqrt(n), 2)]
    rng = np.random.default_rng(0)
    bs = np.array([R[rng.integers(0, 24, 24)].mean() for _ in range(20000)])  # 과제 단위 bootstrap
    out["ci95_task_cluster_bootstrap"] = [round(float(x), 2) for x in np.percentile(bs, [2.5, 97.5])]
    out["mean_of_24_task_avgs"] = round(avg_rep.mean(), 2)
    cats = {}
    for c, *_, r, _ in [(x[0], x[1], x[2], x[3], x[4]) for x in ROWS]:
        cats.setdefault(c, []).extend(r)
    out["category_mean"] = {c: round(float(np.mean(v)), 1) for c, v in cats.items()}
    out["rollouts_below_100"] = int((flat < 100).sum())
    # 본문 실패 원인 비율 43.7/34.5/18.2/3.6 %가 작은 정수 실패 수로 설명되는지 확인
    hits = []
    for N in range(10, 201):
        ok = True
        for p in (43.7, 34.5, 18.2, 3.6):
            if not any(abs(round(100 * k / N, 1) - p) < 1e-9 for k in range(N + 1)):
                ok = False
                break
        if ok:
            hits.append(N)
    out["failure_pct_consistent_denominators_10_200"] = hits
    # Fig. 5 (본문 캡션 값)
    out["fig5_var1_gap_vs_fixed_grasp_pp"] = round(98.0 - 61.0, 1)
    out["fig5_var1_gap_vs_retarget_pp"] = round(98.0 - 8.1, 1)
    out["fig5_var2_gap_vs_fixed_grasp_pp"] = round(82.7 - 10.8, 1)
    out["fig5_var2_gap_vs_retarget_pp"] = round(82.7 - 0.0, 1)
    out["fig5_var2_simtoolreal_equals_table2_red_sweep_fwd"] = bool(abs(np.mean([100, 100, 13.5, 100, 100]) - 82.7) < 0.05)
    Path(__file__).with_name("recompute_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
