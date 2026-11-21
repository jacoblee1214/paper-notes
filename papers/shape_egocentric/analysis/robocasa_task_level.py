"""논문 Table S10(RoboCasa GR-1, 과제별 성공률 %)으로 과제 단위 쌍체 검정을 한다.

목적: 논문 Table 4의 p값은 1,200개 과제-시드 쌍을 독립으로 보고 계산했다(Appendix F).
같은 과제 안의 시드는 독립이 아니므로, 과제 24개를 단위로 한 쌍체 t와 Wilcoxon 검정을 함께 본다.
입력: 논문 Table S10 (6열: GR00T N1.7, pi0.5, w/o both, w/o human, w/o FAST, SHAPE). 외부 데이터는 없다.
출력: analysis/robocasa_task_level_results.json
실행: python3 analysis/robocasa_task_level.py   (numpy, scipy, 1초 미만)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "robocasa_task_level_results.json"
COLS = ["GR00T N1.7", "pi0.5", "w/o both", "w/o human", "w/o FAST", "SHAPE"]
IDX = {c: i for i, c in enumerate(COLS)}
ROWS = [
    ("Bottle → Cabinet Close", [80.7, 76.7, 74.0, 75.3, 72.7, 82.7]),
    ("Can → Drawer Close", [86.0, 44.7, 76.0, 73.3, 60.0, 78.0]),
    ("Cup → Drawer Close", [51.3, 26.7, 48.0, 49.3, 42.0, 48.7]),
    ("Milk → Microwave Close", [71.3, 50.7, 56.7, 60.0, 62.7, 66.0]),
    ("Potato → Microwave Close", [70.7, 56.7, 66.0, 70.7, 60.7, 74.0]),
    ("Wine → Cabinet Close", [55.3, 53.3, 48.0, 46.7, 44.7, 50.0]),
    ("Cuttingboard → Basket", [82.7, 82.0, 78.7, 77.3, 86.7, 82.7]),
    ("Cuttingboard → Cardboardbox", [49.3, 80.7, 52.7, 70.7, 74.7, 74.0]),
    ("Cuttingboard → Pan", [92.7, 95.3, 90.0, 92.0, 89.3, 92.0]),
    ("Cuttingboard → Pot", [71.3, 75.3, 61.3, 62.0, 71.3, 74.0]),
    ("Cuttingboard → Tieredbasket", [41.3, 54.0, 55.3, 51.3, 59.3, 64.7]),
    ("Placemat → Basket", [79.3, 83.3, 67.3, 59.3, 72.7, 78.0]),
    ("Placemat → Bowl", [41.3, 68.0, 59.3, 62.7, 72.7, 72.0]),
    ("Placemat → Plate", [48.0, 76.0, 57.3, 72.7, 82.0, 74.7]),
    ("Placemat → Tieredshelf", [27.3, 32.7, 34.7, 32.0, 49.3, 49.3]),
    ("Plate → Bowl", [56.0, 75.3, 73.3, 83.3, 88.7, 88.0]),
    ("Plate → Cardboardbox", [44.7, 67.3, 58.0, 53.3, 67.3, 71.3]),
    ("Plate → Pan", [39.3, 76.7, 60.7, 66.0, 83.3, 80.0]),
    ("Plate → Plate", [89.3, 72.0, 72.7, 80.7, 80.0, 82.0]),
    ("Tray → Cardboardbox", [73.3, 70.7, 70.7, 67.3, 80.0, 80.0]),
    ("Tray → Plate", [87.3, 72.0, 75.3, 85.3, 84.0, 90.7]),
    ("Tray → Pot", [73.3, 70.0, 63.3, 69.3, 74.7, 76.0]),
    ("Tray → Tieredbasket", [64.0, 59.3, 60.7, 64.7, 64.0, 60.0]),
    ("Tray → Tieredshelf", [26.7, 35.3, 28.0, 39.3, 48.7, 44.7]),
]


def paired(a, b, M):
    d = M[:, IDX[a]] - M[:, IDX[b]]
    t = stats.ttest_rel(M[:, IDX[a]], M[:, IDX[b]])
    w = stats.wilcoxon(M[:, IDX[a]], M[:, IDX[b]])
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return {"a_minus_b": f"{a} - {b}", "mean_diff_pp": round(float(d.mean()), 2),
            "ci95_task_level": [round(float(lo), 2), round(float(hi), 2)],
            "paired_t_p": float(f"{t.pvalue:.2g}"), "wilcoxon_p": float(f"{w.pvalue:.2g}"),
            "tasks_a_higher": int((d > 0).sum()), "tasks_b_higher": int((d < 0).sum()), "ties": int((d == 0).sum())}


def main():
    M = np.array([r[1] for r in ROWS])
    assert M.shape == (24, 6)
    res = {"task_means": {c: round(float(M[:, IDX[c]].mean()), 2) for c in COLS},
           "close_6_means": {c: round(float(M[:6, IDX[c]].mean()), 2) for c in COLS},
           "pnp_18_means": {c: round(float(M[6:, IDX[c]].mean()), 2) for c in COLS}}
    pairs = [("SHAPE", "pi0.5"), ("SHAPE", "GR00T N1.7"), ("SHAPE", "w/o human"), ("w/o human", "pi0.5"),
             ("SHAPE", "w/o FAST"), ("w/o FAST", "w/o both"), ("w/o human", "w/o both"), ("SHAPE", "w/o both")]
    res["task_level_tests"] = [paired(a, b, M) for a, b in pairs]
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
