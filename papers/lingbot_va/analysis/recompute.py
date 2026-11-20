"""LingBot-VA 논문 표 수치를 다시 계산하고 일관성을 검사한다 (numpy, scipy, 수 초).

입력: paper/paper.txt (pdftotext -layout 결과, arXiv v2 기준)
출력: analysis/recompute_results.json
실행: OMP_NUM_THREADS=2 python3 analysis/recompute.py
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TXT = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8")
OUT = ROOT / "analysis" / "recompute_results.json"
res = {}

# ---------- 1. 표 S1 (RoboTwin 50 과제, Easy/Hard, 과제별 Horizon) 파싱 ----------
lines = TXT.splitlines()
start = next(i for i, l in enumerate(lines) if l.strip().startswith("Adjust Bottle"))
end = next(i for i, l in enumerate(lines) if l.strip().startswith("Turn Switch"))
pat = re.compile(r"^\s*([A-Z][A-Za-z0-9 ]+?)\s{2,}(\d)\s+((?:\d+%\s*){9}\d+%)\s*$")
tasks = []
for l in lines[start:end + 1]:
    m = pat.match(l)
    if m:
        vals = [float(v.rstrip('%')) for v in m.group(3).split()]
        tasks.append((re.sub(r"\s+", " ", m.group(1)).strip(), int(m.group(2)), vals))
assert len(tasks) == 50, f"expected 50 tasks, got {len(tasks)}"
# 열 순서: Ours(Easy,Hard) pi0(E,H) pi0.5(E,H) X-VLA(E,H) Motus(E,H)
methods = ["Ours", "pi0", "pi0.5", "X-VLA", "Motus"]
E = np.array([[t[2][2 * i] for i in range(5)] for t in tasks])
H = np.array([[t[2][2 * i + 1] for i in range(5)] for t in tasks])
hz = np.array([t[1] for t in tasks])
res["n_tasks"] = len(tasks)
res["horizon_counts"] = {int(k): int((hz == k).sum()) for k in np.unique(hz)}

# 평균 재계산 vs 표 S1 Average 행 (92.93 91.55 65.92 58.40 82.74 76.76 72.80 72.84 88.66 87.02)
stated_S1 = [92.93, 91.55, 65.92, 58.40, 82.74, 76.76, 72.80, 72.84, 88.66, 87.02]
recomp = []
for i, m in enumerate(methods):
    recomp += [round(float(E[:, i].mean()), 2), round(float(H[:, i].mean()), 2)]
res["S1_mean_recomputed"] = dict(zip([f"{m}_{s}" for m in methods for s in ("Easy", "Hard")], recomp))
res["S1_mean_stated"] = stated_S1
res["S1_mean_max_abs_diff"] = round(float(np.max(np.abs(np.array(recomp) - np.array(stated_S1)))), 3)

# 표 1 horizon 평균 재계산 (Ours vs 두 번째로 좋은 Motus)
t1_stated = {1: (94.18, 93.56), 2: (90.35, 86.95), 3: (93.22, 93.28)}
t1_second = {1: (91.0, 90.6), 2: (85.2, 80.9), 3: (85.0, 84.2)}  # 표 1의 Motus 열
hz_rows = {}
for k in (1, 2, 3):
    mk = hz == k
    e, h = E[mk, 0].mean(), H[mk, 0].mean()
    hz_rows[k] = {
        "n": int(mk.sum()),
        "ours_easy": round(e, 2), "ours_hard": round(h, 2),
        "stated": t1_stated[k],
        "second_best_easy_motus": t1_second[k][0], "gain_easy_recomputed": round(e - t1_second[k][0], 2),
        "gain_hard_recomputed": round(h - t1_second[k][1], 2),
    }
res["table1_horizon"] = hz_rows
# 전 방법의 horizon별 평균 (그림용)
res["horizon_means_all"] = {int(k): {m: {"easy": round(float(E[hz == k, i].mean()), 2),
                                         "hard": round(float(H[hz == k, i].mean()), 2)}
                                     for i, m in enumerate(methods)} for k in (1, 2, 3)}

# 과제 단위 차이의 95% CI (쌍체 t, 과제 50개를 표본으로 사용)
def paired_ci(a, b):
    d = a - b
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    p = stats.ttest_rel(a, b).pvalue
    return dict(mean_diff=round(float(m), 2), ci95=[round(float(m - t * se), 2), round(float(m + t * se), 2)],
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()), p_paired_t=float(p))

res["paired_vs_pi05"] = {"easy": paired_ci(E[:, 0], E[:, 2]), "hard": paired_ci(H[:, 0], H[:, 2])}
res["paired_vs_motus"] = {"easy": paired_ci(E[:, 0], E[:, 4]), "hard": paired_ci(H[:, 0], H[:, 4])}
res["tasks_with_ours_below_motus_easy"] = [t[0] for t, e, m in zip(tasks, E[:, 0], E[:, 4]) if e < m]

# 과제 목록 개수: 본문 "Ours가 다른 방법보다 높음" 과제 수 (Easy 기준, pi0.5 대비)
res["easy_tasks_ours_gt_pi05"] = int((E[:, 0] > E[:, 2]).sum())

# ---------- 2. 표 2 LIBERO 평균과 시드 구간 ----------
libero = {"Spatial": (98.5, 0.3), "Object": (99.6, 0.3), "Goal": (97.2, 0.2), "Long": (98.5, 0.5)}
avg = np.mean([v[0] for v in libero.values()])
res["libero_avg_recomputed"] = round(float(avg), 3)
res["libero_avg_stated"] = 98.5
# 3 시드 x 500 시도 = 1500. 표기 소수 1자리와 맞는 정수 성공 횟수 범위를 구한다.
trials = 1500
res["libero_integer_success_range_1500"] = {
    k: [int(np.ceil((v[0] - 0.05) / 100 * trials)), int(np.floor((v[0] + 0.05) / 100 * trials))]
    for k, v in libero.items()}
# 시드 간 표준편차 ±를 표준오차로 보고 95% 구간 (3 시드, t 분포)
res["libero_seed_ci95"] = {k: [round(v[0] - stats.t.ppf(0.975, 2) * v[1] * np.sqrt(3), 2),
                                round(v[0] + stats.t.ppf(0.975, 2) * v[1] * np.sqrt(3), 2)]
                           for k, v in libero.items()}
# 시도 수 기반 이항 표준오차 (독립 가정, 시드 합산 1500회)
res["libero_binomial_se"] = {k: round(float(np.sqrt(v[0] / 100 * (1 - v[0] / 100) / trials) * 100), 3)
                             for k, v in libero.items()}

# ---------- 3. 실세계 과제 (부록 표 S2–S7): 성공 규칙 재검사 ----------
# 규칙(부록 A): 모든 단계를 끝내야 성공이며, 총점 = 최대점일 때만 성공으로 본다.
# 아래 값은 표에서 옮긴 "Ours" 시도별 (성공 표기, 진행 점수, 최대 점수)이다.
ours_trials = {
    "Make Breakfast": (10, [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                       [9, 10, 10, 10, 10, 10, 9, 9.5, 9, 10, 9, 10, 9, 10, 10, 10, 9.5, 10, 10, 10]),
    "Pick Screws": (5, [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                    [4, 3.5, 4, 5, 4.5, 4, 5, 5, 1, 5, 5, 4, 4, 4, 2.5, 5, 4.5, 4, 3.5, 5]),
    "Fold Clothes": (6, [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [6, 2, 5, 6, 0.5, 0, 0.5, 5, 6, 6, 0.5, 0.5, 0.5, 5, 0.5, 4.5, 0, 3.5, 0.5, 6]),
    "Unpack Delivery": (5, [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
                        [5, 5, 5, 1, 3.5, 3, 5, 3, 5, 5, 5, 5, 3, 5, 5, 3, 4.5, 5, 3.5, 5]),
    "Insert Tubes": (6, [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                     [5, 6, 5, 4, 6, 6, 5, 6, 5, 5, 6, 5, 6, 5, 4, 6, 4, 4, 6, 4]),
    "Fold Pants": (3, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                   None),  # 표 S7의 Progress 열은 아래에서 따로 넣는다
}
# Fold Pants: 표 S7 Ours 진행 점수 (trial 1..20)
ours_trials["Fold Pants"] = (3, ours_trials["Fold Pants"][1], [3, 3, 3, 0, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0])
rw = {}
for name, (mx, succ, prog) in ours_trials.items():
    succ = np.array(succ)
    prog = np.array(prog, dtype=float)
    rule_succ = (prog == mx).astype(int)
    rw[name] = {
        "stated_success_count": int(succ.sum()),
        "count_progress_eq_max": int(rule_succ.sum()),
        "mismatch_trials": [int(i + 1) for i in np.where(succ != rule_succ)[0]],
        "stated_PS_pct": round(float(prog.mean() / mx * 100), 1),
    }
res["realworld_ours_success_rule_check"] = rw

# ---------- 4. 기타 수치 대조 ----------
res["text_checks"] = {
    "RoboTwin_easy_hard_text": {"text": "92.9 (Easy), 91.6 (Hard)", "table1": (92.93, 91.55)},
    "ablation_text_pretrained_lingbot_vs_wan": {"text": "92.10 (Easy), 91.12 (Hard)", "table3_easy_all": 92.9,
                                                "note": "표 3 Easy 전체는 92.9, 본문 92.10은 다른 실험 또는 오기"},
    "abstract_RoboTwin_avg": {"text": "92.0% (결론)", "table1": 92.93,
                              "note": "결론의 92.0은 표 1 Easy 92.9와 다르다"},
    "sec_4_3_1_gain_hard": {"stated_gain_hard_pct": 4.6, "recomputed": round(91.55 - 87.02, 2)},
    "fig8_low_data_claims": {"text": "10 demos: +15.6 (Make Breakfast), +10.3 (RoboTwin Easy)"},
}

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False, default=float)[:6000])
