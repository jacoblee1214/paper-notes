"""Motus 논문 표 재계산과 검정 (CPU, 수 초).

입력: analysis/table14_robotwin.csv (논문 Table 14 과제별 값, pdftotext로 옮긴 뒤 수작업 검수 대상)
      아래 REAL_WORLD (논문 Table 4·5의 부분 점수 분포, 각 10회 시행으로 역산)
출력: analysis/recompute_results.json
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "table14_robotwin.csv", encoding="utf-8")))
cols = [c for c in rows[0] if c != "task"]
M = {c: np.array([float(r[c]) for r in rows]) for c in cols}
N_TRIALS = 100  # 논문 5.2절: 과제당 100회 실행

out = {"n_tasks": len(rows), "averages": {}, "table2_reported": {}, "checks": {}}

# 1. 평균 재계산 (Table 14 Average 행과 비교)
reported_t14 = {"GO1_clean": 37.8, "GO1_rand": 36.24, "pi05_clean": 42.98, "pi05_rand": 43.84,
                "xvla_clean": 72.8, "xvla_rand": 72.84, "wo_pt_clean": 77.56, "wo_pt_rand": 77.00,
                "stage1_clean": 82.26, "stage1_rand": 81.86, "motus_clean": 88.66, "motus_rand": 87.02}
for c in cols:
    out["averages"][c] = {"recomputed": round(M[c].mean(), 2), "reported_t14": reported_t14[c]}

# 2. Table 2 (본문) 평균과 Table 14 평균의 불일치
table2 = {"pi05_clean": 42.98, "pi05_rand": 43.84, "xvla_clean": 72.80, "xvla_rand": 72.84,
          "wo_pt_clean": 72.8, "wo_pt_rand": 77.00, "stage1_clean": 82.86, "stage1_rand": 81.86,
          "motus_clean": 88.66, "motus_rand": 87.02}
out["table2_reported"] = table2
out["checks"]["table2_vs_table14_mismatch"] = {
    c: {"table2": table2[c], "table14": reported_t14[c]} for c in table2 if abs(table2[c] - reported_t14[c]) > 1e-9}

# 3. 헤드라인 "+15% / +45%" 검산
def diff_stats(a, b):
    d = M[a] - M[b]
    t = stats.ttest_rel(M[a], M[b])
    w = stats.wilcoxon(M[a], M[b]) if np.any(d != 0) else None
    nz = d[d != 0]
    sign = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)]
    # 과제 내 이항 분산(n=100)으로 본 평균 차이의 표준오차 (독립 가정, 상한 근사)
    p1, p2 = M[a] / 100, M[b] / 100
    se_binom = math.sqrt(np.sum(p1 * (1 - p1) / N_TRIALS + p2 * (1 - p2) / N_TRIALS)) / len(d)
    return {
        "mean_diff_pts": round(d.mean(), 2),
        "relative_pct": round(100 * (M[a].mean() / M[b].mean() - 1), 1),
        "wins_losses_ties": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())],
        "paired_t_p": float(f"{t.pvalue:.3g}"),
        "wilcoxon_p": float(f"{w.pvalue:.3g}") if w else None,
        "sign_test_p": float(f"{sign:.3g}"),
        "bootstrap95_over_tasks": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
        "binomial_se_of_mean_diff": round(se_binom, 2),
    }

out["checks"]["motus_vs_pi05_clean"] = diff_stats("motus_clean", "pi05_clean")
out["checks"]["motus_vs_pi05_rand"] = diff_stats("motus_rand", "pi05_rand")
out["checks"]["motus_vs_xvla_clean"] = diff_stats("motus_clean", "xvla_clean")
out["checks"]["motus_vs_xvla_rand"] = diff_stats("motus_rand", "xvla_rand")
out["checks"]["motus_vs_stage1_rand"] = diff_stats("motus_rand", "stage1_rand")
out["checks"]["motus_vs_wo_pt_rand"] = diff_stats("motus_rand", "wo_pt_rand")
out["checks"]["motus_vs_stage1_clean"] = diff_stats("motus_clean", "stage1_clean")

# 4. 과제별 최저·최고 (Motus rand가 나쁜 과제)
d_rand = M["motus_rand"] - M["xvla_rand"]
worst = np.argsort(d_rand)[:5]
out["checks"]["motus_vs_xvla_rand_worst5"] = [[rows[i]["task"], float(M["motus_rand"][i]), float(M["xvla_rand"][i])] for i in worst]

# 5. 실세계: Table 4(Put Bread, AC-One)와 Table 5(Get Water, Agilex) 부분 점수 분포로부터 시행별 점수 복원
#    각 행은 (점수, 횟수). 시행 수 합 = 10 (논문 표 합계로 확인)
REAL_WORLD = {
    "put_bread_ac_one": {
        "pi05": [(0.0, 6), (0.2, 3), (0.4, 0), (0.6, 1), (0.8, 0), (1.0, 0)],
        "wo_pt": [(0.0, 4), (0.2, 0), (0.4, 2), (0.6, 1), (0.8, 2), (1.0, 1)],
        "motus": [(0.0, 5), (0.2, 0), (0.4, 1), (0.6, 0), (0.8, 1), (1.0, 3)],
        "reported_pct": {"pi05": 12, "wo_pt": 40, "motus": 42}},
    "get_water_agilex": {
        "pi05": [(0.0, 0), (0.4, 5), (0.8, 4), (1.0, 1)],
        "wo_pt": [(0.0, 8), (0.4, 2), (0.8, 0), (1.0, 0)],
        "motus": [(0.0, 0), (0.4, 0), (0.8, 2), (1.0, 8)],
        "reported_pct": {"pi05": 62, "wo_pt": 8, "motus": 96}},
}
rw = {}
rng = np.random.default_rng(1)
for task, d in REAL_WORLD.items():
    rw[task] = {}
    for k in ["pi05", "wo_pt", "motus"]:
        scores = np.concatenate([[s] * n for s, n in d[k]])
        assert len(scores) == 10, (task, k, len(scores))
        rw[task][k] = {"n": len(scores), "mean_pct": round(100 * scores.mean(), 1),
                       "reported_pct": d["reported_pct"][k]}
    a = np.concatenate([[s] * n for s, n in d["motus"]])
    b = np.concatenate([[s] * n for s, n in d["pi05"]])
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    boots = [100 * (rng.choice(a, 10).mean() - rng.choice(b, 10).mean()) for _ in range(10000)]
    rw[task]["motus_minus_pi05"] = {"mean_diff_pts": round(100 * (a.mean() - b.mean()), 1),
                                    "mannwhitney_p": round(float(u.pvalue), 3),
                                    "bootstrap95": [round(float(np.percentile(boots, 2.5)), 1), round(float(np.percentile(boots, 97.5)), 1)]}
out["real_world_trials"] = rw

# 6. 실세계 Table 3 평균 재계산 (과제 9개 AC-One, 5개 Agilex)
T3 = {"AC-One": {"pi05": [4, 0, 30, 46, 28.125, 8, 5, 0, 12], "wo_pt": [1, 0, 8, 60, 18.75, 0, 5, 100, 40],
                 "motus": [14.5, 62, 36, 100, 75, 92, 65, 82.5, 42]},
      "Agilex-Aloha-2": {"pi05": [27.5, 62, 45, 72.5, 36], "wo_pt": [0, 8, 40, 85, 0],
                         "motus": [39, 96, 47.5, 80, 34]}}
out["table3_recomputed"] = {p: {k: round(float(np.mean(v)), 2) for k, v in d.items()} for p, d in T3.items()}
out["table3_improvement_pts"] = {p: round(float(np.mean(d["motus"]) - np.mean(d["pi05"])), 2) for p, d in T3.items()}

with open(HERE / "recompute_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: out[k] for k in ["averages", "checks"]}, ensure_ascii=False, indent=1)[:6000])
print("table3", out["table3_recomputed"], out["table3_improvement_pts"])
print("rw", json.dumps(rw, ensure_ascii=False)[:1500])
