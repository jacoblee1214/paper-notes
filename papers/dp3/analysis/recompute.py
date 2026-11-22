"""DP3 (arXiv:2403.03954 v7) 논문 표 수치 재계산과 검정. 실행: python3 analysis/recompute.py (수 초, 학습 없음).

입력: 논문 부록 Table XVIII(72 과제별 평균 성공률), Table II·XVII(10 과제), 본문 Table IV, V, VIII–XIV.
출력: analysis/recompute_results.json 과 표준출력.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------- 1. 시뮬레이션 72 과제 (Table XVIII, v7). (domain, 과제, DP3, DP) ----------
T18 = [
    ("Adroit", "Hammer", 100, 45), ("Adroit", "Door", 62, 37), ("Adroit", "Pen", 43, 13),
    ("Bi-DexHands", "Block Stack", 24, 4), ("Bi-DexHands", "Bottle Cap", 83, 61),
    ("Bi-DexHands", "Door Open Outward", 100, 100), ("Bi-DexHands", "Grasp And Place", 69, 65),
    ("Bi-DexHands", "Hand Over", 45, 38), ("Bi-DexHands", "Scissors", 100, 100),
    ("DexArt", "Laptop", 83, 69), ("DexArt", "Faucet", 63, 23), ("DexArt", "Toilet", 82, 58),
    ("DexArt", "Bucket", 46, 46),
    ("DexDeform", "Rope", 93, 97), ("DexDeform", "Bun", 70, 76), ("DexDeform", "Dumpling", 92, 92),
    ("DexDeform", "Wrap", 94, 91), ("DexDeform", "Flip", 97, 99), ("DexDeform", "Folding", 81, 88),
    ("DexMV", "Pour", 99, 90), ("DexMV", "Place Inside", 100, 100),
    ("HORA", "Rotation", 71, 49),
    ("MW-Easy", "Button Press", 100, 99), ("MW-Easy", "Button Press Topdown", 100, 98),
    ("MW-Easy", "Button Press Topdown Wall", 99, 96), ("MW-Easy", "Button Press Wall", 99, 97),
    ("MW-Easy", "Coffee Button", 100, 99), ("MW-Easy", "Dial Turn", 66, 63), ("MW-Easy", "Door Close", 100, 100),
    ("MW-Easy", "Door Lock", 98, 86), ("MW-Easy", "Door Open", 99, 98), ("MW-Easy", "Door Unlock", 100, 98),
    ("MW-Easy", "Drawer Close", 100, 100), ("MW-Easy", "Drawer Open", 100, 93),
    ("MW-Easy", "Faucet Close", 100, 100), ("MW-Easy", "Faucet Open", 100, 100),
    ("MW-Easy", "Handle Press", 100, 81), ("MW-Easy", "Handle Pull", 53, 27),
    ("MW-Easy", "Handle Press Side", 100, 100), ("MW-Easy", "Handle Pull Side", 85, 23),
    ("MW-Easy", "Lever Pull", 79, 49), ("MW-Easy", "Plate Slide", 100, 83),
    ("MW-Easy", "Plate Slide Back", 99, 99), ("MW-Easy", "Plate Slide Back Side", 100, 100),
    ("MW-Easy", "Plate Slide Side", 100, 100), ("MW-Easy", "Reach", 24, 18),
    ("MW-Easy", "Reach Wall", 68, 59), ("MW-Easy", "Window Close", 100, 100),
    ("MW-Easy", "Window Open", 100, 100), ("MW-Easy", "Peg Unplug Side", 75, 74),
    ("MW-Medium", "Basketball", 98, 85), ("MW-Medium", "Bin Picking", 34, 15), ("MW-Medium", "Box Close", 42, 30),
    ("MW-Medium", "Coffee Pull", 87, 34), ("MW-Medium", "Coffee Push", 94, 67), ("MW-Medium", "Hammer", 76, 15),
    ("MW-Medium", "Peg Insert Side", 69, 34), ("MW-Medium", "Push Wall", 49, 20), ("MW-Medium", "Soccer", 18, 14),
    ("MW-Medium", "Sweep", 96, 18), ("MW-Medium", "Sweep Into", 15, 10),
    ("MW-Hard", "Assembly", 99, 15), ("MW-Hard", "Hand Insert", 14, 9), ("MW-Hard", "Pick Out of Hole", 14, 0),
    ("MW-Hard", "Pick Place", 12, 0), ("MW-Hard", "Push", 51, 30), ("MW-Hard", "Push Back", 0, 0),
    ("MW-VeryHard", "Shelf Place", 17, 11), ("MW-VeryHard", "Disassemble", 69, 43),
    ("MW-VeryHard", "Stick Pull", 27, 11), ("MW-VeryHard", "Stick Push", 97, 63),
    ("MW-VeryHard", "Pick Place Wall", 35, 5),
]
assert len(T18) == 72
dom = [t[0] for t in T18]
d3 = np.array([t[2] for t in T18], float)
dp = np.array([t[3] for t in T18], float)

# 영역별 평균 (Table I 대조). 논문 Table I 값
T1_reported = {  # domain: (DP3, DP, n)
    "Adroit": (68.3, 31.7, 3), "Bi-DexHands": (70.2, 61.3, 6), "DexArt": (68.5, 49.0, 4),
    "DexDeform": (87.8, 90.5, 6), "DexMV": (99.5, 95.0, 2), "HORA": (71.0, 49.0, 1),
    "MW-Easy": (90.9, 83.6, 28), "MW-Medium": (61.6, 31.1, 11), "MW-Hard": (31.7, 9.0, 6),
    "MW-VeryHard": (49.0, 26.6, 5),
}
dom_check = {}
for k, (r3, rdp, n) in T1_reported.items():
    m = np.array([d for d in dom if d == k])
    a3 = d3[[i for i, d in enumerate(dom) if d == k]].mean()
    ad = dp[[i for i, d in enumerate(dom) if d == k]].mean()
    assert len(m) == n, (k, len(m), n)
    dom_check[k] = {"n": n, "dp3_calc": round(a3, 2), "dp3_paper": r3, "dp_calc": round(ad, 2), "dp_paper": rdp}
res["table1_domain_check"] = dom_check

m3, mdp = d3.mean(), dp.mean()
res["table1_overall"] = {
    "dp3_mean_calc": round(m3, 3), "dp3_mean_paper": 74.4,
    "dp_mean_calc": round(mdp, 3), "dp_mean_paper": 59.8,
    "rel_improvement_from_calc_means_pct": round((m3 / mdp - 1) * 100, 2),
    "rel_improvement_from_paper_rounded_pct": round((74.4 / 59.8 - 1) * 100, 2),
    "rel_improvement_paper_pct": 24.2,
    "dp3_sd_ddof0": round(d3.std(ddof=0), 2), "dp3_sd_ddof1": round(d3.std(ddof=1), 2), "dp3_sd_paper": 29.9,
    "dp_sd_ddof0": round(dp.std(ddof=0), 2), "dp_sd_ddof1": round(dp.std(ddof=1), 2), "dp_sd_paper": 35.9,
}

# 논문 본문 주장 검사: 90% 이상 과제 수, 10% 미만 과제 수
res["text_claims_72"] = {
    "dp3_tasks_ge90": int((d3 >= 90).sum()), "dp_tasks_ge90": int((dp >= 90).sum()),
    "paper_says": "DP3 nearly 30 (>=90%), DP fewer than 15",
    "dp3_tasks_lt10": int((d3 < 10).sum()), "dp3_lt10_names": [t[1] for t in T18 if t[2] < 10],
    "dp_tasks_lt10": int((dp < 10).sum()),
    "paper_says_lt10": "DP3 none below 10%; DP more than 10 tasks below 10%",
    "dp3_zero_tasks": [t[1] for t in T18 if t[2] == 0],
}

# 대응 비교 (72 과제, 과제 평균을 단위로)
diff = d3 - dp
rng = np.random.default_rng(0)
boot = np.array([rng.choice(diff, size=diff.size, replace=True).mean() for _ in range(10000)])
boot_ratio = []
for _ in range(10000):
    idx = rng.integers(0, diff.size, diff.size)
    boot_ratio.append((d3[idx].mean() / dp[idx].mean() - 1) * 100)
w = stats.wilcoxon(d3, dp)
t = stats.ttest_rel(d3, dp)
res["paired_72"] = {
    "mean_diff_pp": round(diff.mean(), 2),
    "boot95_mean_diff_pp": [round(np.percentile(boot, 2.5), 2), round(np.percentile(boot, 97.5), 2)],
    "boot95_rel_improvement_pct": [round(np.percentile(boot_ratio, 2.5), 1), round(np.percentile(boot_ratio, 97.5), 1)],
    "tasks_dp3_better": int((diff > 0).sum()), "tasks_dp_better": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
    "sign_test_p": round(stats.binomtest(int((diff > 0).sum()), int((diff != 0).sum()), 0.5).pvalue, 6),
    "wilcoxon_p": float(f"{w.pvalue:.3g}"), "paired_t_p": float(f"{t.pvalue:.3g}"),
}

# 영역별 과제 수 민감도: MetaWorld 제외
mask = np.array([not d.startswith("MW") for d in dom])
res["paired_72_excluding_metaworld"] = {
    "n": int(mask.sum()), "dp3_mean": round(d3[mask].mean(), 2), "dp_mean": round(dp[mask].mean(), 2),
    "wilcoxon_p": float(f"{stats.wilcoxon(d3[mask], dp[mask]).pvalue:.3g}"),
}

# ---------- 2. 10 과제 표 (Table II / XVII) 와 Table XVIII 대조 ----------
T10_names = ["Hammer", "Door", "Pen", "Assembly", "Disassemble", "Stick-Push", "Laptop", "Faucet", "Toilet", "Bucket"]
T2 = {  # Table II (Average 열: DP3 74.4, DP 44.0, BCRNN 9.8, BCRNN+3D 14.5, IBC 5.0, IBC+3D 10.4)
    "DP3": [100, 62, 43, 99, 69, 97, 83, 63, 82, 46],
    "DP": [48, 50, 25, 15, 43, 63, 69, 23, 58, 46],
    "BCRNN": [0, 0, 9, 3, 32, 45, 3, 1, 5, 0],
    "BCRNN+3D": [8, 0, 8, 1, 11, 0, 29, 26, 38, 24],
    "IBC": [0, 0, 9, 0, 1, 16, 3, 7, 14, 0],
    "IBC+3D": [0, 0, 10, 18, 3, 50, 1, 7, 15, 0],
}
T2_avg_reported = {"DP3": 74.4, "DP": 44.0, "BCRNN": 9.8, "BCRNN+3D": 14.5, "IBC": 5.0, "IBC+3D": 10.4}
T17_simple = [100, 58, 46, 79, 50, 97, 84, 63, 81, 44]  # Table XVII Simple DP3
T17_dp = [48, 50, 25, 15, 43, 63, 69, 23, 58, 46]
key18 = {t[1]: (t[2], t[3]) for t in T18}
ren = {"Stick-Push": "Stick Push"}
t18_vals = {n: key18[ren.get(n, n)] for n in T10_names}
res["table2_mean_check"] = {k: {"calc": round(np.mean(v), 2), "paper": T2_avg_reported[k]} for k, v in T2.items()}
res["table2_vs_table18_mismatch"] = {
    "DP_table2_xvii": {n: v for n, v in zip(T10_names, T2["DP"])},
    "DP_table18": {n: t18_vals[n][1] for n in ["Hammer", "Door", "Pen"]},
    "note": "Table II/XVII의 DP 열과 Table XVIII의 DP 값이 Adroit 세 과제(Hammer, Door, Pen)에서 다름. 나머지 7개는 일치",
    "dp10_mean_with_table18_values": round(np.mean([t18_vals[n][1] for n in T10_names]), 2),
    "dp3_10_mean_table18": round(np.mean([t18_vals[n][0] for n in T10_names]), 2),
}
dp_t18_10 = np.array([t18_vals[n][1] for n in T10_names], float)
dp3_t18_10 = np.array([t18_vals[n][0] for n in T10_names], float)
res["table17_simple_dp3"] = {
    "mean_simple": round(np.mean(T17_simple), 2), "mean_dp3": round(np.mean(T2["DP3"]), 2),
    "mean_dp_t17": round(np.mean(T17_dp), 2),
    "simple_minus_dp3_pp": round(np.mean(np.array(T17_simple) - np.array(T2["DP3"])), 2),
    "simple_minus_dp3_per_task": dict(zip(T10_names, (np.array(T17_simple) - np.array(T2["DP3"])).tolist())),
}
dd = np.array(T2["DP3"], float) - np.array(T2["DP"], float)
res["paired_10_table2"] = {
    "mean_diff_pp": round(dd.mean(), 2), "paired_t_p": float(f"{stats.ttest_rel(T2['DP3'], T2['DP']).pvalue:.3g}"),
    "wilcoxon_p": float(f"{stats.wilcoxon(T2['DP3'], T2['DP']).pvalue:.3g}"),
    "tasks_dp3_better": int((dd > 0).sum()), "tasks_equal": int((dd == 0).sum()),
}
res["paired_10_table18_values"] = {
    "mean_diff_pp": round(float(dp3_t18_10.mean() - dp_t18_10.mean()), 2),
    "dp_mean": round(float(dp_t18_10.mean()), 2),
}

# ---------- 3. 본문 절제 표 평균 검사 (6 과제: H, D, P, A, DA, SP) ----------
abl = {
    "Table IV Oracle state": ([99, 61, 44, 94, 72, 91], 76.8),
    "Table IV Point cloud (DP3)": ([100, 62, 43, 99, 69, 97], 78.3),
    "Table IV Image": ([48, 50, 25, 15, 43, 63], 40.7),
    "Table IV Depth": ([39, 49, 12, 15, 15, 62], 32.0),
    "Table IV RGB-D": ([57, 47, 14, 15, 14, 61], 34.7),
    "Table IV Voxel": ([54, 33, 18, 10, 17, 62], 32.3),
    "Table V PointNet": ([46, 34, 14, 0, 0, 0], 15.7),
    "Table V PointNet++": ([0, 0, 13, 0, 0, 0], 2.2),
    "Table V PointNeXt": ([0, 0, 14, 0, 0, 0], 2.3),
    "Table V Point Transformer": ([0, 0, 6, 0, 0, 0], 1.0),
    "Table V PointNet++ pre": ([5, 19, 17, 0, 0, 0], 6.8),
    "Table V PointNeXt pre": ([0, 36, 17, 0, 0, 0], 8.8),
    "Table VII w/o cropping": ([98, 69, 14, 19, 32, 40], 45.3),
    "Table XV DP3 long horizon": ([100, 64, 46, 99, 75, 85], 78.2),
    "Table XV DP long horizon": ([68, 44, 16, 12, 14, 44], 33.0),
    "Table XV DP short horizon": ([48, 50, 25, 15, 43, 63], 40.7),
}
res["ablation_mean_check"] = {k: {"calc": round(np.mean(v), 2), "paper": p, "ok": bool(abs(round(float(np.mean(v)), 1) - p) < 0.06)}
                              for k, (v, p) in abl.items()}

# ---------- 4. 실로봇 (Table VIII–XIV). 과제당 10회 ----------
real = {  # (성공 수, 시도 수) 과제별: Roll-Up, Dumpling, Drill, Pour
    "DP3": [9, 7, 8, 10], "DP": [0, 3, 7, 4], "DP-Depth": [4, 2, 1, 1],
}
N = 10


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round((c - h) * 100, 1), round((c + h) * 100, 1)


real_res = {}
for name, v in real.items():
    k, n = sum(v), N * len(v)
    real_res[name] = {"success": k, "trials": n, "rate_pct": round(100 * k / n, 1), "wilson95_pct": wilson(k, n),
                      "per_task_pct": [x * 10 for x in v],
                      "sd_across_tasks_pop": round(float(np.std(np.array(v) * 10)), 1),
                      "sd_across_tasks_sample": round(float(np.std(np.array(v) * 10, ddof=1)), 1),
                      "mean_across_tasks": round(float(np.mean(np.array(v) * 10)), 1)}
res["table8_real"] = real_res
res["table8_fisher"] = {
    "DP3_vs_DP_overall_p": float(f"{stats.fisher_exact([[34, 6], [14, 26]])[1]:.3g}"),
    "DP3_vs_Depth_overall_p": float(f"{stats.fisher_exact([[34, 6], [8, 32]])[1]:.3g}"),
    "DP3_vs_DP_per_task_p": [float(f"{stats.fisher_exact([[a, N - a], [b, N - b]])[1]:.3g}") for a, b in zip(real["DP3"], real["DP"])],
    "power_note": "과제당 10회라 과제 단위 차이는 대부분 검정력이 낮음",
}
# 일반화 실험 (시도 수 작음)
gen = {
    "IX spatial (Pour, 5 positions)": {"DP3": (4, 5), "DP": (0, 5), "DP-Depth": (0, 5)},
    "X appearance (Drill, 5 colors)": {"DP3": (5, 5), "DP": (0, 5), "DP-Depth": (0, 5)},
    "XI instance (Drill, 5 objects)": {"DP3": (5, 5), "DP": (1, 5), "DP-Depth": (1, 5)},
    "XII view (Roll-Up, 3 views)": {"DP3": (3, 3), "DP": (0, 3), "DP-Depth": (0, 3)},
    "XIII clutter (train color, 10)": {"DP3": (8, 10), "DP": (6, 10), "DP3+PointNeXt": (0, 10), "DP3+color": (8, 10)},
}
res["generalization_tables"] = {
    task: {m: {"k": k, "n": n, "wilson95_pct": wilson(k, n)} for m, (k, n) in d.items()} for task, d in gen.items()
}
res["generalization_fisher_p"] = {
    "X_DP3_vs_DP": float(f"{stats.fisher_exact([[5, 0], [0, 5]])[1]:.3g}"),
    "IX_DP3_vs_DP": float(f"{stats.fisher_exact([[4, 1], [0, 5]])[1]:.3g}"),
    "XIII_DP3_vs_DP": float(f"{stats.fisher_exact([[8, 2], [6, 4]])[1]:.3g}"),
    "XI_DP3_vs_DP": float(f"{stats.fisher_exact([[5, 0], [1, 4]])[1]:.3g}"),
    "XII_DP3_vs_DP": float(f"{stats.fisher_exact([[3, 0], [0, 3]])[1]:.3g}"),
}
# 안전 위반율 (Table XIV). 분모 10회 가정 (논문에 명시 없음)
viol = {"DP": [9, 2, 2, 0], "DP-Depth": [2, 3, 3, 2], "DP3": [0, 0, 0, 0]}
res["table14_safety"] = {k: {"mean_pct_calc": round(float(np.mean(np.array(v) * 10)), 2),
                             "counts_if_n10": v} for k, v in viol.items()}
res["table14_safety"]["note"] = "논문은 분모를 밝히지 않음. 평균 32.5, 25.0, 0.0은 과제 4개 단순 평균과 일치하고 10회 가정과 맞음"

# ---------- 5. 추론 속도 (Table XVI) ----------
res["table16_speed"] = {"dp_fps": 12.3, "dp3_fps": 12.7, "simple_fps": 25.3,
                        "simple_over_dp3_pct": round((25.3 / 12.7 - 1) * 100, 1),
                        "dp3_over_dp_pct": round((12.7 / 12.3 - 1) * 100, 1),
                        "note": "논문 본문의 DP3 추론 속도는 DP보다 약 3% 빠름 (같은 장치 측정 조건 미상)"}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    print(json.dumps(res, ensure_ascii=False, indent=1)[:9000])
