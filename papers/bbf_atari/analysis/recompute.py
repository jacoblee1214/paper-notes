"""BBF (arXiv:2305.19452) 수치 재계산과 검정.

입력 1: analysis/table_a1.csv  논문 Table A.1(p.14)의 26 게임 점수 (저자가 쓴 값을 옮김)
입력 2: repo/gr/bigger_better_faster/scores/*.csv  공식 저장소의 실행별 점수 (게임 x 시드 한 줄)
        저장소 얕은 복제가 필요합니다(README 참고). 코드는 실행하지 않고 CSV만 읽습니다.
출력: analysis/recompute_results.json, 표준출력 요약

실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "repo" / "gr" / "bigger_better_faster" / "scores"
OUT = Path(__file__).resolve().parent / "recompute_results.json"
RNG = np.random.default_rng(0)
N_BOOT = 2000
COLS = ["DER", "DrQeps", "SPR", "IRIS", "SRSPR", "EfficientZero", "BBF"]
PAPER_AGG = {  # 논문 Table A.1 하단 행 (Games>Human, IQM, Optimality gap, Median, Mean)
    "DER": (2, 0.183, 0.698, 0.189, 0.350), "DrQeps": (3, 0.280, 0.631, 0.313, 0.465),
    "SPR": (6, 0.337, 0.577, 0.396, 0.616), "IRIS": (9, 0.501, 0.512, 0.289, 1.046),
    "SRSPR": (9, 0.631, 0.433, 0.685, 1.272), "EfficientZero": (14, 1.020, 0.371, 1.116, 1.945),
    "BBF": (12, 1.045, 0.344, 0.917, 2.247),
}


def iqm(x):
    """interquartile mean: 중간 50%의 평균 (rliable 정의, scipy trim_mean 0.25)"""
    return float(stats.trim_mean(np.asarray(x, float), 0.25))


def agg(x):
    x = np.asarray(x, float)
    return {"IQM": iqm(x), "median": float(np.median(x)), "mean": float(x.mean()),
            "optimality_gap": float(np.mean(np.clip(1 - x, 0, None)))}


# ---------- 1. 논문 Table A.1 에서 집계 재계산 ----------
with open(Path(__file__).resolve().parent / "table_a1.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
games26 = [r["game"] for r in rows]
hn = {}
for c in COLS:
    v = [(float(r[c]) - float(r["random"])) / (float(r["human"]) - float(r["random"])) for r in rows]
    hn[c] = np.array(v)

table_check = {}
for c in COLS:
    x = hn[c]
    a = agg(x)
    gh = int(np.sum([float(r[c]) > float(r["human"]) for r in rows]))
    p = PAPER_AGG[c]
    table_check[c] = {
        "games_gt_human": {"recomputed": gh, "paper": p[0]},
        "IQM_per_game_means": {"recomputed": round(a["IQM"], 3), "paper": p[1]},
        "optimality_gap": {"recomputed": round(a["optimality_gap"], 3), "paper": p[2]},
        "median": {"recomputed": round(a["median"], 3), "paper": p[3]},
        "mean": {"recomputed": round(a["mean"], 3), "paper": p[4]},
    }

# Table A.1 의 IQM은 실행(run) 단위 집계입니다. 여기서는 게임 평균 26개로 대신 계산하므로 차이가 날 수 있습니다.

# ---------- 2. 공식 저장소 실행별 점수로 집계 ----------
def load(name):
    """CSV 한 파일을 읽어 {game: [정규화 점수...]}로 돌려준다 (26 게임만)."""
    out = {g: [] for g in games26}
    raw = {g: [] for g in games26}
    with open(SCORES / name, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g = r["game"]
            if g in out:
                out[g].append(float(r["GameScoreNormalized"]))
                raw[g].append(float(r["GameScoreAverage"]))
    return out, raw


def pooled(d):
    return np.concatenate([np.array(d[g]) for g in games26])


def boot_iqm_ci(d, stat=iqm):
    """게임 안에서 실행을 복원추출하는 층화 부트스트랩 95% CI"""
    arrs = [np.array(d[g]) for g in games26]
    vals = []
    for _ in range(N_BOOT):
        vals.append(stat(np.concatenate([RNG.choice(a, len(a), replace=True) for a in arrs])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], vals


def run_counts(d):
    c = [len(d[g]) for g in games26]
    return {"min": min(c), "max": max(c), "total": sum(c)}


FILES = {
    "BBF_RR8": "RR8_BBF.csv",
    "BBF_RR2": "RR2_BBF.csv",
    "BBF_RR8_sticky": "RR8_BBF+sticky.csv",
    "BBF_RR8_no_SPR": "RR8_BBF-SPR.csv",
    "BBF_RR8_no_WD": "RR8_BBF-WD.csv",
    "BBF_RR8_no_annealing": "RR8_BBF-Annealing.csv",
    "BBF_RR8_no_harder_resets": "RR8_BBF-HarderResets.csv",
    "BBF_RR8_no_resets": "RR8_BBF-Resets.csv",
    "BBF_RR8_n10": "RR8_BBF+n=10.csv",
    "BBF_RR8_gamma099": "RR8_BBF+\\gamma=0.99.csv",
    "BBF_RR2_no_SPR": "RR2_BBF-SPR.csv",
    "BBF_RR2_no_WD": "RR2_BBF-WD.csv",
    "BBF_RR2_no_annealing": "RR2_BBF-Annealing.csv",
    "BBF_RR2_no_harder_resets": "RR2_BBF-HarderResets.csv",
    "BBF_RR2_no_resets": "RR2_BBF-Resets.csv",
    "BBF_RR2_n10": "RR2_BBF+n=10.csv",
    "BBF_RR2_gamma099": "RR2_BBF+\\gamma=0.99.csv",
}

data = {}
for k, fn in FILES.items():
    if (SCORES / fn).exists():
        data[k] = load(fn)

pool_summary = {}
for k, (d, raw) in data.items():
    a = agg(pooled(d))
    ci, _ = boot_iqm_ci(d)
    pool_summary[k] = {"runs_per_game": run_counts(d), "IQM": round(a["IQM"], 3),
                       "IQM_CI95_stratified": [round(ci[0], 3), round(ci[1], 3)],
                       "median": round(a["median"], 3), "mean": round(a["mean"], 3),
                       "optimality_gap": round(a["optimality_gap"], 3)}

# BBF_RR8 의 게임별 평균을 논문 표와 비교
d8, raw8 = data["BBF_RR8"]
per_game_cmp = {}
for r in rows:
    g = r["game"]
    m = float(np.mean(raw8[g])) if g in raw8 else None
    per_game_cmp[g] = {"csv_mean_raw": round(m, 1), "paper_raw": float(r["BBF"]),
                       "diff_pct": round(100 * (m - float(r["BBF"])) / float(r["BBF"]), 2)}
max_abs_pct = max(abs(v["diff_pct"]) for v in per_game_cmp.values())

# 같은 CSV를 게임 평균으로 줄인 뒤(표 A.1 하단 행과 같은 방식) 집계
gm_norm = np.array([float(np.mean(d8[g])) for g in games26])
per_game_agg = agg(gm_norm)
top = sorted(per_game_cmp.items(), key=lambda kv: -abs(kv[1]["diff_pct"]))[:4]

# 같은 CSV에서 시드 50개만 골라 게임 평균을 내 봐도 표와 일치하지 않음 (앞 50, 뒤 50 시드)
seed_subset = {}
with open(SCORES / FILES["BBF_RR8"], encoding="utf-8") as f:
    allr = list(csv.DictReader(f))
for mode in ("first50", "last50"):
    vals = {}
    for g in games26:
        rr = sorted([(int(r["seed"]), float(r["GameScoreAverage"])) for r in allr if r["game"] == g])
        sel = rr[:50] if mode == "first50" else rr[-50:]
        vals[g] = float(np.mean([v for _, v in sel]))
    seed_subset[mode] = round(float(np.mean([abs(vals[g] - float(r["BBF"])) / float(r["BBF"]) for g, r in zip(games26, rows)])) * 100, 2)

# ---------- 3. 대응 비교 (같은 게임 쌍) ----------
def paired(k1, k2):
    d1, _ = data[k1]
    d2, _ = data[k2]
    m1 = np.array([np.mean(d1[g]) for g in games26])
    m2 = np.array([np.mean(d2[g]) for g in games26])
    diff = m1 - m2
    w = stats.wilcoxon(m1, m2)
    return {"IQM_diff_point": round(iqm(pooled(d1)) - iqm(pooled(d2)), 3),
            "games_better": int(np.sum(diff > 0)), "games_worse": int(np.sum(diff < 0)),
            "wilcoxon_p_game_level": float(w.pvalue)}

comparisons = {}
for k in data:
    if k.startswith("BBF_RR8") and k != "BBF_RR8" and "sticky" not in k:
        comparisons[f"{k} vs BBF_RR8"] = paired(k, "BBF_RR8")
if "BBF_RR2" in data:
    comparisons["BBF_RR2 vs BBF_RR8"] = paired("BBF_RR2", "BBF_RR8")
    for k in ("BBF_RR2_no_SPR", "BBF_RR2_no_WD", "BBF_RR2_no_resets", "BBF_RR2_no_harder_resets"):
        if k in data:
            comparisons[f"{k} vs BBF_RR2"] = paired(k, "BBF_RR2")

# 26 게임 기준 BBF(RR8) 와 Table A.1 의 BBF 열 IQM 차이, Figure 6 의 0.45 와 비교
gap_table = PAPER_AGG["BBF"][1] - PAPER_AGG["SRSPR"][1]
# 논문 Figure 3 파라미터 수 비율 (x축 값 읽기: 논문 Fig.3)
cnn = [0.20, 0.44, 1.10, 3.06, 9.56, 32.88]
resnet = [0.15, 0.32, 0.79, 2.13, 6.47, 21.80]
ratio = [round(c / r, 3) for c, r in zip(cnn, resnet)]

# 시간 주장 확인: 100k 스텝 = 400k 프레임 @ 60 FPS
minutes_100k = 400_000 / 60 / 60

results = {
    "paper_table_a1_check": table_check,
    "per_game_csv_vs_table_BBF_RR8": {"max_abs_pct_diff": round(max_abs_pct, 2), "largest": top, "per_game": per_game_cmp},
    "csv_BBF_RR8_per_game_mean_aggregates": {k: round(v, 3) for k, v in per_game_agg.items()},
    "seed_subset_test_mean_abs_pct_diff": seed_subset,
    "csv_pooled_runs": pool_summary,
    "paired_comparisons": comparisons,
    "fig6_claim_vs_table": {"paper_claim_IQM_gap": 0.45, "table_IQM_gap_BBF_minus_SRSPR": round(gap_table, 3),
                             "IQM_ratio_BBF_over_SRSPR": round(PAPER_AGG["BBF"][1] / PAPER_AGG["SRSPR"][1], 3)},
    "fig3_param_ratio_CNN_over_ResNet": ratio,
    "minutes_100k_steps_at_60fps": round(minutes_100k, 1),
    "data_scaling_cost_ratio_RR2_1M_vs_RR8_100k": round((1_000_000 * 2) / (100_000 * 8), 2),
    "n_boot": N_BOOT,
}
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

print("== Table A.1 재계산 (게임 평균 기준) ==")
for c, v in table_check.items():
    print(c, v)
print("\n== CSV 실행 단위 집계 (26 게임 pooled, 층화 부트스트랩 95% CI) ==")
for k, v in pool_summary.items():
    print(k, v)
print("\n== 대응 비교 (게임 단위 Wilcoxon) ==")
for k, v in comparisons.items():
    print(k, v)
print("\nmax |CSV 게임 평균 - 표 A.1| (%):", round(max_abs_pct, 2), " 시드 부분집합 평균 절대차 (%):", seed_subset)
print("CSV 게임평균 집계 (표 A.1 하단과 같은 방식):", {k: round(v, 3) for k, v in per_game_agg.items()})
print("가장 큰 차이 4게임:", top)
print("Fig.6 0.45 vs 표:", results["fig6_claim_vs_table"])
print("CNN/ResNet 파라미터 비:", ratio)


# ---------- 4. 55 게임 파일: 학습 곡선과 held-out 29 게임 ----------
def load_all(name, games):
    out = {g: [] for g in games}
    with open(SCORES / name, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["game"] in out:
                out[r["game"]].append(float(r["GameScoreNormalized"]))
    return out


all55 = sorted({r["game"] for r in csv.DictReader(open(SCORES / "RR8_BBF.csv", encoding="utf-8"))})
held29 = [g for g in all55 if g not in games26]
curve = {}
for step in ["20k", "40k", "60k", "80k", "100k", "200k", "500k", "1M"]:
    d = load_all(f"RR2_BBF+sticky_{step}.csv", all55)
    curve[step] = {"runs_per_game": run_counts(d), "IQM_55games": round(iqm(np.concatenate([d[g] for g in all55])), 3)}
held = {}
for name, fn in [("BBF_RR8_sticky_55", "RR8_BBF+sticky.csv"), ("BBF_RR8_det_26_only", "RR8_BBF.csv")]:
    d = load_all(fn, all55)
    held[name] = {
        "IQM_26_games": round(iqm(np.concatenate([d[g] for g in games26])), 3),
        "IQM_29_held_out": round(iqm(np.concatenate([d[g] for g in held29])), 3),
        "runs_29": sum(len(d[g]) for g in held29),
    }
results["curve_RR2_sticky_55games"] = curve
results["heldout_29_check"] = {"n_held_out": len(held29), "held_out_games": held29, **held}
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n== RR2 sticky 학습 곡선 (55 게임, pooled IQM) ==")
for k, v in curve.items():
    print(k, v)
print("== held-out 29 게임 ==", results["heldout_29_check"])

# 26 게임 기준 RR=2 sticky 학습 곡선 (논문 그림 13의 BBF 곡선 대응)
curve26 = {}
for step in ["20k", "40k", "60k", "80k", "100k"]:
    d = load_all(f"RR2_BBF+sticky_{step}.csv", games26)
    curve26[step] = {"IQM_26": round(iqm(np.concatenate([d[g] for g in games26])), 3),
                     "runs_per_game": run_counts(d)}
results["curve_RR2_sticky_26games"] = curve26
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print("== RR2 sticky 26게임 곡선 ==", curve26)


# ---------- 5. 실행 수가 다른 두 팔의 IQM 차이 부트스트랩 (팔마다 독립 복원추출, 게임 층화) ----------
def boot_diff(da, db, n=N_BOOT):
    aa = [np.array(da[g]) for g in games26]
    ab = [np.array(db[g]) for g in games26]
    out = []
    for _ in range(n):
        ia = iqm(np.concatenate([RNG.choice(a, len(a), replace=True) for a in aa]))
        ib = iqm(np.concatenate([RNG.choice(b, len(b), replace=True) for b in ab]))
        out.append(ia - ib)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


diff_ci = {}
for k in data:
    if k in ("BBF_RR8", "BBF_RR2") or "sticky" in k:
        continue
    base = "BBF_RR2" if "RR2" in k else "BBF_RR8"
    if base in data:
        diff_ci[f"{k} - {base}"] = boot_diff(data[k][0], data[base][0])
results["bootstrap_diff_CI95_IQM"] = diff_ci
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n== IQM 차이 95% CI (부트스트랩, 실행 수 반영) ==")
for k, v in diff_ci.items():
    print(k, v)


# ---------- 6. Figure 8 대응: RR=2 절제 실험을 29 held-out 게임에서 ----------
fig8 = {}
for k in ["BBF_RR2", "BBF_RR2_no_SPR", "BBF_RR2_no_WD", "BBF_RR2_no_annealing", "BBF_RR2_no_harder_resets",
          "BBF_RR2_no_resets", "BBF_RR2_n10", "BBF_RR2_gamma099"]:
    fn = FILES[k]
    d = load_all(fn, all55)
    fig8[k] = {"IQM_29": round(iqm(np.concatenate([d[g] for g in held29])), 3),
               "runs_29": sum(len(d[g]) for g in held29)}
base29 = fig8["BBF_RR2"]["IQM_29"]
for k, v in fig8.items():
    v["pct_change_vs_full"] = round(100 * (v["IQM_29"] - base29) / base29, 1)
results["fig8_heldout29_RR2"] = fig8
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n== Figure 8 대응 (RR=2, 29 held-out 게임) ==")
for k, v in fig8.items():
    print(k, v)
