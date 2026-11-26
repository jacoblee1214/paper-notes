"""재계산 결과로 그린 자체 그래프 세 장 (논문 그림은 쓰지 않는다).

출력: report/figures/own_table1_ci.png, own_diff_forest.png, own_generalization.png
실행: python3 analysis/plot_figures.py   (matplotlib, analysis/recompute_results.json 필요)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text())
OUTDIR = ROOT / "report" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

for cand in ["Noto Sans CJK KR", "Noto Sans CJK JP", "Apple SD Gothic Neo"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

TASKS = ["S1", "S2", "S3", "S4"]
TASK_LABEL = {"S1": "S1 버튼 누르기\n(n=42)", "S2": "S2 막대 꺼내기\n(n=37)", "S3": "S3 커튼 뒤 물체\n(n=50)", "S4": "S4 닫힌 캐비닛\n(n=55)"}
METHODS = ["TVC", "WC", "TVC w/ WC", "FTC", "FTC w/ WC", "w/o camera poses", "w/o joint currents", "Ours"]
COLORS = ["#9a9a9a", "#b7b7b7", "#7a8fa6", "#d08a3c", "#e0b070", "#7fb08a", "#4f9a63", "#1f5f8b"]


def plot_table1():
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.3), dpi=150, sharey=True)
    for ax, t in zip(axes, TASKS):
        for i, m in enumerate(METHODS):
            lo, hi = RES["V9_wilson"][m][t]
            row = next(r for r in RES["V1_cells"] if r["method"] == m and r["task"] == t)
            p = row["k"] / row["n"]
            ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color=COLORS[i], capsize=2.5, ms=5)
        ax.set_title(TASK_LABEL[t], fontsize=9)
        ax.set_xlim(0, 1.02)
        ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8)
        ax.tick_params(labelleft=(t == "S1"))
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        ax.set_xlabel("성공률", fontsize=8)
    fig.suptitle("표 1 재계산: 방법별 성공률과 Wilson 95% 구간 (시도 수는 논문 부록 A.1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTDIR / "own_table1_ci.png")
    plt.close(fig)


def plot_forest():
    rows = RES["V10_pairwise_task"]
    pooled = {r["baseline"]: r for r in RES["V10_pairwise_pooled"]}
    bases = [m for m in METHODS if m != "Ours"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    y = 0
    ticks, labels = [], []
    for m in bases:
        for t in TASKS:
            r = next(x for x in rows if x["baseline"] == m and x["task"] == t)
            lo, hi = r["newcombe95"]
            ax.errorbar(r["diff_pts"], y, xerr=[[r["diff_pts"] - lo], [hi - r["diff_pts"]]], fmt="o", color="#7a8fa6", capsize=2, ms=4)
            y += 1
        pr = pooled[m]
        lo, hi = pr["newcombe95"]
        ax.errorbar(pr["diff_pts"], y, xerr=[[pr["diff_pts"] - lo], [hi - pr["diff_pts"]]], fmt="D", color="#1f5f8b", capsize=3, ms=6)
        ticks.append(y - 2.5)
        labels.append(f"Ours - {m}")
        y += 1.6
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("성공률 차이 (퍼센트포인트), Newcombe 95% 구간. 회색 점은 과제별, 파란 마름모는 4과제 합산(184회)", fontsize=8)
    ax.set_title("Ours 대 각 방법의 성공률 차이 (비교마다 독립 표본으로 계산, 같은 초기 조건 쌍 정보는 없음)", fontsize=9)
    ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    fig.tight_layout()
    fig.savefig(OUTDIR / "own_diff_forest.png")
    plt.close(fig)


def plot_generalization():
    cats = RES["V12_categories"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    colors = {"S1": "#1f5f8b", "S2": "#d08a3c", "S3": "#4f9a63", "S4": "#9b5b9a"}
    for i, c in enumerate(cats):
        p = c["k"] / c["n"]
        lo, hi = c["wilson"]
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", color=colors[c["task"]], capsize=2.5, ms=5)
        ax.text(i, 0.02, f'{c["k"]}/{c["n"]}', ha="center", fontsize=6.5, rotation=90, color="#444")
    ax.set_xticks(range(len(cats)), [f'{c["task"]} {c["cat"]}' for c in cats], rotation=70, fontsize=7, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ours 성공률 (Wilson 95% 구간)")
    ax.set_title("표 4: 범주별 성공률. 범주당 시도는 3–25회이고 22개 중 16개가 6회 이하 (기준선의 범주별 값은 보고되지 않음)", fontsize=9)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    fig.tight_layout()
    fig.savefig(OUTDIR / "own_generalization.png")
    plt.close(fig)


plot_table1()
plot_forest()
plot_generalization()
print("wrote", [p.name for p in sorted(OUTDIR.glob("own_*.png"))])
