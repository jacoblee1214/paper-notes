"""재계산 결과(recompute_results.json)로 그래프 두 장을 그린다.

출력: report/figures/own_cells_diff.png  (셀별 FAKTUAL - random 차이와 Welch 95% 구간)
      report/figures/own_budget_curves.png (데모 수에 따른 평균 곡선, 어려운 과제 세 개)
실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))
TXT = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8").splitlines()


def font():
    import matplotlib.font_manager as fm
    for name in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def cells_plot():
    cells = RES["K3_cells"]
    # 셀별 Welch 구간은 시드 원자료가 필요하다. 원자료는 parse_tables로 다시 읽는다.
    import importlib.util
    spec = importlib.util.spec_from_file_location("rc", HERE / "recompute.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    tabs = rc.parse_tables(TXT)
    seed = {}
    for key in ("table5", "table6"):
        for r in tabs[key]:
            seed[(r["task"], r["demos"], r["method"])] = np.array(r["seeds"])
    diffs, lo, hi, labels = [], [], [], []
    for c in cells:
        f = seed[(c["task"], c["demos"], "FAKTUAL")]
        r = seed[(c["task"], c["demos"], "random")]
        d = f.mean() - r.mean()
        se = np.sqrt(f.var(ddof=1) / 3 + r.var(ddof=1) / 3)
        df = (f.var(ddof=1) / 3 + r.var(ddof=1) / 3) ** 2 / (
            (f.var(ddof=1) / 3) ** 2 / 2 + (r.var(ddof=1) / 3) ** 2 / 2)
        tcrit = stats.t.ppf(0.975, df) if se > 0 else 0
        labels.append(f'{c["task"]} {c["demos"]}')
        diffs.append(d)
        lo.append(d - tcrit * se)
        hi.append(d + tcrit * se)
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 7.6), dpi=170)
    ax.axvline(0, color="#888", lw=0.8)
    ax.errorbar(diffs, y, xerr=[np.array(diffs) - np.array(lo), np.array(hi) - np.array(diffs)],
                fmt="o", color="#1f5fae", ecolor="#9db8dc", capsize=2, ms=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("FAKTUAL - random (성공률 차이, 시드 3개 평균)")
    ax.set_title("셀별 차이와 Welch 95% 구간 (논문 표 5, 6 재계산)", fontsize=9)
    ax.grid(axis="x", color="#eee", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_cells_diff.png")
    plt.close(fig)


def budget_plot():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rc", HERE / "recompute.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    tabs = rc.parse_tables(TXT)
    pick = ["Square (PH)", "Transport (MH)", "Square (MH)"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=170, sharey=False)
    for ax, task in zip(axes, pick):
        rows = [r for r in tabs["table6"] if r["task"] == task]
        for method, color, mk in (("FAKTUAL", "#c2185b", "s"), ("random", "#2e7d32", "o")):
            pts = sorted((r["demos"], r["mean"], r["min"], r["max"]) for r in rows if r["method"] == method)
            x = [p[0] for p in pts]
            ax.plot(x, [p[1] for p in pts], marker=mk, color=color, ms=3.5, lw=1.2, label=method)
            ax.fill_between(x, [p[2] for p in pts], [p[3] for p in pts], color=color, alpha=0.10)
        full = [r for r in rows if r["method"] == "full dataset"]
        if full:
            ax.axhline(full[0]["mean"], color="#555", ls="--", lw=0.9, label="full dataset")
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("데모 수", fontsize=8)
        ax.set_ylabel("성공률 (3 시드 평균, 음영은 최소-최대)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(color="#eee", lw=0.6)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("논문 표 6의 데모 수별 곡선 (각 값은 체크포인트 최댓값)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_budget_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    font()
    cells_plot()
    budget_plot()
    print("wrote", FIG / "own_cells_diff.png", FIG / "own_budget_curves.png")
