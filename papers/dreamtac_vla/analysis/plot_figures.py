"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_ci.png : 표 1 성공률과 Wilson 95% 구간 (실행 3회 x 100회 = 과제당 300회 가정)
own_sd_check.png: 표 1 보고 SD와 이항 SD(100회 한 번) 비교, 세 정수 실행의 최소 SD 선
own_real_sim.png: 같은 모델의 실로봇(표 1)과 시뮬레이션(표 4) 성공률 차이
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"
TASKS = ["Peg-in-Hole", "USB Insert", "Gear Assembly", "Tool Stabilize"]
T1 = {
    "ACT": [35.2, 62.6, 22.4, 19.3], "Diffusion Policy": [35.5, 56.3, 33.1, 30.4], "π0": [48.7, 59.4, 45.2, 41.0],
    "ACT (w/ Tactile)": [45.6, 63.1, 40.2, 39.1], "HSA-Only": [60.8, 63.7, 51.5, 42.9],
    "Dream-Only": [75.4, 75.2, 64.9, 68.5], "Full (HSA & Dream)": [95.0, 85.7, 81.1, 74.6],
}
T4 = {"π0": [83.5, 85.9, 77.2, 74.6], "HSA-Only": [89.1, 90.4, 84.6, 82.7],
      "Dream-Only": [93.8, 94.6, 88.9, 90.1], "Full (HSA & Dream)": [98.6, 97.9, 95.2, 94.7]}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_real_ci():
    models = list(T1)
    cols = ["#c9c7bf", "#b3b1a8", "#9a9890", "#e0b97a", "#8fb0d9", "#4f84c4", ACC]
    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    w = 0.12
    x = np.arange(4)
    for i, m in enumerate(models):
        vals = T1[m]
        lo = [wilson(round(v / 100 * 300), 300)[0] for v in vals]
        hi = [wilson(round(v / 100 * 300), 300)[1] for v in vals]
        xs = x + (i - 3) * w
        ax.bar(xs, vals, width=w, color=cols[i], label=m)
        ax.errorbar(xs, vals, yerr=[np.array(vals) - lo, np.array(hi) - np.array(vals)], fmt="none", ecolor=INK, capsize=1.5, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.16))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=200)
    plt.close(fig)


def plot_sd_check():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    cells = rec["V5_wilson_and_sd"]
    xs = [v["binomial_sd_per_100"] for v in cells.values()]
    ys = [v["reported_sd"] for v in cells.values()]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(xs, ys, color=ACC, s=26)
    lim = max(xs) * 1.1
    ax.plot([0, lim], [0, lim], color=GREY, lw=1, ls="--", label="보고 SD = 이항 SD(100회)")
    ax.axhline(0.577, color="#b5483a", lw=1, ls=":", label="정수 %로 3회 실행할 때 0이 아닌 최소 SD (0.58)")
    ax.set_xlabel("이항 SD, 100회 한 번 (%p)")
    ax.set_ylabel("보고된 SD (%p, 표 1)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_sd_check.png", dpi=200)
    plt.close(fig)


def plot_real_sim():
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.3), sharey=True)
    names = list(T4)
    for t, ax in enumerate(axes):
        for i, m in enumerate(names):
            r, s = T1[m][t], T4[m][t]
            ax.plot([r, s], [i, i], color=GREY, lw=2)
            ax.scatter([r], [i], color=ACC, s=34, zorder=3, label="실로봇 (표 1)" if (t == 0 and i == 0) else None)
            ax.scatter([s], [i], color="#e0a030", s=34, zorder=3, label="시뮬레이션 (표 4)" if (t == 0 and i == 0) else None)
        ax.set_title(TASKS[t], fontsize=9)
        ax.set_xlim(30, 102)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.tick_params(labelleft=(t == 0))
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("성공률 (%)", fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "own_real_sim.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_real_ci()
    plot_sd_check()
    plot_real_sim()
    print("wrote", [p.name for p in FIG.glob("own_*.png")])
