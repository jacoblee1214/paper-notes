"""우리가 그린 그래프 두 장 (report/figures/own_*.png). 실행: python3 analysis/plot_figures.py

own_success.png: 표 1 성공률과 Wilson 95% 구간 (크기별, 전체)
own_cost.png:    표 2A 인코딩 영역 수와 인코더 시간 (로컬 재사용의 효과)
색: 검증 완료 범주 팔레트 #1f5fae(TacPilot), #b3561a(DTR), #2e8b6a(ACT). 텍스트는 어두운 잉크색.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import math


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

FIG = Path(__file__).resolve().parents[1] / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COL = {"TacPilot": "#1f5fae", "DTR": "#b3561a", "ACT": "#2e8b6a"}
INK, MUTED, GRID = "#1b1b1a", "#55544f", "#e4e2dc"
plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "DejaVu Sans"], "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK})

# 1. 성공률
T1 = {"ACT": {"4x8": (7, 16), "5x10": (8, 14), "전체": (15, 30)},
      "DTR": {"4x8": (13, 16), "5x10": (8, 14), "전체": (21, 30)},
      "TacPilot": {"4x8": (13, 16), "5x10": (13, 14), "전체": (26, 30)}}
groups = ["4x8", "5x10", "전체"]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
w = 0.26
for i, m in enumerate(["ACT", "DTR", "TacPilot"]):
    xs = np.arange(len(groups)) + (i - 1) * w
    for j, g in enumerate(groups):
        k, n = T1[m][g]
        lo, hi = wilson(k, n)
        p = k / n
        ax.bar(xs[j], p, width=w * 0.92, color=COL[m], label=m if j == 0 else None)
        ax.errorbar(xs[j], p, yerr=[[p - lo], [hi - p]], color=INK, capsize=2.5, lw=0.8)
        ax.text(xs[j], hi + 0.03, f"{k}/{n}", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(np.arange(len(groups)))
ax.set_xticklabels(["4 x 8 cm", "5 x 10 cm", "전체 (30회)"])
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
ax.set_title("표 1 성공률 (Wilson 95% 구간, 논문 값 재표시)", loc="left", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG / "own_success.png")
plt.close(fig)

# 2. 인코딩 비용 (표 2A)
labels = ["Periodic/Full", "Event/Full", "Periodic/Local", "Event/Local"]
regions = [474912, 408096, 206975, 206975]
enc = [30.60, 26.16, 19.82, 19.17]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4), dpi=200)
bar_col = ["#9a9891"] * 2 + [COL["TacPilot"]] * 2  # 회색: 전체 인코딩, 파랑: 로컬 재사용
for ax, vals, title, fmt in [(a1, regions, "인코딩 영역 수 (누적)", "{:,.0f}"),
                             (a2, enc, "인코더 누적 시간 (s)", "{:.2f}")]:
    y = np.arange(len(labels))
    ax.barh(y, vals, color=bar_col, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v * 1.02, yi, fmt.format(v), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels if ax is a1 else [""] * len(labels))
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "회색: 매 주기 전체 인코딩 (Full), 파랑: 로컬 재사용 (Local). 오프라인 재생, 논문 표 2(A).",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIG / "own_cost.png")
plt.close(fig)
print("wrote", FIG)
