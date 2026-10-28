"""Table I 재구성 분포 점그림 (재구성이 유일하지 않은 칸은 첫 번째 해를 표시). 실행: python3 analysis/plot_stats.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})

INK2, GRID, MUTED, ACCENT = "#55555a", "#e4e4e7", "#9a9aa0", "#1b6f6a"
res = json.loads((ROOT / "analysis" / "stats_check_results.json").read_text())
rec = res["nut_reconstruction"]
conds = ["none", "tactile", "hist", "tactile+hist"]
labels = ["없음", "촉각", "히스토리", "촉각+\n히스토리"]
nuts = [("square", "사각 너트"), ("triangular", "삼각 너트"), ("hexagonal", "육각 너트 (미학습)"), ("cross", "십자 너트 (미학습)")]

fig, axes = plt.subplots(1, 4, figsize=(10, 3.2), dpi=200, sharey=True)
rng = np.random.default_rng(0)
for ax, (nut, title) in zip(axes, nuts):
    for i, c in enumerate(conds):
        r = rec[f"{c}/{nut}"]
        vals = np.array(r["example"])
        col = ACCENT if c == "tactile+hist" else MUTED
        # 같은 값끼리 겹치지 않게 가로로 펼친다
        xs = np.zeros_like(vals)
        for v in np.unique(vals):
            idx = np.where(vals == v)[0]
            xs[idx] = (np.arange(len(idx)) - (len(idx) - 1) / 2) * 0.07
        ax.scatter(i + xs, vals, s=14, color=col, edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [vals.mean()] * 2, color="#222", lw=1.6, zorder=4)
    p = res["nut_tests"][f"{nut}: tactile+hist vs hist"]["welch_p"]
    ax.text(2.5, 110, f"p = {p:.2f}", ha="center", color=INK2, fontsize=8)
    ax.plot([2, 3], [106, 106], color=INK2, lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(4), labels, fontsize=8)
    ax.set_ylim(-6, 118)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2)
axes[0].set_ylabel("progress ratio (%)", color=INK2)
fig.text(0.01, 0.005, "점: 표의 평균±SD를 만족하도록 재구성한 시행 10회 (12.5% = 8회전 중 1회전 가정, 해가 여러 개면 하나만 표시)   검은 선: 평균   p: Welch t-test (히스토리 vs 촉각+히스토리)",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = ROOT / "report" / "figures" / "stats_nut_dots.png"
fig.savefig(out)
print("wrote", out)
