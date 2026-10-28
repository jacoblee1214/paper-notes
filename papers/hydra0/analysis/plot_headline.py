"""Table 2 Average 행: 물체·그리퍼 EPE를 모델별로 비교 (±SE). 실행: python3 analysis/plot_headline.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
INK2, GRID = "#55555a", "#e4e4e7"
GRAY, ORANGE, TEAL_L, TEAL = "#b5b5ba", "#d9822b", "#7fb3ad", "#1b6f6a"

# Table 2 Average 행 (논문 값)
rows = [("ATI\n(zero-shot)", 23.19, 2.74, 4.62, 0.14, GRAY), ("Wan-Move\n(zero-shot)", 21.53, 2.76, 4.67, 0.17, GRAY),
        ("Cosmos 2.5\n(6D EE 행동)", 13.23, 0.97, 34.28, 1.02, ORANGE),
        ("Ours\nCosmos 2B", 6.27, 0.67, 13.80, 0.85, TEAL_L), ("Ours\nWan 5B", 6.61, 0.42, 3.88, 0.12, TEAL_L),
        ("Ours\nWan A14B", 6.00, 0.41, 3.83, 0.13, TEAL_L), ("Ours\nA14B 4-step", 5.27, 0.35, 3.29, 0.10, TEAL)]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=200)
for ax, (title, vi, si) in zip(axes, (("물체 EPE (px, 낮을수록 좋음)", 1, 2), ("그리퍼 EPE (px, 낮을수록 좋음)", 3, 4))):
    xs = range(len(rows))
    ax.bar(xs, [r[vi] for r in rows], yerr=[r[si] for r in rows], color=[r[5] for r in rows],
           width=0.62, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    for x, r in zip(xs, rows):
        ax.text(x, r[vi] + r[si] + 0.4, f"{r[vi]:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(list(xs), [r[0] for r in rows], fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)
axes[1].annotate("헤드라인 −90.4%의 기준선\n(제로샷 ATI보다 7.4배 나쁨)", xy=(2, 34.28), xytext=(3.3, 27),
                 fontsize=7.5, color=ORANGE, arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 0.8})
fig.text(0.01, 0.01, "Table 2 Average 행(5개 검증셋 무가중 평균, 물체 EPE는 DROID 제외 4개). 오차막대: 논문 보고 SE.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = ROOT / "report" / "figures" / "headline_epe.png"
fig.savefig(out)
print("wrote", out)
