"""5회 시행 성공률 격자와 본문의 개선 폭을 대조하는 그림(자체 그래프, 논문 그림 아님).

실행: python3 report/plot_planning.py  -> report/figures/own_planning_check.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures" / "own_planning_check.png"

font = next((f for f in ["Apple SD Gothic Neo", "Noto Sans CJK KR", "AppleGothic"]
             if any(f == x.name for x in font_manager.fontManager.ttflist)), None)
if font:
    plt.rcParams["font.family"] = font
plt.rcParams["axes.unicode_minus"] = False

tasks = ["push fruits", "reach & push", "wipe cloth", "stack cubes"]
reported = [10, 35, 31, 11]
grid = [round(100 * k / 5) for k in range(0, 6)]  # 0, 20, ... 100

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=170)
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(tasks, fontsize=9)
ax.set_ylabel("VT-WM minus V-WM (%p)", fontsize=9)
for g in range(0, 101, 20):
    ax.axhline(g, color="#dcdad3", lw=0.6, zorder=0)
ax.bar(range(len(tasks)), reported, color="#1f5fae", width=0.5, label="reported gain (text)", zorder=2)
ax.set_ylim(0, 105)
ax.set_title("Reported gains vs. the 5-trial grid (multiples of 20 %p)", fontsize=10)
ax.legend(fontsize=8, loc="upper right", frameon=False)
for i, v in enumerate(reported):
    ax.text(i, v + 2, f"{v}%", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
