"""근사 판독값(recompute.py의 입력)으로 우리가 그린 그래프 두 장을 만든다.

출력: report/figures/own_ablation_read.png, report/figures/own_scatter_read.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

# 1. 그림 10 읽은 값 (평균 %, 반폭 %)
labels = ["π0.5", "no WD", "no CE", "no ME", "no CE·ME"]
means = [78, 73, 51, 54, 40]
half = [4, 5.5, 4.5, 4.5, 4.5]
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
ax.bar(labels, means, yerr=half, capsize=4, color=["#c9a227", "#2f6f3e", "#3d8a4f", "#3d8a4f", "#8fb878"],
       edgecolor="#333", linewidth=0.6)
ax.set_ylim(0, 100)
ax.set_ylabel("평균 과제 진행도 (%)")
ax.set_title("제거 ablation (mock 집, 그림 10 근사 판독)", fontsize=10)
ax.grid(axis="y", alpha=0.3)
for x, m in enumerate(means):
    ax.text(x, m + half[x] + 2, f"{m}", ha="center", fontsize=8)
ax.text(0, -22, "막대 높이와 오차는 그림에서 눈으로 읽은 값(±2 pt). 원자료 아님.", fontsize=6.5, ha="left",
        transform=ax.transData, color="#555")
fig.tight_layout()
fig.savefig(OUT / "own_ablation_read.png")
plt.close(fig)

# 2. 그림 12 과제별 읽은 값
tasks = ["접시", "서랍", "빨래", "침대"]
series = {
    "π0.5": [91, 85, 80, 58],
    "π0-FAST+Flow (HL 없음)": [55, 71, 73, 48],
    "π0 300k": [41, 32, 31, 20],
    "π0 80k": [19, 5, 39, 14],
}
colors = ["#c9a227", "#d9d6cc", "#2f6f3e", "#8fb878"]
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
width = 0.2
for i, (name, vals) in enumerate(series.items()):
    xs = [j + (i - 1.5) * width for j in range(len(tasks))]
    ax.bar(xs, vals, width=width, label=name, color=colors[i], edgecolor="#333", linewidth=0.5)
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(tasks)
ax.set_ylim(0, 100)
ax.set_ylabel("과제 진행도 (%)")
ax.set_title("과제별 비교 (그림 12 근사 판독, 검정 표시 없음)", fontsize=10)
ax.legend(fontsize=7, loc="upper right", ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "own_scatter_read.png")
plt.close(fig)
print("wrote", OUT)
