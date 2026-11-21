"""우리가 그린 그래프 두 장. 실행: python3 analysis/plot_figures.py -> report/figures/own_*.png
입력 수치는 논문 값(표 3, 그림 2 성공 횟수)과 공개 프로젝트 페이지 값이다. 그림 자체는 논문 그림을 쓰지 않는다.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

KO_FONTS = ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic"]
avail = {f.name for f in font_manager.fontManager.ttflist}
for name in KO_FONTS:
    if name in avail:
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


# 그림 1: 실로봇 towel folding 성공률 (Wilson 95% 구간), 논문 그림 2의 횟수로 계산
labels = ["ACT\n101 ep", "ACT\n562 ep", "π0.5 base\n(미튜닝)", "π0.5\n562 ep"]
succ = [0, 12, 0, 7]
n = 20
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
xs = range(len(labels))
rates = [s / n * 100 for s in succ]
errs = [(r - wilson(s, n)[0], wilson(s, n)[1] - r) for r, s in zip(rates, succ)]
ax.bar(xs, rates, color=["#9aa5b1", "#1f5fae", "#9aa5b1", "#1f5fae"], width=0.6)
ax.errorbar(xs, rates, yerr=[[e[0] for e in errs], [e[1] for e in errs]], fmt="none", ecolor="#333", capsize=4)
for x, s, r in zip(xs, succ, rates):
    ax.text(x, r + 4, f"{s}/20", ha="center", fontsize=9)
ax.set_xticks(list(xs))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(0, 100)
ax.set_ylabel("성공률 (%)")
ax.set_title("PiPER towel folding, 20회 시도 (Wilson 95% 구간)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_success_ci.png")
plt.close(fig)

# 그림 2: 논문 표 3(arXiv v1, 641 에피소드 중 PiPER 기준)과 프로젝트 페이지 표(20개 에피소드)의 IK 위치 오차 중앙값 비교
arms = ["PiPER", "OpenArm v1", "TRLC-DK1", "I2RT YAM", "Metal", "R1 Lite", "ARX X5"]
arxiv = [1.17, 0.14, 0.18, 0.19, 0.09, 0.03, 0.09]
project = [0.80, 0.10, 0.09, 0.09, 0.11, 0.03, 0.09]
fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=200)
import numpy as np
x = np.arange(len(arms))
ax.bar(x - 0.2, arxiv, 0.38, color="#1f5fae", label="arXiv v1 표 3 (전체 에피소드 기준)")
ax.bar(x + 0.2, project, 0.38, color="#9aa5b1", label="프로젝트 페이지 표 (20개 에피소드 기준)")
ax.axhline(1.0, color="#b03a2e", lw=0.8, ls="--")
ax.text(len(arms) - 0.5, 1.03, "1 mm", color="#b03a2e", fontsize=8, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(arms, fontsize=8)
ax.set_ylabel("위치 오차 중앙값 (mm)")
ax.set_title("시뮬 IK 위치 오차 중앙값: 두 출처의 비교", fontsize=10)
ax.legend(fontsize=7.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_table3_versions.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
