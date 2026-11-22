"""보고 수치로 우리가 그린 그래프 세 장 (matplotlib). 논문 그림은 쓰지 않는다.

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
입력: 보고 표 값(하드코딩, 출처는 recompute.py 주석과 같음). 구간은 시행 수가 확정된 표 5에만 Wilson 95%로 그린다.
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for cand in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

INK = "#1b1b1a"
ACC = "#1f5fae"
GREY = "#8c8a83"
ORANGE = "#c8702a"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ------------------------------------------------ 그림 1: 표 5 하위 집단 (시행 수 확정)
SUB = ["기호 1", "기호 2", "기호 3", "수학", "로고", "영양", "색·다국어", "유명인", "CelebA"]
N = [15, 20, 15, 20, 25, 25, 40, 15, 15]
K = {  # 정수 성공 횟수(부록 H.2 표 5 값에서 역산, v1과 CoRL 판본 동일)
    "RT-1": [4, 4, 0, 1, 0, 8, 11, 3, 3],
    "RT-2-PaLI-X-55B": [14, 12, 14, 5, 13, 12, 23, 8, 8],
    "RT-2-PaLM-E-12B": [10, 4, 3, 7, 14, 11, 14, 5, 8],
}
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
xs = list(range(len(SUB)))
w = 0.27
styles = {"RT-1": (GREY, -w), "RT-2-PaLI-X-55B": (ACC, 0), "RT-2-PaLM-E-12B": (ORANGE, w)}
for name, (col, off) in styles.items():
    pts, lo, hi = [], [], []
    for i, k in enumerate(K[name]):
        p = 100 * k / N[i]
        l, u = wilson(k, N[i])
        pts.append(p)
        lo.append(p - l)
        hi.append(u - p)
    ax.bar([x + off for x in xs], pts, width=w, color=col, label=name.replace("RT-2-", "RT-2 "),
           yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8, ecolor=INK))
ax.set_xticks(xs)
ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(SUB, N)], fontsize=8)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 110)
ax.set_title("표 5 하위 집단별 성공률 (Wilson 95% 구간, 시행 수 = 지시문 × 5)", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "출처: arXiv:2307.15818 v1 부록 H.2 표 5, 정수 횟수는 재계산(analysis/recompute.py V3). 구간은 우리가 계산.",
         fontsize=7, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_emergent.png")
plt.close(fig)

# ------------------------------------------------ 그림 2: 표 4 미관측 여섯 셀 (셀 크기 미보고)
CELLS = ["물체\n쉬움", "물체\n어려움", "배경\n쉬움", "배경\n어려움", "환경\n쉬움", "환경\n어려움"]
T4 = {
    "RT-1": [31, 43, 71, 9, 26, 14],
    "MOO": [58, 48, 38, 41, 19, 3],
    "RT-2-PaLI-X-55B": [70, 62, 96, 48, 63, 35],
    "RT-2-PaLM-E-12B": [84, 76, 75, 71, 36, 33],
}
fig, ax = plt.subplots(figsize=(10, 4.4), dpi=160)
xs = list(range(len(CELLS)))
w = 0.2
cols = {"RT-1": GREY, "MOO": "#b9b6ad", "RT-2-PaLI-X-55B": ACC, "RT-2-PaLM-E-12B": ORANGE}
for j, (name, vals) in enumerate(T4.items()):
    ax.bar([x + (j - 1.5) * w for x in xs], vals, width=w, color=cols[name],
           label=name.replace("RT-2-", "RT-2 "))
ax.set_xticks(xs)
ax.set_xticklabels(CELLS, fontsize=8.5)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 110)
ax.set_title("표 4 미관측 셀 성공률 (셀 시행 수는 논문에 없음, 구간 미표시)", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "출처: arXiv:2307.15818 v1 부록 H.1 표 4 (CoRL 2023 판본 표 5와 같음). 막대는 보고값 그대로.",
         fontsize=7, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_cells.png")
plt.close(fig)

# ------------------------------------------------ 그림 3: 공동 미세조정 이득 (표 6, 셀별 차이)
CELLS6 = ["물체 쉬움", "물체 어려움", "배경 쉬움", "배경 어려움", "환경 쉬움", "환경 어려움"]
FT = {"5B": [24, 38, 79, 50, 36, 23], "55B": [60, 62, 75, 38, 57, 19]}
CFT = {"5B": [60, 38, 67, 29, 44, 24], "55B": [70, 62, 96, 48, 63, 35]}
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
xs = list(range(len(CELLS6)))
w = 0.36
for j, (size, col) in enumerate((("5B", GREY), ("55B", ACC))):
    d = [b - a for a, b in zip(FT[size], CFT[size])]
    ax.bar([x + (j - 0.5) * w for x in xs], d, width=w, color=col, label=f"PaLI-X {size}")
    for x, v in zip(xs, d):
        ax.text(x + (j - 0.5) * w, v + (1 if v >= 0 else -4), f"{v:+d}", ha="center", fontsize=7, color=INK)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(CELLS6, fontsize=8.5)
ax.set_ylabel("공동 미세조정 - 미세조정 (%p)")
ax.set_ylim(-30, 45)
ax.set_title("표 6: 공동 미세조정의 셀별 이득 (5B 평균 +2.0, 55B 평균 +10.5)", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "출처: arXiv:2307.15818 v1 부록 H.3 표 6 (CoRL 2023 판본 표 7과 같음). 차이는 우리가 계산.",
         fontsize=7, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_cofinetune.png")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
