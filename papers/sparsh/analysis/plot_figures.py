"""논문 보고 수치로 그린 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_t13_gains.png  : 표 13 '최고 SSL 대 E2E' 여섯 값과 평균·중앙값 (평균이 두 과제에 좌우됨)
own_ci_check.png   : 표 8·9의 보고 95% CI 반폭과 같은 n의 이항 근사 반폭
own_slip_budget.png: 표 7 슬립 F1, 라벨 예산별 E2E 대 Sparsh 계열
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#c0662b"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)


# 1. 표 13 여섯 값
labels = ["힘 (DIGIT)", "힘 (GelSight)", "슬립", "자세", "파지", "bead maze"]
gains = [28.31, 59.74, 242.70, 235.89, 5.14, 19.72]
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
colors = [WARM if g > 100 else ACC for g in gains]
ax.bar(labels, gains, color=colors, width=0.6)
for i, g in enumerate(gains):
    ax.text(i, g + 5, f"{g:.1f}%", ha="center", color=INK, fontsize=9)
mean, med = np.mean(gains), np.median(gains)
ax.axhline(mean, color=INK, lw=1, ls="--")
ax.axhline(med, color=GREY, lw=1, ls=":")
ax.text(5.45, mean + 4, f"평균 {mean:.1f}%", ha="right", fontsize=9, color=INK)
ax.text(5.45, med + 4, f"중앙값 {med:.1f}%", ha="right", fontsize=9, color=GREY)
ax.set_ylabel("E2E 대비 개선 (%)")
ax.set_title("표 13 '최고 SSL 대 E2E': 평균은 슬립·자세에 좌우됨", fontsize=11, color=INK, loc="left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_t13_gains.png")
plt.close(fig)


# 2. 보고 CI 반폭 대 이항 근사
def binom_half(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n)


rows = [
    ("표 8 E2E (n=630)", 0.812, 630, 0.001),
    ("표 8 DINO (n=630)", 0.913, 630, 0.001),
    ("표 8 DINOv2 (n=630)", 0.665, 630, 0.0075),
    ("표 9 E2E (n=1.3k)", 0.784, 1300, 0.001),
    ("표 9 I-JEPA (n=1.3k)", 0.802, 1300, 0.002),
]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
y = np.arange(len(rows))
rep = [r[3] * 100 for r in rows]
ana = [binom_half(r[1], r[2]) * 100 for r in rows]
ax.barh(y + 0.18, ana, height=0.34, color=GREY, label="이항 근사 반폭 (같은 n)")
ax.barh(y - 0.18, rep, height=0.34, color=WARM, label="논문 보고 CI 반폭")
for i in range(len(rows)):
    ax.text(ana[i] + 0.3, y[i] + 0.18, f"{ana[i]:.2f}%p", va="center", fontsize=8, color=INK)
    ax.text(rep[i] + 0.3, y[i] - 0.18, f"{rep[i]:.2f}%p", va="center", fontsize=8, color=INK)
ax.set_yticks(y, [r[0] for r in rows])
ax.invert_yaxis()
ax.set_xlabel("95% CI 반폭 (%p)")
ax.set_title("보고 CI는 같은 시험 수의 이항 근사보다 약 5–30배 좁음", fontsize=11, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_ci_check.png")
plt.close(fig)

# 3. 표 7 슬립 F1, 예산별
budgets = ["전체", "1/3", "1/10", "1/100"]
f1 = {
    "E2E": [0.767, 0.238, 0.299, 0.214],
    "Sparsh (DINO)": [0.685, 0.561, 0.548, 0.489],
    "Sparsh (I-JEPA)": [0.776, 0.791, 0.775, 0.726],
    "Sparsh (V-JEPA)": [0.820, 0.828, 0.800, 0.760],
}
styles = {"E2E": (INK, "o"), "Sparsh (DINO)": (GREY, "s"),
          "Sparsh (I-JEPA)": (ACC, "^"), "Sparsh (V-JEPA)": (WARM, "D")}
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
x = np.arange(len(budgets))
for name, vals in f1.items():
    c, m = styles[name]
    ax.plot(x, vals, marker=m, color=c, label=name, lw=1.6)
ax.set_xticks(x, budgets)
ax.set_xlabel("라벨 예산")
ax.set_ylabel("F1 (높을수록 좋음)")
ax.set_ylim(0.1, 0.9)
ax.set_title("표 7 슬립 검출: E2E는 라벨이 줄면 급락", fontsize=11, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_slip_budget.png")
plt.close(fig)
print("saved", sorted(p.name for p in FIG.glob("own_*.png")))
