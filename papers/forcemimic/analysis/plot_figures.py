"""우리가 만든 그래프 두 장 (matplotlib). 실행: python3 analysis/plot_figures.py
출력: report/figures/own_table1_ci.png, report/figures/own_power.png
입력: 표 I의 보고 값(시행 수와 백분율)과 검정력 계산(recompute.py). 논문 그림은 쓰지 않는다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
INK, ACCENT, GREY = "#1b1b1a", "#1f5fae", "#8a8880"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ------------------------------------------------------------ 그래프 1: 표 I 성공률과 Wilson 구간
methods = ["Raw DP", "Force DP", "Force+Hybrid DP", "HybridIL"]
ns = [20, 10, 10, 20]
motion_k = [16, 6, 8, 20]
len10_k = [11, 1, 2, 17]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, ks, title in zip(axes, [motion_k, len10_k],
                         ["동작 올바름", "연속 10 cm 이상"]):
    ys = np.arange(len(methods))[::-1]
    for y, k, n, m in zip(ys, ks, ns, methods):
        p = k / n
        lo, hi = wilson(k, n)
        color = ACCENT if m == "HybridIL" else GREY
        ax.errorbar(p * 100, y, xerr=[[(p - lo) * 100], [(hi - p) * 100]],
                    fmt="o", color=color, ecolor=color, capsize=4, ms=7)
        ax.text(106, y, f"{k}/{n}", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(methods)
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("성공률 (%) · 95% Wilson 구간")
    ax.set_title(title, fontsize=11, color=INK)
    ax.grid(axis="x", color="#e5e3dc", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("표 I 재계산: 구간이 많이 겹침 (논문 v3 보고 값)", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_table1_ci.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------ 그래프 2: 검정력 곡선
def power_fisher(p1, p2, n1, n2, alpha=0.05):
    pw = 0.0
    for a in range(n1 + 1):
        pa = stats.binom.pmf(a, n1, p1)
        for b in range(n2 + 1):
            if stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1] < alpha:
                pw += pa * stats.binom.pmf(b, n2, p2)
    return pw


grid = list(range(10, 61, 2))
pw_10cm = [power_fisher(0.85, 0.55, n, n) for n in grid]
pw_motion = [power_fisher(1.00, 0.80, n, n) for n in grid]

fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.plot(grid, pw_10cm, color=ACCENT, lw=2, label="연속 10 cm 이상 (85% 대 55%)")
ax.plot(grid, pw_motion, color=GREY, lw=2, ls="--", label="동작 올바름 (100% 대 80%)")
ax.axhline(0.8, color=INK, lw=0.8, ls=":")
ax.axvline(20, color=GREY, lw=0.8)
ax.text(20.8, 0.05, "논문 설계 n = 20", fontsize=9, color=GREY)
ax.set_xlabel("기준선과 HybridIL의 쌍당 시행 수 n")
ax.set_ylabel("검정력 (α = 0.05, 두 쪽 Fisher)")
ax.set_ylim(0, 1)
ax.set_title("표 I 비교의 검정력: 현재 설계는 절반 정도", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_power.png", dpi=160)
plt.close(fig)

print("wrote", FIG / "own_table1_ci.png", FIG / "own_power.png")
