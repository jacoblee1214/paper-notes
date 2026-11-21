"""우리가 만든 그래프 두 장을 report/figures/에 쓴다 (matplotlib). 실행: python3 analysis/plot_figures.py
own_libero_ci.png : LIBERO 네 suite, OpenVLA 대 UniVLA(full), 공식 README의 3 seed 표준편차로 만든 t 구간(df=2)
own_real_ci.png   : 실로봇 과제별 성공률과 Wilson 95% 구간(시행 15회 가정, 분수 역산)
"""
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.rcParams["font.family"] = ["Apple SD Gothic Neo", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
FIG = Path(__file__).resolve().parents[1] / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, BLUE, GREY = "#1b1b1a", "#1f5fae", "#8a8880"

# ---- 1. LIBERO ----
suites = ["Spatial", "Object", "Goal", "Long"]
openvla = [(84.7, 0.9), (88.4, 0.8), (79.2, 1.0), (53.7, 1.3)]
univla = [(96.5, 0.5), (96.8, 0.5), (95.6, 0.4), (92.0, 1.0)]
tc = stats.t.ppf(0.975, 2)
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
x = np.arange(len(suites))
for off, data, col, lab in [(-0.18, openvla, GREY, "OpenVLA"), (0.18, univla, BLUE, "UniVLA (full)")]:
    m = [d[0] for d in data]
    h = [tc * d[1] / math.sqrt(3) for d in data]
    ax.bar(x + off, m, width=0.34, color=col, label=lab, yerr=h, capsize=3, error_kw={"lw": 0.9})
    for xi, mi in zip(x + off, m):
        ax.text(xi, mi + 6.5, f"{mi:.1f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(x, [f"LIBERO-{s}" for s in suites], fontsize=8)
ax.set_ylim(0, 115)
ax.set_ylabel("성공률 (%)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
ax.set_title("LIBERO 네 suite: 3 seed 표준편차 기반 95% 구간 (df=2)", fontsize=9, color=INK)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "값: 공식 README 표 (논문 표 I과 같은 평균). 구간은 README의 ± 값을 3 seed 표준편차로 가정해 계산. 우리가 그린 그래프.",
         fontsize=5.6, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_libero_ci.png")
plt.close(fig)

# ---- 2. 실로봇 ----
tasks = ["Store\nscrew", "Clean\nboard", "Fold\ntowel", "Hanoi"]
real = {"Diffusion Policy": [6, 5, 8, 1], "OpenVLA": [6, 8, 5, 4], "LAPA (OXE)": [9, 6, 5, 7],
        "UniVLA (full)": [14, 15, 7, 13]}
n = 15


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
cols = [GREY, "#b9b6ad", "#6f9bd1", BLUE]
w = 0.2
for i, (name, ks) in enumerate(real.items()):
    pts = [100 * k / n for k in ks]
    lo = [100 * (k / n - wilson(k, n)[0]) for k in ks]
    hi = [100 * (wilson(k, n)[1] - k / n) for k in ks]
    ax.bar(np.arange(4) + (i - 1.5) * w, pts, width=w, color=cols[i], label=name)
    ax.errorbar(np.arange(4) + (i - 1.5) * w, pts, yerr=[lo, hi], fmt="none", ecolor=INK, lw=0.7, capsize=2)
ax.set_xticks(np.arange(4), tasks, fontsize=7)
ax.set_ylabel("성공률 (%)", fontsize=8)
ax.set_ylim(0, 115)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, frameon=False, ncol=4, loc="upper center")
ax.set_title("실로봇 과제별 성공률과 Wilson 95% 구간 (과제당 15회 가정)", fontsize=9, color=INK)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "값: 논문 표 A-V 백분율을 15회 분수로 역산한 값(논문에 시행 수 미보고). 우리가 그린 그래프.",
         fontsize=5.6, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "own_real_ci.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
