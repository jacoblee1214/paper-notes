"""TLA 논문 표 수치를 다시 그린 그래프 세 장 (우리가 그린 그래프, 논문 그림 아님).

출력: report/figures/own_gcr_l1.png, own_insertion_ci.png, own_peg_ci.png
실행: python3 analysis/plot_figures.py   (수 초)
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

KO_FONTS = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"]
avail = {f.name for f in font_manager.fontManager.ttflist}
font = next((f for f in KO_FONTS if f in avail), None)
if font:
    plt.rcParams["font.family"] = font
plt.rcParams["axes.unicode_minus"] = False

INK, GREY, BLUE, ORANGE = "#1b1b1a", "#8a8880", "#1f5fae", "#c8702a"
COL = {"BC": GREY, "DP": "#b9b6ac", "SP-TLA": BLUE, "MP-TLA": ORANGE}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e4dc", lw=0.8)
    ax.set_axisbelow(True)


# 1) 표 I (단일 페그)와 표 II (다중 페그, ID) 의 GCR과 L1 x
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
ax = axes[0]
methods = ["BC", "DP", "SP-TLA"]
gcr = [10.4, 8.5, 12.5]
ax.bar(methods, gcr, color=[COL[m] for m in methods])
for i, v in enumerate(gcr):
    lo, hi = wilson(round(v * 20), 2000)  # 표본 2k 가정
    ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=4, lw=1)
    ax.text(i, hi + 0.4, f"{v}", ha="center", fontsize=8.5)
ax.set_title("표 I  GCR (단일 페그, 2k 가정)", fontsize=9.5)
ax.set_ylabel("GCR (%)")
ax.set_ylim(0, 19)
style(ax)

ax = axes[1]
gcr2 = [18.4, 7.4, 18.4]
m2 = ["BC", "DP", "MP-TLA"]
ax.bar(m2, gcr2, color=[COL[m] for m in m2])
for i, v in enumerate(gcr2):
    lo, hi = wilson(round(v * 40), 4000)  # ID 4k 가정
    ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=4, lw=1)
    ax.text(i, hi + 0.4, f"{v}", ha="center", fontsize=8.5)
ax.set_title("표 II  GCR (다중 페그 ID, 4k 가정)", fontsize=9.5)
ax.set_ylim(0, 24)
style(ax)

ax = axes[2]
x_l1 = {"BC": 0.803, "DP": 0.370, "SP-TLA": 0.079}
ax.bar(list(x_l1), list(x_l1.values()), color=[COL[m] for m in x_l1])
for i, v in enumerate(x_l1.values()):
    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
ax.set_title("표 I  L1 x (mm, 낮을수록 좋음)", fontsize=9.5)
ax.set_ylim(0, 0.95)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_gcr_l1.png", dpi=170)
plt.close(fig)

# 2) 표 III: 간격별 성공률과 Wilson 95% 구간 (각 칸 50회 가정)
fig, ax = plt.subplots(figsize=(8.6, 4.2))
groups = ["2.0 mm", "1.6 mm", "1.0 mm", "합계"]
suc = {"BC": [44, 32, 18, 31], "DP": [58, 43, 20, 40], "SP-TLA": [96, 94, 74, 88], "MP-TLA": [94, 86, 90, 90]}
n_of = {"합계": 150}
width = 0.2
for j, meth in enumerate(suc):
    xs = [i + (j - 1.5) * width for i in range(len(groups))]
    vals = suc[meth]
    ns = [50, 50, 50, 150]
    lows, highs = [], []
    for v, n in zip(vals, ns):
        k = round(v * n / 100)
        lo, hi = wilson(k, n)
        lows.append(v - lo)
        highs.append(hi - v)
    ax.bar(xs, vals, width=width * 0.92, color=COL[meth], label=meth)
    ax.errorbar(xs, vals, yerr=[lows, highs], fmt="none", ecolor=INK, capsize=2.5, lw=0.8)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 115)
ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.set_title("표 III  간격별 성공률 (Wilson 95%, 칸당 50회 가정)", fontsize=9.5)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_insertion_ci.png", dpi=170)
plt.close(fig)

# 3) 표 IV: 페그별 성공률 (ID/OOD), 최고 기준선과 MP-TLA
fig, ax = plt.subplots(figsize=(8.6, 4.0))
pegs = ["Square\n(ID)", "Triangle\n(ID)", "Round\n(OOD)", "Hexagon\n(OOD)"]
t4 = {"BC": [60, 84, 62, 66], "DP": [60, 60, 56, 54], "SP-TLA": [96, 76, 92, 90], "MP-TLA": [94, 92, 94, 96]}
width = 0.2
for j, meth in enumerate(t4):
    xs = [i + (j - 1.5) * width for i in range(4)]
    vals = t4[meth]
    lows, highs = [], []
    for v in vals:
        lo, hi = wilson(round(v * 0.5), 50)
        lows.append(v - lo)
        highs.append(hi - v)
    ax.bar(xs, vals, width=width * 0.92, color=COL[meth], label=meth)
    ax.errorbar(xs, vals, yerr=[lows, highs], fmt="none", ecolor=INK, capsize=2.5, lw=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels(pegs)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 125)
ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.set_title("표 IV  페그 종류별 성공률 (Wilson 95%, 칸당 50회 가정)", fontsize=9.5)
style(ax)
fig.tight_layout()
fig.savefig(FIG / "own_peg_ci.png", dpi=170)
plt.close(fig)
print("wrote", [str(FIG / n) for n in ("own_gcr_l1.png", "own_insertion_ci.png", "own_peg_ci.png")])
