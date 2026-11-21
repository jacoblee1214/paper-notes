"""우리가 만든 그래프 두 장 (논문 그림을 다시 그린 것이 아님).

1) own_fig7_object_ci.png : Fig. 7 판독값과 π0의 객체 단위 Wilson 95% 구간
2) own_fig11_dips.png      : Fig. 11 판독값에서 π0의 5시간 하락과 Paper towel 추세

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return c - h, c + h


# 1) Fig. 7 판독값. 막대는 판독값, 오차 막대는 π0의 객체 단위 Wilson 95%(상한 검정 기준의 폭)
tasks = ["Shirt folding", "Bussing easy", "Bussing hard", "Grocery", "Toast"]
pts = [1, 7, 12, 7, 4]
pi0 = [1.00, 0.97, 0.875, 0.785, 0.75]
parity = [0.90, 0.81, 0.50, 0.345, 0.40]
small = [0.50, 0.45, 0.335, 0.275, None]
fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
x = range(len(tasks))
w = 0.26
ax.bar([i - w for i in x], pi0, w, color=ACC, label="π0")
ax.bar(x, parity, w, color="white", edgecolor=ACC, label="π0 (parity)")
ax.bar([i + w for i in x], [s or 0 for s in small], w, color="#7fc4e8", label="π0-small")
for i in x:
    n = pts[i] * 10
    k = round(pi0[i] * n)
    lo, hi = wilson(k, n)
    ax.errorbar(i - w, pi0[i], yerr=[[pi0[i] - lo], [hi - pi0[i]]], color=INK, capsize=3, lw=1)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{t}\n(n={p*10})" for t, p in zip(tasks, pts)], fontsize=8)
ax.set_ylabel("Average task progress (판독)", fontsize=9)
ax.set_ylim(0, 1.12)
ax.grid(axis="y", color="#dcdad3", lw=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
ax.set_title("Fig. 7 판독 (out-of-box): π0 막대 위 검은 선은 객체 단위 Wilson 95%", fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(OUT / "own_fig7_object_ci.png")
plt.close(fig)

# 2) Fig. 11 판독값: Tupperware 와 Paper towel
hours = [1, 5, 10]
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=200, sharey=True)
panels = [
    ("Tupperware in microwave", {"π0": [0.87, 0.57, 0.76], "π0 (scratch)": [0.67, 0.75, 0.77],
                                 "DP": [0.65, 0.62, 0.67], "ACT": [0.53, 0.63, 0.73]}),
    ("Paper towel replacement", {"π0": [0.82, 0.80, 0.70], "π0 (scratch)": [0.37, 0.57, 0.57],
                                 "DP": [0.23, 0.33, 0.30]}),
]
styles = {"π0": (ACC, "-", "o"), "π0 (scratch)": (ACC, "--", "o"), "DP": ("#d9552b", "-", "o"),
          "ACT": ("#e6a800", "-", "o")}
for ax, (title, series) in zip(axes, panels):
    for name, ys in series.items():
        c, ls, m = styles[name]
        ax.plot(hours, ys, color=c, ls=ls, marker=m, ms=4, lw=1.4, label=name)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks(hours)
    ax.set_xlabel("Fine-tuning data (h)", fontsize=8)
    ax.grid(axis="y", color="#dcdad3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.0)
axes[0].set_ylabel("Average task progress (판독)", fontsize=8)
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
axes[1].legend(fontsize=7, frameon=False, loc="lower left")
fig.suptitle("Fig. 11 판독: 본문의 '5시간 유사' 주장과 Paper towel 추세", fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(OUT / "own_fig11_dips.png")
plt.close(fig)
print("wrote", OUT / "own_fig7_object_ci.png", OUT / "own_fig11_dips.png")
