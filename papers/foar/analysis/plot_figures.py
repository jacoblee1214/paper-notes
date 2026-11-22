"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_asr_ci.png     : 표 I 과제 성공률(20회)과 Wilson 95% 구간, 세 과제 x 일곱 방법
own_chop_ci.png    : 표 II 절단 결과 (개수 평균 ± SD, 배치 성공률과 Wilson 95% 구간)
own_robust_ci.png  : 표 IV 외란 네 조건의 과제 성공률과 Wilson 95% 구간 (RISE, RISE force-token, FoAR)
입력은 analysis/recompute.py의 값과 같다(표에서 옮긴 개수). 실행: python3 analysis/plot_figures.py
"""
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
INK, ACC, GREY, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#b5562b"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8.5)
    ax.grid(axis="x", color="#e6e4dd", lw=0.6)


# ---------- 표 I: 과제 성공률 (20회) ----------
methods = ["ACT", "Diffusion Policy", "RISE", "RISE (force-token)", "RISE (force-concat)", "FoAR (3D-cls)", "FoAR"]
task_counts = {
    "Wiping": [10, 12, 15, 16, 13, 7, 20],
    "Wiping (General)": [10, 10, 16, 16, 19, 8, 20],
    "Peeling": [5, 14, 10, 15, 16, 8, 20],
}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
for ax, (task, ks) in zip(axes, task_counts.items()):
    ys = np.arange(len(methods))[::-1]
    for y, m, k in zip(ys, methods, ks):
        lo, hi = wilson(k, 20)
        color = ACC if m == "FoAR" else (WARM if m == "RISE" else GREY)
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round")
        ax.plot(100 * k / 20, y, "o", color=color, ms=5.5)
        ax.text(102, y, f"{k}/20", va="center", fontsize=7.6, color=INK)
    ax.set_title(task, fontsize=10.5, color=INK, loc="left")
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    style(ax)
axes[0].set_yticks(np.arange(len(methods))[::-1])
axes[0].set_yticklabels(methods)
fig.supxlabel("과제 성공률 (%), 점과 Wilson 95% 구간 (시행 20회)", fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_asr_ci.png", dpi=200)
plt.close(fig)

# ---------- 표 II: 절단 (10회) ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
labels = ["RISE", "FoAR", "시연(오라클)"]
means, sds = [1.8, 3.9, 5.0], [0.6, 0.9, 0.0]
cols = [WARM, ACC, GREY]
for i, (m, s, c) in enumerate(zip(means, sds, cols)):
    a.bar(i, m, color=c, width=0.55, alpha=0.9)
    a.errorbar(i, m, yerr=s, color=INK, capsize=4, lw=1)
a.set_xticks(range(3))
a.set_xticklabels(labels, fontsize=8.5)
a.set_ylabel("절단 개수 (평균 ± SD)", fontsize=9, color=INK)
a.set_title("절단 개수 (10회)", fontsize=10, color=INK, loc="left")
style(a)
a.grid(axis="x", visible=False)
a.grid(axis="y", color="#e6e4dd", lw=0.6)
rates = {"RISE": (3, 10), "FoAR": (7, 10)}
for i, name in enumerate(["RISE", "FoAR"]):
    k, n = rates[name]
    lo, hi = wilson(k, n)
    x = i
    b.plot([lo, hi], [x, x], color=WARM if name == "RISE" else ACC, lw=2.2, solid_capstyle="round")
    b.plot(100 * k / n, x, "o", color=WARM if name == "RISE" else ACC, ms=6)
    b.text(102, x, f"{k}/{n}", va="center", fontsize=8, color=INK)
b.set_yticks([0, 1])
b.set_yticklabels(["RISE", "FoAR"], fontsize=8.5)
b.set_xlim(0, 118)
b.set_xticks([0, 25, 50, 75, 100])
b.set_title("배치 성공률 (%), Wilson 95%", fontsize=10, color=INK, loc="left")
style(b)
fig.tight_layout()
fig.savefig(FIG / "own_chop_ci.png", dpi=200)
plt.close(fig)

# ---------- 표 IV: 외란 4조건, 과제(wipe) 성공률 ----------
scen = ["Original", "Rewrite", "Move", "Rewrite+Move"]
wipe = {
    "RISE": [16, 14, 20, 14],
    "RISE (force-token)": [16, 18, 16, 20],
    "FoAR": [20, 20, 20, 20],
}
fig, ax = plt.subplots(figsize=(7.4, 3.4))
offs = {"RISE": -0.22, "RISE (force-token)": 0.0, "FoAR": 0.22}
colors = {"RISE": WARM, "RISE (force-token)": GREY, "FoAR": ACC}
for name, ks in wipe.items():
    for i, k in enumerate(ks):
        lo, hi = wilson(k, 20)
        x = i + offs[name]
        ax.plot([x, x], [lo, hi], color=colors[name], lw=2)
        ax.plot(x, 100 * k / 20, "o", color=colors[name], ms=5.5, label=name if i == 0 else None)
ax.set_xticks(range(4))
ax.set_xticklabels(scen, fontsize=8.5)
ax.set_ylabel("Wiping (General) 과제 성공률 (%)", fontsize=9, color=INK)
ax.set_ylim(0, 108)
ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(colors=INK, labelsize=8.5)
ax.grid(axis="y", color="#e6e4dd", lw=0.6)
ax.set_title("표 IV 외란 실험: 과제 성공률과 Wilson 95% 구간 (시행 20회 추정)", fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(FIG / "own_robust_ci.png", dpi=200)
plt.close(fig)
print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])
