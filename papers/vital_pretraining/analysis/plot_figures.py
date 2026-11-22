"""그림 7 판독값의 Wilson 95% 구간 그래프 (우리가 그린 그래프, 영문 라벨).

실행: python3 analysis/plot_figures.py  → report/figures/own_success_ci.png
입력은 recompute.py의 READ와 같다(20회 시행, 5%p 판독 단위).
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute import READ, N, wilson  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "report" / "figures" / "own_success_ci.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

panels = [
    ("USB plugging", "USB", ["tactile+vision", "vision", "tactile"]),
    ("Rectangle stacking", "RECT", ["tactile+vision", "vision"]),
    ("Cube stacking", "CUBE", ["tactile+vision", "vision"]),
]
labels = {"tactile+vision": "Tactile+Vision", "vision": "Vision only", "tactile": "Tactile only"}

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, (title, task, inputs) in zip(axes, panels):
    x0 = 0
    ticks, tlabels = [], []
    for inp in inputs:
        for j, mdl in enumerate(["ACT", "DP"]):
            for pre in (False, True):
                key = (task, inp, mdl, pre)
                if key not in READ:
                    continue
                k = READ[key] // 5
                lo, hi = wilson(k, N)
                rate = k / N
                x = x0 + j * 0.9 + (0.35 if pre else 0)
                color = ("#8fb4d9" if mdl == "ACT" else "#8fd19e") if not pre else ("#1f5fae" if mdl == "ACT" else "#2e8b57")
                ax.bar(x, rate * 100, width=0.33, color=color, edgecolor="black", linewidth=0.5)
                ax.errorbar(x, rate * 100, yerr=[[rate * 100 - lo * 100], [hi * 100 - rate * 100]],
                            fmt="none", ecolor="black", capsize=2, linewidth=0.8)
        ticks.append(x0 + 0.45)
        tlabels.append(labels[inp])
        x0 += 2.4
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Success rate (%)  [n = 20 per bar]")
fig.text(0.5, -0.02, "Light: not pretrained, dark: pretrained. Blue = ACT, green = Diffusion Policy. "
         "Whiskers: Wilson 95% CI. Values read from paper Fig. 7 (v2).", ha="center", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("wrote", OUT)
