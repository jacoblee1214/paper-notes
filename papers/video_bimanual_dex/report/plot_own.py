"""직접 그린 그래프 (논문 그림을 쓰지 않는다). 실행: python3 report/plot_own.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
for cand in ["Apple SD Gothic Neo", "Noto Sans CJK KR"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h) * 100, (c + h) * 100

res = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
tasks = ["Pour", "Sweep", "Unscrew", "Clean"]
# 표 I: PriDex 와 w/o HP (150 시도, 논문 수치를 분수로 환산한 값)
groups = {"PriDex": [98.0, 44.7, 40.0, 46.0], "w/o HP": [41.3, 64.7, 16.7, 99.3], "w/o HP & Conf.": [16.0, 50.7, 6.7, 98.7]}
fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=200)
width = 0.26
colors = ["#1f5fae", "#8fb0d9", "#c9c6bd"]
for gi, (name, vals) in enumerate(groups.items()):
    xs = [i + (gi - 1) * width for i in range(4)]
    ks = [round(v / 100 * 150) for v in vals]
    ci = [wilson(k, 150) for k in ks]
    ax.bar(xs, vals, width=width, color=colors[gi], label=name, edgecolor="none")
    ax.errorbar(xs, vals, yerr=[[v - lo for v, (lo, _) in zip(vals, ci)], [hi - v for v, (_, hi) in zip(vals, ci)]],
                fmt="none", ecolor="#333", capsize=3, lw=0.9)
ax.set_xticks(range(4), tasks)
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 110)
ax.set_title("표 I 시뮬레이션 RL 성공률 (태스크당 150회, Wilson 95% 구간)", fontsize=11)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_ci.png")
plt.close(fig)

# 실로봇 표 III: 10회 시행, Wilson 95% 구간
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
real = {"Pour": (2, 5), "Sweep": (3, 8), "Clean": (4, 6)}
for i, (t, (a, b)) in enumerate(real.items()):
    for j, (k, lab, col) in enumerate([(a, "Real-only", "#c9c6bd"), (b, "Sim+Real", "#1f5fae")]):
        x = i + (j - 0.5) * 0.3
        lo, hi = wilson(k, 10)
        ax.bar(x, k / 10 * 100, width=0.28, color=col, label=lab if i == 0 else None)
        ax.errorbar(x, k / 10 * 100, yerr=[[k / 10 * 100 - lo], [hi - k / 10 * 100]], fmt="none", ecolor="#333", capsize=3, lw=0.9)
ax.set_xticks(range(3), list(real))
ax.set_ylabel("성공률 (%)")
ax.set_ylim(0, 110)
ax.set_title("표 III 실로봇 결과 (태스크당 10회, Wilson 95% 구간)", fontsize=11)
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_real_ci.png")
plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
