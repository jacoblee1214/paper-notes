"""재계산 결과를 그래프 세 장으로 그린다 (우리가 그린 그래프, 논문 그림 아님).

출력: report/figures/own_real_ci.png, own_sim_cost.png, own_sparc.png
실행: python3 analysis/plot_figures.py   (matplotlib, numpy, scipy)
"""
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서
for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
COL = {"EgoEngine": "#1f5fae", "Real Robot": "#55544f", "Phantom": "#b5892b", "Human Video": "#9c9a92"}

# 1. 실로봇 과제 성공률 + Wilson 95% (N=20 추정)
tasks = ["Mustard", "Drawer", "Flower", "Hammer"]
methods = ["EgoEngine", "Real Robot", "Phantom", "Human Video"]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
w = 0.2
for j, m in enumerate(methods):
    xs, ys, lo, hi = [], [], [], []
    for i, t in enumerate(tasks):
        r = RES["t3_real_per_task"][m][t]
        xs.append(i + (j - 1.5) * w)
        ys.append(r["sr"])
        lo.append(r["sr"] - r["wilson95"][0])
        hi.append(r["wilson95"][1] - r["sr"])
    ax.bar(xs, ys, width=w * 0.92, color=COL[m], label=m, alpha=0.9)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
ax.set_xticks(range(len(tasks)), tasks)
ax.set_ylabel("성공률 (시도 20회 가정)")
ax.set_ylim(0, 1.05)
ax.set_title("실로봇 과제 성공률 (표 3 보고 값, Wilson 95% 구간)", fontsize=10)
ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_real_ci.png")
plt.close(fig)

# 2. 시뮬레이션: 성공률 대 생성 비용 (표 2). 비용은 시뮬 스텝/궤적 스텝, 로그 축
rows = {
    "TACO": {"Mink (Replay)": (0.17, 1.00), "Spider (MPC)": (0.25, 7923),
             "H2S2R (RL)": (0.83, 73675), "EgoEngine": (0.83, 34842)},
    "Aria": {"Mink (Replay)": (0.10, 1.00), "Spider (MPC)": (0.20, 4382),
             "H2S2R (RL)": (0.90, 20237), "EgoEngine": (0.90, 16560)},
}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=200, sharey=True)
for ax, (ds, d) in zip(axes, rows.items()):
    cmap = {"EgoEngine": COL["EgoEngine"], "H2S2R (RL)": COL["Phantom"]}
    for name, (sr, cost) in d.items():
        c = cmap.get(name, COL["Real Robot"])
        ax.scatter(cost, sr, s=46, color=c, zorder=3, label=name)
        if name not in cmap:
            ax.annotate(name, (cost, sr), xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_xscale("log")
    ax.set_title(f"{ds} (표 2)", fontsize=9)
    ax.set_xlabel("Cost: 궤적 스텝당 시뮬 스텝 (로그)")
    ax.set_xlim(0.5, 1e5 if ds == "TACO" else 4e4)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("성공률 (SR)")
axes[0].set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(FIG / "own_sim_cost.png")
plt.close(fig)

# 3. 동작 매끄러움 SPARC (표 E.1): 값이 클수록 매끄럽다
sp = {"Real": [-8.68, -10.40, -3.21, -4.66], "EgoEngine": [-4.88, -7.49, -3.25, -3.88]}
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
w = 0.36
for j, (k, v) in enumerate(sp.items()):
    xs = [i + (j - 0.5) * w for i in range(4)]
    ax.bar(xs, v, width=w * 0.9, color=COL["EgoEngine"] if k == "EgoEngine" else COL["Real Robot"],
           label="EgoEngine" if k == "EgoEngine" else "실로봇 teleop")
ax.set_xticks(range(4), ["Mustard", "Drawer", "Hammer", "Flower"])  # 표 E.1 순서
ax.set_ylabel("SPARC (클수록 매끄러움, 음수)")
ax.set_title("동작 매끄러움 (표 E.1 보고 값)", fontsize=10)
ax.axhline(0, color="#333", lw=0.6)
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "own_sparc.png")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("own_*.png")), "on", platform.system())
