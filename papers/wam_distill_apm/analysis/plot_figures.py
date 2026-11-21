"""recompute_results.json을 읽어 그래프 세 장을 만든다(우리가 그린 그래프).

출력: report/figures/own_sim_ci.png, own_real_ci.png, own_ablation_ci.png
실행: python3 analysis/plot_figures.py   (matplotlib)
한글 폰트는 macOS Apple SD Gothic Neo → Linux Noto CJK 순서로 찾는다.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ("AppleSDGothicNeo", "Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

INK, GREY, BLUE, ORANGE, TEAL = "#1b1b1a", "#8a8880", "#1f5fae", "#d1603d", "#2a8c82"
COLORS = {"LingBot-VA": GREY, "Flash-WAM(DirectWAM 보고 1/1)": "#b5b2a8", "Flash-WAM": "#b5b2a8",
          "Continued": ORANGE, "APM": BLUE}

# 1) 표 1: 조건별 성공률과 Wilson 95% 구간 (로컬 2,500 에피소드)
import numpy as np
T1 = {"1/1": {"LingBot-VA": (57.0, 50.7), "Flash-WAM(DirectWAM 보고 1/1)": (82.1, 74.2), "Continued": (82.3, 76.1), "APM": (84.3, 77.5)},
      "2/2": {"LingBot-VA": (88.3, 86.0), "Flash-WAM": (90.5, 88.6), "Continued": (89.6, 87.7), "APM": (90.4, 87.1)},
      "4/4": {"LingBot-VA": (90.4, 90.0), "Flash-WAM": (92.0, 90.4), "Continued": (90.5, 89.1), "APM": (92.5, 90.3)}}
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
for ax, budget in zip(axes, ["1/1", "2/2", "4/4"]):
    rows = list(T1[budget].items())
    for i, (name, (c, r)) in enumerate(rows):
        for j, (val, lab) in enumerate([(c, "Clean"), (r, "Random")]):
            cell = RES["table1"][budget]["cells"].get(name)
            if cell is None:
                continue
            lo, hi = (cell["clean_ci95"] if lab == "Clean" else cell["random_ci95"])
            y = i + (0.15 if j == 0 else -0.15)
            ax.errorbar(val, y, xerr=[[val - lo], [hi - val]], fmt="o" if j == 0 else "s",
                        color=COLORS.get(name, INK), ms=4, lw=1.2, capsize=2, mfc="white" if j else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n.replace("(DirectWAM 보고 1/1)", "†") for n, _ in rows])
    ax.set_title(f"{budget} (video/action steps)", fontsize=9.5, color=INK)
    ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    ax.set_xlim(45, 97)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_xlabel("success (%)")
axes[1].set_xlabel("success (%)")
axes[2].set_xlabel("success (%)")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=INK, label="Clean (채움)"),
     Line2D([], [], marker="s", ls="", color=INK, mfc="white", label="Random (빈 원)")]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, 0.965, "표 1 성공률과 Wilson 95% 구간 (조건당 2,500 에피소드, 독립 가정)", ha="center", fontsize=9.5, color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 0.91))
fig.savefig(FIG / "own_sim_ci.png", dpi=180)
plt.close(fig)

# 2) 표 3: 실로봇 성공률과 Wilson 95% 구간 (과제당 24 시도)
real = RES["real_robot"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
order = ["Teacher (8/8)", "Coarse teacher (1/1)", "APM (1/1)"]
for ax, task in zip(axes, ["PenHolder", "FoldTowel"]):
    for i, name in enumerate(order):
        k = real[task][name]["successes"]
        lo, hi = real[task][name]["wilson95_pct"]
        val = 100 * k / 24
        ax.barh(i, val, color=BLUE if "APM" in name else (GREY if "Teacher (8" in name else ORANGE), height=0.55, alpha=0.85)
        ax.errorbar(val, i, xerr=[[val - lo], [hi - val]], color=INK, capsize=3, lw=1)
        ax.text(min(hi, 97) + 1.5, i, f"{k}/24", va="center", fontsize=8.5, color=INK)
    ax.set_title(task, fontsize=9.5, color=INK)
    ax.set_xlim(0, 110)
    ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks(range(3))
axes[0].set_yticklabels(["교사 8단계", "교사 1단계(거친 격자)", "APM 1단계"])
axes[0].invert_yaxis()
axes[0].set_xlabel("success (%)")
axes[1].set_xlabel("success (%)")
fig.text(0.5, 0.965, "표 3 실로봇 성공률과 Wilson 95% 구간 (과제당 24회, Astribot S1)", ha="center", fontsize=9.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(FIG / "own_real_ci.png", dpi=180)
plt.close(fig)

# 3) 표 2: 절제 쌍 비교의 평균 차와 근사 95% 구간(독립 가정)
pairs = [k for k in RES["table2_pairs"]]
labels_short = {
    pairs[0]: "앵커: Restart - Original (F 없음)",
    pairs[1]: "F 추가: Restart+F - Restart (투영 없음)",
    pairs[2]: "투영: APM - Restart+F (투영 없음)",
    pairs[3]: "APM - Continued",
    pairs[4]: "APM - Shared",
    pairs[5]: "Continued - Shared",
    pairs[6]: "Original+F+Proj - Original (F 없음)",
}
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ypos = []
yl = []
for i, key in enumerate(pairs):
    for j, bud in enumerate(["1/1", "2/2", "4/4"]):
        x = RES["table2_pairs"][key][bud]
        y = i * 4 + j
        d = x["avg_diff_pp"]
        lo, hi = x["ci95_pp_approx"]
        col = BLUE if j == 0 else (TEAL if j == 1 else ORANGE)
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ms=4, capsize=2, lw=1.1)
        ypos.append(y)
        yl.append(f"{labels_short[key] if j == 1 else ''}  {bud}")
ax.axvline(0, color=GREY, lw=0.8, ls="--")
ax.set_yticks(ypos)
ax.set_yticklabels(yl, fontsize=7.6)
ax.invert_yaxis()
ax.set_xlabel("평균 성공률 차 (percentage point), 95% 근사 구간")
ax.set_title("표 2 절제 비교 (독립 가정, 조건당 2,500 에피소드; 쌍체 아님)", fontsize=9.5, color=INK)
ax.grid(axis="x", color="#e5e3dc", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_ablation_ci.png", dpi=180)
plt.close(fig)
print("wrote figures to", FIG)
