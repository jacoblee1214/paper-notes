"""우리가 그린 그래프 두 장. 입력은 논문 표 2(과제별 성공률)와 recompute_results.json이다.

  report/figures/own_per_task.png : 셀별 과제 8개의 기준선 대 ReGround (표 2)
  report/figures/own_recovery.png : 논문 회복률 주장(0.91–1.00)과 두 정의로 다시 계산한 값

실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for f in FONT_CANDIDATES:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

TASKS = ["spatial 0", "spatial 2", "object 1", "object 3", "goal 1", "goal 2", "10-1", "10-3"]
T2 = {
    "지연 250 ms": [(.23, .99), (.54, .95), (.51, .99), (.66, .95), (.74, .95), (.74, .94), (.51, .85), (.91, .94)],
    "지연 400 ms": [(.00, .91), (.00, .83), (.14, .89), (.11, .87), (.40, .93), (.43, .95), (.00, .76), (.66, .99)],
    "복합": [(.03, 1.00), (.00, .98), (.00, .99), (.00, .96), (.37, .95), (.00, .92), (.00, .91), (.00, .94)],
    "회전 90°": [(.00, .99), (.00, 1.00), (.00, 1.00), (.00, .99), (.00, .99), (.00, .95), (.00, .88), (.00, .77)],
    "게인 ×3": [(.09, 1.00), (.03, 1.00), (.14, .97), (.03, .95), (.34, 1.00), (.34, .90), (.00, .90), (.34, .49)],
}
BASE_C, RG_C = "#9a9891", "#1f5fae"


def per_task():
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 4.2), sharey=True)
    for ax, (cell, pairs) in zip(axes, T2.items()):
        y = list(range(len(TASKS)))[::-1]
        for yi, (b, r) in zip(y, pairs):
            ax.plot([b, r], [yi, yi], color="#cfcdc6", lw=2, zorder=1)
        ax.scatter([b for b, _ in pairs], y, color=BASE_C, s=28, zorder=2, label="기준선 (Base)")
        ax.scatter([r for _, r in pairs], y, color=RG_C, s=28, zorder=3, label="ReGround")
        ax.set_title(cell, fontsize=10.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="x", color="#e6e4dd", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(list(range(len(TASKS)))[::-1])
    axes[0].set_yticklabels(TASKS, fontsize=9)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False, bbox_to_anchor=(0, -0.28), ncol=2)
    fig.suptitle("과제별 시험 성공률 (표 2에서 옮김): 기준선 대 ReGround, 과제 8개", fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "own_per_task.png", dpi=200)
    plt.close(fig)


def recovery():
    res = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))["recovery"]
    cells = list(res)
    labels = {"lat250": "지연 250", "lat400": "지연 400", "comp": "복합", "rot90": "회전 90°", "gain3": "게인 ×3"}
    x = list(range(len(cells)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhspan(0.91, 1.00, color="#eef3fa", zorder=0)
    ax.text(1.6, 0.915, "논문 주장 0.91–1.00", fontsize=8.5, color="#1f5fae", va="bottom", ha="left")
    ax.scatter(x, [res[c]["vs_nominal_val"] for c in cells], color=RG_C, s=45, zorder=3,
               label="명목 0.99 기준 (검증 명목 성공률)")
    ax.scatter(x, [res[c]["vs_oracle"] for c in cells], color="#c0622b", marker="D", s=35, zorder=3,
               label="Oracle 기준 (표 1)")
    ax.axhline(1.0, color="#cfcdc6", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in cells], fontsize=9)
    ax.set_ylabel("회복률 (잃은 성공 대비)", fontsize=9)
    ax.set_ylim(0.85, 1.10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("회복률: 논문 주장과 두 정의의 재계산 (π0.5, 표 1)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "own_recovery.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    per_task()
    recovery()
    print("wrote", OUT / "own_per_task.png", "and", OUT / "own_recovery.png")
