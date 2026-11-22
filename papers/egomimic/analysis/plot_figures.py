"""우리가 그린 그래프 두 장. 입력은 recompute.py의 값(논문 표 수치)이다.

실행: python3 analysis/plot_figures.py → report/figures/own_sr_ci.png, own_score.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from recompute import SR, PTS, wilson

FIG = Path(__file__).resolve().parents[1] / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
METHODS = ["ACT", "MimicPlay", "EgoMimic w/o human", "EgoMimic"]
COLORS = ["#9aa3ad", "#c9a86a", "#6b9bd1", "#1f5fae"]


def sr_plot():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, task in zip(axes, ["Laundry (n=40)", "Groceries SR (n=50)"]):
        d = SR[task]
        xs, ys, lo, hi = [], [], [], []
        for m in METHODS:
            lo_, hi_ = wilson(d[m], d["n"])
            xs.append(m.replace(" w/o human", "\n(w/o human)"))
            ys.append(d[m] / d["n"] * 100)
            lo.append(ys[-1] - lo_ * 100)
            hi.append(hi_ * 100 - ys[-1])
        ax.bar(range(4), ys, color=COLORS, yerr=[lo, hi], capsize=4)
        ax.set_xticks(range(4))
        ax.set_xticklabels(xs, fontsize=8)
        ax.set_title(task, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("성공률 (%)")
    fig.suptitle("Table III 성공률과 Wilson 95% 구간 (우리가 계산, 논문 값 재구성)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_sr_ci.png", dpi=170)
    plt.close(fig)


def score_plot():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    tasks = list(PTS.keys())
    width = 0.2
    for i, m in enumerate(METHODS):
        vals = [PTS[t][m] for t in tasks]
        ax.bar([j + (i - 1.5) * width for j in range(len(tasks))], vals, width, color=COLORS[i], label=m.replace(" w/o human", " (w/o human)"))
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.replace(" (Table III)", "") for t in tasks], fontsize=9)
    ax.set_ylabel("점수 (pts)")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Table III 점수 (단일 실행, 구간 없음)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_score.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    sr_plot()
    score_plot()
    print("wrote", FIG)
