"""우리가 그린 그래프 세 장 (논문 그림 미사용). 입력: recompute.py와 같은 표 값.

출력: report/figures/own_calvin_k.png, own_realrobot.png, own_table7_scatter.png
실행: python3 analysis/plot_figures.py   (matplotlib, 표준 라이브러리 외 numpy 불필요)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "report", "figures")
os.makedirs(OUT, exist_ok=True)

for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

import sys  # noqa: E402
sys.path.insert(0, HERE)
from recompute import T1, T2  # noqa: E402

T7_ROWS = [
("rotate blue block right",94.9,71.2,51.6),("move slider right",99.3,99.6,86.9),("lift red block slider",98.5,91.7,50.6),
("turn off led",100,100,95.6),("push into drawer",82.9,79.8,74.7),("lift blue block drawer",100,93.8,80.0),
("lift pink block slider",97.8,93.8,56.4),("place in slider",91.3,89.1,34.8),("open drawer",99.4,100,94.2),
("rotate red block right",98.6,81.9,45.3),("lift red block table",97.7,76.7,36.5),("lift pink block table",94.1,72.0,73.8),
("move slider left",99.2,99.5,90.7),("turn on lightbulb",99.4,98.6,79.8),("rotate blue block left",97.1,71.2,66.7),
("push blue block left",84.1,72.7,56.1),("close drawer",99.5,98.8,91.3),("turn off lightbulb",99.3,100,79.6),
("turn on led",100,98.7,95.6),("stack block",80.1,45.7,43.2),("push pink block right",61.8,60.3,50.0),
("push red block left",82.3,77.8,54.7),("lift blue block table",97.1,66.2,64.9),("place in drawer",98.9,98.6,95.0),
("rotate red block left",95.3,70.5,61.5),("push pink block left",89.6,84.5,73.8),("lift blue block slider",97.0,90.1,54.7),
("push red block right",54.2,49.3,43.6),("lift pink block drawer",100,81.8,50.0),("rotate pink block right",91.5,79.1,51.7),
("unstack block",100,84.4,100.0),("rotate pink block left",96.4,70.4,66.0),("push blue block right",53.6,50.0,33.9),
("lift red block drawer",100,100,87.5)]

PAL = {"GR-1": "#1f5fae", "HULC": "#9a9a95", "RT-1": "#c98a3b", "MT-R3M": "#7fa6d6", "MCIL": "#bdbdb8"}


def fig_calvin_k():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), sharey=True)
    for ax, split in zip(axes, ["ABCD→D", "ABC→D", "10% data"]):
        for m, (sr, _avg) in T1[split].items():
            ax.plot(range(1, 6), sr, marker="o", lw=2.2 if m == "GR-1" else 1.4,
                    color=PAL.get(m, "#999"), label=m)
        ax.set_title(split, fontsize=11)
        ax.set_xlabel("연속 성공 수 k")
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("P(연속 k개 성공)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("CALVIN 표 1 값 그대로 그린 성공률 곡선 (1000 chain, 우리가 그림)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5, frameon=False)
    fig.savefig(os.path.join(OUT, "own_calvin_k.png"), dpi=170)
    plt.close(fig)


def fig_realrobot():
    cats = ["Seen Objects", "Unseen Instances", "Unseen Categories", "Articulated"]
    keys = ["seen", "unseen_inst", "unseen_cat", "artic"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    width = 0.26
    for i, m in enumerate(["RT-1", "MT-R3M", "GR-1"]):
        vals = [T2[m][k] for k in keys]
        xs = [j + (i - 1) * width for j in range(len(cats))]
        ax.bar(xs, vals, width=width, color=PAL[m], label=m)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("성공률 (논문 표 2)")
    ax.set_title("실로봇 표 2 값 (시도 횟수 미보고: 구간 추정 없음)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "own_realrobot.png"), dpi=170)
    plt.close(fig)


def fig_table7():
    gr1 = [r[1] for r in T7_ROWS]
    wo = [r[2] for r in T7_ROWS]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    colors = ["#c98a3b" if "block" in name else "#9a9a95" for name, *_ in T7_ROWS]
    ax.scatter(wo, gr1, c=colors, s=28, edgecolor="white", lw=0.5)
    ax.plot([0, 105], [0, 105], color="#333", lw=0.8, ls="--")
    ax.set_xlim(30, 103)
    ax.set_ylim(40, 103)
    ax.set_xlabel("GR-1 w/o Video Prediction & Pre-training (%)")
    ax.set_ylabel("GR-1 (%)")
    ax.set_title("표 7 작업별 성공률 (34행, 주황 = block 작업)", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "own_table7_scatter.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_calvin_k()
    fig_realrobot()
    fig_table7()
    print("wrote figures to", os.path.abspath(OUT))
