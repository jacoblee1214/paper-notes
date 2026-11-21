"""자체 그래프 세 장 (report/figures/own_*.png). 논문 그림은 쓰지 않는다.

실행: python3 analysis/plot_figures.py  (CPU, matplotlib)
데이터는 analysis/recompute.py의 Table 5 전사값을 그대로 쓴다.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recompute import SCREW, COFFEE  # noqa: E402

OUT = os.path.join(HERE, "..", "report", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
COL = {"EXPERT-ONLY": "#1b6ca8", "FROZEN-LM": "#2a9d8f", "FULL-FT": "#e9c46a", "LORA": "#e76f51", "LORA-LM": "#8e44ad"}
TASKS = {"ScrewLightbulb": SCREW, "CoffeePressButton": COFFEE}


def fig_corr():
    """사전 RL 반복 잡음 성공률 대 DSRL 후 학습 성공률 (과제별 산점도)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (tname, tab) in zip(axes, TASKS.items()):
        for rec, ck in tab.items():
            x = [r[1] for r in ck.values()]
            y = [r[3] for r in ck.values()]
            ax.scatter(x, y, color=COL[rec], label=rec, s=40)
        ax.set_title(tname)
        ax.set_xlabel("사전 RL 반복 잡음 학습 성공률 (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("DSRL 후 학습 성공률 (%)")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("SFT 시점 성공률과 조종 후 성공률 (체크포인트 20개 × 과제)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "own_corr_scatter.png"), dpi=160)
    plt.close(fig)


def fig_means():
    """레시피별 DSRL 후 추가 평가 성공률 평균 (체크포인트 4개 평균, 과제별)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    recs = list(SCREW.keys())
    x = np.arange(len(recs))
    w = 0.38
    for k, (tname, tab) in enumerate(TASKS.items()):
        vals = [np.mean([r[4] for r in tab[rec].values()]) for rec in recs]
        ax.bar(x + (k - 0.5) * w, vals, w, label=tname, color=["#1b6ca8", "#8ab6d6"][k])
    ax.set_xticks(x)
    ax.set_xticklabels(recs, rotation=0, fontsize=8)
    ax.set_ylabel("DSRL 후 추가 평가 성공률 평균 (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("레시피별 추가 평가 성공률 (체크포인트 4개 평균)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "own_recipe_means.png"), dpi=160)
    plt.close(fig)


def fig_ckpt():
    """체크포인트 대 DSRL 후 추가 평가 성공률 (레시피별 선, 과제별 패널)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (tname, tab) in zip(axes, TASKS.items()):
        for rec, ck in tab.items():
            steps = sorted(ck)
            ax.plot(steps, [ck[s][4] for s in steps], marker="o", color=COL[rec], label=rec)
        ax.set_title(tname)
        ax.set_xlabel("SFT 스텝 (천)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("DSRL 후 추가 평가 성공률 (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("SFT 진행과 조종 후 추가 평가 성공률", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "own_ckpt_extrap.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_corr()
    fig_means()
    fig_ckpt()
    print("written to", OUT)
