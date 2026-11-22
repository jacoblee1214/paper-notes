"""Phantom 논문 표 수치의 Wilson 95% 구간 그래프 두 장을 만든다(우리가 그린 그림).

입력: analysis/recompute.py 와 같은 표 값(논문 Table 1, 6). 출력: report/figures/own_*.png
실행: python3 analysis/plot_figures.py   (numpy, matplotlib)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from recompute import wilson, N  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "report" / "figures"


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#c0632b"


def fig_table1():
    cols = ["Pick/Place\nBook", "Stack\nCups", "Tie\nRope", "Rotate\nBox", "Grasp\nBrush", "Sweep\n>0", "Sweep\n>2", "Sweep\n>4"]
    # 0.75 은 25회에서 k/25가 아니므로 19/25(반올림)로 그린다. 그림에 표시한다.
    inp = [23, 18, 16, 18, 22, 20, 18, 10]
    msk = [23, 13, 15, 19, 19, 19, 18, 17]
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=150)
    for k, color, off, lab in [(inp, ACC, -0.13, "Hand Inpaint"), (msk, GREY, 0.13, "Hand Mask")]:
        lo = [wilson(v, N)[0] for v in k]
        hi = [wilson(v, N)[1] for v in k]
        p = [v / N for v in k]
        ax.errorbar(x + off, p, yerr=[np.array(p) - lo, np.array(hi) - np.array(p)], fmt="o", color=color,
                    ms=5, capsize=3, lw=1.2, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("성공률 (25회)", fontsize=9)
    ax.set_title("논문 표 1 재표시: 과제별 성공률과 Wilson 95% 구간", fontsize=10, color=INK)
    ax.axvspan(4.5, 7.5, color="#eef3fa", zorder=0)
    ax.text(6, 1.0, "Sweep 단계별 (같은 시행)", ha="center", fontsize=7.5, color=INK)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.text(0.01, 0.01, "Mask의 Grasp Brush·Sweep>0 0.75는 25회에서 불가능한 값이라 19/25로 그림 (V1). 출처: 논문 Table 1.",
             fontsize=6.8, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "own_t1_ci.png")
    plt.close(fig)


def fig_table6():
    labels = ["로봇 50", "사람 50", "로봇 100", "사람 100", "사람 300"]
    k = [13, 11, 22, 16, 21]
    cols = [GREY, ACC, GREY, ACC, ACC]
    p = [v / N for v in k]
    lo = [wilson(v, N)[0] for v in k]
    hi = [wilson(v, N)[1] for v in k]
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=150)
    x = np.arange(len(k))
    for i in range(len(k)):
        ax.errorbar(x[i], p[i], yerr=[[p[i] - lo[i]], [hi[i] - p[i]]], fmt="o", color=cols[i], ms=6, capsize=4)
        ax.text(x[i] + 0.08, p[i] + 0.01, f"{k[i]}/25", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("성공률", fontsize=9)
    ax.set_title("논문 표 6 재표시: 로봇 대 사람 시연 수, Wilson 95% 구간", fontsize=10, color=INK)
    ax.text(0.02, 0.03, "n=25 두 군 비교에서 검출 가능한 최소 차이 약 37 %p (80% 검정력, 근사). "
            "p=0.78, 0.095는 '차이 없음'이 아니라 검정력 부족.",
            transform=ax.transAxes, fontsize=7.2, color=INK)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "own_t6_ci.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_table1()
    fig_table6()
    print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
