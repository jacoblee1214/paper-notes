"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 여기서 만들지 않는다).

own_table1_ci.png   : 표 1 평균과 시드 기준 95% 구간 (세 정책, 네 칸)
own_coverage.png    : 표 3 학습 조합 수 6/12/18과 고정 보류 6쌍, 전체 24쌍 (SD 막대)
own_hand_heatmap.png: 표 4 미학습 손·그리퍼 × 학습 팔 (평균 SR)
실행: python3 analysis/plot_figures.py   (numpy, scipy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recompute as rc  # noqa: E402

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
INK, ACC, GREY, LIGHT = "#1b1b1a", "#1f5fae", "#9a9890", "#8fb0d9"


def plot_table1():
    groups = [("seen_obj_seen", "학습 객체\n학습 조합 18"), ("seen_obj_held", "학습 객체\n보류 6쌍"),
              ("unseen_obj_seen", "YCB 객체\n학습 조합 18"), ("unseen_obj_held", "YCB 객체\n보류 6쌍")]
    pols = [("GRU", GREY), ("DexFormer1.0", LIGHT), ("Ours", ACC)]
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    width = 0.26
    for j, (pol, col) in enumerate(pols):
        means = [rc.T1[pol][k][0] for k, _ in groups]
        lo = [rc.ci95(*rc.T1[pol][k])[0] for k, _ in groups]
        hi = [rc.ci95(*rc.T1[pol][k])[1] for k, _ in groups]
        x = np.arange(len(groups)) + (j - 1) * width
        ax.bar(x, means, width, color=col, label=pol.replace("DexFormer1.0", "DexFormer 1.0"))
        ax.errorbar(x, means, yerr=[np.array(means) - lo, np.array(hi) - means], fmt="none",
                    ecolor=INK, capsize=2.5, lw=0.9)
        for xi, m in zip(x, means):
            ax.text(xi, 2, f"{m:.1f}", ha="center", va="bottom", fontsize=6.5,
                    color="white" if col != LIGHT else INK)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([g[1] for g in groups], fontsize=8)
    ax.set_ylabel("Stable SR (%)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    ax.set_title("표 1: 평균과 시드 5개 기준 95% 구간 (논문 값 재구성)", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png", dpi=200)
    plt.close(fig)


def plot_coverage():
    ns = [6, 12, 18]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, lab, col, mk in [("common", "보류 6쌍 (고정)", ACC, "o"), ("all", "전체 24쌍", GREY, "s")]:
        m = [rc.T3[n][key][0] for n in ns]
        sd = [rc.T3[n][key][1] for n in ns]
        ax.errorbar(ns, m, yerr=sd, marker=mk, color=col, capsize=3, lw=1.4, label=lab)
        for n, v in zip(ns, m):
            ax.text(n + 0.25, v + 3, f"{v:.1f}", fontsize=8, color=col)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n}쌍 학습" for n in ns], fontsize=8.5)
    ax.set_ylabel("Stable SR (%)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("표 3: 학습 조합 수에 따른 SR (SD 막대, 시드 5개)", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_coverage.png", dpi=200)
    plt.close(fig)


def plot_heatmap():
    cols = rc.HANDS + rc.GRIPPERS
    data = rc.T4_MEAN
    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=45, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(rc.ARMS)))
    ax.set_yticklabels(rc.ARMS, fontsize=8.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 25 else INK)
    ax.axvline(3.5, color=INK, lw=0.8, ls="--")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Stable SR (%)", fontsize=8)
    ax.set_title("표 4: 미학습 손(왼쪽 4열)과 그리퍼(오른쪽 4열)의 평균 SR (논문 값)", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_hand_heatmap.png", dpi=200)
    plt.close(fig)


def main():
    plot_table1()
    plot_coverage()
    plot_heatmap()
    for name in ["own_table1_ci.png", "own_coverage.png", "own_hand_heatmap.png"]:
        print("wrote", FIG / name)


if __name__ == "__main__":
    main()
