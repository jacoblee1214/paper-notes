"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_ahr_bounds.png : 사용자 연구 AHR 평균과 피험자 단위 95% 구간 상한 (분산 상한 가정, recompute.py와 같은 식)
own_view_recall.png: 표 1 뷰 리콜 PSNR, SSIM, LPIPS (구간 없음: 샘플별 분산 없음)
own_vbench_gap.png : 표 2 VBench, 제안 방법 대 DaS와 Wan2.1 차이 (구간 없음)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "analysis"))
from recompute import AHR, VIEW, VBENCH, VB_NAMES, N_SUBJ  # noqa: E402


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
COL = {"TrajectoryCrafter": "#9aa0a6", "DaS": "#6d8fb8", "Wan2.1-Inpainting": "#b7a36b", "Ours": "#1f5fae"}


def half_width(m, n):
    return 1.96 * math.sqrt((m - 1.0) * (4.0 - m)) / math.sqrt(n)


def plot_ahr():
    crits = list(AHR.keys())
    methods = list(COL.keys())
    x = np.arange(len(crits))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    for i, mth in enumerate(methods):
        means = [AHR[c][mth] for c in crits]
        errs = [half_width(m, N_SUBJ) for m in means]
        ax.bar(x + (i - 1.5) * w, means, w, yerr=errs, capsize=3, color=COL[mth], label=mth,
               error_kw={"elinewidth": 0.8})
    ax.set_xticks(x, crits)
    ax.set_ylim(1, 4.2)
    ax.set_ylabel("평균 순위 (AHR, 1–4, 높을수록 좋음)")
    ax.set_title("사용자 연구 AHR: 피험자 단위 95% 구간 상한 (분산 상한 가정)", fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_ahr_bounds.png")
    plt.close(fig)


def plot_view():
    names = list(VIEW.keys())
    labels = ["PSNR (dB, ↑)", "SSIM (↑)", "LPIPS (↓)"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), dpi=200)
    for j, ax in enumerate(axes):
        vals = [VIEW[n][j] for n in names]
        ax.bar(range(len(names)), vals, color=[COL[n] for n in names])
        ax.set_xticks(range(len(names)), ["TC", "DaS", "Wan", "Ours"], fontsize=8)
        ax.set_title(labels[j], fontsize=9)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        for k, v in enumerate(vals):
            ax.text(k, v, f"{v:g}", ha="center", va="bottom", fontsize=6.5)
    fig.suptitle("뷰 리콜 일관성 (표 1 왼쪽, 구간 없음)  TC=TrajectoryCrafter, Wan=Wan2.1-Inpainting", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_view_recall.png")
    plt.close(fig)


def plot_vbench():
    base = ["DaS", "Wan2.1-Inpainting"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    y = np.arange(len(VB_NAMES))
    h = 0.38
    for i, b in enumerate(base):
        d = [VBENCH["Ours"][k] - VBENCH[b][k] for k in range(6)]
        ax.barh(y + (i - 0.5) * h, d, h, color=COL[b], label=f"Ours - {b}")
        for k, v in enumerate(d):
            ax.text(v, y[k] + (i - 0.5) * h, f" {v:+.4f}", va="center", fontsize=6.5)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y, VB_NAMES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("VBench 차이 (표 2, 구간 없음)")
    ax.set_title("VBench 차이: Temporal Flickering 대 Wan2.1 (+0.115)만 크고 나머지는 0.02 이하", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_vbench_gap.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_ahr()
    plot_view()
    plot_vbench()
    for p in sorted(FIG.glob("own_*.png")):
        print("wrote", p)
