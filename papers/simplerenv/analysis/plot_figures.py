"""SimplerEnv 분석 그래프 세 장 (우리가 그림, 논문 그림 아님).

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png   (numpy, scipy, matplotlib; 수 초)
입력은 analysis/recompute.py 의 표 값과 함수를 그대로 쓴다.
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
spec = importlib.util.spec_from_file_location("rc", ROOT / "analysis" / "recompute.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

from matplotlib import font_manager

# 한글 폰트: Linux Noto CJK -> macOS Apple SD Gothic Neo 순서로 있는 것을 쓴다
_KO = next((f for f in ["Noto Sans CJK KR", "Apple SD Gothic Neo"]
            if any(f == e.name for e in font_manager.fontManager.ttflist)), "DejaVu Sans")
plt.rcParams["font.family"] = [_KO, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#8a8880"
COLORS = ["#1f5fae", "#4f9a6b", "#c4623a", "#8a5fb0", "#b08a2e", "#7a7a7a"]
SHORT = ["RT-1 (Conv.)", "RT-1 (15%)", "RT-1-X", "RT-2-X", "Octo-Base", "RT-1 (Begin)"]


def fig_real_ci():
    """Pick Coke Can 평균: 실로봇 성공률(Wilson 95%)과 시뮬 Visual Matching 값."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
    x = np.arange(6)
    real = rc.REAL[:, 3]
    lo, hi = zip(*[rc.wilson(round(p * 75), 75) for p in real])
    ax.errorbar(x - 0.12, real, yerr=[real - np.array(lo), np.array(hi) - real], fmt="o", color=ACC,
                capsize=3, label="실로봇 (n = 75, Wilson 95%)")
    ax.scatter(x + 0.12, rc.SIM_VM[:, 3], marker="s", color="#c4623a", label="시뮬 Visual Matching (n = 300)")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=8, rotation=12)
    ax.set_ylabel("Pick Coke Can 평균 성공률", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("표 IV: 실로봇 불확실성과 시뮬 값 (정책 6개)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png")
    plt.close(fig)


def fig_scatter():
    """네 Google 과제의 실로봇 대 시뮬 (Visual Matching), 정책 6개."""
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=160)
    idx = [3, 4, 7, 8]
    names = ["Pick Coke Can", "Move Near", "Open/Close", "Open+Apple"]
    markers = ["o", "^", "s", "D"]
    for k, c in enumerate(idx):
        r = rc.REAL[:, c]
        s = rc.SIM_VM[:, c]
        ax.scatter(r, s, marker=markers[k], s=34, color=COLORS[k], alpha=0.85, label=names[k], edgecolor="white", lw=0.5)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.9, ls="--")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("실로봇 성공률", fontsize=9)
    ax.set_ylabel("시뮬 성공률 (Visual Matching)", fontsize=9)
    ax.grid(color="#e5e3dc", lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Google Robot: 과제별 쌍(정책 6개)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_scatter.png")
    plt.close(fig)


def fig_mmrv_noise(n_draw=6000):
    """실로봇 이항 잡음만 넣었을 때 MMRV 분포 (시뮬 고정). 점 추정이 잡음 바닥 안에 있는지 본다."""
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=160, sharey=False)
    for ax, (c, title) in zip(axes, [(3, "Pick Coke Can 평균 (n = 75)"), (4, "Move Near (n = 60)")]):
        n = rc.N_REAL[c]
        ks = np.round(rc.REAL[:, c] * n).astype(int)
        vals = np.array([rc.mmrv(rc.SIM_VM[:, c], rng.binomial(n, ks / n) / n) for _ in range(n_draw)])
        point = rc.mmrv(rc.SIM_VM[:, c], rc.REAL[:, c])
        ax.hist(vals, bins=np.linspace(0, max(0.2, vals.max()), 36), color="#b9cde6", edgecolor="white")
        ax.axvline(point, color="#c4623a", lw=1.6, label=f"논문 값 {point:.3f}")
        ax.set_xlabel("MMRV (실로봇 잡음만 재표본)", fontsize=8.5)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    axes[0].set_ylabel("재표본 횟수", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_mmrv_noise.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_real_ci()
    fig_scatter()
    fig_mmrv_noise()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
