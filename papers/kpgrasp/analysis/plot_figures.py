"""우리가 그린 그래프 세 장을 report/figures/에 만든다 (수 초, matplotlib).

  own_ablation.png    : 표 3 GSR 막대 (시드 분산 미보고 표시)
  own_realworld_ci.png: 실제 로봇 83/100과 Wilson 95% 구간
  own_fig5_gap.png    : 그림 5의 KPGrasp 대 pose-joint 차이 (판독값, 규모 축별)

한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
실행: python3 analysis/plot_figures.py
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"


def korean_font():
    for path in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                 "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
        if Path(path).exists():
            return font_manager.FontProperties(fname=path).get_name()
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, GREY, RED, BLUE, ORANGE = "#1b1b1a", "#8a8880", "#c8102e", "#1f5fae", "#f28e2b"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#dcdad3", lw=0.6)


def ablation():
    rows = [("Pose-Joint (Euclidean flow)", 62.5), ("Pose-Joint (SE(3) flow)", 55.0),
            ("Pose-Joint (24 tokens)", 63.5), ("Pose-Joint (34 tokens)", 66.1),
            ("Keypoint (1 token)", 55.4), ("Keypoint (no ranking)", 67.0), ("Keypoint (ours)", 76.3)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    colors = [GREY] * 4 + [BLUE] + [GREY] + [RED]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=colors, height=0.62)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.8, i, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(40, 84)
    ax.set_xlabel("GSR (%)", color=INK)
    ax.tick_params(labelsize=8.5)
    style(ax)
    ax.set_title("표 3 재정리: 같은 시드 한 번의 결과, 분산 미보고", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "own_ablation.png")
    plt.close(fig)


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def realworld():
    k, n = 83, 100
    lo, hi = wilson(k, n)
    fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=200)
    ax.barh([0], [100 * k / n], color=BLUE, height=0.5)
    ax.errorbar([100 * k / n], [0], xerr=[[100 * k / n - lo], [hi - 100 * k / n]], fmt="none", ecolor=INK, capsize=5)
    ax.axvline(50, color=GREY, ls="--", lw=0.8)
    ax.text(50.8, 0.36, "50%", fontsize=8, color=GREY)
    ax.text(hi + 1.2, 0, f"83/100 (Wilson 95%: {lo:.1f}–{hi:.1f})", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("성공률 (%) · 객체 20개 × 5회", color=INK)
    ax.tick_params(labelsize=8.5)
    style(ax)
    ax.set_title("실제 로봇 (논문 5.4절): 시도 독립 가정 구간", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "own_realworld_ci.png")
    plt.close(fig)


def fig5_gap():
    panels = {
        "데이터 크기 (비율)": ([0.001, 0.01, 0.1, 1.0], [34.2, 54.6, 69.5, 76.3], [33.0, 43.5, 51.9, 62.5]),
        "모델 크기 (DiT 블록)": ([2, 4, 8, 16], [56.3, 71.9, 76.3, 77.6], [48.6, 58.6, 62.5, 65.3]),
        "배치 크기": ([8, 64, 512, 4096], [29.1, 57.2, 70.3, 76.3], [15.1, 34.9, 53.4, 62.5]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), dpi=200)
    for ax, (title, (x, kp, pj)) in zip(axes, panels.items()):
        gap = [a - b for a, b in zip(kp, pj)]
        ax.plot(range(len(x)), gap, marker="o", color=RED, lw=1.6)
        for i, g in enumerate(gap):
            ax.text(i, g + 1.1, f"{g:.1f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(x)), [str(v) for v in x], fontsize=8)
        ax.set_ylim(0, 27)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dcdad3", lw=0.6)
    axes[0].set_ylabel("KPGrasp - pose-joint (pp)", fontsize=8.5, color=INK)
    fig.suptitle("그림 5 판독값의 차이: 규모가 커져도 차이가 계속 벌어지지는 않음", fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "own_fig5_gap.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ablation()
    realworld()
    fig5_gap()
    for name in ["own_ablation.png", "own_realworld_ci.png", "own_fig5_gap.png"]:
        print("wrote", OUT / name)
