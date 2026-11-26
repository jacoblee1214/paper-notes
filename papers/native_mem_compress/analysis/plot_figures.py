"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sim_ci.png      : 표 1 시뮬레이션 성공률(과제 5개, 방법 7개)과 Wilson 95% 구간 (100회)
own_real_ci.png     : 표 2 실로봇 최종 단계 성공률과 Wilson 95% 구간 (50회)
own_ablation.png    : 표 4 절제 표의 과제별 성공률 히트맵과 평균 열의 재계산 값
own_token_budget.png: 역사 프레임 수에 따른 시각 토큰 수 (전체 해상도 vs 프레임당 토큰 1개)와 공개 설정의 메모리 상한
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


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"
PALETTE = ["#9a9890", "#c7a86b", "#d98a4a", "#b5493f", "#7b6bb0", "#4a9a7c", "#1f5fae"]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def dotplot(ax, tasks, methods, data, n, colors):
    m = len(methods)
    for i, (name, col) in enumerate(zip(methods, colors)):
        xs, ys, lo, hi = [], [], [], []
        for j in range(len(tasks)):
            pct = data[name][j]
            k = round(pct * n / 100)
            a, b = wilson(k, n)
            xs.append(j + (i - (m - 1) / 2) * 0.11)
            ys.append(pct)
            lo.append(max(0.0, pct - a))
            hi.append(max(0.0, b - pct))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=4.5, color=col, ecolor=col, elinewidth=1.1,
                    capsize=0, label=name, zorder=3 if name == "Ours" else 2)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks)
    ax.set_ylim(-3, 104)
    ax.set_ylabel("성공률 (%)")
    ax.grid(axis="y", color="#e5e3dc", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_sim():
    tasks = ["Click\nButtons", "Click Buttons\n(hard)", "Swap\nBlocks", "Put Back\nBlock", "Observe\nand Pickup"]
    data = {"π0.5": [0, 0, 24, 11, 9], "X-VLA": [7, 12, 16, 18, 9], "MemER": [12, 8, 18, 12, 2],
            "Mem-0": [0, 1, 67, 90, 4], "HAMLET*": [4, 17, 11, 3, 10], "MEM-short*": [0, 39, 4, 15, 6],
            "Ours": [94, 88, 94, 100, 44]}
    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=200)
    dotplot(ax, tasks, list(data), data, 100, PALETTE)
    ax.set_title("표 1 시뮬레이션 성공률과 Wilson 95% 구간 (과제당 100회, 논문 값을 다시 그림)", fontsize=10.5, color=INK)
    ax.legend(ncol=7, fontsize=7.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), handletextpad=0.2,
              columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png")
    plt.close(fig)


def fig_real():
    tasks = ["Click Buttons\n(S3)", "Put Back Block\n(S2)", "Grocery Checkout\nScanning, unseen (S3)"]
    data = {"π0.5": [2, 14, 28], "π0.5 + RTC": [16, 24, 64], "Mem-0": [0, 0, 0], "MEM-short*": [40, 0, 52],
            "Ours": [96, 100, 100]}
    cols = ["#9a9890", "#d98a4a", "#b5493f", "#4a9a7c", "#1f5fae"]
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=200)
    dotplot(ax, tasks, list(data), data, 50, cols)
    ax.set_title("표 2 실로봇 최종 단계 성공률과 Wilson 95% 구간 (과제당 50회)", fontsize=10.5, color=INK)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png")
    plt.close(fig)


def fig_ablation():
    rows = ["Unfrozen VLA", "w/o Stage1", "Sparse update (0.5Hz)", "Short horizon (2s)", "Short horizon (4s)",
            "Short horizon (6s)", "Ours"]
    cols = ["Click\nButtons", "Click Buttons\n(hard)", "Swap\nBlocks", "Put Back\nBlock", "Observe\nand Pickup"]
    vals = np.array([[38, 24, 45, 0, 9], [94, 80, 92, 17, 7], [83, 63, 91, 24, 9], [26, 87, 0, 18, 39],
                     [69, 87, 0, 18, 37], [80, 88, 0, 18, 44], [94, 88, 94, 100, 44]], float)
    reported = [23.2, 58.0, 53.8, 34.0, 43.0, 46.0, 84.0]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=200, gridspec_kw={"width_ratios": [3.3, 1.5]})
    ax.imshow(vals, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels(rows, fontsize=8.5)
    for i in range(7):
        for j in range(5):
            ax.text(j, i, f"{vals[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if vals[i, j] > 55 else INK)
    ax.set_title("표 4 절제: 과제별 성공률 (%)", fontsize=10, color=INK)
    comp = vals.mean(axis=1)
    y = np.arange(7)
    ax2.barh(y - 0.18, reported, 0.34, color=ACC, label="논문 Avg. 열")
    ax2.barh(y + 0.18, comp, 0.34, color=GREY, label="다섯 칸 평균(재계산)")
    for i in range(7):
        if abs(reported[i] - comp[i]) > 0.05:
            ax2.text(max(reported[i], comp[i]) + 2, i, f"{reported[i]:.1f} 대 {comp[i]:.1f}", va="center", fontsize=8,
                     color="#b5493f")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 125)
    ax2.set_title("평균 열 비교", fontsize=10, color=INK)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png")
    plt.close(fig)


def fig_tokens():
    frames = np.array([1, 2, 5, 10, 20, 50, 100, 160, 200, 500, 1000, 5000])
    views = 3
    full = frames * views * 256
    one = frames * views
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=200)
    ax.loglog(frames, full, "-o", ms=3.5, color="#b5493f", label="프레임당 256 토큰 x 3 시점 (전체 해상도 역사)")
    ax.loglog(frames, one, "-o", ms=3.5, color=ACC, label="프레임당 1 토큰 x 3 시점 (NativeMEM)")
    ax.axhline(256, color=GREY, ls="--", lw=1)
    ax.axhline(512, color=GREY, ls="--", lw=1)
    ax.text(1.05, 275, "공개 설정의 메모리 상한 256 토큰 (시뮬 과제 대부분)", fontsize=7.8, color="#55544f")
    ax.text(1.05, 560, "상한 512 토큰 (cover_blocks, 실로봇 설정 예)", fontsize=7.8, color="#55544f")
    ax.scatter([5, 9], [3840, 6912], color="#b5493f", zorder=5)
    ax.annotate("Native History K=4\n3,840 토큰 (표 6)", (5, 3840), (1.0, 60000), fontsize=7.8,
                arrowprops=dict(arrowstyle="-", color=GREY))
    ax.annotate("K=8 6,912 토큰\n(학습 용량 초과라고 서술)", (9, 6912), (4.5, 400000), fontsize=7.8,
                arrowprops=dict(arrowstyle="-", color=GREY))
    ax.scatter([160], [480], color=ACC, zorder=5)
    ax.annotate("160프레임 = 480 토큰", (160, 480), (30, 70), fontsize=7.8, arrowprops=dict(arrowstyle="-", color=GREY))
    ax.set_xlabel("입력에 든 프레임 수 (현재 프레임 포함, 프레임당 3 시점)")
    ax.set_ylabel("VLA 입력에 더해지는 토큰 수")
    ax.set_title("역사 길이에 따른 토큰 수 (산술, 논문 식 4와 C.2의 계산)", fontsize=10.5, color=INK)
    ax.grid(color="#e5e3dc", lw=0.6, which="both")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_token_budget.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_sim()
    fig_real()
    fig_ablation()
    fig_tokens()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
