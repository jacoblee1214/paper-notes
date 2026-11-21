"""자체 그래프 두 장 (report/figures/own_*.png). 입력은 recompute.py와 같은 표 값이다.

  own_real_ci.png : 표 I, 실로봇 성공률 (사전학습, ID), Wilson 95% 구간 (칸당 n=50, Rice-Cooker는 20)
  own_libero.png  : 표 II, LIBERO 과제별 평균 ± SD (시드 3, ± = SD 가정)

실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
한글 폰트: Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"

CANDIDATES = [
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "Apple SD Gothic Neo"),
]
for path, name in CANDIDATES:
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
INK, GRAY, ACC = "#1b1b1a", "#8a8880", "#1f5fae"
COLORS = {"UWM": ACC, "DP": "#8a8880", "PAD": "#c47b2c", "GR1": "#4c9a5f"}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# 표 I, 사전학습 ID 칸 (k/n은 rate x n으로 복원)
TASKS = ["Stack-Bowls", "Block-Cabinet", "Paper-Towel", "Hang-Towel", "Rice-Cooker"]
ID_RATE = {
    "UWM": [0.86, 0.76, 0.78, 0.82, 0.60],
    "DP": [0.48, 0.60, 0.52, 0.64, 0.35],
    "PAD": [0.08, 0.00, 0.42, 0.52, 0.00],
    "GR1": [0.66, 0.66, 0.60, 0.66, 0.40],
}
N = [50, 50, 50, 50, 20]


def plot_real():
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    width = 0.2
    for j, method in enumerate(["UWM", "GR1", "DP", "PAD"]):
        xs, ys, lo, hi = [], [], [], []
        for i, (r, n) in enumerate(zip(ID_RATE[method], N)):
            k = round(r * n)
            a, b = wilson(k, n)
            x = i + (j - 1.5) * width
            xs.append(x)
            ys.append(r)
            lo.append(r - a)
            hi.append(b - r)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=COLORS[method], ms=5, capsize=3,
                    label=method if method != "UWM" else "UWM (사전학습)", lw=1.2)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS, fontsize=8)
    ax.set_ylabel("성공률 (ID, 사전학습)", fontsize=9, color=INK)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color="#dcdad3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(OUT / "own_real_ci.png")
    plt.close(fig)


# 표 II, LIBERO (평균, SD)
LIBERO_TASKS = ["Book-Caddy", "Soup-Cheese", "Bowl-Drawer", "Moka-Moka", "Mug-Mug"]
LIBERO = {
    "UWM": [(0.91, 0.07), (0.93, 0.01), (0.80, 0.02), (0.68, 0.02), (0.65, 0.01)],
    "DP": [(0.73, 0.10), (0.88, 0.02), (0.77, 0.02), (0.65, 0.03), (0.53, 0.05)],
    "PAD": [(0.78, 0.04), (0.47, 0.04), (0.74, 0.05), (0.59, 0.08), (0.25, 0.04)],
    "GR1": [(0.77, 0.03), (0.65, 0.05), (0.62, 0.03), (0.46, 0.04), (0.38, 0.05)],
}


def plot_libero():
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    width = 0.2
    for j, method in enumerate(["UWM", "DP", "PAD", "GR1"]):
        for i, (m, s) in enumerate(LIBERO[method]):
            x = i + (j - 1.5) * width
            ax.errorbar([x], [m], yerr=[[s], [s]], fmt="o", color=COLORS[method], ms=5, capsize=3,
                        label=method if i == 0 else None, lw=1.2)
    ax.set_xticks(range(len(LIBERO_TASKS)))
    ax.set_xticklabels(LIBERO_TASKS, fontsize=8)
    ax.set_ylabel("성공률 (평균 ± SD, 시드 3)", fontsize=9, color=INK)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#dcdad3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(OUT / "own_libero.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot_real()
    plot_libero()
    print("wrote", OUT / "own_real_ci.png", "and own_libero.png")
