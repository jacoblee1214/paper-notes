"""우리가 만든 그래프 두 장을 report/figures/에 저장한다 (논문 그림은 여기서 그리지 않는다).

own_sr_ci.png      : Table 1·3의 성공률(n=50)과 Wilson 95% 구간, 방법별 비교
own_scaling_lpips.png: Table 5 데이터 크기 곡선(PushT)과 Table 4 LPIPS 비교(Granular 기준)
실행: python3 analysis/plot_figures.py   (matplotlib, numpy; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
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
N = 50


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_sr():
    # Table 1 (PushT 제외 4개 환경) + Table 3 (WallRandom, PushObj) 성공률
    envs = ["Maze", "Wall", "Reach", "PushT", "WallRandom", "PushObj"]
    data = {
        "Maze":        {"IRIS": .74, "DreamerV3": 1.00, "TD-MPC2": .00, "Ours": .98},
        "Wall":        {"IRIS": .04, "DreamerV3": 1.00, "TD-MPC2": .00, "Ours": .96},
        "Reach":       {"IRIS": .18, "DreamerV3": .64, "TD-MPC2": .00, "Ours": .92},
        "PushT":       {"IRIS": .32, "DreamerV3": .30, "TD-MPC2": .00, "Ours": .90},
        "WallRandom":  {"IRIS": .06, "DreamerV3": .76, "Ours": .82},
        "PushObj":     {"IRIS": .14, "DreamerV3": .18, "Ours": .34},
    }
    methods = ["IRIS", "DreamerV3", "TD-MPC2", "Ours"]
    colors = {"IRIS": "#c9c6bd", "DreamerV3": "#9a9890", "TD-MPC2": "#6e6c66", "Ours": ACC}
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    width = 0.2
    x = np.arange(len(envs))
    for j, m in enumerate(methods):
        vals, lo, hi = [], [], []
        for e in envs:
            v = data[e].get(m)
            if v is None:
                vals.append(np.nan); lo.append(0); hi.append(0); continue
            k = round(v * N)
            a, b = wilson(k, N)
            vals.append(100 * v); lo.append(max(0.0, 100 * v - a)); hi.append(max(0.0, b - 100 * v))
        pos = x + (j - 1.5) * width
        ax.bar(pos, np.nan_to_num(vals), width, color=colors[m], label=m,
               yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], capsize=2, error_kw={"lw": 0.8, "ecolor": INK})
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 115)
    ax.axvline(3.5, color=GREY, lw=0.8, ls="--")
    ax.text(1.5, 108, "Table 1 (n = 50)", ha="center", color=INK, fontsize=9)
    ax.text(4.5, 108, "Table 3 (n = 50)", ha="center", color=INK, fontsize=9)
    ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("DINO-WM 성공률과 Wilson 95% 구간 (논문 보고값을 n = 50으로 환산)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_sr_ci.png", dpi=200)
    plt.close(fig)


def plot_scaling_lpips():
    sizes = [200, 1000, 5000, 10000, 18500]
    sr = [0.08, 0.48, 0.72, 0.88, 0.92]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    ax.plot(sizes, [100 * v for v in sr], marker="o", color=ACC, lw=1.6)
    for s, v in zip(sizes, sr):
        ax.annotate(f"{100*v:.0f}%", (s, 100 * v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("학습 궤적 수 (로그 축)")
    ax.set_ylabel("PushT 성공률 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("데이터 크기 곡선 (Table 5, n = 50 가정)", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    methods = ["R3M", "ResNet", "DINO CLS", "AVDC", "Ours"]
    gran = [0.080, 0.080, 0.086, 0.106, 0.035]
    cols = ["#c9c6bd"] * 4 + [ACC]
    ax.bar(methods, gran, color=cols)
    for i, v in enumerate(gran):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("LPIPS (낮을수록 좋음)")
    ax.set_title("Granular 예측 프레임 LPIPS (Table 4)", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_scaling_lpips.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_sr()
    plot_scaling_lpips()
    print("wrote", FIG / "own_sr_ci.png", FIG / "own_scaling_lpips.png")
