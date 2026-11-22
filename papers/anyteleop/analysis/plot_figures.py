"""그래프 세 장을 report/figures/에 저장한다 (우리가 그림; 논문 보고 수치만 사용).

  own_sim_ci.png   : 표 III 시뮬 모방학습, 평균과 시드 기준 t 95% 구간 (n=3 시드)
  own_real_ci.png  : 표 IV 실로봇 10 과제, AnyTeleop 10회 대 기준선 (Wilson 95%, 기준선 n=10 가정)
  own_budget.png   : 표 II 프로파일링 평균 ± SD 대 프레임 시간 예산 (25 Hz, 120 Hz)

실행: python3 analysis/plot_figures.py   (matplotlib, scipy)
"""
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"


def korean_font():
    """Linux Noto CJK, macOS Apple SD Gothic Neo 순서로 찾는다."""
    cands = ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"]
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in cands:
        if c in names:
            return c
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
COL = {"RL": "#9a9a95", "Baseline": "#c0692b", "Ours": "#1f5fae"}


def fig_sim():
    T3 = {
        "Relocate\nFloating": {"RL": (36.3, 15.3), "Baseline": (49.7, 18.3), "Ours": (53.7, 12.2)},
        "Flip Mug\nFloating": {"RL": (33.7, 15.0), "Baseline": (51.3, 34.7), "Ours": (47.3, 28.3)},
        "Open Door\nFloating": {"RL": (69.3, 38.0), "Baseline": (64.7, 14.7), "Ours": (73.3, 9.0)},
        "Relocate\nArm-hand": {"RL": (33.7, 29.3), "Baseline": (40.3, 36.7), "Ours": (70.0, 9.8)},
        "Flip Mug\nArm-hand": {"RL": (31.0, 28.7), "Baseline": (36.0, 32.4), "Ours": (53.7, 24.0)},
        "Open Door\nArm-hand": {"RL": (34.7, 31.7), "Baseline": (51.3, 30.7), "Ours": (79.7, 15.5)},
    }
    tc = stats.t.ppf(0.975, df=2)  # n=3 시드, 표본 SD 가정
    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=200)
    for i, (task, d) in enumerate(T3.items()):
        for j, m in enumerate(["RL", "Baseline", "Ours"]):
            mean, sd = d[m]
            half = tc * sd / math.sqrt(3)
            x = i + (j - 1) * 0.24
            ax.errorbar(x, mean, yerr=half, fmt="o", color=COL[m], ms=5, capsize=3, lw=1.3,
                        label=m if i == 0 else None)
    ax.set_xticks(range(len(T3)))
    ax.set_xticklabels(list(T3.keys()), fontsize=8)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.set_title("표 III 시뮬 모방학습: 평균과 시드 기준 t 95% 구간 (3 시드)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "own_sim_ci.png")
    plt.close(fig)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fig_real():
    T4 = [("Pickup Box", 1.0, 0.9), ("Pickup Fabric", 1.0, 0.9), ("Box Rotation", 0.6, 0.6),
          ("Scissor", 0.8, 0.7), ("Cup Stack", 0.9, 0.6), ("Two Cup Stack", 0.7, 0.3),
          ("Pour Cubes", 0.7, 0.7), ("Cup Into Plate", 1.0, 0.8), ("Open Drawer", 1.0, 0.9),
          ("Drawer + Pickup", 0.9, 0.6)]
    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=200)
    for i, (task, o, b) in enumerate(T4):
        for j, (lab, v, c) in enumerate([("Ours", o, COL["Ours"]), ("Telekinesis [54]", b, COL["Baseline"])]):
            k = round(v * 10)
            lo, hi = wilson(k, 10)
            x = i + (j - 0.5) * 0.3
            ax.errorbar(x, v * 100, yerr=[[v * 100 - lo * 100], [hi * 100 - v * 100]], fmt="o", color=c,
                        ms=5, capsize=3, lw=1.3, label=lab if i == 0 else None)
    ax.set_xticks(range(len(T4)))
    ax.set_xticklabels([t[0] for t in T4], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("표 IV 실로봇 10 과제: 성공률과 Wilson 95% 구간 (10회 기준, 기준선 시도 수 가정)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "own_real_ci.png")
    plt.close(fig)


def fig_budget():
    items = [("Hand pose\nRGB, 데스크톱", 26, 5, 40), ("Hand pose\nRGB, 노트북", 34, 5, 40),
             ("Motion\n데스크톱", 8, 3, 1000 / 120), ("Motion\n노트북", 11, 5, 1000 / 120)]
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=200)
    for i, (lab, m, s, budget) in enumerate(items):
        ax.bar(i, m, yerr=s, capsize=4, color=COL["Ours"] if m <= budget else COL["Baseline"], alpha=0.85)
        ax.hlines(budget, i - 0.4, i + 0.4, colors="black", linestyles="--", lw=1.2)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([t[0] for t in items], fontsize=8)
    ax.set_ylabel("프레임 시간 (ms)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("표 II 프로파일링: 평균 ± SD 대 목표 주파수 예산 (점선: 25 Hz = 40 ms, 120 Hz = 8.3 ms)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "own_budget.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_sim()
    fig_real()
    fig_budget()
    print("wrote", sorted(os.listdir(OUT)))
