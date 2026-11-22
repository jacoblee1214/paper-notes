"""우리가 만든 그래프 두 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_grip_ratio.png  : 표 3의 기록 대비 grip 비율 (pose 0.31, vision 1.55, touch 0.99)과 기준선 1.0
own_success_ci.png  : 표 3 성공 횟수와 Clopper-Pearson 95% 구간 (held-out 20회, 시뮬 테스트 36회)
실행: python3 analysis/plot_figures.py   (numpy, scipy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "analysis"
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


def plot_grip():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    g = rec["v2_grip_and_text"]["grip_ratio_table3"]
    labels = ["관절 포즈 모방\n(pose)", "비전 기반 재생\n(vision)", "촉각 기반 재생\n(touch, 논문 제안)"]
    vals = [g["pose"], g["vision"], g["touch"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = [GREY, "#8fb0d9", ACC]
    ax.bar(range(3), vals, color=colors, width=0.58)
    ax.axhline(1.0, color=INK, lw=1, ls="--")
    ax.text(2.42, 1.02, "기록 = 1.0", ha="right", va="bottom", fontsize=8, color=INK)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}×", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("grip / 기록된 힘")
    ax.set_ylim(0, 1.9)
    ax.set_title("보유 기록 대비 파지력 비율 (논문 표 3)", fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01, "집계 방식(평균·중앙값)과 기록별 값은 논문에 없음. 값은 표 3에서 옮김.",
             fontsize=6.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_grip_ratio.png", dpi=180)
    plt.close(fig)


def plot_success():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["v1_success"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    names = {"pose": "pose", "vision": "vision", "touch": "touch (제안)"}
    cols = {"pose": GREY, "vision": "#8fb0d9", "touch": ACC}
    for ax, (key, title) in zip(axes, [("held_out", "held-out 기록 (20회)"), ("sim_test", "시뮬 테스트 (36회)")]):
        block = rec[key]
        for i, c in enumerate(["pose", "vision", "touch"]):
            r = block[c]
            p = 100 * r["rate"]
            lo, hi = 100 * r["clopper_pearson95"][0], 100 * r["clopper_pearson95"][1]
            ax.bar(i, p, color=cols[c], width=0.58)
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, capsize=3, lw=1)
            ax.text(i, 3, f"{r['k']}/{r['n']}", ha="center", va="bottom", color="white", fontsize=8.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([names[c] for c in ["pose", "vision", "touch"]], fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_ylim(0, 108)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("성공률 (%)")
    fig.suptitle("표 3 성공 횟수와 Clopper-Pearson 95% 구간", fontsize=10, x=0.01, ha="left")
    fig.text(0.01, 0.01, "성공 기준: 미끄럼 ≤ 5 mm, 내려놓기 오차 ≤ 30 mm (논문 표 3 캡션). 값은 논문에서 옮김.",
             fontsize=6.5, color="#55544f")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(FIG / "own_success_ci.png", dpi=180)
    plt.close(fig)


def main():
    plot_grip()
    plot_success()
    print("wrote", FIG / "own_grip_ratio.png", FIG / "own_success_ci.png")


if __name__ == "__main__":
    main()
