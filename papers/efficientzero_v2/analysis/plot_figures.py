"""분석 리포트와 게시본에 넣을 그래프 세 장을 만든다(모두 우리가 그린 그래프, 논문 그림 아님).

입력: analysis/recompute.py의 표 값(같은 파일에서 import)
출력: report/figures/own_atari_sensitivity.png, own_dmc_diff.png, own_ablation.png
실행: python3 analysis/plot_figures.py
"""
import importlib.util
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("rc", HERE / "recompute.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

# 한글 폰트: Linux Noto CJK -> macOS Apple SD Gothic Neo 순서로 찾는다
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for p in FONT_CANDIDATES:
    if os.path.exists(p):
        plt.rcParams["font.family"] = matplotlib.font_manager.FontProperties(fname=p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
C_EZ, C_DV, C_GREY = "#1f5fae", "#b5651d", "#8a8a85"


def fig_atari_sensitivity():
    """Atari 정규화 평균: 보고값과 재계산값, 그리고 상위 게임을 뺐을 때."""
    norm = sorted([(rc.normed(rc.ATARI[g][10], rc.ATARI[g][0], rc.ATARI[g][1]), g) for g in rc.ATARI], reverse=True)
    vals = [v for v, _ in norm]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    labels = ["보고값\n(평균)", "재계산\n(평균)", "재계산\n(중앙값)", "상위 1개\n제외 평균", "상위 2개\n제외 평균"]
    values = [2.428, np.mean(vals), np.median(vals), np.mean(vals[1:]), np.mean(vals[2:])]
    colors = [C_GREY, C_EZ, C_EZ, "#6a9bd1", "#6a9bd1"]
    bars = a1.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    a1.set_ylabel("EZ-V2 정규화 점수")
    a1.set_title("Atari 100k: 보고값과 재계산 (26게임)", fontsize=11)
    a1.set_ylim(0, max(values) * 1.18)
    top = norm[:8][::-1]
    a2.barh([g for _, g in top], [v for v, _ in top], color=C_EZ)
    a2.set_xlabel("게임별 정규화 점수 (EZ-V2)")
    a2.set_title("평균을 끌어올리는 게임", fontsize=11)
    for i, (v, g) in enumerate(top):
        a2.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_atari_sensitivity.png", dpi=170)
    plt.close(fig)


def fig_dmc_diff():
    """DMC 과제별 EZ-V2 - DreamerV3 점수 차이 (Proprio 50k/100k, Vision 100k/200k)."""
    groups, data = [], []
    for name, M, a, b in [("Proprio 50k", rc.DMC_EASY, 3, 2), ("Proprio 100k", rc.DMC_HARD, 3, 2),
                          ("Vision 100k", rc.DMC_EASY, 7, 6), ("Vision 200k", rc.DMC_HARD, 7, 6)]:
        groups.append(name)
        data.append([v[a] - v[b] for v in M.values()])
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    rng = np.random.default_rng(0)
    for gi, d in enumerate(data):
        x = gi + rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(x, d, s=34, color=C_EZ, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.hlines(np.mean(d), gi - 0.25, gi + 0.25, color=C_DV, linewidth=2)
        ax.text(gi + 0.28, np.mean(d), f"평균 {np.mean(d):+.0f}", va="center", fontsize=8.5, color=C_DV)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel("EZ-V2 점수 - DreamerV3 점수 (Table 2)")
    ax.set_title("DMC: 과제별 차이 (양수는 EZ-V2 우세, 과제 10개씩)", fontsize=11)
    ax.set_xlim(-0.5, len(groups) - 0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_dmc_diff.png", dpi=170)
    plt.close(fig)


def fig_ablation():
    """구성요소 제거 ablation: 과제별 원본 대 변형 (Table 4·5)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (title, tab) in zip(axes, [("Proprio (Table 4)", rc.ABL_PROPRIO), ("Vision (Table 5)", rc.ABL_VISION)]):
        y = np.arange(len(rc.ABL_TASKS_PROPRIO))
        ax.hlines(y, np.minimum(tab["Original"], tab["Sample MCTS"]), np.maximum(tab["Original"], tab["Sample MCTS"]),
                  color="#d9d9d4", linewidth=3, zorder=1)
        for var, mk, col in [("Multi-step TD", "s", "#7a5ea8"), ("GAE", "^", "#3c9a6b"), ("Sample MCTS", "x", C_DV)]:
            ax.scatter(tab[var], y, marker=mk, s=28, color=col, label=var, zorder=2)
        ax.scatter(tab["Original"], y, s=46, color=C_EZ, label="원본 EZ-V2", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(rc.ABL_TASKS_PROPRIO, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("점수")
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_atari_sensitivity()
    fig_dmc_diff()
    fig_ablation()
    print("wrote figures to", FIG)
