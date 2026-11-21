"""재계산 결과를 그래프 세 장으로 그린다(우리가 그린 그래프, 논문 그림 아님).

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
입력: analysis/recompute_results.json, 논문 표 값(아래 상수, analysis/recompute.py와 같은 값)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux(Noto CJK) → macOS(Apple SD Gothic Neo) 순서로 찾는다
for cand in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, ACC, GREY = "#1b1b1a", "#55544f", "#1f5fae", "#9a988f"

TASKS = ["PegInsert", "GearMesh", "NutThread", "BallSort", "ObjectSearch"]
SIM = {
    "MDF++": [85.3, 78.7, 92.0, 93.3, 85.3],
    "MDF++(policy)": [62.7, 64.7, 90.7, 90.0, 86.6],
    "UWM": [88.6, 72.0, 89.3, 53.3, 68.0],
    "DP": [88.7, 81.3, 91.3, 28.0, 54.0],
}
N = 150


def wilson(p_pct, n=N, z=1.959964):
    p = p_pct / 100
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return 100 * (c - h), 100 * (c + h)


def plot_sim():
    fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
    colors = {"MDF++": ACC, "MDF++(policy)": GREY, "UWM": "#b07a2a", "DP": "#6b4c9a"}
    width = 0.19
    for j, (name, vals) in enumerate(SIM.items()):
        xs = [i + (j - 1.5) * width for i in range(len(TASKS))]
        lo = [v - wilson(v)[0] for v in vals]
        hi = [wilson(v)[1] - v for v in vals]
        ax.bar(xs, vals, width=width * 0.92, color=colors[name], label=name, alpha=0.9)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.7, capsize=2)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS, color=INK, fontsize=9)
    ax.set_ylabel("성공률 (%)", color=INK)
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_title("시뮬 과제별 성공률 (표 1 값, 과제당 150회 가정, 막대 위 Wilson 95% 구간)", fontsize=9.5, color=INK, pad=24)
    fig.text(0.01, 0.01, "우리가 그린 그래프. 표 1 값을 다시 그렸고 MDF 계열 최댓값 행은 뺐습니다.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "own_sim_ci.png")
    plt.close(fig)


def plot_real():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), dpi=200, gridspec_kw={"width_ratios": [1, 1]})
    data = {
        "Bolt Screwing": {"MDF++": (16, 95.0), "UWM": (10, 57.5), "MDF++(frozen-ViT)": (6, 46.5)},
        "Nut Mating": {"MDF++": (8, 83.0), "UWM": (5, 61.0), "MDF++(frozen-ViT)": (3, 19.0)},
    }
    for ax, (task, rows) in zip(axes, data.items()):
        names = list(rows)
        ks = [rows[n][0] for n in names]
        scores = [rows[n][1] for n in names]
        cols = [ACC, "#b07a2a", GREY]
        ax.bar(range(len(names)), [100 * k / 20 for k in ks], color=cols, width=0.6)
        for i, k in enumerate(ks):
            lo, hi = wilson(100 * k / 20, n=20)
            ax.errorbar(i, 100 * k / 20, yerr=[[100 * k / 20 - lo], [hi - 100 * k / 20]], fmt="none",
                        ecolor=INK, elinewidth=0.8, capsize=3)
            ax.text(i, hi + 3, f"{k}/20\n점수 {scores[i]:.1f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7.5, color=INK)
        ax.set_ylim(0, 115)
        ax.set_title(task, fontsize=9.5, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("성공률 (%)", color=INK)
    fig.suptitle("실로봇 성공률과 Wilson 95% 구간 (논문 표 9, 과제당 20회)", fontsize=9.5, color=INK)
    fig.text(0.01, 0.01, "우리가 그린 그래프. 점수는 표 9의 평균 점수, 구간은 성공률 기준입니다.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIG / "own_real_ci.png")
    plt.close(fig)


def plot_rep():
    # 표 2 표현 진단: 촉각 유효 랭크와 RGB 프로브 MSE (논문 표 2 값)
    labels = ["Pretrained ViT", "Self-denoising only", "Naive E2E", "MDF++"]
    tac_rank = [20.28, 2.22, 5.99, 33.34]
    probe = [0.163, 0.574, 0.520, 0.043]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=200)
    cols = [GREY, GREY, GREY, ACC]
    axes[0].bar(labels, tac_rank, color=cols)
    axes[0].set_title("촉각 특징의 유효 랭크 (높을수록 다양)", fontsize=9, color=INK)
    axes[1].bar(labels, probe, color=cols)
    axes[1].set_title("RGB 특징의 행동 프로브 MSE (낮을수록 예측 가능)", fontsize=9, color=INK)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7, rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "우리가 그린 그래프. 논문 표 2 값을 다시 그렸습니다(RGB 유효 랭크는 생략).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "own_rep_diag.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_sim()
    plot_real()
    plot_rep()
    print("wrote figures to", FIG)
