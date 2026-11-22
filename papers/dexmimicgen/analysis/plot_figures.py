"""우리가 논문 표 값으로 그린 그래프 세 장 (논문 그림을 복사하지 않음).

  own_task_policy.png   : 과제별 원천 시연과 세 정책의 성공률 (표 I, v2, ±는 SD 가정 막대)
  own_bcrnn_versions.png: 표 I 의 BC-RNN 열, arXiv v1 과 v2 비교 (K2)
  own_realworld_ci.png  : 실로봇 20회 가정의 Wilson 95% 구간 (W1)

실행: python3 analysis/plot_figures.py   → report/figures/own_*.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recompute import (T1_SRC, T1_GMM, T1_BCRNN_V1, T1_BCRNN_V2, T1_DP, TASKS, wilson)  # noqa: E402

FIG = HERE.parents[0] / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
INK, GREY = "#1b1b1a", "#8a8880"
COL = {"원천 시연": "#8a8880", "BC-RNN-GMM": "#c97b2a", "BC-RNN": "#2f6fb0", "DP": "#2a8c6b"}


def fig_task_policy():
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=200)
    x = np.arange(len(TASKS))
    series = [("원천 시연", T1_SRC), ("BC-RNN-GMM", T1_GMM), ("BC-RNN", T1_BCRNN_V2), ("DP", T1_DP)]
    w = 0.2
    for k, (name, data) in enumerate(series):
        means = np.array([d[0] for d in data])
        sds = np.array([d[1] for d in data])
        ax.bar(x + (k - 1.5) * w, means, w, yerr=sds, capsize=2, color=COL[name], label=name,
               error_kw={"elinewidth": 0.8, "ecolor": INK})
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("성공률 (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("표 I (v2): 과제별 성공률, 막대 ±는 보고된 값 (SD 가정)", fontsize=9.5, pad=30, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_task_policy.png")
    plt.close(fig)


def fig_bcrnn_versions():
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
    y = np.arange(len(TASKS))
    for i in range(len(TASKS)):
        a, b = T1_BCRNN_V1[i][0], T1_BCRNN_V2[i][0]
        ax.plot([a, b], [i, i], color=GREY, lw=1.2, zorder=1)
    ax.scatter([d[0] for d in T1_BCRNN_V1], y, color="#b8b5ad", s=34, zorder=2, label="v1 (2024-10-31)")
    ax.scatter([d[0] for d in T1_BCRNN_V2], y, color=COL["BC-RNN"], s=34, zorder=3, label="v2 (2025-03-06)")
    ax.set_yticks(y)
    ax.set_yticklabels(TASKS, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("BC-RNN 성공률 (%)", fontsize=9)
    ax.set_xlim(40, 102)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("표 I 의 BC-RNN 열: v1 과 v2 비교 (9개 과제 중 8개가 바뀜)", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_bcrnn_versions.png")
    plt.close(fig)


def fig_realworld():
    n = 20
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
    rows = [("DexMimicGen 데이터 (18/20 가정)", 18), ("원천 시연만 (0/20 가정)", 0)]
    for i, (label, k) in enumerate(rows):
        lo, hi = wilson(k, n)
        p = k / n
        ax.plot([lo * 100, hi * 100], [i, i], color=COL["DP"] if k else COL["원천 시연"], lw=3, solid_capstyle="butt")
        ax.scatter([p * 100], [i], color=INK, zorder=3, s=22)
        ax.text(hi * 100 + 2, i, f"{k}/{n}, 95% [{lo*100:.0f}, {hi*100:.0f}]%", va="center", fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xlabel("실로봇 성공률 (%), Wilson 95% 구간", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("실로봇 Can Sorting: 논문 보고값 90%, 0%의 불확실성 (20회 가정)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_realworld_ci.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_task_policy()
    fig_bcrnn_versions()
    fig_realworld()
    print("wrote figures to", FIG)
