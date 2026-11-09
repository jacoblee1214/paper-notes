"""리포트 그래프 (CPU, matplotlib). 실행: python3 analysis/plot_figures.py → report/figures/*.png
팔레트는 dataviz 검증 통과 세트(Vision #0072B2, TacRGB #A8560B, TacFF #7B3FA0). 한 축, 범례 + 직접 라벨.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
T1 = json.loads((ROOT / "analysis" / "concat_toy_results.json").read_text())
T2 = json.loads((ROOT / "analysis" / "lowdim_toy_results.json").read_text())

for f in ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

C = {"Vision": "#0072B2", "TacRGB": "#A8560B", "TacFF": "#7B3FA0"}
INK, MUTED, GRID = "#1b1b1a", "#55544f", "#dcdad3"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


# 1) 시뮬 과제별 차이(TacFF - Vision, TacRGB - Vision): n=1500 가정 대 n=150 가정의 Newcombe 구간
def fig_sim_gap():
    tasks = ["S1", "S2", "S3", "S4", "S5"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    for j, (comp, color, lbl) in enumerate([("TacFF_vs_Vision", C["TacFF"], "TacFF - Vision"),
                                             ("TacRGB_vs_Vision", C["TacRGB"], "TacRGB - Vision")]):
        rows1 = R["T3_sim_tests"][comp + "_n1500"]["rows"]
        rows150 = R["T3_sim_tests"][comp + "_n150"]["rows"]
        for i in range(5):
            x = i + (j - 0.5) * 0.28
            r1, r2 = rows1[i], rows150[i]
            ax.plot([x, x], r1["newcombe_pp"], color=color, lw=3, alpha=0.35, solid_capstyle="butt")
            ax.plot([x, x], r2["newcombe_pp"], color=color, lw=1.4)
            ax.plot(x, r1["diff_pp"], "o", color=color, ms=5, label=lbl if i == 0 else None)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(5), tasks)
    ax.set_ylabel("성공률 차이 (pp)", color=INK, fontsize=10)
    ax.set_title("시뮬 과제별 차이: 굵은 띠는 n=1500(논문 가정), 가는 선은 n=150(3 seed × 50 초기조건)", fontsize=9, color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "sim_gap_ci.png")
    plt.close(fig)


# 2) 실물 과제별 성공률 + Wilson 95% (n = 20)
def fig_real():
    tasks = ["R1", "R2", "R3", "R4", "R5"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    offs = {"Vision": -0.24, "TacRGB": 0.0, "TacFF": 0.24}
    for m in ("Vision", "TacRGB", "TacFF"):
        for i, t in enumerate(tasks):
            k = R["T2_real_wilson"][m][t]["k"]
            lo, hi = [v / 100 for v in R["T2_real_wilson"][m][t]["wilson"]]
            x = i + offs[m]
            ax.plot([x, x], [lo * 100, hi * 100], color=C[m], lw=1.2, alpha=0.6)
            ax.plot(x, 100 * k / 20, "o", color=C[m], ms=6, label=m if i == 0 else None)
    ax.set_xticks(range(5), tasks)
    ax.set_ylim(-3, 103)
    ax.set_ylabel("성공률 (%)", color=INK, fontsize=10)
    ax.set_title("실물 과제별 성공률 (20회, 막대는 Wilson 95% 구간): 구간이 서로 많이 겹친다", fontsize=9, color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "real_ci.png")
    plt.close(fig)


# 3) 토이 A: 표본 수에 따른 정확도 (평균 ± 표준오차, 40 seed)
def fig_concat():
    ns = [30, 60, 120, 250, 500, 1000]
    style_map = [("oracle_z", "잠재 z 오라클", "#55544f", "--"), ("V", "Vision (V)", C["Vision"], "-"),
                 ("V+T_std", "V + 촉각 RGB (표준화)", C["TacRGB"], "-"),
                 ("V+F_std", "V + 힘장 (표준화)", C["TacFF"], "-"),
                 ("V+T+F_std", "V + 촉각 RGB + 힘장", "#333333", ":")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    for key, lbl, col, ls in style_map:
        mean = [T1["E1_sample_size"][key][str(n)]["mean"] for n in ns]
        se = [T1["E1_sample_size"][key][str(n)]["sd"] / math.sqrt(40) for n in ns]
        ax.errorbar(ns, mean, yerr=se, color=col, ls=ls, marker="o", ms=3.5, lw=1.4, capsize=2, label=lbl)
    ax.set_xscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("학습 표본 수 n (로그 눈금)", color=INK, fontsize=10)
    ax.set_ylabel("무잡음 라벨 정확도", color=INK, fontsize=10)
    ax.set_title("토이 A: 중복 모달리티를 이어 붙이면 작은 n에서 손해 (평균 ± SE, 40 seed)", fontsize=9, color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "concat_toy.png")
    plt.close(fig)


# 4) 토이 B: 원시 영상 대 PCA-2 대 오라클 (s = 잡음 표준편차, 3 패널)
def fig_lowdim():
    ns = [10, 20, 40, 80, 160, 320, 640]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), dpi=200, sharey=True)
    for ax, s in zip(axes, [0.5, 1.0, 2.0]):
        rows = T2["results"][f"s={s}"]
        for key, lbl, col, ls in [("oracle_z", "z 오라클", "#55544f", "--"),
                                  ("pca2", "PCA-2", C["TacFF"], "-"),
                                  ("raw_pixels", "원시 영상", C["TacRGB"], "-")]:
            mean = [rows[key][str(n)]["mean"] for n in ns]
            se = [rows[key][str(n)]["sd"] / math.sqrt(30) for n in ns]
            ax.errorbar(ns, mean, yerr=se, color=col, ls=ls, marker="o", ms=3, lw=1.3, capsize=1.5, label=lbl)
        ax.set_xscale("log")
        ax.set_xticks([10, 40, 160, 640], ["10", "40", "160", "640"])
        ax.set_title(f"잡음 s = {s}  (PCA 정렬 {rows['pca2_subspace_alignment_n2000']:.2f})", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("n", color=INK, fontsize=9)
        style(ax)
    axes[0].set_ylabel("정확도", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("토이 B: 2차원 요약(PCA-2)은 잡음이 신호보다 크면 신호 부분공간을 못 찾는다", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "lowdim_toy.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_sim_gap()
    fig_real()
    fig_concat()
    fig_lowdim()
    print("wrote figures to", FIG)
