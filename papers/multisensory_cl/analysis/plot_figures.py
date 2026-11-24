"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_scores_ci.png : 표 3·4의 과제별 점수 비율과 Wilson 95% 구간(점수/시행 수를 비율로 본 근사)
own_pooled_diff.png: MuSe와 각 기준선의 합산(n = 40) 차이와 Newcombe 95% 구간
own_f_t_error.png  : 그림 4와 표 2의 F/T 예측 오차(그림 판독 값)와 MuSe 대비 상대 증가
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; analysis/recompute_results.json 필요)
한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
import math
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
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#9a9890", "#b5472f"

N = {"wipe": 15, "peg": 15, "pp": 10}
TASK = {"wipe": "닦기", "peg": "페그", "pp": "집어 놓기"}
FWD = {"No F/T": {"wipe": 5, "peg": 9, "pp": 7.5}, "No Pretrain": {"wipe": 8, "peg": 6, "pp": 5},
       "Diffusion Policy": {"wipe": 10, "peg": 10, "pp": 6}, "MuSe": {"wipe": 11.5, "peg": 13, "pp": 7.5}}
BWD = {"Pretraining Only": {"wipe": 8.5, "peg": 8, "pp": 5.5}, "No ER": {"wipe": 0.5, "peg": 2, "pp": 5.5},
       "Diffusion Policy": {"wipe": 6.5, "peg": 5, "pp": 6}, "MuSe": {"wipe": 12.5, "peg": 10, "pp": 6.5}}
COLORS = {"MuSe": RED, "No F/T": "#3a3fb5", "No Pretrain": "#8a8577", "Diffusion Policy": "#6aa3d9",
          "Pretraining Only": "#8a8ad4", "No ER": "#d9b8a8"}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_scores():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
    for ax, tab, title in ((axes[0], BWD, "후진 전이 (사전학습 과제, 표 4)"), (axes[1], FWD, "전진 전이 (미세조정 과제, 표 3)")):
        methods = list(tab)
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            xs, ys, lo, hi = [], [], [], []
            for j, t in enumerate(("wipe", "peg", "pp")):
                c = tab[m][t]
                p = 100 * c / N[t]
                l, h = wilson(c, N[t])
                xs.append(j + (i - (len(methods) - 1) / 2) * w)
                ys.append(p)
                lo.append(p - l)
                hi.append(h - p)
            ax.bar(xs, ys, width=w * 0.92, color=COLORS[m], label=m)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{TASK[t]} (n = {N[t]})" for t in ("wipe", "peg", "pp")])
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_ylim(0, 125)
        ax.set_yticks(range(0, 101, 20))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)
    axes[0].set_ylabel("점수 / 시행 수 (%)")
    fig.text(0.5, 0.005, "막대: 논문 점수/n. 오차 막대: Wilson 95%(점수를 비율로 본 근사, 시행 독립 가정)", ha="center", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_scores_ci.png", dpi=200)
    plt.close(fig)


def plot_pooled():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    rows = rec["V11_pooled_fisher_n40"]["tests"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    labels = []
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = r["diff_ci_pp"]
        color = RED if lo > 0 else GREY
        ax.plot([lo, hi], [y, y], color=color, lw=2.2)
        ax.plot(r["diff_pp"], y, "o", color=color, ms=6)
        labels.append(("전진" if r["set"] == "fwd" else "후진") + f": MuSe - {r['vs']}")
        ax.text(hi + 1.5, y, f"{r['diff_pp']:+.1f} [{lo:.1f}, {hi:.1f}]", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1], fontsize=8.5)
    ax.set_xlabel("합산 성공률 차이 (%p, 3과제 합 n = 40, Newcombe 95%)")
    ax.set_xlim(-10, 85)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_pooled_diff.png", dpi=200)
    plt.close(fig)


def plot_ft_error():
    vals = {"MuSe": 8.42, "No VP": 9.90, "No ER": 12.38, "No Early": 12.06, "No Late": 11.52,
            "Linear": 11.80, "Additive": 11.80, "Diffusion Policy": 18.27}
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    names = list(vals)
    cols = [RED if n == "MuSe" else ("#6aa3d9" if n == "Diffusion Policy" else "#c9b9ae") for n in names]
    ax.bar(names, [vals[n] for n in names], color=cols)
    for i, n in enumerate(names):
        rel = "" if n == "MuSe" else f"\n(+{100 * (vals[n] / 8.42 - 1):.0f}%)"
        ax.text(i, vals[n] + 0.3, f"{vals[n]:.2f}{rel}", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("F/T 예측 L2 오차 (사전학습 과제)")
    ax.set_ylim(0, 23)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.005, "그림 4 판독 값과 표 2 값을 다시 그림. 영 예측·직전 값 유지 기준선과 분산은 논문에 없음", ha="center", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "own_f_t_error.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_scores()
    plot_pooled()
    plot_ft_error()
    print("wrote 3 figures in", FIG)
