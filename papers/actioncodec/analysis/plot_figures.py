"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_libero_ci.png : 표 1(c) LIBERO 평균 성공률과 Wilson 95% 구간 (스위트당 500회, 합 2,000회 가정)
own_fig3_sweep.png: 그림 3 판독 값(LIBERO-Goal, 20k 스텝, 500회 가정)의 토큰 수 n과 어휘 크기 S 스윕
own_so100.png     : 그림 7 판독 값(SO100, 과제 10 x 20회 = 200회)의 Pick과 Place 성공 횟수와 Wilson 구간
own_toy_or.png    : 토이 T1의 OR과 L1 대 양자화 단계 수 S
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c8641e"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_libero():
    rows = [("String (T=8)", 49.6), ("Binning (T=8)", 53.4), ("VQVLA's", 60.5), ("MiniVLA's", 82.6), ("FAST", 90.6),
            ("AC 256M", 92.3), ("AC 500M", 92.5), ("KI", 94.3), ("AC 2.2B (AR)", 95.5), ("PD", 95.6), ("BAR", 97.4)]
    N = 2000
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = np.arange(len(rows))
    for i, (name, p) in enumerate(rows):
        k = round(p / 100 * N)
        lo, hi = wilson(k, N)
        col = GREY if name in ("String (T=8)", "Binning (T=8)", "VQVLA's", "MiniVLA's", "FAST") else ACC
        ax.barh(i, p, color=col, height=0.62)
        ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, capsize=2.5, lw=1)
        ax.text(p + 0.4 if False else 101.2, i, f"{p:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(40, 104)
    ax.set_xlabel("LIBERO 4개 스위트 평균 성공률 (%), 막대 끝의 선은 Wilson 95% 구간 (n = 2,000 가정)", fontsize=8)
    ax.axvline(90.6, color=ORG, lw=0.8, ls="--")
    ax.text(90.4, len(rows) - 0.35, "FAST 90.6", color=ORG, fontsize=7.5, ha="right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_libero_ci.png", dpi=200)
    plt.close(fig)


def plot_fig3():
    n_sweep = [("n=8", 73.4, 0.047), ("n=16", 67.2, 0.029), ("n=32", 60.0, 0.020), ("n=64", 29.2, 0.016)]
    s_sweep = [("S=512", 77.0, 0.031), ("S=1024", 75.6, 0.024), ("S=2048", 67.2, 0.029), ("S=4096", 65.8, 0.029)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for ax, data, title, chosen in ((axes[0], n_sweep, "토큰 수 n (S = 2048)", "n=16"),
                                    (axes[1], s_sweep, "어휘 크기 S (n = 16)", "S=2048")):
        for i, (name, p, l1) in enumerate(data):
            k = round(p / 100 * 500)
            lo, hi = wilson(k, 500)
            ax.bar(i, p, color=ORG if name == chosen else ACC, width=0.6)
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, capsize=3, lw=1)
            ax.text(i, 3, f"L1\n{l1:.3f}", ha="center", color="white", fontsize=7.5)
            ax.text(i, hi + 1.2, f"{p:.1f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([d[0] for d in data], fontsize=8)
        ax.set_title(title, fontsize=9)
        style(ax)
    axes[0].set_ylabel("LIBERO-Goal 성공률 (%), 20k 스텝", fontsize=8)
    fig.text(0.5, 0.005, "논문 그림 3에 인쇄된 값을 읽은 것(그림 판독 값), 500회 가정, 주황 막대는 논문이 고른 설정(OR 40% 고정)", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "own_fig3_sweep.png", dpi=200)
    plt.close(fig)


def plot_so100():
    names = ["ActionCodec", "w/o CT", "FAST", "pi0"]
    pick = [148, 142, 118, 121]
    place = [71, 61, 49, 46]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    w = 0.36
    for j, (lab, vals, col) in enumerate((("Pick", pick, ACC), ("Place", place, ORG))):
        for i, k in enumerate(vals):
            lo, hi = wilson(k, 200)
            p = 100 * k / 200
            ax.bar(i + (j - 0.5) * w, p, width=w, color=col, label=lab if i == 0 else None)
            ax.errorbar(i + (j - 0.5) * w, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, capsize=2.5, lw=1)
            ax.text(i + (j - 0.5) * w, hi + 1.5, f"{k}", ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("성공률 (%), Wilson 95% 구간 (n = 200)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("SO100-ShapeSorter (그림 7 판독 값의 합, 막대 위는 성공 횟수)", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_so100.png", dpi=200)
    plt.close(fig)


def plot_toy():
    res = json.loads((ANA / "toy_or_tradeoff_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    cols = [ACC, ORG]
    for r, col in zip(res, cols):
        rows = [x for x in r["pca_quant"] if x["n_tokens"] == 16]
        S = [x["S"] for x in rows]
        axes[0].plot(S, [100 * x["OR"] for x in rows], "o-", color=col, label=f"측정 (필터 폭 {r['sigma_t']})")
        axes[0].plot(S, [100 * x["OR_pred_from_step"] for x in rows], "x", color=INK, ms=6)
        axes[1].plot(S, [x["L1"] for x in rows], "o-", color=col, label=f"필터 폭 {r['sigma_t']}")
    axes[0].set_xscale("log", base=2)
    axes[1].set_xscale("log", base=2)
    axes[0].set_xlabel("성분당 양자화 단계 수 S", fontsize=8)
    axes[1].set_xlabel("성분당 양자화 단계 수 S", fontsize=8)
    axes[0].set_ylabel("OR (%)", fontsize=8)
    axes[1].set_ylabel("L1 재구성 오차 (신호 단위)", fontsize=8)
    axes[0].set_title("OR: 측정(원)과 간격 기반 예측(x)", fontsize=9)
    axes[1].set_title("재구성 오차", fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False)
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_or.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_libero()
    plot_fig3()
    plot_so100()
    plot_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
