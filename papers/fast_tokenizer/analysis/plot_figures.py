"""우리가 만든 그래프 두 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_tokens.png : Table I의 1초 청크 토큰 수 (naive 대 FAST), 논문 표 값을 그대로 옮김
own_toy_spectrum.png  : 토이 T1의 0 아닌 DCT 계수 수와 naive 토큰 수 (95% 구간), 두 신호 종류
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#c4622d"


def fig_table1():
    rows = [("BridgeV2\n(7 dim, 5 Hz)", 35, 20), ("DROID\n(7 dim, 15 Hz)", 105, 29),
            ("Bussing\n(7 dim, 20 Hz)", 140, 28), ("Shirt Fold\n(14 dim, 50 Hz)", 700, 53)]
    labels = [r[0] for r in rows]
    naive = np.array([r[1] for r in rows])
    fast = np.array([r[2] for r in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - 0.2, naive, 0.4, color=GREY, label="naive binning (D × Hz)")
    ax.bar(x + 0.2, fast, 0.4, color=ACC, label="FAST (논문 Table I)")
    for i in range(len(rows)):
        ax.text(x[i] + 0.2, fast[i] + 10, f"{naive[i] / fast[i]:.1f}×", ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("1초 청크당 토큰 수")
    ax.set_title("논문 Table I: 토큰 수와 압축률 (위 숫자는 재계산 압축률)", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_tokens.png", dpi=200)
    plt.close(fig)


def fig_toy():
    res = json.loads((ANA / "toy_fast_spectrum_results.json").read_text(encoding="utf-8"))
    rates = res["rates_hz"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=False)
    for ax, (sig, label) in zip(axes, [("sinusoid", "무작위 사인 합"), ("cubic_spline", "4점 3차 스플라인")]):
        d = res["signals"][sig]
        x = np.array(rates, dtype=float)
        nz = np.array([d[str(r)]["nonzero_dct_mean"] for r in rates])
        hw = np.array([d[str(r)]["nonzero_dct_ci95_halfwidth"] for r in rates])
        naive = np.array([d[str(r)]["naive_tokens_D_x_hz"] for r in rates])
        ax.plot(x, naive, "-o", color=GREY, ms=3, label="naive (D × Hz)")
        ax.errorbar(x, nz, yerr=hw, fmt="-o", color=ACC, ms=3, capsize=2, label="0 아닌 DCT 계수 (BPE 전)")
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel("샘플 수 / 1초 (Hz)", fontsize=8)
        ax.set_xticks(rates)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("1초 청크당 개수", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("토이 T1: 주파수가 높아질 때 BPE 이전 계수 수 (300 궤적, 95% 구간)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_spectrum.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_table1()
    fig_toy()
    print("wrote", FIG)
