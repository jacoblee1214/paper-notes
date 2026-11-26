"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (recompute_results.json, toy_seen_fraction_results.json 사용).

own_fdr_heatmap.png    : 표 1의 FDR 60칸과 동점 구간(|FDR| <= 5, 분모 400의 표준오차 5점)
own_fdh_stability.png  : 표집 잡음만 고려한 FDH 재현 비율과 color 1위 비율 (분모 400 가정)
own_table3_ci.png      : 표 3(실로봇) 성공률과 Wilson 95 % 구간 (시행 수 48 가정은 표에 적힘)
own_seen_fraction.png  : 공개 코드 기준 6x6 평가에서 훈련 칸 비율, 합성 과제의 전체 격자 대 미훈련 칸 성공률
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
MODELS = ["pi0.5", "pi0", "OpenVLA-oft", "GR00T-N1.7", "XVLA", "Genie-Envisioner"]
SHORT = ["π0.5", "π0", "OFT", "GR00T", "XVLA", "Genie"]


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#d9822b"
REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
sys_path = str(ANA)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("rc", ANA / "recompute.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def plot_heatmap():
    pairs = list(rc.T1.keys())
    M = np.array([rc.T1[p] for p in pairs])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 35 else INK, fontweight="bold" if abs(v) <= 5 else "normal")
    ax.set_xticks(range(6))
    ax.set_xticklabels(SHORT, fontsize=9)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"({a}, {b})" for a, b in pairs], fontsize=8.5)
    fig.colorbar(im, ax=ax, label="FDR (%) 양수 = 앞 인자 우세", shrink=0.85)
    ax.set_title("표 1 FDR (굵은 글자: |FDR| ≤ 5, 분모 400이면 표준오차 약 5점)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_fdr_heatmap.png", dpi=200)
    plt.close(fig)


def plot_stability():
    v = REC["V9_fdr_sampling_noise"]["models"]
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(x - 0.19, [100 * v[m]["p_same_fdh"] for m in MODELS], 0.38, color=ACC, label="보고된 FDH 문자열과 같을 비율")
    ax.bar(x + 0.19, [100 * v[m]["p_color_strict_top"] for m in MODELS], 0.38, color=ORG, label="color가 엄격히 1위일 비율")
    for xi, m in zip(x, MODELS):
        ax.text(xi - 0.19, 100 * v[m]["p_same_fdh"] + 1.5, f"{100 * v[m]['p_same_fdh']:.0f}", ha="center", fontsize=8)
        ax.text(xi + 0.19, 100 * v[m]["p_color_strict_top"] + 1.5, f"{100 * v[m]['p_color_strict_top']:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)
    ax.set_ylabel("%")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", ncol=1)
    ax.set_title("롤아웃 표집 잡음만 넣은 FDH 안정성 (표 1을 참값, N = 400, 4,000회 모의)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_fdh_stability.png", dpi=200)
    plt.close(fig)


def plot_table3():
    cols = list(rc.T3.keys())
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    x = np.arange(len(cols))
    w = 0.26
    for off, meth, col in ((-w, "L", GREY), (0, "Random", ORG), (w, "V", ACC)):
        k = [rc.T3[c][meth] for c in cols]
        val = np.array([100 * a / 48 for a in k])
        lo = np.array([rc.wilson(a, 48)[0] for a in k])
        hi = np.array([rc.wilson(a, 48)[1] for a in k])
        ax.bar(x + off, val, w, color=col, alpha=0.9, label=meth if meth != "V" else "V (제안)")
        ax.errorbar(x + off, val, yerr=[val - lo, hi - val], fmt="none", ecolor=INK, capsize=2.5, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("표 3 실로봇: 열마다 48회, Wilson 95 % 구간 (학습 변동 미포함)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_table3_ci.png", dpi=200)
    plt.close(fig)


def plot_seen():
    toy = json.loads((ANA / "toy_seen_fraction_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    fs = [6, 12, 18]
    strat = [("V", ACC), ("L", GREY), ("Random", ORG)]
    x = np.arange(3)
    for k, (s, col) in enumerate(strat):
        axes[0].bar(x + (k - 1) * 0.26, [100 * toy[f"f{f}/{s}"]["seen_fraction"] for f in fs], 0.26, color=col, label=s)
        axes[1].bar(x + (k - 1) * 0.26, [toy[f"f{f}/{s}"]["overall_pct"] for f in fs], 0.26, color=col, alpha=0.5)
        axes[1].bar(x + (k - 1) * 0.26, [toy[f"f{f}/{s}"]["unseen_only_pct"] for f in fs], 0.26, color=col)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"f = {f}" for f in fs])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("6×6 평가 격자에서 훈련 칸 비율 (%)")
    axes[0].set_title("공개 코드의 지지집합 크기로 계산", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("합성 과제 성공률 (%)")
    axes[1].set_title("합성 과제: 연한 막대 = 전체 격자, 진한 막대 = 미훈련 칸만", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_seen_fraction.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_heatmap()
    plot_stability()
    plot_table3()
    plot_seen()
    print("wrote figures to", FIG)
