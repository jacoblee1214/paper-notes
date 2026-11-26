"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (recompute_results.json 사용).

own_table1_ci.png     : 표 1 평균 성공률과 Wilson 95% 구간 (운동 80회, 지시 70회 가정)
own_as_vs_sr.png      : 조합 과제에서 AS, 독립 곱 기준선, 실제 SR (두 플랫폼 합산)
own_fig3_vs_table1.png: 그림 3 막대에서 읽은 성공 수와 표 1 성공 수 비교
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
MODELS = ["Pi0.5", "Motus", "LingBot-VLA", "GROOT N1.6", "SmolVLA"]


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#d9822b"
REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def plot_table1():
    ci = REC["V4_wilson"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
    w = 0.38
    for ax, plat, title in zip(axes, ("Franka", "Cobot"), ("Franka Panda (단일 팔)", "Cobot Magic (양팔)")):
        x = np.arange(len(MODELS))
        for off, key, col, lab in ((-w / 2, "motor", ACC, "운동 (n=80)"), (w / 2, "instr", ORG, "지시 (n=70)")):
            k = [ci[f"{plat}/{m}/{key}"]["k"] for m in MODELS]
            n = [ci[f"{plat}/{m}/{key}"]["n"] for m in MODELS]
            val = np.array([100 * a / b for a, b in zip(k, n)])
            lo = np.array([ci[f"{plat}/{m}/{key}"]["wilson"][0] for m in MODELS])
            hi = np.array([ci[f"{plat}/{m}/{key}"]["wilson"][1] for m in MODELS])
            ax.bar(x + off, val, w, color=col, alpha=0.85, label=lab)
            ax.errorbar(x + off, val, yerr=[val - lo, hi - val], fmt="none", ecolor=INK, capsize=2.5, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=20, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("성공률 SR (%)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("표 1 재계산: 평균 SR과 Wilson 95% 구간 (시행 수 가정: 과제당 10회)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png", dpi=200)
    plt.close(fig)


def plot_as_vs_sr():
    v7 = REC["V7_product_baseline"]
    ci = REC["V4_wilson"]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(MODELS))
    w = 0.26
    asv = [v7[m]["AS_mean"] for m in MODELS]
    prod = [v7[m]["independent_product"] for m in MODELS]
    plo = [v7[m]["product_interval_95"][0] for m in MODELS]
    phi = [v7[m]["product_interval_95"][1] for m in MODELS]
    obs, olo, ohi = [], [], []
    for m in MODELS:
        k = ci[f"Franka/{m}/comp"]["k"] + ci[f"Cobot/{m}/comp"]["k"]
        n = 240
        p = k / n
        z = 1.959964
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        obs.append(100 * p)
        olo.append(100 * (c - h))
        ohi.append(100 * (c + h))
    ax.bar(x - w, asv, w, color=GREY, label="AS (원자 PSR 평균, 표 2)")
    ax.bar(x, prod, w, color=ORG, alpha=0.85, label="원자 SR의 독립 곱 (그림 3 막대, 구간은 Beta 표본)")
    ax.errorbar(x, prod, yerr=[np.array(prod) - plo, np.array(phi) - prod], fmt="none", ecolor=INK, capsize=2.5, lw=1)
    ax.bar(x + w, obs, w, color=ACC, alpha=0.9, label="조합 과제 SR (표 2, n=240 가정, Wilson)")
    ax.errorbar(x + w, obs, yerr=[np.array(obs) - olo, np.array(ohi) - obs], fmt="none", ecolor=INK, capsize=2.5, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("AS와 SR은 다른 척도: 독립 곱 기준선은 AS보다 SR에 훨씬 가깝다 (두 플랫폼 평균)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_as_vs_sr.png", dpi=200)
    plt.close(fig)


def plot_fig3_vs_t1():
    v8 = REC["V8_fig3_vs_table1"]["per_model"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, key, tot, title in ((axes[0], "motor", 160, "운동 원자 (160회)"), (axes[1], "instr", 140, "지시 원자 (140회)")):
        x = np.arange(len(MODELS))
        t1 = [v8[m][f"table1_{key}_succ_of_{tot}"] for m in MODELS]
        f3 = [v8[m][f"fig3_{key}_succ_of_{tot}"] for m in MODELS]
        ax.bar(x - 0.19, t1, 0.38, color=ACC, alpha=0.9, label="표 1에서 환산")
        ax.bar(x + 0.19, f3, 0.38, color=ORG, alpha=0.9, label="그림 3 막대에서 읽음 (그림 판독 값)")
        for xi, a, b in zip(x, t1, f3):
            ax.text(xi - 0.19, a + 1.5, f"{a:.0f}", ha="center", fontsize=7.5)
            ax.text(xi + 0.19, b + 1.5, f"{b:.0f}", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=20, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("성공 수 (두 플랫폼 합)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("그림 3 막대의 합과 표 1의 합: Pi0.5와 GROOT는 일치, 나머지 세 모델은 차이", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig3_vs_table1.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_as_vs_sr()
    plot_fig3_vs_t1()
    print("wrote figures to", FIG)
