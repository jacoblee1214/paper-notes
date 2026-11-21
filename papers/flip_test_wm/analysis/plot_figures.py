"""우리가 만든 그래프 두 장을 report/figures/ 에 저장한다. 논문 그림 조각은 만들지 않는다.

  own_pair_ci.png        모델별 쌍 정확도(시뮬·실사)와 Wilson 95% 구간, 우연 수준 25%
  own_readout_ceiling.png 판독 정확도 a에서 독립 오류를 가정한 쌍 정확도 상한 a^2와 본문 값 대조

입력: 논문 Table 3의 개수(recompute.py와 같은 값), analysis/recompute_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

INK, MUTED, ACC, ACC2, WARN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a"
N = 150


def wilson(k, n=N, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def pair_ci():
    pm = RES["per_model"]
    names = sorted(pm, key=lambda m: -pm[m]["overall_pct"])
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)
    y = np.arange(len(names))
    h = 0.38
    for off, key, k_key, color, label in ((-h / 2, "sim_pair_acc_pct", "sim_solved", ACC2, "시뮬레이션"),
                                          (h / 2, "real_pair_acc_pct", "real_solved", ACC, "실사")):
        vals = [pm[m][key] for m in names]
        ks = [pm[m][k_key] for m in names]
        if label == "실사":
            ks = [RES["per_model"][m]["real_solved"] for m in names]
        else:
            ks = [RES["per_model"][m]["sim_solved"] for m in names]
        cis = [wilson(k) for k in ks]
        lo = [v - c[0] for v, c in zip(vals, cis)]
        hi = [c[1] - v for v, c in zip(vals, cis)]
        ax.barh(y + off, vals, height=h, color=color, alpha=0.9, label=label)
        ax.errorbar(vals, y + off, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.axvline(25, color=WARN, ls="--", lw=1)
    ax.text(25.4, len(names) - 0.45, "우연 25%", color=WARN, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("쌍 정확도 (%), 오차막대 = Wilson 95% (n = 150 쌍)", fontsize=9)
    ax.set_xlim(0, 60)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("논문 Table 3에서 다시 셈: 쌍 정확도와 구간", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_pair_ci.png")
    plt.close(fig)


def readout_ceiling():
    a = np.linspace(0.5, 1.0, 200)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    ax.plot(a * 100, (a ** 2) * 100, color=ACC, lw=2, label="독립 오류 가정: 쌍 정확도 = a²")
    ax.axhline(86.7, color=ACC2, ls="--", lw=1.2)
    ax.text(51, 88.2, "본문 told-outcome 실사 Omni 86.7% (판독 단일 판정기)", fontsize=8, color=ACC2)
    marks = [
        (79, "사람 라벨 일치 79% (Gemini 3.6, 본문 §3.3)", (52, 42)),
        (85.8, "두 판정기 일치 85.8% (6600 영상)", (52, 62)),
    ]
    for x, label, (tx, ty) in marks:
        y = (x / 100) ** 2 * 100
        ax.plot([x], [y], "o", color=INK, ms=4)
        ax.annotate(f"{label}\n→ 쌍 정확도 {y:.1f}%", (x, y), xytext=(tx, ty), fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel("영상 하나의 판독 정확도 a (%)", fontsize=9)
    ax.set_ylabel("쌍 정확도 상한 (%)", fontsize=9)
    ax.set_xlim(50, 100)
    ax.set_ylim(20, 100)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("판독 오류가 쌍 정확도를 얼마나 깎을 수 있나 (재계산)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_readout_ceiling.png")
    plt.close(fig)


if __name__ == "__main__":
    pair_ci()
    readout_ceiling()
    print("wrote", FIG / "own_pair_ci.png", FIG / "own_readout_ceiling.png")
