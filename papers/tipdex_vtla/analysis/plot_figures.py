"""재계산 결과를 그래프 세 장으로 그린다 (우리가 그린 그래프, 공개 가능).

실행: python3 analysis/plot_figures.py
  -> report/figures/own_real_ci.png, own_sim_ci.png, own_ablation_ci.png
입력: analysis/recompute_results.json (recompute.py가 만든 파일)
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


def setup_font():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def ci_bar(ax, y, mean, lo, hi, color, label=None):
    ax.errorbar([mean], [y], xerr=[[mean - lo], [hi - mean]], fmt="o", color=color,
                capsize=3, label=label, markersize=5)


def real_ci():
    t2 = RES["table2"]
    methods = ["ACT", "π0.5", "T-Rex", "N0-VTLA", "TipDex-VTLA"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, cond, title in [(axes[0], "normal", "표준 조명 (N)"), (axes[1], "unseen", "처음 보는 조명 (U)")]:
        for i, m in enumerate(methods):
            r = t2[cond][m]
            col = "#1f5fae" if m == "TipDex-VTLA" else "#777777"
            ci_bar(ax, i, r["mean_pct_recomputed"], r["wilson95_pct"][0], r["wilson95_pct"][1], col)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods if cond == "normal" else [""] * len(methods))
        ax.set_xlim(0, 100)
        ax.set_xlabel("과제 성공률 (%), 80회 pooled, Wilson 95% 구간")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png")
    plt.close(fig)


def sim_ci():
    s = RES["table1_pairwise"]
    names = ["π0.5 (Without Tactile)", "T-Rex", "N0-VTLA", "TipDex-VTLA"]
    # 시뮬 행: 800회 pooled의 평균과 Wilson 구간을 직접 계산
    from math import sqrt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sim_counts = {
        "π0.5 (Without Tactile)": 331, "T-Rex": 650, "N0-VTLA": 665, "TipDex-VTLA": 679,
    }
    z = 1.959964
    for i, m in enumerate(names):
        k, n = sim_counts[m], 800
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        col = "#1f5fae" if m == "TipDex-VTLA" else "#777777"
        ci_bar(ax, i, 100 * p, 100 * (c - h), 100 * (c + h), col)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("UniVTAC 8 과제 pooled 성공률 (%), 800회, Wilson 95% 구간")
    ax.set_title("시뮬레이션: 저자가 같은 프로토콜로 돌린 4개 방법 (공개 결과 행 제외)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png")
    plt.close(fig)
    return s


def ablation_ci():
    t3 = RES["table3"]
    names = ["Full", "w/o VASE", "w/o Tactile Verification", "w/o Gate Head"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    offsets = {"normal": 0.18, "unseen": -0.18}
    for cond, label, col in [("normal", "표준 조명 (N)", "#9aa9c4"), ("unseen", "처음 보는 조명 (U)", "#1f5fae")]:
        for i, m in enumerate(names):
            r = t3[cond][m]
            ax.errorbar([r["mean_pct_recomputed"]], [i + offsets[cond]],
                        xerr=[[r["mean_pct_recomputed"] - r["wilson95_pct"][0]],
                              [r["wilson95_pct"][1] - r["mean_pct_recomputed"]]],
                        fmt="o", color=col, capsize=3, markersize=5,
                        label=label if i == 0 else None)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("실세계 과제 성공률 (%), 조건당 80회, Wilson 95% 구간")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2, fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("구성요소 절제 (논문 표 3 재계산)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_ci.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    setup_font()
    real_ci()
    sim_ci()
    ablation_ci()
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
