"""우리가 그린 그래프 두 장 (논문 그림은 쓰지 않는다: 라이선스 CC0, CC BY 아님).

  report/figures/own_fig2_overall.png : Fig. 2 세 과제 합(60 시행)의 조건별 전체 성공률과 Wilson 95% 구간 (읽은 값 기준)
  report/figures/own_table1_ci.png    : Table 1 보상 정확도 (n = 40)와 Wilson 95% 구간

실행: python3 analysis/plot_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute import READ, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for path in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False

COND_LABEL = {"Base": "기본 정책", "VL8": "시각 8단계", "VL16": "시각 16단계", "TS": "촉각 샘플링",
              "TG": "촉각 가이던스", "Naive": "단순 결합", "Ours": "ViTaL"}
COLOR = {"Base": "#9a948c", "VL8": "#a9c4de", "VL16": "#4f86b8", "TS": "#b5d9b5", "TG": "#5e9e6e",
         "Naive": "#eacb9f", "Ours": "#e07b39"}


def fig2_overall():
    keys = list(COND_LABEL)
    counts = {c: sum(round(READ[t][c][0] * 20 / 100) for t in READ) for c in keys}
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=200)
    for i, c in enumerate(keys):
        k = counts[c]
        p = k / 60
        lo, hi = wilson(k, 60)
        ax.bar(i, 100 * p, color=COLOR[c], edgecolor="#333", linewidth=0.5, width=0.7)
        ax.errorbar(i, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="none", ecolor="#222", capsize=3, lw=0.9)
        ax.text(i, 100 * hi + 2, f"{100 * p:.0f}%\n({k}/60)", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([COND_LABEL[c] for c in keys], rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("전체 성공률 (%)", fontsize=9)
    ax.set_title("세 과제 합 전체 성공률 (Fig. 2 읽은 값, n = 60, Wilson 95%)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "own_fig2_overall.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def table1_ci():
    t1 = {
        "시각 GT": [32, 28, 40], "시각 예측": [33, 29, 40],
        "촉각 GT": [34, 28, 32], "촉각 예측": [36, 31, 31],
    }
    tasks = ["Wiping", "Insertion", "Pipette"]
    fig, ax = plt.subplots(figsize=(8.2, 3.9), dpi=200)
    width = 0.2
    colors = ["#4f86b8", "#a9c4de", "#5e9e6e", "#b5d9b5"]
    for j, (name, ks) in enumerate(t1.items()):
        for i, k in enumerate(ks):
            p = k / 40
            lo, hi = wilson(k, 40)
            x = i + (j - 1.5) * width
            ax.bar(x, 100 * p, width=width * 0.92, color=colors[j], edgecolor="#333", linewidth=0.4,
                   label=name if i == 0 else None)
            ax.errorbar(x, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="none", ecolor="#222", capsize=2, lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("선호 순서 정확도 (%)", fontsize=9)
    ax.set_title("Table 1 보상 정확도 (n = 40, Wilson 95%)", fontsize=10)
    ax.legend(fontsize=7, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "own_table1_ci.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for f in (fig2_overall(), table1_ci()):
        print("wrote", f)
