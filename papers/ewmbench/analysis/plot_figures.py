"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다 (matplotlib, CPU).

입력: recompute.py 의 값 (논문 표 2, 그림 6B 라벨, 사람 순위, n = 100 가정).
실행: python3 analysis/plot_figures.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# 한글 폰트: Linux(Noto CJK) → macOS(Apple SD Gothic Neo) 순서로 찾는다.
KOREAN_FONTS = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "Apple SD Gothic Neo"),
]
KO_FAMILY = next((name for path, name in KOREAN_FONTS if os.path.exists(path)), None)
for path, _ in KOREAN_FONTS:
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
        break

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recompute as rc  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "report", "figures")
os.makedirs(OUT, exist_ok=True)

# dataviz 기본 팔레트 (references/palette.md, light). 슬롯 1 파랑, 2 주황, 3 청록, 4 노랑.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad3"

plt.rcParams.update({
    "font.family": [KO_FAMILY or "DejaVu Sans", "DejaVu Sans"],
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9.5,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def overall_ci():
    """단일 모델 Overall과 n = 100 최악 가정의 95% 반폭(0.277)."""
    half = rc.sensitivity(100)["half_width95_single_model"]
    items = sorted(rc.T2.items(), key=lambda kv: kv[1]["overall"], reverse=True)
    names = [k for k, _ in items]
    vals = [v["overall"] for _, v in items]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = list(range(len(names)))[::-1]
    ax.errorbar(vals, y, xerr=half, fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.4,
                capsize=3, markersize=6)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Overall (표 2 값, 세 축 합)")
    ax.set_xlim(2.4, 5.4)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("표 2 Overall과 최악 가정의 95% 반폭 (n = 100)")
    fig.text(0.01, 0.01, "오차 막대는 에피소드별 값이 없어 SD 0.5 가정으로 계산한 상한입니다.",
             fontsize=7.8, color=INK2)
    save(fig, "own_overall_ci.png")


def fig6b_mismatch():
    """그림 6B 라벨과 표 2 Overall의 나란한 막대 (두 계열이므로 범례와 직접 라벨)."""
    names = list(rc.FIG6B_EWM)
    table = [rc.T2[m]["overall"] for m in names]
    fig6 = [rc.FIG6B_EWM[m] for m in names]
    xs = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    b1 = ax.bar([x - w / 2 for x in xs], table, w, color=BLUE, label="표 2 Overall")
    b2 = ax.bar([x + w / 2 for x in xs], fig6, w, color=ORANGE, label="그림 6B 라벨")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.2f}",
                    ha="center", fontsize=8.2, color=INK2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names)
    ax.set_ylabel("점수")
    ax.set_ylim(0, 6.2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_title("같은 순서, 다른 값: 표 2와 그림 6B (EWMBench)")
    diffs = [f"{n} {d:+.2f}" for n, d in zip(names, [f - t for f, t in zip(fig6, table)])]
    fig.text(0.01, 0.01, "차이(그림 − 표): " + ", ".join(diffs), fontsize=7.8, color=INK2)
    save(fig, "own_fig6b_mismatch.png")


def rank_agreement():
    """사람, EWMBench(표 2), VBench(그림 6B)의 순위를 모델별 선으로 잇는다."""
    names = ["LTX_FT", "Kling", "Hailuo", "OpenSora"]
    human = rc.HUMAN_RANK
    ewm = rc.ranks_from_scores({m: rc.T2[m]["overall"] for m in names})
    vb = rc.ranks_from_scores(rc.FIG6B_VBENCH)
    series = [("사람 (그림 6A)", human, INK, "-"),
              ("EWMBench (표 2)", ewm, BLUE, "-"),
              ("VBench (그림 6B)", vb, AQUA, "--")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for m in names:
        pts = [human[m], ewm[m], vb[m]]
        ax.plot([0, 1, 2], pts, color=GRID, linewidth=1.0, zorder=1)
    for idx, (label, ranks, color, style) in enumerate(series):
        ax.scatter([idx] * len(names), [ranks[m] for m in names], s=46, color=color, zorder=3, label=label)
    for m in names:
        ax.annotate(m, (2, vb[m]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=INK2)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["사람 (그림 6A)", "EWMBench (표 2)", "VBench (그림 6B)"])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(4.5, 0.5)
    ax.set_ylabel("순위 (1 = 최고)")
    ax.set_xlim(-0.3, 2.9)
    ax.set_title("순위 일치: EWMBench는 사람과 같고 VBench는 3쌍이 다름")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5, ncol=3, bbox_to_anchor=(0, -0.28))
    fig.text(0.01, 0.01, "4개 모델만 비교. Kendall tau: EWMBench 1.0, VBench 0.0. 완전 일치 우연 확률 0.042.",
             fontsize=7.8, color=INK2)
    save(fig, "own_rank_agreement.png")


if __name__ == "__main__":
    overall_ci()
    fig6b_mismatch()
    rank_agreement()
