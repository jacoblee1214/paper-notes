"""우리가 그린 그래프 두 장을 report/figures/ 에 만든다. 논문 그림은 쓰지 않는다.

실행: python3 analysis/plot_figures.py
  own_seq_ci.png    : 종이접기 7단계 순차 성공률과 Wilson 95% 구간 (사례 단위 n=150 가정)
  own_calvin_check.png : 표 2 Avg. Len. 보고값과 단계 합 비교 (V1)
한글 폰트는 Linux Noto CJK, macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recompute as rc  # noqa: E402  (표 값과 Wilson 함수 재사용)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"


def pick_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleSDGothicNeo", "NanumGothic"):
        if cand in names:
            return cand
    return "DejaVu Sans"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["font.family"] = pick_font()
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 그림 1: 순차 성공률 + Wilson 구간 (사례 단위 n=150)
    series = [
        ("VideoWorld 2 (Craft)", "#1f5fae", "-"),
        ("VideoWorld (Craft)", "#8a8a85", "--"),
        ("Wan 2.2 14B (Craft-text)", "#c0692b", ":"),
        ("VideoWorld 2 (OpenX&Craft)", "#0e7c6b", "-."),
        ("CoLA (OpenX&Craft)", "#8a8a85", "--"),
    ]
    steps = list(range(1, 8))
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for name, color, ls in series:
        vals = rc.T1[name]
        lo = [rc.wilson(v, rc.N_A)[0] for v in vals]
        hi = [rc.wilson(v, rc.N_A)[1] for v in vals]
        yerr = [[max(0.0, v - l) for v, l in zip(vals, lo)], [max(0.0, h - v) for v, h in zip(vals, hi)]]
        label = name.replace("(Craft-text)", "(텍스트)").replace("(OpenX&Craft)", "(OpenX+Craft)")
        ax.errorbar(steps, vals, yerr=yerr, color=color, ls=ls, marker="o", ms=3.5, lw=1.6,
                    capsize=2, elinewidth=0.8, label=label, alpha=0.95)
    ax.set_xlabel("단계 (종이접기 7단계)")
    ax.set_ylabel("순차 성공률 (%)")
    ax.set_xticks(steps)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.2, frameon=False, loc="lower left")
    ax.set_title("표 1 재계산: 순차 성공률과 Wilson 95% 구간 (사례 단위 n=150 가정)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_seq_ci.png")
    plt.close(fig)

    # 그림 2: CALVIN Avg. Len. 보고값과 단계 합 (V1)
    names = list(rc.T2.keys())
    reported = [rc.T2[k][1] for k in names]
    sums = [round(sum(rc.T2[k][0]) / 100, 3) for k in names]
    import numpy as np
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=160)
    ax.bar(x - 0.2, reported, width=0.4, color="#1f5fae", label="보고값")
    ax.bar(x + 0.2, sums, width=0.4, color="#b8b6ae", label="단계 합 (재계산)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" ", 1)[0] for n in names])
    ax.set_xlabel("표 2 행 번호 (1–7)")
    ax.set_ylabel("평균 길이 (Avg. Len.)")
    ax.set_ylim(0, 3.2)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("표 2 검사: 보고된 평균 길이와 단계 합 (V1). 행 3에서 차이가 가장 큼", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_calvin_check.png")
    plt.close(fig)
    print("wrote", FIG / "own_seq_ci.png", FIG / "own_calvin_check.png")


if __name__ == "__main__":
    main()
