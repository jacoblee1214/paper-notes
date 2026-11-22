"""Table 1 전체 성공률을 논문 수치로 직접 그린다 (논문 그림 아님).

실행: python3 analysis/plot_figures.py → report/figures/mobile_aloha_cotrain.png
입력: analysis/recompute_results.json (recompute.py가 만든 Wilson CI와 횟수)
"""
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text())
OUT = ROOT / "report" / "figures" / "mobile_aloha_cotrain.png"


def korean_font():
    cands = (["AppleSDGothicNeo", "Apple SD Gothic Neo"] if platform.system() == "Darwin"
             else ["Noto Sans CJK KR", "Noto Sans KR", "NanumGothic"])
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in cands:
        if c in names:
            return c
    return "DejaVu Sans"


def main():
    plt.rcParams["font.family"] = korean_font()
    plt.rcParams["axes.unicode_minus"] = False
    tasks = list(RES["A_whole_task"].keys())
    co = [RES["A_whole_task"][t]["co_pct"] for t in tasks]
    no = [RES["A_whole_task"][t]["no_pct"] for t in tasks]
    co_ci = [RES["A_whole_task"][t]["co_wilson95"] for t in tasks]
    no_ci = [RES["A_whole_task"][t]["no_wilson95"] for t in tasks]
    ns = [RES["A_whole_task"][t]["N"] for t in tasks]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    ys = list(range(len(tasks)))[::-1]
    h = 0.36
    ax.barh([y + h / 2 for y in ys], co, height=h, color="#1b6f6a", label="공동 학습 (co-train)")
    ax.barh([y - h / 2 for y in ys], no, height=h, color="#c9c6bd", label="공동 학습 없음")
    for y, (lo, hi), v in zip([y + h / 2 for y in ys], co_ci, co):
        ax.plot([lo, hi], [y, y], color="#222", lw=1)
    for y, (lo, hi), v in zip([y - h / 2 for y in ys], no_ci, no):
        ax.plot([lo, hi], [y, y], color="#222", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{t} (n={n})" for t, n in zip(tasks, ns)], fontsize=8.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("전체 과제 성공률 (%), Wilson 95% 구간", fontsize=9)
    ax.set_title("표 1 재계산: ACT의 과제별 전체 성공률 (논문 수치, 1 시드)", fontsize=10)
    ax.grid(axis="x", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
