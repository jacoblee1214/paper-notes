"""우리가 만든 그래프 두 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_six_task.png : Table 2의 six-task score와 분산 상한 기반 95% 구간 (recompute_results.json의 K1)
own_h2r_ci.png   : Table 3 H2R 성공률과 Wilson 95% 구간, joint 스트림 0/240 표기 (K5)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib. 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"


def plot_six_task(rec):
    rows = [("SHAPE joint", "SHAPE (joint)"), ("SHAPE EEF", "SHAPE (EEF)"),
            ("w/o human joint", "w/o human (joint)"), ("w/o FAST joint", "w/o FAST (joint)"),
            ("w/o both joint", "w/o both (joint)"), ("pi0.5 joint", "pi0.5 (joint)"),
            ("GR00T N1.7 joint", "GR00T N1.7 (joint)")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, (key, label) in zip(y, rows):
        r = rec["K1_six_task_scores"][key]
        s = r["score"]
        lo, hi = r["ci95"]
        color = ACC if key == "SHAPE joint" else (GREY if ("pi0.5" in key or "GR00T" in key) else INK)
        ax.errorbar(s, yi, xerr=[[s - lo], [hi - s]], fmt="o", color=color, capsize=3, ms=6)
        ax.text(hi + 0.01, yi, f"{s:.2f}", va="center", fontsize=8.5, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], fontsize=9)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("six-task score (논문 Table 2 셀에서 재계산, 95% 구간)")
    ax.axvline(0.63, color=GREY, lw=0.8, ls="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("실로봇 6과제 점수 재계산 (K1)", fontsize=10, loc="left")
    fig.tight_layout()
    out = FIG / "own_six_task.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_h2r(rec):
    labels = ["w/o both", "w/o human", "w/o FAST", "SHAPE"]
    counts = {"w/o both": 0, "w/o human": 4, "w/o FAST": 7, "SHAPE": 21}
    ci = rec["K5_h2r_wilson95"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(labels))
    pct = np.array([100 * counts[k] / 60 for k in labels])
    lo = np.array([ci[k][0] for k in labels])
    hi = np.array([ci[k][1] for k in labels])
    ax.bar(x, pct, color=[GREY, GREY, INK, ACC], width=0.55)
    ax.errorbar(x, pct, yerr=[pct - lo, hi - pct], fmt="none", ecolor=INK, capsize=4, lw=1)
    for xi, k in zip(x, labels):
        ax.text(xi, hi[xi] + 1.5, f"{counts[k]}/60", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("H2R 성공률 (%), EEF 스트림", fontsize=9)
    ax.set_ylim(0, 55)
    ax.text(-0.35, 46, "joint 스트림: 0/240 (모든 모델)", ha="left", fontsize=8.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("사람 데이터만으로 학습한 H2R 전이 (K5, Wilson 95% 구간)", fontsize=10, loc="left")
    fig.tight_layout()
    out = FIG / "own_h2r_ci.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    for p in (plot_six_task(rec), plot_h2r(rec)):
        print("wrote", p)
