"""논문 표 II 값으로 우리가 그린 그래프 두 장을 만든다 (재계산은 recompute.py).

출력: report/figures/own_avg_seed.png   (평균 성공률 ± 논문 보고 SD, 두 학습 조건)
      report/figures/own_task_wilson.png (rand-obj 조건, task별 성공률과 Wilson 95% 구간, 150 시도)
실행: python3 analysis/plot_figures.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import recompute as R  # noqa: E402

FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for f in FONT_CANDIDATES:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

MODELS = ["DP-T", "DP-C", "ACT", "π0.5", "GR00T N1.5"]
COLS = {"DP-T": 0, "DP-C": 2, "ACT": 4, "π0.5": 6, "GR00T N1.5": 8}
C_OBJ, C_FULL = "#1f5fae", "#c8772e"


def load():
    text = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8")
    t2 = R.parse_rows(text, 10, R.TASKS + ["Avg."], 0)
    return t2


def plot_avg(t2):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    width = 0.38
    xs = range(len(MODELS))
    for k, (regime, off, color) in enumerate([("rand-obj", -width / 2, C_OBJ), ("rand-full", width / 2, C_FULL)]):
        means = [t2["Avg."][COLS[m] + k][0] for m in MODELS]
        sds = [t2["Avg."][COLS[m] + k][1] for m in MODELS]
        ax.bar([x + off for x in xs], means, width, yerr=sds, capsize=3, color=color,
               label=regime, error_kw={"lw": 0.9})
        for x, v in zip(xs, means):
            ax.text(x + off, v + 3.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(MODELS, fontsize=8.5)
    ax.set_ylabel("11개 task 평균 성공률 (%)", fontsize=8.5)
    ax.set_ylim(0, 72)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("논문 표 II 평균 (오차 막대: 논문 보고 ± 값)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_avg_seed.png")
    plt.close(fig)


def plot_task_wilson(t2):
    tasks = R.TASKS
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=160)
    offs = {"DP-T": -0.24, "DP-C": 0.0, "π0.5": 0.24}
    colors = {"DP-T": "#7a7a74", "DP-C": "#2f8f5b", "π0.5": C_OBJ}
    for m, off in offs.items():
        ys, lo, hi = [], [], []
        for t in tasks:
            p = t2[t][COLS[m]][0]
            k = round(p * 1.5)
            l, h = R.wilson(k, 150)
            ys.append(p)
            lo.append(p - l)
            hi.append(h - p)
        xs = [i + off for i in range(len(tasks))]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=3.2, lw=0.8, capsize=1.8,
                    color=colors[m], label=m)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.replace(" /B", " (양손)") for t in tasks], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("성공률 (%)", fontsize=8.5)
    ax.set_ylim(-3, 105)
    ax.axhline(0, color="#999", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    ax.set_title("rand-obj 조건 task별 성공률 (점: 논문 표 II, 구간: 150 시도 기준 Wilson 95%)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_task_wilson.png")
    plt.close(fig)


def main():
    t2 = load()
    plot_avg(t2)
    plot_task_wilson(t2)
    print("wrote", FIG / "own_avg_seed.png", FIG / "own_task_wilson.png")


if __name__ == "__main__":
    main()
