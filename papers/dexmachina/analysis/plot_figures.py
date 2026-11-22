"""재계산 결과를 그래프 두 장으로 그린다. 논문 그림은 쓰지 않는다.

입력: analysis/recompute_results.json (recompute.py가 만든다)
출력: report/figures/own_cells_ci.png      손별 물체별 Ours 값과 95% 구간
      report/figures/own_vs_baseline.png   Ours − 기준선 (물체별, 손별)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HANDS = ["inspire_hand", "allegro_hand", "xhand", "schunk_hand"]
HAND_KO = {"inspire_hand": "Inspire", "allegro_hand": "Allegro", "xhand": "XHand", "schunk_hand": "Schunk"}
OBJ = ["Ketchup-100", "Box-200", "Mixer-170", "Ketchup-300", "Mixer-300", "Notebook-300", "Waffleiron-300"]
INK, ACC, GREY, RED = "#1b1b1a", "#1f5fae", "#8a8a85", "#b3412e"


def set_korean_font():
    for name in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleSDGothicNeo"]:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def main():
    set_korean_font()
    d = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))
    cells = d["cells"]
    labels = d["fig3_labels"]

    # 그래프 1: 손별 물체별 Ours 평균과 95% 구간
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.6), sharex=True)
    for ax, h in zip(axes, HANDS):
        xs = list(range(len(OBJ)))
        means = [cells[f"{h}|{o}"]["mean"] for o in OBJ]
        lo = [cells[f"{h}|{o}"]["ci95"][0] for o in OBJ]
        hi = [cells[f"{h}|{o}"]["ci95"][1] for o in OBJ]
        err = [[m - l for m, l in zip(means, lo)], [u - m for m, u in zip(means, hi)]]
        ax.errorbar(xs, means, yerr=err, fmt="o", color=ACC, ecolor=GREY, capsize=3, ms=5)
        ax.axhline(0, color=INK, lw=0.6)
        ax.axhline(100, color=INK, lw=0.4, ls=":")
        ax.set_ylim(-20, 120)
        ax.set_ylabel(f"{HAND_KO[h]}\nAUC×100", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(range(len(OBJ)))
    axes[-1].set_xticklabels(OBJ, rotation=20, ha="right", fontsize=8)
    axes[0].set_title("DexMachina (Ours): 물체별 값과 t 95% 구간 (seed 5개)", fontsize=10, loc="left")
    fig.text(0.01, 0.005, "구간은 집계 파일의 SEM을 ddof=1로 고쳐 t(4)로 계산. 구간이 0 아래나 100 위로 나가는 것은 정규 근사의 한계.",
             fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG_DIR / "own_cells_ci.png", dpi=180)
    plt.close(fig)

    # 그래프 2: Ours − 기준선 (ObjDex 재구현, Task + Aux), 물체별
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
    colors = {"inspire_hand": "#5b8db8", "allegro_hand": "#2f5d8a", "xhand": "#8fb9a8", "schunk_hand": "#d08c60"}
    width = 0.2
    for ax, (bi, bname) in zip(axes, [(1, "Task Rew (ObjDex) 재구현"), (2, "Task + Aux (커리큘럼 없음)")]):
        for k, h in enumerate(HANDS):
            for j in range(len(OBJ)):
                b = labels[h][j][bi]
                if b is None:
                    continue
                v = labels[h][j][3] - b
                x = j + (k - 1.5) * width
                ax.bar(x, v, width=width * 0.92, color=colors[h],
                       hatch="///" if v < 0 else None, edgecolor=RED if v < 0 else colors[h], lw=0.6)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xticks(range(len(OBJ)))
        ax.set_xticklabels(OBJ, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(f"Ours - {bname}", fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[h]) for h in HANDS]
    axes[1].legend(handles, [HAND_KO[h] for h in HANDS], fontsize=7, frameon=False, loc="upper right")
    axes[0].set_ylabel("차이 (%p, 빨강 = 기준선이 더 높음)", fontsize=8)
    fig.text(0.01, 0.005, "기준선 값은 Fig. 3 막대 라벨을 판독한 값(0.1 단위). 기준선 오차막대는 판독하지 않음.",
             fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_DIR / "own_vs_baseline.png", dpi=180)
    plt.close(fig)
    print("wrote", FIG_DIR / "own_cells_ci.png", FIG_DIR / "own_vs_baseline.png")


if __name__ == "__main__":
    main()
