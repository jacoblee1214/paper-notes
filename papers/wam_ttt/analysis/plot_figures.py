"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_main_results.png : 표 C.1 New 블록의 과제별 진행도(WAM-TTT, LDA, WAM-ICL, WAM-Cotrain)와 분산 상계 95% 구간, 방법별 Orig/New 평균
own_grid_check.png   : 표 C.1 108칸이 시행 25회, 부록 D 루브릭 격자에서 도달 가능한지
own_toy_fast_weight.png : 토이 T1(닫힌 해, 외적, softmax 읽기), T2(SGD 단계별 목표 코사인)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 먼저 recompute.py와 toy_fast_weight.py 실행)
한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
"""
import json
import math
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
INK, ACC, GREY, WARN = "#1b1b1a", "#1f5fae", "#9a9890", "#c4572a"

# recompute.py와 같은 값 (표 C.1)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("recompute_vals", ANA / "recompute.py")
# recompute.py를 import하면 계산이 한 번 더 돌아 json을 다시 쓴다(수 초). 값만 쓰고 결과는 같다.
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
C1, TASKS, METHODS = rc.C1, rc.TASKS, rc.METHODS
SHORT = ["Transfer\nBottle", "Table\nBussing", "Deliver\nDrink", "Swap\nPlace", "Pour\nWater", "Stamp\nPaper", "Flip\nSteak", "Pyramid", "Multi-step\nSteak", "Avg."]
COL = {"WAM-TTT": ACC, "LDA": GREY, "WAM-ICL": WARN, "WAM-Cotrain": "#6a8f3a"}


def bound_half(m, n=25):
    m = m / 100
    return 1.96 * math.sqrt(m * (1 - m) / n) * 100


def plot_main():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.0), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    x = np.arange(10)
    ms = ["LDA", "WAM-Cotrain", "WAM-ICL", "WAM-TTT"]
    for j, m in enumerate(ms):
        vals = C1[m]["New"]
        halfs = [bound_half(v) for v in vals[:9]]
        # 평균 열 구간: 9칸 독립 분산 상계
        half_avg = 1.96 * math.sqrt(sum((bound_half(v) / 1.96) ** 2 for v in vals[:9])) / 9
        halfs.append(half_avg)
        ax.errorbar(x + (j - 1.5) * 0.17, vals, yerr=halfs, fmt="o", ms=4.5, lw=1.1, capsize=2, color=COL[m], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=7.8)
    ax.set_ylabel("진행도 (%)")
    ax.set_ylim(-5, 108)
    ax.set_title("표 C.1 New (낯선 가정집): 과제별 진행도와 분산 상계 95% 구간 (n = 25)", fontsize=10)
    ax.axvline(8.5, color=GREY, lw=0.6, ls="--")
    ax.legend(fontsize=8, ncol=4, loc="upper right", frameon=False)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax2 = axes[1]
    order = ["WAM-ICL", "pi05", "EgoScale", "WAM-Cotrain", "LDA", "WAM-TTT"]
    y = np.arange(len(order))
    o = [C1[m]["Orig"][9] for m in order]
    n_ = [C1[m]["New"][9] for m in order]
    ax2.barh(y - 0.18, o, 0.34, color="#c9d6e8", label="Orig (표준 큐비클)")
    ax2.barh(y + 0.18, n_, 0.34, color=ACC, label="New (낯선 가정집)")
    for yy, a, b in zip(y, o, n_):
        ax2.text(a + 1, yy - 0.18, f"{a:.1f}", va="center", fontsize=7.5)
        ax2.text(b + 1, yy + 0.18, f"{b:.1f}", va="center", fontsize=7.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(order, fontsize=8.5)
    ax2.set_xlim(0, 75)
    ax2.set_xlabel("9과제 평균 진행도 (%)")
    ax2.set_title("방법별 평균 (표 C.1의 Avg. 열)", fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "own_main_results.png", dpi=170)
    plt.close(fig)


def plot_grid():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    bad = {(c["method"], c["setting"], c["task"]) for c in rec["V4_feasibility_C1"]["infeasible_cells"]}
    rows = [(m, s) for m in METHODS for s in ("Orig", "New")]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for i, (m, s) in enumerate(rows):
        for j, t in enumerate(TASKS):
            v = C1[m][s][j]
            isbad = (m, s, t) in bad
            ax.add_patch(plt.Rectangle((j, len(rows) - 1 - i), 1, 1, color=("#f3d3c3" if isbad else "#e6eef8"), ec="white"))
            ax.text(j + 0.5, len(rows) - 1 - i + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=7.8, color=(WARN if isbad else INK))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(9) + 0.5)
    ax.set_xticklabels([s.replace("\n", " ") for s in SHORT[:9]], fontsize=7.5, rotation=20)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([f"{m} {s}" for m, s in reversed(rows)], fontsize=8)
    n_bad = len(bad)
    ax.set_title(f"표 C.1: 시행 25회와 부록 D 루브릭 격자에서 도달할 수 없는 칸(주황) {n_bad}/108 (Table Bussing은 항목 수 N = 1–6 허용)", fontsize=9.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(FIG / "own_grid_check.png", dpi=170)
    plt.close(fig)


def plot_toy():
    toy = json.loads((ANA / "toy_fast_weight_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    ax = axes[0]
    labs, cl, ch, cs = [], [], [], []
    for r in toy["T1"]:
        labs.append(("비등방" if r["aniso"] else "등방") + f"\nL={r['L']}")
        cl.append(r["cos_closed_vs_target"])
        ch.append(r["cos_hebb_vs_target"])
        cs.append(r["cos_softmax_vs_target"])
    x = np.arange(len(labs))
    ax.bar(x - 0.26, cl, 0.26, color=ACC, label="닫힌 해 W* (식 A.2)")
    ax.bar(x, ch, 0.26, color="#9bb7dc", label="외적 근사 (식 A.3)")
    ax.bar(x + 0.26, cs, 0.26, color=GREY, label="softmax 읽기 (식 A.1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("저장한 값과의 코사인")
    ax.set_ylim(0, 1.05)
    ax.set_title("T1: 임의 키-값 L쌍을 d = 48 선형 기억에 쓸 때", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax2 = axes[1]
    for r in toy["T2"]:
        if r["mode"] != "structured" or r["L"] != 256:
            continue
        steps = [h["step"] for h in r["hist"]]
        cs_ = [h["cos_vs_target_V"] for h in r["hist"]]
        ax2.plot(steps, cs_, marker="o", ms=3.5, lw=1.2, ls="-" if r["kind"] == "linear" else "--",
                 color=(ACC if r["eta"] == 0.1 else WARN), label=f"{r['kind']} η={r['eta']}")
    ax2.set_xscale("log")
    ax2.axhline(0.9, color=GREY, lw=0.6, ls=":")
    ax2.set_xlabel("inner SGD 단계 수 (N)")
    ax2.set_ylabel("읽기 출력과 목표 값의 코사인")
    ax2.set_title("T2: 구조 있는 대응(L = 256)에서 단계 수와 η", fontsize=10)
    ax2.legend(fontsize=7.5, frameon=False)
    ax2.grid(lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_fast_weight.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_main()
    plot_grid()
    plot_toy()
    print("wrote figures in", FIG)
