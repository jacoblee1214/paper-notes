"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_taskmix.png   : 범주 묶음에서 ID 유형비로 보정한 기대 성공률 대 관측 성공률 (단일 팔, 양팔)
own_orderstab.png : 인접 순위 쌍의 격차 대 과제 부트스트랩 순서 유지 확률
own_aux.png       : 보조 데이터 공동학습 효과(표 20)와 π0.5 시연 수 확장의 범주 묶음 추세(그림 4)
own_roboarena.png : RoboArena Elo 대 MESA·BiMESA 성공률 (순위 일치 정도)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 먼저 python3 analysis/recompute.py)
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
INK, ACC, ORG, GREY, RED = "#1b1b1a", "#1f5fae", "#d9822b", "#9a9890", "#b3261e"
REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
LAB = {"pi0.5": "π0.5", "pi0": "π0", "FAST": "π0-FAST", "PG-FM": "PG-FM", "GR00T": "GR00T", "DP": "DP", "PG-Bin": "PG-Bin", "GPT": "GPT-6.1-Sol"}


def plot_taskmix():
    t = REC["V13_type_adjusted"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, (key, title, models) in zip(
        axes,
        [("single", "단일 팔 MESA-Category (과제 20개)", ["DP", "PG-FM", "GR00T", "pi0", "FAST", "pi0.5"]),
         ("bimanual", "BiMESA-Category (과제 17개)", ["DP", "PG-Bin", "PG-FM", "GR00T", "pi0", "FAST", "pi0.5", "GPT"])]):
        pm = t[key]["Category"]["per_method"]
        y = np.arange(len(models))[::-1]
        for yi, m in zip(y, models):
            e, o = pm[m]["id_mix_adjusted_expected"], pm[m]["observed"]
            ax.plot([o, e], [yi, yi], color=GREY, lw=1.4, zorder=1)
            ax.scatter([e], [yi], color=ACC, s=38, zorder=3, label="ID 성공률을 Category 과제 유형비로 가중한 기대치" if yi == y[0] else None)
            ax.scatter([o], [yi], color=ORG, s=38, zorder=3, label="Category 관측" if yi == y[0] else None)
            ax.text(max(e, o) + 2, yi, f"{o - e:+.0f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([LAB[m] for m in models], fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("성공률 (%)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", color="#e5e3dc", lw=0.6)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "own_taskmix.png", dpi=170)
    plt.close(fig)


def plot_orderstab():
    v = REC["V10_task_bootstrap"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    xs, ys, cs = [], [], []
    for emb, col in (("single", ACC), ("bimanual", ORG)):
        for s, d in v[emb].items():
            for a in d["adjacent"]:
                ax.scatter(a["gap"], a["p_order_kept"], color=col, s=22, alpha=0.8, label=("단일 팔" if emb == "single" else "양팔") if (s == "ID" and a is d["adjacent"][0]) else None)
    ax.axhline(0.975, color=RED, lw=1, ls="--")
    ax.text(17.5, 0.9755, "0.975", color=RED, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("인접 순위 쌍의 성공률 격차 (포인트)")
    ax.set_ylabel("과제 부트스트랩에서\n순서가 유지된 비율")
    ax.set_ylim(0.4, 1.02)
    ax.grid(color="#e5e3dc", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("인접 순위 쌍 55개: 격차 3포인트 안팎에서 과제 표집에 따라 순서가 흔들림", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_orderstab.png", dpi=170)
    plt.close(fig)


def plot_aux():
    a = REC["V7_aux_cotraining"]
    suites = ["ID", "Spatial", "Instance", "Composite", "Category"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    w = 0.26
    for k, (m, col) in enumerate((("DP", GREY), ("pi0", ORG), ("pi0.5", ACC))):
        d = [a[m][s]["delta"] for s in suites]
        xs = np.arange(5) + (k - 1) * w
        ax.bar(xs, d, width=w, color=col, label=LAB[m])
        for x, s in zip(xs, suites):
            p = a[m][s]["paper_mcnemar_p"]
            if p <= 0.05:
                mark = "**" if a[m][s]["significant_bonferroni_15"] else "*"
                dv = a[m][s]["delta"]
                ax.text(x, dv + (0.6 if dv >= 0 else -0.6), mark, ha="center", va="bottom" if dv >= 0 else "top", fontsize=10, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(-6, 14.5)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(suites, fontsize=9)
    ax.set_ylabel("Aux 추가 시 성공률 변화 (포인트)")
    ax.set_title("표 20: 보조 과제 공동학습 효과 (*: McNemar p<0.05, **: 15검정 Bonferroni 통과)", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)
    ax = axes[1]
    demos = [10, 50, 100]
    series = {"ID": (53.5, 64.0, 62.8), "Instance": (47.3, 51.3, 55.4), "Composite": (40.0, 44.7, 46.7), "Category": (32.1, 28.5, 26.4)}
    cols = {"ID": GREY, "Instance": ORG, "Composite": "#6a8f3a", "Category": RED}
    for s, v in series.items():
        ax.plot(demos, v, marker="o", color=cols[s], label=s, lw=1.6 if s == "Category" else 1.1)
    ax.set_xscale("log")
    ax.set_xticks(demos)
    ax.set_xticklabels(["10", "50", "100"])
    ax.set_xlabel("과제당 시연 수")
    ax.set_ylabel("π0.5 성공률 (%)")
    ax.set_title("논문 그림 4에서 읽은 값: Category는 시연이 늘수록 하락", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#e5e3dc", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_aux.png", dpi=170)
    plt.close(fig)


def plot_roboarena():
    elo = {"PG-Bin": 771, "PG-FM": 1527, "pi0": 1449, "FAST": 1564, "pi0.5": 1594}
    mesa = {"PG-FM": 22.8, "pi0": 31.7, "FAST": 35.3, "pi0.5": 52.0}
    bim = {"PG-Bin": 18.3, "PG-FM": 30.4, "pi0": 29.6, "FAST": 39.4, "pi0.5": 48.7}
    r = REC["V8_roboarena_rank"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    for ax, d, title, key in ((axes[0], mesa, "MESA (단일 팔, 정책 4개)", "single_vs_roboarena"),
                              (axes[1], bim, "BiMESA (양팔, 정책 5개)", "bimanual_vs_roboarena")):
        for m, v in d.items():
            ax.scatter(elo[m], v, color=ACC, s=40, zorder=3)
            ax.annotate(LAB[m], (elo[m], v), xytext=(5, 4), textcoords="offset points", fontsize=8.5)
        ax.set_title(f"{title}: Kendall τ = {r[key]['kendall_tau']}, 순열 p = {r[key]['perm_p_one_sided']}", fontsize=9)
        ax.set_xlabel("RoboArena Elo")
        ax.grid(color="#e5e3dc", lw=0.6)
    axes[0].set_ylabel("성공률 (%)")
    fig.tight_layout()
    fig.savefig(FIG / "own_roboarena.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_taskmix()
    plot_orderstab()
    plot_aux()
    plot_roboarena()
    print("wrote 4 figures to", FIG)
