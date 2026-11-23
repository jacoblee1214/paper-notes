"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_cost.png        : 표 1 비용(SC, MD w/o TF, MD). 상자 쌓기는 로그축, 나머지 과제는 선형축 확대
own_valid.png       : 표 1 Valid % 와 Wilson 95% 구간 (노드 수 가정: 과제별 예산, Parkour 1은 18)
own_parent_toy.png  : 토이, 섬 안의 유효 노드 비율에 따른 부모 선택 가중치
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 python3 analysis/recompute.py, python3 analysis/toy_parent_weights.py 를 실행해 JSON을 만들어 둘 것.
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
INK, ACC, GREY, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#c0662b"

TASKS = ["Banana", "Box Stacking", "Climb Table", "Long-Dist.", "Clutter", "Parkour 1", "Parkour 2", "Under-Table"]
SC = [None, 398.05, 16.16, 13.61, None, None, None, 115.64]
WO = [21.17, 398.05, 16.47, 10.70, 11.96, 15.97, 15.90, 18.87]
MD = [20.08, 10.53, 15.97, 10.42, 11.67, 15.47, 15.51, 17.84]


def fig_cost():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.0, 1.5]})
    # (a) 상자 쌓기 로그축
    labels = ["SC", "MD w/o TF", "MD"]
    vals = [398.05, 398.05, 10.53]
    ax1.bar(labels, vals, color=[GREY, WARM, ACC])
    ax1.set_yscale("log")
    for i, v in enumerate(vals):
        ax1.text(i, v * 1.1, f"{v:g}", ha="center", fontsize=9)
    ax1.set_title("Box Stacking (로그축)", fontsize=11)
    ax1.set_ylabel("TO 비용 (낮을수록 좋음)")
    ax1.set_ylim(5, 900)
    # (b) 나머지 7개 과제
    idx = [i for i, t in enumerate(TASKS) if t != "Box Stacking"]
    x = np.arange(len(idx))
    w = 0.27
    sc = [SC[i] if SC[i] is not None and SC[i] < 60 else np.nan for i in idx]
    ax2.bar(x - w, sc, w, color=GREY, label="SC (해 없음은 빈칸)")
    ax2.bar(x, [WO[i] for i in idx], w, color=WARM, label="MD w/o TF")
    ax2.bar(x + w, [MD[i] for i in idx], w, color=ACC, label="MD")
    ax2.set_xticks(x)
    ax2.set_xticklabels([TASKS[i] for i in idx], rotation=25, ha="right", fontsize=9)
    ax2.set_ylim(8, 24)
    ax2.set_title("나머지 7개 과제 (세로축 8-24 확대; Under-Table의 SC 115.64는 범위 밖)", fontsize=10)
    for k, i in enumerate(idx):
        rel = (WO[i] - MD[i]) / WO[i] * 100
        ax2.text(k + w, MD[i] + 0.3, f"-{rel:.1f}%", ha="center", fontsize=8, color=ACC)
    ax2.legend(fontsize=8, loc="upper right")
    fig.suptitle("표 1 비용: MD가 8개 과제 모두 낮지만 상자 쌓기 외에는 2.4-5.5% 차이", fontsize=12)
    fig.text(0.5, 0.005, "값은 논문 표 1. 과제당 한 번의 결과(표에 시드 수 없음). 막대 위 %는 MD가 MD w/o TF보다 낮은 비율.", ha="center", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(FIG / "own_cost.png", dpi=160)
    plt.close(fig)


def fig_valid():
    res = json.loads((ANA / "recompute_results.json").read_text())
    per = res["V5_valid_pct_pooled"]["per_task"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(per))
    w = 0.36
    for off, key, col, lab in ((-w / 2, "wo", WARM, "MD w/o TF"), (w / 2, "md", ACC, "MD")):
        pct = [p[f"{key}_pct"] for p in per]
        lo = [p[f"{key}_pct"] - 100 * p[f"{key}_wilson"][0] for p in per]
        hi = [100 * p[f"{key}_wilson"][1] - p[f"{key}_pct"] for p in per]
        ax.bar(x + off, pct, w, color=col, label=lab, yerr=[lo, hi], capsize=2, error_kw={"lw": 1, "ecolor": INK})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TASKS[i]}\n(n={p['n']})" for i, p in enumerate(per)], fontsize=8.5)
    ax.set_ylabel("Valid % (노드 수 n 가정)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("표 1 Valid %와 Wilson 95% 구간: 과제별로는 구간이 모두 겹침", fontsize=11.5)
    fig.text(0.5, 0.005, "n은 공식 저장소 run_search.sh의 과제별 노드 예산을 가정(논문에 없음). Parkour 1은 정수가 되는 최소값 18, Banana MD는 19/27(70.37)로 대체.", ha="center", fontsize=7.5, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "own_valid.png", dpi=160)
    plt.close(fig)


def fig_toy():
    res = json.loads((ANA / "toy_parent_weights_results.json").read_text())
    f = [r["valid_frac"] for r in res]
    mass = [r["valid_mass_mean"] for r in res]
    ratio = [r["weight_ratio_cheapest_vs_costliest_valid_median"] for r in res]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4))
    a.plot(f, mass, "o-", color=ACC, label="선택 확률 중 유효 노드의 몫")
    a.plot(f, f, "--", color=GREY, label="균등 선택이면")
    a.set_xlabel("섬 안의 유효 노드 비율")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=8.5)
    a.set_title("(a) 유효 노드가 받는 선택 확률", fontsize=11)
    b.semilogy(f, ratio, "o-", color=WARM)
    b.axvline(0.5, color=GREY, lw=1, ls=":")
    b.set_xlabel("섬 안의 유효 노드 비율")
    b.set_ylabel("가중치 비 (가장 싼 유효 노드 / 가장 비싼 유효 노드, 중앙값)")
    b.set_title("(b) 비용 차이가 선택에 반영되는 정도", fontsize=11)
    b.annotate("50% 아래: 비용과 무관\n(median = 0, MAD = 0)", (0.3, 1.3), fontsize=8.5)
    fig.suptitle("토이: A.2 부모 선택 식 (F = 1/(1+비용) 가정, λ = 10, 비용 10-20)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "own_parent_toy.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_cost()
    fig_valid()
    fig_toy()
    print("wrote 3 figures to", FIG)
