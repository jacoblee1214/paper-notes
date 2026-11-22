"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_real_tiers.png     : 실로봇 시스템 처리량(논문 표 5, 그림 4)을 계층별로 다시 그린 막대와 Poisson 95% 구간
own_sim_tradeoff.png   : 시뮬레이션 총 처리량의 상대 변화 (표 4, 기준선 RR·EDF 평균 대비)와 잡음 폭
own_toy_vs_paper.png   : 토이 시뮬레이터의 기아율 대 논문 표 4 (독립 재구현의 검증)
own_latency_capacity.png: 표 1 지연 직선 적합과 용량·수요 (유체 모델)
실행: python3 analysis/plot_figures.py   (numpy, scipy, matplotlib; recompute.py와 toy_mirror_sim.py를 먼저 실행)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

from paper_tables import SEEDS_REAL, T1_L40S, T4, T4_SCHED, T5  # noqa: E402

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
INK, ACC, ORG, GREY = "#1b1b1a", "#1f5fae", "#c8641e", "#9a9890"
R = json.load(open(ANA / "recompute_results.json"))
TOY = json.load(open(ANA / "toy_mirror_sim_results.json"))


def poisson_ci(k, z=1.959964):
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0
    hi = stats.chi2.ppf(0.975, 2 * (k + 1)) / 2
    return lo, hi


def fig_real():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, cfg, title in zip(axes, ["all_fast", "one_fast", "half_fast"], ["All fast (10 fast)", "One fast (1 fast + 9 slow)", "Half fast (5 + 5)"]):
        cells = [(c, s) for (c, s) in T5 if c == cfg]
        for i, key in enumerate(cells):
            tf, ts, tot, *_ = T5[key]
            nf = {"all_fast": 10, "one_fast": 1, "half_fast": 5}[cfg]
            ns = 10 - nf
            fast_tot, slow_tot = tf * nf, (ts or 0) * ns
            ax.bar(i, fast_tot, color=ACC, width=0.65)
            ax.bar(i, slow_tot, bottom=fast_tot, color=GREY, width=0.65)
            k = round(tot * SEEDS_REAL)
            lo, hi = poisson_ci(k)
            ax.errorbar(i, tot, yerr=[[tot - lo / SEEDS_REAL], [hi / SEEDS_REAL - tot]], color=INK, capsize=3, lw=1)
            ax.text(i, hi / SEEDS_REAL + 2, f"{tot:.0f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([s.replace("LA w=", "LA@") for _, s in cells], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("시스템 처리량 (조각/분)")
    axes[0].set_ylim(0, 125)
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(color=ACC, label="빠른 로봇 계층"), Patch(color=GREY, label="느린 로봇 계층")],
                   fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("실로봇 시스템 처리량 (논문 표 5 재작도, 오차 막대는 총량의 Poisson 95% 구간, 3시드 합산)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_tiers.png", dpi=200)
    plt.close(fig)


def fig_sim():
    rel = R["V5b_total_throughput_relative_to_baselines"]
    noise = R["V5a_noise_from_weight_invariant_all_fast"]["pooled_relative_sd_of_throughput_%"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, cfg, title in zip(axes, ["one_fast", "half_fast"], ["One fast", "Half fast"]):
        Ns = [2, 4, 6, 8, 10]
        for s, col, lab in [("la1", GREY, "LA@1"), ("la3", ACC, "LA@3"), ("la5", ORG, "LA@5")]:
            ax.plot(Ns, [rel[cfg][str(n)][s]["vs_mean(RR,EDF)_%"] for n in Ns], "o-", color=col, label=lab, lw=1.5, ms=4)
        ax.axhspan(-2 * noise, 2 * noise, color="#e6e4dc", alpha=0.7, label="±2 SD 잡음(가중치 불변 all-fast 3개 값에서 추정)")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("로봇 수 N")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("총 처리량의 상대 변화 (%) vs 평균(RR, EDF)")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    axes[1].set_ylim(-45, 10)
    fig.suptitle("시뮬레이션(LIBERO, b=3) 총 처리량: LA@1·3은 대체로 잡음 범위, LA@5는 half-fast에서 큰 하락 (논문 표 4 재계산)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_tradeoff.png", dpi=200)
    plt.close(fig)


def fig_toy():
    rows = TOY["table4_like"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]
    for name, p, col, mk in [("rr", "rr", GREY, "s"), ("edf", "edf", INK, "o"), ("laB_w1", "la1", ACC, "^"),
                             ("laB_w3", "la3", "#3b8f5f", "v"), ("laB_w5", "la5", ORG, "D")]:
        xs = [T4[r["shape"]][r["N"]][T4_SCHED.index(p)][1] for r in rows]
        ys = [r[name]["starv"] for r in rows]
        ax.scatter(xs, ys, s=18, color=col, marker=mk, label=name.replace("laB_w", "LA(B)@").upper() if name.startswith("la") else name.upper(), alpha=0.85)
    ax.plot([0, 60], [0, 60], color=INK, lw=0.7)
    ax.set_xlabel("논문 표 4 평균 기아율 (%)")
    ax.set_ylabel("토이 시뮬레이터 기아율 (%)")
    ax.set_title("15개 설정(3 구성 × N=2..10)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    labels = ["LA@3\none-fast", "LA@5\none-fast", "LA@3\nhalf-fast", "LA@5\nhalf-fast"]
    paper = [T4["one_fast"][10][3][1], T4["one_fast"][10][4][1], T4["half_fast"][10][3][1], T4["half_fast"][10][4][1]]
    A, B = [], []
    for shape in ["one_fast", "half_fast"]:
        r = [x for x in rows if x["shape"] == shape and x["N"] == 10][0]
        A += [r["laA_w3"]["starv"], r["laA_w5"]["starv"]]
        B += [r["laB_w3"]["starv"], r["laB_w5"]["starv"]]
    x = np.arange(4)
    ax.bar(x - 0.27, paper, 0.26, color=INK, label="논문 표 4 (N=10)")
    ax.bar(x, B, 0.26, color=ACC, label="토이 B (한계 이득 점수)")
    ax.bar(x + 0.27, A, 0.26, color=ORG, label="토이 A (후보 청크만 보상)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("평균 기아율 (%)")
    ax.set_title("가중 Lookahead: 보상을 어떻게 읽느냐에 따라 달라짐", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_vs_paper.png", dpi=200)
    plt.close(fig)


def fig_latency():
    b = np.arange(1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    ax.plot(b, T1_L40S, "o", color=ACC, label="표 1 (L40S, π0.5)")
    slope, icpt = np.polyfit(b, T1_L40S, 1)
    ax.plot(b, slope * b + icpt, "-", color=GREY, label=f"직선 적합: {icpt:.0f} + {slope:.1f}·b ms")
    ax.set_xticks(b)
    ax.set_xlabel("배치 크기 b")
    ax.set_ylabel("추론 지연 (ms)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    cap = [1000 * k / t for k, t in zip(b, T1_L40S)]
    ax.plot(b, cap, "o-", color=ACC, label="GPU 용량 (요청/초)")
    for lab, d, col in [("시뮬 all-fast N=10 (33.3)", 33.3, ORG), ("실로봇 all-fast (30.0)", 30.0, "#b08900"),
                        ("실로봇 half-fast (22.5)", 22.5, "#3b8f5f"), ("실로봇 one-fast (16.5)", 16.5, GREY)]:
        ax.axhline(d, color=col, lw=1, ls="--")
        ax.text(5.3, d + 0.25, lab, fontsize=7.5, va="bottom", color=col)
    ax.set_xlim(0.8, 8.6)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("배치 크기 b")
    ax.set_ylabel("요청/초")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("배치 지연은 b에 거의 선형(절편 약 40 ms): 큰 배치는 로봇당 비용을 줄이지만 모든 청크를 늦춘다", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_latency_capacity.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_real()
    fig_sim()
    fig_toy()
    fig_latency()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
