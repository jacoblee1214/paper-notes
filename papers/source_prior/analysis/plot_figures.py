"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_gap.png : 표 1 과제별 LeaP 대 기준선 격차와 Welch 95% 구간(시드 3개)
own_real_ci.png    : 실로봇(그림 4 막대값, 20회)의 Wilson 95% 구간과 보고 평균
own_ablation.png   : 절제 표 6-8 과제별 값(시드 0, 100회, 이항 95% 구간)
own_toy_source.png : 토이의 소스 변형별 에너지 거리(NFE 6)
실행: python3 plot_figures.py   (numpy, scipy, matplotlib; 먼저 recompute.py, toy_source_prior.py)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

ANA = Path(__file__).resolve().parent
FIG = ANA.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"
COL = ["#1f5fae", "#d98b2b", "#8a5fb5", "#7d7d7d", "#2e8b57"]
REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
DATA = REC["data"]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_gap():
    T1 = DATA["T1"]
    tasks = list(T1)
    names = ["A2A", "VITA", "NoPrior", "BridgePolicy"]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for jj, nm in enumerate(names, start=1):
        for i, t in enumerate(tasks):
            m0, s0 = T1[t][0]; m1, s1 = T1[t][jj]
            d = m0 - m1
            se = np.sqrt(s0 ** 2 / 3 + s1 ** 2 / 3)
            if se == 0:
                lo = hi = d
            else:
                df = se ** 4 / ((s0 ** 2 / 3) ** 2 / 2 + (s1 ** 2 / 3) ** 2 / 2)
                h = stats.t.ppf(0.975, df) * se
                lo, hi = d - h, d + h
            y = i + (jj - 2.5) * 0.17
            ax.plot([lo, hi], [y, y], color=COL[jj], lw=1.1, alpha=.8)
            ax.plot(d, y, "o", color=COL[jj], ms=3.6, label=nm if i == 0 else None)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t.replace("_", " ") for t in tasks], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("LeaP 평균 - 기준선 평균 (퍼센트포인트), 95% Welch 구간 (시드 3, 표본 SD)")
    ax.set_title("표 1 과제별 격차: 15과제 중 NoPrior·Bridge는 전부 앞서고 A2A·VITA는 12승 3패", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_gap.png", dpi=170)
    plt.close(fig)


def plot_real():
    rr = DATA["real_robot_bars"]
    meths = ["LeaP", "VITA", "A2A", "NoPrior"]
    cols = [COL[0], COL[2], COL[1], COL[3]]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    tasks = list(rr)
    w = 0.2
    for mi, m in enumerate(meths):
        for ti, t in enumerate(tasks):
            v = rr[t][mi]
            lo, hi = wilson(round(v / 5), 20)
            x = ti + (mi - 1.5) * w
            ax.bar(x, v, w * .9, color=cols[mi], label=m if ti == 0 else None)
            ax.plot([x, x], [lo, hi], color=INK, lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylabel("성공률 (%) 20회, Wilson 95%"); ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 20)); ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.set_title("그림 4 막대값 (과제별)", fontsize=10)
    ax = axes[1]
    k = REC["V5"]["pooled_successes_of_60"]
    rep = REC["V5"]["avg_reported"]
    for mi, m in enumerate(meths):
        lo, hi = wilson(k[m], 60)
        ax.bar(mi, 100 * k[m] / 60, .6, color=cols[mi])
        ax.plot([mi, mi], [lo, hi], color=INK, lw=1)
        ax.plot(mi, rep[m], "D", color="white", mec=INK, ms=5, label="본문 보고 평균" if mi == 0 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels(meths, fontsize=8)
    ax.set_ylim(0, 105); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("3과제 합산 60회 (막대=막대값 합, 마름모=본문 평균)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "own_real_ci.png", dpi=170); plt.close(fig)


def plot_ablation():
    ab = DATA["ablation"]
    tasks = ["pick diverse bottles", "open laptop", "handover block"]
    panels = [("T6", "(a) 프라이어 입력"), ("T7", "(b) 프라이어 형태"), ("T8", "(c) 지도 손실")]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, (tb, title) in zip(axes, panels):
        rows = ab[tb]
        names = list(rows)
        if tb == "T8":
            names = [n for n in names if n != "direct mu"]
        w = 0.8 / len(names)
        for ni, n in enumerate(names):
            for ti in range(3):
                v = rows[n][ti]
                lo, hi = wilson(v, 100)
                x = ti + (ni - (len(names) - 1) / 2) * w
                ax.bar(x, v, w * .92, color=cmap(ni), label=n if ti == 0 else None)
                ax.plot([x, x], [lo, hi], color=INK, lw=.7)
        ax.set_xticks(range(3)); ax.set_xticklabels(tasks, fontsize=8)
        ax.set_title(title, fontsize=10); ax.legend(fontsize=6.5, ncol=3, loc="upper center")
        ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20)); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("성공률 (%) 시드 0, 100회, Wilson 95%")
    fig.suptitle("절제 표 6-8을 과제별로 풀면: handover block 한 과제가 평균 격차의 대부분을 만든다", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "own_ablation.png", dpi=170); plt.close(fig)


def plot_toy():
    p = ANA / "toy_source_prior_results.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    names = list(d["A"])
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7), sharey=True)
    for ax, s, title in zip(axes, "ABC", ["설정 A (단봉, 이분산)", "설정 B (쌍봉)", "설정 C (4스텝 청크, 공유 mu·sigma)"]):
        vals = [d[s][n]["nfe6"]["energy"] for n in names]
        cols = [COL[3], "#b0b0b0", "#d9a35a", "#d9a35a", "#e7c9a0", COL[0]]
        ax.bar(range(len(names)), [min(v, 0.075) for v in vals], color=cols)
        for i, v in enumerate(vals):
            ax.text(i, min(v, 0.075) + .0015, f"{v:.3f}" + (" (축 밖)" if v > 0.075 else ""), ha="center", fontsize=7)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
        ax.set_title(title, fontsize=10); ax.set_ylim(0, 0.085)
        ax.set_xlabel("NFE 6, 시드 2 평균", fontsize=8)
    axes[0].set_ylabel("에너지 거리 (낮을수록 좋음)")
    fig.tight_layout(); fig.savefig(FIG / "own_toy_source.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    plot_gap(); plot_real(); plot_ablation(); plot_toy()
    print("wrote figures to", FIG)
