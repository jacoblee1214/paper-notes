"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sim_real.png     : 실제 대 시뮬 성공률 산점도, SimFoundry와 PolaRiS (표 G.1, G.2)
own_bias.png         : 과제별 시뮬 - 실제 평균 편차와 셀 단위 Fisher 유의 개수 (표 G.1)
own_noise_ceiling.png: 토이 T1, 25회 시행 잡음에서 기대되는 평균 r 분포와 관측값
own_cousin_claims.png: 초록의 17/21/40% 재계산과 평가 방식 (표 G.4–G.6)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK, macOS Apple SD Gothic Neo 순서로 찾는다)
"""
import json
import sys
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
sys.path.insert(0, str(ANA))
import recompute as rc  # noqa: E402  (표 입력값을 재사용하고 결과 JSON을 다시 쓴다)

REC = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))


def korean_font():
    for name in ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, WARN = "#1b1b1a", "#1f5fae", "#9a9890", "#c0522b"
TASK_COL = ["#1f5fae", "#c0522b", "#2e8b57", "#8a5fb3", "#b8860b", "#2aa6a6", "#555555"]
POL_MARK = {"pi0": "o", "pi05": "s", "n16": "^", "n17": "D", "dz": "v"}


def plot_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
    for ax, tbl, name, key in ((axes[0], rc.G1, "SimFoundry (표 G.1)", "G1_simfoundry"),
                               (axes[1], rc.G2, "PolaRiS (표 G.2)", "G2_polaris")):
        for i, (t, d) in enumerate(tbl.items()):
            for p, (r_, s_) in d.items():
                ax.scatter(r_, s_, c=TASK_COL[i], marker=POL_MARK[p], s=34, alpha=0.85,
                           label=t if p == "pi0" else None, edgecolors="white", linewidths=0.4)
        ax.plot([0, 100], [0, 100], color=GREY, ls="--", lw=1)
        ax.set_xlim(-4, 104)
        ax.set_ylim(-4, 104)
        ax.set_xlabel("실제 성공률 (%)")
        ax.set_title(f"{name}\n과제별 r 평균 {REC[key]['mean_r']:.3f}, 풀링 r "
                     f"{REC['pooled']['pearson_29_pairs' if key == 'G1_simfoundry' else 'polaris_pearson_29_pairs']:.3f}",
                     fontsize=10)
    axes[0].set_ylabel("시뮬 성공률 (%)")
    axes[0].legend(fontsize=7, loc="upper left", frameon=False, title="과제 (색)", title_fontsize=7)
    fig.text(0.5, -0.01, "마커: ○ π0, □ π0.5, △ GR00T N1.6, ◇ N1.7, ▽ DreamZero. 점선은 y = x. 각 점은 25회 시행 기준.",
             ha="center", fontsize=8, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_real.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_bias():
    tasks = list(rc.G1)
    mean_bias = [np.mean([s_ - r_ for r_, s_ in rc.G1[t].values()]) for t in tasks]
    cells = {t: [c for c in REC["cell_fisher"]["cells"] if c["task"] == t] for t in tasks}
    sig = [sum(c["fisher_p"] < 0.05 for c in cells[t]) for t in tasks]
    tot = [len(cells[t]) for t in tasks]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    cols = [WARN if t in ("Stack Dishware", "Store Marker", "Throw Away Trash") else ACC for t in tasks]
    ax.bar(range(len(tasks)), mean_bias, color=cols, width=0.62)
    ax.axhline(0, color=INK, lw=0.8)
    for i, (b, s, n) in enumerate(zip(mean_bias, sig, tot)):
        ax.text(i, b - 2 if b < 0 else b + 1, f"{b:+.0f}\n(p<0.05: {s}/{n})", ha="center",
                va="top" if b < 0 else "bottom", fontsize=8)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("시뮬 - 실제 평균 (%p)")
    ax.set_ylim(-60, 22)
    ax.set_title("상관은 높지만 미세조정 과제(주황)는 시뮬이 크게 낮다 (표 G.1, Fisher 25 대 25)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_bias.png", dpi=170)
    plt.close(fig)


def plot_ceiling():
    h = REC["toy_hist"]
    bins = np.array(h["bins"])
    mid = (bins[:-1] + bins[1:]) / 2
    w = bins[1] - bins[0]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(mid, np.array(h["noise_ceiling_counts"]) / 4000, width=w * 0.92, color=ACC, alpha=0.85,
           label="시뮬 = 실제인 이상적 환경 (25회 시행 잡음만)")
    ax.bar(mid, np.array(h["null_counts"]) / 4000, width=w * 0.92, color=GREY, alpha=0.6,
           label="시뮬 점수를 정책 사이에서 섞은 귀무")
    ax.axvline(REC["G1_simfoundry"]["mean_r"], color=WARN, lw=2, label=f"관측 평균 r = {REC['G1_simfoundry']['mean_r']:.3f}")
    ax.set_xlabel("7개 과제 평균 Pearson r")
    ax.set_ylabel("비율")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("토이 T1: 관측 r은 시행 잡음 천장 수준이고 귀무와는 멀다", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_noise_ceiling.png", dpi=170)
    plt.close(fig)


def plot_cousins():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    keys = [k for k in rc.G4 if k[1].startswith("Real")]
    fig5a = ["YAM Pot on Stove", "YAM Throw Away Trash", "DROID Store Marker", "DROID Throw Away Trash"]
    vals = [(rc.G4[k][1] - rc.G4[k][0], k) for k in keys]
    vals.sort(key=lambda x: x[0])
    ax.barh(range(len(vals)), [v for v, _ in vals],
            color=[ACC if k[0] in fig5a else GREY for _, k in vals])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([f"{k[0]} / {k[1].replace('Real ', '')}" for _, k in vals], fontsize=7)
    ax.axvline(17, color=WARN, ls="--", lw=1.2)
    ax.axvline(REC["object_cousin_avg_gain_pp"]["real_all_6task_12cells"], color=INK, ls=":", lw=1.2)
    ax.text(17.5, 0.2, "주장 17", color=WARN, fontsize=8)
    ax.text(REC["object_cousin_avg_gain_pp"]["real_all_6task_12cells"] - 0.4, 1.3, "12셀 평균 15.1", color=INK, fontsize=8, ha="right")
    ax.set_xlabel("실제 로봇 성공률 변화 (%p, +9 cousins - twin only)")
    ax.set_title("객체 cousin: 파랑 = 그림 5A에 실린 4과제 (평균 17.0)", fontsize=9)
    ax = axes[1]
    labels = ["객체\n(실제 로봇)", "장면\n(시뮬)", "작업\n(시뮬)"]
    vals = [17, 21, 40]
    cols = [ACC, WARN, WARN]
    ax.bar(range(3), vals, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v}", ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("초록의 평균 향상 (%p)")
    ax.set_title("초록은 셋 다 \"실제 로봇\"이라 쓰지만\n장면·작업은 시뮬 평가 (표 G.5, G.6)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_cousin_claims.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_scatter()
    plot_bias()
    plot_ceiling()
    plot_cousins()
    print("wrote", FIG)
