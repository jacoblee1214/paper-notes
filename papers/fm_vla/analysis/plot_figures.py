"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 여기서 만들지 않는다).

own_table1_ci.png : 표 1의 주요 방법별·과제별 성공률과 Wilson 95% 구간 (시행 18회, 표 1의 k/18)
own_diff_forest.png : FM-VLA와 기준선·절제의 성공률 차이와 Newcombe 95% 구간, Fisher p (recompute_results.json)
own_token_ablation.png : 힘 토큰 수별 성공률(그림 3 막대 라벨, 그림 판독 값, 18회 가정)과 행동 전문가 접미 토큰 수
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 python3 analysis/recompute.py 를 실행해 recompute_results.json을 만들어 둘 것.
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
INK, ACC, GREY, SOFT, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#7fa6d6", "#c07a2c"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)


def plot_table1():
    methods = [("π0.5 (이력 없음)", [13, 2, 0], GREY), ("π-MEM (시각 기억)", [14, 6, 9], WARM),
               ("상태 이력만", [18, 2, 2], SOFT), ("FM-VLA", [18, 13, 14], ACC)]
    tasks = ["Cups (과제 1)", "Buttons (과제 2)", "Wipe (과제 3)"]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(3)
    w = 0.2
    for j, (name, ks, col) in enumerate(methods):
        pct = np.array([100 * k / 18 for k in ks])
        lo = np.array([wilson(k, 18)[0] for k in ks])
        hi = np.array([wilson(k, 18)[1] for k in ks])
        xs = x + (j - 1.5) * w
        ax.bar(xs, pct, width=w, color=col, label=name)
        ax.errorbar(xs, pct, yerr=np.vstack([pct - lo, hi - pct]), fmt="none", ecolor=INK, capsize=2, lw=0.8)
    ax.axhline(100 / 3, color=INK, lw=0.7, ls=":")
    ax.text(2.45, 100 / 3 + 1.5, "1/3", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 108)
    ax.set_title("표 1 주요 행: 성공률과 Wilson 95% 구간 (시행 18회, 점선은 N을 무작위로 고를 때의 1/3)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png", dpi=200)
    plt.close(fig)


def plot_forest():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V4_V6_fm_vla_vs_each"]
    pick = [("pi-MEM", "Cups"), ("pi-MEM", "Buttons"), ("pi-MEM", "Wipe"), ("pi-MEM", "pooled(54)"),
            ("FM-VLA(Q-Former)", "Buttons"), ("FM-VLA(Q-Former)", "Wipe"), ("FM-VLA(Q-Former)", "pooled(54)"),
            ("state_only", "Buttons"), ("state_only", "Wipe"), ("force_only", "Buttons"), ("force_only", "Wipe")]
    label = {"pi-MEM": "π-MEM", "FM-VLA(Q-Former)": "Q-Former", "state_only": "상태 이력만", "force_only": "힘 이력만"}
    scope = {"pooled(54)": "합산 54회"}
    rows = [next(r for r in rec if r["vs"] == v and r["scope"] == s) for v, s in pick]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        d, lo, hi = r["diff_pp"], r["newcombe95_pp"][0], r["newcombe95_pp"][1]
        col = ACC if lo > 0 else GREY
        ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ecolor=INK, capsize=3)
        ax.text(101, yi, f"p={r['fisher_p']:.3g}", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"대 {label[r['vs']]}: {scope.get(r['scope'], r['scope'])} ({r['k1']}/{r['n']} 대 {r['k2']}/{r['n']})"
                        for r in rows], fontsize=8)
    ax.set_xlabel("FM-VLA 성공률 빼기 비교 대상 성공률, (percentage point, Newcombe 95% 구간)")
    ax.set_xlim(-25, 118)
    ax.set_title("FM-VLA와의 차이: 0을 넘는 구간은 파란색 (Fisher 양측, 보정 없음)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_diff_forest.png", dpi=200)
    plt.close(fig)


def plot_tokens():
    K = [4, 8, 16, 32]
    ks = [2, 14, 10, 6]  # 그림 3 라벨 11.1, 77.8, 55.6, 33.3 (그림 판독 값), 18회 가정
    pct = np.array([100 * k / 18 for k in ks])
    lo = np.array([wilson(k, 18)[0] for k in ks])
    hi = np.array([wilson(k, 18)[1] for k in ks])
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    xs = np.arange(4)
    ax.bar(xs, pct, width=0.5, color=ACC)
    ax.errorbar(xs, pct, yerr=np.vstack([pct - lo, hi - pct]), fmt="none", ecolor=INK, capsize=3, lw=0.8)
    for xi, p in zip(xs, pct):
        ax.text(xi, 3, f"{p:.1f}", ha="center", color="white", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"K = {k}" for k in K])
    ax.set_ylabel("Wipe 성공률 (%)")
    ax.set_ylim(0, 108)
    ax2 = ax.twinx()
    tot = [30 + k + 1 for k in K]
    ax2.plot(xs, tot, color=WARM, marker="s", lw=1.2)
    ax2.axhline(50, color=WARM, ls="--", lw=0.8)
    ax2.text(-0.4, 51.5, "본문이 드는 한계 50", color=WARM, fontsize=8, ha="left")
    for xi, t in zip(xs, tot):
        ax2.text(xi + 0.06, t - 3.4, str(t), color=WARM, fontsize=8)
    ax2.set_ylabel("접미 토큰 수 (잡음 행동 30 + K + 상태 1)", color=WARM)
    ax2.set_ylim(0, 108)
    ax.set_title("힘 토큰 수 K와 성공률 (그림 3 라벨, 그림 판독 값·18회 가정, Wilson 95% 구간)", fontsize=9.5)
    clean(ax)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_token_ablation.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_forest()
    plot_tokens()
    print("wrote own_table1_ci.png own_diff_forest.png own_token_ablation.png in", FIG)
