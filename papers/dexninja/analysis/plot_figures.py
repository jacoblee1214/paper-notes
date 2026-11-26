"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_main_ci.png      : 표 1 풀링 Progress/Success/Integrity와 Wilson 95% 구간(3개 학습 방식, 각 100회)
own_scaling_ablation.png : 부록 표 3 시뮬 에피소드 수에 따른 성공률(로그 축)과 표 2 절제별 성공률 차이의 Newcombe 95% 구간
own_full_system_repeats.png : 같은 "완성 시스템"의 세 측정(표 1 25회, 부록 표 2·4 각 10회)의 식품별 성공률
own_ranking_mc.png   : toy_ranking_mc.py 결과 (시행 잡음만으로 나올 확률)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 recompute.py와 toy_ranking_mc.py를 돌려 json을 만들어 둔다.
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
INK, ACC, GREY, LIGHT = "#1b1b1a", "#1f5fae", "#9a9890", "#8fb0d9"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def bars_with_ci(ax, ks, n, labels, colors, fs=8):
    pct = [100 * k / n for k in ks]
    lo = [wilson(k, n)[0] for k in ks]
    hi = [wilson(k, n)[1] for k in ks]
    ax.bar(range(len(ks)), pct, color=colors, width=0.62)
    ax.errorbar(range(len(ks)), pct, yerr=[np.array(pct) - np.array(lo), np.array(hi) - np.array(pct)], fmt="none", ecolor=INK, capsize=3, lw=1)
    for i, k in enumerate(ks):
        ax.text(i, 2, f"{k}/{n}", ha="center", va="bottom", color="white", fontsize=fs)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)


def plot_main():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["table1"]["pooled_wilson95"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)
    for ax, (m, title) in zip(axes, [("P", "Progress"), ("S", "Success"), ("I", "Integrity")]):
        ks = [rec[r][m][0] for r in ("Sim-only", "Real-only", "Sim+Real")]
        bars_with_ci(ax, ks, 100, ["Sim-only", "Real-only", "Sim+Real"], [GREY, GREY, ACC])
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("성공 비율 (%)")
    fig.suptitle("표 1: 학습 방식별 풀링 결과(식품 4종 x 25회)와 Wilson 95% 구간", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_main_ci.png", dpi=170)
    plt.close(fig)


def plot_scaling_ablation():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    Ns = [10, 50, 100, 300]
    ks = [36, 50, 56, 63]
    pct = np.array(ks, float)
    lo = np.array([wilson(k, 100)[0] for k in ks])
    hi = np.array([wilson(k, 100)[1] for k in ks])
    ax.errorbar(Ns, pct, yerr=[pct - lo, hi - pct], fmt="o-", color=ACC, capsize=3)
    ax.axhline(31, color=GREY, ls="--", lw=1)
    ax.text(300, 32, "Real-only 31%", ha="right", va="bottom", fontsize=8, color=GREY)
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("추가 시뮬레이션 에피소드 수 N (로그 축)")
    ax.set_ylabel("성공률 (%), 100회")
    ax.set_ylim(20, 80)
    ax.set_title("부록 표 3: 시뮬 에피소드 수와 성공률", fontsize=10)
    ax = axes[1]
    rows = [c for c in rec["table2"]["vs_full"] if c["name"].startswith("S:")]
    labels = [c["name"].split(": ")[1].replace(" vs Full", "") for c in rows]
    d = [c["diff_pp"] for c in rows]
    lo = [c["newcombe95_pp"][0] for c in rows]
    hi = [c["newcombe95_pp"][1] for c in rows]
    y = np.arange(len(rows))[::-1]
    ax.errorbar(d, y, xerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)], fmt="o", color=ACC, ecolor=INK, capsize=3)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("성공률 차이 (%p, Full 기준) 와 Newcombe 95% 구간")
    ax.set_title("표 2: 절제별 성공률 차이 (각 100회)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_scaling_ablation.png", dpi=170)
    plt.close(fig)


def plot_repeats():
    foods = ["Strawberry", "Apple", "Banana", "Potato"]
    data = {"표 1 (25회)": ([12, 8, 18, 12], 25), "부록 표 2 Ours (10회)": ([6, 3, 9, 5], 10), "부록 표 4 Full (10회)": ([7, 5, 7, 5], 10)}
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    w = 0.26
    cols = [ACC, LIGHT, GREY]
    for j, (name, (ks, n)) in enumerate(data.items()):
        x = np.arange(4) + (j - 1) * w
        pct = np.array([100 * k / n for k in ks])
        lo = np.array([wilson(k, n)[0] for k in ks])
        hi = np.array([wilson(k, n)[1] for k in ks])
        ax.bar(x, pct, width=w, color=cols[j], label=name)
        ax.errorbar(x, pct, yerr=[pct - lo, hi - pct], fmt="none", ecolor=INK, capsize=2, lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(foods)
    ax.set_ylim(0, 105)
    ax.set_ylabel("성공률 (%)와 Wilson 95% 구간")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("같은 완성 시스템의 세 측정: 식품별 성공률", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_full_system_repeats.png", dpi=170)
    plt.close(fig)


def plot_mc():
    res = json.loads((ANA / "toy_ranking_mc_results.json").read_text(encoding="utf-8"))
    A, B, C = res["A_observed_as_truth"], res["B_all_equal_truth"], res["C_table1_direction"]
    items = [
        ("A. 관측값이 참값일 때\nFull이 6개 중 최고(동률 포함)", A["P(Full가 6개 중 단독 최고 또는 공동 최고)"]),
        ("A. 관측값이 참값일 때\n궤적 정제 제거가 절제 중 최저(동률 포함)", A["P(w/o TrajRef가 절제 중 최저, 동률 포함)"]),
        ("B. 모두 같은 참값(44%)일 때\n최고-최저 범위가 11 %p 이상", B["P(범위 >= 11 %p)"]),
        ("B. 모두 같은 참값(44%)일 때\n5개 절제가 모두 Full 이하", B["P(5개 절제 모두 Full 이하)"]),
        ("C. 참값 같음(40%)일 때\n네 식품 모두 Sim+Real >= Real-only", C["P(네 식품 모두 Sim+Real >= Real-only | 참값 같음 0.40)"]),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    y = np.arange(len(items))[::-1]
    ax.barh(y, [100 * v for _, v in items], color=[ACC, ACC, GREY, GREY, LIGHT], height=0.6)
    for yi, (_, v) in zip(y, items):
        ax.text(100 * v + 1, yi, f"{100 * v:.0f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([t for t, _ in items], fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("해당 사건이 일어날 확률 (%), 몬테카를로 20만 회")
    ax.set_title("토이: 이항 잡음에 대한 표 1·표 2 결과의 견고성", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_ranking_mc.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_main()
    plot_scaling_ablation()
    plot_repeats()
    plot_mc()
    print("wrote 4 figures to", FIG)
