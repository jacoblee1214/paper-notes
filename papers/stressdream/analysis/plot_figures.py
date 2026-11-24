"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_recall_policy.png : 회수율(그림 5, n = 100 가정)과 정책 성공률(그림 8, n = 120 가정)의 Wilson 95% 구간
own_task_scatter.png  : 그림 22의 과제별 회수율 대 '정답이 실패 아닌 궤적에서 실패를 상상한 비율' (그림 판독 값)
own_typical_toy.png   : 토이 T1, 정규화 계수 배수에 따른 세 통계의 이탈과 보상 이득
own_drift_budget.png  : 반복당 노름 클립이 허용하는 잡음 이동 상한 (알고리즘 1 대 코드 설정)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 analysis/recompute.py 와 analysis/toy_typical_set.py 를 실행해 결과 JSON을 만들어 둘 것.
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#e08a00"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def plot_recall_policy():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    labels = ["Nominal\n(N=1)", "Best-of-N", "StressDream"]
    ks = [54, 71, 94]
    cols = [GREY, ACC, ORG]
    for i, (k, c) in enumerate(zip(ks, cols)):
        lo, hi = wilson(k, 100)
        ax.bar(i, k, color=c, width=0.6)
        ax.errorbar(i, k, yerr=[[k - lo], [hi - k]], color=INK, capsize=4, lw=1.2)
        ax.text(i, hi + 2, f"{k}%\n[{lo:.0f}, {hi:.0f}]", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, 118)
    ax.set_ylabel("실패 궤적 회수율 (%)")
    ax.set_title("조종 회수율 (n = 100 가정)", fontsize=11, color=INK)
    clean(ax)
    ax = axes[1]
    ks = [47, 85]
    labels = ["Nominal", "Robust"]
    for i, (k, c) in enumerate(zip(ks, [GREY, ORG])):
        lo, hi = wilson(k, 120)
        ax.bar(i, 100 * k / 120, color=c, width=0.55)
        ax.errorbar(i, 100 * k / 120, yerr=[[100 * k / 120 - lo], [hi - 100 * k / 120]], color=INK, capsize=4, lw=1.2)
        ax.text(i, hi + 2, f"{100 * k / 120:.1f}%\n[{lo:.0f}, {hi:.0f}]", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(2), labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("정책 성공률 (%)")
    ax.set_title("π0.5 미세조정 (6과제 x 20회 = 120 가정)", fontsize=11, color=INK)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_recall_policy.png", dpi=180)
    plt.close(fig)


OFFS = {"Coffee Bean": (-4.0, 2.2), "Knife Put": (0.8, 2.2), "Coffee Bag": (1.2, -3.0), "Block Stack": (1.2, -3.0),
        "Utensil Pick": (1.2, -3.0), "Candy Bag": (1.5, -3.0)}


def plot_task_scatter():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V8_fig22"]
    L = rec["그림22L_판독"]
    Rr = rec["그림22R_판독"]
    tasks = ["Block Stack", "Utensil Pick", "Knife Put", "Coffee Bean", "Coffee Bag", "Candy Bag"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    key = [("nominal", "Nominal (N=1)", GREY, "s"), ("bon", "Best-of-N (N=20)", ACC, "^"), ("stressdream", "StressDream", ORG, "o")]
    for i, t in enumerate(tasks):
        xs = [Rr[k][i] for k, *_ in key]
        ys = [L[k][i] for k, *_ in key]
        ax.plot(xs, ys, color="#cfcdc6", lw=1, zorder=1)
        dx, dy = OFFS.get(t, (1.2, -3.0))
        ax.text(xs[2] + dx, ys[2] + dy, t, fontsize=7.5, color=INK)
    for k, lab, c, m in key:
        ax.scatter(Rr[k], L[k], color=c, marker=m, s=46, label=lab, zorder=3, edgecolor="white", linewidth=0.6)
    ax.plot([0, 100], [0, 100], color="#cfcdc6", ls="--", lw=0.9)
    ax.set_xlim(-3, 58)
    ax.set_ylim(20, 104)
    ax.set_xlabel("정답이 실패 아닌 궤적 중 WM이 실패를 상상한 비율 (%, 그림 22 오른쪽 판독)")
    ax.set_ylabel("정답 실패 궤적 회수율 (%, 그림 22 왼쪽 판독)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_task_scatter.png", dpi=180)
    plt.close(fig)


def plot_toy():
    r = json.loads((ANA / "toy_typical_set_results.json").read_text(encoding="utf-8"))
    base = r["기준(무작위 가우시안 200개)"]
    sens = r["계수민감도"]
    mult = [s["계수배수"] for s in sens]
    xs = np.arange(len(mult))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    z_spec = [(s["spec"] - base["spec"][0]) / base["spec"][1] for s in sens]
    z_iso = [(s["iso"] - base["iso"][0]) / base["iso"][1] for s in sens]
    z_norm = [s["norm_dev_sd"] / 1.0 for s in sens]
    ax.plot(xs, z_spec, "-o", color=ORG, label="스펙트럼 (기준 표준편차 배)")
    ax.plot(xs, z_iso, "-s", color=ACC, label="등방성")
    ax.plot(xs, z_norm, "-^", color=GREY, label="노름 (표준편차 배)")
    ax.axhline(0, color="#cfcdc6", lw=0.9)
    ax.set_xticks(xs, [f"x{m:g}" for m in mult])
    ax.set_xlabel("세 항 계수 배수 (x1 = Dubins 표 3의 1.0, 0.5, 5.0)")
    ax.set_ylabel("가우시안 기준 대비 이탈")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([-1, 0, 1, 3, 10, 30, 50])
    ax.set_yticklabels(["-1", "0", "1", "3", "10", "30", "50"])
    ax.set_title("매끈한 저주파 보상 방향, 10회 이동", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8)
    clean(ax)
    ax = axes[1]
    gain = [s["보상이득(sd)"] for s in sens]
    ax.bar(xs, gain, color=ORG, width=0.55)
    for x, g in zip(xs, gain):
        ax.text(x, g + 0.15, f"{g:.1f}", ha="center", fontsize=9)
    ax.set_xticks(xs, [f"x{m:g}" for m in mult])
    ax.set_xlabel("세 항 계수 배수")
    ax.set_ylabel("선형 보상 이득 (표준편차 단위)")
    ax.set_ylim(0, 11.5)
    ax.set_title("보상 이득", fontsize=10.5, color=INK)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_typical_toy.png", dpi=180)
    plt.close(fig)


def plot_drift():
    r = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V4_drift"]
    names = ["Vista", "Ctrl-World", "Dubins"]
    alg = [r[n]["알고리즘1_상한/거리_%"] for n in names]
    code = [r[n]["코드_상한/거리_%"] for n in names]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - 0.2, alg, 0.38, color=ACC, label="알고리즘 1 (노름 클립 1.0)")
    ax.bar(x + 0.2, code, 0.38, color=ORG, label="공개 설정 (노름 클립 20 / 20 / 5)")
    for xi, a, c in zip(x, alg, code):
        ax.text(xi - 0.2, a + 2, f"{a:.1f}", ha="center", fontsize=8.5)
        ax.text(xi + 0.2, c + 2, f"{c:.0f}", ha="center", fontsize=8.5)
    ax.axhline(100, color="#cfcdc6", ls="--", lw=0.9)
    ax.text(-0.45, 103, "독립 두 표본 사이 거리 (100%)", fontsize=7.5, color=GREY, ha="left")
    ax.set_xticks(x, names)
    ax.set_ylabel("이동 상한 / sqrt(2D) (%)")
    ax.set_ylim(0, 128)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_drift_budget.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_recall_policy()
    plot_task_scatter()
    plot_toy()
    plot_drift()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
