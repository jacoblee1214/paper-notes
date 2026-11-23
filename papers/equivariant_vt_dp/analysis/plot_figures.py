"""우리가 만든 그래프를 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sim_ci.png    : 표 1 과제별 성공률과 Wilson 95% 구간 (과제당 40회)
own_ablation.png  : 표 9 과제 평균(8과제)과 Lift Bottle를 뺀 7과제 평균
own_data_eff.png  : 표 10 데모 수 50·100에서의 과제 평균과 95% 구간
own_real_ci.png   : 표 2 실로봇 과제별 성공 횟수와 Wilson 95% 구간 (과제당 10회)
own_toy_equiv.png : 토이 T3, 센서 국소 불변 가정이 깨지는 정도에 따른 동변성 오차
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 recompute.py, toy_equivariance.py를 실행해 둘 것.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import recompute as rc  # noqa: E402

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
INK, ACC, GREY, ORG, GRN = "#1b1b1a", "#1f5fae", "#9a9890", "#c4661f", "#3b8a5a"
TASK_SHORT = ["Grasp", "HDMI", "Hole", "Tube", "Bottle", "Can", "Key", "Shelf"]


def ks(row):
    return rc.counts(row)[0]


def mean_ci(k, n=rc.N_SIM):
    """과제 평균과 95% 구간 (Agresti-Coull 분산)."""
    t = len(k)
    m = np.mean(k) / n * 100
    se = math.sqrt(sum(rc.ac_var(a, n) for a in k)) / t * 100
    return m, m - 1.959964 * se, m + 1.959964 * se


def plot_sim():
    cols = {"ManiFeel": GREY, "VITAL": "#c9b458", "UniVTAC": "#b98a6a", "EquiDiff + Tac": GRN, "VISTA": ACC}
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    names = list(rc.T7)
    off = np.linspace(-0.3, 0.3, len(names))
    for j, m in enumerate(names):
        k = ks(rc.T7[m])
        pct = [100 * a / rc.N_SIM for a in k]
        lo = [rc.wilson(a, rc.N_SIM)[0] for a in k]
        hi = [rc.wilson(a, rc.N_SIM)[1] for a in k]
        x = np.arange(8) + off[j]
        ax.errorbar(x, pct, yerr=[np.array(pct) - lo, np.array(hi) - pct], fmt="o", ms=4, lw=1,
                    color=cols[m], label=m, capsize=0)
    ax.set_xticks(range(8))
    ax.set_xticklabels(TASK_SHORT)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(-3, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title("표 1 재작성: 과제별 성공률과 Wilson 95% 구간 (40회, 학습 시드 1개)", fontsize=10, pad=26)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_ci.png", dpi=170)
    plt.close(fig)


def plot_ablation():
    names = list(rc.T9)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, drop, title in [(axes[0], None, "8과제 평균 (논문 그림 4 왼쪽과 같은 값)"),
                            (axes[1], 4, "Lift Bottle 제외 7과제 평균")]:
        for i, m in enumerate(names):
            k = ks(rc.T9[m])
            if drop is not None:
                k = [a for j, a in enumerate(k) if j != drop]
            mean, lo, hi = mean_ci(k)
            ax.bar(i, mean, color=ACC if m == "VISTA" else GREY, width=0.62)
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor=INK, capsize=3, lw=1)
            ax.text(i, 3, f"{mean:.1f}", ha="center", color="white", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["ISP", "global-l0", "Early", "Concat", "VISTA"], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("과제 평균 성공률 (%)")
    fig.suptitle("표 9 재작성: 융합 절제. 막대는 평균, 선은 95% 구간 (롤아웃 변동만)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=170)
    plt.close(fig)


def plot_data_eff():
    cols = {"ManiFeel": GREY, "VITAL": "#c9b458", "UniVTAC": "#b98a6a", "EquiDiff + Tac": GRN, "VISTA": ACC}
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for m, c in cols.items():
        ms, los, his = [], [], []
        for d in (50, 100):
            mean, lo, hi = mean_ci(ks(rc.T10[(m, d)]))
            ms.append(mean); los.append(lo); his.append(hi)
        x = np.array([50, 100]) + (list(cols).index(m) - 2) * 2.0
        ax.errorbar(x, ms, yerr=[np.array(ms) - los, np.array(his) - ms], fmt="o-", color=c, label=m, lw=1.4, ms=5, capsize=2)
    ax.set_xticks([50, 100])
    ax.set_xlabel("과제당 데모 수")
    ax.set_ylabel("8과제 평균 성공률 (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("표 10 재작성: 데모 수별 평균과 95% 구간", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_data_eff.png", dpi=170)
    plt.close(fig)


def plot_real():
    tasks = list(rc.T2["VISTA"])
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    for j, (m, c) in enumerate([("EquiDiff + Tac", GRN), ("VISTA", ACC)]):
        k = [rc.T2[m][t] for t in tasks]
        pct = [10 * a for a in k]
        lo = [rc.wilson(a, rc.N_REAL)[0] for a in k]
        hi = [rc.wilson(a, rc.N_REAL)[1] for a in k]
        x = np.arange(3) + (j - 0.5) * 0.3
        ax.errorbar(x, pct, yerr=[np.array(pct) - lo, np.array(hi) - pct], fmt="o", color=c, label=m, capsize=3, lw=1.2)
        for xi, a in zip(x, k):
            ax.text(xi + 0.04, 100 * a / 10 + 3, f"{a}/10", fontsize=7, color=c)
    ax.set_xlim(-0.5, 2.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(tasks)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(-3, 108)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.set_title("표 2 재작성: 실로봇 성공 횟수와 Wilson 95% 구간 (과제당 10회)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_toy():
    res = json.loads((ANA / "toy_equivariance_results.json").read_text(encoding="utf-8"))
    b = np.array(res["T3_beta_sweep"]["betas"][1:])
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.loglog(b, res["T3_beta_sweep"]["attention_mean_rel_err"][1:], "o-", color=ACC, label="교차 주의 융합 (식 2–4)")
    ax.loglog(b, res["T3_beta_sweep"]["concat_mean_rel_err"][1:], "s--", color=ORG, label="단순 연결")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.01", "0.1", "1"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.01", "0.1", "1"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.005, 1.5)
    ax.set_xlabel("β: 센서 국소 특징이 손목 자세에 새는 정도 (토이의 임의 단위)")
    ax.set_ylabel("동변성 상대 오차")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    ax.set_title("토이: 센서 국소 불변 가정이 깨질 때의 오차 (β = 0이면 두 융합 모두 약 1e-15)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_equiv.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_sim()
    plot_ablation()
    plot_data_eff()
    plot_real()
    plot_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
