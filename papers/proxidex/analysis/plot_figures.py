"""우리가 만든 그래프를 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sim_gap.png   : 표 6 과제별 ProxiDex - CordViP 차이(점)와 시드 수 가정(3, 5)에 따른 Welch 95% 구간
own_real_ci.png   : 표 2 합산 성공률(310회 중)과 Wilson 95% 구간, 방법별
own_occlusion.png : 그림 5에서 읽은 가림 회복률과 Wilson 95% 구간 (시행 수 100 가정)
own_hdyn.png      : 표 8 Hdyn별 6과제 평균과 Toilet을 뺀 5과제 평균
own_toy_noise.png : 토이, 물체 자세 오차에 따른 근접도 γ 오차와 단계 라벨 뒤바뀜 비율
실행: python3 analysis/plot_figures.py   (numpy, scipy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 recompute.py, toy_proximity_noise.py를 실행해 둘 것.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

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


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def welch_ci(d, s1, s2, k):
    se = math.sqrt(s1 ** 2 / k + s2 ** 2 / k)
    if se == 0:
        return d, d
    df = se ** 4 / ((s1 ** 2 / k) ** 2 / (k - 1) + (s2 ** 2 / k) ** 2 / (k - 1))
    t = stats.t.ppf(0.975, df)
    return d - t * se, d + t * se


def plot_sim_gap():
    mu_p, sd_p = np.array(rc.T6["ProxiDex"][0]), np.array(rc.T6["ProxiDex"][1])
    mu_c, sd_c = np.array(rc.T6["CordViP"][0]), np.array(rc.T6["CordViP"][1])
    d = mu_p - mu_c
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    x = np.arange(10)
    for k, col, off in ((3, GREY, -0.12), (5, ACC, 0.12)):
        lo, hi = zip(*[welch_ci(d[j], sd_p[j], sd_c[j], k) for j in range(10)])
        ax.errorbar(x + off, d, yerr=[d - np.array(lo), np.array(hi) - d], fmt="o", color=col, ms=5, capsize=3, lw=1.3,
                    label=f"시드 {k}개 가정의 95% 구간")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(rc.T6_TASKS, rotation=0, fontsize=9)
    ax.set_ylabel("ProxiDex - CordViP (점)", color=INK)
    ax.set_title("표 6 과제별 격차와, 시드 수를 가정했을 때의 구간 (논문은 시드 수 미기재)", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_sim_gap.png", dpi=170)
    plt.close(fig)


def plot_real_ci():
    names = ["DP3", "ManiFlow", "AFRO", "CordViP", "ProxiDex-1", "ProxiDex-2", "ProxiDex"]
    labels = ["DP3", "ManiFlow", "AFRO", "CordViP", "ProxiDex-1\n(U-Net)", "ProxiDex-2\n(표준 Transformer)", "ProxiDex"]
    tot = sum(rc.N2)
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    for i, m in enumerate(names):
        s = sum(rc.T2[m])
        lo, hi = rc.wilson(s, tot)
        p = s / tot
        col = ACC if m == "ProxiDex" else (ORG if m.startswith("ProxiDex-") else GREY)
        ax.errorbar(i, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="o", color=col, ms=6, capsize=4, lw=1.4)
        ax.text(i, 100 * hi + 1.5, f"{s}/{tot}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(30, 85)
    ax.set_ylabel("합산 성공률 (%)", color=INK)
    ax.set_title("표 2를 11개 설정으로 합친 성공률과 Wilson 95% 구간 (310회, 시행이 독립이라고 가정)", fontsize=10.5, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=170)
    plt.close(fig)


def plot_occlusion():
    n = 100
    durs = ["2 s", "4 s", "6 s"]
    keys = ["w/o FDM", "w/o IDM", "w/o Prox", "w/o Trajectory Gating", "ProxiDex"]
    cols = ["#d4d3cc", "#aeada5", "#7e7d77", "#b7a3d6", ACC]
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    w = 0.16
    for j, k in enumerate(keys):
        for i in range(3):
            v = rc.FIG5[k][i]
            lo, hi = rc.wilson(v, n)
            x = i + (j - 2) * w
            ax.bar(x, v, width=w * 0.92, color=cols[j], label=k if i == 0 else None)
            ax.errorbar(x, v, yerr=[[v - 100 * lo], [100 * hi - v]], color=INK, lw=0.9, capsize=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(durs)
    ax.set_xlabel("가림 시간", color=INK)
    ax.set_ylabel("회복 성공률 (%)", color=INK)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper right")
    ax.set_title("그림 5 판독 값과 Wilson 95% 구간 (그림 판독 값, 시행 수 100 가정)", fontsize=10.5, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_occlusion.png", dpi=170)
    plt.close(fig)


def plot_hdyn():
    hs = [1, 2, 4, 8, 10]
    allm = [np.mean(rc.T8[h][0]) for h in hs]
    exm = [np.mean([v for v, t in zip(rc.T8[h][0], rc.T7_TASKS) if t != "Toilet"]) for h in hs]
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.plot(range(5), allm, "o-", color=ACC, label="6과제 평균 (표 8)")
    ax.plot(range(5), exm, "s--", color=ORG, label="Toilet을 뺀 5과제 평균")
    for i in range(5):
        ax.text(i, allm[i] + 0.12, f"{allm[i]:.1f}", ha="center", fontsize=8.5, color=ACC)
        ax.text(i, exm[i] - 0.4, f"{exm[i]:.1f}", ha="center", fontsize=8.5, color=ORG)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Hdyn = {h}" for h in hs])
    ax.set_ylabel("성공률 (%)", color=INK)
    ax.set_ylim(74, 78.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("표 8: Hdyn = 2의 우위(+0.7)는 Toilet 열이 대부분", fontsize=10.5, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_hdyn.png", dpi=170)
    plt.close(fig)


def plot_toy():
    res = json.loads((ANA / "toy_proximity_noise_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    cols = {"5.0": GREY, "10.0": ACC, "20.0": ORG}
    for sg, row in res["by_sigma"].items():
        s = [float(k) for k in row["by_s"]]
        axes[0].plot(s, [v["mean_abs_err_over_gaps"] for v in row["by_s"].values()], "o-", color=cols[sg], label=f"σ = {float(sg):.0f} mm")
        axes[1].plot(s, [100 * v["phase_flip_rate_gap_le_12mm"] for v in row["by_s"].values()], "o-", color=cols[sg], label=f"σ = {float(sg):.0f} mm")
    axes[0].set_ylabel("γ의 평균 절대 오차", color=INK)
    axes[1].set_ylabel("단계 라벨이 뒤바뀌는 비율 (%)\n(손바닥-물체 간격 12 mm 이하)", color=INK)
    for ax in axes:
        ax.axvline(8, color=GREY, lw=0.8, ls=":")
        ax.axvline(50, color=GREY, lw=0.8, ls=":")
        ax.set_xlabel("물체 자세 병진 오차 RMS (mm)", color=INK)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("토이: 자세 오차 대 근접도 (점선은 논문 그림 18의 8 mm와 50 mm, τ와 σ는 가정)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_noise.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_sim_gap()
    plot_real_ci()
    plot_occlusion()
    plot_hdyn()
    plot_toy()
    print("wrote figures to", FIG)
