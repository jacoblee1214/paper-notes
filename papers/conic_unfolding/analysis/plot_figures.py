"""우리가 만든 그래프를 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1.png   : 표 1 반복 수 중앙값 [IQR] (로그 축)과 성공 수
own_batch.png    : 표 2 배치 시간을 인스턴스당 비용으로 환산하고 표 6 단일 인스턴스 값과 비교
own_fig1_ci.png  : 그림 1 SCP 20 시나리오 성공 수와 Wilson 95% 구간
own_ood.png      : 표 3 OOD 속도비 (반복, 시간)
own_toy_psd.png  : 토이 T1 PSD 투영 역전파 오차 (float64, float32)
own_toy_du.png   : 토이 T2 max-cut n=12 반복 수와 학습된 일정
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#d9822b"

from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402


def logfmt(ax, which="y"):
    """로그 축 눈금을 1e-3 꼴 글자로 쓴다 (한글 폰트에 수학 마이너스 글리프가 없어서)."""
    f = FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}" if v > 0 else "")
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_formatter(f)
    axis.set_minor_formatter(NullFormatter())

T1 = {  # 중앙값, Q1, Q3, 성공 수 (표 1)
    "Lyapunov": {"상수": (590, 397, 998, 100), "적응형": (788, 384, 1904, 47), "학습": (235, 182, 342, 100)},
    "Max Cut": {"상수": (212, 200, 224, 100), "적응형": (550, 404, 1034, 100), "학습": (69, 65, 79, 100)},
    "Lovasz theta": {"상수": None, "적응형": (1449, 134, 7848, 77), "학습": (74, 69, 81, 100)},
    "Robust KF": {"상수": (314, 312, 317, 100), "적응형": (314, 312, 317, 100), "학습": (26, 24, 32, 100)},
    "Robust MPC": {"상수": (521, 380, 708, 99), "적응형": (81, 59, 102, 100), "학습": (16, 14, 19, 100)},
    "Double Int.": {"상수": (59, 37, 87, 100), "적응형": (684, 367, 1534, 49), "학습": (18, 10, 31, 100)},
    "Unicycle": {"상수": (3954, 2672, 5306, 90), "적응형": None, "학습": (63, 60, 68, 99)},
}


def plot_table1():
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    arms = ["상수", "적응형", "학습"]
    cols = [GREY, "#b9b6ab", ACC]
    w = 0.26
    x = np.arange(len(T1))
    for j, arm in enumerate(arms):
        for i, (cls, d) in enumerate(T1.items()):
            v = d[arm]
            xx = i + (j - 1) * w
            if v is None:
                ax.text(xx, 12, "0/100", ha="center", va="bottom", fontsize=7, color=INK, rotation=90)
                continue
            ax.bar(xx, v[0], width=w * 0.92, color=cols[j], label=arm if i == 0 else None)
            ax.errorbar(xx, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="none", ecolor=INK, capsize=2, lw=0.9)
            if v[3] < 100:
                ax.text(xx, v[2] * 1.12, f"{v[3]}/100", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_yscale("log")
    logfmt(ax)
    ax.set_ylim(8, 2e4)
    ax.set_xticks(x)
    ax.set_xticklabels(list(T1), fontsize=9)
    ax.set_ylabel("ε = 1e-3 도달 반복 수 (중앙값, 막대 = IQR)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    ax.set_title("표 1 (논문 값): 성공이 100/100이 아닌 칸에 성공 수를 적음", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1.png", dpi=170)
    plt.close(fig)


def plot_batch():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V4_table2"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, cls, title in zip(axes, ["Max Cut", "Lovasz theta"], ["Max Cut (n=50)", "Lovasz theta (n=100)"]):
        r = rec[cls]
        xs = np.arange(2)
        for k, (arm, col, lab) in enumerate((("learned", ACC, "학습 (배치, GPU)"), ("cosmojl", ORG, "COSMO.jl (순차)"),
                                             ("clarabel", GREY, "Clarabel (순차)"))):
            ys = [r[arm]["per_instance_B16"], r[arm]["per_instance_B64"]]
            ax.bar(xs + (k - 1) * 0.26, ys, width=0.24, color=col, label=lab)
        ax.axhline(r["table6_cosmojl_per_instance"], color=ORG, ls="--", lw=1, label="표 6의 COSMO.jl 단일 인스턴스")
        ax.set_yscale("log")
        logfmt(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels(["B = 16", "B = 64"])
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("인스턴스당 시간 (s) = 배치 시간 / B")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("표 2를 인스턴스당 비용으로 환산 (논문 값에서 계산)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_batch.png", dpi=170)
    plt.close(fig)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_fig1():
    labs = ["COSMO", "Clarabel", "학습 COSMO"]
    ks = [4, 12, 16]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    pct = [100 * k / 20 for k in ks]
    lo = [wilson(k, 20)[0] for k in ks]
    hi = [wilson(k, 20)[1] for k in ks]
    ax.bar(range(3), pct, color=[GREY, ORG, ACC], width=0.6)
    ax.errorbar(range(3), pct, yerr=[np.array(pct) - lo, np.array(hi) - pct], fmt="none", ecolor=INK, capsize=4, lw=1)
    for i, k in enumerate(ks):
        ax.text(i, 3, f"{k}/20", ha="center", color="white", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs)
    ax.set_ylabel("성공 시나리오 비율 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("그림 1 (20 시나리오): Wilson 95% 구간\n학습 대 Clarabel Fisher p = 0.30, 짝 검정 p = 0.125", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig1_ci.png", dpi=170)
    plt.close(fig)


def plot_ood():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))["V5_table3"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3), sharey=False)
    for ax, cls in zip(axes, ["Max Cut", "Robust KF", "Robust MPC"]):
        rows = [r for r in rec if r["class"] == cls]
        xs = ["1×", "5×", "10×"]
        ax.plot(xs, [r["iter_reported"] for r in rows], "o-", color=ACC, label="반복 (보고)")
        ax.plot(xs, [r["time_reported"] for r in rows], "s-", color=ORG, label="시간 (보고)")
        ax.plot(xs, [r["iter_ratio_of_medians"] for r in rows], "o", mfc="none", color=ACC, label="반복 (중앙값의 비)")
        ax.plot(xs, [r["time_ratio_of_medians"] for r in rows], "s", mfc="none", color=ORG, label="시간 (중앙값의 비)")
        ax.set_title(cls, fontsize=9)
        ax.set_ylim(0, 14)
    axes[0].set_ylabel("속도비 (배)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("표 3: 보고된 쌍별 중앙값 속도비(채움)와 중앙값의 비(빈 표식), 20 인스턴스", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_ood.png", dpi=170)
    plt.close(fig)


def plot_toy_psd():
    r = json.loads((ANA / "toy_dk_cg_checks_results.json").read_text(encoding="utf-8"))
    gaps = ["1e+00", "1e-02", "1e-04", "1e-06", "1e-08"]
    g64 = r["a_gap_sweep"]
    g32 = r["a32_float32_vs_float64_reference"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
    xs = [float(g) for g in gaps]
    ax = axes[0]
    ax.plot(xs, [g64[g]["dk"]["median_rel_err"] for g in gaps], "o-", color=ACC, label="DK")
    ax.plot(xs, [g64[g]["autograd"]["median_rel_err"] for g in gaps], "x--", color=INK, label="autograd (eigh)")
    ax.plot(xs, [g64[g]["denom_1e-6"]["median_rel_err"] for g in gaps], "^-", color=ORG, label="분모 정규화 ε=1e-6")
    ax.set_title("float64 (중심 차분 기준, 한계 약 1e-8)", fontsize=9)
    ax = axes[1]
    ax.plot(xs, [g32[g]["dk"] for g in gaps], "o-", color=ACC, label="DK")
    ax.plot(xs, [g32[g]["autograd"] for g in gaps], "x--", color=INK, label="autograd (eigh)")
    ax.set_title("float32 (float64 DK 기울기 기준)", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        logfmt(ax)
        logfmt(ax, "x")
        ax.invert_xaxis()
        ax.set_xlabel("고유값 간격 δ")
    axes[0].set_ylabel("상대 기울기 오차 (중앙값)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("토이 T1: 식 14 군집 행렬(n=40)에서 PSD 투영 역전파 오차 (우리가 계산)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_psd.png", dpi=170)
    plt.close(fig)


def plot_toy_du():
    r = json.loads((ANA / "toy_du_maxcut_results.json").read_text(encoding="utf-8"))
    b = r["baselines_test"]
    names = ["기본값\n(100, 0.1, 1.6)", "격자 탐색\n상수", "적응형\n(게이트 5)", "학습\n(개루프)"]
    vals = [b["default_const(100,0.1,1.6)"], b["grid_tuned_const"], b["adaptive_residual_balance_gate5"], r["learned_open_loop_test"]]
    med = [v["median_charged"] for v in vals]
    q = [v["q1_q3_solved_only"] for v in vals]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.bar(range(4), med, color=[GREY, GREY, GREY, ACC], width=0.6)
    ax.errorbar(range(4), med, yerr=[[m - x[0] for m, x in zip(med, q)], [x[1] - m for m, x in zip(med, q)]],
                fmt="none", ecolor=INK, capsize=3)
    for i, m in enumerate(med):
        ax.text(i, 3, f"{m:.0f}", ha="center", color="white", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("ε = 1e-3 도달 반복 수 (중앙값, 막대 = IQR)")
    ax.set_title("max-cut n=12, 새 그래프 100개 (모두 수렴)", fontsize=9)
    ax = axes[1]
    s = r["learned_schedule_every_10"]
    ks = np.arange(0, 60, 10)
    ax.plot(ks, s["rho_eq"], "o-", color=ACC, label="ρ_eq")
    ax.set_yscale("log")
    logfmt(ax)
    ax.plot(ks, s["rho_psd"], "s-", color=ORG, label="ρ_psd")
    ax2 = ax.twinx()
    ax2.plot(ks, s["alpha"], "^--", color=INK, label="α")
    ax2.set_ylim(1.0, 1.9)
    ax.set_xlabel("반복 k")
    ax.set_title("학습된 일정 (10 반복마다)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc=(0.02, 0.5))
    ax2.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_du.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_batch()
    plot_fig1()
    plot_ood()
    plot_toy_psd()
    plot_toy_du()
    print("wrote figures in", FIG)
