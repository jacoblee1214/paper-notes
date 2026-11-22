"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1.png    : 표 1의 목표 오차(평균 ± SD)와 안전율을 과제·방법별로 다시 그림
own_realtime.png  : 표 1의 단계당 계획 시간과 시뮬 시간 간격 dt = 0.02 s (실시간 계수)
own_cp_toy.png    : 토이 T1, 논문의 87% 반지름을 쓸 때 길이 T 궤적이 관 안에 머물 확률
own_smoothing.png : 접촉 평활화 가중치(코드 exp(-d/κ) 대 논문 exp(-d²/κ))와 하드 접촉 반경
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
plt.rcParams["mathtext.fontset"] = "dejavusans"
INK, ACC, GREY, RED, GREEN = "#1b1b1a", "#1f5fae", "#9a9890", "#b3412f", "#3f7f4f"
COL = {"Ours": ACC, "MPPI": GREY, "PPO": "#d89b3c", "APG": GREEN}

# 표 1 (본문 8쪽): 목표 오차 평균, SD, 안전율 평균, SD, 단계당 시간 s
T1 = {
    "Lift rope": {"Ours": (0.0548, 0.0003, 100.0, 0.0, 0.050), "MPPI": (0.0681, 0.0038, 100.0, 0.0, 0.177),
                  "PPO": (0.0587, 0.0016, 96.5, 0.2, 0.0004), "APG": (0.0573, 0.0017, 100.0, 0.0, 0.0004)},
    "Drag rope": {"Ours": (0.0360, 0.0077, 99.9, 0.1, 0.070), "MPPI": (0.0814, 0.0034, 1.0, 0.1, 0.234),
                  "PPO": (0.0320, 0.0092, 97.0, 0.4, 0.0004), "APG": (0.0302, 0.0061, 93.0, 0.4, 0.0004)},
    "Fold cloth": {"Ours": (0.0611, 0.0072, 100.0, 0.0, 0.758), "MPPI": (0.0738, 0.0079, 64.0, 6.1, 18.81)},
    "Flatten cloth": {"Ours": (0.0234, 0.0089, 100.0, 0.0, 0.711), "MPPI": (0.0526, 0.0102, 80.0, 0.2, 11.10)},
}


def plot_table1():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    tasks = list(T1)
    for ax, (idx_m, idx_s, title, ylab) in zip(axes, [(0, 1, "목표 오차 (평균 ± 표시된 SD, 시도 수 미공개)", "목표 오차 (m)"),
                                                     (2, 3, "안전율 (평균 ± 표시된 SD)", "안전율 (%)")]):
        x0 = np.arange(len(tasks))
        w = 0.19
        for j, m in enumerate(["Ours", "MPPI", "PPO", "APG"]):
            xs, ys, es = [], [], []
            for i, t in enumerate(tasks):
                if m in T1[t]:
                    xs.append(i + (j - 1.5) * w)
                    ys.append(T1[t][m][idx_m])
                    es.append(T1[t][m][idx_s])
            ax.bar(xs, ys, width=w * 0.92, color=COL[m], label=m)
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, capsize=2, lw=0.9)
        ax.set_xticks(x0)
        ax.set_xticklabels(tasks, fontsize=8)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[1].set_ylim(0, 108)
    axes[0].legend(fontsize=8, frameon=False)
    fig.text(0.5, -0.02, "Drag rope에서는 PPO·APG의 목표 오차가 Ours보다 작다. 천 과제에는 RL 기준선이 없다. 우리가 표 1 값을 다시 그림.", ha="center", fontsize=8, color="#55544f")
    fig.tight_layout()
    fig.savefig(FIG / "own_table1.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_realtime():
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    tasks = list(T1)
    w = 0.19
    for j, m in enumerate(["Ours", "MPPI", "PPO", "APG"]):
        xs, ys = [], []
        for i, t in enumerate(tasks):
            if m in T1[t]:
                xs.append(i + (j - 1.5) * w)
                ys.append(T1[t][m][4])
        ax.bar(xs, ys, width=w * 0.92, color=COL[m], label=m)
    ax.axhline(0.02, color=RED, lw=1.2, ls="--")
    ax.text(1.55, 0.0235, "시뮬 dt = 0.02 s (코드)", color=RED, fontsize=8, ha="center")
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}" if v > 0 else ""))
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylabel("단계당 시간 (s, 로그 축)", fontsize=9)
    ax.set_title("표 1 단계당 실행 시간: 막대가 점선 위이면 시뮬 시간보다 느림", fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_realtime.png", dpi=170)
    plt.close(fig)


def plot_cp():
    r = json.loads((ANA / "toy_cp_coverage_results.json").read_text(encoding="utf-8"))["c_paper_87pct_radius"]
    Ts = [5, 10, 20, 50, 200]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for rho, c in [("0.0", RED), ("0.5", "#d89b3c"), ("0.9", GREEN), ("0.99", ACC)]:
        ax.plot(range(len(Ts)), [r[f"T={T}"][f"rho={rho}"] for T in Ts], "o-", color=c, label=f"시간 상관 ρ = {rho}")
    ax.axhline(0.87, color=GREY, ls=":", lw=1)
    ax.text(2.0, 0.845, "단계별 87% (상한)", fontsize=8, color=GREY, ha="center")
    ax.axhline(0.9, color=INK, ls="--", lw=0.8)
    ax.text(len(Ts) - 1, 0.92, "목표 1 - δ = 0.9", fontsize=8, ha="right")
    ax.set_xticks(range(len(Ts)))
    ax.set_xticklabels([f"T = {T}" for T in Ts], fontsize=8)
    ax.set_ylim(-0.03, 1.02)
    ax.set_ylabel("궤적 전체가 관 안에 머물 확률", fontsize=9)
    ax.set_title("토이: 단계별 87%를 덮는 반지름의 궤적 수준 포함 확률", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_cp_toy.png", dpi=170)
    plt.close(fig)


def plot_smoothing():
    d = np.linspace(0, 25, 400)
    kappa = 1.0  # mm
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.semilogy(d, np.exp(-d / kappa), color=ACC, label="코드: exp(-d/κ)")
    ax.semilogy(d, np.exp(-(d * 1e-3) ** 2 / 1e-3), color=GREY, label="논문 §4.1: exp(-|c|^2/κ) (거리 m 단위로 가정)")
    ax.axvline(17, color=RED, ls="--", lw=1)
    ax.text(17.3, 1e-6, "Lift rope 하드 접촉 반경\n(밧줄 2 + 그리퍼 15 mm)", fontsize=8, color=RED)
    ax.axvline(7, color=GREEN, ls="--", lw=1)
    ax.text(7.3, 1e-14, "Drag rope 7 mm", fontsize=8, color=GREEN)
    ax.set_xlabel("그리퍼와 밧줄의 최근접 거리 d (mm), κ = 1e-3", fontsize=9)
    ax.set_ylabel("그립 강성 가중치 (최대 대비)", fontsize=9)
    ax.set_ylim(1e-18, 2)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}" if v > 0 else ""))
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_smoothing.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_table1()
    plot_realtime()
    plot_cp()
    plot_smoothing()
    print("wrote 4 figures in", FIG)
