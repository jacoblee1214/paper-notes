"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_ci.png   : 표 1의 성공률과 Wilson 95% 구간 (과제별, 방법 네 개)
own_fig5_read.png   : 논문 그림 5에서 읽은 값(그림 판독 값)을 다시 그린 곡선과 누적 개입량 대용 비율
own_single_seed.png : 토이 1, 시드 SD에 따른 Fisher 검정의 '유의' 비율
own_hil_toy.png     : 토이 2, 복도 환경의 개입 비율 곡선과 총 개입 비율
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
INK, ACC, RED, ORG, GREY = "#1b1b1a", "#1f5fae", "#b84545", "#e88737", "#9a9890"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


T1 = {"Cube": (30, [20, 25, 25, 30]), "LEGO": (30, [19, 24, 23, 29]), "P&P": (30, [13, 19, 18, 27]),
      "Drawer": (20, [11, 14, 15, 19]), "Long Horizon": (20, [7, 13, 12, 17])}
METH = ["BC", "ResFiT", "Residual RL", "WHIRL"]
MCOL = [GREY, "#6d8fb5", RED, ACC]


def fig_table1():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    for j, (m, c) in enumerate(zip(METH, MCOL)):
        for i, (t, (n, ks)) in enumerate(T1.items()):
            k = ks[j]
            lo, hi = wilson(k, n)
            x = i + (j - 1.5) * w
            ax.errorbar(x, 100 * k / n, yerr=[[100 * k / n - lo], [hi - 100 * k / n]], fmt="o", color=c, ms=5, capsize=2,
                        label=m if i == 0 else None)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{t}\n(n={v[0]})" for t, v in T1.items()])
    ax.set_ylabel("성공률 (%) 와 Wilson 95% 구간")
    ax.set_ylim(15, 103)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="lower center", frameon=False)
    ax.set_title("표 1의 성공 수로 다시 그린 성공률 (과제마다 구간이 겹침)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png", dpi=170)
    plt.close(fig)


def fig_fig5():
    d = json.loads((ANA / "fig_digitized.json").read_text())["fig5"]
    r = json.loads((ANA / "recompute_results.json").read_text())["V7_fig5_figure_read"]
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1])
    names = list(d.keys())
    axs = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
           fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4])]
    for ax, t in zip(axs[:5], names):
        for key, c, lab in (("wo_WM", RED, "w/o WM"), ("WHIRL", ACC, "WHIRL")):
            a = np.array(d[t][key])
            ax.plot(a[:, 0], a[:, 1], "-o", color=c, ms=3, lw=1.4, label=lab)
        ax.set_title(f"{t}", fontsize=10)
        ax.set_ylim(-0.02, None)
        ax.grid(alpha=0.2)
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_ylabel("굴림 개입 비율 (그림 판독 값)")
    axs[3].set_ylabel("굴림 개입 비율 (그림 판독 값)")
    ax = fig.add_subplot(gs[1, 4:6])
    red = [r[t]["reduction_pct_common"] for t in names]
    ax.bar(range(5), red, color=[ACC if x > 10 else ORG for x in red])
    ax.axhline(84, color=GREY, ls="--", lw=1)
    ax.text(4.4, 85, "84% (본문 값)", ha="right", va="bottom", fontsize=8, color=GREY)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Cube", "LEGO", "P&P", "Drawer", "LH"], fontsize=8)
    ax.set_ylabel("곡선 아래 면적 감소 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("누적 개입량 대용 (공통 구간)", fontsize=10)
    fig.suptitle("논문 그림 5의 마커를 읽어 다시 그림 (오차 약 ±0.005). 가로축은 상호작용 스텝", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig5_read.png", dpi=170)
    plt.close(fig)


def fig_single_seed():
    d = json.loads((ANA / "toy_single_seed_results.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for a, n in zip(ax, ("N=20", "N=30")):
        ss = [0.0, 0.05, 0.10, 0.15]
        nul = [d[n][f"s={s}"]["null_same_mean"] for s in ss]
        alt = [d[n][f"s={s}"]["alt_plus_0.15"] for s in ss]
        a.plot(ss, nul, "-o", color=RED, label="두 방법 평균이 같음 (거짓 양성)")
        a.plot(ss, alt, "-o", color=ACC, label="WHIRL 평균이 +0.15 (검출력)")
        a.axhline(0.05, color=GREY, ls="--", lw=1)
        a.set_title(f"체크포인트당 {n}번 굴림", fontsize=10)
        a.set_xlabel("시드별 진짜 성공률의 SD (가정값)")
        a.grid(alpha=0.25)
    ax[0].set_ylabel("Fisher 단측 p < 0.05 비율")
    ax[0].legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "own_single_seed.png", dpi=170)
    plt.close(fig)


def fig_hil():
    d = json.loads((ANA / "toy_hil_corridor_results.json").read_text())["Z10_q0.85"]["variants"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = {"noWM": RED, "aI0": ORG, "whirl": ACC, "bonus": "#3a8f5c"}
    labs = {"noWM": "모델 없음", "aI0": "αi = 0", "whirl": "actor 항 αi = 0.4", "bonus": "보상 보너스 0.4"}
    for k, v in d.items():
        y = np.array(v["rolling_curve_every10"])
        ax[0].plot(np.arange(len(y)) * 10, y, color=cols[k], label=labs[k], lw=1.6)
    ax[0].set_xlim(0, 150)
    ax[0].set_xlabel("에피소드")
    ax[0].set_ylabel("굴림 개입 비율 (20 시드 평균)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].grid(alpha=0.25)
    ks = list(d.keys())
    ax[1].bar(range(4), [100 * d[k]["operator_step_share_total"] for k in ks], color=[cols[k] for k in ks])
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([labs[k] for k in ks], fontsize=8)
    ax[1].set_ylabel("학습 전체 중 운영자 스텝 (%)")
    for i, k in enumerate(ks):
        ax[1].text(i, 100 * d[k]["operator_step_share_total"] + 0.05, f"떨어뜨림 {d[k]['train_drops_per_seed']}", ha="center", fontsize=8)
    ax[1].set_ylim(0, 4.8)
    fig.tight_layout()
    fig.savefig(FIG / "own_hil_toy.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_table1()
    fig_fig5()
    fig_single_seed()
    fig_hil()
    print("wrote 4 figures")
