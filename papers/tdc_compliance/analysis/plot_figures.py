"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_pooled_ci.png   : Table 1 시뮬 학습 방법의 합산 성공 횟수(SR-ND 80회, SR-D 20회)와 Wilson 95% 구간
own_table5_ci.png   : Table 5 절제 다섯 줄의 SR-ND·SR-D와 Wilson 구간 (논문 값과 재계산이 일치)
own_sensitivity.png : Fig. 6을 읽은 값(시행 20회 가정)에 Wilson 구간을 다시 그린 민감도 곡선
own_toy_contact.png : toy T1(벽 강성에 따른 최대 힘)과 T2(자세 오프셋에 따른 정상 힘)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 recompute.py, toy_admittance.py를 실행해 둘 것.
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c95a24"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e4dd", lw=0.7)
    ax.set_axisbelow(True)


def fig_pooled(R):
    rows = [(m, v["succ_nd"], v["succ_d"]) for m, v in R["V3_failures"].items()]
    rows.sort(key=lambda r: r[1])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, idx, n, title in ((axes[0], 1, 80, "SR-ND 합산 (과제 4개 × 20회 = 80회)"),
                              (axes[1], 2, 20, "SR-D 합산 (과제 4개 × 5회 = 20회)")):
        for i, r in enumerate(rows):
            k = r[idx]
            lo, hi = wilson(k, n)
            ours = r[0] == "Ours"
            ax.errorbar(100 * k / n, i, xerr=[[100 * k / n - lo], [hi - 100 * k / n]], fmt="o",
                        color=ORG if ours else ACC, ecolor=ORG if ours else GREY, capsize=3, ms=6)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(-3, 103)
        ax.set_xlabel("성공률 (%)")
        clean(ax)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=9)
    fig.suptitle("Table 1 시뮬 학습 방법: 합산 성공률과 Wilson 95% 구간 (시행 수 20+5회 가정)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_pooled_ci.png", dpi=170)
    plt.close(fig)


def fig_table5(R):
    labels = ["1 사전학습 없음\n+ 적응형", "2 L-A", "3 M-A", "4 H-A", "5 TDC\n(사전학습 + 적응형)"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, key, n, title in ((axes[0], "k_nd", 80, "SR-ND (80회)"), (axes[1], "k_d", 20, "SR-D (20회)")):
        for i in range(5):
            k = R["V7_table5"][str(i + 1)][key]
            lo, hi = wilson(k, n)
            ax.errorbar(100 * k / n, 4 - i, xerr=[[100 * k / n - lo], [hi - 100 * k / n]], fmt="s",
                        color=ORG if i == 4 else ACC, ecolor=GREY, capsize=3, ms=6)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(-3, 103)
        ax.set_xlabel("성공률 (%)")
        clean(ax)
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(labels[::-1], fontsize=9)
    fig.suptitle("Table 5 절제: 재계산한 Wilson 구간은 논문 값과 일치", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_table5_ci.png", dpi=170)
    plt.close(fig)


def fig_sens():
    from recompute import FIG6  # noqa: WPS433

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=True)
    cols = {"MO": "#d89a14", "PH": "#2f7fb5", "WW": "#2e8b6a", "DO": "#8a5ea8"}
    ax = axes[0]
    xs = np.arange(4)
    for j, t in enumerate(["MO", "PH", "WW", "DO"]):
        ys = np.array(FIG6["a_stiffness"][t])
        ci = np.array([wilson(round(y * 20 / 100), 20) for y in ys])
        ax.errorbar(xs + (j - 1.5) * 0.06, ys, yerr=[ys - ci[:, 0], ci[:, 1] - ys], color=cols[t], marker="o",
                    ms=4, capsize=2, label=t, lw=1.3)
    ax.set_xticks(xs)
    ax.set_xticklabels(FIG6["a_stiffness"]["x"])
    ax.set_xlabel("강성 k (이동/회전)")
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    for t, key in (("PH", "PH_x"), ("WW", "WW_x")):
        ys = np.array(FIG6["b_force"][t])
        ci = np.array([wilson(round(y * 20 / 100), 20) for y in ys])
        axes[1].errorbar(FIG6["b_force"][key], ys, yerr=[ys - ci[:, 0], ci[:, 1] - ys], color=cols[t], marker="o",
                         ms=4, capsize=2, label=t, lw=1.3)
    axes[1].set_xlabel("목표 힘 (N)")
    axes[1].legend(fontsize=8, loc="lower left")
    ys = np.array(FIG6["c_ratio"]["MO"])
    ci = np.array([wilson(round(y * 20 / 100), 20) for y in ys])
    axes[2].errorbar(FIG6["c_ratio"]["x"], ys, yerr=[ys - ci[:, 0], ci[:, 1] - ys], color=cols["MO"], marker="o",
                     ms=4, capsize=2, lw=1.3)
    axes[2].set_xlabel("강성 비 k_high / k_low (MO)")
    axes[0].set_ylabel("성공률 (%)")
    axes[0].set_ylim(0, 105)
    for a in axes:
        a.grid(color="#e6e4dd", lw=0.7)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Fig. 6을 읽은 값(그림 판독, 시행 20회 가정)에 Wilson 95% 구간을 다시 그림", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_sensitivity.png", dpi=170)
    plt.close(fig)


def fig_toy():
    T = json.loads((ANA / "toy_admittance_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax = axes[0]
    for key, lab, c in (("target_4N_approach", "목표 4 N, 자유 공간 접근", ACC),
                        ("target_4N_at_rest", "목표 4 N, 정지에서 시작", GREY)):
        r = T["T1_contact"][key]
        ax.plot([x["ke"] for x in r], [x["peak_force"] for x in r], marker="o", label=lab, color=c)
    ax.axhline(4, color=ORG, ls="--", lw=1)
    ax.text(1.2e3, 5.2, "목표 4 N", color=ORG, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("벽 강성 ke (N/m)")
    ax.set_ylabel("최대 접촉 힘 (N)")
    ax.set_title("T1 힘 제어 축의 접촉 순간 최대 힘", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(color="#e6e4dd", lw=0.7)
    ax = axes[1]
    S = T["T2_static"]
    es = np.array(S["e_m"]) * 100
    for key, lab, c in (("L-A_50", "L-A (k = 50)", "#7aa6d8"), ("M-A_200", "M-A (k = 200)", ACC),
                        ("H-A_800", "H-A (k = 800)", "#102440")):
        ax.plot(es, S[key], marker="o", label=lab, color=c)
    ax.plot(es, S["TDC_force_axis_4N"], marker="s", label="TDC 힘 제어 축 (4 N)", color=ORG)
    ax.set_xlabel("지령 위치가 표면 아래로 간 깊이 e (cm), ke = 10⁴ N/m")
    ax.set_ylabel("정상 접촉 힘 (N)")
    ax.set_title("T2 등방성 어드미턴스의 정상 힘은 자세 오프셋에 비례", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(color="#e6e4dd", lw=0.7)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_contact.png", dpi=170)
    plt.close(fig)


def main():
    import sys

    sys.path.insert(0, str(ANA))
    R = json.loads((ANA / "recompute_results.json").read_text())
    fig_pooled(R)
    fig_table5(R)
    fig_sens()
    fig_toy()
    print("figures in", FIG)


if __name__ == "__main__":
    main()
