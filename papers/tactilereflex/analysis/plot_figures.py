"""리포트 그래프 (CPU, matplotlib). 실행: python3 analysis/plot_figures.py → report/figures/*.png

모두 우리가 직접 그린 그래프다(논문 그림 아님, 공개본에 넣어도 됨).
팔레트는 Okabe-Ito 계열 색맹 안전 세트. 한 축, 직접 라벨.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TC = json.loads((ROOT / "analysis" / "table_check_results.json").read_text())
TA = json.loads((ROOT / "analysis" / "toy_threshold_results.json").read_text())
TB = json.loads((ROOT / "analysis" / "toy_control_results.json").read_text())

for f in ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, GRID = "#1b1b1a", "#55544f", "#dcdad3"
BLUE, ORANGE, GREEN, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#7a7a75"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


# ---------------------------------------------------------------- 그림 1: 절제 실험 성공률과 Wilson 95% 구간
def fig_ablation():
    cfg = TC["T1_ablation"]["config"]
    names = ["A 슬립만", "B 슬립+보호", "C 슬립+해제", "D 전체"]
    keys = ["A", "B", "C", "D"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    for i, k in enumerate(keys):
        kk, n = cfg[k]["k"], cfg[k]["n"]
        lo, hi = cfg[k]["wilson95"]
        ax.errorbar(kk / n * 100, i, xerr=[[kk / n * 100 - lo * 100], [hi * 100 - kk / n * 100]],
                    fmt="o", color=BLUE if k == "D" else GREY, ms=6, capsize=3, lw=1.4)
        ax.text(102, i, f"{kk}/{n}", va="center", fontsize=9, color=INK)
    pooled = TC["T1_ablation"]["pooled_ABC"]
    lo, hi = pooled["wilson95"]
    ax.errorbar(pooled["k"] / pooled["n"] * 100, 4, xerr=[[pooled["k"] / pooled["n"] * 100 - lo * 100],
                                                        [hi * 100 - pooled["k"] / pooled["n"] * 100]],
                fmt="s", color=ORANGE, ms=6, capsize=3, lw=1.4)
    ax.text(102, 4, f"{pooled['k']}/{pooled['n']}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(5))
    ax.set_yticklabels(names + ["A+B+C 합산"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-3, 115)
    ax.set_xlabel("성공률 (%), Wilson 95% 구간, 구성당 N = 5", fontsize=9, color=MUTED)
    ax.set_title("표 V 절제 실험: 한 구성 5회는 구간이 매우 넓다", fontsize=10, color=INK, loc="left")
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_ci.png")
    plt.close(fig)


# ---------------------------------------------------------------- 그림 2: 토이 A, 드리프트별 오경보율
def fig_threshold():
    names = ["R1_noise_P99.9", "R2_paper_style_min_slip", "R3_in_sample_Youden", "R4_midpoint"]
    labels = ["R1 잡음 P99.9", "R2 논문형(최소 슬립 바로 아래)", "R3 표본 내 최적(Youden)", "R4 중점"]
    cols = [BLUE, ORANGE, GREEN, PURPLE]
    drifts = ["drift_1.0", "drift_1.25", "drift_1.5"]
    xs = [1.0, 1.25, 1.5]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    for n, lab, c in zip(names, labels, cols):
        ys = [TA["results"][d][n]["false_alarm_per_min_median"] for d in drifts]
        ax.plot(xs, ys, marker="o", color=c, lw=1.8, ms=5, label=lab)
    ax.set_xlabel("시험 때 잡음 척도 / 보정 때 (드리프트 배수)", fontsize=9, color=MUTED)
    ax.set_ylabel("오경보 (회/분, 12 Hz)", fontsize=9, color=MUTED)
    ax.set_xticks(xs)
    ax.set_title("토이 A: 드리프트에서 오경보율이 규칙마다 다르게 오른다 (합성 데이터)", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_threshold.png")
    plt.close(fig)


# ---------------------------------------------------------------- 그림 3: 토이 B, 여유 폭별 성공률과 검출 지연
def fig_control():
    abl = TB["ablation"]
    ws = list(abl.keys())
    x = [float(w.split("=")[1]) for w in ws]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=200, gridspec_kw={"width_ratios": [1.6, 1]})
    series = [("A_slip_only", "A 슬립만", GREY), ("D_full", "D 전체", BLUE), ("C_slip_release", "C 슬립+해제", ORANGE)]
    for key, lab, c in series:
        ys = [abl[w][key]["rate"] * 100 for w in ws]
        ax1.plot(x, ys, marker="o", color=c, lw=1.8, ms=5, label=lab)
    fixed = TB["fixed"]
    ax1.plot(x, [fixed[w]["F0_fixed_1200"]["rate"] * 100 for w in ws], ls="--", color=MUTED, lw=1.2, label="고정 e=1200")
    ax1.set_xlabel("여유 폭 W (힘 단위; 안티슬립 1회 증분 = 0.3)", fontsize=8.5, color=MUTED)
    ax1.set_ylabel("성공률 (%), N = 400", fontsize=8.5, color=MUTED)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=7.5, frameon=False, loc="center right")
    ax1.set_title("구성별 성공률", fontsize=9, color=INK, loc="left")
    style(ax1)
    delay = TB["delay"]
    hzs = list(delay.keys())
    med = [delay[h]["delay_median_ms"] for h in hzs]
    p90 = [delay[h]["delay_p90_ms"] for h in hzs]
    labels = [h.replace("Hz", " Hz") for h in hzs]
    ax2.bar(range(len(hzs)), med, color=BLUE, width=0.55, label="중앙값")
    ax2.scatter(range(len(hzs)), p90, color=ORANGE, zorder=3, label="90%")
    ax2.set_xticks(range(len(hzs)))
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("슬립 시작 → 검출 (ms)", fontsize=8.5, color=MUTED)
    ax2.legend(fontsize=7.5, frameon=False)
    ax2.set_title("검출 지연 (토이)", fontsize=9, color=INK, loc="left")
    style(ax2)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_control.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_ablation()
    fig_threshold()
    fig_control()
    print("wrote figures to", FIG)
