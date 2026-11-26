"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_latency_events.png : 그림 7(a)의 지연 구간별 비율과 Wilson 95% 구간(사건 167개 가정) + 표 6의 속도별 평균·중앙값·최대
own_chance_latency.png : 표 1의 보고 지연 대 같은 FPR의 우연 수준 지연(프레임, 로그 축)
own_intervals.png      : 사건 수준 비율(76%, 43%, 9%)과 실로봇 100/100, 95/100의 95% 구간

실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
입력: analysis/recompute_results.json (analysis/recompute.py 가 만듦)
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
INK, ACC, GREY, WARM = "#1b1b1a", "#1f5fae", "#9a9890", "#c4572b"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def plot_latency():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    labels = ["1 프레임\n(시작)", "2–5", "6–21", "21 초과"]
    pct = [43, 33, 15, 9]
    k = [72, 55, 25, 15]
    lo_hi = [wilson(x, 167) for x in k]
    err = np.array([[p - lo for p, (lo, hi) in zip(pct, lo_hi)], [hi - p for p, (lo, hi) in zip(pct, lo_hi)]])
    ax.bar(labels, pct, color=ACC, width=0.6)
    ax.errorbar(labels, pct, yerr=err, fmt="none", ecolor=INK, capsize=4, lw=1.2)
    for i, p in enumerate(pct):
        ax.text(i, p + err[1][i] + 1.5, f"{p}%", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("슬립 사건 비율 (%)")
    ax.set_title("그림 7(a) 값 + Wilson 95% (n = 167 가정)", fontsize=10, color=INK)
    ax.set_ylim(0, 62)
    style(ax)
    ax = axes[1]
    speeds = [2, 5, 10, 15, 18]
    mean = [14.9, 6.8, 3.4, 5.8, 2.9]
    med = [3.0, 3.0, 1.0, 2.5, 1.0]
    mx = [106, 39, 27, 35, 20]
    x = np.arange(5)
    ax.bar(x - 0.2, mean, 0.38, color=ACC, label="평균")
    ax.bar(x + 0.2, med, 0.38, color=GREY, label="중앙값")
    ax.scatter(x, mx, marker="v", color=WARM, zorder=3, label="최대")
    for i, v in enumerate(mx):
        ax.text(i + 0.12, v + 3, str(v), color=WARM, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} mm/s" for s in speeds])
    ax.set_ylabel("지연 (프레임, 1 프레임 = 4.17 ms)")
    ax.set_title("표 6: 속도별 평균·중앙값·최대", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_latency_events.png", dpi=170)
    plt.close(fig)


def plot_chance():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    rep = {  # 표 1 보고 지연 (프레임)
        "T1 base Piezo ID": 1.00, "T1 base XL ID": 1.00, "T1 base AND ID": 1.00, "T1 SS Piezo ID": 20.50, "T1 SS XL ID": 4.67, "T1 SS fused ID": 2.00,
        "T1 base Piezo OOD": 2.00, "T1 base XL OOD": 2.00, "T1 base AND OOD": 2.00, "T1 SS Piezo OOD": 36.17, "T1 SS XL OOD": 3.00, "T1 SS fused OOD": 1.50,
        "T2 palm": 3.00, "T2 Tesollo 2F": 3.00, "T2 Tesollo 3F": 5.00}
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    lim = [0.8, 80]
    ax.plot(lim, lim, color=GREY, lw=1, ls="--")
    ax.text(1.0, 1.25, "보고 지연 = 우연 수준", fontsize=8, color=GREY, rotation=33)
    for name, lat in rep.items():
        ch = rec["V7"]["rows"][name]["chance_median_frames"]
        base = "base" in name
        ax.scatter(ch, lat, s=42, color=WARM if base else ACC, zorder=3)
    ax.scatter([], [], color=WARM, label="해석적 기준선")
    ax.scatter([], [], color=ACC, label="SlipSense 계열")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(0.8, 60)
    ax.set_xlabel("같은 FPR의 우연 수준 중앙값 지연 (프레임)")
    ax.set_ylabel("보고된 중앙값 지연 (프레임)")
    ax.set_title("표 1·2 지연 대 우연 수준 (프레임별 독립 가정)", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_chance_latency.png", dpi=170)
    plt.close(fig)


def plot_intervals():
    items = [("76% 이내 5 프레임 (127/167)", 127, 167), ("시작 프레임 검출 (72/167)", 72, 167), ("21 프레임 초과 (15/167)", 15, 167),
             ("실로봇 슬립 검출 (100/100)", 100, 100), ("실로봇 낙하 방지 (95/100)", 95, 100)]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for i, (lab, k, n) in enumerate(items[::-1]):
        lo, hi = wilson(k, n)
        p = 100 * k / n
        ax.plot([lo, hi], [i, i], color=ACC if "실로봇" not in lab else WARM, lw=3, solid_capstyle="round")
        ax.scatter([p], [i], color=INK, zorder=3, s=22)
        ax.text(max(hi, p) + 1.5, i, f"{p:.1f}%  [{lo:.1f}, {hi:.1f}]", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items[::-1]], fontsize=8.5)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("비율 (%) 과 Wilson 95% 구간 (사건 수 167은 그림 판독 가정, 실로봇 시행 100)")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_intervals.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_latency()
    plot_chance()
    plot_intervals()
    print("wrote", FIG)
