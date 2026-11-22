"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_ci.png : 표 1 과제별 성공률과 Wilson 95% 구간 (F2F-AP 대 기준선 4종)
own_ablation.png  : 표 2 절제 성공률과 Wilson 95% 구간 (병 잡기, 20회)
own_toy_latency.png: 지연 토이: 속도 변화, 총 지연, 관측 잡음에 따른 파지 성공률
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK -> macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY = "#1b1b1a", "#1f5fae", "#9a9890"
COLORS = {"Syn. UMI": "#c9c7bf", "Asyn. UMI": "#9a9890", "RTC": "#d98c3f", "VLASH": "#6b8f71", "F2F-AP": ACC}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_table1(rec):
    t1 = rec["V1_table1_wilson"]
    methods = list(t1.keys())
    tasks = [r["task"] for r in t1[methods[0]]]
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    w = 0.16
    x = np.arange(len(tasks))
    for j, m in enumerate(methods):
        xs = x + (j - 2) * w
        pct = np.array([r["sr"] for r in t1[m]])
        lo = np.array([r["wilson_lo"] for r in t1[m]])
        hi = np.array([r["wilson_hi"] for r in t1[m]])
        ax.errorbar(xs, pct, yerr=[pct - lo, hi - pct], fmt="o", ms=4.5, color=COLORS[m], ecolor=COLORS[m],
                    capsize=2, lw=1, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={t1['F2F-AP'][i]['n']}, RTC {t1['RTC'][i]['n']})" for i, t in enumerate(tasks)], fontsize=7.5)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(-3, 100)
    ax.grid(axis="y", color="#e4e2da", lw=0.6)
    ax.legend(ncol=5, fontsize=8, loc="upper center", frameon=False)
    ax.set_title("표 1 재계산: 과제별 성공률과 Wilson 95% 구간", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_ci.png", dpi=170)
    plt.close(fig)


def plot_ablation(rec):
    t2 = rec["V6_ablation_vs_full"]
    names = ["Fewer Flow Points", "Diff. Line Color", "Low Contrastive Wt.", "w/o Contrastive Loss", "w/o Future State",
             "w/o Heatmap", "MF-UMI", "RTC-MF-UMI", "Low Freq. Inference", "300ms Latency", "500ms Latency", "600ms Latency"]
    labels = ["Full F2F-AP"] + names
    ks = [11] + [t2[n]["k"] for n in names]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    y = np.arange(len(labels))[::-1]
    for yi, lab, k in zip(y, labels, ks):
        lo, hi = wilson(k, 20)
        col = ACC if lab == "Full F2F-AP" else GREY
        ax.errorbar(100 * k / 20, yi, xerr=[[100 * k / 20 - lo], [hi - 100 * k / 20]], fmt="o", color=col, ecolor=col, capsize=3)
        ax.text(103, yi, f"{k}/20", va="center", fontsize=8)
    ax.axvline(55, color=ACC, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-3, 115)
    ax.set_xlabel("성공률 (%), 병 잡기 20회")
    ax.set_title("표 2 재계산: 절제별 성공률과 Wilson 95% 구간 (점선 = 전체 모델 11/20)", fontsize=10)
    ax.grid(axis="x", color="#e4e2da", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=170)
    plt.close(fig)


def plot_toy():
    toy = json.loads((ANA / "toy_latency_intercept_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), sharey=True)
    panels = [("B_speed_change_std", "물체 속도 변화 크기 (m/s, 표준편차)"),
              ("C_true_latency_s", "총 지연 L (s)"),
              ("E_obs_noise_m", "관측 잡음 (m)")]
    sty = {"stale": (GREY, "낡은 관측 유지"), "flow": (ACC, "흐름 외삽 (등속)"), "oracle": ("#d98c3f", "오라클")}
    for ax, (key, xl) in zip(axes, panels):
        xs = list(toy[key].keys())
        for p, (c, lab) in sty.items():
            ax.plot(range(len(xs)), [100 * toy[key][x][p] for x in xs], "-o", color=c, label=lab, ms=4)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=8)
        ax.set_xlabel(xl, fontsize=9)
        ax.grid(color="#e4e2da", lw=0.6)
    axes[0].set_ylabel("파지 성공률 (%)")
    axes[0].legend(fontsize=8, frameon=False, loc="center right")
    fig.suptitle("지연 토이 (1차원, 6000회/점, numpy): 속도 변화, 총 지연, 관측 잡음", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_latency.png", dpi=170)
    plt.close(fig)


def main():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    plot_table1(rec)
    plot_ablation(rec)
    plot_toy()
    print("wrote", [p.name for p in sorted(FIG.glob("own_*.png"))])


if __name__ == "__main__":
    main()
