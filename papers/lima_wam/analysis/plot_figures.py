"""LiMA 분석의 그래프 네 장을 만든다 (우리가 그린 그래프, 논문 그림 아님).

  report/figures/own_sr_ci.png      : 표 1 성공률(과제 6개 풀링 120회)과 Wilson 95% 구간, 과제별 LiMA - 기준선 차이
  report/figures/own_latency.png    : 성공률 평균 대 지연(표 1), 표 3(b)(c)의 지연 모양
  report/figures/own_ablation.png   : 표 2의 시드 복원값과 표 3의 칸별 성공률(칸당 20회 가정)과 Wilson 구간
  report/figures/own_generalization.png : 그림 5 판독값(칸당 20회 가정)으로 그린 일반화 막대

입력: analysis/recompute_results.json. 실행: python3 analysis/plot_figures.py
한글 폰트: Noto Sans CJK KR -> Apple SD Gothic Neo 순서로 찾는다.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

for cand in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
INK, ACC, GREY, ORANGE, GREEN, RED = "#1b1b1a", "#1f5fae", "#8a8880", "#c0602a", "#2f7d5b", "#b03030"
COL = {"LiMA": ACC, "GR00T N1.6": ORANGE, "VPP": GREY, "InternVLA-A1": GREEN, "Cosmos-Policy": "#7a4fa0"}
METHODS = ["GR00T N1.6", "VPP", "InternVLA-A1", "Cosmos-Policy", "LiMA"]
TASKS = ["Stack Cup", "Roll T-shirt", "Cook Rice", "Make Sandwich", "Make Coffee", "Assemble Package"]
SHORT = ["Stack", "Roll", "Cook Rice", "Sandwich", "Coffee", "Assemble"]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def own_sr_ci():
    pooled = {m: RES["V2_means"]["per_method"][m]["sr_sum_counts_of_120"] for m in METHODS}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for i, m in enumerate(METHODS):
        k = pooled[m]
        lo, hi = wilson(k, 120)
        a1.bar(i, 100 * k / 120, 0.62, color=COL[m])
        a1.errorbar(i, 100 * k / 120, yerr=[[100 * k / 120 - lo], [hi - 100 * k / 120]], color=INK, capsize=3, lw=1)
        a1.text(i, hi + 1.5, f"{k}/120", ha="center", fontsize=8.5)
    a1.set_xticks(range(5), [m.replace("-", "-\n") if m == "Cosmos-Policy" else m for m in METHODS], fontsize=8.5)
    a1.set_ylim(0, 100)
    a1.set_ylabel("성공률 (%) 와 Wilson 95% 구간")
    a1.set_title("표 1 성공률, 과제 6개 풀링 (과제당 20회)", fontsize=10)
    pair = RES["V9_sr_comparisons"]["per_task"]
    bases = ["GR00T N1.6", "VPP", "InternVLA-A1", "Cosmos-Policy"]
    w = 0.2
    for j, b in enumerate(bases):
        d = [next(r["diff_pp"] for r in pair if r["task"] == t and r["baseline"] == b) for t in TASKS]
        a2.bar([i + (j - 1.5) * w for i in range(6)], d, w, color=COL[b], label=b)
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_xticks(range(6), SHORT, fontsize=8.5)
    a2.set_ylabel("LiMA - 기준선 (포인트)")
    a2.set_title("과제별 성공률 차이 (과제당 1회 = 5 포인트)", fontsize=10)
    a2.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "own_sr_ci.png", dpi=170)
    plt.close(fig)


def own_latency():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    time = {"GR00T N1.6": 270, "VPP": 225, "InternVLA-A1": 360, "Cosmos-Policy": 600, "LiMA": 325}
    for m in METHODS:
        s = RES["V2_means"]["per_method"][m]["sr_mean"]
        a1.scatter(time[m], s, s=90, color=COL[m], zorder=3)
        a1.annotate(m, (time[m], s), textcoords="offset points", xytext=(6, 6), fontsize=9)
    a1.set_xlabel("표 1 지연 (ms, H100). LiMA는 Dreamer를 4번에 1번 부르는 평균값")
    a1.set_ylabel("과제 6개 성공률 평균 (%)")
    a1.set_xlim(180, 660)
    a1.set_ylim(50, 76)
    a1.set_title("성공률 대 지연 (표 1)", fontsize=10)
    a1.grid(alpha=0.25)
    b = RES["V4_table3b_latency_vs_steps"]
    c = RES["V4_table3c_latency_vs_ratio"]
    a2.plot(b["steps"], b["time_ms"], "o-", color=ACC, label="표 3(b) 디노이징 스텝 수 (5, 10, 15, 20)")
    a2.set_xlabel("표 3(b): 스텝 수")
    a2.set_ylabel("지연 (ms)")
    a2.grid(alpha=0.25)
    a3 = a2.twiny()
    a3.plot([1, 2, 3, 4], c["time_ms"], "s--", color=ORANGE, label="표 3(c) 슬로우 몫 (1:9 ~ 4:6)")
    a3.set_xlabel("표 3(c): 슬로우 몫 Ns (Ns:Nf = Ns:10-Ns)", color=ORANGE)
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a3.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    a2.set_title("표 3의 지연: 스텝 수는 거의 직선, 비율은 +30, +15, +15", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_latency.png", dpi=170)
    plt.close(fig)


def own_ablation():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    v5 = RES["V5_table2_seed_reconstruction"]
    variants = ["w/o Future Visual Intent", "Gaussian Init. + Cross-Attn", "Flow Matching", "LiMA"]
    labels = ["w/o Future\nVisual Intent", "Gaussian Init.\n+ Cross-Attn", "Flow\nMatching", "LiMA"]
    for t_i, (task, dx, col) in enumerate([("Cook Rice", -0.18, ORANGE), ("Stack Cup", 0.18, ACC)]):
        for i, v in enumerate(variants):
            r = next(x for x in v5 if x["variant"] == v and x["task"] == task)
            seeds = r["triplets_5pct_grid"][0]["seeds"]
            a1.scatter([i + dx] * 3, seeds, s=28, color=col, alpha=0.8, label=task if i == 0 else None)
            a1.plot([i + dx - 0.07, i + dx + 0.07], [r["mean"]] * 2, color=INK, lw=2)
    a1.set_xticks(range(4), labels, fontsize=8)
    a1.set_ylabel("성공률 (%)")
    a1.set_ylim(45, 105)
    a1.set_title("표 2: 시드별 성공률 (평균 ± SD에서 복원한 값, 시드당 20회)", fontsize=9.5)
    a1.legend(fontsize=8, loc="lower right")
    a1.grid(alpha=0.25)
    cells = [("3a Full", 80), ("3a w/o Adapt.", 60), ("3a w/o Tokens", 55), ("3a w/o Fast-Slow", 80),
             ("3b 5스텝", 65), ("3b 10스텝", 80), ("3b 15스텝", 85), ("3b 20스텝", 80),
             ("3c 1:9", 50), ("3c 2:8", 65), ("3c 3:7", 80), ("3c 4:6", 70)]
    colors = [ACC] * 4 + [GREEN] * 4 + [ORANGE] * 4
    for i, (name, v) in enumerate(cells):
        lo, hi = wilson(round(v * 20 / 100), 20)
        a2.bar(i, v, 0.7, color=colors[i])
        a2.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=2.5, lw=1)
    a2.set_xticks(range(12), [c[0] for c in cells], rotation=55, ha="right", fontsize=8)
    a2.set_ylabel("성공률 (%) 와 Wilson 95% 구간")
    a2.set_ylim(0, 105)
    a2.set_title("표 3: 칸당 20회 가정 (논문에 시행 수 없음)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=170)
    plt.close(fig)


def own_generalization():
    v = RES["V12_generalization"]["read_values"]
    settings = ["Base", "Cluttered", "Unseen Object", "Unseen Background", "Unseen Lighting"]
    names = ["Base", "Cluttered", "Unseen\nObject", "Unseen\nBackground", "Unseen\nLighting"]
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    w = 0.26
    for j, m in enumerate(["LiMA", "Cosmos-Policy", "GR00T N1.6"]):
        for i, s in enumerate(settings):
            val = v[m][s]
            lo, hi = wilson(round(val * 20 / 100), 20)
            ax.bar(i + (j - 1) * w, val, w, color=COL[m], label=m if i == 0 else None)
            ax.errorbar(i + (j - 1) * w, val, yerr=[[val - lo], [hi - val]], color=INK, capsize=2, lw=0.9)
    ax.set_xticks(range(5), names, fontsize=9)
    ax.set_ylabel("성공률 (%) 와 Wilson 95% 구간")
    ax.set_ylim(0, 100)
    ax.set_title("Cook Rice 일반화 (논문 그림 5에서 읽은 값, 칸당 20회 가정, 눈금 ±2.5 포인트)", fontsize=10)
    ax.legend(fontsize=8.5, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_generalization.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    own_sr_ci()
    own_latency()
    own_ablation()
    own_generalization()
    print("wrote 4 figures in", FIG)
