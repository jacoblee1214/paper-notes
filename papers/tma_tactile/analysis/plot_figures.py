"""자체 그래프 두 장을 그린다. 논문 그림은 쓰지 않는다(라이선스가 CC BY 4.0이 아님).

입력: analysis/recompute_results.json (recompute.py가 만든 값)
출력: report/figures/own_real_sg.png (실로봇 S/G, Wilson 95% 구간)
      report/figures/own_sim_ablation.png (Table I, 시드 SD 막대)
실행: python3 analysis/plot_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
font_path = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
if font_path:
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False


def wilson(k, n, z=1.959964):
    import math
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def real_plot():
    tot = RES["real_totals"]
    methods = ["TMA", "Reactive", "RMA"]
    labels = ["TMA (제안)", "Reactive (이진 접촉)", "RMA (촉각 없음)"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    xs = range(len(methods))
    width = 0.36
    for i, m in enumerate(methods):
        s, g, n = tot[m]["S"], tot[m]["G"], tot[m]["n"]
        ps = 100 * s / n
        pg = 100 * g / n
        lo, hi = wilson(s, n)
        glo, ghi = wilson(g, n)
        ax.bar(i - width / 2, ps, width, color="#0f6b78", label="문 열림 S" if i == 0 else None)
        ax.errorbar(i - width / 2, ps, yerr=[[ps - lo], [hi - ps]], color="#222", capsize=4, lw=1)
        ax.bar(i + width / 2, pg, width, color="#c9d6dd", label="파지 G" if i == 0 else None)
        ax.errorbar(i + width / 2, pg, yerr=[[pg - glo], [ghi - pg]], color="#222", capsize=4, lw=1)
        ax.text(i - width / 2, hi + 2, f"{s}/{n}", ha="center", fontsize=9)
        ax.text(i + width / 2, ghi + 2, f"{g}/{n}", ha="center", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 110)
    ax.set_title("실로봇 손잡이 열기 (손잡이 4종 × 10회, 방법당 n = 40)", fontsize=10)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "막대: 합계 성공률. 오차 막대: Wilson 95%. 출처: 논문 Fig. 6의 칸별 주석을 합산 (재계산).",
             fontsize=6.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "own_real_sg.png")
    plt.close(fig)


def sim_plot():
    sim = RES["sim_ci"]
    names = list(sim.keys())
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    width = 0.36
    for i, name in enumerate(names):
        for j, (metric, color) in enumerate([("G", "#0f6b78"), ("S", "#c9d6dd")]):
            m = sim[name][metric]["mean"]
            sd = sim[name][metric]["sd_seed"]
            x = i + (j - 0.5) * width
            ax.bar(x, m, width, color=color, label=(f"{metric}" if i == 0 else None))
            ax.errorbar(x, m, yerr=sd, color="#222", capsize=4, lw=1)
            ax.text(x, m + sd + 1.2, f"{m:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("시뮬레이션 촉각 신호 절제 (Table I, 시드 3개)", fontsize=10)
    ax.legend(title="지표", loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8, frameon=False, title_fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "막대: 시드 평균. 오차 막대: 시드 간 SD (논문 값, SD 가정). 출처: 논문 Table I (재계산).",
             fontsize=6.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "own_sim_ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    real_plot()
    sim_plot()
    print("saved", sorted(p.name for p in OUT.glob("own_*.png")))
