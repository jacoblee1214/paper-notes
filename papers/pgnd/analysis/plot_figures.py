"""우리가 만든 그래프 두 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다. 이유는 01 §0과 README 참조).

own_table1_ci.png : 표 I 세 지표, 카테고리별 Ours와 최선 기준선의 평균과 95% 구간 (Welch, 시험 수 40 또는 20)
own_table2_sd.png : 표 II J-Score와 F-Score의 평균 대 SD, 0–100 척도에서 가능한 SD 상한 sqrt(m(100-m))
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 입력은 analysis/recompute_results.json과 상수)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
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
INK, ACC, GREY, LIGHT = "#1b1b1a", "#1f5fae", "#9a9890", "#8fb0d9"


def plot_table1():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    rows = rec["table1_welch_vs_best_baseline"]
    cats = ["Cloth", "Rope", "Plush", "Box", "Bag", "Bread"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True)
    for ax, metric in zip(axes, ["MDE", "CD", "EMD"]):
        sub = {r["cat"]: r for r in rows if r["metric"] == metric}
        x = np.arange(len(cats))
        w = 0.38
        o_m = [sub[c]["ours"][0] for c in cats]
        o_lo = [sub[c]["ours_ci95"][0] for c in cats]
        o_hi = [sub[c]["ours_ci95"][1] for c in cats]
        b_m = [sub[c]["base"][0] for c in cats]
        b_n = [sub[c]["n"] for c in cats]
        # 기준선 95% 구간: 같은 방식으로 계산
        b_s = [sub[c]["base"][1] for c in cats]
        b_ci = [stats.t.ppf(0.975, n - 1) * s / np.sqrt(n) for n, s in zip(b_n, b_s)]
        ax.bar(x - w / 2, o_m, w, color=ACC, label="Ours")
        ax.errorbar(x - w / 2, o_m, yerr=[np.array(o_m) - np.array(o_lo), np.array(o_hi) - np.array(o_m)],
                    fmt="none", ecolor=INK, capsize=2, lw=0.9)
        ax.bar(x + w / 2, b_m, w, color=LIGHT, label="최선 기준선")
        ax.errorbar(x + w / 2, b_m, yerr=b_ci, fmt="none", ecolor=INK, capsize=2, lw=0.9)
        ax.set_title(metric + " (낮을수록 좋음)", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8, rotation=20, ha="right")
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("오차 (m), 평균과 95% 구간", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.text(0.01, 0.01, "표 I 평균 ± SD에서 계산. 시험 수 40(Cloth, Rope) 또는 20. 최선 기준선은 셀마다 다름 (GBND 또는 Particle, MPM 제외).",
             fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = FIG / "own_table1_ci.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def plot_table2_sd():
    means_sd = {
        "GBND": [(56.7, 13.9), (19.2, 15.6), (42.7, 10.5), (83.0, 5.5), (78.0, 82.8), (49.8, 13.9)],
        "Particle": [(58.5, 18.7), (24.4, 17.7), (49.0, 12.5), (82.2, 6.1), (74.5, 11.7), (39.3, 15.4)],
        "MPM": [(40.5, 20.3), (12.5, 9.5), (37.5, 14.8), None, (34.3, 5.5), (42.9, 10.3)],
        "Ours": [(63.2, 16.4), (29.5, 16.5), (59.7, 11.1), (82.4, 5.0), (76.8, 8.8), (55.8, 10.1)],
    }
    cats = ["Cloth", "Rope", "Plush", "Box", "Bag", "Bread"]
    styles = {"Ours": (ACC, "o"), "GBND": ("#b5651d", "s"), "Particle": ("#6b6a63", "^"), "MPM": (GREY, "x")}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    m = np.linspace(0.5, 99.5, 400)
    ax.fill_between(m, 0, np.sqrt(m * (100 - m)), color="#eef3fa", zorder=0)
    ax.plot(m, np.sqrt(m * (100 - m)), color=ACC, lw=0.8, ls="--", label="0–100 척도에서 가능한 SD 상한")
    for name, vals in means_sd.items():
        c, mk = styles[name]
        pts = [(v[0], v[1], cats[i]) for i, v in enumerate(vals) if v is not None]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=c, marker=mk, s=28, label=name, zorder=3)
        if name == "GBND":
            for p in pts:
                if p[2] == "Bag":
                    ax.annotate("GBND Bag\n78.0 ± 82.8", xy=(p[0], p[1]), xytext=(p[0] - 30, p[1] - 6),
                                fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 95)
    ax.set_xlabel("J-Score 평균 (%)", fontsize=8)
    ax.set_ylabel("SD (%p)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("표 II J-Score: 평균과 SD (0–100 척도 검사)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    out = FIG / "own_table2_sd.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main():
    plot_table1()
    plot_table2_sd()


if __name__ == "__main__":
    main()
