"""Own plots for the Wan analysis (reads recompute_results.json, writes report/figures/own_*.png).

Only our own figures; no paper figures are used. Korean font: first available of
Noto Sans CJK (Linux) and Apple SD Gothic Neo (macOS).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

KOREAN = ["Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo", "AppleGothic"]
available = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in KOREAN if f in available), "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False

R = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))


def win_ci_plot():
    """Net win-minus-loss gap with 95% CI for Table 3 (T2V) and Table 7 (I2V), vs CN-TopA."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, key, title in [(axes[0], "table3_ci", "T2V (표 3)"), (axes[1], "table7_ci", "I2V (표 7)")]:
        dims = list(R[key].keys())
        opps = list(R[key][dims[0]].keys())
        y = 0
        labels = []
        for d in dims:
            for opp in opps:
                c = R[key][d][opp]
                color = "#1f5fae" if c["ci95_low"] > 0 else "#b03a2e"
                ax.errorbar(c["g"], -y, xerr=[[c["g"] - c["ci95_low"]], [c["ci95_high"] - c["g"]]],
                            fmt="o", color=color, capsize=3, ms=4)
                labels.append(f"{d} / {opp}")
                y += 1
            y += 0.5
        ax.axvline(0, color="#777", lw=0.8, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("승-패 차이 (win - loss), 95% 구간 (보수적)")
        ax.grid(axis="x", alpha=0.25)
    # y labels for the left axis only, from the T2V panel
    dims3 = list(R["table3_ci"].keys())
    opps3 = list(R["table3_ci"][dims3[0]].keys())
    ticks, names, y = [], [], 0
    for d in dims3:
        for opp in opps3:
            ticks.append(-y)
            names.append(f"{d} / {opp}")
            y += 1
        y += 0.5
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(names, fontsize=7)
    axes[1].tick_params(labelleft=False)
    fig.suptitle("사람 평가 차이와 신뢰구간 재계산 (파란 = 0 초과, 빨강 = 0 포함)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "own_winrate_ci.png", dpi=160)
    plt.close(fig)


def wanbench_plot():
    """Unweighted mean vs reported weighted score; and sensitivity of Wan14B-Sora gap to per-prompt SD."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    um = R["wanbench_unweighted_mean"]
    rw = R["wanbench_reported_weighted"]
    names = list(um.keys())
    xs = range(len(names))
    axes[0].bar([x - 0.2 for x in xs], [um[n] for n in names], width=0.4, color="#9aa7b8", label="단순 평균 (계산)")
    axes[0].bar([x + 0.2 for x in xs], [rw[n] for n in names], width=0.4, color="#1f5fae", label="가중 점수 (논문)")
    axes[0].set_xticks(list(xs))
    axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylim(0.6, 0.75)
    axes[0].set_title("Wan-Bench: 가중치에 따라 순위가 바뀜", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    sens = R["wanbench_sens_wan14_vs_sora"]["by_sd"]
    sds = [float(s) for s in sens.keys()]
    zs = [sens[k]["z"] for k in sens.keys()]
    axes[1].plot(sds, zs, "o-", color="#1f5fae")
    axes[1].axhline(1.96, color="#b03a2e", ls="--", lw=1)
    axes[1].text(sds[-1], 2.05, "z = 1.96 (p = 0.05)", ha="right", fontsize=8, color="#b03a2e")
    axes[1].set_xlabel("프롬프트별 점수 표준편차 s (가정)")
    axes[1].set_ylabel("Wan 14B - Sora 차이의 z")
    axes[1].set_title("가중 점수 차 0.024의 유의성은 s에 달림", fontsize=10)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "own_wanbench_sensitivity.png", dpi=160)
    plt.close(fig)


def vbench_plot():
    """VBench total: reported vs recomputed (4Q+S)/5, showing exact agreement."""
    rows = R["vbench_total_check"]
    names = list(rows.keys())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ys = range(len(names))
    ax.scatter([rows[n]["reported"] for n in names], list(ys), color="#1f5fae", label="인쇄 총점", zorder=3)
    ax.scatter([rows[n]["calc"] for n in names], list(ys), marker="x", color="#b03a2e", label="(4·품질 + 의미)/5", zorder=3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("VBench 총점 (%)")
    ax.set_title("표 4: 계산식과 인쇄값 일치 (최대 차 0.006)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "own_vbench_check.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    win_ci_plot()
    wanbench_plot()
    vbench_plot()
    print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
