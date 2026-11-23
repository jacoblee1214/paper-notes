"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_sr_ci.png          : Table 1 과제별 성공 횟수(15회)와 Wilson 95% 구간 (TeleDexter와 기준선)
own_stage_survival.png : Fig. 6의 단계별 생존 시행 수를 PDF 벡터에서 복원한 정수로 다시 그린 그림
own_ablation_ci.png    : Table 5(행동 마스킹)와 Table 3(자율 정책)의 Wilson 구간
own_consistency.png    : Table 6의 목표당 성공률과 연속 성공 평균의 관계, Fig. 5 학습 곡선 끝값
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
먼저 recompute.py를 실행해 둘 것.
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
INK, ACC, GREY, ORG, GRN = "#1b1b1a", "#1f5fae", "#9a9890", "#c95a24", "#2a8f6a"
R = json.loads((ANA / "recompute_results.json").read_text())


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#e6e4dd", lw=0.7)
    ax.set_axisbelow(True)


def fig_sr_ci():
    tasks = ["CylinderReorient", "CuboidReorient", "BunnyReorient", "HammerUse", "BrushSweep", "ScrewdriverUse", "BulbReplace"]
    T1 = {"TeleDexter": [12, 12, 10, 10, 11, 11, 13], "DexRT": [1, 4, 0, 0, 0, 1, 0], "GeoRT": [0] * 7, "DexGen": [0] * 7}
    cols = {"TeleDexter": ORG, "DexRT": ACC, "GeoRT": GREY, "DexGen": GRN}
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    off = {"TeleDexter": -0.27, "DexRT": -0.09, "GeoRT": 0.09, "DexGen": 0.27}
    for m, ks in T1.items():
        for i, k in enumerate(ks):
            lo, hi = wilson(k, 15)
            y = i + off[m]
            ax.plot([lo, hi], [y, y], color=cols[m], lw=2, alpha=0.75)
            ax.plot(100 * k / 15, y, "o", color=cols[m], ms=5, label=m if i == 0 else None)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("성공률 % (15회, Wilson 95% 구간)")
    ax.set_xlim(-2, 102)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.09))
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_sr_ci.png", dpi=200)
    plt.close(fig)


def fig_stage():
    d = R["V3_fig6_vs_table1"]
    tasks = ["HammerUse", "BrushSweep", "ScrewdriverUse", "BulbReplace"]
    names = {"Ours": ("TeleDexter", ORG, "-"), "DexRT": ("DexRT", ACC, "-"), "GeoRT": ("GeoRT", GREY, "--"),
             "DexGen": ("DexGen", "#7b3294", ":"), "SimToolReal_cat": ("SimToolReal†", GRN, "-"), "SimToolReal_all": ("SimToolReal‡", "#66c2a5", "--")}
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), sharey=True)
    for ax, t in zip(axes, tasks):
        for row in [r for r in d if r["task"] == t]:
            lab, c, ls = names[row["method"]]
            ys = [15] + row["counts"]
            ax.plot(range(len(ys)), ys, ls, color=c, lw=1.8, marker="o", ms=3, label=lab)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("마친 단계 수")
        clean(ax, "y")
    axes[0].set_ylabel("시행 수 (/15)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="upper center", ncol=6)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "own_stage_survival.png", dpi=200)
    plt.close(fig)


def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    a = R["V7_action_mask"]
    tasks = ["HammerUse", "ScrewdriverUse", "CuboidReorient"]
    for i, t in enumerate(tasks):
        for j, (key, col, lab) in enumerate((("with", ORG, "w/ AM"), ("without", GREY, "w/o AM"))):
            k = int(a[t][key].split("/")[0])
            lo, hi = wilson(k, 15)
            y = i + (-0.15 if j == 0 else 0.15)
            ax.plot([lo, hi], [y, y], color=col, lw=2.2, alpha=0.8)
            ax.plot(100 * k / 15, y, "o", color=col, label=lab if i == 0 else None)
        ax.text(101, i - 0.3, "p=%.3g" % a[t]["p_fisher"], va="center", fontsize=8, color=INK)
    ax.set_yticks(range(3))
    ax.set_yticklabels(tasks, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-2, 125)
    ax.set_xlabel("실로봇 SR % (15회, Wilson)")
    ax.set_title("Table 5: 행동 마스킹 절제", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    clean(ax)
    ax = axes[1]
    b = R["V8_autonomous"]
    for i, t in enumerate(["BulbInstall", "HammerDriver", "BrushForward"]):
        sr = b[t]["sr_recomputed"]
        lo, hi = b[t]["sr_wilson"]
        ax.plot([lo, hi], [i, i], color=ACC, lw=2.2, alpha=0.8)
        ax.plot(sr, i, "o", color=ACC)
        ax.text(101, i, "%.1f%% [%.0f, %.0f]" % (sr, lo, hi), va="center", fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["BulbInstall", "HammerDriver", "BrushForward"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-2, 140)
    ax.set_xlabel("자율 정책 SR % (시연 50개, 15회, Wilson)")
    ax.set_title("Table 3: 확산 정책", fontsize=10)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_ci.png", dpi=200)
    plt.close(fig)


def fig_consistency():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    m = np.linspace(0.5, 60, 300)
    ax.plot(m, 100 * m / (m + 1), color=GREY, lw=1.8, label="시행마다 실패 1회 가정: m/(m+1)")
    for name, (mc, per) in {"Cylinder": (41.1, 97.6), "Cuboid": (12.1, 92.3)}.items():
        ax.plot(mc, per, "o", color=ORG, ms=7)
        ax.annotate(name, (mc, per), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax.set_xlabel("평균 연속 성공 m (Table 6)")
    ax.set_ylabel("목표당 성공률 %")
    ax.set_title("Table 6: 두 지표의 관계", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    clean(ax, "both")
    ax = axes[1]
    f = R["V12_fig5"]["final"]
    labs = ["Ours", "no_curriculum", "PPO"]
    shown = ["SAPG+커리큘럼", "커리큘럼 없음", "PPO"]
    x = np.arange(3)
    ax.bar(x - 0.2, [f["reward_1e4"][k] / f["reward_1e4"]["Ours"] for k in labs], 0.38, color=ORG, label="보상 (Ours = 1)")
    ax.bar(x + 0.2, [f["consecutive_successes"][k] / f["consecutive_successes"]["Ours"] for k in labs], 0.38, color=ACC, label="연속 성공 (Ours = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(shown, fontsize=9)
    ax.set_title("Fig. 5 학습 곡선 끝값 (곡선 좌표에서 복원)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    clean(ax, "y")
    fig.tight_layout()
    fig.savefig(FIG / "own_consistency.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_sr_ci()
    fig_stage()
    fig_ablation()
    fig_consistency()
    print("wrote 4 figures in", FIG)
