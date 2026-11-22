"""DP3 분석의 그래프 두 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_task_scatter.png : 시뮬레이션 72 과제, 과제별 DP 대 DP3 (Table XVIII), 대각선 위·아래 과제 수
own_real_ci.png      : 실로봇 성공률(Table VIII)과 일반화 실험(IX–XIII)의 Wilson 95% 구간
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
INK, ACC, GREY, SOFT = "#1b1b1a", "#1f5fae", "#9a9890", "#8fb0d9"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def plot_scatter(rec):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rc", ANA / "recompute.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    dp = rc.dp
    d3 = rc.d3
    dom = rc.dom
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    is_mw = np.array([d.startswith("MW") for d in dom])
    ax.scatter(dp[~is_mw], d3[~is_mw], s=22, color=ACC, label="비-MetaWorld (22)", alpha=0.85)
    ax.scatter(dp[is_mw], d3[is_mw], s=22, color=GREY, label="MetaWorld (50)", alpha=0.85)
    ax.plot([0, 100], [0, 100], color=INK, lw=0.8, ls="--")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Diffusion Policy 성공률 (%)")
    ax.set_ylabel("DP3 성공률 (%)")
    ax.set_title("72 과제 (3 시드 평균, 부록 Table XVIII)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    n_up = int((d3 > dp).sum())
    n_dn = int((d3 < dp).sum())
    n_eq = int((d3 == dp).sum())
    ax.text(4, 92, f"DP3 우위 {n_up}\nDP 우위 {n_dn}\n동률 {n_eq}", fontsize=8.5, color=INK, va="top")
    ax = axes[1]
    labels = ["Roll-Up", "Dumpling", "Drill", "Pour"]
    x = np.arange(len(labels))
    w = 0.38
    dp3_v = [9, 7, 8, 10]
    dp_v = [0, 3, 7, 4]
    ax.bar(x - w / 2, [10 * v for v in dp3_v], w, color=ACC, label="DP3")
    ax.bar(x + w / 2, [10 * v for v in dp_v], w, color=GREY, label="Diffusion Policy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("성공률 (%, 과제당 10회)")
    ax.set_title("실로봇 4 과제 (표 VIII, 각 10회)", fontsize=10)
    for i, (a, b) in enumerate(zip(dp3_v, dp_v)):
        ax.text(i - w / 2, 10 * a + 2, f"{a}/10", ha="center", fontsize=7.5)
        ax.text(i + w / 2, 10 * b + 2, f"{b}/10", ha="center", fontsize=7.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "own_task_scatter.png", dpi=200)
    plt.close(fig)


def plot_real(rec):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    labels = ["DP3", "Diffusion\nPolicy", "DP\n(Depth)"]
    keys = ["DP3", "DP", "DP-Depth"]
    succ = [rec["table8_real"][k]["success"] for k in keys]
    pct = [100 * s / 40 for s in succ]
    lo = [wilson(s, 40)[0] for s in succ]
    hi = [wilson(s, 40)[1] for s in succ]
    yerr = np.array([np.array(pct) - np.array(lo), np.array(hi) - np.array(pct)])
    x = np.arange(3)
    ax.bar(x, pct, color=[ACC, GREY, GREY], width=0.6)
    ax.errorbar(x, pct, yerr=yerr, fmt="none", ecolor=INK, capsize=3, lw=1)
    for i, s in enumerate(succ):
        ax.text(i, 3, f"{s}/40", ha="center", color="white", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("성공률 (%)")
    ax.set_title("실로봇 전체 (4 과제 × 10회, Wilson 95%)", fontsize=9.5)
    ax = axes[1]
    gen = rec["generalization_tables"]
    rows = [
        ("IX 공간 (5 위치)", "IX spatial (Pour, 5 positions)"),
        ("X 외형 (5 색)", "X appearance (Drill, 5 colors)"),
        ("XI 물체 (5 종)", "XI instance (Drill, 5 objects)"),
        ("XII 시점 (3 시점)", "XII view (Roll-Up, 3 views)"),
        ("XIII 어수선 (10회)", "XIII clutter (train color, 10)"),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, key) in zip(y, rows):
        for off, m, c in [(0.2, "DP3", ACC), (-0.2, "DP", GREY)]:
            d = gen[key][m]
            p = 100 * d["k"] / d["n"]
            lo_, hi_ = d["wilson95_pct"]
            ax.errorbar([p], [yi + off], xerr=[[p - lo_], [hi_ - p]], fmt="o", color=c, ms=5, capsize=2, lw=1)
            ax.text(103, yi + off, f"{d['k']}/{d['n']}", va="center", fontsize=7.5, color=c)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(-2, 115)
    ax.set_xlabel("성공률 (%) · 파랑 DP3, 회색 Diffusion Policy")
    ax.set_title("일반화 실험 (Wilson 95%)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_real_ci.png", dpi=200)
    plt.close(fig)


def main():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    plot_scatter(rec)
    plot_real(rec)
    for name in ["own_task_scatter.png", "own_real_ci.png"]:
        print("wrote", FIG / name)


if __name__ == "__main__":
    main()
