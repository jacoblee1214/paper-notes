"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다. 라이선스는 arXiv non-exclusive).

own_ablation_ci.png : 표 2 과제별 성공률과 Wilson 95% 구간 (세 조건)
own_xemb_init.png   : 표 3 교차 로봇 성공률과 초기 자세 20/20의 Wilson 구간
own_collection.png  : 그림 8 재구성 (n = 10 성공 횟수 Wilson 구간, 성공 시도 평균 시간)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib, 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c86a1c"
TASKS = ["Cube", "Multi-obj", "Plug", "Cap", "Tea", "Drawer", "Egg", "Biman."]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * min(c - h, p) if k > 0 else 0.0, 100 * max(c + h, p) if k < n else 100.0


def plot_ablation(rec):
    counts = rec["table2_counts"]
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    x = np.arange(len(TASKS))
    styles = [("RealDexUMI", ACC, -0.22), ("w/o tactile", ORG, 0.0), ("State-as-action", GREY, 0.22)]
    for name, col, off in styles:
        ks = counts[name]
        pct = np.array([100 * k / 20 for k in ks])
        lo = np.array([wilson(k, 20)[0] for k in ks])
        hi = np.array([wilson(k, 20)[1] for k in ks])
        ax.errorbar(x + off, pct, yerr=[pct - lo, hi - pct], fmt="o", ms=5, color=col, capsize=2.5,
                    lw=1.1, label=f"{name} (평균 {100*sum(ks)/160:.2f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=9)
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.13))
    ax.set_title("표 2 재계산: 과제별 성공률과 Wilson 95% 구간 (각 20회)", fontsize=10, color=INK, pad=24)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation_ci.png", dpi=170)
    plt.close(fig)


def plot_xemb_init(rec):
    t3 = rec["table3"]
    init = rec["init_pose"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    embs = ["FR3", "RM65", "Adam-U"]
    x = np.arange(len(embs))
    for off, task, col in [(-0.1, "Cube", ACC), (0.1, "Drawer", ORG)]:
        ks = t3[task]["counts"]
        pct = np.array([100 * k / 20 for k in ks])
        lo = np.array([wilson(k, 20)[0] for k in ks])
        hi = np.array([wilson(k, 20)[1] for k in ks])
        ax.errorbar(x + off, pct, yerr=[pct - lo, hi - pct], fmt="o", ms=6, color=col, capsize=3, lw=1.2, label=task)
    ax.set_xticks(x)
    ax.set_xticklabels(embs)
    ax.set_ylim(0, 110)
    ax.set_ylabel("성공률 (%)")
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.42))
    ax.set_title("표 3: 로봇별 성공률 (각 20회)", fontsize=10, color=INK)

    ax = axes[1]
    k = init["counts"]
    lo, hi = init["wilson"]
    ax.bar([0], [100 * k / 20], color=ACC, width=0.5)
    ax.errorbar([0], [100 * k / 20], yerr=[[max(0, 100 * k / 20 - 100 * lo)], [max(0, 100 * hi - 100 * k / 20)]],
                fmt="none", ecolor=INK, capsize=4)
    ax.set_xticks([0])
    ax.set_xticklabels(["cube, 초기 자세 4개\n(20회)"], fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.text(0, 103, f"{k}/20", ha="center", fontsize=9, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("초기 자세 강건성 (그림 6)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_xemb_init.png", dpi=170)
    plt.close(fig)


def plot_collection(rec):
    f8 = rec["fig8"]
    tasks = ["Cap", "Tea"]
    methods = ["Human", "AVP", "Manus", "RealDexUMI"]
    cols = {"Human": GREY, "AVP": "#d98a3d", "Manus": "#5a9a57", "RealDexUMI": ACC}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    x = np.arange(len(tasks))
    w = 0.2
    ax = axes[0]
    handles = []
    for i, m in enumerate(methods):
        vals = [100 * f8[t][m]["succ"] / 10 for t in tasks]
        lo = [wilson(f8[t][m]["succ"], 10)[0] for t in tasks]
        hi = [wilson(f8[t][m]["succ"], 10)[1] for t in tasks]
        vv = np.array(vals)
        handles.append(ax.bar(x + (i - 1.5) * w, vv, w, color=cols[m], label=m))
        ax.errorbar(x + (i - 1.5) * w, vv, yerr=[vv - np.array(lo), np.array(hi) - vv], fmt="none", ecolor=INK, capsize=2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Cap twisting", "Tea picking"], fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("성공률 (%), n = 10")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("그림 8 재구성: 성공률 (Wilson 95%)", fontsize=9.5, color=INK)

    ax = axes[1]
    for i, m in enumerate(methods):
        vals = [f8["time_s"][t][m] for t in tasks]
        ax.bar(x + (i - 1.5) * w, vals, w, color=cols[m], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(["Cap twisting", "Tea picking"], fontsize=9)
    ax.set_ylabel("성공 시도의 평균 시간 (s)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("그림 8 재구성: 평균 시간 (SD 미보고)", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, methods, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.savefig(FIG / "own_collection.png", dpi=170)
    plt.close(fig)


def main():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    plot_ablation(rec)
    plot_xemb_init(rec)
    plot_collection(rec)
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))


if __name__ == "__main__":
    main()
