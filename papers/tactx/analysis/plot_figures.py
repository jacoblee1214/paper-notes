"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_transfer_ci.png  : 표 2 센서 간 전이(30칸 평균)와 표 8 같은 센서 평균, Wilson 95% 구간(시행 수 가정)
own_same_vs_cross.png: 비전 전용과 TACTX의 과제별 같은 센서(표 8) 대 센서 간(표 2) 성공률
own_toy_transitive.png: 토이 T1, D–F 쌍을 보지 않은 조건과 모두 본 조건의 D–F 정확 검색률
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
앞에 python3 analysis/recompute.py 와 python3 analysis/toy_transitive.py 를 실행해 두어야 한다.
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c4622d"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def fig_transfer(r):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    # 센서 간(표 2, 900회 가정)과 같은 센서(표 8, 150회)
    items = [("비전 전용\n센서 간", r["V3_pooled"]["Vision"], 900, GREY),
             ("이진 접촉\n센서 간", r["V3_pooled"]["Binary"], 900, "#c9c7bf"),
             ("TACTX\n센서 간", r["V3_pooled"]["TactX"], 900, ACC),
             ("비전 전용\n같은 센서", r["V6_table8_pooled_150"]["Vision"], 150, GREY),
             ("원시 촉각\n같은 센서", r["V6_table8_pooled_150"]["GT"], 150, "#7a7871"),
             ("TACTX\n같은 센서", r["V6_table8_pooled_150"]["TactX"], 150, ACC)]
    for i, (lab, d, n, col) in enumerate(items):
        k = d["successes"] if "successes" in d else d["k"]
        lo, hi = wilson(k, n)
        v = 100 * k / n
        ax.bar(i, v, color=col, width=0.7)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=3, lw=1)
        ax.text(i, hi + 2, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([x[0] for x in items], fontsize=8)
    ax.axvline(2.5, color=GREY, lw=0.8, ls=":")
    ax.set_ylim(0, 100)
    ax.set_ylabel("성공률 (%)")
    ax.set_title("방법별 평균과 Wilson 95% 구간\n(센서 간 900회, 같은 센서 150회 가정)", fontsize=10)
    style(ax)

    ax = axes[1]
    tasks = list(r["V2_by_task"]["Vision"].keys())
    x = np.arange(len(tasks))
    w = 0.26
    for j, (m, col, lab) in enumerate([("Vision", GREY, "비전 전용"), ("Binary", "#c9c7bf", "이진 접촉"), ("TactX", ACC, "TACTX")]):
        vals = [r["V2_by_task"][m][t] for t in tasks]
        ax.bar(x + (j - 1) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("센서 간 성공률 (%)")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("과제별 센서 간 평균 (6방향 평균, 표 2)", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "own_transfer_ci.png", dpi=170)
    plt.close(fig)


def fig_same_cross(r):
    tasks = list(r["V2_by_task"]["Vision"].keys())
    t8 = r["V6_table1_vs_table8"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, (m, name, col) in zip(axes, [("Vision", "비전 전용 (촉각 입력 없음)", GREY), ("TactX", "TACTX", ACC)]):
        same = 10 * np.array(t8[m]["from_table8"])
        cross = np.array([r["V2_by_task"][m][t] for t in tasks])
        x = np.arange(len(tasks))
        ax.bar(x - 0.18, same, 0.36, color=col, label="같은 센서 (표 8)")
        ax.bar(x + 0.18, cross, 0.36, color=col, alpha=0.45, label="센서 간 (표 2)")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=9)
        ax.set_title(f"{name}: 평균 {same.mean():.1f}% → {cross.mean():.1f}% (-{same.mean() - cross.mean():.1f} %p)", fontsize=10)
        ax.set_ylim(0, 105)
        style(ax)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("성공률 (%)")
    fig.tight_layout()
    fig.savefig(FIG / "own_same_vs_cross.png", dpi=170)
    plt.close(fig)


def fig_toy():
    p = ANA / "toy_transitive_results.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    methods = [("recon", "재구성+KL"), ("l2", "L2 당김"), ("nce", "InfoNCE만"), ("tactx", "TACTX형")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for ax, key, title in zip(axes, ["pos_cos", "top1"], ["D–F 정답 쌍 코사인 (평균)", "D–F 정확 검색률 (top-1)"]):
        x = np.arange(len(methods))
        for j, (cond, col, lab) in enumerate([("transitive", ACC, "D–F 쌍 학습 안 함 (D–E, E–F만)"), ("all3", ORG, "D–F 쌍도 학습")]):
            vals = [d[f"{cond}/{m}"][key][0] for m, _ in methods]
            errs = [d[f"{cond}/{m}"][key][1] for m, _ in methods]
            ax.bar(x + (j - 0.5) * 0.36, vals, 0.36, yerr=errs, color=col, label=lab, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in methods], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        style(ax)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_transitive.png", dpi=170)
    plt.close(fig)


def main():
    r = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    fig_transfer(r)
    fig_same_cross(r)
    fig_toy()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))


if __name__ == "__main__":
    main()
