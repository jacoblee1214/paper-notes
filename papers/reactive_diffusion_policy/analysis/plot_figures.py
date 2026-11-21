"""우리가 그린 그래프 세 장을 report/figures/에 만든다 (논문 그림은 쓰지 않는다).

  own_scores_ci.png : 피넬링·닦기 조건별 평균과 최대 분산 가정의 95% 상한 구간
  own_lift_wilson.png : 양팔 들기 클램프·들기 성공 (Wilson 95%)
  own_gain35.png    : "> 35%" 주장의 상대 향상과 절대 점수 차이

실행: python3 analysis/plot_figures.py   (matplotlib, 수 초)
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recompute as rc  # noqa: E402  (원자료와 Wilson 함수 재사용)

OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def korean_font():
    for name in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


plt.rcParams["font.family"] = korean_font()
plt.rcParams["axes.unicode_minus"] = False
COL = {"DP": "#8a8a85", "DP w. tactile emb.": "#c9a227", "RDP (Force)": "#1f5fae", "RDP (GelSight)": "#5b8fd1"}


def fig_scores():
    conds = ["섭동 없음", "접촉 전 섭동", "접촉 후 섭동"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, (task, tab) in zip(axes, (("피넬링", rc.PEEL), ("닦기", rc.WIPE))):
        keys = ["DP", "DP w. tactile emb.", "RDP (Force)"]
        width = 0.26
        for i, k in enumerate(keys):
            means = tab[k][:3]
            # 최대 분산 가정: 분산 m(1-m), 표준오차 상한 sqrt(m(1-m)/10), 95% 폭 1.96 SE
            half = [1.96 * math.sqrt(m * (1 - m) / 10) for m in means]
            xs = [j + (i - 1) * width for j in range(3)]
            ax.bar(xs, means, width, color=COL[k], label=k if task == "피넬링" else None)
            ax.errorbar(xs, means, yerr=half, fmt="none", ecolor="#333", capsize=3, lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(conds, fontsize=8.5)
        ax.set_title(f"{task} (조건당 10회)", fontsize=10.5)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", color="#e5e3dc", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("점수 (논문 표 II, III)")
    fig.legend(loc="upper center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "own_scores_ci.png", dpi=180)
    plt.close(fig)


def fig_lift():
    names = ["DP", "DP w. tactile emb.", "RDP (GelSight + MCTac)", "RDP (Force)"]
    labels = ["DP", "DP\n(촉각 임베딩)", "RDP\n(GelSight+MCTac)", "RDP\n(힘)"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.36
    for i, (metric, off, color) in enumerate((("clamp", -width / 2, "#8a8a85"), ("lift", width / 2, "#1f5fae"))):
        ks, lo, hi, pts = [], [], [], []
        for n in names:
            d = rc.LIFT[n]
            if metric == "clamp":
                k = d["soft"][0] + d["hard"][0]
            else:
                k = d["soft"][1] + d["hard"][1]
            w = rc.wilson(k, 20)
            pts.append(k / 20)
            lo.append(k / 20 - w[0])
            hi.append(w[1] - k / 20)
        xs = [j + off for j in range(len(names))]
        ax.bar(xs, pts, width, color=color, alpha=0.9, label="클램프 성공" if metric == "clamp" else "들기 성공")
        ax.errorbar(xs, pts, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=3, lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("성공 비율 (20회)")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    ax.set_title("양팔 들기: 성공 비율과 Wilson 95% 구간 (표 IV 기준)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "own_lift_wilson.png", dpi=180)
    plt.close(fig)


def fig_gain():
    import json
    claim = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))["claim_gt35_check"]
    tasks = ["피넬링", "닦기", "양팔 들기"]
    keys = ["Peeling", "Wiping", "Lifting"]
    rel = [claim[k]["rel_gain_percent"] for k in keys]
    absd = [claim[k]["abs_gain_points"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.bar(tasks, rel, color="#1f5fae")
    ax.axhline(35, color="#c0392b", ls="--", lw=1)
    ax.text(-0.45, 45, "35%", color="#c0392b", fontsize=8.5, ha="left")
    ax.set_ylabel("상대 향상 (%)")
    ax.set_ylim(0, 880)
    ax.set_title("상대 향상 (최고 기준선 대비)", fontsize=10)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)
    for i, v in enumerate(rel):
        ax.text(i, v + 12, f"{v:.0f}%", ha="center", fontsize=8.5)
    ax = axes[1]
    ax.bar(tasks, absd, color="#c9a227")
    ax.axhline(35, color="#c0392b", ls="--", lw=1)
    ax.text(-0.45, 37, "35점", color="#c0392b", fontsize=8.5, ha="left")
    ax.set_ylabel("절대 점수 차이 (점)")
    ax.set_title("절대 점수 차이 (최고 기준선 대비)", fontsize=10)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", color="#e5e3dc", lw=0.6)
    ax.set_axisbelow(True)
    for i, v in enumerate(absd):
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "own_gain35.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_scores()
    fig_lift()
    fig_gain()
    print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
