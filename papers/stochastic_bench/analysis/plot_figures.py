"""우리가 만든 그래프 세 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_table1_subsets.png : 표 1의 시드 부분집합별 성공률 (Spot, Stretch)과 표 2 값
own_seed_planning.png  : 필요 시드 수(식 1, 3)와 시드 난이도 상관이 낮을 때의 검출 한계
own_rule_toy.png       : 토이, "차이 > 2.49 SE" 규칙의 오탐과 검출 확률
실행: python3 analysis/plot_figures.py   (numpy, matplotlib; 먼저 recompute.py, toy_rule_check.py 실행)
한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서로 찾는다.
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
INK, ACC, ORG, GREY = "#1b1b1a", "#1f5fae", "#c2561b", "#9a9890"


def fig_table1(rec):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)
    for ax, task, title in zip(axes, ("drawer", "door"), ("서랍 열기", "문 열기")):
        v = rec["V8"][task]
        groups = [("부분집합 1", 0), ("부분집합 2", 1), ("부분집합 3", 2)]
        for j, (lab, i) in enumerate(groups):
            ax.scatter(j - 0.08, v["spot_subsets"][i], color=ACC, s=55, zorder=3, label="Spot" if j == 0 else None)
            ax.scatter(j + 0.08, v["stretch_subsets"][i], color=ORG, s=55, marker="s", zorder=3,
                       label="Stretch" if j == 0 else None)
        ax.axhline(v["table2_spot_pct"], color=ACC, ls="--", lw=1, alpha=0.7)
        ax.axhline(v["table2_stretch_pct"], color=ORG, ls="--", lw=1, alpha=0.7)
        ax.text(2.45, v["table2_spot_pct"], "표 2 Spot", color=ACC, fontsize=8, va="bottom", ha="right")
        ax.text(2.45, v["table2_stretch_pct"], "표 2 Stretch", color=ORG, fontsize=8, va="bottom", ha="right")
        ax.set_xticks(range(3))
        ax.set_xticklabels([g[0] for g in groups])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.set_ylabel("성공률 (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("표 1: 무작위 시드 부분집합 세 개의 성공률 (점선은 표 2 값)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_table1_subsets.png", dpi=170)
    plt.close(fig)


def fig_planning(rec):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    deltas = np.linspace(0.05, 0.4, 60)
    zs = 2.4865
    for task, col in (("door", ACC), ("drawer", ORG)):
        v = rec["V9"][task]
        for p, ls in ((0.0, "-"), (0.5, "--")):
            k = 2 * zs ** 2 * ((1 - p) * v["sigma2_seed"] + v["B_over_J"]) / deltas ** 2
            ax.plot(deltas * 100, k, color=col, ls=ls,
                    label=f"{'문' if task == 'door' else '서랍'}, p = {p}")
    ax.set_yscale("log")
    ax.set_xlabel("검출하려는 차이 δ (퍼센트포인트)")
    ax.set_ylabel("로봇당 필요 시드 수 K (α=0.10, 검정력 80%)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    ax.set_title("식 (1), (3), 역산한 분산 사용")
    ax = axes[1]
    for task, col in (("door", ACC), ("drawer", ORG)):
        t = rec["V6"][task]["margin_vs_rho"]
        rhos = [1.0, 0.8, 0.5, 0.0]
        ax.plot(rhos, [100 * t[f"rho{r}_p0.5"] for r in rhos], color=col, marker="o",
                label=f"{'문' if task == 'door' else '서랍'}, p = 0.5")
        ax.plot(rhos, [100 * t[f"rho{r}_p1.0"] for r in rhos], color=col, marker="s", ls="--",
                label=f"{'문' if task == 'door' else '서랍'}, p = 1")
    ax.invert_xaxis()
    ax.set_xlabel("두 로봇의 시드 난이도 상관 ρ (논문은 ρ = 1 가정)")
    ax.set_ylabel("검출 가능 한계 (퍼센트포인트)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("ρ < 1이면 시드 공유의 이득이 줄어듦")
    fig.tight_layout()
    fig.savefig(FIG / "own_seed_planning.png", dpi=170)
    plt.close(fig)


def fig_rule(toy):
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    labels = ["문\nK=6/7", "문\nK=30", "서랍\nK=6/8", "서랍\nK=30"]
    keys = [("door", "null_true_gap0"), ("door", "null_K30"), ("drawer", "null_true_gap0"), ("drawer", "null_K30")]
    keys_a = [("door", "alt_true_gap_observed"), ("door", "alt_K30"), ("drawer", "alt_true_gap_observed"), ("drawer", "alt_K30")]
    x = np.arange(4)
    ax.bar(x - 0.2, [100 * toy[a][b]["rule_z_yes"] for a, b in keys], 0.38, color=GREY,
           label="귀무(참 차이 0): 규칙이 \"구분 가능\"이라 함 (%)")
    ax.bar(x + 0.2, [100 * toy[a][b]["rule_z_yes"] for a, b in keys_a], 0.38, color=ACC,
           label="참 차이 = 관측 차이: 규칙이 \"구분 가능\"이라 함 (%)")
    for i, (k0, k1) in enumerate(zip(keys, keys_a)):
        ax.text(i - 0.2, 100 * toy[k0[0]][k0[1]]["rule_z_yes"] + 1.5, f"{100 * toy[k0[0]][k0[1]]['rule_z_yes']:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, 100 * toy[k1[0]][k1[1]]["rule_z_yes"] + 1.5, f"{100 * toy[k1[0]][k1[1]]['rule_z_yes']:.0f}", ha="center", fontsize=8)
    ax.axhline(100 * toy["normal_one_sided_p_at_zsum"], color=INK, ls=":", lw=1)
    ax.text(1.5, 100 * toy["normal_one_sided_p_at_zsum"] + 6, "점선: 정규 가정\n오탐 0.64%", fontsize=7.5, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 140)
    ax.set_ylabel("비율 (%)")
    ax.legend(fontsize=7.5, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("토이: 규칙 \"관측 차이 > 2.49 × SE\" (40,000회 반복)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_rule_toy.png", dpi=170)
    plt.close(fig)


def main():
    rec = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
    toy = json.loads((ANA / "toy_rule_check_results.json").read_text(encoding="utf-8"))
    fig_table1(rec)
    fig_planning(rec)
    fig_rule(toy)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
