"""우리가 그린 그래프 세 장. 논문 그림은 쓰지 않는다 (라이선스 판단은 README 참조).

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
입력: recompute.py의 표 값 (아래 상수는 논문 표 III·IV와 같다).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10,
})
INK, ACC, GREY, SOFT = "#1b1b1a", "#1f5fae", "#8a8880", "#c9d7ec"

RES = json.loads((HERE / "recompute_results.json").read_text(encoding="utf-8"))


def fig_ssr_ci():
    """표 II: SSR과 Wilson 95% 구간 (n = 95,880), 파지 속도는 글자로 적음."""
    names = ["DexGraspNet", "SpringGrasp", "FRoGGeR", "BODex (Ours)"]
    ssr = [RES["V2_table2_ssr"][n]["SSR"] for n in names]
    lo = [RES["V2_table2_ssr"][n]["wilson95"][0] for n in names]
    hi = [RES["V2_table2_ssr"][n]["wilson95"][1] for n in names]
    speed = [0.40, 0.67, 1.20, 61.5]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    y = range(len(names))
    colors = [SOFT] * 3 + [ACC]
    ax.barh(list(y), ssr, color=colors, edgecolor=INK, linewidth=0.5, height=0.6)
    ax.errorbar(ssr, list(y), xerr=[[s - l for s, l in zip(ssr, lo)], [h - s for s, h in zip(ssr, hi)]],
                fmt="none", ecolor=INK, capsize=3, linewidth=1)
    for i, (s, v) in enumerate(zip(ssr, speed)):
        ax.text(s + 2, i, f"{s:.1f}%   (S {v} grasp/s)", va="center", color=INK, fontsize=9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlim(0, 115)
    ax.set_xlabel("시뮬레이션 성공률 SSR (%), Allegro, 95,880 파지 기준 Wilson 95% 구간")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("표 II 재계산: 구간이 좁아 차이는 확실하지만 독립성 가정이 있음", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_ssr_ci.png", dpi=200)
    plt.close(fig)


def fig_version_change():
    """v1 대 v3: 같은 표 II 행의 SSR 변화 (지표 정의도 바뀜)."""
    names = ["DexGraspNet", "SpringGrasp", "FRoGGeR", "BODex (Ours)"]
    v1 = [RES["V10_v1_to_v3_table2_ssr"][n]["SSR_v1"] for n in names]
    v3 = [RES["V10_v1_to_v3_table2_ssr"][n]["SSR_v3"] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, (a, b) in enumerate(zip(v1, v3)):
        ax.plot([a, b], [i, i], color=GREY, linewidth=2, zorder=1)
        ax.scatter([a], [i], color="white", edgecolor=GREY, s=50, zorder=2)
        ax.scatter([b], [i], color=ACC, s=50, zorder=3)
        ax.text(max(a, b) + 2, i, f"{a:.1f} → {b:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("SSR (%)  ○ v1 (2024-12-21)   ● v3 (2025-09-03)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("버전에 따라 기준선 순위가 바뀜 (SpringGrasp 37.2 → 12.1)", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_version_change.png", dpi=200)
    plt.close(fig)


def fig_ablation():
    """표 III: 두 요인의 2x2 중 세 칸. ✗✗ 칸은 표에 없다 (빈 칸으로 표시)."""
    cells = {  # (hand): [(label, value)] 순서: 둘 다, C2F만, pregrasp만, 없음(없음)
        "Shadow": [75.8, 57.8, 68.4, None],
        "Allegro": [79.9, 68.2, 78.1, None],
    }
    labels = ["둘 다", "C2F만 (pregrasp 없음)", "pregrasp만 (C2F 없음)", "둘 다 없음"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.36
    for k, hand in enumerate(cells):
        xs = [i + (k - 0.5) * width for i in range(4)]
        vals = cells[hand]
        for x, v in zip(xs, vals):
            if v is None:
                ax.text(x, 3, "표에 없음", ha="center", va="bottom", fontsize=8, color=GREY, rotation=90)
                continue
            ax.bar(x, v, width=width, color=ACC if hand == "Allegro" else SOFT, edgecolor=INK, linewidth=0.5)
            ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 95)
    ax.set_ylabel("SSR (%)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SOFT, ec=INK), plt.Rectangle((0, 0), 1, 1, color=ACC, ec=INK)],
              labels=["Shadow", "Allegro"], frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("표 III: 둘 다 없는 칸이 빠져 있어 두 요인의 상호작용을 알 수 없음", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=200)
    plt.close(fig)


def main():
    fig_ssr_ci()
    fig_version_change()
    fig_ablation()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))


if __name__ == "__main__":
    main()
