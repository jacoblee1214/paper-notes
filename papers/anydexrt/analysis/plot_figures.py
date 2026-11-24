"""우리가 만든 그래프 네 장을 report/figures/에 저장한다 (논문 그림은 쓰지 않는다).

own_v1_v2.png    : 표 1의 AnyDexRT v1 대 v2, 두 기준선(칸별 GMC·LMC)
own_pinch.png    : 표 4 핀치 성공률(시도 합산·에피소드 24개 가정)의 Wilson 구간, 작업자 군집을 넣은 p
own_toy.png      : 토이 결과(뒤집힌 시드 수, 시드 간 위치 SD)
own_fig56_read.png: 그림 5·6에서 읽은 값 (그림 판독)
실행: python3 analysis/plot_figures.py   (numpy, matplotlib, recompute.py와 토이 결과 json 필요;
      한글 폰트는 Linux Noto CJK → macOS Apple SD Gothic Neo 순서)
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
INK, ACC, GREY, ORG = "#1b1b1a", "#1f5fae", "#9a9890", "#c2571a"

RES = json.loads((ANA / "recompute_results.json").read_text(encoding="utf-8"))
T = json.loads((ANA / "toy_anchor_symmetry_results.json").read_text(encoding="utf-8"))
HANDS = ["Inspire", "Ability", "XHand", "Wuji", "Allegro", "Leap", "Shadow"]

# 표 1 값 (recompute.py와 같은 값)
OPT = {"Inspire": (53.9, 43.2), "Ability": (47.7, 39.5), "XHand": (46.9, 33.8), "Wuji": (80.2, 74.2),
       "Allegro": (89.5, 76.1), "Leap": (70.8, 60.0), "Shadow": (45.0, 38.5)}
GEO = {"Inspire": (71.9, 50.6), "Ability": (65.0, 36.2), "XHand": (75.4, 51.3), "Wuji": (87.3, 77.0),
       "Allegro": (85.5, 74.8), "Leap": (73.4, 53.2), "Shadow": (89.8, 75.2)}
V1 = {"Inspire": (80.7, 89.5), "Ability": (84.1, 88.5), "XHand": (76.0, 88.5), "Wuji": (93.1, 92.3),
      "Allegro": (87.4, 92.1), "Leap": (54.5, 89.0), "Shadow": (83.6, 91.4)}
V2 = {"Inspire": (84.7, 92.1), "Ability": (85.0, 90.5), "XHand": (89.7, 94.8), "Wuji": (93.2, 94.1),
      "Allegro": (92.2, 94.0), "Leap": (92.3, 93.2), "Shadow": (90.8, 94.1)}


def fig_v1_v2():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    x = np.arange(len(HANDS))
    for ax, k, name in ((axes[0], 0, "GMC (전역 운동 일관성)"), (axes[1], 1, "LMC (지역 운동 일관성)")):
        ax.scatter(x, [OPT[h][k] for h in HANDS], marker="s", color=GREY, label="Optimization", zorder=3)
        ax.scatter(x, [GEO[h][k] for h in HANDS], marker="^", color=ORG, label="GeoRT", zorder=3)
        ax.scatter(x - 0.12, [V1[h][k] for h in HANDS], marker="o", facecolor="white", edgecolor=ACC, label="AnyDexRT v1", zorder=3)
        ax.scatter(x + 0.12, [V2[h][k] for h in HANDS], marker="o", color=ACC, label="AnyDexRT v2", zorder=3)
        for i, h in enumerate(HANDS):
            ax.plot([i - 0.12, i + 0.12], [V1[h][k], V2[h][k]], color=ACC, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(HANDS, rotation=30, fontsize=9)
        ax.set_title(name, fontsize=10.5, color=INK)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(25, 100)
    axes[0].set_ylabel("점수 (×10⁻², 표 1 값)")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].annotate("Leap v1 → v2: +37.8", xy=(5, 73), xytext=(2.6, 38), fontsize=8.5,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    fig.tight_layout()
    fig.savefig(FIG / "own_v1_v2.png", dpi=170)
    plt.close(fig)


def fig_pinch():
    p = RES["V5_pinch_pooled_assumed"]
    names = ["Optimization", "GeoRT", "AnyDexRT"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for i, m in enumerate(names):
        lo, hi = p[m]["wilson95_at_mid_A"]
        mid = p[m]["reported_pct"]
        ax.errorbar(i, mid, yerr=[[mid - lo], [hi - mid]], fmt="o", color=ACC if m == "AnyDexRT" else INK, capsize=4)
        ax.text(i + 0.08, mid, f"{mid:.1f}%\n({lo:.0f}–{hi:.0f})", fontsize=8.5, va="center")
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    ax.set_xlim(-0.4, 2.8)
    ax.set_ylim(15, 80)
    ax.set_ylabel("핀치 성공률 (%)")
    ax.set_title("표 4 성공률과 Wilson 95% 구간\n(24 에피소드, 성공 10개, 시도 합산 가정)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]
    rows = [("시도를 독립으로\n풀링", "AnyDexRT_vs_Optimization", "AnyDexRT_vs_GeoRT", "p_independent_attempts"),
            ("작업자 군집 ρ=0.1", "cluster_rho0.1_AnyDexRT_vs_Optimization", "cluster_rho0.1_AnyDexRT_vs_GeoRT", "p_adj"),
            ("작업자 군집 ρ=0.3", "cluster_rho0.3_AnyDexRT_vs_Optimization", "cluster_rho0.3_AnyDexRT_vs_GeoRT", "p_adj")]
    t = RES["V5_pinch_tests_assumed"]
    y = np.arange(len(rows))
    for j, (col, off) in enumerate((("vs Optimization", -0.15), ("vs GeoRT", 0.15))):
        vals = [max(t[r[1 + j]][r[3]], 1e-30) for r in rows]
        ax.barh(y + off, [-np.log10(v) for v in vals], height=0.28, color=[ACC, ORG][j], label=col)
    ax.axvline(-np.log10(0.05), color=INK, lw=0.8, ls="--")
    ax.text(-np.log10(0.05) + 0.2, 2.55, "p = 0.05", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("-log10 p (클수록 구분이 분명)")
    ax.set_title("AnyDexRT 대 기준선: 두 비율 z 검정\n(ρ는 이 분석의 예시 값)", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "own_pinch.png", dpi=170)
    plt.close(fig)


def fig_toy():
    keys = [("partial_noanchor", "앵커 없음"), ("partial_anchor", "앵커 8개"),
            ("full_chamfer_anchor_redundant", "양방향\nChamfer\n+ 여분"), ("partial_anchor_redundant", "부분\nChamfer\n+ 여분")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    x = np.arange(len(keys))
    cols = [GREY, ACC, ORG, ACC]
    axes[0].bar(x, [T[k]["flipped_seeds"] for k, _ in keys], color=cols)
    axes[0].set_ylabel("뒤집힌 시드 수 (5개 중)")
    axes[0].set_ylim(0, 5)
    axes[1].bar(x, [T[k]["seed_std_of_mapped_positions"] for k, _ in keys], color=cols)
    for xi, k in zip(x, [k for k, _ in keys]):
        axes[1].text(xi, T[k]["seed_std_of_mapped_positions"], f'{T[k]["seed_std_of_mapped_positions"]:.4f}', ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("시드 간 매핑 위치 SD")
    axes[2].bar(x, [T[k]["dist_rel_err_mean"] for k, _ in keys], color=cols)
    axes[2].set_ylabel("거리 보존 상대 오차")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in keys], fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("앵커 없이는 뒤집힘", fontsize=10)
    axes[1].set_title("시드 안정성", fontsize=10)
    axes[2].set_title("양방향 Chamfer의 왜곡", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy.png", dpi=170)
    plt.close(fig)


def fig_fig56():
    f5 = RES["V9_fig5_read_values"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    a = f5["angles"]
    ax.plot(a, f5["AnyDexRT"], "o-", color=ACC, label="AnyDexRT")
    ax.plot(a, f5["GeoRT"], "s--", color=ORG, label="GeoRT")
    ax.plot(a, f5["Opt"], "^-.", color=GREY, label="Optimization")
    ax.set_xticks(a)
    ax.set_xlabel("입력 회전 (y축, 도)")
    ax.set_ylabel("LMC")
    ax.set_ylim(0, 1)
    ax.set_title("그림 5에서 읽은 값 (그림 판독, Wuji)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower center")
    ax = axes[1]
    ratios = RES["V9_fig6_read_ratio_geort_over_any"]
    hs = list(ratios)
    ax.bar(range(len(hs)), [ratios[h] for h in hs], color=ACC)
    ax.axhline(10, color=INK, lw=0.8, ls="--")
    ax.text(len(hs) - 0.6, 10.5, "10배", fontsize=8, ha="right")
    ax.axhline(100, color=INK, lw=0.8, ls=":")
    ax.text(len(hs) - 0.6, 90, "100배(두 자릿수)", fontsize=8, ha="right", va="top")
    ax.set_yscale("log")
    ax.set_ylim(1, 200)
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels(hs, rotation=30, fontsize=8.5)
    ax.set_ylabel("GeoRT SD / AnyDexRT SD")
    ax.set_title("그림 6에서 읽은 시드 SD의 비 (그림 판독)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_fig56_read.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_v1_v2()
    fig_pinch()
    fig_toy()
    fig_fig56()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
