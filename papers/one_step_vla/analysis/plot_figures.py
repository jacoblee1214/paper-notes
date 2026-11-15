"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림은 쓰지 않는다.

  own_table2_ci.png     표 2 셀별 Δ (ref - 1-step)와 독립 근사 90% 구간, ±3 pp 마진
  own_toy_w1.png       토이 A: 모드 간격에 따른 W1 (1스텝, N=10, N=1000)
  own_realworld_ci.png  표 3 실로봇 태스크별 Δ와 논문 보고 90% 구간

입력: analysis/table_check_results.json, analysis/toy_one_step_gmm_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
T2 = json.loads((ROOT / "analysis" / "table_check_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_one_step_gmm_results.json").read_text(encoding="utf-8"))

# 한글 폰트: Linux Noto CJK 다음 macOS Apple SD Gothic Neo 순서로 찾는다
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for cand in FONT_CANDIDATES:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False


def table2():
    rows = T2["table2_rows"]
    labels = [f"{r['model']} · {r['metric']}" for r in rows]
    y = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(rows) + 1.2))
    ax.axvspan(-3, 3, color="#e7f1e4", zorder=0)
    colours = {"equivalent": "#2e7d32", "different": "#c62828", "inconclusive": "#888888"}
    for yi, r in zip(y, rows):
        lo, hi = r["ci90_pp_independent"]
        d = r["delta_pp_ref_minus_1step"]
        ax.plot([lo, hi], [yi, yi], color=colours[r["verdict_independent_ci"]], lw=2)
        ax.plot(d, yi, "o", color=colours[r["verdict_independent_ci"]], ms=5)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Δ = 기준 - 1스텝 (pp), 독립 근사 90% 구간")
    ax.set_title("표 2 셀 판정 (독립 근사, n이 알려진 19셀)", fontsize=10)
    ax.set_xlim(-25, 25)
    fig.tight_layout()
    fig.savefig(FIG / "own_table2_ci.png", dpi=170)
    plt.close(fig)


def toy():
    rows = TOY["rows"]
    d_s = [r["d_over_s"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(d_s, [r["W1_one_step_vs_target"] for r in rows], "o-", color="#c62828", label="1 스텝")
    ax.plot(d_s, [r["W1_10step_vs_target"] for r in rows], "s-", color="#1565c0", label="10 스텝")
    ax.plot(d_s, [r["W1_1000step_vs_target"] for r in rows], "^-", color="#2e7d32", label="1000 스텝")
    ax.set_xlabel("모드 간격 / 성분 표준편차 (d/s)")
    ax.set_ylabel("1-Wasserstein 거리 (목표와)")
    ax.set_title("토이 A: 정확한 속도장, 1차원 두 모드", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_w1.png", dpi=170)
    plt.close(fig)


def realworld():
    tasks = T2["table3_tasks"]
    models = ["DiT", "X-VLA", "FLOWER", "pi0.5", "LAP", "XR-0"]
    tnames = list(dict.fromkeys(t["task"] for t in tasks))
    fig, axes = plt.subplots(1, len(tnames), figsize=(9.6, 3.4), sharey=True)
    for ax, task in zip(axes, tnames):
        sub = [t for t in tasks if t["task"] == task]
        ys = list(range(len(models)))[::-1]
        for yi, m in zip(ys, models):
            r = next(t for t in sub if t["model"] == m)
            lo, hi = r["reported_ci90"]
            d = r["net_count_ref_minus_1step"]
            colour = "#c62828" if r["starred_reported"] else "#555"
            ax.plot([lo, hi], [yi, yi], color="#bbb", lw=5, solid_capstyle="butt")
            ax.plot(d, yi, "o", color=colour, ms=5)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_title(task.replace("->", "->"), fontsize=8.5)
        ax.set_xlim(-30, 30)
        ax.grid(alpha=0.2, axis="x")
    axes[0].set_yticks(list(range(len(models)))[::-1])
    axes[0].set_yticklabels(models, fontsize=8)
    fig.supxlabel("Δ = 기준 - 1스텝 (pp, 논문 보고 90% 구간 회색, 빨간 점은 논문 별표)", fontsize=8.5)
    fig.suptitle("표 3 실로봇 태스크별 차이 (태스크당 n=100)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_realworld_ci.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    table2()
    toy()
    realworld()
    print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
