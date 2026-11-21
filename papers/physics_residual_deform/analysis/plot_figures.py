"""논문 수치로 우리가 그린 그래프 세 장을 만든다 (CPU, 수 초).

실행: python3 analysis/plot_figures.py -> report/figures/own_*.png
입력: recompute.py의 표 값과 검정 함수 (같은 폴더).
팔레트: references/palette.md의 범주형 슬롯 1–6 (라이트 모드). 검증은 validate_palette.js 통과.
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
import recompute as rc  # noqa: E402

OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#dcdad3"

for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_mde_bars():
    methods = ["Spring-Mass", "MPM", "GBND", "PGND", "Diff. SM", "PGRD"]
    objs = rc.OBJECTS
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
    style(ax)
    w = 0.13
    for j, m in enumerate(methods):
        xs, ys, es = [], [], []
        for i, o in enumerate(objs):
            cell = rc.T1["MDE"][o][j]
            if cell is None:
                continue
            xs.append(i + (j - 2.5) * w)
            ys.append(cell[0])
            es.append(cell[1])
        ax.bar(xs, ys, width=w * 0.92, color=SLOTS[j], label=m, yerr=es,
               error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    # PGRD 값 직접 표시 (대비 경고 대응: 값을 글자로 씀)
    for i, o in enumerate(objs):
        v, s = rc.T1["MDE"][o][5]
        ax.text(i + (5 - 2.5) * w, v + s + 0.6, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(objs)))
    ax.set_xticklabels(objs, color=INK)
    ax.set_ylabel("MDE (cm), 평균 ± SD", color=INK2, fontsize=9)
    ax.set_ylim(0, 36)
    ax.text(len(objs) - 0.5, 33.5, "MPM의 Teddy는 평가하지 않음", ha="right", fontsize=8, color=INK2)
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(OUT / "own_mde_bars.png", facecolor=SURFACE)
    plt.close(fig)


def fig_forest():
    # PGRD 대 객체·지표별 최선 기준선 차이 (95% Welch 구간, 가정 E1)
    rows = []
    for metric in ["MDE", "CD", "EMD"]:
        for o in rc.OBJECTS:
            cells = rc.T1[metric][o]
            ours_m, ours_s = cells[5]
            cands = [(m, c) for m, c in zip(rc.METHODS[:5], cells[:5]) if c is not None]
            bm, bc = min(cands, key=lambda x: x[1][0])
            n = rc.N_EVAL
            se = math.sqrt(bc[1] ** 2 / n + ours_s ** 2 / n)
            d = ours_m - bc[0]
            t, p = rc.welch(ours_m, ours_s, n, bc[0], bc[1], n)
            half = 2.09 * se  # 95% 근사 (df≈19–38의 t 값 근처)
            rows.append((f"{metric} {o} (vs {bm})", d, half, p))
    fig, ax = plt.subplots(figsize=(8.6, 6.8), dpi=200)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ys = list(range(len(rows)))[::-1]
    for y, (lab, d, h, p) in zip(ys, rows):
        color = SLOTS[0] if p < 0.05 else INK2
        ax.errorbar(d, y, xerr=h, fmt="o", color=color, ecolor=INK2, elinewidth=1, capsize=2, ms=4)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5, color=INK)
    ax.set_xlabel("PGRD - 최선 기준선 (cm, 음수면 PGRD가 낮음), 95% 구간", color=INK2, fontsize=9)
    ax.text(0.99, 0.01, "파란 점: p < 0.05 (Teddy EMD 한 개)", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "own_forest_tracking.png", facecolor=SURFACE)
    plt.close(fig)


def fig_planning():
    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=200)
    style(ax)
    labels = ["PGRD", "Spring-Mass"]
    ks = [8, 2]
    ns = [10, 10]
    # Wilson 95%
    def wilson(k, n, z=1.96):
        ph = k / n
        den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    xs = [0, 1]
    for x, k, n, col in zip(xs, ks, ns, [SLOTS[5], SLOTS[1]]):
        ph = k / n
        lo, hi = wilson(k, n)
        ax.bar(x, ph, width=0.5, color=col)
        ax.errorbar(x, ph, yerr=[[ph - lo], [hi - ph]], fmt="none", ecolor=INK, capsize=4)
        ax.text(x, hi + 0.03, f"{k}/{n}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=INK)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("케이블 재배선 성공률 (Wilson 95%)", color=INK2, fontsize=9)
    ax.text(0.5, -0.22, "과제 1개, 시행 10회, Fisher 양측 p = 0.023", transform=ax.transAxes,
            ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "own_planning_success.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    fig_mde_bars()
    fig_forest()
    fig_planning()
    print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
