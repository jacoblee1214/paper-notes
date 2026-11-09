"""리포트용 그래프 세 장. 실행: python3 analysis/plot_figures.py (table_check.py, ba_toy.py 먼저 실행)

1) iter_curves.png: 표 2 — 반복 1–4에서 BA-T와 ViSTA† w/ iterative training
2) ablation_ladder.png: 표 3 — (a)→(b)→(h)→(g) 사다리와 leave-one-out (c)(d)(e)
3) toy_order.png: ba_toy.py — 갱신 순서별 회전 오차, 대응점 수 N에 따른 포즈·깊이 오차
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import table_check as tc  # noqa: E402

FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                          "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e7"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9f"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def plain_log(axis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)


def iter_curves():
    its = [1, 2, 3, 4]
    panels = [(tc.I_ROT, "회전 AUC@5° ↑"), (tc.I_TRA, "이동 AUC@0.05 m ↑"), (tc.I_CORR, "3D 대응 오차 Corr. (m) ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))
    for ax, (i, title) in zip(axes, panels):
        b = [tc.T2_BAT[k][i] for k in its]
        v = [tc.T2_VISTA_ITER[k][i] for k in its]
        ax.plot(its, b, color=BLUE, lw=2, marker="o", ms=5)
        ax.plot(its, v, color=ORANGE, lw=2, marker="s", ms=5)
        ax.axhline(tc.T3["a"][i], color=MUTED, lw=1, ls="--")
        ax.text(4.05, b[-1], "BA-T", color=INK, va="center", fontsize=8)
        ax.text(4.05, v[-1], "ViSTA†\n반복 학습", color=INK, va="center", fontsize=8)
        ax.text(2.3, tc.T3["a"][i], "ViSTA† 반복 없음", color=INK2, va="bottom", fontsize=7)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks(its)
        ax.set_xlabel("반복 k", color=INK2)
        ax.set_xlim(0.8, 4.9)
        style(ax)
    fig.suptitle("표 2 (7Scenes, 디코더 38M 동일): BA-T는 반복 1에서 이미 ViSTA† 반복 4를 넘는다",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "iter_curves.png", dpi=200)
    plt.close(fig)


def ablation_ladder():
    rows = [("a", "(a) ViSTA†"), ("b", "(b) +반복 학습"), ("h", "(h) +잠재 잔차"), ("g", "(g) 전체 BA-T"),
            ("c", "(c) w/o geometry 변환"), ("d", "(d) w/o 반복 감독"), ("e", "(e) w/o 대응 매칭")]
    panels = [(tc.I_ROT, "회전 AUC@5° ↑"), (tc.I_TRA, "이동 AUC@0.05 m ↑"), (tc.I_CORR, "Corr. (m) ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), sharey=True)
    ys = list(range(len(rows)))[::-1]
    for ax, (i, title) in zip(axes, panels):
        for y, (k, _) in zip(ys, rows):
            col = BLUE if k in "abhg" else MUTED
            val = tc.T3[k][i]
            ax.barh(y, val, color=col, height=0.62)
            ax.text(val, y, f" {val:.3f}", va="center", fontsize=7, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlim(0, max(tc.T3[k][i] for k, _ in rows) * 1.28)
        style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([n for _, n in rows])
    fig.suptitle("표 3 (v2): 파랑 = 사다리 (a)→(b)→(h)→(g), 회색 = 전체 모델에서 하나씩 뺀 것",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "ablation_ladder.png", dpi=200)
    plt.close(fig)


def toy_order():
    d = json.loads((ROOT / "analysis" / "ba_toy_results.json").read_text())
    t2 = d["T2_update_order"]["baseline0.3_init10.0deg"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))
    ax = axes[0]
    for name, col, lab, mk in (("joint", BLUE, "joint (Schur)", "o"), ("camera_first(paper)", AQUA, "카메라 먼저 (논문)", "^"),
                               ("geometry_first(code)", ORANGE, "geometry 먼저 (코드)", "s")):
        y = t2[name]["rot_deg_median_by_iter"]
        ax.plot(range(len(y)), y, color=col, lw=2, marker=mk, ms=4.5, label=lab)
    ax.set_yscale("log")
    plain_log(ax.yaxis)
    ax.set_xlabel("반복", color=INK2)
    ax.set_title("회전 오차 중앙값 (°), 기선 0.3, 초기 10°", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    t3 = d["T3_dimensionality"]
    ns = [10, 40, 160]
    rot = [t3[f"N{n}"]["rot_deg_median"] for n in ns]
    dep = [t3[f"N{n}"]["depth_rel_median"] for n in ns]
    pri = [t3[f"N{n}"]["depth_prior_rel_median"] for n in ns]
    ax = axes[1]
    ax.plot(ns, rot, color=BLUE, lw=2, marker="o", ms=5)
    for n, v in zip(ns, rot):
        ax.text(n, v, f" {v:.3f}", fontsize=7, color=INK, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    plain_log(ax.yaxis)
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("대응점 수 N", color=INK2)
    ax.set_title("포즈: 회전 오차 (°) — N에 따라 계속 준다", fontsize=8.5, color=INK)
    style(ax)
    ax = axes[2]
    ax.plot(ns, dep, color=BLUE, lw=2, marker="o", ms=5)
    ax.plot(ns, pri, color=MUTED, lw=1.5, ls="--", marker="s", ms=4)
    ax.text(ns[-1], dep[-1], " BA 후", fontsize=7, color=INK, va="bottom")
    ax.text(ns[-1], pri[-1], " 단안 사전", fontsize=7, color=INK2, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, max(pri) * 1.25)
    ax.set_xlabel("대응점 수 N", color=INK2)
    ax.set_title("geometry: 상대 깊이 오차 — 금방 바닥", fontsize=8.5, color=INK)
    style(ax)
    fig.suptitle("ba_toy.py (numpy 고전 BA, 시드 10개 중앙값)", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "toy_order.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    iter_curves()
    ablation_ladder()
    toy_order()
    print("saved:", sorted(p.name for p in OUT.glob("*.png") if not p.name.startswith("paper_")))
