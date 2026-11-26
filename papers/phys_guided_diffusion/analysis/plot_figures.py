"""우리가 만든 그래프 세 장을 report/figures/ 에 저장한다. 논문 그림 조각은 여기서 만들지 않는다.

  own_visuomotor.png   표 2·3의 과제 성공률(평균 ± SD, 4회 실행)과 DDIM† 기준선
  own_grasp_strict.png 표 1의 SR(1)과 그림 6에서 읽은 SR_all(그림 판독 값, 쉬움 3 : 어려움 2 가중)
  own_toy_guidance.png 토이: 실행 가능률 대 유효율, 모드 안 퍼짐 비율

입력: analysis/table_check_results.json, analysis/toy_guidance_results.json
실행: python3 analysis/plot_figures.py   (matplotlib, CPU, 수 초)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TC = json.loads((ROOT / "analysis" / "table_check_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "analysis" / "toy_guidance_results.json").read_text(encoding="utf-8"))

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
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

INK, MUTED, ACC, ACC2, WARN, GRN = "#1b1b1a", "#6b6a64", "#1f5fae", "#d9822b", "#a33a3a", "#2f8f5b"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color="#e6e4dd", lw=0.6)
    ax.set_axisbelow(True)


def fig_visuomotor():
    T2 = TC["data"]["T2"]
    names = ["DDIM", "기울기(σ)", "기울기(x_t)", "최적화(L-BFGS)"]
    cols = [MUTED, "#9a9992", "#c9c7bf", ACC]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6), sharey=False)
    for ax, (task, arm) in zip(axes, [("Drawer", "Dynaarm"), ("Drawer", "Franka"), ("PickPlace", "Dynaarm"), ("PickPlace", "Franka")]):
        rows = T2[task][arm]
        vals = [rows[i][0] for i in (1, 2, 3, 4)]
        sds = [rows[i][1] for i in (1, 2, 3, 4)]
        ax.bar(range(4), vals, yerr=sds, color=cols, capsize=3, ecolor=INK, error_kw={"lw": 0.9})
        ax.axhline(rows[0][0], color=WARN, ls="--", lw=1)
        ax.text(-0.45, rows[0][0] + 1.5, "DDIM† %.1f" % rows[0][0], color=WARN, fontsize=8, ha="left")
        for i, v in enumerate(vals):
            ax.text(i, v + sds[i] + 1.5, "%.1f" % v, ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["DDIM", "σ", "x_t", "ours"], fontsize=8)
        ax.set_title(("서랍 열기" if task == "Drawer" else "집어서 놓기") + " · " + arm, fontsize=10, color=INK)
        ax.set_ylim(0, 112)
        style(ax)
    axes[0].set_ylabel("SR_task [%] (평균 ± SD, 4회)", fontsize=9)
    fig.suptitle("표 2·3의 과제 성공률: 서랍 열기는 +2pp(실행 간 변동 안), 집어서 놓기에서 차이가 큼", fontsize=10.5, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "own_visuomotor.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_grasp_strict():
    v13 = TC["V13_fig6_strict_metric"]
    T1 = TC["data"]["T1"]
    methods = ["Gradient", "Theseus", "IPOPT"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, arm in zip(axes, ["Dynaarm", "Franka"]):
        x = np.arange(3)
        sr1 = [T1[arm][m][0] for m in methods]
        sra = [v13[f"{arm}/{m}"]["SRall_weighted_3to2"] for m in methods]
        ax.bar(x - 0.18, sr1, 0.36, color=ACC, label="SR(1) (표 1)")
        ax.bar(x + 0.18, sra, 0.36, color=ACC2, label="SR_all (그림 6 판독, 3:2 가중)")
        for i in range(3):
            ax.text(x[i] - 0.18, sr1[i] + 1.2, "%.1f" % sr1[i], ha="center", fontsize=8)
            ax.text(x[i] + 0.18, sra[i] + 1.2, "%.1f" % sra[i], ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["기울기 안내", "Theseus", "IPOPT"], fontsize=9)
        ax.set_title(arm, fontsize=10)
        ax.set_ylim(0, 85)
        style(ax)
    axes[0].set_ylabel("성공률 [%]", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("같은 기준선 대비 이득: SR(1)에서 +11, +20pp, 엄격한 SR_all에서 약 +6, +9pp (그림 판독 값)", fontsize=10.5, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "own_grasp_strict.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_toy():
    keys = [("U", "무안내", MUTED, "o"), ("P", "투영", WARN, "s"), ("G(g=0.2)", "기울기 g=0.2", ACC2, "^"), ("G(g=0.8)", "기울기 g=0.8", ACC2, "v"),
            ("G(g=1.6)", "기울기 g=1.6", "#b5651d", "D"), ("S(g=1.0)", "σ-안내 g=1", "#7a4fa3", "P"),
            ("O1(x_K 자유)", "최적화 O1", ACC, "*"), ("O2(x_K에 사전항)", "최적화 O2", GRN, "*")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, c in zip(axes[:2], ("c=1.0", "c=1.5")):
        mo = TOY["mean_over_seeds"][c]
        for k, lab, col, mk in keys:
            ax.scatter(mo[k]["feasible"], mo[k]["valid"], s=95 if mk != "*" else 190, color=col, marker=mk, label=lab, zorder=3)
        ax.plot([0, 1.02], [0, 1.02], color="#cccbc3", lw=0.8, ls=":")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("실행 가능률 (J ≤ 0.01)", fontsize=9)
        ax.set_ylabel("유효율 (실행 가능이며 모드 3σ 이내)", fontsize=9)
        ax.set_title("실행 가능 경계 x1 ≥ %s" % c[2:], fontsize=10)
        style(ax)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    mo = TOY["mean_over_seeds"]["c=1.0"]
    labs = [lab for _, lab, _, _ in keys]
    vals = [mo[k]["spread_ratio"] for k, _, _, _ in keys]
    cols = [col for _, _, col, _ in keys]
    axes[2].barh(range(len(keys)), vals, color=cols)
    axes[2].axvline(mo["U"]["spread_ratio"], color=INK, ls="--", lw=1)
    axes[2].set_yticks(range(len(keys)))
    axes[2].set_yticklabels(labs, fontsize=8)
    axes[2].invert_yaxis()
    for i, v in enumerate(vals):
        axes[2].text(v + 0.01, i, "%.2f" % v, va="center", fontsize=8)
    axes[2].set_xlabel("모드 안 퍼짐 비율 (점선 = 무안내)", fontsize=9)
    axes[2].set_title("유효 표본이 모드 안에서 퍼진 정도 (c = 1.0)", fontsize=10)
    style(axes[2])
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_guidance.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_visuomotor()
    fig_grasp_strict()
    fig_toy()
    print("wrote 3 figures in", FIG)
