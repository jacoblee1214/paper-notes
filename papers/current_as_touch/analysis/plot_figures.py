"""recompute_results.json을 읽어 우리가 그린 그래프 두 장을 report/figures/에 만든다(CPU, 수 초).

- own_forest.png: 현재 방식 대 비교 차이(pp)와 Newcombe 95% 구간, Fisher p
- own_bottle_load.png: 병 들기 로드별 안정 비율과 Wilson 95% 구간(n=12)
실행: python3 analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text(encoding="utf-8"))
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

from matplotlib import font_manager  # noqa: E402


def _korean_font():
    """Linux(Noto CJK)와 macOS(Apple SD Gothic Neo) 후보를 차례로 찾는다."""
    for name in ("Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "DejaVu Sans"


plt.rcParams.update({"font.family": [_korean_font()], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
INK, MUTED, ACCENT, GREY = "#1b1b1a", "#55544f", "#1f5fae", "#9a9890"


def forest():
    rows = [
        ("표 1 기초 실패율 (novice, 잡기 실패) w/o - w/", RES["T1_fisher_grasp_fail_w_vs_wo"]["novice"]),
        ("표 1 기초 실패율 (skilled, 잡기 실패) w/o - w/", RES["T1_fisher_grasp_fail_w_vs_wo"]["skilled"]),
        ("표 2 닦기 성공 (novice) w/ - Retargeting", RES["T2_pairs"][0]),
        ("표 2 닦기 성공 (novice) w/ - w/o", RES["T2_pairs"][1]),
        ("표 2 닦기 성공 (skilled) w/ - w/o", RES["T2_pairs"][2]),
        ("표 3 병 안정 150 g  w/ - w/o", RES["T3_stable"]["150g"]["diff_pp_w_minus_wo"]),
        ("표 3 병 안정 250 g  w/ - w/o", RES["T3_stable"]["250g"]["diff_pp_w_minus_wo"]),
        ("표 3 병 안정 350 g(OOD)  w/ - w/o", RES["T3_stable"]["350g"]["diff_pp_w_minus_wo"]),
        ("표 4 카드 엄격 성공 w/ - w/o", RES["T4_strict"]),
        ("표 4 카드 허용 성공 w/ - w/o", RES["T4_tolerant"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=160)
    for i, (label, r) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = r["newcombe95_pp"]
        d = r["diff_pp"]
        ax.plot([lo, hi], [y, y], color=ACCENT, lw=2)
        ax.plot(d, y, "o", color=ACCENT, ms=5)
        ax.text(102, y, f"p={r['fisher_p']:.3f}" if r["fisher_p"] >= 0.001 else "p<0.001",
                va="center", ha="left", fontsize=8, color=MUTED)
    ax.axvline(0, color=GREY, lw=1, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8, color=INK)
    ax.set_xlim(-50, 100)
    ax.set_xlabel("차이 (percentage point), 점: 추정치, 선: Newcombe 95% 구간")
    ax.set_title("논문 표 원시 횟수로 다시 계산한 차이 (양수 = 현재 방식 우위)", fontsize=9.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_forest.png")
    plt.close(fig)


def bottle():
    loads = ["0g", "150g", "250g", "350g"]
    wo = [RES["T3_stable"][l]["w/o"] for l in loads]
    w = [RES["T3_stable"][l]["w/"] for l in loads]
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=160)
    x = range(len(loads))
    width = 0.36
    for offset, data, color, name in ((-width / 2, wo, GREY, "w/o Current"), (width / 2, w, ACCENT, "w/ Current")):
        heights = [d["pct"] for d in data]
        err_lo = [d["pct"] - d["wilson95"][0] for d in data]
        err_hi = [d["wilson95"][1] - d["pct"] for d in data]
        ax.bar([i + offset for i in x], heights, width, color=color, label=name)
        ax.errorbar([i + offset for i in x], heights, yerr=[err_lo, err_hi], fmt="none", ecolor=INK, capsize=3, lw=0.8)
        for i, d in zip(x, data):
            ax.text(i + offset, d["wilson95"][1] + 2, f"{d['pct']:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["0 g", "150 g", "250 g (학습 부하)", "350 g (학습 밖)"], fontsize=8.5)
    ax.set_ylabel("안정 유지 비율 (%)")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("병 들기 과제: 부하별 안정 비율 (n=12, Wilson 95% 구간)", fontsize=9.5, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "own_bottle_load.png")
    plt.close(fig)


if __name__ == "__main__":
    forest()
    bottle()
    print("wrote", FIG / "own_forest.png", FIG / "own_bottle_load.png")
